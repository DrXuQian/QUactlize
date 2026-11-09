#!/usr/bin/env python3
"""Build the decode GEMV workload manifest from GGUF headers and the caller's load/TP rules.

Caller rules are transcribed from DrXuQian/llama.cpp pr/kpack-ffn (30cdcb24):
  * src/llama-kpack-merge.cpp: gate/up merge for QWEN3/QWEN35 dense FFN and QWEN35MOE shared
    expert when both halves share qtype/shape and are plain weights (Q8_0, Q2_K..Q6_K);
    create_tensor_gate_up_exps requests merged experts (same rule).
  * src/llama-model.cpp llama_meta_device_get_split_state: TP axes, segments, granularity.
  * Quactlize admission (dense/api.h, moe/matmul.h): Q8_0,Q2_K..Q6_K, N%256==0,
    K%256==0 (Q3_K/Q6_K: K%512==0).
Per-token decode call counts: one call per layer that owns the tensor; output head once.
MTP/nextn layers are excluded (disabled by default in the caller).
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QT = {8: 'Q8_0', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K'}
NAMES = {0: 'F32', 1: 'F16', 30: 'BF16', 20: 'IQ4_NL', 21: 'IQ3_S', 23: 'IQ4_XS', 39: 'MXFP4', 26: 'I32'}
BLOCK = {8: 32, 10: 256, 11: 256, 12: 256, 13: 256, 14: 256, 39: 32, 20: 32, 21: 256, 23: 256,
         0: 1, 1: 1, 30: 1}
MODELS = ['qwen3-32b', 'qwen38-27b', 'qwen35-35b-a3b', 'qwen35-122b-a10b', 'deepseek-v4-flash']


def qname(q):
    return QT.get(q) or NAMES.get(q) or f'type{q}'


def load(model):
    d = json.loads((ROOT / f'inventory/headers/{model}.json').read_text())
    meta, tensors = {}, {}
    for s in d['shards']:
        meta.update(s['metadata'])
        for t in s['tensors']:
            tensors[t['name']] = t
    return d, meta, tensors


def quactlize_ok(q, n, k):
    if q not in QT:
        return 'format_not_quactlize'
    if n % 256:
        return 'n_not_multiple_of_256'
    if k % (512 if q in (11, 14) else 256):
        return 'k_alignment'
    return None


def lcm(a, b):
    return a * b // math.gcd(a, b)


def split_even(ne, gran, world):
    """Equal tensor_split (tensor_split==nullptr): ne*(j+1)/world rounded down to gran."""
    out, low = [], 0
    for j in range(world - 1):
        high = ne * (j + 1) // world
        high -= high % gran
        out.append(high - low)
        low = high
    out.append(ne - low)
    return out


class Arch:
    def __init__(self, meta):
        self.arch = meta['general.architecture']
        a = self.arch
        g = lambda key, default=None: meta.get(f'{a}.{key}', default)
        self.n_layer = g('block_count')
        self.nextn = g('nextn_predict_layers', 0)
        self.n_embd = g('embedding_length')
        self.heads = g('attention.head_count')
        self.kv = g('attention.head_count_kv')
        self.head_k = g('attention.key_length', self.n_embd // self.heads if self.heads else 0)
        self.d_state = g('ssm.state_size')
        self.n_group = g('ssm.group_count')
        self.dt_rank = g('ssm.time_step_rank')
        self.interval = g('full_attention_interval')
        self.o_groups = g('attention.output_group_count')
        self.o_rank = g('attention.output_lora_rank')
        self.topk = g('expert_used_count')
        self.experts = g('expert_count')

    def is_recr(self, il):
        if self.arch in ('qwen35', 'qwen35moe'):
            return il < self.n_layer - self.nextn and (il + 1) % self.interval != 0
        return False


def tp_local(arch, name, q, n, k, world):
    """Return list of (n_local, k_local) per rank following the caller's split function."""
    if world == 1:
        return [(n, k)]
    dsv4 = arch.arch == 'deepseek4'
    m = re.fullmatch(r'blk\.(\d+)\.(.+)\.weight', name)
    il = int(m[1]) if m else 0
    base = m[2] if m else name.replace('.weight', '')
    blck = BLOCK[q]
    mirrored = [(n, k)] * world
    # --- axis selection (dsv4 block first) ---
    axis = None
    if dsv4:
        if base == 'attn_q_b':
            axis = 1
        elif base == 'attn_output_b':
            axis = 0
        elif base in ('ffn_up_shexp', 'ffn_gate_shexp'):
            axis = 1
        elif base == 'ffn_down_shexp':
            axis = 0
    if axis is None:
        if base in ('attn_q', 'attn_k', 'attn_v', 'attn_qkv', 'attn_gate', 'ssm_alpha', 'ssm_beta'):
            axis = 1
        elif base in ('attn_output', 'ssm_out'):
            axis = 0
        elif base in ('ffn_up', 'ffn_gate', 'ffn_up_exps', 'ffn_gate_exps', 'ffn_gate_up', 'ffn_gate_up_exps'):
            axis = 1
        elif base == 'ffn_gate_up_shexp':
            axis = 1 if dsv4 else 'M'
        elif base in ('ffn_down', 'ffn_down_exps'):
            axis = 0
        elif name == 'output.weight':
            axis = 'M' if dsv4 else 1
        else:
            axis = 'M'
    if axis == 'M':
        return mirrored
    ne = n if axis == 1 else k
    # --- segments ---
    segments = [(ne, 1)]
    if arch.arch in ('qwen35', 'qwen35moe'):
        key_dim = arch.d_state * arch.n_group
        ratio = arch.dt_rank // arch.n_group
        if base == 'attn_qkv':
            segments = [(key_dim, 2 + ratio)]
        elif base in ('attn_gate', 'ssm_out'):
            segments = [(key_dim, ratio)]
        elif base.startswith('ffn_gate_up'):
            segments = [(ne // 2, 2)]
    elif base.startswith('ffn_gate_up'):
        segments = [(ne // 2, 2)]
    # --- granularity ---
    if base.startswith('ffn_'):
        gran = lcm(blck, 128)
    elif arch.is_recr(il):
        gran = lcm(lcm(blck, 128), arch.d_state)
        if base not in ('attn_qkv', 'attn_gate', 'ssm_out'):
            gran = 1
    else:
        n_gqa = arch.heads // arch.kv
        n_embd_q = n_gqa * arch.head_k
        perf = blck
        while perf < 128 and perf * world < n_embd_q:
            perf *= 2
        gq = lcm(n_embd_q, perf)
        if dsv4 and base == 'attn_q_b':
            gran = (arch.heads // arch.o_groups) * arch.head_k
        elif dsv4 and base == 'attn_output_b':
            gran = arch.o_rank
        elif base == 'attn_q':
            gran = lcm(2 * n_embd_q, perf) if arch.arch in ('qwen35', 'qwen35moe', 'qwen3next') else gq
        elif base == 'attn_output':
            gran = gq
        elif base in ('attn_k', 'attn_v'):
            gran = gq // n_gqa
        else:
            gran = 1
    per_rank = [0] * world
    for size, repeat in segments:
        for r, e in enumerate(split_even(size, gran, world)):
            per_rank[r] += e * repeat
    if sum(per_rank) != ne:
        raise ValueError(f'split coverage {name}')
    return [(e, k) if axis == 1 else (n, e) for e in per_rank]


def model_ops(model):
    d, meta, tensors = load(model)
    arch = Arch(meta)
    world = d['tp']
    main_layers = arch.n_layer - arch.nextn
    ops = []          # quactlize ops
    native = Counter()  # non-quactlize matrices (by family)

    def add(role, names, q, n, k, kind, experts=1, topk=1, channels=1, merged=False):
        reason = quactlize_ok(q, n, k)
        if reason:
            native[(role, qname(q), n, k, experts, reason)] += 1
            return
        ops.append(dict(role=role, tensors=names, q=q, n=n, k=k, kind=kind, experts=experts,
                        topk=topk, channels=channels, merged=merged))

    def dims(t):
        dg = t['dims_gguf']
        return dg[1], dg[0], (dg[2] if len(dg) > 2 else 1)  # n, k, experts

    for il in range(main_layers):
        pre = f'blk.{il}.'
        names = {x[len(pre):-len('.weight')]: x for x in tensors if x.startswith(pre) and x.endswith('.weight')}
        # dense attention / linear attention projections
        for base in ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'attn_qkv', 'attn_gate', 'ssm_out',
                     'ssm_alpha', 'ssm_beta', 'attn_q_a', 'attn_q_b', 'attn_kv', 'attn_output_b',
                     'attn_compressor_kv', 'attn_compressor_gate', 'indexer.attn_q_b', 'indexer.proj',
                     'indexer_compressor_kv', 'indexer_compressor_gate', 'attn_output_a'):
            if base not in names:
                continue
            t = tensors[names[base]]
            n, k, e = dims(t)
            if base == 'attn_output_a':
                native[('attn_output_a(grouped)', qname(t['qtype']), n, k, 1, 'grouped_output_projection')] += 1
                continue
            for r, (nl, kl) in enumerate(tp_local(arch, names[base], t['qtype'], n, k, world)[:1]):
                add(base, [names[base]], t['qtype'], nl, kl, 'dense')
        # FFN
        if 'ffn_gate' in names and 'ffn_up' in names:  # dense FFN
            g, u = tensors[names['ffn_gate']], tensors[names['ffn_up']]
            gn, gk, _ = dims(g)
            mergeable = (arch.arch in ('qwen3', 'qwen35') and g['qtype'] == u['qtype'] and
                         dims(g) == dims(u) and g['qtype'] in QT)
            if mergeable:
                nl, kl = tp_local(arch, pre + 'ffn_gate_up.weight', g['qtype'], 2 * gn, gk, world)[0]
                add('ffn_gate_up', [names['ffn_gate'], names['ffn_up']], g['qtype'], nl, kl, 'dense', merged=True)
            else:
                for base in ('ffn_gate', 'ffn_up'):
                    t = tensors[names[base]]
                    n, k, _ = dims(t)
                    nl, kl = tp_local(arch, names[base], t['qtype'], n, k, world)[0]
                    add(base, [names[base]], t['qtype'], nl, kl, 'dense')
        if 'ffn_down' in names:
            t = tensors[names['ffn_down']]
            n, k, _ = dims(t)
            nl, kl = tp_local(arch, names['ffn_down'], t['qtype'], n, k, world)[0]
            add('ffn_down', [names['ffn_down']], t['qtype'], nl, kl, 'dense')
        if 'ffn_gate_exps' in names:
            g, u = tensors[names['ffn_gate_exps']], tensors[names['ffn_up_exps']]
            gn, gk, ge = dims(g)
            if g['qtype'] == u['qtype'] and dims(g) == dims(u) and g['qtype'] in QT:
                nl, kl = tp_local(arch, pre + 'ffn_gate_up_exps.weight', g['qtype'], 2 * gn, gk, world)[0]
                add('ffn_gate_up_exps', [names['ffn_gate_exps'], names['ffn_up_exps']], g['qtype'], nl, kl,
                    'moe', experts=ge, topk=arch.topk, channels=1, merged=True)
            else:
                for base in ('ffn_gate_exps', 'ffn_up_exps'):
                    t = tensors[names[base]]
                    n, k, e = dims(t)
                    nl, kl = tp_local(arch, names[base], t['qtype'], n, k, world)[0]
                    add(base, [names[base]], t['qtype'], nl, kl, 'moe', experts=e, topk=arch.topk, channels=1)
            t = tensors[names['ffn_down_exps']]
            n, k, e = dims(t)
            nl, kl = tp_local(arch, names['ffn_down_exps'], t['qtype'], n, k, world)[0]
            add('ffn_down_exps', [names['ffn_down_exps']], t['qtype'], nl, kl, 'moe', experts=e,
                topk=arch.topk, channels=arch.topk)
        if 'ffn_gate_shexp' in names:
            g, u = tensors[names['ffn_gate_shexp']], tensors[names['ffn_up_shexp']]
            gn, gk, _ = dims(g)
            if arch.arch == 'qwen35moe' and g['qtype'] == u['qtype'] and dims(g) == dims(u) and g['qtype'] in QT:
                nl, kl = tp_local(arch, pre + 'ffn_gate_up_shexp.weight', g['qtype'], 2 * gn, gk, world)[0]
                add('ffn_gate_up_shexp', [names['ffn_gate_shexp'], names['ffn_up_shexp']], g['qtype'], nl, kl,
                    'dense', merged=True)
            else:
                for base in ('ffn_gate_shexp', 'ffn_up_shexp'):
                    t = tensors[names[base]]
                    n, k, _ = dims(t)
                    nl, kl = tp_local(arch, names[base], t['qtype'], n, k, world)[0]
                    add(base, [names[base]], t['qtype'], nl, kl, 'dense')
            t = tensors[names['ffn_down_shexp']]
            n, k, _ = dims(t)
            nl, kl = tp_local(arch, names['ffn_down_shexp'], t['qtype'], n, k, world)[0]
            add('ffn_down_shexp', [names['ffn_down_shexp']], t['qtype'], nl, kl, 'dense')
    if 'output.weight' in tensors:
        t = tensors['output.weight']
        n, k, _ = dims(t)
        nl, kl = tp_local(arch, 'output.weight', t['qtype'], n, k, world)[0]
        add('output', ['output.weight'], t['qtype'], nl, kl, 'dense')
    # Aggregate identical operator families (same contract) with per-token call counts.
    fam = defaultdict(lambda: dict(calls=0, tensors=[]))
    for o in ops:
        key = (o['kind'], o['q'], o['n'], o['k'], o['experts'], o['topk'], o['channels'], o['role'])
        fam[key]['calls'] += 1
        fam[key]['tensors'] += o['tensors']
    families = []
    for (kind, q, n, k, e, topk, ch, role), v in sorted(fam.items(), key=lambda x: (x[0][7], x[0][1])):
        families.append(dict(role=role, kind=kind, qtype=q, format=QT[q], n=n, k=k, experts=e, topk=topk,
                             channels=ch, calls_per_token=v['calls'], example_tensors=v['tensors'][:2],
                             compute='bf16' if kind == 'moe' else 'f16'))
    natives = [dict(role=r, format=f, n=n, k=k, experts=e, reason=why, count=c)
               for (r, f, n, k, e, why), c in sorted(native.items())]
    return dict(model=model, repo=d['repo'], revision=d['revision'], tp=world,
                architecture=arch.arch, main_layers=main_layers, families=families, not_quactlize=natives)


def main():
    out = dict(schema='qgemv.workload-manifest.v1', caller='DrXuQian/llama.cpp pr/kpack-ffn 30cdcb24',
               quactlize_baseline='5b5b9a455ce9800cf85d3a2f4d707fa5207b4072', m_values=list(range(1, 9)),
               models=[model_ops(m) for m in MODELS])
    shapes = {}
    for m in out['models']:
        for f in m['families']:
            key = (f['kind'], f['qtype'], f['n'], f['k'], f['experts'], f['topk'], f['channels'])
            shapes.setdefault(key, []).append((m['model'], f['role'], f['calls_per_token']))
    out['unique_shapes'] = [dict(kind=k[0], qtype=k[1], format=QT[k[1]], n=k[2], k=k[3], experts=k[4], topk=k[5],
                                 channels=k[6], consumers=v) for k, v in sorted(shapes.items())]
    path = ROOT / 'inventory/workload_manifest.json'
    path.write_text(json.dumps(out, indent=1) + '\n')
    for m in out['models']:
        print(f"== {m['model']} TP{m['tp']} {m['architecture']} layers={m['main_layers']}")
        for f in m['families']:
            print(f"   {f['role']:22s} {f['kind']:5s} {f['format']:5s} N={f['n']:6d} K={f['k']:5d} "
                  f"E={f['experts']:3d} top{f['topk']} ch{f['channels']} calls={f['calls_per_token']}")
        for f in m['not_quactlize']:
            print(f"   [native] {f['role']:22s} {f['format']:7s} N={f['n']} K={f['k']} E={f['experts']} {f['reason']} x{f['count']}")
    print('unique quactlize shapes:', len(out['unique_shapes']))


if __name__ == '__main__':
    main()
