#!/usr/bin/env python3
"""Rule-generated prototype configs for a set of shapes (plus the production baseline arm).

Warp layouts must match tools/mma_proto_tu.cu; code = wn*1000 + wk*10 + nb.
"""
import argparse, json
from pathlib import Path

COMBOS = {2: [(1,4),(1,8),(1,16),(2,4),(2,8),(4,2),(4,4)],
          4: [(1,4),(1,8),(2,2),(2,4),(4,1),(4,2)],
          8: [(1,2),(1,4),(2,1),(2,2),(2,4),(4,1),(4,2),(8,1)]}
BPG = {12: 1, 8: 1, 13: 2, 14: 2}
# Staged reader (tools/mma_staged_tu.cu): code = wn*1000 + wk*10 + stages.
STAGED = {2: [(8,1),(8,2),(16,1)],
          4: [(2,2),(2,4),(4,1),(4,2),(4,4),(8,1),(8,2)],
          8: [(1,2),(1,4),(2,1),(2,2),(2,4),(4,1),(4,2)]}


def staged_bytes(q, cta_n, wk, stages):
    """Mirror of mma_staged.cuh StageLayout: shared bytes of all stages."""
    sb = (2 if wk == 1 else wk) if q == 14 else wk
    pitch = cta_n * 2 + 16
    rows = (128 if q == 8 else 64) + {13: 16, 14: 32}.get(q, 0) + (8 if q == 8 else 0)
    units = sb * cta_n * 16 if q in (12, 13) else (sb // 2) * cta_n * 36 if q == 14 else 0
    per = (sb * rows * pitch + units + 127) // 128 * 128
    return stages * per


def staged_configs(q, n, k, m, rows, ts, splits, min_ctas):
    out = []
    us = 2 if q == 14 else 1
    tp = 1 if m <= 1 else 2
    for t in ts:
        for wn, wk in STAGED[t]:
            cta_n = wn * 8 * t
            if n % cta_n: continue
            tiles = n // cta_n
            for split in splits:
                if k % (256 * us * split): continue
                sb = k // 256 // split
                if sb < wk: continue
                ctas = tiles * rows * split
                if ctas < min_ctas: continue
                if split > 1 and tiles * rows >= 512: continue
                for st in (2, 3, 4):
                    b = staged_bytes(q, cta_n, wk, st)
                    if b > 256 * 1024 or b < wk * tp * cta_n * 4: continue
                    out.append(dict(t=t, w=wn*1000+wk*10+st, split=split))
    return out


def nbs(q, t):
    out = [2, 4]
    if BPG[q] == 1 and t != 8:
        out.append(8)
    return out


def configs(q, n, k, m, kind, rows, ts, splits, min_ctas):
    out = []
    for t in ts:
        for wn, wk in COMBOS[t]:
            cta_n = wn * 8 * t
            if n % cta_n: continue
            tiles = n // cta_n
            for split in splits:
                if k % (256 * split): continue
                sb = k // 256 // split
                if sb < wk: continue                       # every k-slice warp owns a superblock
                ctas = tiles * rows * split
                warps = ctas * wn * wk
                if ctas < min_ctas or warps < 256 or warps > 16384: continue
                if split > 1 and tiles * rows >= 512: continue   # enough CTAs without Split-K
                for nb in nbs(q, t):
                    out.append(dict(t=t, w=wn*1000+wk*10+nb, split=split))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib', required=True)
    ap.add_argument('--shapes', required=True, help='q:n:k:m[:channels] list (channels => MoE), comma separated')
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-base', action='store_true')
    ap.add_argument('--t', default='2,4,8')
    ap.add_argument('--splits', default='1,2,4,8')
    ap.add_argument('--min-ctas', type=int, default=32)
    ap.add_argument('--staged', action='store_true', help='staged reader configs (M<=2)')
    a = ap.parse_args()
    ts = [int(x) for x in a.t.split(',')]
    splits = [int(x) for x in a.splits.split(',')]
    cases = []
    for spec in a.shapes.split(','):
        f = spec.split(':')
        q, n, k, m = map(int, f[:4]); moe = len(f) > 4
        kind = 'moe' if moe else 'dense'
        compute = 1 if moe else 0
        base = dict(kind=kind, qtype=q, n=n, k=k, m=m, compute=compute)
        if moe: base.update(experts=256, topk=8, channels=int(f[4]) if f[4] else 1)
        rows = m * 8 if moe else 1
        pre = (f"e-q{q}-n{n}-k{k}-c{base['channels']}-m{m}" if moe else f"d-q{q}-n{n}-k{k}-m{m}")
        if not a.no_base:
            cases.append(dict(base, id=pre + '-base'))
        gen = (staged_configs(q, n, k, m, rows, ts, splits, a.min_ctas) if a.staged else
               configs(q, n, k, m, kind, rows, ts, splits, a.min_ctas))
        for c in gen:
            cases.append(dict(base, id=f"{pre}-t{c['t']}c{c['w']}s{c['split']}",
                              proto=dict(lib=a.lib, compute=compute, t=c['t'], w=c['w'], split=c['split'])))
    Path(a.out).write_text(json.dumps(cases, indent=0) + '\n')
    print(len(cases), 'cases')


if __name__ == '__main__':
    main()
