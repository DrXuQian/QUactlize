"""Bounded Q4 dense M2..8 reader scan and complete-call Tensor Core controls.

The SIMT lift changes only the activation/output row bases and grid.y.
Rows do not share a CTA; this is not a new multi-row B-reuse algorithm.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re

from dev.gemv_cuda.build import digest, replace_once
from dev.gemv_ppu import medium_refine, reader_followup, reader_reuse, small_latency
from dev.gemv_ppu.bload_source import REFERENCE_SHA256, reference_fp32
from dev.gemv_ppu.config_space import Config
from dev.gemv_ppu.run import SHAPES
from dev.gemv_ppu.run_cold_shapes import recipe as control_recipe

ROOT = medium_refine.ROOT
SCHEMA = 'quactlize.q4-smallm-ppu.v1'
MS = tuple(range(2, 9))
REVIEW = ROOT / 'docs/measurements/q4_medium_refine_20260912/review.json'
PACKAGES = ('q4-medium-refine-v1', 'q4-small-latency-v1',
            'q4-reader-followup-v1', 'q4-reader-reuse-v1', 'q4-config-sweep-v1',
            'q4-cold-shapes-v1', 'q4-h800-port-v1', 'q4-simt-ab-v1')


@dataclass(frozen=True)
class Candidate:
    family: str
    recipe: tuple

    @property
    def key(self):
        if self.family == 'meta':
            return small_latency.Candidate('meta', *self.recipe).key
        if self.family == 'medium':
            return medium_refine.Candidate(*self.recipe).key
        v, c, w, p = self.recipe
        return f'v{v}-c{c}-w{w}-p{p}'

    @property
    def arithmetic(self):
        return 'PER_WEIGHT_FP16' if self.family == 'meta' else 'FP32_GROUP_AFFINE'

    def geometry(self, n, k, m):
        if not 1 <= m <= 8 or (n, k) not in SHAPES:
            raise ValueError('small-M geometry outside contract')
        if self.family == 'meta':
            g = small_latency.Candidate('meta', *self.recipe).geometry(n, k)
        elif self.family == 'medium':
            g = medium_refine.Candidate(*self.recipe).geometry()
        else:
            g = Config(*self.recipe[1:]).geometry(n, k)
        return dict(g, grid_x=g['grid'], grid_y=m, grid=g['grid'] * m,
                    rows_per_cta=1, inter_cta_split=1)


@lru_cache(maxsize=1)
def closure():
    rows = json.loads(REVIEW.read_text())['combined_cases']
    if len(rows) != 6 or {tuple(r['shape'][1:]) for r in rows} != set(SHAPES):
        raise ValueError('M1 closure denominator differs')
    for row in rows:
        if row['shape'][0] != 1 or row['reference_delta_pct'] > 5:
            raise ValueError('M1 control was not admitted')
        if digest(ROOT / 'prebuilt/ppu0010' / row['package'] / 'manifest.json') != row['package_manifest_sha256']:
            raise ValueError('M1 control package changed')
    return {(r['shape'][1], r['shape'][2]): r for r in rows}


@lru_cache(maxsize=6)
def selected(n, k):
    key = closure()[n, k]['selected']
    if (n, k) == (512, 2048):
        c = small_latency.lookup(n, k, key)
        return Candidate('meta', (c.header, c.loading, c.a_mode, c.warps))
    if (n, k) == (1024, 5120):
        return Candidate('medium', medium_refine.lookup(key).recipe)
    return Candidate('reuse', tuple(map(int, re.fullmatch(r'v(\d+)-c(\d+)-w(\d+)-p(\d+)', key).groups())))


@lru_cache(maxsize=6)
def inventory(n, k):
    base = selected(n, k)
    if base.family == 'meta':
        rows = [Candidate('meta', (0, 1, 1, w)) for w in (4, 8, 16)]
        rows += [Candidate('meta', (1, 1, 1, 16))]
    elif base.family == 'medium':
        keys = ('p2-w8-r1-u0', 'p2-w10-r0-u1', 'p2-w16-r1-u0',
                'p4-w10-r1-u0', 'p4-w16-r1-u0', 'p4-w20-r0-u1',
                'p4-w20-r1-u0', 'p4-w20-r1-u1')
        rows = [Candidate('medium', medium_refine.lookup(key).recipe) for key in keys]
    else:
        _, c, w, p = base.recipe
        # M adds independent CTAs. Challenge the M1 winner with fewer K
        # workers and P4/P8, retaining the two proven H32 reuse variants.
        warps = sorted({4, 8, w})
        rows = [Candidate('reuse', (v, c, ww, pp)) for v in (6, 7)
                for ww in warps for pp in (4, 8)]
    if base not in rows:
        raise ValueError('previous winning reader omitted')
    return rows


def lookup(n, k, key):
    rows = [c for c in inventory(n, k) if c.key == key]
    if len(rows) != 1:
        raise ValueError('unknown small-M reader: ' + key)
    return rows[0]


def reference_recipes(n, k):
    # Keep the actual cold M1 reference winner and a small intra-CTA
    # N/K-warp neighborhood. Third field is NOT inter-CTA Split-K.
    old = tuple(control_recipe('raw-reference', n, k))
    return sorted({old, (1, 8, 1), (1, 8, 2), (2, 4, 2), (4, 4, 2)})


def requests():
    return [(12, 0, m, n, k, 1, m) for n, k in SHAPES for m in MS]


def plan():
    return dict(schema=SCHEMA, qtype=12, m=list(MS), shapes=[list(s) for s in SHAPES],
        denominator=len(MS) * len(SHAPES), cache='ROTATING_WEIGHTS_AT_LEAST_2_25_L2',
        cases=[dict(n=n, k=k, anchor=selected(n, k).key,
                    readers=[dict(key=c.key, family=c.family, recipe=c.recipe,
                                  geometry_m1=c.geometry(n, k, 1), arithmetic=c.arithmetic)
                             for c in inventory(n, k)],
                    reference=reference_recipes(n, k)) for n, k in SHAPES],
        tc_scope='CURRENT_POLICY_PLUS_ITS_FIVE_PARENT_UNION_SPLIT_1_2_4_8_NOT_GLOBAL_OPTIMUM',
        simt_scope='ONE_ROW_PER_CTA_SINGLE_MULTIROW_LAUNCH_NO_CROSS_ROW_B_REUSE',
        pruning=['Only Q4 dense M2..8; no indexed/grouped or other quant formats',
                 'Preserve every M1 winning body/config and nearby fewer-worker readers',
                 'No new shared-A, AIU, precision change or inter-CTA SIMT reducer',
                 'TC controls are real FQ parents with FP16 A; timing includes reducer'])


def payload(n, k):
    if (n, k) not in SHAPES:
        raise ValueError('unknown payload shape')
    return f'libq4_smallm_n{n}_k{k}.so'


def kernel_parts(n, k):
    family = selected(n, k).family
    if family == 'meta':
        s = small_latency.source(n, k, 'meta')
        marker = f'extern "C" int q4_latency_run_{n}_{k}('
        name, output = 'q4_small_pipeline', 'output'
    elif family == 'medium':
        s, marker = medium_refine.source(), 'extern "C" int q4_medium_run('
        name, output = 'q4_medium_refined', 'out_ptr'
    else:
        if (n, k) in reader_followup.SHAPES:
            s = reader_followup.source(n, k)
            marker = f'extern "C" int q4_followup_run_{n}_{k}('
        else:
            s = reader_reuse.source(n, k)
            marker = f'extern "C" int q4_reader_run_{n}_{k}('
        name, output = 'q4_reader_reuse', 'out_ptr'
    prefix = s[:s.index(marker)]
    pos = prefix.index('__global__ void ' + name + '(')
    start = prefix.rfind('\ntemplate<', 0, pos) + 1
    end = prefix.index('\n}\n', pos) + 3
    original = prefix[start:end]
    lifted = replace_once(original, '__global__ void ' + name + '(',
                          '__global__ void q4_smallm_' + family + '(')
    opening = lifted.index('{', lifted.index('__global__')) + 1
    bases = ('\n    a_ptr=static_cast<__half const*>(a_ptr)+size_t(blockIdx.y)*K;\n'
             f'    {output}+=size_t(blockIdx.y)*N;\n')
    lifted = lifted[:opening] + bases + lifted[opening:]
    return prefix, original, lifted, name


def template_args(c, n, k):
    if c.family == 'meta':
        return (*c.recipe, n, k)
    if c.family == 'medium':
        p, w, r, u = c.recipe
        mode = int(p == 2)
        return (r, u, 1, mode, mode, 4, w, p, n, k)
    return (*c.recipe, n, k)


def source(n, k):
    prefix, original, lifted, name = kernel_parts(n, k)
    body = prefix + '\nnamespace QKG_CONCAT(kpack_q,QKG_QTYPE) {\n' + lifted + '\n}\n'
    body += '''extern "C" int q4_smallm_run(int id,int rows,int control,
        void const* a,void const* low,void const* units,void* output,void* stream) {
    if(rows<1 || rows>8 || (control!=0 && control!=1) || !a || !low || !units || !output ||
       (uintptr_t(a)&15) || (uintptr_t(low)&15) || (uintptr_t(units)&15) || (uintptr_t(output)&3)) return QKG_INVALID;
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    using namespace QKG_CONCAT(kpack_q,QKG_QTYPE);
'''
    for i, c in enumerate(inventory(n, k)):
        g = c.geometry(n, k, 1)
        t = ','.join(map(str, template_args(c, n, k)))
        args = 'a,static_cast<uint8_t const*>(low),static_cast<uint8_t const*>(units),static_cast<float*>(output)'
        control_args = f'static_cast<__half const*>(a)+size_t(row)*{k},static_cast<uint8_t const*>(low),static_cast<uint8_t const*>(units),static_cast<float*>(output)+size_t(row)*{n}'
        body += f'''    if(id=={i}) {{
        if(control) {{ // Untimed original-body per-row numerical control only.
            for(int row=0;row<rows;++row)
                {name}<{t}><<<{g['grid']},{g['threads']},0,static_cast<hggcStream_t>(stream)>>>({control_args});
        }} else q4_smallm_{c.family}<{t}><<<dim3({g['grid']},rows),{g['threads']},0,static_cast<hggcStream_t>(stream)>>>({args});
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }}
'''
    return body + '    return QKG_INVALID;\n}\n'


def reference_source():
    path = ROOT / 'dev/gemv_ppu/reference/gemv_ref.cuh'
    if digest(path) != REFERENCE_SHA256:
        raise ValueError('raw reference differs')
    body = reference_fp32(path.read_text(), ppu=True)
    body += '''
extern "C" int q4_smallm_reference(int c,int w,int kw,int rows,int n,int k,
        void const* a,void const* raw,void* output,void* stream) {
    if(rows<1 || rows>8 || !a || !raw || !output || (uintptr_t(a)&15) ||
       (uintptr_t(raw)&15) || (uintptr_t(output)&3)) return -1;
    if(hggcGetLastError()!=hggcSuccess) return -2;
'''
    for n, k in SHAPES:
        for c, w, kw in reference_recipes(n, k):
            body += f'''    if(n=={n} && k=={k} && c=={c} && w=={w} && kw=={kw}) {{
        q4k_gemv_fp32::launch_q4k_gemv<{c},{w},{kw}>(static_cast<half const*>(a),
            static_cast<q4k_gemv_fp32::block_q4_K const*>(raw),static_cast<float*>(output),rows,n,k,static_cast<hggcStream_t>(stream));
        return int(hggcGetLastError());
    }}
'''
    return body + '    return -1;\n}\n'


def access(c, n, k, m, bases=None):
    if c.family == 'meta':
        base = small_latency.access(small_latency.Candidate('meta', *c.recipe), n, k, bases)
    elif c.family == 'medium':
        base = medium_refine.access(medium_refine.Candidate(*c.recipe), bases)
    else:
        base = reader_reuse.access(reader_reuse.Reader(c.recipe[0], Config(*c.recipe[1:])), n, k, bases)
    # Warp footprints are unchanged, but one multirow call executes M copies
    # of those requests. Do not confuse logical lane bytes with unique B
    # storage/DRAM bytes or silently label an M1 total as an M-row total.
    total_name = 'logical_lane_bytes_per_call' if c.family == 'reuse' else 'logical_global_lane_bytes'
    per_row = dict(base[total_name])
    base[total_name] = {name:m*value for name,value in per_row.items()}
    base['logical_lane_bytes_one_row'] = per_row
    return dict(base, shape=[m, n, k], multirow_geometry=c.geometry(n, k, m),
                row_a_offset_bytes=2*k, row_output_offset_bytes=4*n,
                weight_row_offset_bytes=0, unique_weight_bytes=n*k*9//16,
                cross_row_b_reuse='CACHE_ONLY_NOT_EXPLICIT')


def verify(bundle, *, sources=True):
    from quactlize.runtime.compiler import Compiler, FLAGS, validate_parent
    from quactlize.runtime.tuning import digest as identity_digest

    # Freeze within a scan, but recheck the actual receipt/package bindings
    # at each admission boundary rather than trusting a stale memoized plan.
    closure.cache_clear()
    selected.cache_clear()
    inventory.cache_clear()
    m = json.loads((bundle / 'manifest.json').read_text())
    if m['schema'] != SCHEMA or m['plan'] != json.loads(json.dumps(plan())) or m['production_changed'] is not False:
        raise ValueError('small-M manifest contract differs')
    if m['review_sha256'] != digest(REVIEW):
        raise ValueError('M1 selected receipt differs')
    for name, row in m['payloads'].items():
        path = (bundle / name).resolve(strict=True)
        if not path.is_relative_to(bundle.resolve()) or digest(path) != row['sha256'] or path.read_bytes()[:4] != b'\x7fELF':
            raise ValueError('small-M image differs: ' + name)
    if set(m['payloads']) != {payload(n,k) for n,k in SHAPES} | {'libq4_smallm_reference.so'}:
        raise ValueError('small-M payload denominator differs')
    for row in m['modules']:
        parent = row['parent']
        validate_parent(parent)
        if parent['qtype'] != 12 or parent['route'] != 'fq-dense' or parent['ap'] != 0:
            raise ValueError('TC control changed activation/weight arithmetic')
        if (row['identity']['flags'] != FLAGS or
            row['identity']['generator'] != digest(ROOT/'quactlize/runtime/compiler.py') or
            row['key'] != identity_digest(dict(identity=row['identity'],parent=parent,source=Compiler.source(None,parent,'')))):
            raise ValueError('TC generated parent identity differs')
        path = (bundle / row['path']).resolve(strict=True)
        if not path.is_relative_to(bundle.resolve()) or digest(path) != row['sha256'] or path.read_bytes()[:4] != b'\x7fELF':
            raise ValueError('TC image differs')
    selected_requests = m['tc_selection']
    if [r['request'] for r in selected_requests] != [list(r) for r in requests()] or any(r['status'] != 'SELECTED' for r in selected_requests):
        raise ValueError('TC policy request denominator differs')
    parents = {r['parent']['symbol']: r['parent'] for r in m['modules']}
    if {r['parent'] for r in selected_requests} != set(parents) or len(parents) != len(m['modules']):
        raise ValueError('TC parent union differs')
    if digest(bundle / 'isa-stats.json') != m['isa_sha256']:
        raise ValueError('small-M ISA receipt differs')
    for name, sha in m['source_hashes'].items() if sources else []:
        path = (ROOT / name).resolve(strict=True)
        if not path.is_relative_to(ROOT) or digest(path) != sha:
            raise ValueError('small-M source differs: ' + name)
    return m
