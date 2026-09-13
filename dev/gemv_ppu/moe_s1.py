"""Bounded Q4 indexed MoE S1 port, ahead of TC/SIMT policy selection."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from dev.gemv_cuda.build import digest

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='quactlize.q4-moe-s1.v1'
# Four previously measured MoE weight families, plus merged gate/up outputs.
SHAPES=((512,2048),(512,3072),(2048,512),(3072,512),(1024,2048),(1024,3072))
TOKENS=tuple(range(1,9))


@dataclass(frozen=True)
class Recipe:
    reader: int
    variant: int
    warps: int
    values: int

    @property
    def key(self):
        return f'r{self.reader}-v{self.variant}-w{self.warps}-p{self.values}'

    @property
    def tile_n(self): return 8 if self.reader==0 else 4*self.values

    @property
    def arithmetic(self): return 'PER_WEIGHT_FP16' if self.reader==0 else 'FP32_GROUP_AFFINE'

    def geometry(self,n,k,rows):
        workers=self.warps*(4 if self.reader==0 else 8)
        groups=k//(128 if self.reader==0 else 32)
        divisor=self.warps if self.reader==0 else workers
        return dict(grid=rows*n//self.tile_n,threads=self.warps*32,tile_n=self.tile_n,
                    k_workers=workers,k_passes=(groups+divisor-1)//divisor,
                    shared_bytes=self.warps*self.tile_n*4,rows_per_cta=1,split=1)


def inventory():
    # No new reader arithmetic. Preserve the small-N winner and challenge it
    # with fewer K workers once eight or more expert rows provide concurrency.
    return ([Recipe(0,0,w,1) for w in (4,8,16)] +
            [Recipe(1,2,20,4)] +
            [Recipe(2,v,w,p) for v in (6,7) for w in (2,4,8) for p in (4,8)])


def lookup(key):
    hits=[r for r in inventory() if r.key==key]
    if len(hits)!=1: raise ValueError('unknown explicit S1 recipe: '+key)
    return hits[0]


def payload(n,k): return f'libq4_moe_s1_n{n}_k{k}.so'


def source(n,k):
    if (n,k) not in SHAPES: raise ValueError('unlisted MoE shape')
    s='#include <hggc_runtime.h>\n#include "q4_s1_kernel.cuh"\n'
    s+='using namespace quactlize::execution::q4_s1;\n'
    s+='static bool compiled(qkg_q4_s1_config_v1 const& f) {\n    return '
    s+=' ||\n        '.join(f'(f.reader=={r.reader} && f.variant=={r.variant} && f.warps=={r.warps} && f.values=={r.values})' for r in inventory())+';\n}\n'
    s+=f'''extern "C" int quactlize_q4_s1_query_v1(qkg_call_v1 const* c,qkg_q4_s1_config_v1 const* f,
    quactlize_ppu_placed_arrangement_v2 const* a,qkg_sizes_v1* out) {{
    if (!c || !f || !out) return QKG_INVALID;
    if (c->n!={n} || c->k!={k}) return QKG_SHAPE;
    qkg_sizes_v1 sizes{{}};
    int rc=validate(*c,*f,a,sizes);
    if (rc) return rc;
    if (!compiled(*f)) return QKG_INVALID;
    *out=sizes;
    return QKG_OK;
}}
extern "C" int quactlize_q4_s1_run_v1(qkg_call_v1 const* c,qkg_q4_s1_config_v1 const* f,
    quactlize_ppu_placed_arrangement_v2 const* a) {{
    qkg_sizes_v1 sizes{{}};
    int rc=quactlize_q4_s1_query_v1(c,f,a,&sizes);
    if (rc) return rc;
    rc=buffers(*c,sizes);
    if (rc) return rc;
'''
    for r in inventory():
        s+=f'''    if(f->reader=={r.reader} && f->variant=={r.variant} && f->warps=={r.warps} && f->values=={r.values})
        return launch<{r.reader},{r.variant},{r.warps},{r.values},{n},{k}>(*c);
'''
    return s+'    return QKG_INVALID;\n}\n'


def access(r,n,k,rows,input_type,bases):
    # Reuse the exact lane-address derivation of the admitted dense bodies.
    from dev.gemv_ppu import small_latency, reader_reuse
    from dev.gemv_ppu.config_space import Config
    if r.reader==0:
        model=small_latency.access(small_latency.Candidate('meta',r.variant,1,1,r.warps),n,k,bases)
    elif r.reader==1:
        model=small_latency.access(small_latency.Candidate('affine',1,0,0,r.warps,4,r.values),n,k,bases)
    else:
        model=reader_reuse.access(reader_reuse.Reader(r.variant,Config(4,r.warps,r.values)),n,k,bases)
    # A footprint must be recomputed for F32 (not double the transaction
    # counts). Individual entries use explicit byte addresses/widths.
    return dict(dense_f16_warp_model=model,input_type=input_type,
        f32_note='A requests use 4-byte source elements and register F16 rounding; see indexed_a_model',
        indexed_geometry=r.geometry(n,k,rows),weight_layout='UNCHANGED_CANONICAL_KPACK4',
        expert_low_stride_bytes=n*k//2,expert_unit_stride_bytes=n*k//16,
        cross_row_b_reuse='CACHE_ONLY_NO_EXPLICIT_STAGING')


def plan():
    return dict(schema=SCHEMA,shapes=[list(x) for x in SHAPES],tokens=list(TOKENS),experts=256,topk=8,
        input_type='F32_ROUNDED_TO_F16_IN_REGISTERS',output_type='F32',
        recipes=[asdict(r)|dict(key=r.key,arithmetic=r.arithmetic) for r in inventory()],
        scope='Q4_S1_INDEXED_OPERATOR_NOT_MOE_CHAIN_OR_PRODUCTION_SELECTOR',
        launch_scope='ONE_KERNEL_IDS_A_WEIGHT_LOOKUP_DOT_DIRECT_OUTPUT_NO_GATHER_SCATTER',
        pruning=['Q4 first; Q5/Q6 down paths are not covered by Q4 admission',
                 'Single-CTA-row readers only; no inter-CTA Split-K or new arithmetic',
                 'Four existing MoE families plus two fused gate/up widths',
                 'Three small-reader warp counts and a bounded C4 P4/P8 reuse neighborhood',
                 'Retain the actual small-N M1 winner and medium P4/W20 reader'])


def verify(bundle,sources=True):
    bundle=Path(bundle)
    m=json.loads((bundle/'manifest.json').read_text())
    if m['schema']!=SCHEMA or m['plan']!=plan(): raise ValueError('MoE S1 plan differs')
    if set(m['payloads'])!={payload(n,k) for n,k in SHAPES}: raise ValueError('MoE S1 payload union differs')
    for name,sha in m['payloads'].items():
        if digest(bundle/name)!=sha: raise ValueError('MoE S1 payload differs: '+name)
    if sources:
        for name,sha in m['source_hashes'].items():
            if digest(ROOT/name)!=sha: raise ValueError('MoE S1 source differs: '+name)
    return m
