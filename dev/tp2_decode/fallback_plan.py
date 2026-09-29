"""Bounded production-fallback gate, with the last confirmed minima frozen."""
from dataclasses import asdict, replace
from dev.gemv_model.plan import Candidate, Point
from dev.tp2_decode import plan as prior

SCHEMA = 'quactlize.decode-fallback-gate.v1'
REFERENCE_MANIFEST = '4addcd47773099db4d636f1c0e0d51b8ac1982624edf109733e17ccca7546460'
RESULT_ARCHIVE = '5944aa7547e4dd5c9b87dae0f906102d7df5d4e39f42d7007c25731408fee075'
# Minimum of incumbent and confirmed candidates, NOT always the new candidate.
BEST = {
    'tp2-q8-qkv': '8', 'tp2-q8-attn-gate': '8', 'tp2-q8-attn-q': '2',
    'tp2-q8-out': '8', 'tp2-q8-shared-projection': '4',
    'tp2-q8-shared-down': '2', 'tp2-q8-attn-kv': '4',
    'tp2-q5-down': '1', 'q5-routed-down': 'incumbent',
    'q8-shared-down': '3', 'q8-ssm-out': 'incumbent',
}
MODEL_POINTS = tuple(p for p in prior.POINTS if p.name in BEST)
UNSEEN = (Point('unseen-q8-n768-k2048', 8, 768, 2048),
          Point('unseen-q8-n1536-k3072', 8, 1536, 3072))
REDUCERS = tuple(Point(f'reducer-q{q}-bf16', q, 768, 1024, 1) for q in (10,11,12,13,14))
POINTS = MODEL_POINTS + UNSEEN + REDUCERS
FIELDS = ('variant','columns','warps','values','split')


def record_config(value, name='production-fallback', *, vector_reduce=None):
    return asdict(Candidate(name, **{k:int(value[k]) for k in FIELDS},
                            vector_reduce=int(value['split'])>1 if vector_reduce is None else vector_reduce))


def kernel_names(p, cfg):
    """Exact production identities; checked against the emitted native image."""
    v,c,w,a,s=(cfg[k] for k in FIELDS)
    if p.q==8 and v>=4:
        if (v,c,w,a,s,p.n,p.k)==(5,8,4,4,8,2048,4096):
            producer='q8_vector::kernel_model<1,0,1,8,4,4,false,2048,4096,8>'
        elif (v,c,w,a,s,p.n,p.k)==(5,8,4,4,1,8192,2048):
            producer='q8_vector::kernel_model<1,0,1,8,4,4,true,8192,2048,1>'
        elif (v,c,w,a,s)==(5,4,2,4,1):
            producer='q8_vector::kernel_s1<1,0,1,4,2,4,true>'
        elif (v,c,w,a,s,p.n,p.k)==(5,4,8,4,1,4096,2048):
            producer='q8_vector::kernel_s1<1,0,1,4,8,4,false>'
        else:
            hoist=p.compute==0 and s==1 and v==5 and a==4 and (c,w) in ((8,4),(4,2),(4,8))
            producer=f'q8_vector::kernel<1,{p.compute},{v-4},{c},{w},{a},{str(hoist).lower()}>'
    elif p.q==13 and (v,c,w,a,s,p.compute,p.mode,p.n,p.k)==(3,4,2,8,1,1,2,2048,512):
        producer='register_reuse_model<13,1,3,4,2,8,1,3,2048,512>'
    else:
        changes=3 if p.q==13 and p.compute==1 and (v,c,w,a)==(3,4,2,8) else 0
        producer=f'register_reuse<{p.q},1,{v},{c},{w},{a},{p.compute},{changes}>'
    reducer=None if s==1 else (f'reduce_decode<{s},float>' if 'false,2048,4096,8>' in producer
                              else f'reduce_decode_rows<{s}>')
    return producer,reducer


def production_config(p,cfg):
    """Implementation metadata for the aligned M1 timing/profile call only."""
    result=record_config(cfg)
    producer,_=kernel_names(p,cfg)
    result['hoist']=p.q==8 and ('true>' in producer or 'true,' in producer)
    result['fixed']='kernel_model<' in producer or 'register_reuse_model<' in producer
    result['changes']=3 if p.q==13 and p.compute==1 and tuple(cfg[k] for k in FIELDS[:-1])==(3,4,2,8) else 0
    return result


def frozen_kernel_names(p,cfg):
    """Old execution image identities, not the changed source's launcher."""
    # JSON stores the TC tuple as a list, including [] for SIMT-only points.
    # Normalize just that representation; keep every geometry/precision guard.
    p=replace(p,tc=tuple(p.tc))
    v,c,w,a,s=(cfg[k] for k in FIELDS)
    if p.name=='q5-routed-down':
        if (v,c,w,a,s)!=(3,4,2,8,1):raise ValueError('frozen Q5 recipe differs')
        return 'register_reuse_model<13,1,3,4,2,8,1,3>',None
    if p.name=='q8-ssm-out':
        if (v,c,w,a,s)!=(5,8,4,4,8):raise ValueError('frozen Q8 recipe differs')
        return 'q8_vector::kernel_model<1,0,1,8,4,4,false,2048,4096,8>','reduce_decode<8,float>'
    if p not in UNSEEN+REDUCERS or (p.q==8 and v>=4):
        raise ValueError('frozen incumbent identity not declared')
    return (f'register_reuse<{p.q},1,{v},{c},{w},{a},{p.compute},0>',
            None if s==1 else f'register_reuse_reduce<{p.q}>')
