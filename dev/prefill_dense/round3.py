"""Packed N-pair reuse, then exact integer-to-FP32 construction on that mapping.

Matched R2 controls; no new IO, cuBLAS algorithm, or production selection.
Keep shared exchange and conversion as separately measurable changes.
"""
from . import round2
from .spec import workloads
import re

ROUND=3
DEQUANT_VARIANTS=range(14)
DYNAMIC_VARIANTS=range(8,14)
# R2 review-1792635753263379836: same variant wins both output dtypes.
# These are incumbents, not new production table entries or a global optimum.
CONTROLS=(4,4,4,4,5,6,4)
PAIRS={4:(8,9),5:(10,11),6:(12,13)}

def pool(point):
    result=[]
    for dtype,storage in (('bf16',1),('fp16',0)):
        io=2 if point in (4,6)or(storage and point==5)else 1
        for variant in (CONTROLS[point],*PAIRS[CONTROLS[point]]):
            result.append(dict(name=f'r3-{dtype}-dq{variant}',storage=storage,vector_io=io,
                dequant_variant=variant,math_mode=16,algorithm=-1))
    return result

def stages(point,arm):return ('dequant',)

def composed(point,rows):
    result=[]
    for storage in (1,0):
        candidates=[r for r in rows if r['arm']['storage']==storage]
        control=next(r for r in candidates if r['arm']['dequant_variant']==CONTROLS[point])
        choice=min(candidates,key=lambda r:r['median_us']['dequant'])
        result.append(control['arm']|dict(name=f'composed-{storage}-control'))
        result.append(choice['arm']|dict(name=f'composed-{storage}-screen-choice'))
    return result

def profile_jobs(results):
    pairs=[(0,'r3-fp16-dq4','r3-fp16-dq8'),(0,'r3-fp16-dq8','r3-fp16-dq9'),
           (5,'r3-bf16-dq6','r3-bf16-dq12'),(5,'r3-bf16-dq12','r3-bf16-dq13'),
           (6,'r3-bf16-dq4','r3-bf16-dq8'),(4,'r3-fp16-dq5','r3-fp16-dq10')]
    admitted={r['point']:{a['arm']['name']for a in r['arms']}for r in results if r['status']=='PASS'}
    jobs=[]
    for pair,(point,control,candidate)in enumerate(pairs):
        if not {control,candidate}<=admitted.get(point,set()):raise ValueError('R3 profile arm not admitted')
        for repeat,order in enumerate(((control,candidate),(candidate,control))):
            for arm in order:jobs.append(dict(point=point,stage=1,arm=arm,pair=pair,repeat=repeat,
                                              role='control'if arm==control else'candidate'))
    return jobs

def check_profile(path,job):
    arm=next(a for a in pool(job['point'])if a['name']==job['arm'])
    variant=arm['dequant_variant'];w=workloads()[job['point']]
    tn,tk=(16,256)if variant in (6,12,13)else(32,128)if variant in (4,8,9)else(32,256)
    expected='dense_dequant_pair'if variant>=8 else'dense_dequant_unsigned'
    if job['stage']!=1:raise ValueError('R3 ACU must isolate dequant, not GEMM or the complete call')
    result=round2.check_profile(path,job,expected_kernel=expected,geometry=([w.n//tn,w.k//tk,1],128))
    # Numerical equivalence alone cannot distinguish Fast=false/true. Bind
    # the actual specialization as well, rather than trusting the arm label.
    name=result['launches'][0][1]
    match=re.search(expected+r'<\(gguf_scale::KType\)([234]), (true|false), (\d+), (\d+), (true|false|\d+)>',name)
    fast=variant in (9,11,13)
    expected_args=(str(w.qtype-10),'true'if arm['storage']else'false',str(tn),str(tk),
                   ('true'if fast else'false')if variant>=8 else'128')
    if not match or match.groups()!=expected_args:raise ValueError('ACU dtype/code conversion specialization differs')
    return result|dict(variant=variant,storage=arm['storage'],tile=[tn,tk],
        fast_fp32=fast,candidate_binding='PRIOR_NUMERIC_RECEIPT_AND_KERNEL_SPECIALIZATION_AND_LAUNCH_GEOMETRY')

cycle_pairs=round2.cycle_pairs
