"""Separate bank mapping, cooperative metadata, N width, and IO load depth.

R3 incumbents remain controls. Components are independently timed; only the
final full-call confirmation combines them. No provider or production change.
"""
import re
from . import round2
from .spec import workloads

ROUND=4
DEQUANT_VARIANTS=range(22)
DYNAMIC_VARIANTS=range(14,22)
IO_VARIANTS=(0,1,2,3)
DYNAMIC_IO=3
CONTROLS=(8,8,8,8,10,12,8)
BANK={8:14,10:15,12:16}
GEOMETRY={8:(32,128),10:(32,256),12:(16,256),14:(32,128),15:(32,256),
          16:(16,256),17:(32,128),18:(32,256),19:(16,256),20:(64,128),21:(64,256)}

def pool(point):
    result=[]
    for dtype,storage in (('bf16',1),('fp16',0)):
        io=2 if point in (4,6)or(storage and point==5)else 1
        for variant in (CONTROLS[point],BANK[CONTROLS[point]],17,18,19,20,21):
            result.append(dict(name=f'r4-{dtype}-dq{variant}',storage=storage,vector_io=io,
                dequant_variant=variant,math_mode=16,algorithm=-1))
        result.append(dict(name=f'r4-{dtype}-io3',storage=storage,vector_io=3,
            dequant_variant=CONTROLS[point],math_mode=16,algorithm=-1))
    return result

def stages(point,arm):
    if arm['vector_io']==3:return ('input','output')
    if arm['dequant_variant']==CONTROLS[point]:return ('dequant','input','output')
    return ('dequant',)

def composed(point,rows):
    result=[]
    for storage in (1,0):
        candidates=[r for r in rows if r['arm']['storage']==storage]
        control=next(r for r in candidates if r['arm']['vector_io']!=3 and
                     r['arm']['dequant_variant']==CONTROLS[point])
        dq=min((r for r in candidates if 'dequant'in r['median_us']),key=lambda r:r['median_us']['dequant'])
        io=min((r for r in candidates if 'input'in r['median_us']),
               key=lambda r:r['median_us']['input']+r['median_us']['output'])
        result.append(control['arm']|dict(name=f'composed-{storage}-control'))
        result.append(dq['arm']|dict(name=f'composed-{storage}-screen-choice',vector_io=io['arm']['vector_io']))
    return result

def profile_jobs(results):
    admitted={r['point']:r for r in results if r['status']=='PASS'}
    # Eight ABBA pairs: matched bank/metadata on large Q4/Q6, small-shape
    # winners, and large input/output ILP. Add at most two unmatched large
    # winners so a wider geometry does not require a second Box handoff.
    pairs=[(0,1,'r4-fp16-dq8','r4-fp16-dq14'),(0,1,'r4-fp16-dq14','r4-fp16-dq17'),
           (5,1,'r4-bf16-dq12','r4-bf16-dq16'),(5,1,'r4-bf16-dq16','r4-bf16-dq19'),
           (0,3,'r4-bf16-dq8','r4-bf16-io3'),(5,0,'r4-bf16-dq12','r4-bf16-io3')]
    for point,storage in ((4,0),(6,1)):
        if point not in admitted:raise ValueError('R4 point not admitted')
        choices=[r for r in admitted[point]['arms']if r['arm']['storage']==storage and
                 r['arm']['dequant_variant']>=14 and 'dequant'in r['median_us']]
        choice=min(choices,key=lambda r:r['median_us']['dequant'])['arm']['name']
        pairs.append((point,1,f'r4-{"bf16"if storage else"fp16"}-dq{CONTROLS[point]}',choice))
    for point,storage,covered in ((0,0,{8,14,17}),(5,1,{12,16,19})):
        if point not in admitted:raise ValueError('R4 point not admitted')
        choices=[r for r in admitted[point]['arms']if r['arm']['storage']==storage and 'dequant'in r['median_us']]
        choice=min(choices,key=lambda r:r['median_us']['dequant'])['arm']
        if choice['dequant_variant']not in covered:
            pairs.append((point,1,f'r4-{"bf16"if storage else"fp16"}-dq{CONTROLS[point]}',choice['name']))
    jobs=[]
    for pair,(point,stage,control,candidate)in enumerate(pairs):
        names={r['arm']['name']for r in admitted.get(point,{}).get('arms',[])}
        if not {control,candidate}<=names:raise ValueError('R4 profile arm not admitted')
        for repeat,order in enumerate(((control,candidate),(candidate,control))):
            for arm in order:jobs.append(dict(point=point,stage=stage,arm=arm,pair=pair,repeat=repeat,
                                              role='control'if arm==control else'candidate'))
    return jobs

def check_profile(path,job):
    arm=next(a for a in pool(job['point'])if a['name']==job['arm']);w=workloads()[job['point']]
    if job['stage']==1:
        variant=arm['dequant_variant'];tn,tk=GEOMETRY[variant]
        expected='dense_dequant_pair'if variant<14 else'dense_dequant_cooperative'
        result=round2.check_profile(path,job,expected_kernel=expected,geometry=([w.n//tn,w.k//tk,1],128))
        args=(str(w.qtype-10),'true'if arm['storage']else'false',str(tn),str(tk),'true'if variant>=17 else'false')
        match=re.search(expected+r'<\(gguf_scale::KType\)([234]), (true|false), (\d+)u?, (\d+)u?, (true|false)>',result['launches'][0][1])
        if not match or match.groups()!=args:raise ValueError('ACU cooperative specialization differs')
        details=dict(variant=variant,tile=[tn,tk],cooperative_metadata=variant>=17)
    elif job['stage']in (0,3):
        vector=arm['vector_io'];count=w.m*(w.k if job['stage']==0 else w.n)
        expected={1:'dense_convert_vec8',2:'dense_convert_stripe4',3:'dense_convert_full4'}[vector]
        if count%4096:raise ValueError('R4 IO profile requires full flat CTAs')
        result=round2.check_profile(path,job,expected_kernel=expected,
            geometry=([count//(4096 if vector==3 else 2048),1,1],256))
        args=('true'if arm['storage']else'false','true'if job['stage']==0 else'false')
        if vector!=3:args+=('true',)
        match=re.search(expected+r'<(true|false), (true|false)'+(r', (true|false)'if vector!=3 else'')+'>',result['launches'][0][1])
        if not match or match.groups()!=args:raise ValueError('ACU IO specialization differs')
        details=dict(vector_io=vector,input=job['stage']==0)
    else:raise ValueError('R4 ACU must isolate a helper, not GEMM or the complete call')
    return result|details|dict(storage=arm['storage'],
        candidate_binding='PRIOR_NUMERIC_RECEIPT_AND_KERNEL_SPECIALIZATION_AND_LAUNCH_GEOMETRY')

cycle_pairs=round2.cycle_pairs
