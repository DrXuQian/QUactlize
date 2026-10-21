"""Bounded seven-family candidate inventory; not a production selector."""
from quactlize.dispatch.table import Workload

STAGES={0:'input',1:'dequant',2:'gemm',3:'output',4:'reset',-1:'complete'}

def workloads():
    return [Workload(q,n,k,2048,storage='f32',compute='f16') for q,n,k in
        [(12,25600,5120),(12,8192,5120),(12,5120,8192),(12,5120,25600),
         (12,1024,5120),(14,5120,25600),(14,1024,5120)]]

def arms():
    return [dict(name='bf16-old',storage=1,vector_io=0,dequant_variant=-1,math_mode=16,algorithm=-1),
            dict(name='bf16-vec',storage=1,vector_io=1,dequant_variant=-1,math_mode=16,algorithm=-1)]+[
        dict(name=f'{dtype}-packed{v}',storage=storage,vector_io=1,dequant_variant=v,math_mode=16,algorithm=-1)
        for dtype,storage in [('bf16',1),('fp16',0)] for v in (1,2,3)]+[
        dict(name=f'fp16-tensor{v}',storage=0,vector_io=1,dequant_variant=v,math_mode=16,algorithm=99)
        for v in (1,2,3)]

def legacy_config(w):
    # Explicit historical full-expansion controls; NOT SF config IDs.
    return 11 if w.qtype==12 else 5 if w.n>1024 else 2

def profile_jobs(results):
    """Bounded helper/Q6/provider probes, never cold rings inside ACU."""
    jobs=[]
    for point,stages in [(0,(2,3)),(5,(0,1)),(6,(1,))]:
        row=next(r for r in results if r['point']==point and r['status']=='PASS')
        pool=row['arms']
        for stage in stages:
            # Same-dtype old vs vector/packed, then the best full-FP16 path.
            bf=min((r for r in pool if r['arm']['storage']==1 and r['arm']['name']!='bf16-old'),
                   key=lambda r:r['median_us']['complete'])
            if stage==1:
                # Isolate the new exchange at the SAME dtype even if a full
                # call currently wins with the old dequant + faster IO.
                bf=min((r for r in pool if r['arm']['storage']==1 and r['arm']['dequant_variant']>0),
                       key=lambda r:r['median_us']['dequant'])
            fp=min((r for r in pool if r['arm']['storage']==0),key=lambda r:r['median_us']['complete'])
            for arm in ('bf16-old',bf['arm']['name'],fp['arm']['name']):
                jobs.append(dict(point=point,stage=stage,arm=arm))
    return jobs
