"""Bounded structural experiment, NOT a best-of-all-configs sweep."""
from dataclasses import asdict
from quactlize.dispatch.table import Workload
from dev.gemv_simt.general_reader import QTYPES


def configs(q):
    if q==8:
        return [dict(variant=5,columns=8,warps=4,values=4,split=1,hoist=True),
                dict(variant=5,columns=4,warps=8,values=4,split=1,hoist=True)]
    result=[dict(variant=3,columns=4,warps=8,values=8,split=1),
            dict(variant=3,columns=4,warps=2,values=8,split=1),
            dict(variant=3,columns=8,warps=4,values=4,split=1)]
    if q==12:result.append(dict(variant=3,columns=4,warps=4,values=4,split=1))
    return result


def workload(q,mode,m,n=512,k=2048,compute='f16',experts=16,router='real',channels=8):
    return Workload(q,n,k,m,operation=mode,compute=compute,experts=1 if mode=='dense' else experts,
                    topk=8 if mode!='dense' else 1,channels=channels if mode!='dense' else 1,
                    storage='f32',input_stride=k,output_stride=n,router='' if mode=='dense' else router)


def inventory():
    # Reuse existing Q6 control geometry and the six Q4 cold-shape registry.
    # Numerical coverage M1..8 is done on each small fixture, without a cold
    # ring per M or an 8x expansion of ACU. Only M1/M8 have cold timing.
    points=[]
    for q in QTYPES:
        for compute in ('f16','bf16'):
            for mode in ('dense','indexed','grouped'):
                focus=mode=='indexed' and compute=='bf16'
                points.append(dict(workload=asdict(workload(q,mode,1,compute=compute,
                                                           experts=256 if focus else 16,
                                                           router='cluster' if focus else 'real')),
                                   numeric_m=list(range(1,9)),timing_m=[1,8],kind='all-format-control'))
    for q in QTYPES:
        for n,k in ((1024,5120),(4096,2048),(4096,4096),(5120,8192),(8192,5120)):
            points.append(dict(workload=asdict(workload(q,'dense',1,n,k)),numeric_m=[1,8],timing_m=[1,8],kind='cold-shape-extension'))
        points.append(dict(workload=asdict(workload(q,'indexed',8,1024,5120,'bf16',32)),
                           numeric_m=[8],timing_m=[8],kind='moe-shape-extension'))
        points.append(dict(workload=asdict(workload(q,'indexed',8,compute='bf16',experts=256,router='spread')),
                           numeric_m=[8],timing_m=[8],kind='spread-routing-control'))
    for q in QTYPES:
        points.append(dict(workload=asdict(workload(q,'indexed',1,3072,512,'bf16',16,channels=1)),
                           numeric_m=[1,3,8],timing_m=[1,8],kind='short-k-shared-activation-control'))
    return points


def plan(indices=None):
    all_points=inventory()
    selected=list(range(len(all_points))) if indices is None else indices
    if not selected or len(set(selected))!=len(selected) or any(i not in range(len(all_points)) for i in selected):
        raise ValueError('invalid or duplicate point inventory')
    points=[dict(point=i,**all_points[i]) for i in selected]
    return dict(schema='simt-general-readers-v1',points=points,point_count=len(points),
                numeric_config_pairs=sum(len(p['numeric_m'])*len(configs(p['workload']['qtype'])) for p in points),
                timing_config_pairs=sum(len(p['timing_m'])*len(configs(p['workload']['qtype'])) for p in points),
                configs={str(q):configs(q) for q in QTYPES},
                arithmetic='F32_STORAGE_F16_OR_BF16_ACTIVATION_ORDERED_F32_GROUP_AFFINE',
                scope='S1_READER_GENERALIZATION_NOT_GLOBAL_OPTIMUM_OR_MODEL_TPOT',
                omissions=['SPLIT_K','TWO_ROW_B_REUSE','PRODUCTION_ROUTING'],production_changed=False)
