"""Bounded before/after implementation gate, not a configuration Cartesian sweep."""
from dataclasses import asdict
from dev.gemv_simt.general_plan import inventory as previous, configs, workload

QTYPES = (10, 11, 12, 13, 14)
SPLITS = (1, 2, 4, 8)


def inventory():
    result=[]
    for old in previous():
        if old['workload']['qtype'] not in QTYPES:
            continue
        row=dict(old)
        row['numeric_m']=[m for m in old['numeric_m'] if m in (1,3,8)]
        result.append(dict(row,point=len(result)))
    # The generic implementation must also compete with the four current
    # model-specific incumbents, not only with the unfixed generic baseline.
    for q,n,k,channels in ((12,1024,2048,1),(13,2048,512,8),
                          (12,1024,3072,1),(13,3072,512,8)):
        result.append(dict(point=len(result),workload=asdict(workload(q,'indexed',1,n,k,'bf16',256,channels=channels)),
                           numeric_m=[1,8],timing_m=[1,8],kind='shipping-fixed-path-control'))
    return result


def plan():
    points=inventory()
    return dict(schema='simt-common-split-v1',points=points,splits=list(SPLITS),
        configs={str(q):configs(q) for q in QTYPES},
        numeric_pairs=sum(len(p['numeric_m'])*len(configs(p['workload']['qtype']))*len(SPLITS) for p in points),
        timing_pairs=sum(len(p['timing_m'])*len(configs(p['workload']['qtype']))*len(SPLITS) for p in points),
        baseline='CURRENT_PRODUCTION_RECIPE_LAUNCH_NOT_EXPERIMENTAL_Q6_CONTROL',
        score='SUM_OF_SEPARATELY_MEASURED_PRODUCER_AND_REDUCER',
        rounds=6,samples=5,scope='IMPLEMENTATION_ADMISSION_NOT_NEW_CLOCK_CONFIG_SELECTION',
        production_changed=False,omissions=['Q8','TC_COMPARISON','MODEL_TPOT','NEW_CLOCK_GLOBAL_SELECTION'])
