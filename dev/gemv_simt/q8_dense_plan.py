"""Frozen regressions plus existing dense controls; not a new full sweep."""
from dataclasses import asdict
from dev.gemv_simt.general_plan import configs,workload
from dev.gemv_simt.q8_dense_source import ARMS


def inventory():
    points=[]
    for n,k in ((512,2048),(1024,5120),(4096,2048),(4096,4096),(5120,8192),(8192,5120)):
        points.append(dict(workload=asdict(workload(8,'dense',1,n,k)),
                           numeric_m=list(range(1,9)) if n<=1024 else [1,8],timing_m=[1,8]))
    for n,k in ((1024,5120),(5120,8192)):
        points.append(dict(workload=asdict(workload(8,'dense',1,n,k,'bf16')),numeric_m=[1,8],timing_m=[1,8]))
    return points


def plan():
    points=[dict(point=i,**row) for i,row in enumerate(inventory())]
    return dict(schema='q8-dense-ablation-v1',points=points,configs=configs(8),arms=list(ARMS),
                numeric_pairs=sum(len(r['numeric_m']) for r in points)*len(configs(8))*(len(ARMS)-1),
                timing_pairs=sum(len(r['timing_m']) for r in points)*len(configs(8))*(len(ARMS)-1),
                profiles=[dict(point=1,m=1,config=1),dict(point=4,m=8,config=0)],
                acu_reports=42,production_changed=False,
                score='MATCHED_CLOCK_CE_CYCLES',scope='Q8_DENSE_S1_CAUSAL_READER_TEST_NOT_GLOBAL_SELECTION',
                frequency_policy='DO_NOT_SCALE_OR_MERGE_1GHZ_TIMES_WITH_1P7GHZ_TIMES',
                omissions=['SPLIT_K','MOE_CHANGES','MODEL_TPOT','PRODUCTION_ROUTING'])
