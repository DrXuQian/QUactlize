"""Eight-point causal experiment; not the model-wide MoE config sweep."""
from dataclasses import asdict
import json
from pathlib import Path

from dev.strategy.refresh import request,original_recipe
from dev.strategy.resources import module_spec
from dev.strategy.vector import spec,execution_contract
from quactlize.dispatch.table import Workload,Recipe
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ROOT=Path(__file__).resolve().parents[2]
COHORT='moe-token-pair'
COHORTS=(COHORT,COHORT+'-q4')


def workloads(cohort=COHORT):
    # Actual separate gate/up and down families in Qwen35 GGUF headers. Paired
    # G4/U4 physical-2N weights belong to the BACKUP fusion experiment instead.
    if cohort not in COHORTS:raise ValueError('unknown bounded MoE pair cohort')
    return [Workload(q,n,k,m,operation='indexed',compute='bf16',experts=256,topk=8,channels=ch,router=router)
            for q,n,k,ch in ((12,512,2048,1),(13,2048,512,8))
            if cohort==COHORT or q==12
            for m,router in ((1,'spread'),(4,'repeat2'),(8,'spread'),(8,'cluster'))]


def production_entry(row):
    prefix='quactlize::execution::'
    if row['kind']==0:return 'cutlass::device_kernel'
    if row['kind']==2:
        compute=row['request'][-1]
        if compute not in (0,1):raise ValueError('invalid Q4 production compute type')
        return prefix+'q4_decode::'+('kernel_bf16' if compute==1 else 'kernel')
    names={'common-packet-3d':'common_reader','register-reuse':'register_reuse',
           'register-reuse-fixed':'register_reuse_model','measured-constant-split':'measured_decode_kernel'}
    if row['implementation']['producer']=='q8-vector':
        return prefix+'simt::q8_vector::kernel'
    return prefix+'simt::'+names[row['implementation']['producer']]


def make_plan(originals,cohort=COHORT):
    ws=workloads(cohort)
    if [x['request'] for x in originals]!=[request(w) for w in ws]:
        raise ValueError('MoE pair production control inventory differs')
    points=[];tc={};vectors={}
    for w,old in zip(ws,originals):
        if old['status']:raise ValueError('current indexed production control unavailable')
        arms=[]
        def add(r,role,entry,aid=None):
            item=module_spec(w,r) if r.kind=='tc' else spec(w,r)
            key=digest(item);(tc if r.kind=='tc' else vectors)[key]=item
            arms.append(dict(id=aid or r.key,recipe=asdict(r),roles=[role],aliases=[],required=True,
                             reasons=['BOUNDED_SAME_EXPERT_REUSE_AB_NOT_CONFIG_SEARCH'],spec=item,
                             module=key,expected_producer=entry))
        original=original_recipe(old)
        add(original,'original',production_entry(old),'original')
        for c,warps,p in ((4,2,8),(8,4,4)):
            for split in (1,4):
                for rows in ((1,) if w.m==1 else (1,2)):
                    r=Recipe.make('simt','fq',variant=3,columns=c,warps=warps,values=p,split=split,moe_pair_rows=rows)
                    entry='quactlize::execution::simt::'+('common_reader' if rows==1 else 'moe_token_pair_reader')
                    add(r,'control' if rows==1 else 'rule',entry)
        sf=sorted({max(0,dict(a['recipe']['parameters']).get('dequant_config',-1))
                   for a in arms if a['recipe']['route']=='sf'})
        points.append(dict(id=w.key,workload=asdict(w),cohort='decode',selection=old,
            execution_contract=execution_contract(w),arms=arms,sf_configs=sf,
            same_expert_pair=True,excluded=[],consumers=[]))
    result=dict(schema='quactlize.smallm-asys.v1',phase='regression-asys',cohort=cohort,
        points=points,tc=tc,vectors=vectors,
        counts=dict(points=len(points),arms=sum(len(p['arms']) for p in points),tc_modules=len(tc),
                    vector_modules=len(vectors),families=len({(w.qtype,w.n,w.k) for w in ws})),
        timing_method='ASYS_GPU_ACTIVITY',selection_metric='MEAN_COMPONENT_SUM',noise_policy='DIAGNOSTIC_NOT_VETO',
        historical_times_reused=False,production_changed=False,native_comparison=False,
        scope='INDEXED_BF16_TWO_ROW_REUSE_CAUSAL_AB_NOT_FULL_SWEEP_OR_PRODUCTION_ADMISSION',
        sources={p:sha(ROOT/p) for p in ('dev/strategy/moe_token_pair_plan.py','dev/strategy/vector.py',
            'dev/strategy/moe_token_pair.py','dev/strategy/moe_token_pair.hpp','dev/strategy/moe_token_pair.cuh',
            'quactlize/execution/simt_common.cuh','quactlize/execution/simt_common_scope.hpp',
            'quactlize/execution/simt_strategy.hpp','quactlize/execution/measured_decode.inc')})
    result['sha256']=digest(result)
    return json.loads(json.dumps(result))
