"""Host-only contracts for the model-driven, model-name-independent entry."""
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dev.model_tuning import inventory, plans, policy, runner
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.tuning import digest
from tools.gguf_internal_shape_inventory import _synthetic_gguf, read_gguf_header


def loader(extra=(), *, merged=False, up_type=12):
    tensors=[('blk.0.attn_q.weight',(2048,4096),8),
             ('blk.0.ffn_gate_shexp.weight',(2048,512),8),
             ('blk.0.ffn_up_shexp.weight',(2048,512),8),
             ('blk.0.ffn_down_exps.weight',(512,2048,256),13)]
    tensors += ([('blk.0.ffn_gate_up_exps.weight',(2048,1024,256),12)] if merged else
                [('blk.0.ffn_gate_exps.weight',(2048,512,256),12),
                 ('blk.0.ffn_up_exps.weight',(2048,512,256),up_type)])
    data=_synthetic_gguf([('general.name','not-a-hardcoded-model'),
        ('general.architecture','qwen35moe'),('qwen35moe.expert_count',256),
        ('qwen35moe.expert_used_count',8)],tensors+list(extra))
    header=read_gguf_header(io.BytesIO(data),'fixture')
    return lambda path:([dict(path=path,header=header)],header['tensors'])


def collected(**kwargs):
    return inventory.collect([dict(name='unseen-network',path='/fixture',tp=1)],
        ms=[1,4,128],loader=loader(),**kwargs)


def test_generic_models_actual_types_and_fusion_contracts():
    value=collected()
    assert value['status']=='READY' and not value['gpu_work'] and not value['weights_read']
    assert {p['workload']['qtype'] for p in value['points']}=={8,12,13}
    for p in value['points']:
        w,c=p['workload'],p['contract']
        assert w['compute']==('f16' if w['operation']=='dense' else 'bf16')
        assert w['storage']==c['input_type']==c['output_type']==c['accumulator']=='f32'
        if c['form']=='swiglu':
            assert c['weight_n']==2*w['n'] and c['output_n']==w['n']
            assert c['layout']==inventory.PAIRED and w['fusion']=='gate-up-swiglu'
        elif c['form']=='concat':
            assert w['n']==2*c['logical_n'] and w['fusion']=='none'
            assert not c['activation_included'] and c['scope']=='PROJECTION_ONLY'
    grouped=[p for p in value['points'] if p['contract']['form']=='projection' and p['workload']['qtype']==12]
    assert all(len(p['consumers'])==2 for p in grouped)


def test_deduplicate_across_models_but_preserve_consumers():
    one=collected()
    both=inventory.collect([dict(name=n,path='/'+n) for n in ('first','second')],
        ms=[1,4,128],loader=loader())
    assert len(one['points'])==len(both['points'])
    assert all({c['model'] for c in p['consumers']}=={'first','second'} for p in both['points'])


def test_merged_gateup_not_doubled_twice_and_mixed_pair_not_fused():
    value=inventory.collect([dict(name='x',path='/x')],ms=[1],loader=loader(merged=True))
    assert {p['contract']['weight_n'] for p in value['points']
            if p['workload']['operation']=='indexed' and p['contract']['form']!='projection'}=={1024}
    mixed=inventory.collect([dict(name='x',path='/x')],ms=[1],loader=loader(up_type=13))
    assert not [p for p in mixed['points'] if p['workload']['operation']=='indexed' and p['contract']['form']!='projection']
    assert any(e['reason']=='GATE_UP_TYPE_SHAPE_OR_PARTITION_MISMATCH' for e in mixed['excluded'])


def test_tp_requires_caller_extents_not_division():
    with pytest.raises(ValueError,match='no guessed'):
        inventory.collect([dict(name='x',path='/x',tp=2)],ms=[1],loader=loader())
    def partition(model,rows,tensors):
        records={}
        for t in tensors:
            k,n,*_=t['dims_gguf']
            down='down' in t['name']
            records[t['name']]={(r,t['qtype'],n if down else n//2,k//2 if down else k) for r in range(2)}
        return records,dict(source='test-only-caller-proof'),{}
    p=inventory.collect([dict(name='x',path='/x',tp=2)],ms=[1],loader=loader(),partitioner=partition)
    fused=[r for r in p['points'] if r['workload']['operation']=='indexed' and r['contract']['form']=='concat']
    assert all(r['workload']['n']==512 and r['workload']['k']==2048 for r in fused)
    assert all({c['rank'] for c in r['consumers']}=={0,1} for r in fused)


def test_k_split_does_not_move_nonlinearity_before_allreduce():
    def partition(model,rows,tensors):
        return {t['name']:{(r,t['qtype'],t['dims_gguf'][1],t['dims_gguf'][0]//2) for r in range(2)}
                for t in tensors},{},{}
    p=inventory.collect([dict(name='x',path='/x',tp=2)],ms=[1],loader=loader(),partitioner=partition)
    assert not [r for r in p['points'] if r['contract']['form']=='swiglu']
    assert any(e['reason']=='SWIGLU_REQUIRES_ALLREDUCE_BEFORE_EPILOGUE' for e in p['excluded'])


def test_inventory_tamper_and_unrecognized_format_are_visible():
    p=inventory.collect([dict(name='x',path='/x')],ms=[1],loader=loader(extra=[
        ('blk.1.ffn_gate_exps.weight',(2048,512,256),39)]))
    assert any(e['qtype']==39 and e['reason']=='UNSUPPORTED_DTYPE' for e in p['excluded'])
    p['points'][0]['workload']['m']=3
    with pytest.raises(ValueError,match='digest'):inventory.validate(p)


def test_one_dimensional_shared_router_is_native_not_invalid_matrix():
    p=inventory.collect([dict(name='x',path='/x')],ms=[1],loader=loader(extra=[
        ('blk.0.ffn_gate_inp_shexp.weight',(2048,),0)]))
    assert any(x['tensor']=='blk.0.ffn_gate_inp_shexp.weight' for x in p['ignored'])


def test_engine_partition_does_not_fake_paired_timings():
    p=collected();result=plans.partition(p)
    assert set(result['groups'])=={'smallm-dense','smallm-indexed','dense-prefill','moe-prefill'}
    pending={x['point'] for x in result['pending']}
    assert pending=={x['id'] for x in p['points'] if x['contract']['form']=='swiglu'}
    assert len(result['bindings'])+len(pending)==len(p['points'])


def point(m=1,n=1024,k=2048,*,indexed=False,router='spread'):
    w=Workload(12,n,k,m,operation='indexed' if indexed else 'dense',
        compute='bf16' if indexed else 'f16',experts=256 if indexed else 1,
        topk=8 if indexed else 1,router=router if indexed else '')
    return inventory._point(w)


def binding(split=1):
    return dict(recipe=asdict(Recipe.make('simt','fq',variant=3,columns=4,values=8,warps=4,split=split)))


def tc_binding(split=1):
    return dict(recipe=asdict(Recipe.make('tc','sf',qtype=12,tm=16,tn=64,tk=256,
        wm=16,wn=16,stages=2,split=split,ap=0,dn=16)))


PROVENANCE=dict(timing_method='ASYS_GPU_ACTIVITY',quactlize_commit='a'*40,
                source_identity='source',target_identity='device')


def measured(p, costs=(10,10.05),baseline=12):
    maker=binding if p['workload']['m']<=8 else tc_binding
    return policy.record(p,[dict(binding=maker(i+1),samples_us=[[v*.99,v,v*1.01],[v,v,v]],correctness='PASS')
                            for i,v in enumerate(costs)],baseline,PROVENANCE,proof={'receipt':'raw-hash'})


def test_arithmetic_mean_and_exact_precedence():
    p=point();r=measured(p)
    assert r['mean_us']==10
    out=policy.export([r],expected=[p['id']])
    assert out['status']=='READY_FOR_INTEGRATION'
    got=policy.lookup(out,p,target_identity='device',source_identity='source')
    assert got['kind']=='EXACT' and got['measured_us']==10
    assert not out['production_changed'] and not out['native_no_regression_admitted']
    assert policy.lookup(out,p,target_identity='device',source_identity='changed')['kind'].startswith('MISS')


def test_no_commit_or_failed_numeric_is_not_evidence():
    with pytest.raises(ValueError,match='commit'):
        policy.record(point(),[],10,PROVENANCE|{'quactlize_commit':''},proof={'receipt':'raw'})
    with pytest.raises(ValueError,match='no valid'):
        policy.record(point(),[dict(binding=binding(),correctness='FAIL')],10,PROVENANCE,proof={'receipt':'raw'})
    with pytest.raises(ValueError,match='timing'):policy.samples([1,float('nan'),2])


def test_static_router_selection_not_runtime_readback():
    a,b=point(indexed=True),point(indexed=True,router='cluster')
    out=policy.export([measured(a,(9,10)),measured(b,(12,10))])
    assert len(out['exact'])==1
    row=next(iter(out['exact'].values()))
    assert row['binding']==binding(2) and len(row['profiles'])==2
    assert policy.lookup(out,b,target_identity='device',source_identity='source')['measured_us']==10


def test_router_regression_does_not_get_hidden_by_average():
    a,b=point(indexed=True),point(indexed=True,router='cluster')
    out=policy.export([measured(a,(9,13)),measured(b,(13,9))])
    assert out['status']=='INCOMPLETE' and not out['exact']
    assert out['pending'][0]['reason']=='NO_COMMON_NONREGRESSING_ROUTER_CONFIG'


def test_hard_contract_isolation():
    a=point();b=inventory._point(replace(Workload(**a['workload']),compute='bf16'))
    assert policy.bucket_key(a)!=policy.bucket_key(b)
    c=inventory._point(replace(Workload(**a['workload']),fusion='gate-up-swiglu'),'swiglu')
    assert policy.bucket_key(a)!=policy.bucket_key(c)
    assert policy.bucket_key(point(m=1))!=policy.bucket_key(point(m=2))


def test_bucket_holdout_requires_three_families_and_same_config():
    rows=[measured(point(m=128,n=n,k=2048)) for n in (1280,1536,1792)]
    assert not policy.fit(rows[:2])['holdout_validated']
    value=policy.fit(rows)
    assert value['holdout_validated'] and value['transfer_scope']=='NK_TRANSFER'
    out=policy.export(rows)
    got=policy.lookup(out,point(m=160,n=1792),target_identity='device',source_identity='source')
    assert got['kind']=='BUCKET_PROPOSAL_REQUIRES_RESOURCE_QUERY' and got['measured_us'] is None


def test_single_nk_bucket_never_claims_cross_nk_generalization():
    rows=[measured(point(m=m)) for m in (128,160,192)]
    out=policy.export(rows)
    bucket=next(iter(out['buckets'].values()))
    assert bucket['holdout_validated'] and bucket['nk']==[1024,2048]
    assert policy.lookup(out,point(m=224),target_identity='device',source_identity='source')['kind'].startswith('BUCKET')
    assert policy.lookup(out,point(m=224,n=1280),target_identity='device',source_identity='source')['kind'].startswith('MISS')


def test_large_dense_fq_never_becomes_bucket_fallback():
    bad=tc_binding();bad['recipe']['route']='fq'
    p=point(m=5120)
    assert not policy.legal(bad,p)
    assert policy.legal(tc_binding(),p)


def test_alias_binding_keeps_both_fresh_captures_not_two_candidates():
    p=point();cfg=lambda n:dict(binding=binding(),correctness='PASS',samples_us=[n,n,n])
    r=policy.record(p,[cfg(9),cfg(11)],12,PROVENANCE,proof={'receipt':'raw'})
    assert len(r['configurations'])==1 and r['mean_us']==10


def test_new_expert_topology_does_not_merge_router_finalists():
    from dev.moe_confirm.refine_plan import routing_key
    a=point(indexed=True)['workload'];b=dict(a,experts=128)
    assert routing_key(a)!=routing_key(b)


def test_q8_prefill_finalists_need_neither_sf_expansion_nor_full():
    from dev.moe_confirm.refine_plan import finalists
    w=Workload(8,512,2048,16,'indexed','bf16',experts=32,topk=4,router='spread')
    recipe=asdict(Recipe.make('tc','sf',qtype=8,tm=16,tn=64,tk=64,wm=16,wn=16,stages=2,split=1,dequant_config=0))
    p=dict(id=w.key,workload=asdict(w),arms=[dict(id='original',recipe=recipe,roles=['original'])])
    screen=dict(decision=dict(statistics={'original':dict(mean_us=10,samples_us=[[10]*3]*4)}),
                component_records={},helper_components={'sf':{},'full':{}})
    got=finalists([p],{p['id']:screen})[p['id']]
    assert got['sf_configs']==got['full_configs']==[]


def test_export_binds_source_and_binary_without_repeating_the_manifests():
    from dev.model_tuning.evidence import provenance
    run=dict(source_revision={'quactlize_commit':'a'*40},source={'large-source-list':'source'},
             harness={'large-helper-list':'harness'},images={'main':'binary'},sdk={'id':'sdk'},
             device={'name':'PPU','ordinal':0,'pci':'one','compute_units':72})
    p=provenance(run)
    assert p['binary_manifest_sha256']==digest(run['images']) and 'source' not in p
    other=provenance(dict(run,device=dict(run['device'],ordinal=1,pci='two')))
    assert p['target_identity']==other['target_identity']
    with pytest.raises(ValueError,match='binary'):provenance(dict(run,images={}))


def test_runtime_miss_does_not_claim_new_measurement():
    out=policy.export([measured(point())],expected=[point()['id'],point(2)['id']])
    assert out['status']=='INCOMPLETE' and out['missing']==[point(2)['id']]


def test_delivery_rule_withholds_target_and_never_reads_winner_latency():
    from dev.model_tuning.candidates import decode_delivery_peers
    w=Workload(14,1024,5120,4)
    seed=Recipe.make('tc','fq',qtype=14,tm=16,tn=64,tk=128,wm=16,wn=16,stages=2,
        ap=0,dn=16,persistent=-1,split=1,grid_mode=0,grid_b=0,symbol='seed')
    peer=Recipe.make('tc','fq',**(dict(seed.parameters)|dict(dn=64)))
    class Observation:
        def __init__(self,w,r):self.workload,self.recipe=w,r
        @property
        def us(self):pytest.fail('rule read a latency')
    donor=Observation(replace(w,n=4096,output_stride=0),peer)
    poison=Observation(w,Recipe.make('tc','fq',**(dict(seed.parameters)|dict(dn=32))))
    got=decode_delivery_peers(w,[seed],[donor,poison])
    assert {dict(r.parameters)['dn'] for r in got}=={64}
    assert {dict(r.parameters)['split'] for r in got}=={1,2,4,8}


def test_multiple_full_arms_use_their_own_reset_samples():
    from copy import deepcopy
    from test_moe_prefill_campaign import compose_fixture
    from dev.strategy.moe_prefill_search import compose
    p,value=compose_fixture();other=deepcopy(value)
    full=deepcopy(p['arms'][-1]);full['id']='full-second';p['arms'].append(full)
    other['arms']={'full-second':other['arms']['rule-full']}
    other['reset'].update(mean_us=2.,samples_us=[[2.]*3]*4)
    out=compose(p,[value,other])
    assert out['decision']['statistics']['rule-full']['mean_us']==13
    assert out['decision']['statistics']['full-second']['mean_us']==14
    assert set(out['arm_resets'])>={'rule-full','full-second'}


def test_cli_plan_reads_headers_only(tmp_path):
    from tools.tune_model import main
    rows,_=loader()('/abs/model.gguf');snap=tmp_path/'headers.json'
    snap.write_text(json.dumps({'/abs/model.gguf':rows}))
    output=tmp_path/'run'
    assert main(['plan','--model','/abs/model.gguf','--headers-snapshot',str(snap),
                 '--m','1','128','--output',str(output)])==0
    value=json.loads((output/'inventory.json').read_bytes())
    assert value['ms']==[1,128] and not (output/'runs').exists()
    assert main(['plan','--resume','--output',str(output)])==0
    with pytest.raises(ValueError,match='resume ms'):
        main(['plan','--resume','--output',str(output),'--m','2'])


def test_external_plan_requires_correct_backend_and_control():
    p=point();w=Workload(**p['workload'])
    plan=inventory.sealed(dict(schema='quactlize.smallm-asys.v1',
        model_tuning=dict(inventory='id',backend='smallm-dense'),
        points=[dict(id=w.key,workload=asdict(w),arms=[dict(id='original')])]))
    assert plans.validate_engine(plan,'smallm')==plan
    plan['points'][0]['arms']=[];plan=inventory.sealed(plan)
    with pytest.raises(ValueError,match='incumbent'):plans.validate_engine(plan,'smallm')


def test_runner_same_point_stays_on_one_device_and_no_native(tmp_path):
    ps=[dict(workload=asdict(Workload(12,1024,2048,m))) for m in (128,5120)]
    p=inventory.sealed(dict(points=ps));path=tmp_path/'plan.json';path.write_text(json.dumps(p))
    args=SimpleNamespace(devices='1,2,3',sdk='/sdk',jobs=2,l2_bytes=1234,asys=None,smi=None,
        budget_hours=8,llama_source=None,configs_per_batch=16)
    cmds=runner.commands(dict(backend='dense-prefill',plan=str(path),sha256=p['sha256']),args,tmp_path/'run')
    assert len(cmds)==1 and '--no-native' in cmds[0][0]
    assert cmds[0][2]['CUDA_VISIBLE_DEVICES']=='1'
