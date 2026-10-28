"""Three actual-header inventories, all M1..8; no fabricated device results."""
from copy import deepcopy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from dev.strategy import moe_model_plan as models
from dev.strategy.moe_inventory import collect
from dev.strategy.refresh import request
from dev.strategy.smallm_asys import assignments, check_producers
from quactlize.dispatch.planning import FIELDS, plan_smallm
from quactlize.dispatch.table import Workload
from tools.gguf_internal_shape_inventory import _synthetic_gguf, read_gguf_header, qtype_name

ROOT = Path(__file__).resolve().parents[1]


def synthetic_inventory(*, deepseek_qtype=None, shapes=None):
    # Test-only shapes, not claims about the deployed checkpoints. Vary E/top
    # too: an out-of-table family must retain the real v2 control, not vanish.
    shapes = shapes or [(12,13,2048,512,256,8), (10,14,2048,2048,256,8), (11,8,4096,2048,128,6)]
    inputs, headers = [], {}
    for (name,tp,_,_), (gate,down,k,n,e,top) in zip(models.MODELS, shapes):
        if tp==4 and deepseek_qtype is not None:gate=down=deepseek_qtype
        path='/synthetic/'+name
        arch='deepseek4' if tp==4 else 'qwen35moe'
        tensors=[('blk.0.ffn_gate_exps.weight',(k,n,e),gate),
                 ('blk.0.ffn_up_exps.weight',(k,n,e),gate),
                 ('blk.0.ffn_down_exps.weight',(n,k,e),down)]
        data=_synthetic_gguf([('general.name',name),('general.architecture',arch),
            (arch+'.expert_count',e),(arch+'.expert_used_count',top)],tensors)
        header=read_gguf_header(io.BytesIO(data),name)
        headers[path]=([dict(path=path,header=header)],header['tensors'])
        inputs.append(dict(name=name,tp=tp,path=path))

    def partition(model, rows, tensors):
        world=model['tp'];result={}
        for t in tensors:
            k,n,_=t['dims_gguf'];down='down' in t['name']
            result[t['name']]={(r,t['qtype'],n if down else n//world,k//world if down else k)
                               for r in range(world)}
        return result,dict(identity=dict(operation='MUL_MAT_ID')),{}

    return collect(inputs,loader=lambda path:headers[path],partitioner=partition,ms=list(range(1,9)))


@pytest.fixture(scope='module')
def planned(tmp_path_factory):
    out=tmp_path_factory.mktemp('moe-models')
    inventory=synthetic_inventory()
    return models.make_plan(out,inventory),inventory,out


def test_all_models_formats_local_shapes_and_m_are_visible(planned):
    plan,inventory,_=planned
    assert plan['counts']['families']==6 and plan['counts']['points']==216
    assert {p['workload']['qtype'] for p in plan['points']}=={8,10,11,12,13,14}
    assert {m['model']:m['tp'] for m in plan['coverage']['models']}=={n:tp for n,tp,_,_ in models.MODELS}
    assert all(m['points']==72 and len(m['families'])==2 for m in plan['coverage']['models'])
    assert plan['coverage']['actual_model_weights_used'] is False
    assert not plan['historical_times_reused'] and not plan['production_changed']
    for point in plan['points']:
        w=Workload(**point['workload'])
        assert point['id']==w.key and point['consumers']
        assert point['arms'][0]['id']=='original'
        assert w.storage=='f32' and w.compute=='bf16' and w.fusion=='none'
        assert all(a['recipe']['kind']=='simt' for a in point['arms'][1:])
        assert all('moe_pair_rows' not in dict(a['recipe']['parameters']) for a in point['arms'])
        for c in point['consumers']:
            k,n,e=c['global_dims']
            assert e==w.experts and 0<=c['rank']<c['tp']
            assert (w.n,w.k)==((n,k//c['tp']) if c['role']=='moe_expert_down' else (n//c['tp'],k))
    for m in range(1,9):
        assert len([p for p in plan['points'] if p['workload']['m']==m])==(6 if m==1 else 30)
    # Gate/up share one timing, but every layer/rank occurrence remains mapped.
    q4=next(p for p in plan['points'] if p['workload']['qtype']==12)
    assert len(q4['consumers'])==2
    q11=next(p for p in plan['points'] if p['workload']['qtype']==11)
    assert len(q11['consumers'])==8 and q11['workload']['n']==512


@pytest.mark.parametrize('fault',['missing_model','missing_m','blocked_tp','no_experts','unsupported_shape'])
def test_no_false_complete_inventory(fault):
    inventory=synthetic_inventory()
    if fault=='missing_model':inventory['models'].pop()
    elif fault=='missing_m':inventory['points']=[p for p in inventory['points'] if p['workload']['m']!=5]
    elif fault=='blocked_tp':inventory['blocked']=[dict(reason='TP_LOCAL_RECORDS_MISSING')]
    elif fault=='no_experts':inventory['models'][0]['expert_weights']=0
    else:inventory['excluded']=[dict(reason='NATIVE_ONLY_SHAPE: unsupported')]
    with pytest.raises(ValueError):models.expanded_points(inventory)


@pytest.mark.parametrize('reason,qtype',[
    ('UNSUPPORTED_QUANT_OR_FLOAT',39), ('MTP_DISABLED_BY_DEFAULT',12)])
def test_empty_model_reports_format_and_exclusion_without_skipping(reason,qtype):
    inventory=synthetic_inventory()
    name=models.MODELS[-1][0]
    inventory['points']=[p for p in inventory['points']
                         if all(c['model']!=name for c in p['consumers'])]
    inventory['excluded'].append(dict(model=name,qtype=qtype,reason=reason,
                                      tensor='blk.0.ffn_gate_exps.weight'))
    report=models.coverage_report(inventory)
    missing=next(row for row in report['models'] if row['model']==name)
    assert report['status']=='INCOMPLETE'
    assert missing['points']==0 and missing['missing']==list(range(1,9))
    assert missing['formats']==[] and missing['tokens']==[]
    assert missing['excluded']==[dict(qtype=qtype,format=qtype_name(qtype),reason=reason,records=1)]
    with pytest.raises(ValueError,match=name) as error:
        models.expanded_points(inventory)
    assert reason in str(error.value)


def test_mxfp4_checkpoint_is_not_misreported_as_missing_m_or_reinterpreted_q4():
    inventory=synthetic_inventory(deepseek_qtype=39)
    assert inventory['status']=='PASS' and not inventory['blocked']
    report=models.coverage_report(inventory)
    assert [m['model'] for m in report['models'] if m['missing']]==['deepseek-v4-flash']
    assert report['models'][-1]['excluded']==[dict(qtype=39,format='MXFP4',
        reason='UNSUPPORTED_QUANT_OR_FLOAT',records=3)]
    with pytest.raises(ValueError,match='no supported expert workloads') as error:
        models.expanded_points(inventory)
    assert 'MXFP4' in str(error.value) and 'deepseek-v4-flash' in str(error.value)
    assert not any(c['model']=='deepseek-v4-flash' for p in inventory['points'] for c in p['consumers'])
    # Explicit scope, never inferred from formats that happen to survive.
    inventory['requested_models']=[m[0] for m in models.MODELS[:2]]
    inventory['models']=inventory['models'][:2]
    inventory['excluded']=[]
    assert len(models.expanded_points(inventory))==144


@pytest.mark.parametrize('names',['',[],['typo'],['qwen35-35b-a3b']*2,39])
def test_model_selector_rejects_empty_duplicate_or_unknown_scope(names):
    with pytest.raises(ValueError,match='distinct names'):models.select_models(names)


def test_explicit_model_paths_do_not_touch_deferred_deepseek(tmp_path):
    paths={}
    for name,_,flag,_ in models.MODELS[:2]:
        paths[flag]=tmp_path/(name+'.gguf');paths[flag].touch()
    args=SimpleNamespace(model_root=tmp_path/'no-default-models',**paths,
        moe_models=','.join(m[0] for m in models.MODELS[:2]),deepseek_model=tmp_path/'absent.gguf')
    assert [m['name'] for m in models.model_paths(args)]==[m[0] for m in models.MODELS[:2]]
    args.moe_models=None
    with pytest.raises(FileNotFoundError):models.model_paths(args)


def test_shared_family_cannot_hide_missing_m_for_one_model():
    inventory=synthetic_inventory()
    # Share a family across two models. The second model has all M from its
    # own families, but this particular shared family is missing M=5 for it.
    other=models.MODELS[1][0]
    for p in inventory['points']:
        if p['workload']['qtype']==12 and p['workload']['m']!=5:
            consumer=dict(p['consumers'][0],model=other,tp=2)
            p['consumers'].append(consumer)
    report=models.coverage_report(inventory)
    assert all(not row['missing'] for row in report['models'])
    assert len(report['incomplete_families'])==1
    row=report['incomplete_families'][0]
    assert row['model']==other and row['missing']==[5]
    with pytest.raises(ValueError,match='M1..8 coverage missing'):
        models.expanded_points(inventory)
    point=next(p for p in inventory['points'] if p['workload']['qtype']==12 and p['workload']['m']==5)
    point['consumers'].append(dict(point['consumers'][0],model=other,tp=2))
    assert models.coverage_report(inventory)['status']=='PASS'
    assert len(models.expanded_points(inventory))==216


def test_real_tc_incumbent_retained_without_new_tc_candidates(planned):
    plan,_,_=planned
    tc=[p for p in plan['points'] if p['arms'][0]['recipe']['kind']=='tc']
    assert tc  # synthetic Q6 family uses an existing TC choice at some M
    for p in tc:
        assert sum(a['recipe']['kind']=='tc' for a in p['arms'])==1
        assert p['selection']['control_scope']=='POC_INDEXED_BF16_AUTO_HOST_CHAIN'
    # Unknown expert topology is not labeled an exact measured-table match.
    p=next(p for p in plan['points'] if p['workload']['experts']==128)
    assert p['selection']['policy']==11  # QKS_COMPUTE_INITIAL


def test_producer_guards_keep_generic_flat_grid_and_common_3d(planned):
    for p in planned[0]['points']:
        data=dict(arms={},component_records={},arm_components={})
        w=p['workload']
        for a in p['arms']:
            f=dict(a['recipe']['parameters']);entry=a['expected_producer']
            grid=([w['n']//(f['columns']*f['values']),w['m']*w['topk'],f.get('split',1)]
                  if a['recipe']['kind']=='simt' else [1,1,1])
            if not entry.endswith('::common_reader'):
                grid=[grid[0]*grid[1]*grid[2],1,1]
            node=dict(name=entry,symbol=entry,grid=grid,block=[f.get('warps',4)*32,1,1])
            data['component_records'][a['id']]=dict(kind='producer',identity=dict(node=node))
            data['arm_components'][a['id']]=[a['id']]
            data['arms'][a['id']]=dict(receipt=dict(execution_contract=p['execution_contract']))
        check_producers(p,data)
        wrong=deepcopy(data)
        aid=next(a['id'] for a in p['arms'] if a['recipe']['kind']=='simt')
        wrong['component_records'][aid]['identity']['node']['grid'][0]+=1
        with pytest.raises(ValueError,match='grid/block'):check_producers(p,wrong)
    a=assignments(planned[0],7)
    assert len({p for ids in a for p in ids})==planned[0]['counts']['points']
    assert sum(map(len,a))==planned[0]['counts']['points'] and all(a)


def test_cpp_controls_match_real_public_dispatch_not_python_guess(planned,tmp_path):
    plan,_,_=planned
    (tmp_path/'catalog.inc').write_text('static std::vector<Image> kImages; static char kJitSource[]="";\n')
    binary=tmp_path/'public-control'
    subprocess.run(['g++','-std=c++17','-O1','-I'+str(ROOT),'-I'+str(tmp_path),
        str(ROOT/'tests/moe_model_control_host.cpp'),'-ldl','-pthread','-o',str(binary)],check=True)
    keys=sorted({tuple(p['selection']['request']) for p in plan['points']} | {
        (q,2,n,k,e,top,ch,m,1) for q in (8,10,11,12,13,14) for n,k in ((768,1536),(8192,512))
        for e,top in ((256,8),(128,6)) for ch in (1,top) for m in (1,4,8)})
    host=plan_smallm(tmp_path,keys,indexed_fallbacks=True)['requests']
    text=subprocess.check_output([str(binary)],text=True,input=''.join(' '.join(map(str,k))+'\n' for k in keys))
    actual=[json.loads(line) for line in text.splitlines()]
    assert len(host)==len(actual)==len(keys)
    for a,b in zip(host,actual):
        assert a['status']==b['status']==0 and a['kind']==b['kind'],(a,b)
        if a['kind']==0:
            assert a['parent']['symbol']==b['parent'] and a['split']==b['split']
        else:
            fields=FIELDS if a['kind']==1 else ('reader','variant','columns','warps','values')
            assert [a['config'][f] for f in fields]==b['config'],(a,b)


@pytest.mark.parametrize('cohort,subset',[(models.COHORT,False),('moe-config-models',False),('moe-config-models',True)])
def test_three_model_plan_cli_real_caller_cpu_only(tmp_path,cohort,subset):
    llama=os.environ.get('LLAMA_TP_SOURCE')
    if not llama:pytest.skip('set LLAMA_TP_SOURCE for actual CPU TP derivation')
    from test_tp_header import fixtures
    command=[sys.executable,str(ROOT/'tools/run_smallm_regressions_asys.py'),
        '--cohort',cohort,'--output',str(tmp_path/'plan'),
        '--llama-source',llama,'--sdk','/no-sdk','--devices','7','--plan-only']
    selected=models.MODELS[:2] if subset else models.MODELS
    if subset:command+=['--moe-models',','.join(m[0] for m in selected)]
    for name,tp,flag,_ in selected:
        arch='deepseek4' if tp==4 else 'qwen35moe'
        meta,tensors=fixtures(arch)
        meta[arch+'.expert_count']=256;meta[arch+'.expert_used_count']=8
        k,n=(4096,2048) if tp==4 else (2048,512*tp)
        tensors += [dict(name='blk.0.ffn_gate_exps.weight',qtype=12,dims_gguf=[k,n,256]),
                    dict(name='blk.0.ffn_up_exps.weight',qtype=12,dims_gguf=[k,n,256]),
                    dict(name='blk.0.ffn_down_exps.weight',qtype=13,dims_gguf=[n,k,256])]
        if tp==1:
            tensors.append(dict(name='blk.8.ffn_up_exps.weight',qtype=14,dims_gguf=[k,n,256]))
        path=tmp_path/(name+'.gguf')
        path.write_bytes(_synthetic_gguf(list(meta.items()),[(t['name'],tuple(t['dims_gguf']),t['qtype']) for t in tensors]))
        command += ['--'+flag.replace('_','-'),str(path)]
    for extra in ([],['--resume']):
        result=subprocess.run(command+extra,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        assert result.stdout.count('MOE_MODEL_COVERAGE ')==len(selected)
    plan=json.loads((tmp_path/'plan/plan.json').read_text())
    # Identical actual local shapes across Qwen35/122 deduplicate, not 3x72.
    assert plan['counts']['points']==(72 if subset else 144)
    assert len(json.loads((tmp_path/'plan/moe-headers.json').read_text()))==len(selected)
    assert plan['coverage']['requested_models']==[m[0] for m in selected]
    assert plan['coverage']['deferred_models']==(['deepseek-v4-flash'] if subset else [])
    models.verify_inventory(tmp_path/'plan',plan,','.join(m[0] for m in selected))
    if subset:
        with pytest.raises(ValueError,match='resume requested models differ'):
            models.verify_inventory(tmp_path/'plan',plan,','.join(m[0] for m in models.MODELS))
        # Omitting --moe-models on resume inherits the frozen subset, not all three.
        models.verify_inventory(tmp_path/'plan',plan)
    assert any(e['model']=='qwen35-35b-a3b' and e['reason']=='MTP_DISABLED_BY_DEFAULT'
               for e in plan['coverage']['excluded'])
    assert not (tmp_path/'plan/run.json').exists()
    assert not (tmp_path/'plan/images.json').exists()
    path=tmp_path/'plan/moe-inventory.json'
    data=json.loads(path.read_text());data['models'][0]['expert_weights']+=1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='inventory changed'):
        models.verify_inventory(tmp_path/'plan',plan)
