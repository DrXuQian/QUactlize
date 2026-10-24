"""TP inventory and bounded plans are host tests, not new GPU evidence."""
from collections import Counter
import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from dev.prefill_asys.cohort import SCHEMA as COHORT_SCHEMA, collect, tp_records, local_shapes
from dev.prefill_asys.supplement import make_plan, validate, execution_case, policy_lookup
from dev.prefill_asys.finalists import load, bind_cases, partition, case_identity
from dev.strategy.resources import inventory, resolve_point
from quactlize.dispatch.dense_prefill import EVIDENCE
from quactlize.dispatch.table import Workload
from quactlize.runtime.tuning import digest
from tools.run_tp_prefill_supplement import probe, probe_command, model_path
from dev.prefill_point.bench import tc_helper_choice

ROOT = Path(__file__).resolve().parents[1]


def cohort(workloads):
    d = dict(schema=COHORT_SCHEMA, points=[dict(id=w.key,workload=asdict(w),consumers=[],tensors=[],call_count=None)
        for w in workloads],status='READY',blocked=[],excluded=[],models=[])
    d['sha256'] = digest(d)
    return d


def policy():
    return json.loads(EVIDENCE.read_text())


def plan(workloads):
    return make_plan(cohort(workloads),policy(),digest(json.loads(
        (ROOT/'policies/dense_full_helpers_20261023.json').read_text())))


def test_native_tp_metadata_preserves_repeat_and_replication(tmp_path):
    tensors = [dict(name='blk.0.attn_qkv.weight',qtype=12,dims_gguf=[5120,9216]),
               dict(name='blk.0.ffn_down.weight',qtype=14,dims_gguf=[5120,3072]),
               dict(name='blk.0.attn_gate.weight',qtype=8,dims_gguf=[3072,512])]
    log = tmp_path/'tp.log'
    log.write_text('SPLIT_STATE: {} -> blk.0.attn_qkv.weight[NONE, 1, {512x3, 1024x3, 512x3, 1024x3}]\n'
                   'SPLIT_STATE: {} -> blk.0.ffn_down.weight[NONE, 0, {1280x1, 1280x1, 1280x1, 1280x1}]\n'
                   'SPLIT_STATE: {} -> blk.0.attn_gate.weight[NONE, MIRRORED, {0x1, 0x1, 0x1, 0x1}]\n')
    records,_ = tp_records([log],4,tensors)
    for t,axis in zip(tensors,('n','k','replicated')):
        rows,error = local_shapes(t,4,records)
        assert error is None and len(rows)==4 and {r[-1] for r in rows}=={axis}
    assert [r[2] for r in sorted(records[tensors[0]['name']])]==[1536,3072,1536,3072]


@pytest.mark.parametrize('extents',['1024x1,1024x1','512x1,512x1,512x1,512x1','1,2,3,4'])
def test_incomplete_native_segments_are_not_guessed(tmp_path,extents):
    t = dict(name='blk.0.attn_q.weight',qtype=12,dims_gguf=[4096,4096])
    log = tmp_path/'log'
    log.write_text(f'SPLIT_STATE: {{}} -> {t["name"]}[NONE, 1, {{{extents}}}]\n')
    with pytest.raises(ValueError):
        tp_records([log],4,[t])


def test_native_and_quactlize_conflict_stays_blocked(tmp_path):
    t = dict(name='blk.0.attn_q.weight',qtype=12,dims_gguf=[4096,4096])
    log=tmp_path/'log'
    log.write_text('SPLIT_STATE: {} -> blk.0.attn_q.weight[NONE, 1, {2048x1,2048x1}]\n'
                   '[quactlize-dense] tensor=blk.0.attn_q.weight device=0 q=12 m=64 n=4096 k=4096 route=0\n')
    d=collect([dict(name='tp',path='/model',tp=2,tp_logs=[log])],[128],loader=lambda _:([],[t]))
    assert d['status']=='INCOMPLETE'
    assert d['blocked'][0]['reason']=='TP_LOCAL_RECORDS_AMBIGUOUS'


def test_all_old_exact_controls_and_helper_choices_are_retained():
    data=policy()
    workloads=[Workload(q,n,k,m) for q,m,n,k in (p['shape'] for p in data['points'])]
    p=plan(workloads)
    assert len(p['points'])==225
    assert load(p)==p
    for row,old in zip(sorted(p['points'],key=lambda x:x['id']), sorted(data['points'],key=lambda x:x['key'])):
        assert row['table_coverage']=='EXACT' and row['incumbent_recipe']==old['winner']
        want=execution_case(data['recipes'][old['winner']])
        assert any(case_identity(c)==want and 'model-config' in c['roles'] for c in row['cases'])
        assert len(row['cases'])<=32


def test_new_q2_to_q8_shapes_have_bounded_f16_complete_comparisons():
    workloads=[Workload(q,n,k,m) for q in (8,10,11,12,13,14)
               for n,k in ((512,1024),(1024,2048),(2048,512),(4096,3072),(3072,6144))
               for m in (128,256,512,1024,2048)]
    p=plan(workloads)
    assert p['counts']['points']==150 and p['counts']['families']==30
    assert max(len(x['cases']) for x in p['points'])<=32
    assert not p['performance_admitted'] and not p['production_changed']
    assert Counter(x['table_coverage'] for x in p['points'])['OUTSIDE_NEW_TABLE']>0
    for row in p['points']:
        assert all('gpu_sum_us' not in c and 'median_us' not in c for c in row['cases'])
        if row['workload']['qtype']==8:
            assert {38,41}<={c.get('helper_options',{}).get('dequant_variant') for c in row['cases']}
    parts=[partition(p['points'],i,7) for i in range(7)]
    assert sorted(x['id'] for part in parts for x in part)==sorted(x['id'] for x in p['points'])
    inv=inventory(p)
    receipts={k:dict(status='READY',request=v['request'],device=dict(compute_units=72),occupancy=2)
              for k,v in inv['queries'].items()}
    for row in p['points']:
        bound=bind_cases(resolve_point(row,receipts),receipts)
        assert [c['id'] for c in bound]==[c['id'] for c in row['cases']]


def test_supplement_plan_only_uses_existing_runner_without_sdk(tmp_path):
    p=plan([Workload(12,3072,6144,2048)])
    src=tmp_path/'plan.json';src.write_text(json.dumps(p))
    output=tmp_path/'dry'
    subprocess.run([sys.executable,str(ROOT/'tools/run_prefill_model_asys.py'),
        '--finalist-plan',str(src),'--output',str(output),'--plan-only',
        '--sdk','/no-sdk','--llama-source','/no-llama'],check=True,capture_output=True)
    got=json.loads((output/'plan.json').read_text())
    assert got['points']==p['points'] and got['counts']==p['counts']
    assert not (output/'build').exists()


@pytest.mark.parametrize('fault',['control','helper','budget','precision','route'])
def test_changed_or_partial_plans_fail_even_when_resealed(fault):
    p=plan([Workload(12,5120,8192,2048)])
    point=p['points'][0]
    if fault=='control':
        for c in point['cases']:c['roles']=[r for r in c['roles'] if r!='model-config']
    elif fault=='helper':point['cases'][0]['helper_options']['vector_io']=99
    elif fault=='budget':p['max_cases']=1
    elif fault=='precision':point['workload']['compute']='bf16'
    elif fault=='route':point['cases']=[c for c in point['cases'] if c['arm']!='sf']
    p['sha256']=digest({k:v for k,v in p.items() if k!='sha256'})
    with pytest.raises(ValueError):validate(p)


def test_probe_and_path_resolution_are_explicit(tmp_path):
    d=tmp_path/'bench_model_zoo'/'model'/'Q4_K_M';d.mkdir(parents=True)
    path=d/'first.gguf';path.write_bytes(b'header')
    assert model_path(tmp_path,'model','first.gguf',None)==path
    cmd=probe_command(Path('/build/bin/llama-batched-bench'),path,4)
    assert cmd[cmd.index('-sm')+1]=='tensor' and cmd[cmd.index('-ts')+1]=='1,1,1,1'
    assert cmd[cmd.index('--verbosity')+1]=='5' and '--no-warmup' not in cmd
    other=tmp_path/'model';other.mkdir();(other/'first.gguf').write_bytes(b'other')
    with pytest.raises(ValueError,match='expected one'):model_path(tmp_path,'model','first.gguf',None)
    assert model_path(tmp_path,'model','first.gguf',path)==path


@pytest.mark.parametrize('warmup_flag', ['', '--no-warmup'])
def test_probe_accepts_tp_binary_without_a_warmup_switch(tmp_path,monkeypatch,warmup_flag):
    build=tmp_path/'build'
    (build/'bin').mkdir(parents=True)
    binary=build/'bin/llama-batched-bench';binary.write_bytes(b'host-only test binary identity')
    args=SimpleNamespace(llama_build=build,sdk=tmp_path/'sdk')
    help_text='--split-mode --tensor-split --verbosity '+warmup_flag
    def help_call(command,**kwargs):
        assert command==[str(binary),'--help']
        return subprocess.CompletedProcess(command,0,stdout=help_text,stderr='')
    monkeypatch.setattr('tools.run_tp_prefill_supplement.subprocess.run',help_call)
    calls=[]
    monkeypatch.setattr('tools.run_dense_model_ab.execute',lambda *args:calls.append(args))
    out=tmp_path/'probe';model=tmp_path/'model.gguf'
    result=probe(args,model,'qwen35-122b-tp2',2,out,['1','3','5','7'])
    assert result==out/'run.log' and len(calls)==1
    command,env,directory,label=calls[0]
    assert '--no-warmup' not in command and command[command.index('-ts')+1]=='1,1'
    assert env['CUDA_VISIBLE_DEVICES']=='1,3'
    assert env['GGML_QUACTLIZE_DENSE']=='0' and env['GGML_META_DEBUG']=='1'
    receipt=json.loads((out/'identity.json').read_text())
    assert receipt['warmup']=='NATIVE_DEFAULT_NOT_TIMED'
    assert receipt['scope']=='NATIVE_PLACEMENT_INVENTORY_NOT_PERFORMANCE'


@pytest.mark.parametrize('missing', ['--split-mode','--tensor-split','--verbosity'])
def test_probe_still_requires_actual_inventory_options(tmp_path,monkeypatch,missing):
    build=tmp_path/'build';(build/'bin').mkdir(parents=True)
    (build/'bin/llama-batched-bench').write_bytes(b'host-only identity')
    help_text=' '.join(f for f in ('--split-mode','--tensor-split','--verbosity','--no-warmup') if f!=missing)
    monkeypatch.setattr('tools.run_tp_prefill_supplement.subprocess.run',
        lambda command,**_:subprocess.CompletedProcess(command,0,stdout='',stderr=help_text))
    def unexpected(*_):raise AssertionError('invalid binary must not start a model run')
    monkeypatch.setattr('tools.run_dense_model_ab.execute',unexpected)
    with pytest.raises(ValueError,match='required TP inventory option '+missing):
        probe(SimpleNamespace(llama_build=build,sdk=tmp_path/'sdk'),tmp_path/'model',
              'tp',2,tmp_path/'probe',['1','2'])


def test_fixed_tc_io_does_not_consult_the_old_helper_lookup():
    class NoLookup:
        def choice(self,*_,**__):
            raise AssertionError('explicit measured IO must not be replaced by a second selector')
    for io in (-1,1,2,3,4):
        got=tc_helper_choice(NoLookup(),None,'cohort-current',dict(vector_io=io))
        assert got==(dict(vector_io=io,storage=0) if io>=0 else None)
    with pytest.raises(ValueError):tc_helper_choice(NoLookup(),None,'cohort-current',dict(vector_io=0))
    assert tc_helper_choice(None,None,'legacy',None) is None
