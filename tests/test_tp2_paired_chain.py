"""Host-only contract tests; no claim of PPU numeric/performance admission."""
import ctypes as C
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from dev.bf16_compute.fixture import round_compute
from dev.bf16_compute.native import Projection, MixedPlanV2
from dev.gemv_model.access import access
from dev.gemv_model.source import source
from dev.tp2_chain.plan import POINTS, ARMS, candidates, stages, projection_config
from dev.tp2_chain.bench import Bench, swiglu, finish_gold, normalized_error
from dev.tp2_chain.run import summarize, profiles, fresh_path, collect
from quactlize.execution.native import Call
from quactlize.dispatch.native import MoeFinish
from quactlize.fusion.native import MappedCall, Layout


def test_inventory_and_retained_baselines():
    assert [(p.n,p.k) for p in POINTS] == [(512,3072),(512,2048)]
    for p in POINTS:
        cs=candidates(p)
        assert len(cs)==2 and all(c.split==1 and c.fixed for c in cs)
        assert (cs[0].warps,cs[0].values)==(8,8)
        assert (cs[1].warps,cs[1].values)==(4,4)
        assert cs[1].changes==(3 if p.k==3072 else 1)
        text=source(p,cs)
        assert 'call.round_projection=fusion->call.round_projection' in text
        assert 'fusion->call.round_projection!=1' in text
        assert 'Paired16Finish' in text and 'c.n=1024' in text
        for c in cs:
            a=access(p,c,dict(A=0,low=0,high=0,units=0))
            assert a['paired_N4'] and a['fixed_shape']
            assert a['B_contiguous_bytes_per_k_worker_group']==2*c.tile_n
            assert set(a['aggregate_source_footprint'])=={'32','64','128'}
    assert len(stages('unfused'))==5
    assert all(len(stages(a))==4 for a in ARMS[1:])
    assert projection_config(13)==dict(variant=3,columns=4,warps=2,values=8,split=1)


def test_bf16_boundaries_and_negative_oracles():
    gate=np.array([[243383.484375,3.14159]],'f4')
    up=np.array([[2.125,-5.125]],'f4')
    actual=swiglu(gate,up)
    assert np.isfinite(actual).all() and actual[0,0]>65504
    g,u=round_compute(gate,'bf16'),round_compute(up,'bf16')
    expected=g/(1+np.exp(-g))*u
    np.testing.assert_array_equal(actual,expected)
    activation=np.tile(actual,(8,1))
    down=[np.array([[.125,.25],[-.5,.125]],'f4') for _ in range(8)]
    weights=np.arange(1,9,dtype='f4')[None,:]/64
    gold,denom,dots,absolute=finish_gold(activation,down,weights)
    assert np.isfinite(gold).all() and np.all(denom>0)
    assert normalized_error(gold,gold,denom)==0
    with pytest.raises(ValueError):normalized_error(np.zeros_like(gold),gold,denom)
    with pytest.raises(ValueError):normalized_error(np.full_like(gold,np.nan),gold,denom)
    with pytest.raises(ValueError):normalized_error(gold[:,:1],gold,denom)


def test_complete_call_sample_validation():
    result=summarize([[float(i+1)]*15 for i in range(6)])
    assert result['median_us']==3.5
    for data in ([[1.]*15]*5,[[1.]*14]*6,[[float('nan')]*15]*6,[[0.]*15]*6):
        with pytest.raises(ValueError):summarize(data)


def csv_profile(p,arm):
    names=['void prepare_detail::once_with_finish<cutlass::bfloat16_t>()']
    cfg=candidates(p)[1]
    if arm=='unfused':
        names+=['void measured_decode_kernel<12,2>()','void moe_chain_swiglu_compute<cutlass::bfloat16_t>()']
    else:names += [f'void point_12_1024_{p.k}_arm_1()']
    names+=['void measured_decode_kernel<13,2>()','void moe_weighted_finish<true,cutlass::bfloat16_t>()']
    rows=[{'ID':i,'Kernel Name':name,'Grid Size':f'({8*1024//cfg.tile_n},1,1)',
           'Block Size':f'({cfg.warps*32},1,1)'} for i,name in enumerate(names)]
    stream=io.StringIO();writer=csv.DictWriter(stream,list(rows[0]),quoting=csv.QUOTE_ALL)
    writer.writeheader();writer.writerows(rows)
    return stream.getvalue()


def test_profile_requires_every_stage_and_exact_paired_geometry():
    p=POINTS[0]
    assert len(profiles(csv_profile(p,'unfused'),p,'unfused'))==5
    text=csv_profile(p,'paired-1')
    assert len(profiles(text,p,'paired-1'))==4
    for changed in (text.replace('moe_weighted_finish','omitted_finish'),
                    text.replace('bfloat16_t','half_t'),
                    text.replace('point_12_1024_3072_arm_1','point_12_1024_2048_arm_1'),
                    text.replace('(128,1,1)','(256,1,1)'),
                    text.replace('measured_decode_kernel<13','measured_decode_kernel<12')):
        with pytest.raises(ValueError):profiles(changed,p,'paired-1')


class FakeRuntime:
    stream=C.c_void_p(71)
    def __init__(self):self.ptr=0x100000
    def allocate(self,size):self.ptr+=((size+1023)//1024)*1024;return self.ptr
    def fill(self,*args):pass


def fake_bench():
    b=object.__new__(Bench);b.rt=FakeRuntime();b.point=POINTS[0];b.copies=2;b.scratch=[]
    for i,name in enumerate(('input','ids','logits','weights','output','gate_output','down_input','down_output')):
        setattr(b,name,mock.Mock(ptr=0x1000*(i+1)))
    b.rings={name:mock.Mock(at=lambda copy:dict(low=0xA000+copy*4096,high=None,units=0xC000+copy*4096))
             for name in ('canonical','paired','down')}
    def query(call,size):C.cast(size,C.POINTER(C.c_uint64)).contents.value=65536;return 0
    def bind(call,device,scratch,size,part):
        c=C.cast(call,C.POINTER(Call)).contents;p=C.cast(part,C.POINTER(Projection)).contents
        p.m,p.n,p.k,p.experts=c.rows,c.n,c.k,c.experts
        p.a,p.output,p.outputs,p.directory_header=scratch,scratch+8192,scratch+32768,scratch+40000
        p.io.output=c.output
        return 0
    b.query,b.bind=query,bind
    return b


@pytest.mark.parametrize('arm',ARMS)
def test_callbacks_match_production_chain_buffers_and_phase(arm):
    b=fake_bench();b.configure(1);seen=[];b.layout=Layout()
    assert b.calls[0].output==b.parts[0].output
    assert b.calls[1].a==b.parts[1].a
    assert b.retained.weights==b.parts[1].outputs and b.finish.weights==b.output.ptr
    def prepare(d,f,stream):
        assert C.cast(d,C.POINTER(MixedPlanV2)).contents.compute_type==1
        seen.append('prepare-snapshot');return 0
    def activate(d,phase,stream):
        assert phase==2;seen.append('activation');return 0
    def shipping(call,cfg):
        assert cfg.split==1
        return lambda:(seen.append('projection' if call.qtype==12 else 'down') or 0)
    def paired(*args):
        arg=args[0] if arm=='paired-shipping' else args[1]
        c=C.cast(arg,C.POINTER(MappedCall)).contents
        assert c.call.input.call.n==512 and c.call.round_projection==1
        assert c.call.input.call.output==b.parts[1].a
        assert c.status==b.parts[0].directory_header+4
        seen.append('paired-projection-activation');return 0
    b.prepare_fn,b.activate_fn=prepare,activate
    b.finish_fn=lambda *args:(seen.append('weighted-finish') or 0)
    b.shipping=mock.Mock(prepare=shipping);b.paired_run=b.candidate_run=paired
    assert b.launch(arm,1)()==0
    assert tuple(seen)==stages(arm)


def test_stage_failure_stops_dependent_work():
    b=fake_bench();b.configure(1)
    b.shipping=mock.Mock(prepare=lambda *a:lambda:0)
    b.prepare_fn=lambda *a:17;b.activate_fn=mock.Mock();b.finish_fn=mock.Mock()
    assert b.launch('unfused')()==17
    b.activate_fn.assert_not_called();b.finish_fn.assert_not_called()


def test_profile_resume_preserves_failed_attempt_siblings(tmp_path):
    old=tmp_path/'point-unfused.acu.log'
    old.write_text('previous failure')
    assert fresh_path(tmp_path,'point-unfused.acu')!=tmp_path/'point-unfused.acu'
    assert old.read_text()=='previous failure'
    assert fresh_path(tmp_path,'new.json')==tmp_path/'new.json'


def test_fetch_is_exactly_scoped_to_the_selected_cohort(tmp_path):
    from tools.run_decode_focus_box import fetch
    with mock.patch('tools.run_decode_focus_box.subprocess.run') as run:
        valid=dict(branch='artifacts/tp2-paired-chain-v1',commit='a'*40,
                   path='prebuilt/ppu0010/tp2-paired-chain-v1',manifest_sha256='b'*64)
        for changed,cohort in ((valid,'tp2-focus'),(valid,'unknown'),
                               (dict(valid,path='prebuilt/ppu0010/kpack-model-runtime-v1'),'tp2-chain'),
                               (dict(valid,commit='bad'),'tp2-chain'),
                               (dict(valid,branch='--all'),'tp2-chain')):
            with pytest.raises(ValueError):fetch(changed,tmp_path,cohort)
        run.assert_not_called()


def test_preflight_needs_both_points_and_reports_runtime_differences(tmp_path):
    from tools.run_paired_chain_box import preflight
    from quactlize.runtime.compiler import LIBRARIES, sha
    sdk=tmp_path/'sdk';(sdk/'lib').mkdir(parents=True);(sdk/'asight/bin').mkdir(parents=True)
    (sdk/'asight/bin/acu').write_bytes(b'inspector')
    for name in LIBRARIES:(sdk/'lib'/f'lib{name}.so').write_bytes(b'runtime-'+name.encode())
    bundle=tmp_path/'bundle';bundle.mkdir();(bundle/'manifest.json').write_text('{}')
    manifest=dict(records=[dict(point=dict(name=p.name)) for p in POINTS],
                  runtime={f'lib{n}.so':sha(sdk/'lib'/f'lib{n}.so') for n in LIBRARIES})
    with mock.patch('tools.run_paired_chain_box.verify',return_value=manifest):
        assert preflight(bundle,sdk)['runtime_matches_build']
        manifest['runtime']['libhggc.so']='f'*64
        # Runtime differences are evidence, not a spurious compiler identity
        # veto. The independent device numerical gate remains mandatory.
        assert not preflight(bundle,sdk)['runtime_matches_build']
        manifest['records'].pop()
        with pytest.raises(ValueError):preflight(bundle,sdk)
    with mock.patch('tools.run_paired_chain_box.verify',return_value=dict(records=[dict(point=dict(name=p.name)) for p in POINTS])):
        with pytest.raises(ValueError,match='SDK root'):preflight(bundle,tmp_path)


def test_duplicate_physical_device_never_starts_timing(tmp_path):
    args=SimpleNamespace(bundle=tmp_path,output=tmp_path/'out',devices='6,7',
                         sdk=tmp_path,resume=False)
    with mock.patch('dev.tp2_chain.run.verify'), \
            mock.patch('dev.tp2_chain.run.subprocess.check_output',return_value='PAIRED_CHAIN_DEVICE {"pci":"0000:01:00.0"}\n'), \
            mock.patch('dev.tp2_chain.run.run_point') as worker:
        with pytest.raises(ValueError,match='same physical'):collect(args)
        worker.assert_not_called()


def test_box_script_preserves_failure_results_and_outer_shell(tmp_path):
    root=Path(__file__).resolve().parents[1]
    sdk=tmp_path/'sdk';(sdk/'asight/bin').mkdir(parents=True)
    acu=sdk/'asight/bin/acu';acu.write_text('#!/bin/sh\nexit 0\n');acu.chmod(0o700)
    python=tmp_path/'fake-python'
    python.write_text('''#!/usr/bin/env python3
import pathlib, sys
if sys.argv[1] == '-c': raise SystemExit(0)
out=pathlib.Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir()
(out/'failure.json').write_text('{"status":"FAIL"}')
print('PAIRED_CHAIN_TEST retained failure', flush=True)
raise SystemExit(7)
''')
    python.chmod(0o700)
    env=dict(os.environ,PPU_SDK=str(sdk),RESULT_ROOT=str(tmp_path),PYTHON=str(python))
    env.pop('PAIRED_CHAIN_RUN',None)
    process=subprocess.run(['bash','-c','bash tools/run_paired_chain_box.sh\nprintf "outer_shell_rc=%s\\n" "$?"'],
                           cwd=root,env=env,text=True,capture_output=True,check=True)
    assert 'outer_shell_rc=7' in process.stdout
    assert 'Current Docker shell is preserved.' in process.stdout
    archives=list(tmp_path.glob('paired-chain.*.results.tgz'))
    assert len(archives)==1
    with tarfile.open(archives[0]) as archive:
        assert json.load(archive.extractfile('results/failure.json'))=={'status':'FAIL'}
        assert 'runner_rc=7' in archive.extractfile('runner-status.txt').read().decode()
        assert 'retained failure' in archive.extractfile('console.log').read().decode()
