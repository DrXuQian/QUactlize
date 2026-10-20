import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as N

import numpy as np
import pytest

from dev.gemv_simt.split_plan import plan, inventory, configs, QTYPES, SPLITS
from dev.gemv_simt.split_source import source, body
from dev.gemv_simt.general_reader import ROOT, reuse_body


def test_frozen_bounded_inventory():
    p=plan()
    assert len(p['points'])==74 and p['numeric_pairs']==2224 and p['timing_pairs']==1776
    assert p['splits']==[1,2,4,8] and not p['production_changed']
    assert {r['workload']['qtype'] for r in p['points']}==set(QTYPES)
    for r in p['points']:
        assert set(r['timing_m'])<=set(r['numeric_m'])<=set((1,3,8))
    assert 'Q8' in p['omissions'] and 'NEW_CLOCK_GLOBAL_SELECTION' in p['omissions']
    fixed=[r['workload'] for r in p['points'] if r['kind']=='shipping-fixed-path-control']
    assert {(w['qtype'],w['n'],w['k'],w['channels']) for w in fixed}=={
        (12,1024,2048,1),(12,1024,3072,1),(13,2048,512,8),(13,3072,512,8)}
    assert all(w['experts']==256 and w['topk']==8 for w in fixed)


def test_same_optimized_math_only_split_grid_is_new():
    assert body()==reuse_body().replace('general_reuse_body','split_reuse_body').replace(
        'partition=0','partition=blockIdx.z')
    assert 'g=partition*Workers+worker' in body()
    assert 'g+=split*Workers' in body()
    assert '(int64_t(row)*split+partition)*call.n' in body()
    for q in QTYPES:
        for compute in ('f16','bf16'):
            text=source(q,compute)
            assert 'simt::query_v2(*d,cfg,a,sizes)' in text and 'simt::buffers_v2(*d,sizes)' in text
            assert 'measured_decode(*d,cfg)' in text and f'measured_decode_launch<{q}>' in text
            assert f'simt::launch_reduction<{q}>' in text
            assert 'general_control_' not in text
            for i,c in enumerate(configs(q)):
                assert f'simt::launch_v2<{q},3,{c["columns"]},{c["warps"]},{c["values"]}>(*d,split)' in text
                for mode in range(3):
                    assert f'split_candidate_{i}_{mode}<<<dim3(call.n/{c["columns"]*c["values"]},call.rows,split)' in text
    with pytest.raises(ValueError):source(8,'f16')


@pytest.mark.parametrize('q',QTYPES)
def test_partition_oracle_against_direct_gguf(q):
    from dev.gemv_simt.split_bench import Weights
    from dev.gemv_simt.general_fixture import raw_weight, grouped
    from dev.gemv_simt.general_plan import workload
    from dev.strategy.indexed import point
    from dev.strategy.pilot_native import scale_activation_data
    weights=Weights(q,256,512,16)
    group=weights.group
    for compute in ('f16','bf16'):
        for mode in ('dense','indexed','grouped'):
            w=workload(q,mode,3,256,512,compute,channels=1)
            data=weights.inputs(point(w)|{'mode':0 if mode=='dense' else 2},2)
            if mode=='grouped':data=grouped(data,16,1,8)
            if compute=='bf16':scale_activation_data(data,2**20)
            b=N(w=w,rows=3 if mode=='dense' else 24,data=data)
            for c in configs(q):
                workers=c['warps']*32//c['columns']
                for split in SPLITS:
                    want,denom=weights.partial_oracle(b,c,split)
                    np.testing.assert_allclose(want.sum(axis=1),data['gold'][:,weights.spots],rtol=1e-13,atol=1e-13)
                    # Independently enumerate each CTA's strided K groups,
                    # decode raw GGUF, and dot actual rounded activations.
                    for row in (0,b.rows-1):
                        expert=int(data['owners'][row])%16
                        _,gold=raw_weight(q,256,512,710+expert*31+q*1009)
                        arow=row//w.topk*w.channels+row%w.topk%w.channels if mode=='indexed' else row
                        a=data['a'][arow].astype('f8')
                        for s in range(split):
                            ks=[k for worker in range(workers)
                                for g in range(s*workers+worker,512//group,split*workers)
                                for k in range(g*group,(g+1)*group)]
                            direct=gold[weights.spots][:,ks].astype('f8')*a[ks]
                            np.testing.assert_allclose(want[row,s],direct.sum(axis=1),rtol=1e-13,atol=1e-13)
                            np.testing.assert_allclose(denom[row,s],np.abs(direct).sum(axis=1),rtol=1e-13,atol=1e-13)
                    if split>1:
                        assert not np.array_equal(want,np.roll(want,1,axis=1))


def test_all_partitions_cover_k_once_even_when_some_empty():
    for groups in (16,32,48,64,128,320,512):
        for workers in (16,32,64):
            for split in SPLITS:
                pieces=[[g for w in range(workers) for g in range(s*workers+w,groups,split*workers)] for s in range(split)]
                assert sorted(g for p in pieces for g in p)==list(range(groups))
                for s,p in enumerate(pieces):assert all((g//workers)%split==s for g in p)
                # Flattened shipping and 3D candidate enumerate identical
                # (N tile,row,partition) ownership, even with different order.
                ntiles,rows=5,24
                old={(i%ntiles,i//ntiles//split,i//ntiles%split) for i in range(ntiles*rows*split)}
                new={(x,y,z) for x in range(ntiles) for y in range(rows) for z in range(split)}
                assert old==new


def test_split_request_coalescing_and_uniform_packet_phases():
    from tools.run_simt_split import access_receipt
    from dev.gemv_simt.general_plan import workload
    for q in QTYPES:
        for c in configs(q):
            w=workload(q,'indexed',8,k=2048,compute='bf16')
            result=access_receipt(w,c,8,dict(A=0,low=0,high=0,units=0))
            for row in result['rows']:
                old=[x for x in row['shipping'] if not x['name'].startswith('metadata')]
                new=[x for x in row['candidate'] if not x['name'].startswith('metadata')]
                assert old==new
                assert all(x in row['shipping'] for x in row['candidate'])
                live=[g for g in row['lane_groups'] if g<w.k//(32 if q in (12,13) else 16)]
                assert len(live) in (0,32)
                if live:assert len({g//8%({10:2,11:4,14:4}.get(q,1)) for g in live})==1


def test_kernel_factory_binds_current_partial_and_storage(monkeypatch):
    import dev.gemv_simt.split_bench as b
    from dev.gemv_simt.general_plan import workload
    from quactlize.execution.native import arrangement
    class Function:
        def __call__(self,*args):self.args=args;return 0
    fn=Function()
    monkeypatch.setattr(b.C,'CDLL',lambda *a,**kw:N(split_reader_run=fn))
    w=workload(14,'indexed',3)
    context=N(w=w,p={'mode':2},rows=24,a=N(ptr=4096),out=N(ptr=8192),ids=N(ptr=12288),offsets=None,
              r=N(stream=N(value=0)),ring=N(planes=[dict(low=20480,high=24576,units=28672)]),
              arr=arrangement(14),partial=N(ptr=32768,size=24*512*8*4))
    for s in SPLITS:
        core=b.Kernel(context,'unused',s)
        call=core.calls[0].call
        assert call.workspace==(32768 if s>1 else None)
        assert call.workspace_bytes==(24*512*8*4 if s>1 else 0)
        assert call.input_type==1 and call.topk==8 and call.channels==8
        assert core.launch(2,1,stage=1)==0 and fn.args[-4:]==(2,s,1,1)


def test_resume_preserves_cells_and_rejects_corruption(tmp_path):
    from tools.run_simt_split import read_cell, sealed, cell_path, cells
    spec=inventory()[0];identity={'m':1,'split':2,'config':0,'source':'fixed'}
    path=cell_path(tmp_path,0,1,0,2)
    sealed(path,dict(status='PASS',identity=identity))
    before=path.read_bytes()
    assert read_cell(path,identity) and path.read_bytes()==before
    assert len(cells(spec))==36
    with pytest.raises(ValueError):read_cell(path,identity|{'split':4})
    with pytest.raises(ValueError):sealed(path,dict(status='PASS',identity=identity))
    value=json.loads(path.read_text());value['status']='FAIL';path.write_text(json.dumps(value))
    with pytest.raises(ValueError):read_cell(path,identity)


def test_plan_entry_and_failed_only_resume_without_gpu(tmp_path):
    output=tmp_path/'results'
    command=[sys.executable,str(ROOT/'tools/run_simt_split.py'),'--sdk',str(tmp_path),
             '--output',str(output),'--plan-only']
    subprocess.run(command,check=True,capture_output=True)
    assert json.loads((output/'plan.json').read_text())==plan()
    subprocess.run(command+['--resume'],check=True,capture_output=True)
    assert not (output/'build').exists()
    frozen=plan();frozen['splits']=[1];(output/'plan.json').write_text(json.dumps(frozen))
    result=subprocess.run(command+['--resume'],capture_output=True)
    assert result.returncode and b'resume plan changed' in result.stderr


def test_timer_and_direct_acu_scopes_are_separate():
    from dev.gemv_simt.split_bench import timing, profile
    text=inspect.getsource(timing)
    assert 'isolate(0,weight_ring=True' in text and 'isolate(1,weight_ring=False' in text
    assert 'reducer_measured_once_for_identical_implementation' in text
    assert 'host_launch_cost_added=False' in text and 'param.func!=' in text
    text=inspect.getsource(profile)
    assert 'Captured' not in text and 'Ring(' not in text
    timed=text[text.index("checked(start(),"):text.index('finally:')]
    assert timed.count('core.launch(')==1
    from dev.native_components.profile import acu_command
    command=acu_command('acu','out',['python','child'],graph_profiling='none')
    assert command[command.index('--launch-count')+1]=='1'
    assert command[command.index('--profile-from-start')+1]=='no'
    assert command[command.index('--graph-profiling')+1]=='none'


@pytest.mark.parametrize('damage',('none','missing-partition','missing-producer','bad-oracle','missing-reducer','combined-timer','sum','samples'))
def test_completed_cell_needs_full_evidence(damage):
    from tools.run_simt_split import validate_cell
    from dev.strategy.components import statistics_checked
    good=dict(bad=0,nonfinite=0,relative_l1_error=.001)
    numeric=dict(whole=dict(checks=[good]*12,raw_equal=True,zero_a='PASS',omitted='DETECTED',plane_negatives=['low','units']),
                 partials=dict(checks=[good]*12,raw_equal=True,omitted_producer='DETECTED',omitted_reducer='DETECTED'))
    timing=dict(producer=[statistics_checked([[2.]*5]*6),statistics_checked([[1.]*5]*6)],
                reducer=statistics_checked([[.5]*5]*6),component_sum_us=[2.5,1.5],
                timed_kernel_types_per_interval=1,host_launch_cost_added=False,raw_equal=True)
    row=dict(numerical=numeric,timing=timing,ring=dict(copies=2))
    if damage=='missing-partition':numeric['partials']['checks']=[]
    elif damage=='missing-producer':numeric['partials']['omitted_producer']='NOT_TESTED'
    elif damage=='bad-oracle':numeric['whole']['checks']=[good|dict(relative_l1_error=.01)]*12
    elif damage=='missing-reducer':timing['reducer']=None
    elif damage=='combined-timer':timing['timed_kernel_types_per_interval']=2
    elif damage=='sum':timing['component_sum_us']=[2.,1.]
    elif damage=='samples':timing['producer'][1]['samples_us']=[[1.]*5]*2
    if damage=='none':assert validate_cell(row,inventory()[0],1,2) is row
    else:
        with pytest.raises(ValueError):validate_cell(row,inventory()[0],1,2)


def test_shell_static_and_no_nested_fetch():
    path=ROOT/'tools/run_simt_split_box.sh'
    subprocess.run(['bash','-n',str(path)],check=True)
    text=path.read_text()
    assert 'git submodule update' not in text and 'git clone' not in text and '\nrm ' not in text
    assert 'git -C "${ROOT}/third_party/actlize" archive' in text
    assert '--resume' in text and 'PIPESTATUS[0]' in text
