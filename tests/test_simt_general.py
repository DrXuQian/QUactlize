import subprocess
import pytest
from dev.gemv_simt.general_reader import ROOT,source,reuse_body,codes_body,vector_body,QTYPES
from dev.gemv_simt.general_plan import configs,plan


def test_actual_host_maps_and_metadata(tmp_path):
    exe=tmp_path/'host'
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra',f'-I{ROOT}',
                    str(ROOT/'tests/general_reader_host.cpp'),'-o',str(exe)],check=True)
    assert 'GENERAL_READER_HOST PASS' in subprocess.check_output([str(exe)],text=True)


def test_no_reduction_or_dot_math_rewrite():
    text=(ROOT/'quactlize/execution/simt_kernel.cuh').read_text()
    candidate=reuse_body()
    for begin,end in [('dot[p].x=fmaf','\n                    }'),
                      ('total[p].x+=fmaf','\n        }'),
                      ('    float values[P];','\n}\n')]:
        a=text.index(begin);b=text.index(end,a)
        assert text[a:b] in candidate
    text=(ROOT/'quactlize/execution/simt_format.cuh').read_text()
    assert text[text.index('    // Two exact integer'):text.index('\n} // namespace')] in codes_body()


def test_all_format_and_mode_bodies_and_s1_only():
    for q in QTYPES:
        for compute in ('f16','bf16'):
            text=source(q,compute,configs(q))
            for i in range(len(configs(q))):
                for mode in range(3): assert f'general_candidate_{i}_{mode}<<<dim3(' in text
            assert 'simt::query_v2' in text and 'simt::buffers_v2' in text
            assert 'general_control_0<<<call.rows*' in text
            assert 'call.input_type!=QKG_F32' in text
    with pytest.raises(ValueError,match='S1 only'):
        source(14,'bf16',[configs(14)[0]|{'split':2}])


def test_plan_is_bounded_with_complete_m_numeric_coverage():
    p=plan();assert p['point_count']==84
    assert p['numeric_config_pairs']<1200 and p['timing_config_pairs']<500
    for q in QTYPES:
        for compute in ('f16','bf16'):
            for mode in ('dense','indexed','grouped'):
                row,=[r for r in p['points'] if r['kind']=='all-format-control' and
                       (r['workload']['qtype'],r['workload']['compute'],r['workload']['operation'])==(q,compute,mode)]
                assert row['numeric_m']==list(range(1,9))
    assert not p['production_changed'] and 'SPLIT_K' in p['omissions']


def test_no_runtime_n_division_in_new_geometry():
    for text in (reuse_body(),vector_body()):
        assert 'tile=blockIdx.x,partition=0,row=blockIdx.y;' in text
        assert 'blockIdx.x%' not in text and 'blockIdx.x/' not in text
        assert 'general_locate<Mode>' in text


def test_actual_locate_and_grouped_call_contract(tmp_path):
    exe=tmp_path/'locate'
    subprocess.run(['g++','-std=c++17','-O2',f'-I{ROOT}',
                    str(ROOT/'tests/general_locate_host.cpp'),'-o',str(exe)],check=True)
    assert 'GENERAL_LOCATE_HOST PASS' in subprocess.check_output([str(exe)],text=True)


def test_latest_q6_and_q8_optimized_controls_are_retained():
    text=source(14,'bf16',configs(14))
    assert 'q6_warp_body<14,1,3,4,8,8,1,0>(call,split);' in text
    assert 'q6_pipeline_body<14,1,3,4,2,8,1,0>(call,split);' in text
    assert 'q8_vector::kernel_body<1,0,1,8,4,4,true>' in source(8,'f16',configs(8))
    assert 'register_reuse_body<13,1,3,4,2,8,1,3>' in source(13,'bf16',configs(13))


@pytest.mark.parametrize('q',QTYPES)
def test_strong_fixture_and_ragged_grouping(q):
    import numpy as np
    from dev.gemv_simt.general_fixture import Weights,grouped,raw_weight
    from dev.gemv_simt.general_plan import workload
    from dev.strategy.indexed import point
    from dev.bf16_compute.fixture import round_compute
    w=Weights(q,256,512,16)
    p=point(workload(q,'indexed',3,256,512,'bf16'))
    data=w.inputs(p,1);g=grouped(data,16,8,8)
    assert len(g['offsets'])==17 and g['offsets'][0]==0 and g['offsets'][-1]==24
    assert np.all(np.diff(g['offsets'])>=0) and np.any(np.diff(g['offsets'])==0)
    assert np.array_equal(g['a'],round_compute(g['a'],'bf16'))
    for e in range(16):
        lo,hi=g['offsets'][e:e+2]
        assert np.all(g['owners'][lo:hi]==e)
    # The independent direct matrix multiplication must agree with the
    # memory-bounded category sums used by the device gate.
    for row in (0,7,23):
        e=int(data['owners'][row]);raw,gold=raw_weight(q,256,512,710+e*31+q*1009)
        want=data['a'][row].astype('f8')@gold.astype('f8').T
        assert np.allclose(want,data['gold'][row],rtol=1e-13,atol=1e-13)


def test_receipt_resume_and_corruption_rejected(tmp_path):
    import json
    from tools.run_simt_general import sealed,read_cell
    p=tmp_path/'cell.json';identity={'m':3,'config':1,'source':'abc'}
    assert read_cell(p,identity) is None
    sealed(p,dict(status='PASS',identity=identity,proof='all'))
    assert read_cell(p,identity)['proof']=='all'
    with pytest.raises(ValueError,match='refusing to replace'):sealed(p,{'status':'PASS'})
    with pytest.raises(ValueError,match='identity/hash'):read_cell(p,identity|{'m':2})
    data=json.loads(p.read_text());data['proof']='changed';p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='identity/hash'):read_cell(p,identity)


@pytest.mark.parametrize('selection,count',[('all',84),('controls',36),('0,4,34',3)])
def test_actual_plan_entry_without_gpu(tmp_path,selection,count):
    import json,sys
    output=tmp_path/'plan'
    command=[sys.executable,str(ROOT/'tools/run_simt_general.py'),'--sdk',str(tmp_path),
             '--output',str(output),'--points',selection,'--plan-only']
    subprocess.run(command,check=True,capture_output=True)
    p=json.loads((output/'plan.json').read_text())
    assert p['point_count']==count and not (output/'build').exists()
    subprocess.run(command+['--resume'],check=True,capture_output=True)
    bad=subprocess.run(command[:-1]+['--resume','--plan-only','--points','1'],capture_output=True)
    assert bad.returncode and b'resume plan changed' in bad.stderr


def test_box_and_direct_profile_have_no_hidden_sweep():
    subprocess.run(['bash','-n',str(ROOT/'tools/run_simt_general_box.sh')],check=True)
    text=(ROOT/'tools/run_simt_general_box.sh').read_text()
    assert 'git submodule update' not in text and 'RESUME_RUN' in text and '--resume' in text
    import inspect
    from dev.gemv_simt.general_bench import profile
    text=inspect.getsource(profile)
    assert 'Captured' not in text and 'ring.copies' not in text
    timed=text[text.index("checked(start(),"):text.index('finally:')]
    assert timed.count('core.launch(')==1
    assert 'hggcProfilerStart' in text and 'graphs=0 calls=1' in text


def test_kernel_factory_matches_public_grouped_abi(monkeypatch):
    from types import SimpleNamespace as N
    import dev.gemv_simt.general_bench as b
    from dev.gemv_simt.general_plan import workload
    from quactlize.execution.native import arrangement
    class Function:
        def __call__(self,*args):return 0
    monkeypatch.setattr(b.C,'CDLL',lambda *args,**kwargs:N(general_reader_run=Function()))
    for operation,mode in (('dense',0),('grouped',1),('indexed',2)):
        w=workload(14,operation,3)
        context=N(w=w,p={'mode':mode},rows=w.m*(w.topk if mode else 1),
                  a=N(ptr=4096),out=N(ptr=8192),ids=N(ptr=12288) if mode==2 else None,
                  offsets=N(ptr=16384) if mode==1 else None,r=N(stream=N(value=0)),
                  ring=N(planes=[{'low':20480,'high':24576,'units':28672}]),arr=arrangement(14))
        core=b.Kernel(context,'unused')
        c=core.calls[0].call
        assert c.channels==(w.channels if mode==2 else 1) and c.topk==(w.topk if mode==2 else 1)
        assert bool(c.offsets)==(mode==1) and bool(c.ids)==(mode==2)
        assert c.rows==context.rows and c.a==4096 and c.output==8192


def test_request_footprint_preserves_planes_and_reduces_only_metadata():
    from dev.gemv_simt.general_access import pattern
    from quactlize.execution.simt_codegen import Config
    for q in QTYPES:
        for f in configs(q):
            c=Config(**{k:f[k] for k in ('variant','columns','warps','values','split')})
            for warp in range(c.warps):
                p=pattern(q,c,512,2048,warp=warp)
                x,y=p['control'],p['candidate']
                if q!=8:
                    assert [s for s in x['streams'] if not s['name'].startswith('metadata')]==[
                        s for s in y['streams'] if not s['name'].startswith('metadata')]
                    assert all(s in x['streams'] for s in y['streams'])
                if q==14 and c.columns==4 and c.values==8:assert x==y


def test_later_cell_failure_does_not_erase_completed_peer(tmp_path):
    import json,sys
    from tools.run_simt_general import read_cell,sealed
    first=tmp_path/'first.json';second=tmp_path/'second.json'
    identity={'source':'fixed','m':1}
    sealed(first,dict(identity=identity,status='PASS',checks=[{'error':0.0}]))
    original=first.read_bytes()
    script='import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from tools.run_simt_general import read_cell;assert read_cell(Path(sys.argv[2]),{"source":"fixed","m":1});raise SystemExit(7)'
    assert subprocess.run([sys.executable,'-c',script,str(ROOT),str(first)]).returncode==7
    assert first.read_bytes()==original and read_cell(second,identity) is None
    sealed(second,dict(identity=identity,status='PASS',checks=[{'error':0.0}]))
    assert read_cell(first,identity) and read_cell(second,identity)


def test_two_dimensional_geometry_has_identical_work_ownership():
    for n in (256,512,1024,3072,5120,8192):
        for rows in (1,2,3,4,5,6,7,8,24,64):
            for tile in (16,32):
                old={(block%(n//tile),block//(n//tile)) for block in range(rows*(n//tile))}
                new={(x,y) for y in range(rows) for x in range(n//tile)}
                assert old==new and len(old)==rows*n//tile
