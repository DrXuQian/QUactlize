import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from dev.gemv_simt.general_plan import configs,workload
from dev.gemv_simt.general_reader import vector_body
from dev.gemv_simt.q8_dense_source import ARMS,ROOT,body,geometry,original_body,source,symbol_name
from dev.gemv_simt.q8_dense_plan import inventory,plan


def test_prior_combined_body_is_exact_apart_from_template_name():
    previous=vector_body().replace('template<int Mode,int Input,','template<int Input,')
    previous=previous.replace('void general_vector_body(', 'void dense_ablation_5(')
    previous=previous.replace('general_locate<Mode>','general_locate<QKG_DENSE>')
    assert previous==body(5)


def test_dot_dequant_and_reduction_unchanged():
    old=original_body()
    for arm in range(1,len(ARMS)):
        candidate=body(arm)
        start=old.index('        float d[P];')
        assert old[start:]==candidate[candidate.index('        float d[P];'):]
        assert 'uint64_t nk=uint64_t(c.n)*c.k;' in candidate
    with pytest.raises(ValueError,match='untouched'):body(0)


def test_orthogonal_axes_and_signed_loop_repair():
    for arm in range(1,len(ARMS)):
        spec=ARMS[arm];text=body(arm)
        assert ('general_locate<QKG_DENSE>' in text)==spec['mode']
        assert ('row=blockIdx.y' in text)==spec['grid']
        assert ('unsigned tid' in text)==spec['unsigned']
        assert ('for(unsigned g' in text)==(spec['unsigned'] and not spec['signed_loop'])
    assert 'for(int g' in body(6)


def test_work_ownership_for_all_dense_shapes_and_m():
    for spec in inventory():
        for m in range(1,9):
            w=workload(8,'dense',m,spec['workload']['n'],spec['workload']['k'])
            for cfg in configs(8):
                tiles=w.n//(cfg['columns']*cfg['values'])
                expected={(x,y) for y in range(m) for x in range(tiles)}
                for arm in range(len(ARMS)):
                    grid,block=geometry(w,cfg,arm)
                    actual=({(x,y) for y in range(grid[1]) for x in range(grid[0])} if ARMS[arm]['grid'] else
                            {(x%tiles,x//tiles) for x in range(grid[0])})
                    assert actual==expected and block==[32*cfg['warps'],1,1]


def test_dense_only_s1_validation_and_both_precisions():
    for compute in ('f16','bf16'):
        text=source(8,compute,configs(8))
        assert 'call.mode!=QKG_DENSE' in text and 'call.input_type!=QKG_F32' in text
        assert 'simt::query_v2' in text and 'simt::buffers_v2' in text
        assert text.count('__global__ void ')==14
        for c in range(2):
            for arm in range(7):assert symbol_name(c,arm)+'<<<' in text
    with pytest.raises(ValueError):source(14,'f16',configs(8))
    with pytest.raises(ValueError):source(8,'f16',[configs(8)[0]|{'split':2}])
    with pytest.raises(ValueError):source(8,'f16',[configs(8)[0]|{'hoist':False}])


def test_plan_bounded_and_contains_both_regressions():
    p=plan()
    assert json.loads(json.dumps(p))==p
    assert len(p['points'])==8 and p['numeric_pairs']==336 and p['timing_pairs']==192
    assert p['acu_reports']==len(p['profiles'])*len(ARMS)*3==42
    assert p['profiles']==[dict(point=1,m=1,config=1),dict(point=4,m=8,config=0)]
    assert p['configs']==configs(8) and not p['production_changed']
    assert '1GHZ' in p['frequency_policy'] and '1P7GHZ' in p['frequency_policy']


def test_actual_plan_and_resume_entry_no_device(tmp_path):
    out=tmp_path/'plan'
    cmd=[sys.executable,str(ROOT/'tools/run_q8_dense.py'),'--sdk',str(tmp_path),'--output',str(out),'--plan-only']
    subprocess.run(cmd,check=True,capture_output=True)
    assert json.loads((out/'plan.json').read_text())==plan() and not (out/'build').exists()
    subprocess.run(cmd+['--resume'],check=True,capture_output=True)
    old=json.loads((out/'plan.json').read_text());old['acu_reports']=0;(out/'plan.json').write_text(json.dumps(old))
    p=subprocess.run(cmd+['--resume'],capture_output=True)
    assert p.returncode and b'frozen plan changed' in p.stderr


def test_pair_adapter_maps_only_requested_variant(monkeypatch):
    import dev.gemv_simt.q8_dense_bench as bench
    calls=[]
    monkeypatch.setattr(bench.GeneralKernel,'__init__',lambda *a:None)
    monkeypatch.setattr(bench.GeneralKernel,'launch',lambda self,c,a,copy: calls.append((c,a,copy)))
    for candidate in range(1,len(ARMS)):
        k=bench.Kernel(None,None,candidate)
        k.launch(1,0,2);k.launch(1,1,3)
        assert calls[-2:]==[(1,0,2),(1,candidate,3)]


def test_receipt_cannot_cross_variants_or_source(tmp_path):
    from tools.run_simt_general import sealed,read_cell
    p=tmp_path/'cell.json';key=dict(source='a',point=1,m=1,config=1,variant=2)
    sealed(p,dict(status='PASS',identity=key,raw_equal=True))
    before=p.read_bytes()
    assert read_cell(p,key)['raw_equal']
    for new in (key|{'variant':3},key|{'source':'b'},key|{'config':0}):
        with pytest.raises(ValueError):read_cell(p,new)
    assert p.read_bytes()==before


def test_single_direct_profile_and_shell_syntax():
    from dev.gemv_simt.general_bench import profile
    text=inspect.getsource(profile)
    timed=text[text.index("checked(start(),"):text.index('finally:')]
    assert timed.count('core.launch(')==1 and 'Captured' not in text
    shell=ROOT/'tools/run_q8_dense_box.sh'
    subprocess.run(['bash','-n',str(shell)],check=True)
    text=shell.read_text()
    assert 'git submodule update' not in text and 'RESUME_RUN' in text and 'EXPECTED_CE_GHZ' in text
    assert '--graph-profiling' not in text


@pytest.mark.parametrize('expected_ce_hz',[1.7e9,1e9])
def test_profile_all_variants_with_exact_proof_and_reuse(tmp_path,monkeypatch,expected_ce_hz):
    import csv
    from dev.gemv_simt.q6_cycles import CYCLES,RATES
    from quactlize.runtime.compiler import sha
    from tools import run_q8_dense as run
    output=tmp_path/'output';output.mkdir()
    authority={'expected_ce_hz':expected_ce_hz};focus=dict(point=1,m=1,config=1)
    key=dict(authority='fixed',point=1,configs=configs(8),arms=list(ARMS))
    monkeypatch.setattr(run,'identity',lambda *a:key)
    w=workload(8,'dense',1,1024,5120)
    for variant in range(1,len(ARMS)):
        nodes=[]
        for arm in (0,variant):
            grid,block=geometry(w,configs(8)[1],arm)
            nodes.append(dict(symbol=symbol_name(1,arm),name=symbol_name(1,arm),grid=grid,block=block))
        run.sealed(run.cell_path(output,1,1,1,variant),dict(status='PASS',identity=key|dict(m=1,config=1,variant=variant),timing=dict(nodes=nodes)))
    calls=[];reports={}
    def execute(command,log):
        calls.append(command)
        if '--export' in command:
            target=Path(command[command.index('--export')+1]);arm=int(command[command.index('--profile-arm')+1])
            target.with_suffix('.acurep').write_bytes(b'fake-test-report')
            result=Path(command[command.index('--result')+1]);prior=run.cell_path(output,1,1,1,arm or 5)
            proof=dict(status='PASS',identity=key|dict(m=1,config=1,profile_arm=arm),
                       prior_sha256=sha(prior),graphs=0,timed_calls=1,weight_copies=1,
                       checks=[dict(bad=0,nonfinite=0,relative_l1_error=0.0)]*4)
            run.sealed(result,proof);reports[str(target.with_suffix('.acurep'))]=arm
        else:
            arm=reports[command[command.index('--import')+1]]
            grid,block=geometry(w,configs(8)[1],arm)
            row={'ID':'0','Kernel Name':symbol_name(1,arm),'Grid Size':str(tuple(grid)),'Block Size':str(tuple(block))}
            row.update({k:900-arm for k in CYCLES})
            row.update({k:1800000 if k=='device__attribute_memory_clock_rate' else 1.7e9 for k in RATES})
            with log.open('x') as fp:
                writer=csv.DictWriter(fp,fieldnames=list(row),quoting=csv.QUOTE_ALL);writer.writeheader();writer.writerow(row)
        return 0
    monkeypatch.setattr(run,'execute',execute)
    a=NS(output=output,sdk=tmp_path/'sdk',l2_bytes=67108864,acu=tmp_path/'acu')
    attempt=output/'first';attempt.mkdir()
    result=run.profile_focus(a,focus,attempt,authority)
    if expected_ce_hz!=1.7e9:
        assert result['status']=='FAIL' and len(result['reports'])==21
        assert all('CE clock differs' in r['error'] for r in result['reports'])
        assert not list((output/'profiles').rglob('r*-a*.json'))
        return
    assert result['status']=='PASS' and len(result['reports'])==21 and len(result['comparisons'])==6
    assert len(calls)==42
    attempt=output/'second';attempt.mkdir()
    result=run.profile_focus(a,focus,attempt,authority)
    assert result['status']=='PASS' and len(calls)==42  # no repeated GPU work
    report=Path(result['reports'][0]['raw_csv']);report.write_text('corrupted')
    attempt=output/'third';attempt.mkdir()
    with pytest.raises(ValueError,match='hash/path'):run.profile_focus(a,focus,attempt,authority)
