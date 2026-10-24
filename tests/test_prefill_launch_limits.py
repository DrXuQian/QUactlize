"""Host launcher boundaries only; no SDK, device, kernel or timing claims."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.run_prefill_cohort_multi import worker_environment, worker_limits

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('explicit',[{},dict(jobs=3,configs_per_batch=4)])
def test_worker_limits_do_not_inherit_build_or_capture_environment(monkeypatch,explicit):
    monkeypatch.setenv('JOBS','192');monkeypatch.setenv('CONFIGS_PER_BATCH','64')
    env=worker_environment(Path('/new-run'),'7',2,6,'frozen',finalists=True,**explicit)
    wanted=('3','4') if explicit else ('8','16')
    assert (env['JOBS'],env['CONFIGS_PER_BATCH'])==wanted
    assert (env['GPU'],env['SHARD'],env['SHARDS'])==('7','2','6')


@pytest.mark.parametrize('jobs,batch',[(0,16),(65,16),(192,16),(True,16),('8',16),
                                     (8,0),(8,33),(8,False),(8,'16')])
def test_invalid_limits_fail_before_workers(jobs,batch):
    with pytest.raises(ValueError):worker_limits(jobs,batch)
    with pytest.raises(ValueError):worker_environment(Path('/never-created'),'1',0,1,'sha',
                                                      jobs=jobs,configs_per_batch=batch)


@pytest.mark.parametrize('jobs,batch',[(8,16),(2,4)])
def test_outer_runner_forwards_and_records_limits(tmp_path,monkeypatch,jobs,batch):
    from test_q8_tp_refine import review
    from tools import run_q8_tp_refine as runner
    path=tmp_path/'review.json';path.write_text(json.dumps(review()))
    monkeypatch.setenv('JOBS','192');monkeypatch.setenv('CONFIGS_PER_BATCH','64')
    calls=[]
    def launch(command,env):
        calls.append(command)
        assert command[command.index('--jobs')+1]==str(jobs)
        assert command[command.index('--configs-per-batch')+1]==str(batch)
        return 0
    monkeypatch.setattr(runner,'run_coordinator',launch)
    argv=[runner.__file__,'--review',str(path),'--phase','helpers','--output',str(tmp_path/'run'),
          '--run','--confirm-idle','--gpus','1,2,3,4,5,7']
    if jobs!=8:argv+=['--jobs',str(jobs),'--configs-per-batch',str(batch)]
    monkeypatch.setattr(sys,'argv',argv)
    assert runner.main()==0 and len(calls)==1
    request=json.loads((tmp_path/'run/request.json').read_text())
    assert request['worker_limits']==dict(jobs=jobs,configs_per_batch=batch)


@pytest.mark.parametrize('runner,plan_flag',[
    ('run_q8_tp_refine.py','--review'),('run_prefill_cohort_multi.py','--finalist-plan'),
    ('run_prefill_model_asys.py','--finalist-plan')])
@pytest.mark.parametrize('flag,value',[('--jobs','192'),('--configs-per-batch','64')])
def test_invalid_cli_is_specific_and_precedes_setup(tmp_path,runner,plan_flag,flag,value):
    output=tmp_path/'not-created'
    result=subprocess.run([sys.executable,str(ROOT/'tools'/runner),plan_flag,'/no-plan',
        '--output',str(output),flag,value],capture_output=True,text=True)
    assert result.returncode==2
    assert flag+' must be in ' in result.stderr and 'got '+value in result.stderr
    assert not output.exists() and 'Traceback' not in result.stderr


def test_actual_plan_only_worker_accepts_isolated_limits(tmp_path,monkeypatch):
    from dev.prefill_asys.io_refine import make_plan
    from test_dense_compact import split_review
    path=tmp_path/'cohort.json';path.write_text(json.dumps(make_plan(split_review(),compact=True)))
    monkeypatch.setenv('JOBS','192');monkeypatch.setenv('CONFIGS_PER_BATCH','64')
    env=worker_environment(tmp_path,'7',0,1,'sha',finalists=True)
    result=subprocess.run([sys.executable,str(ROOT/'tools/run_prefill_model_asys.py'),
        '--finalist-plan',str(path),'--output',str(tmp_path/'dry'),'--plan-only',
        '--device',env['GPU'],'--jobs',env['JOBS'],'--configs-per-batch',env['CONFIGS_PER_BATCH'],
        '--sdk','/no-sdk','--llama-source','/no-llama'],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads((tmp_path/'dry/plan.json').read_text())['counts']['configs']==72
    assert not (tmp_path/'dry/build').exists()


def test_real_shell_forwards_logged_limits_to_worker(tmp_path,monkeypatch):
    # Exercise the real shell but replace only its final Python process with
    # an explicit argv-printing double. No pretend numerical/timing result.
    run=tmp_path/'retained';(run/'source/tools').mkdir(parents=True);(run/'results').mkdir()
    (run/'source/tools/run_prefill_model_asys.py').touch();(run/'results/plan.json').write_text('{}')
    plan=tmp_path/'plan.json';plan.write_text('{}')
    stub=tmp_path/'argv-python'
    stub.write_text(f'#!{sys.executable}\nimport json,sys\nprint("HOST_ARGV "+json.dumps(sys.argv[1:]))\n')
    stub.chmod(0o700)
    monkeypatch.setenv('JOBS','192');monkeypatch.setenv('CONFIGS_PER_BATCH','64')
    env=worker_environment(tmp_path,'7',0,1,'sha',finalists=True,jobs=3,configs_per_batch=4)
    env.update(PYTHON=str(stub),PLAN_ONLY='1',FINALIST_PLAN=str(plan),RESUME_RUN=str(run))
    result=subprocess.run(['bash',str(ROOT/'tools/run_prefill_model_asys_box.sh')],env=env,
                          capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    argv=json.loads(next(x.removeprefix('HOST_ARGV ') for x in result.stdout.splitlines() if x.startswith('HOST_ARGV ')))
    assert argv[argv.index('--jobs')+1]=='3' and argv[argv.index('--configs-per-batch')+1]=='4'
    assert 'jobs=3 configs_per_batch=4' in result.stdout
