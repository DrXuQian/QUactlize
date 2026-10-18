"""Host-only parallel handoff checks; the wrapper never claims full timing."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'tools/run_kpack_strategy_prepare_box.sh'


def test_invalid_task_does_not_start_a_campaign(tmp_path):
    p=subprocess.run(['bash',str(SCRIPT),'all-timing'],capture_output=True,text=True,
                     env=dict(os.environ,RUN_ROOT=str(tmp_path)))
    assert p.returncode==2 and 'Usage:' in p.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('task,cohort,max_modules',[
    # R26 broadened prefill geometry and current decode reader coverage. R27
    # adds two observed delivery parents (264/74 current modules). These are
    # host proposal bounds, NOT the closure runner's zero-recompile inventory.
    ('prefill-resources','prefill',280),('device-pilot','pilot',80)])
def test_plan_only_and_resume_have_no_sdk_device_or_claimed_timings(tmp_path,task,cohort,max_modules):
    # Actual host rule generation and resource inventory, not a fake GPU.
    env=dict(os.environ,PYTHON=sys.executable,RUN_ROOT=str(tmp_path),PLAN_ONLY='1',
             PPU_SDK=str(tmp_path/'MISSING_SDK'),CUDA_VISIBLE_DEVICES='')
    cmd=['bash',str(SCRIPT),task]
    p=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    runs=[v for v in tmp_path.glob('kpack-rule-prepare.*') if v.is_dir()]
    assert len(runs)==1
    run=runs[0];plan=json.loads((run/'plan/plan.json').read_text())
    assert plan['cohort']==cohort
    inv=json.loads((run/'resources/inventory.json').read_text())
    assert 0<inv['counts']['modules']<=max_modules
    assert not (run/'cache').exists() and not (run/'resources/authority.json').exists()
    assert not (run/'pilot').exists()
    assert 'scope=PREPARATION_NOT_FULL_TIMING' in p.stdout
    assert 'Current Docker shell is preserved' in p.stdout
    assert 'STRATEGY_PREPARE_DONE' not in p.stdout
    frozen=(run/'plan/plan.json').read_bytes()
    resumed=subprocess.run(cmd,cwd=ROOT,env=dict(env,RESUME_DIR=str(run)),capture_output=True,text=True)
    assert resumed.returncode==0,resumed.stdout+resumed.stderr
    assert (run/'plan/plan.json').read_bytes()==frozen
    wrong='device-pilot' if task=='prefill-resources' else 'prefill-resources'
    failed=subprocess.run(['bash',str(SCRIPT),wrong],cwd=ROOT,
        env=dict(env,RESUME_DIR=str(run)),capture_output=True,text=True)
    assert failed.returncode!=0 and 'resume task differs' in failed.stderr
    assert (run/'plan/plan.json').read_bytes()==frozen
