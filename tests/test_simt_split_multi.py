"""Host-only orchestration checks, never synthetic device/performance evidence."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace as N

import pytest

import tools.run_simt_split_multi as multi
from dev.strategy.journal import atomic_json
from dev.gemv_simt.split_plan import plan, inventory, SPLITS
from tools.run_simt_split import cells, build_args, build_root


def probe(card):
    return dict(device=dict(host='unit-test-host',pci=f'0000:{int(card):02x}:00.0',ordinal=0,
        compute_units=72,warp_size=32,max_threads=1024,DriverGetVersion=13000,RuntimeGetVersion=13000),
        sdk='unit-test-sdk',l2=dict(bytes=67108864,reported=0,source='OPERATOR_VERIFIED'))


@pytest.mark.parametrize('workers',(1,2,3,7,8))
def test_complete_nonoverlapping_whole_family_assignments(workers):
    shards=multi.assign_families(workers)
    assert shards==multi.assign_families(workers)
    assert len(shards)==workers and all(shards)
    assert sorted(i for ids in shards for i in ids)==list(range(74))
    assert sum(plan(ids)['numeric_pairs'] for ids in shards)==2224
    assert sum(plan(ids)['timing_pairs'] for ids in shards)==1776
    assert all(len(cells(inventory()[i]))>=12 for ids in shards for i in ids)
    if workers==7:
        loads=[sum(multi.family_work(inventory()[i]) for i in ids) for ids in shards]
        assert max(loads)/min(loads)<1.1


@pytest.mark.parametrize('ids',([],[-1],[74],[1,1],[1.0]))
def test_invalid_point_selection_rejected(ids):
    with pytest.raises(ValueError):plan(ids)


def test_physical_not_visible_ordinal_check():
    assert len(multi.check_devices([probe(c) for c in ('1','2','3','4','5','6','7')]))==7
    # All child-visible ordinals are zero, as expected, but PCI is distinct.
    with pytest.raises(ValueError,match='same physical'):multi.check_devices([probe('1'),probe('1')])


@pytest.mark.parametrize('field',('sdk','l2','compute_units','DriverGetVersion'))
def test_mixed_devices_rejected(field):
    a,b=probe('1'),probe('2')
    if field=='sdk':b['sdk']='different'
    elif field=='l2':b['l2']['bytes']*=2
    else:b['device'][field]+=1
    with pytest.raises(ValueError,match='mixed'):multi.check_devices([a,b])


def fake_summary(ids,profile=False):
    frozen=plan(ids)
    return dict(status='PASS',expected=frozen['numeric_pairs'],completed=frozen['numeric_pairs'],
        points=[dict(point=p['point'],rc=0,completed=len(cells(p)),expected=len(cells(p))) for p in frozen['points']],
        timings=[dict(point=p['point'],m=m,config=c,split=s) for p in frozen['points'] for m,c,s in cells(p) if m in p['timing_m']],
        profiles=[dict(status='PASS',identity=dict(point=p['point'],split=s)) for p in frozen['points'] for s in SPLITS
            if profile and p['kind']=='all-format-control' and p['workload']['compute']=='bf16' and p['workload']['operation']=='indexed'])


def test_aggregate_exact_coverage_and_profile_denominators():
    shards=multi.assign_families(7);summaries=[fake_summary(ids,True) for ids in shards]
    out=multi.aggregate(shards,summaries,[0]*7,True)
    assert out['status']=='PASS' and out['completed']==2224 and out['timing_pairs']==1776
    assert len(out['profiles'])==20 and not out['performance_admitted'] and not out['production_changed']
    for damage in ('omit','duplicate','wrong-owner','profile','rc'):
        rows=deepcopy(summaries);rc=[0]*7
        if damage=='omit':rows[0]['timings'].pop()
        elif damage=='duplicate':rows[0]['timings'].append(rows[0]['timings'][0])
        elif damage=='wrong-owner':rows[0]['timings'][0]['point']=shards[1][0]
        elif damage=='profile':next(r for r in rows if r['profiles'])['profiles'].pop()
        else:rc[0]=1
        assert multi.aggregate(shards,rows,rc,True)['status']=='INCOMPLETE'


def test_worker_uses_one_shared_image_root_and_its_own_points(tmp_path):
    a=N(sdk=Path('/sdk'),output=tmp_path,l2_bytes=67108864,acu=Path('/acu'))
    output=tmp_path/'gpu-1';cmd=multi.worker_command(a,[1,4,25],output)
    assert cmd[cmd.index('--build-root')+1]==str(tmp_path/'build')
    assert cmd[cmd.index('--points')+1]=='1,4,25' and '--no-pack' in cmd and '--resume' not in cmd
    assert build_root(a)==tmp_path/'build' and build_args(a)==[]
    a.build_root=tmp_path/'shared'
    assert build_root(a)==tmp_path/'shared' and build_args(a)==['--build-root',str(tmp_path/'shared')]
    output.mkdir();assert '--resume' in multi.worker_command(a,[1,4,25],output)


def test_real_plan_cli_without_gpu_and_resume_rebinding(tmp_path):
    path=tmp_path/'run'
    command=[sys.executable,str(multi.ROOT/'tools/run_simt_split_multi.py'),'--sdk',str(tmp_path),
             '--output',str(path),'--devices','1,2,3,4,5,6,7','--plan-only']
    subprocess.run(command,check=True,capture_output=True)
    first=(path/'multi.json').read_bytes()
    subprocess.run(command+['--resume'],check=True,capture_output=True)
    assert first==(path/'multi.json').read_bytes() and not (path/'build').exists()
    failed=subprocess.run(command+['--resume','--devices','1,2,3'],capture_output=True)
    assert failed.returncode and b'resume binding changed' in failed.stderr


def test_fake_process_coordinator_builds_once_and_reuses_on_resume(tmp_path,monkeypatch):
    """Exercise orchestration using explicitly fake children, no GPU calls."""
    cards=list(map(str,range(1,8)));commands=[];builds=[]
    a=N(output=tmp_path/'run',sdk=tmp_path/'sdk',l2_bytes=67108864,acu=None,resume=False,plan_only=False)
    monkeypatch.setattr(multi,'harness_identity',lambda root:{'unit_test_only':True})
    def fake_probe(cmd,**kw):
        output=Path(cmd[cmd.index('--output')+1]);card=kw['env']['CUDA_VISIBLE_DEVICES']
        assert '--probe-only' in cmd
        atomic_json(output/'device.json',probe(card))
        return N(returncode=0)
    monkeypatch.setattr(multi.subprocess,'run',fake_probe)
    def fake_build(sdk,output):
        builds.append(output);atomic_json(output/'manifest.json',dict(unit_test_only=True))
    monkeypatch.setattr(multi,'build',fake_build)
    monkeypatch.setattr(multi,'shared_identity',lambda *args:dict(unit_test_only=True))
    def fake_child(cmd,**kw):
        commands.append(cmd)
        assert builds==[a.output/'build']
        ids=list(map(int,cmd[cmd.index('--points')+1].split(',')))
        output=Path(cmd[cmd.index('--output')+1]);card=kw['env']['CUDA_VISIBLE_DEVICES']
        assert kw['start_new_session'] and card in cards
        assert cmd[cmd.index('--build-root')+1]==str(a.output/'build')
        atomic_json(output/'summary.json',fake_summary(ids))
        atomic_json(output/'authority.json',dict(device=probe(card),harness={'unit_test_only':True}))
        return N(returncode=0,poll=lambda:0)
    monkeypatch.setattr(multi.subprocess,'Popen',fake_child)
    assert multi.run(a,cards)==0 and len(commands)==7 and len(builds)==1
    first=(a.output/'multi.json').read_bytes();a.resume=True
    assert multi.run(a,cards)==0 and len(commands)==14 and len(builds)==1
    assert all('--resume' in cmd for cmd in commands[7:])
    assert first==(a.output/'multi.json').read_bytes()


def test_one_archive_excludes_libraries_objects_and_nested_archives(tmp_path):
    atomic_json(tmp_path/'summary.json',dict(unit_test_only=True))
    atomic_json(tmp_path/'gpu-1/points/0/m1-c0-s2.json',dict(unit_test_only=True))
    for name in ('test.so','test.o','review-old.tgz','weights.bin'):(tmp_path/name).write_bytes(b'not-evidence')
    (tmp_path/'sample.acurep').write_bytes(b'test-counter-data')
    archive=multi.package(tmp_path)
    with tarfile.open(archive) as reader:
        assert set(reader.getnames())=={'summary.json','gpu-1/points/0/m1-c0-s2.json','transfer-manifest.json'}


def test_shell_multidevice_dispatch_and_safe_resume():
    shell=multi.ROOT/'tools/run_simt_split_box.sh'
    subprocess.run(['bash','-n',str(shell)],check=True)
    text=shell.read_text()
    assert 'args+=(--devices "${GPUS}")' in text and 'results/multi.json' in text
    assert 'do not reshard' in text and 'git submodule update' not in text
    single=(multi.ROOT/'tools/run_simt_split.py').read_text()
    assert single.count('*build_args(a)')==2
    assert 'point belongs to another worker' in single
