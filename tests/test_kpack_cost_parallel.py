import copy
import json
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from tools import run_kpack_cost_parallel as parallel
from tools.kpack_cost_supplement import plan
from tools.run_kpack_cost_supplement import same_device_class
from tools.run_kpack_cost_supplement import perform
from quactlize.runtime.tuning import digest
from test_kpack_cost_supplement import record


def probe(device):
    return dict(device=dict(ordinal=0,pci=f'0000:{device:02x}:00.0',visible_devices=str(device),
                           compute_units=72,l2_bytes=64<<20,warp=32),
                runtime={'lib':'same'},packages={'numpy':'same'})


def test_all_weights_are_indivisible_and_finite():
    points=plan()['points'];names=parallel.work_items(points)
    assert len(names)==len(set(names))==86
    assert set(names)=={p['weight_id'] for p in points}
    assert sum(sum(p['weight_id']==n for p in points) for n in names)==458
    assert parallel.work_items(list(reversed(points)))==names


def test_same_model_cross_card_is_admitted_without_changing_identity():
    records=[probe(i) for i in range(8)];before=copy.deepcopy(records)
    parallel.verify_devices(records)
    assert records==before
    assert same_device_class(records[0]['device'],records[7]['device'])


@pytest.mark.parametrize('fault',['duplicate','missing-pci','l2','cu','warp','runtime','packages'])
def test_foreign_or_aliased_workers_fail_before_measurement(fault):
    records=[probe(i) for i in range(8)]
    if fault=='duplicate':records[-1]['device']['pci']=records[0]['device']['pci']
    elif fault=='missing-pci':records[-1]['device']['pci']=''
    elif fault in ('l2','cu','warp'):
        records[-1]['device'][{'l2':'l2_bytes','cu':'compute_units','warp':'warp'}[fault]]+=1
    else:records[-1][fault]['different']='plant'
    with pytest.raises(ValueError):parallel.verify_devices(records)


def test_one_process_per_device_dynamic_queue_and_failure_continuation(tmp_path,monkeypatch):
    bundle=tmp_path/'bundle';sdk=tmp_path/'sdk';bundle.mkdir();sdk.mkdir()
    (bundle/'manifest.json').write_text('{}')
    original=plan();chosen=parallel.work_items(original['points'])[:16]
    planned=original|dict(points=[p for p in original['points'] if p['weight_id'] in chosen],plan_sha256='plan')
    monkeypatch.setattr(parallel,'verify',lambda *args:({'runtime':{'lib':'same'}},planned))
    monkeypatch.setattr(parallel,'probe',lambda a,d:probe(d))
    monkeypatch.setattr(parallel,'command',lambda a,out:['fake',str(out)])
    lock=threading.Lock();active=set();started=[];finished=[]
    class Child:
        def __init__(self,cmd,env,**kwargs):
            self.device=int(env['CUDA_VISIBLE_DEVICES']);self.name=Path(cmd[1]).name
            self.pid=100+self.device
            assert kwargs['start_new_session']
            assert env['OMP_NUM_THREADS']=='1'
            with lock:
                assert self.device not in active
                active.add(self.device);started.append(self.name)
        def wait(self):
            time.sleep(.01*(1+self.device%3))
            with lock:active.remove(self.device);finished.append(self.name)
            return 1 if self.name==chosen[3] else 0
    monkeypatch.setattr(parallel.subprocess,'Popen',Child)
    observed={}
    def collect(output,plan,assignment,authority):
        observed.update(assignment)
        assert len(assignment)==17 and len(started)==len(set(started))==17
        assert len(finished)==17 and not active
        assert assignment[chosen[3]]['status']=='FAIL'
        assert sum(t['status']=='PASS' for t in assignment.values())==16
        return dict(status='INCOMPLETE',complete=16,expected_components=17)
    monkeypatch.setattr(parallel,'collect',collect)
    a=SimpleNamespace(sdk=sdk,bundle=bundle,output=tmp_path/'results',devices=list(range(8)),summarize_only=False)
    assert parallel.run(a)==1
    assert {t['device'] for t in observed.values()}==set(range(8))
    assert json.loads((a.output/'assignment.json').read_text())==observed


def test_box_entry_never_builds_and_preserves_docker_shell(tmp_path):
    script=parallel.ROOT/'tools/run_kpack_cost_parallel_ppu_box.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    source=script.read_text()
    assert 'build_kpack_cost_supplement.py' not in source and 'run_kpack_cost_parallel.py' in source
    r=subprocess.run(['bash','-c','PPU_SDK="$2/absent" bash "$1"; printf "PARENT_ALIVE\\n"',
                      'bash',str(script),str(tmp_path)],text=True,capture_output=True)
    assert r.returncode==0 and 'PARENT_ALIVE' in r.stdout


def test_merge_keeps_cross_card_provenance_and_partial_success(tmp_path):
    points=[];assignment={}
    root=dict(plan_sha256='plan',bundle_sha256='bundle',runtime={'lib':'same'},sources={'code':'same'},
              probes={str(i):probe(i) for i in (0,1)})
    for i,n in enumerate((256,512)):
        name=f'q12-n{n}-k512-e1'
        p=dict(id=name+'-t128-real',weight_id=name,q=12,n=n,k=512,experts=1,active_ids=[0],
               routes={'0':{'historical':None}})
        points.append(p);assignment[name]=dict(device=i,status='FAIL')
        path=tmp_path/'weights'/name
        for folder in ('components','failures'):(path/folder).mkdir(parents=True)
        authority=dict(plan_sha256='plan',bundle_sha256='bundle',runtime=root['runtime'],sources=root['sources'],
                       python_packages=probe(i)['packages'],device=probe(i)['device'])
        (path/'authority.json').write_text(json.dumps(authority))
        args=SimpleNamespace(output=path,authority=digest(authority))
        perform(args,p['id']+'-r0-current',lambda i=i:record()|dict(device=probe(i)['device']))
    result=parallel.collect(tmp_path,dict(points=points),assignment,root)
    assert result['complete']==2 and result['status']=='INCOMPLETE'
    assert {p['device']['pci'] for p in result['components'].values()}=={probe(i)['device']['pci'] for i in (0,1)}
    # Same-card or source impersonation must not silently become a result.
    wrong=copy.deepcopy(root);wrong['probes']['1']['device']=probe(0)['device']
    with pytest.raises(ValueError,match='provenance'):parallel.collect(tmp_path,dict(points=points),assignment,wrong)
