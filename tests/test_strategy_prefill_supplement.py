"""Host-only append/reuse checks. Synthetic samples are NOT PPU results."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from dev.strategy.components import component_key, statistics_checked
from dev.strategy.journal import ArmJournal, atomic_json
from dev.strategy.prefill_supplement import (
    MEASUREMENT_SOURCE, SupplementJournal, canonicalize, check_harness, compatible,
    import_passes, job_key, measurement_signature,
)
from dev.strategy.resources import module_spec
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.tuning import digest
from tools.run_kpack_strategy_prefill import jobs, summarize


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr('dev.strategy.prefill_supplement.check_harness', lambda *a:None)
    old, new = tmp_path/'old', tmp_path/'new'
    w = Workload(12, 256, 512, 128)
    r = Recipe.make('tc', 'fq', symbol='tc_host_test', qtype=12, tm=16, tn=32, tk=128,
                    wm=16, wn=16, stages=2, ap=0, dn=32, split=1, persistent=-1, grid_mode=0, grid_b=0)
    spec = module_spec(w, r)
    arms = [dict(id=aid, recipe=asdict(r), roles=[role], aliases=[], spec=spec, module='module',
                 binding=dict(query='query', launch=dict(algorithm=0, split=1, grid=0, work=64)))
            for aid, role in [('original', 'original'), ('b'*64, 'control')]]
    point = dict(id=w.key, workload=asdict(w), arms=arms, sf_configs=[], full_configs=[])
    plan = json.loads(json.dumps(dict(points=[point], blocked=[], counts=dict(requested_points=1),
                                      scope='HOST_TEST', excluded=[])))
    plan['plan_sha256'] = digest(plan)
    authority = dict(plan=plan['plan_sha256'], resource=dict(device='physical-card', sdk='sdk', cache='cache'),
        harness={}, tc={'module':'image'}, body_source='body', oracle='oracle', helpers='helpers',
        dequant='dequant', l2={'bytes':64}, protocol='unchanged', scope='HOST_TEST')
    atomic_json(old/'authority.json', authority)
    atomic_json(old/'execution-plan.json', plan)
    journal = ArmJournal(old/'arms', dict(sha256=digest(authority)), 'a'*32)
    node = dict(name='void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<X>>(Params)',
                symbol='sym', grid=[64,1,1], block=[32,1,1])
    cid, identity = component_key(w,r,node,authority=journal.authority,source='body',cache='cold')
    component = dict(statistics_checked([[3.]*3]*3), identity=identity, component_id=cid,
                     kind='producer', timed_kernel_types=1, calls_per_graph=16,
                     first_use_excluded=True, host_launch_cost_added=False)
    component['record_sha256'] = digest(component)
    atomic_json(old/'components'/(cid+'.json'), component)
    for arm in plan['points'][0]['arms']:
        journal.start(point, arm)
        journal.finish(point, arm, 'PASS', data={'correctness':'PASS','components':[cid]})
    candidate = deepcopy(plan)
    p = candidate['points'][0]
    p['arms'][1]['id'] = 'c'*64; p['arms'][1]['roles'] = ['rule']
    added = deepcopy(p['arms'][1]); added.update(id='d'*64, module='new-module')
    p['arms'].append(added)
    candidate['plan_sha256'] = digest({k:v for k,v in candidate.items() if k!='plan_sha256'})
    current = dict(authority, plan=candidate['plan_sha256'], tc=dict(authority['tc'], **{'new-module':'new-image'}))
    atomic_json(new/'authority.json', current)
    return old,new,plan,candidate,authority,current,cid


def snapshot(root):
    return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_reuse_alias_preserves_raw_samples_and_only_new_image_is_due(tmp_path, monkeypatch):
    old,new,_,plan,_,current,cid = fixture(tmp_path,monkeypatch)
    original = snapshot(old)
    index = import_passes(old,new,plan,current,jobs,Path('.'))
    assert index['passed_jobs'] == 2 and snapshot(old) == original
    journal = SupplementJournal(new/'arms', {'sha256':digest(current)}, 'b'*32)
    p = plan['points'][0]
    assert [journal.due(p,a) for a in p['arms']] == [False,False,True]
    bound = index['receipts'][p['id']+'/c'+'c'*63]
    assert (new/bound['path']).read_bytes() == (old/'arms'/p['id']/('b'*64)/'current.json').read_bytes()
    assert journal.read(p,p['arms'][1])['identity']['arm']['roles'] == ['control']
    assert (new/'components'/(cid+'.json')).read_bytes() == (old/'components'/(cid+'.json')).read_bytes()
    with pytest.raises(ValueError, match='already complete'): journal.start(p,p['arms'][1])
    assert import_passes(old,new,plan,current,jobs,Path('.')) == index
    rows = summarize(new,plan,journal)
    assert rows[0]['status'] == 'INCOMPLETE'


@pytest.mark.parametrize('damage', ['device','sdk','cache','helpers','dequant','oracle','protocol','image'])
def test_semantically_similar_but_different_authority_cannot_reuse(tmp_path,monkeypatch,damage):
    old,new,_,plan,_,current,_ = fixture(tmp_path,monkeypatch)
    current = deepcopy(current)
    if damage in ('device','sdk','cache'): current['resource'][damage]='changed'
    elif damage=='image': current['tc']['module']='recompiled-image'
    else: current[damage]='changed'
    with pytest.raises(ValueError): import_passes(old,new,plan,current,jobs,Path('.'))
    assert not (new/'supplement/index.json').exists()


def test_same_image_different_launch_is_not_a_match(tmp_path,monkeypatch):
    _,_,_,plan,_,current,_ = fixture(tmp_path,monkeypatch)
    p=plan['points'][0];a=deepcopy(p['arms'][0]);b=deepcopy(a)
    b['binding']['launch']['split']=2
    assert job_key(p,a,current)!=job_key(p,b,current)


def test_resume_rejects_changed_imported_evidence(tmp_path,monkeypatch):
    old,new,_,plan,_,current,cid = fixture(tmp_path,monkeypatch)
    import_passes(old,new,plan,current,jobs,Path('.'))
    path=new/'components'/(cid+'.json');v=json.loads(path.read_text());v['samples_us'][0][0]=99
    atomic_json(path,v)
    with pytest.raises(ValueError): import_passes(old,new,plan,current,jobs,Path('.'))


def test_symbol_reuse_never_adds_an_unproposed_geometry(tmp_path,monkeypatch):
    _,_,old,_,_,_,_ = fixture(tmp_path,monkeypatch)
    p=old['points'][0];r=Recipe.from_dict(p['arms'][0]['recipe'])
    renamed=Recipe.make(r.kind,r.route,**(dict(r.parameters)|{'symbol':'new-label'}))
    point=dict(id=p['id'],workload=p['workload'],candidates=[asdict(renamed)])
    aliases=canonicalize([point],old)
    assert len(aliases)==1 and point['candidates']==[asdict(r)]
    assert len(point['candidates'])==1
    with pytest.raises(ValueError,match='every point'): canonicalize([],old)


def test_current_measured_frontend_still_has_reviewed_source():
    path=Path(__file__).resolve().parents[1]/'tools/run_kpack_strategy_prefill.py'
    assert measurement_signature(path.read_text())==MEASUREMENT_SOURCE


@pytest.mark.parametrize('interpreter', sorted({sys.executable} | {
    path for name in ('python3.10', 'python3.11', 'python3.12', 'python3.13')
    if (path := shutil.which(name))
}))
def test_reviewed_boundary_across_available_python_versions(interpreter):
    root=Path(__file__).resolve().parents[1]
    script='''
from pathlib import Path
from dev.strategy.prefill_supplement import (
    BASE_RUNNER, RUNNER, MEASUREMENT_SOURCE, check_harness, measurement_signature,
)
from quactlize.runtime.tuning import digest
import dev.strategy.prefill_supplement as mod

files={RUNNER:BASE_RUNNER, 'unchanged-timer.py':'same'}
old=dict(files=files, sha256=digest(files))
mod.BASE_HARNESS=old['sha256']
check_harness(old, old, Path('.'))
source=Path(RUNNER).read_text(encoding='utf-8')
assert measurement_signature(source)==MEASUREMENT_SOURCE
for original,changed in [('def child(args):','def child(args):\\n    return 0'),
                         ('def jobs(point):','def jobs(point):\\n    return []'),
                         ('def child(args):','@changed_semantics\\ndef child(args):')]:
    assert original in source
    assert measurement_signature(source.replace(original,changed,1))!=MEASUREMENT_SOURCE
print(MEASUREMENT_SOURCE)
'''
    result=subprocess.run([interpreter,'-c',script],cwd=root,text=True,capture_output=True,check=True)
    assert result.stdout.strip()==MEASUREMENT_SOURCE


@pytest.mark.parametrize('source, message', [
    ('def jobs(p): return []\n', 'missing measured functions'),
    ('def child(a): return 0\ndef jobs(p): return []\ndef child(a): return 1\n',
     'duplicate measured function'),
])
def test_missing_or_shadowed_measured_function_rejected(source,message):
    with pytest.raises(ValueError,match=message): measurement_signature(source)


def test_decorators_and_body_edits_are_not_ignored():
    source='def child(a): return 0\ndef jobs(p): return []\n'
    before=measurement_signature(source)
    assert measurement_signature('@decorator\n'+source)!=before
    assert measurement_signature(source.replace('return 0','return 1'))!=before
    assert measurement_signature(source.replace('return []','return [p]'))!=before
    assert measurement_signature(source.replace('def child','async def child'))!=before


def test_source_boundary_rejects_timer_or_child_changes(tmp_path,monkeypatch):
    import dev.strategy.prefill_supplement as mod
    root=Path(__file__).resolve().parents[1]
    before={mod.RUNNER:mod.BASE_RUNNER, 'timer.py':'same'}
    old=dict(files=before,sha256=digest(before))
    monkeypatch.setattr(mod,'BASE_HARNESS',old['sha256'])
    files=dict(before, **{'dev/strategy/prefill_candidates.py':'new-host-rules'})
    mod.check_harness(old,dict(files=files,sha256=digest(files)),root)
    files['timer.py']='changed'
    with pytest.raises(ValueError,match='measurement inputs'):
        mod.check_harness(old,dict(files=files,sha256=digest(files)),root)
    path=tmp_path/mod.RUNNER;path.parent.mkdir(parents=True)
    path.write_text('def child(args): return 0\ndef jobs(p): return []\n')
    with pytest.raises(ValueError,match='measured child'):
        mod.check_harness(old,old,tmp_path)


def test_supplement_export_keeps_receipts_and_binding_roles_separate(tmp_path,monkeypatch):
    import lzma
    from tools.pack_kpack_strategy_review import main
    old,new,_,plan,_,current,_=fixture(tmp_path,monkeypatch)
    import_passes(old,new,plan,current,jobs,Path('.'))
    atomic_json(new/'summary.json',dict(status='INCOMPLETE',points=[]))
    output=tmp_path/'review.xz'
    assert main([str(new),'--output',str(output)])==0
    with lzma.open(output,'rt') as stream: rows=[json.loads(line) for line in stream]
    assert rows[-1]['export_status']=='COMPLETE'
    bindings=[r for r in rows if r['type']=='reuse_binding']
    assert len(bindings)==2
    assert next(r for r in bindings if r['id']=='c'*64)['roles']==['rule']
    sources=[r for r in rows if r['type']=='arm']
    assert next(r for r in sources if r['id']=='b'*64)['roles']==['control']
    assert all(r['measurement_origin']=='SUPPLEMENT_PASS_UNCHANGED' for r in sources)


def test_import_keeps_the_earlier_isolated_receipt_authority(tmp_path,monkeypatch):
    from dev.strategy.prefill_retry import import_passes as import_retry
    old,new,old_plan,plan,original,current,cid=fixture(tmp_path,monkeypatch)
    middle=tmp_path/'provider-migration'
    middle_authority=dict(original,harness={'epoch':'second'})
    atomic_json(middle/'authority.json',middle_authority)
    atomic_json(middle/'execution-plan.json',old_plan)
    import_retry(old,middle,old_plan,middle_authority,jobs)
    current=dict(current,harness=middle_authority['harness'])
    index=import_passes(middle,new,plan,current,jobs,Path('.'))
    assert set(index['source_authorities'])=={digest(original),digest(middle_authority)}
    journal=SupplementJournal(new/'arms',{'sha256':digest(current)},'b'*32)
    p=plan['points'][0]
    assert journal.read(p,p['arms'][0])['identity']['authority']=={'sha256':digest(original)}
    assert (new/'components'/(cid+'.json')).read_bytes()==(old/'components'/(cid+'.json')).read_bytes()
