"""Host-only setup recovery; synthetic receipts are not numerical/timing evidence."""
from contextlib import ExitStack
import fcntl
import json
from types import SimpleNamespace

import pytest

from dev.prefill_asys.finalists import partition, counts, load
from dev.strategy.journal import atomic_json
from quactlize.runtime.tuning import digest
from tools.run_prefill_cohort_multi import retry_setup, run_cohort
from test_dense_m_coverage import span


GPUS=['4','5','6','7']
LIMITS=dict(jobs=8,configs_per_batch=16)


@pytest.fixture
def retry_plan(span):
    # Four real families are enough for recovery boundaries. The full 405
    # point inventory is exercised in test_dense_m_coverage, not rewritten
    # to disk for each negative filesystem test.
    selected=[p for p in span['points'] if p['workload']['m']==8192][:4]
    value=span|dict(points=selected,counts=counts(selected))
    value['sha256']=digest({k:v for k,v in value.items() if k!='sha256'})
    return load(json.loads(json.dumps(value)))


@pytest.fixture
def previous(tmp_path,retry_plan):
    span=retry_plan
    root=tmp_path/'previous';root.mkdir();(root/'.coordinator.lock').touch()
    atomic_json(root/'cohort.json',span)
    atomic_json(root/'coordinator.json',dict(gpus=GPUS,cohort=span['sha256'],
        revision='old-source',worker_limits=LIMITS,native_reference=False))
    atomic_json(root/'summary-1.json',dict(status='INCOMPLETE'))
    for i,gpu in enumerate(GPUS):
        out=root/('gpu-'+gpu)/'qwen32-prefill-asys.fixture/results';out.mkdir(parents=True)
        points=[dict(p,native_comparison=False) for p in partition(span['points'],i,4)]
        plan=dict(points=points,model_identity=dict(finalist_plan_sha256=span['sha256'],
            shard=i,shards=4,points=[],native_reference=False))
        plan['plan_sha256']=digest(plan);atomic_json(out/'plan.json',plan)
        if gpu=='6':
            authority=dict(plan=plan['plan_sha256'])
            atomic_json(out/'authority.json',authority)
            atomic_json(out/'execution-plan.json',dict(authority=digest(authority),points=points))
            atomic_json(out/'summary.json',dict(status='SCREEN_COMPLETE',
                points=[dict(id=p['id'],status='SCREEN_COMPLETE') for p in points]))
        else:
            (out/'reducer-gate-1.log').write_text(
                'ValueError: reducer gate workload differs from native dense prefill\n')
    return root


def test_retry_keeps_original_shards_and_does_not_touch_old_evidence(previous,retry_plan):
    before={p:p.read_bytes() for p in previous.rglob('*') if p.is_file()}
    with ExitStack() as locks:
        retry=retry_setup(previous,retry_plan,GPUS,LIMITS,True,locks)
        assert list(retry['retained'])==['6'] and list(retry['failures'])==['4','5','7']
        # Hold the old lock through the new run to exclude an old-source resume.
        with (previous/'.coordinator.lock').open('r') as held:
            with pytest.raises(BlockingIOError):fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert before=={p:p.read_bytes() for p in previous.rglob('*') if p.is_file()}


@pytest.mark.parametrize('fault',['active','gpus','limits','native','cohort','plan','authority',
    'coverage','unknown','partial','batches','unterminated','nested'])
def test_retry_rejects_changed_or_timed_failures(previous,retry_plan,fault):
    gpus=GPUS;limits=LIMITS;no_native=True
    bad=previous/'gpu-4/qwen32-prefill-asys.fixture/results'
    good=previous/'gpu-6/qwen32-prefill-asys.fixture/results'
    with ExitStack() as held:
        if fault=='active':
            lock=held.enter_context((previous/'.coordinator.lock').open('r'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        elif fault=='gpus':gpus=list(reversed(GPUS))
        elif fault=='limits':limits=dict(jobs=4,configs_per_batch=16)
        elif fault=='native':no_native=False
        elif fault=='cohort':atomic_json(previous/'cohort.json',retry_plan|dict(sha256='changed'))
        elif fault=='plan':
            path=bad/'plan.json';value=json.loads(path.read_text());value['points'].pop()
            value['plan_sha256']=digest({k:v for k,v in value.items() if k!='plan_sha256'})
            atomic_json(path,value)
        elif fault=='authority':atomic_json(good/'authority.json',dict(plan='wrong'))
        elif fault=='coverage':
            path=good/'summary.json';value=json.loads(path.read_text());value['points'].pop()
            atomic_json(path,value)
        elif fault=='unknown':(bad/'reducer-gate-1.log').write_text('different error')
        elif fault=='partial':atomic_json(bad/'authority.json',dict(plan='already timed'))
        elif fault=='batches':(bad/'batches').mkdir()
        elif fault=='unterminated':(previous/'summary-1.json').unlink()
        elif fault=='nested':
            path=previous/'coordinator.json';value=json.loads(path.read_text())
            atomic_json(path,value|dict(setup_retry={'previous':'another'}))
        with ExitStack() as locks,pytest.raises(ValueError):
            retry_setup(previous,retry_plan,gpus,limits,no_native,locks)


@pytest.mark.parametrize('rc',[0,1])
def test_coordinator_retries_only_setup_failures_and_reviews_retained(previous,retry_plan,tmp_path,monkeypatch,rc):
    from tools import run_prefill_cohort_multi as runner
    launches=[];reviews=[];(previous/'tc-cache').mkdir()
    class Finished:
        returncode=rc  # Even exit 0 without new measurements cannot complete the cohort.
        def poll(self):return self.returncode
        def wait(self,timeout):return self.returncode
    def launch(command,**kw):
        launches.append(kw['env']);return Finished()
    monkeypatch.setattr(runner.subprocess,'Popen',launch)
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**k:'new-source\n')
    monkeypatch.setattr(runner.subprocess,'run',lambda cmd:reviews.append(cmd) or SimpleNamespace(returncode=0))
    a=SimpleNamespace(output=tmp_path/'retry',resume=False,retry_setup_from=previous,
                      no_native=True,finalist_plan=previous/'cohort.json')
    with ExitStack() as locks:assert run_cohort(a,retry_plan,GPUS,LIMITS,locks)==1
    assert [(e['GPU'],e['SHARD'],e['SHARDS']) for e in launches]==[('4','0','4'),('5','1','4'),('7','3','4')]
    assert all(e['SOURCE_REV']=='new-source' and 'RESUME_RUN' not in e for e in launches)
    assert all(e['TC_CACHE']==str(previous/'tc-cache') for e in launches)
    assert reviews[0][-2:]==['--run',str(previous/'gpu-6/qwen32-prefill-asys.fixture')]
    assert not (a.output/'gpu-6').exists()
    receipt=json.loads((a.output/'coordinator.json').read_text())
    assert receipt['setup_retry']['revision']=='old-source'
    assert list(receipt['setup_retry']['retained'])==['6']
    # A retry's resume keeps both the old completed shard and the NEW source.
    for gpu in ('4','5','7'):(a.output/('gpu-'+gpu)/'qwen32-prefill-asys.retry').mkdir()
    launches.clear();a.resume=True;a.retry_setup_from=None
    with ExitStack() as locks:assert run_cohort(a,retry_plan,GPUS,LIMITS,locks)==1
    assert all(e['SOURCE_REV']=='new-source' and e['RESUME_RUN'].endswith('qwen32-prefill-asys.retry') for e in launches)
