"""Reparse the old preflight in place without changing raw measurement authority."""
from copy import deepcopy
import json

import pytest

from quactlize.runtime.tuning import digest
from tools import run_decode_refresh as base
from tools.inspect_asys_component_coverage import repair_moe_preflight, repair_moe_batch
from tools.run_moe_prefill_sweep import batch_path, component_cache, saved
from tools.run_smallm_regressions_asys import sealed


def fixture(tmp_path, returncode=0, phase='screen'):
    attempt=tmp_path/'results/gpu-3/attempts/attempt';attempt.mkdir(parents=True)
    run=dict(sha256='old-run',cards=['3'],devices=[dict(pci='original-card')],
             source_revision=dict(quactlize_commit='old-commit'),
             source=dict(sha256='old-source'),harness=dict(sha256='old-harness'))
    authority=dict(run=run['sha256'],shard=0,device=run['devices'][0],phase=phase)
    point=dict(id='point',workload=dict(qtype=12,n=512,k=2048,experts=256,channels=1,
                                      compute=1,storage=2),
               arms=[dict(id='full-arm',recipe=dict(kind='full' if phase=='screen' else 'tc'))])
    base.sealed(attempt/'job.json',dict(run=run['sha256'],point=point,phase=phase,arms=['full-arm']))
    capture=dict(job=sealed(attempt/'job.json')['sha256'],point=point['id'],phase=phase,
                 arms={'full-arm':{}},authority=authority)
    base.sealed(attempt/'capture.json',capture)
    (attempt/'profile.process.json').write_text(json.dumps(dict(returncode=returncode)))
    (attempt/'trace.sqlite').write_bytes(b'retained raw trace')
    identity=dict(authority=authority);cid=digest(identity)
    provenance=dict(quactlize_commit='old-commit',source_sha256='old-source',
                    harness_sha256='old-harness',parser_revision=dict(quactlize_commit='new-parser'))
    data=dict(status='PASS',reset={'mean_us':.2},latency_provenance=provenance,
              component_records={cid:dict(component_id=cid,identity=identity,kind='output',
                                          latency_provenance=provenance)})
    return attempt,run,point,capture,data


def test_repaired_batch_is_readable_by_existing_resume_and_is_idempotent(tmp_path):
    attempt,run,point,capture,data=fixture(tmp_path)
    raw={p:p.read_bytes() for p in attempt.iterdir()};before=deepcopy((run,capture,data))
    result=repair_moe_preflight(attempt,capture,data,run)
    assert result['status']=='REPAIRED'
    from pathlib import Path
    receipt=saved(Path(result['batch']),run)
    assert receipt['data']==data and receipt['run']=='old-run'
    assert component_cache(attempt.parents[1],run,0,'screen',point)==data['component_records']
    assert repair_moe_preflight(attempt,capture,data,run)['status']=='ALREADY_REPAIRED'
    assert all(p.read_bytes()==b for p,b in raw.items())
    assert (run,capture,data)==before


@pytest.mark.parametrize('fault',['failed-process','wrong-run','failed-data'])
def test_reparse_never_publishes_failed_or_mismatched_batch(tmp_path,fault):
    attempt,run,point,capture,data=fixture(tmp_path,1 if fault=='failed-process' else 0)
    if fault=='wrong-run':run['sha256']='another-run'
    if fault=='failed-data':data['status']='FAIL'
    with pytest.raises(ValueError):repair_moe_preflight(attempt,capture,data,run)
    assert not (attempt.parents[2]/'preflight.json').exists()
    assert not (attempt.parents[1]/'batches').exists()


def test_confirmation_repair_preserves_failure_and_raw_and_is_old_resume_readable(tmp_path):
    attempt,run,point,capture,data=fixture(tmp_path,phase='confirm')
    folder=attempt.parents[1];path=batch_path(folder,point['id'],'confirm',['full-arm'])
    base.sealed(path,dict(status='FAIL',run=run['sha256'],point=point['id'],error='one interval missing'))
    failure=path.read_bytes();raw={p:p.read_bytes() for p in attempt.iterdir()}
    result=repair_moe_batch(attempt,capture,data,run)
    assert result['status']=='REPAIRED'
    receipt=saved(path,run)
    assert receipt['phase']=='confirm' and receipt['data']==data
    from pathlib import Path
    backup=Path(receipt['previous_failure'])
    assert backup.read_bytes()==failure
    assert component_cache(folder,run,0,'confirm',point)==data['component_records']
    assert not (attempt.parents[2]/'preflight.json').exists()
    repaired=path.read_bytes()
    assert repair_moe_batch(attempt,capture,data,run)['status']=='ALREADY_REPAIRED'
    assert path.read_bytes()==repaired and backup.read_bytes()==failure
    assert all(p.read_bytes()==b for p,b in raw.items())


@pytest.mark.parametrize('fault',['phase','arms','device','failed-process','existing-run','preflight'])
def test_confirmation_repair_rejects_wrong_binding_without_touching_receipt(tmp_path,fault):
    attempt,run,point,capture,data=fixture(tmp_path,phase='confirm')
    path=batch_path(attempt.parents[1],point['id'],'confirm',['full-arm'])
    base.sealed(path,dict(status='FAIL',run='wrong' if fault=='existing-run' else run['sha256'],
                          point=point['id'],error='keep this'))
    before=path.read_bytes()
    if fault=='phase':capture['phase']='screen'
    elif fault=='arms':capture['arms']={'different':{}}
    elif fault=='device':capture['authority']['device']={}
    elif fault=='failed-process':
        (attempt/'profile.process.json').write_text('{"returncode":1}')
    with pytest.raises(ValueError):
        repair_moe_batch(attempt,capture,data,run,preflight=fault=='preflight')
    assert path.read_bytes()==before
    assert not list(path.parent.glob('*.failed-*.json'))
    assert not (attempt.parents[1]/'components').exists()
