"""Reparse the old preflight in place without changing raw measurement authority."""
from copy import deepcopy
import json

import pytest

from quactlize.runtime.tuning import digest
from tools import run_decode_refresh as base
from tools.inspect_asys_component_coverage import repair_moe_preflight
from tools.run_moe_prefill_sweep import component_cache, saved
from tools.run_smallm_regressions_asys import sealed


def fixture(tmp_path, returncode=0):
    attempt=tmp_path/'results/gpu-3/attempts/attempt';attempt.mkdir(parents=True)
    run=dict(sha256='old-run',cards=['3'],devices=[dict(pci='original-card')],
             source_revision=dict(quactlize_commit='old-commit'),
             source=dict(sha256='old-source'),harness=dict(sha256='old-harness'))
    authority=dict(run=run['sha256'],shard=0,device=run['devices'][0],phase='screen')
    point=dict(id='point',workload=dict(qtype=12,n=512,k=2048,experts=256,channels=1,
                                      compute=1,storage=2),
               arms=[dict(id='full-arm',recipe=dict(kind='full'))])
    base.sealed(attempt/'job.json',dict(run=run['sha256'],point=point,phase='screen',arms=['full-arm']))
    capture=dict(job=sealed(attempt/'job.json')['sha256'],point=point['id'],phase='screen',
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
