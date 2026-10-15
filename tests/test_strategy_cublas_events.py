"""Ordinary provider events and offline reuse; host fixtures are not PPU timings."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from dev.strategy.cublas_events import summarize, review_existing
from dev.strategy.cublas_components import event_components
from dev.strategy.components import recommendation, statistics_checked
from dev.strategy.journal import atomic_json
from quactlize.dispatch.table import Workload
from quactlize.runtime.tuning import digest
from tools.run_kpack_strategy import record_load


def proof(*, old=False):
    provider = dict(provider='CUBLAS_PPU_SDK', entry='cublasGemmEx', compute='CUBLAS_COMPUTE_32F',
                    math_mode=16, output='BF16_M_N')
    rows = [dict(round=r, sample=s, phase=p, copy=s,
        **({'event_span_us':{'input':2., 'provider':40., 'output':3.}[p]} if not old or p=='provider' else {}))
        for r in range(3) for s in range(3) for p in ('input','provider','output')]
    return dict(correctness='PASS', profiler=False, cuda_graph=False, annotations=None,
        zero_a='PASS', omitted_producer='DETECTED', changed_input='PASS',
        pointer_negatives=['BF16_LAST_COPY_ZERO'],
        numerical=[dict(nonfinite=0, cells=128*1024, relative_l1_error=.001) for _ in range(11)],
        weights=[dict(bad=0, bits=1024*5120) for _ in range(2)],
        receipt=dict(copies=3, active_ring_bytes=300, provider=provider), fixture=dict(l2_bytes=100),
        samples_per_round=3, warmup_calls=12, event_warmup_calls=1, rounds=3,
        provider_loaded_images={'blas.so':'sha'}, measured=rows,
        event_scope='AUXILIARY_PROVIDER_API_SPAN_INCLUDING_GAPS_NOT_SINGLE_KERNEL' if old else
                    'SEPARATE_API_EVENTS_PROVIDER_INCLUDES_INTERNAL_REDUCTION_AND_LAUNCH_GAPS')


def saved(root, *, old=True):
    authority=dict(source='immutable-historical-source')
    atomic_json(root/'authority.json',authority)
    atomic_json(root/'events/result.json',dict(status='PASS',authority_sha256=digest(authority),data=proof(old=old)))
    atomic_json(root/'events.process.json',dict(rc=0,timeout=False))
    atomic_json(root/'events.command.json',['python3','historical-runner.py','--worker','events'])


def test_provider_is_a_black_box_call_not_a_single_kernel():
    data=summarize(proof())
    assert data['timers']['provider']['median_us']==40.
    assert data['timers']['provider']['scope']=='PROVIDER_CALL_EVENTS'
    assert data['timers']['provider']['internal_reduction_included'] is True
    assert data['timers']['provider']['kernel_count']=='OPAQUE_PROVIDER'
    assert data['timers']['input']['median_us']==2. and data['timers']['output']['median_us']==3.
    assert data['missing_adapter_timers']==[] and data['dequant_timed'] is False
    assert data['performance_admitted'] is False


@pytest.mark.parametrize('edit', [
    lambda d:d.update(profiler=True), lambda d:d.update(cuda_graph=True),
    lambda d:d.update(annotations={'profiler':'yes'}), lambda d:d.update(correctness='PENDING'),
    lambda d:d.update(omitted_producer='SKIPPED'), lambda d:d.update(pointer_negatives=[]),
    lambda d:d.update(numerical=[]), lambda d:d['numerical'][0].update(relative_l1_error=float('nan')),
    lambda d:d['numerical'][0].update(relative_l1_error=.005), lambda d:d['weights'][0].update(bad=1),
    lambda d:d.update(warmup_calls=11), lambda d:d.update(event_warmup_calls=0),
    lambda d:d['receipt'].update(active_ring_bytes=100),
    lambda d:d['receipt']['provider'].update(compute='CUBLAS_COMPUTE_16F'),
    lambda d:d['measured'].pop(), lambda d:d['measured'][0].update(phase='provider'),
    lambda d:d['measured'][1].update(event_span_us=-1.),
    lambda d:d['measured'][1].update(event_span_us=float('inf')),
])
def test_invalid_or_profiled_measurements_are_not_promoted(edit):
    data=proof();edit(data)
    with pytest.raises(ValueError):summarize(data)


def test_provider_only_reuse_does_not_invent_unmeasured_adapter_times():
    old=proof(old=True)
    with pytest.raises(ValueError,match='missing event time'):summarize(old)
    result=summarize(old,provider_only=True)
    assert result['missing_adapter_timers']==['input','output']
    assert set(result['timers'])=={'provider'}
    with pytest.raises(ValueError,match='adapter timers'):
        event_components(result,Workload(11,1024,5120,128),'authority','body')


def test_offline_review_retains_authority_and_never_changes_old_files(tmp_path):
    root=tmp_path/'old/probe';saved(root)
    before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    data=review_existing(root.parent)
    assert data['device_launches']==0 and data['reused_measurements']
    assert data['timers']['provider']['median_us']==40.
    assert data['original_authority']=={'source':'immutable-historical-source'}
    assert before=={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('filename,edit', [
    ('events.process.json',lambda r:r.update(rc=1)),
    ('events.process.json',lambda r:r.update(timeout=True)),
    ('events/result.json',lambda r:r.update(authority_sha256='wrong')),
    ('events/result.json',lambda r:r.update(status='FAIL')),
    ('events.command.json',lambda r:r.append('--profiled')),
])
def test_offline_review_rejects_failed_or_wrong_worker(tmp_path,filename,edit):
    saved(tmp_path);path=tmp_path/filename;row=json.loads(path.read_text());edit(row);atomic_json(path,row)
    with pytest.raises(ValueError):review_existing(tmp_path)


def test_event_components_compose_provider_once_without_extra_internal_reducer(tmp_path):
    from dev.strategy.prefill_retry import checked_component
    data=summarize(proof()); w=Workload(11,1024,5120,128)
    rows=event_components(data,w,'authority','body')
    assert [r['kind'] for r in rows]==['adapter','provider','postprocess']
    assert rows[1]['timed_kernel_types'] is None
    records={r['component_id']:r for r in rows}
    for i,r in enumerate(rows):
        path=tmp_path/f'{i}.json';atomic_json(path,r);assert record_load(path,r['identity'])==r
        atomic_json(tmp_path/'components'/(r['component_id']+'.json'),r)
        assert checked_component(tmp_path,r['component_id'],'authority')[0]==r
    records['baseline']=statistics_checked([[50.]*3 for _ in range(3)])
    result=recommendation({'original':['baseline'],'full':[r['component_id'] for r in rows]},records)
    assert result['costs_us']['full']==45. and 'MIXED_METHODS' in result['scope']
    assert result['production_changed'] is False


def test_offline_cli_needs_neither_sdk_nor_gpu(tmp_path):
    source=Path(__file__).resolve().parents[1]
    old=tmp_path/'old';saved(old)
    run=subprocess.run([sys.executable,str(source/'tools/probe_kpack_cublas_ordinary.py'),
        '--sdk','/deliberately-absent-sdk','--review-existing',str(old),'--output',str(tmp_path/'review')],
        capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    assert 'device_launches' in run.stdout and (tmp_path/'review.results.tgz').is_file()
    assert not (tmp_path/'review/ordinary.sqlite').exists()


def test_current_measurement_and_campaign_have_no_asys_or_range_dependency():
    import inspect
    import dev.strategy.cublas_ordinary as ordinary
    import tools.probe_kpack_cublas_ordinary as probe
    import tools.run_kpack_strategy_prefill as campaign
    for fn in (ordinary.measure,probe.worker,probe.run_events,probe.run,campaign.main):
        text=inspect.getsource(fn)
        assert 'asys' not in text.lower() and 'Ranges(' not in text and 'parse_trace' not in text
    wrapper=Path(__file__).resolve().parents[1]/'tools/run_kpack_cublas_ordinary_box.sh'
    assert 'asight/' not in wrapper.read_text()
