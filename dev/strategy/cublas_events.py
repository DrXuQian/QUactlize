"""Host-only validation of ordinary cuBLAS CUDA-event samples, not traces."""
import json
import math
from pathlib import Path
import statistics

from dev.strategy.components import statistics_checked
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

PHASES = ('input', 'provider', 'output')
PROVIDER_SCOPE = 'PROVIDER_CALL_EVENTS'
ADAPTER_SCOPE = 'SINGLE_KERNEL_CALL_EVENTS'


def summarize(data, *, provider_only=False):
    if (data.get('correctness') != 'PASS' or data.get('profiler') is not False or
            data.get('cuda_graph') is not False or data.get('annotations') is not None):
        raise ValueError('requires successful unprofiled ordinary event measurements')
    for key, want in (('zero_a', 'PASS'), ('omitted_producer', 'DETECTED'), ('changed_input', 'PASS')):
        if data.get(key) != want: raise ValueError('missing correctness control: '+key)
    if data.get('pointer_negatives') != ['BF16_LAST_COPY_ZERO']:
        raise ValueError('last weight-copy negative missing')
    numbers = data.get('numerical', [])
    if len(numbers) < 8 or any(r.get('nonfinite') != 0 or r.get('cells', 0) <= 0 or
            not math.isfinite(r.get('relative_l1_error', math.nan)) or
            not 0 <= r['relative_l1_error'] < .005 for r in numbers):
        raise ValueError('independent numerical coverage incomplete')
    weights = data.get('weights', [])
    if len(weights) != 2 or any(r.get('bad') != 0 or r.get('bits', 0) <= 0 for r in weights):
        raise ValueError('expanded BF16 weight proof missing')
    copies = data['receipt']['copies']
    count = data['samples_per_round']
    if (type(copies) is not int or copies < 2 or count != max(3, copies) or data.get('rounds') != 3 or
            data.get('warmup_calls', 0) < max(11, 2*copies) or data['warmup_calls'] % copies or
            data.get('event_warmup_calls') != 1 or data['fixture']['l2_bytes'] <= 0 or
            data['receipt']['active_ring_bytes'] < 2.25*data['fixture']['l2_bytes']):
        raise ValueError('warmup or rotating-weight coverage differs')
    provider = data['receipt']['provider']
    if (provider.get('provider') != 'CUBLAS_PPU_SDK' or provider.get('entry') != 'cublasGemmEx' or
            provider.get('compute') != 'CUBLAS_COMPUTE_32F' or provider.get('math_mode') != 16 or
            provider.get('output') != 'BF16_M_N' or not data.get('provider_loaded_images')):
        raise ValueError('cuBLAS precision/provider identity missing')
    measured = data['measured']
    expected = [(r, s, p, (r*count+s) % copies) for r in range(3) for s in range(count) for p in PHASES]
    if [(r.get('round'), r.get('sample'), r.get('phase'), r.get('copy')) for r in measured] != expected:
        raise ValueError('event sample inventory is incomplete, duplicate or reordered')
    frontend = data.get('submission_frontend', 'PYTHON_PER_CALL_SYNC')
    if frontend not in ('PYTHON_PER_CALL_SYNC', 'CPP_PREBOUND_CONTINUOUS'):
        raise ValueError('unknown event submission frontend')
    if frontend == 'CPP_PREBOUND_CONTINUOUS':
        if (data.get('per_sample_synchronizations') != 0 or data.get('inter_round_checks') != 0 or
                any(type(r.get('host_submit_us')) not in (int, float) or
                    not math.isfinite(r['host_submit_us']) or r['host_submit_us'] < 0 for r in measured)):
            raise ValueError('native continuous submission evidence incomplete')
    timers, missing = {}, []
    for phase in PHASES:
        samples = [r for r in measured if r['phase'] == phase]
        if any('event_span_us' not in r for r in samples):
            if provider_only and phase != 'provider' and all('event_span_us' not in r for r in samples):
                missing.append(phase); continue
            raise ValueError('missing event time for '+phase)
        rounds = [[r['event_span_us'] for r in samples if r['round'] == i] for i in range(3)]
        timer = statistics_checked(rounds)
        timer.update(scope=PROVIDER_SCOPE if phase == 'provider' else ADAPTER_SCOPE,
                     timing_method='CUDA_EVENTS_ORDINARY_CALL', first_use_excluded=True,
                     internal_reduction_included=phase == 'provider',
                     kernel_count='OPAQUE_PROVIDER' if phase == 'provider' else 1,
                     host_launch_cost_added=False, submission_frontend=frontend)
        if frontend == 'CPP_PREBOUND_CONTINUOUS':
            host = [[r['host_submit_us'] for r in samples if r['round'] == i] for i in range(3)]
            medians = [statistics.median(r) for r in host]
            timer['host_submission'] = dict(samples_us=host, round_medians_us=medians,
                median_us=statistics.median(medians), scope='CPU_SUBMISSION_DIAGNOSTIC_NOT_SUBTRACTED')
        timers[phase] = timer
    return dict(status='PASS', timers=timers, missing_adapter_timers=missing,
        scope='CUBLAS_PROVIDER_ONLY' if missing else 'SEPARATE_PROVIDER_AND_ADAPTER_API_EVENTS',
        original_event_scope=data.get('event_scope'), submission_frontend=frontend, cuda_graph=False, profiler=False,
        dequant_timed=False, production_changed=False, performance_admitted=False)


def review_existing(path):
    """No SDK/device access, no mutation and no re-labelling of old receipts."""
    root = Path(path).resolve(strict=True)
    if (root/'probe/authority.json').is_file(): root = root/'probe'
    files = {name: root/name for name in ('authority.json', 'events/result.json',
                                         'events.process.json', 'events.command.json')}
    values = {name: json.loads(p.read_text()) for name, p in files.items()}
    authority = values['authority.json']; result = values['events/result.json']
    process = values['events.process.json']; command = values['events.command.json']
    if (process.get('rc') != 0 or process.get('timeout') is not False or result.get('status') != 'PASS' or
            result.get('authority_sha256') != digest(authority)):
        raise ValueError('saved ordinary event worker failed or its authority differs')
    if not isinstance(command, list) or command[-2:] != ['--worker', 'events'] or 'profile' in command:
        raise ValueError('saved command was not the unprofiled event worker')
    data = summarize(result['data'], provider_only=True)
    data.update(reused_measurements=True, device_launches=0, original_authority=authority,
                evidence={n: dict(path=str(p), sha256=sha(p)) for n, p in files.items()})
    return data
