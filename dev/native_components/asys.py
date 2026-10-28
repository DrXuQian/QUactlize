"""Bounded native baseline, using the same per-record GPU clock as candidates."""
from dataclasses import asdict
import math
import statistics

from dev.strategy.asys_components import parse_component_rows
from dev.strategy.smallm_asys import METHOD
from dev.prefill_asys.trace import activities
from quactlize.dispatch.table import Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

SCHEMA = 'quactlize.native-smallm-asys.v1'
PROTOCOL = dict(rounds=6, samples=5, order='FORWARD_REVERSE_ALTERNATING')


def plan(qtypes=(13, 14), n=1024, k=5120):
    if not qtypes or len(set(qtypes)) != len(qtypes) or any(q not in (8,10,11,12,13,14) for q in qtypes):
        raise ValueError('qtypes must be unique GGML IDs from 8,10,11,12,13,14')
    if n <= 0 or k <= 0 or n % 128 or k % 256:
        raise ValueError('bounded baseline requires positive N/128 and K/256')
    points = []
    for q in qtypes:
        for m in range(1,9):
            w = Workload(q,n,k,m)
            points.append(dict(id=w.key, workload=asdict(w), cohort='decode'))
    result = dict(schema=SCHEMA, points=points, qtypes=list(qtypes), n=n, k=k, protocol=PROTOCOL,
        timing_method=METHOD, producer_cache='ROTATING_WEIGHT_RING_AT_LEAST_2P25_L2',
        arithmetic='NATIVE_UNMODIFIED; Q8_1_ACTIVATION_QUANTIZATION_REPORTED_IF_SELECTED',
        selection_metric='ARITHMETIC_MEAN_OF_SEPARATE_COMPONENT_SUMS',
        scope='NATIVE_SMALLM_BASELINE_NOT_PAIRED_AB_OR_MODEL_TPOT', production_changed=False)
    result['sha256'] = digest(result)
    return result


def validate_capture(data, point, authority):
    if (data.get('status') != 'CAPTURED_NOT_TIMED' or data.get('timing_method') != METHOD or
            data.get('id') != point['id'] or data.get('workload') != point['workload'] or
            data.get('authority') != authority or data.get('protocol') != PROTOCOL):
        raise ValueError('native capture method/point/authority/protocol differs')
    if (data.get('correctness') != 'PASS' or data.get('linked_backend_comparison') != 'PASS_TWO_INPUTS' or
            data.get('omitted_producer') != 'DETECTED' or
            data.get('last_weight_copy_negative') != 'DETECTED_COPY0_UNCHANGED'):
        raise ValueError('native extraction/numerical controls incomplete')
    ring = data['ring']
    if (ring['copies'] < 2 or ring['l2_bytes'] <= 0 or
            ring['active_read_bytes'] != ring['copies']*ring['minimum_read_bytes'] or
            ring['active_read_bytes'] < 2.25*ring['l2_bytes']):
        raise ValueError('native weight ring is not cold')
    proofs = data.get('proof', [])
    if len(proofs) != 3*ring['copies']+4 or {p['copy'] for p in proofs} != set(range(ring['copies'])):
        raise ValueError('native per-copy numerical coverage differs')
    w = point['workload']
    expected_cells = w['m']*w['n']*(w['topk'] if w['operation'] == 'indexed' else 1)
    quantized = data.get('native_activation_contract') == 'Q8_1_QUANTIZED_FROM_F32'
    for p in proofs:
        if (p.get('status') != 'PASS' or p.get('finite_cells_per_arm') != expected_cells or
                p.get('spot_cells') != 64 or p.get('oracle_activation') != ('NATIVE_Q8_1' if quantized else 'ORIGINAL_F32')):
            raise ValueError('native numerical oracle coverage/contract differs')
        error = p['arithmetic_spot_normalized_error'] if quantized else max(p['raw_spot_normalized_error'])
        if not isinstance(error,(int,float)) or not math.isfinite(error) or not 0 <= error < (.00005 if quantized else .005):
            raise ValueError('native numerical oracle failed')
    components = data['components']
    if (not components or [c['stage'] for c in components] != list(range(len(components))) or
            sum(c['role']=='producer' for c in components) != 1):
        raise ValueError('native stage coverage differs')
    for c in components:
        if (c['entry'] != 'KERNEL' or c['timing'] != METHOD or c['identity'] != authority or
                c['role'] not in ('producer','adapter','expansion','reducer','prepare','postprocess') or
                'quactlize' in c['name'].lower() or c['correctness'] != 'PASS' or
                c['timed_kernel_types'] != 1 or not c['first_use_excluded'] or c['host_launch_cost_added']):
            raise ValueError('native isolated stage contract differs')
        rotate = c['role'] in ('producer','expansion')
        copies = ring['copies'] if rotate else 1
        if c['calls_per_graph'] != copies*math.ceil(16/copies):
            raise ValueError('native stage does not traverse the declared ring')


def parse_rows(data, point, authority, rows, provenance):
    validate_capture(data,point,authority)
    records = {str(c['stage']):dict(identity=dict(node={k:c[k] for k in
        ('symbol','grid','block','shared_bytes')}, native=authority),
        calls_per_graph=c['calls_per_graph']) for c in data['components']}
    schedule = [dict(component=str(s['stage']), **{k:s[k] for k in ('round','sample','kernels')})
                for s in data['trace_schedule']]
    records = parse_component_rows(records,schedule,rows,provenance)
    components = [dict(c, **{k:v for k,v in records[str(c['stage'])].items() if k != 'identity'})
                  for c in data['components']]
    costs = [[sum(c['samples_us'][r][s] for c in components) for s in range(5)] for r in range(6)]
    flat = [x for r in costs for x in r]
    return dict(data, status='PARSED_NOT_ADMITTED', components=components, sum_samples_us=costs,
        sum_mean_us=statistics.fmean(flat), sum_min_us=min(flat), sum_max_us=max(flat),
        latency_provenance=provenance, interval_policy='PER_RECORD_END_MINUS_START_NO_ADJACENCY_REQUIREMENT',
        no_regression_admitted=False, model_tpot_admitted=False)


def parse(data, point, authority, database, report, provenance):
    return parse_rows(data,point,authority,activities(database,launch_shared=True),
        dict(provenance,timing_method=METHOD,sqlite_sha256=sha(database),trace_sha256=sha(report)))
