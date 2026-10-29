"""Scaffolding-only contract. No GEMM or production-selector performance claims."""
import ctypes as C
from dataclasses import asdict, dataclass
import hashlib
import json
import statistics
from collections import defaultdict

import numpy as np

from dev.moe_route.protocol import Call, Config, Sizes, Case as RouteCase, ids_for, oracle


class Value(C.Structure):
    _fields_ = [('data', C.c_void_p)] + [(n, C.c_int32) for n in
        ('dtype', 'order', 'state', 'partial_count')] + [(n, C.c_int64) for n in
        ('row_stride', 'partial_stride')] + [('round_type', C.c_int32), ('reserved', C.c_int32)]


class Prepare(C.Structure):
    _fields_ = [('version', C.c_uint32), ('size', C.c_uint32), ('route', Call), ('config', Config),
                ('columns', C.c_int32), ('dtype', C.c_int32), ('input_row_stride', C.c_int64)] + [
        (n, C.c_void_p) for n in ('input', 'output', 'inverse_rows', 'expert_rows', 'active_experts', 'active_count')]


class Transition(C.Structure):
    _fields_ = [('version', C.c_uint32), ('size', C.c_uint32)] + [(n, C.c_int32) for n in
        ('tokens', 'topk', 'hidden', 'merged')] + [('gate', Value), ('up', Value),
        ('output_dtype', C.c_int32), ('output_order', C.c_int32), ('output_stride', C.c_int64)] + [
        (n, C.c_void_p) for n in ('output', 'row_ids', 'inverse_rows', 'stream')]


class Finish(C.Structure):
    _fields_ = [('version', C.c_uint32), ('size', C.c_uint32)] + [(n, C.c_int32) for n in
        ('tokens', 'topk', 'columns', 'reserved')] + [('down', Value)] + [
        (n, C.c_void_p) for n in ('inverse_rows', 'weights', 'output')] + [
        (n, C.c_int64) for n in ('weights_stride', 'output_stride')] + [('stream', C.c_void_p)]


@dataclass(frozen=True)
class Case:
    tokens: int
    columns: int = 257
    hidden: int = 129
    output: int = 259
    experts: int = 256
    topk: int = 8
    dtype: int = 2
    merged: int = 1
    state: int = 0
    splits: int = 1
    down_splits: int = 1
    gate_order: int = 0
    up_order: int = 0
    output_order: int = 0
    down_order: int = 0
    profile: str = 'random'
    padding: int = 3
    projection_dtype: int = -1  # -1: use dtype; 0: final producer storage is F32.
    phase: str = 'all'

    @property
    def key(self):
        value=asdict(self)
        # Preserve the original mechanism-cohort identities, not its timings.
        for name,default in (('padding',3),('projection_dtype',-1),('phase','all')):
            if value[name]==default:value.pop(name)
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]

    @property
    def route_case(self): return RouteCase(self.tokens, self.experts, self.topk, 1, self.profile, padding=self.padding)


def encode(values, dtype):
    values = np.asarray(values, dtype=np.float32)
    if dtype == 0: return values.copy()
    if dtype == 1: return values.astype(np.float16)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7fff) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def decode(values, dtype):
    if dtype == 2: return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    return np.asarray(values).astype(np.float32)


def rounded(values, dtype): return decode(encode(values, dtype), dtype)


def projected(planes, dtype):
    result = np.zeros(planes.shape[1:], dtype=np.float32)
    for plane in planes: result = np.add(result, plane, dtype=np.float32)
    return rounded(result, dtype)


def activation(gate, up):
    return np.multiply(np.divide(gate, np.add(np.float32(1), np.exp(-gate), dtype=np.float32),
                                 dtype=np.float32), up, dtype=np.float32)


def weighted(down, weights):
    result = np.zeros((down.shape[0], down.shape[2]), dtype=np.float32)
    for slot in range(down.shape[1]):
        value = np.multiply(down[:, slot], weights[:, slot, None], dtype=np.float32)
        result = value if slot == 0 else np.add(result, value, dtype=np.float32)
    return result


def workloads(smoke=False):
    tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 32, 64, 65, 128, 257, 512, 1024, 2048, 4096, 5120]
    if smoke: tokens = [0, 1, 8, 9, 65, 257, 2048]
    cases = [Case(m, state=i % 3, splits=(1, 4, 1)[i % 3], down_splits=(1, 2, 4, 8)[i % 4],
                  merged=int(i % 2 == 0), gate_order=i % 2, up_order=(i+1) % 2,
                  output_order=(i//2) % 2, down_order=(i//3) % 2,
                  profile='cluster' if i % 2 else 'random') for i, m in enumerate(tokens)]
    cases += [Case(17, experts=33, topk=6, dtype=1, merged=0, state=1, splits=2, down_splits=2),
              Case(9, experts=1024, topk=32, dtype=2, state=1, splits=8),
              Case(1, experts=1, topk=1, dtype=1, state=2)]
    if not smoke:
        cases += [Case(8193, columns=17, hidden=19, output=23, state=1, splits=2),
                  Case(2048, columns=2048, hidden=512, output=2048),
                  Case(5120, columns=2048, hidden=512, output=2048, state=2)]
    cases += [Case(1, columns=2048, hidden=512, output=2048),
              Case(8, columns=2048, hidden=512, output=2048, state=1, splits=4, down_splits=8),
              Case(65, columns=33, hidden=31, output=37, dtype=0, topk=6, experts=33,
                   state=1, splits=4, down_splits=8, down_order=1)]
    return cases


def parse(capture, rows, apis):
    groups = defaultdict(list)
    for row in rows: groups[row['correlation_id']].append(row)
    calls = [a for a in apis if a['name'].startswith(('hggcGraphLaunch', 'cudaGraphLaunch'))]
    if len(calls) != len(capture['schedule']) or any(c['return_value'] for c in calls):
        raise ValueError('scaffolding graph launch coverage/status differs')
    samples = defaultdict(list); observed = defaultdict(int); missing = 0
    for item, call in zip(capture['schedule'], calls):
        block = groups[call['correlation_id']]
        if not 16 <= len(block) <= item['repeats']: raise ValueError('too few/extra measured activities')
        for row in block:
            if (row['kind'] != 'kernel' or item['kernel'] not in row['name'] or
                    row['grid'] != item['grid'] or row['block'] != item['block'] or row['end'] <= row['start']):
                raise ValueError('scaffolding activity/geometry differs: '+str(row))
        samples[item['component']].append(statistics.fmean((r['end']-r['start'])/1000 for r in block))
        observed[item['component']] += len(block); missing += item['repeats']-len(block)
    costs = {k: statistics.fmean(v) for k, v in samples.items()}
    points = []
    for point in capture['points']:
        arms = {arm: sum(costs[k] for k in keys) for arm, keys in point['components'].items()}
        points.append(dict(point, arms_us=arms,
                           delta_pct=100*(arms['fused']/arms['unfused']-1) if arms['unfused'] else None,
                           versus_fused_pct={k:100*(v/arms['fused']-1) for k,v in arms.items()} if arms['fused'] else {},
                           versus_unfused_pct={k:100*(v/arms['unfused']-1) for k,v in arms.items()} if arms['unfused'] else {}))
    return dict(status='PASS', points=points, component_us=costs, samples_us=dict(samples),
                observed_counts=dict(observed), missing_intervals=missing, imputed_intervals=0,
                device=capture['device'], baseline='SAME_MATH_UNFUSED_HELPERS_NOT_LLAMA_GEMM',
                timing_method='ASYS_SEPARATE_COMPONENT_MEANS_SUM', production_changed=False,
                no_regression_admitted=False, model_tpot_admitted=False)
