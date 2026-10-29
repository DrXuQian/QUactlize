"""Integer routing contract, bounded cases and Asys-only component accounting."""
import ctypes as C
from dataclasses import asdict, dataclass
from collections import defaultdict
import statistics

from quactlize.runtime.tuning import digest


class Call(C.Structure):
    _fields_ = [('version', C.c_uint32), ('size', C.c_uint32)] + [
        (n, C.c_int32) for n in ('tokens', 'experts', 'topk', 'channels', 'inverse', 'reserved')] + [
        ('ids_stride', C.c_int64), ('source_stride', C.c_int64)] + [
        (n, C.c_void_p) for n in ('ids', 'src_rows', 'dst_rows', 'offsets', 'workspace')] + [
        ('workspace_bytes', C.c_uint64), ('stream', C.c_void_p)]


class Config(C.Structure):
    _fields_ = [('version', C.c_uint32), ('size', C.c_uint32), ('algorithm', C.c_int32), ('tile_rows', C.c_int32)]


class Sizes(C.Structure):
    _fields_ = [('workspace_bytes', C.c_uint64)] + [(n, C.c_int32) for n in ('algorithm', 'tile_rows', 'tiles', 'stages')]


@dataclass(frozen=True)
class Case:
    tokens: int
    experts: int = 256
    topk: int = 8
    channels: int = 1
    profile: str = 'random'
    padding: int = 0
    inverse: int = 0

    @property
    def key(self): return digest(asdict(self))[:16]


def workloads(smoke=False):
    edge = [Case(0), Case(17, 33, 6, 3, padding=5, inverse=1),
            Case(513, 1, 1, 1, padding=3), Case(4097, 1024, 32, 32),
            Case(5121), Case(32769, profile='cluster'), Case(65536, 64, 4, 4),
            Case(31, 17, 5, 2, padding=3, inverse=1), Case(33, channels=8), Case(65, channels=8),
            Case(1,1,1,1), Case(512,1,1,1), Case(256,64,2,2),
            Case(128,64,4,4), Case(32,33,16,16), Case(16,64,32,32)]
    if smoke:
        return [Case(m, channels=8) for m in (1, 4, 8, 16, 32, 64, 129, 2048, 5120)] + edge
    return [Case(m, channels=c, profile=p) for m in
            (*range(1, 9), 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 5120)
            for p in ('spread', 'cluster', 'repeat2', 'repeat4', 'random') for c in (1, 8)] + edge


def ids_for(case, seed):
    import numpy as np
    m, e, k = case.tokens, case.experts, case.topk
    rng = np.random.default_rng(4517 + seed)
    if case.profile == 'random':
        # Independent rows, unique expert slots. CPU fixture only, outside profiling.
        ids = np.empty((m, k), dtype=np.int32)
        for i in range(m): ids[i] = rng.choice(e, k, replace=False)
    else:
        groups = {'spread': max(1, e//k), 'cluster': 1, 'repeat2': 2, 'repeat4': 4}[case.profile]
        tokens = np.arange(m, dtype=np.int64)
        ids = ((tokens[:, None] % groups)*k + np.arange(k) + seed*7) % e
    storage = np.full((m, k+case.padding), -777, dtype=np.int32)
    storage[:, :k] = ids
    return storage


def oracle(ids, case):
    import numpy as np
    flat = ids[:, :case.topk].reshape(-1)
    order = np.argsort(flat, kind='stable').astype(np.int32)
    offsets = np.r_[0, np.cumsum(np.bincount(flat, minlength=case.experts))].astype(np.int32)
    if case.inverse:
        source = np.empty_like(order); source[order] = np.arange(len(order), dtype=np.int32)
    else:
        source = ((order//case.topk)*(case.channels+case.padding) +
                  (order % case.topk) % case.channels).astype(np.int32)
    return source, order, offsets


def tiled_model(ids, case, tile):
    """Independent host simulation of the tiled ownership/prefix decomposition."""
    import numpy as np
    flat = ids[:, :case.topk].reshape(-1); n = len(flat)
    chunks = (n+tile-1)//tile
    hist = np.zeros((chunks, case.experts), dtype=np.int64)
    ranks = np.empty(n, dtype=np.int64)
    for chunk in range(chunks):
        for row in range(chunk*tile, min(n, (chunk+1)*tile)):
            expert = flat[row]; ranks[row] = hist[chunk, expert]; hist[chunk, expert] += 1
    prefix = np.cumsum(hist, axis=0)-hist
    counts = hist.sum(axis=0); offsets = np.r_[0, np.cumsum(counts)].astype(np.int32)
    compact = offsets[flat] + prefix[np.arange(n)//tile, flat] + ranks
    order = np.empty(n, dtype=np.int32); order[compact] = np.arange(n, dtype=np.int32)
    source = compact.astype(np.int32) if case.inverse else (
        (order//case.topk)*(case.channels+case.padding)+(order % case.topk) % case.channels).astype(np.int32)
    return source, order, offsets


def register_model(ids, case, ballot=False):
    """Model warp ballots/lane-prefix ranks; no shared selected-list."""
    import numpy as np
    flat = ids[:, :case.topk].reshape(-1); n = len(flat)
    limit=512 if ballot else 256
    if not 0 < n <= limit: raise ValueError(f'register candidate scope is 1..{limit} routed rows')
    source, order = np.empty(n, dtype=np.int32), np.empty(n, dtype=np.int32)
    offsets = np.empty(case.experts+1, dtype=np.int32)
    for e in range(case.experts):
        lower = [0]*32; masks = []; ballot_begin=0
        for g in range((n+31)//32):
            hit = 0; lower_mask=0
            for lane in range(32):
                r = g*32+lane
                if r<n:
                    lower[lane] += int(flat[r]<e)
                    if flat[r]<e: lower_mask |= 1<<lane
                    if flat[r]==e: hit |= 1<<lane
            masks.append(hit)
            ballot_begin += lower_mask.bit_count()
        begin, count = ballot_begin if ballot else sum(lower), 0
        for g, hit in enumerate(masks):
            for lane in range(32):
                if hit&(1<<lane):
                    r = g*32+lane; to = begin+count+(hit&((1<<lane)-1)).bit_count()
                    order[to] = r
                    if case.inverse: source[r] = to
                    else: source[to] = (r//case.topk)*(case.channels+case.padding)+(r%case.topk)%case.channels
            count += hit.bit_count()
        offsets[e] = begin
        if e==case.experts-1: offsets[e+1] = begin+count
    return source, order, offsets


def parse(capture, rows, apis):
    calls = [a for a in apis if a['name'] in ('hggcGraphLaunch', 'cudaGraphLaunch')]
    schedule = capture['schedule']; groups = defaultdict(list)
    for row in rows: groups[row.get('correlation_id')].append(row)
    if (len(calls) != len(schedule) or any(a['return_value'] for a in calls) or
            len({a['correlation_id'] for a in calls}) != len(calls) or
            set(groups) != {a['correlation_id'] for a in calls}):
        raise ValueError('Asys graph calls do not cover the measured schedule')
    samples = defaultdict(list); observed = defaultdict(int); missing = 0
    for item, launch in zip(schedule, calls):
        block = groups[launch['correlation_id']]
        if not 16 <= len(block) <= item['repeats']:
            raise ValueError('Asys sample has too few/extra observed GPU activities')
        for row in block:
            if (row['kind'] != 'kernel' or item['kernel'] not in row['name'] or
                    row['grid'] != item['grid'] or row['block'] != item['block'] or row['end'] <= row['start']):
                raise ValueError('Asys kernel or launch geometry differs from the requested component')
        samples[item['component']].append(statistics.fmean((r['end']-r['start'])/1000 for r in block))
        observed[item['component']] += len(block)
        missing += item['repeats']-len(block)
    costs = {key: statistics.fmean(v) for key, v in samples.items()}
    result = []
    for p in capture['points']:
        arms = {name: sum(costs[k] for k in keys) for name, keys in p['components'].items()}
        native = arms.pop('native', None)
        best = min(arms, key=arms.get)
        result.append(dict(case=p['case'], key=p['key'], arms_us=arms, native_us=native,
            best=best, best_us=arms[best], delta_pct=100*(arms[best]/native-1) if native else None,
            auto=p['auto'], auto_us=arms[p['auto']],
            auto_delta_pct=100*(arms[p['auto']]/native-1) if native else None,
            no_regression=arms[best] <= native if native else None,
            native_status=p['native_status'], correctness='PASS'))
    return dict(status='PASS', points=result, component_us=costs, samples_us=dict(samples),
                observed_counts=dict(observed), missing_intervals=missing, imputed_intervals=0,
                timing_method='ASYS_GPU_ACTIVITY_MEAN_SEPARATE_COMPONENT_SUM',
                device=capture['device'], production_changed=False)
