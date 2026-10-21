"""Read actual GPU activities. No HGTX-range guesses or host/event fallback."""
from collections import defaultdict
from pathlib import Path
import re
import sqlite3
import statistics

from dev.strategy.components import role as kernel_role
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest


def activities(path):
    path = Path(path).resolve(strict=True)
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory = sqlite3.Row
        if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok': raise ValueError('invalid Asys SQLite')
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        kt = next((t for t in ('HGPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_KERNEL') if t in tables),None)
        if kt is None: raise ValueError('Asys has no GPU kernel activity; no host-timing fallback')
        strings = dict(db.execute('SELECT id,value FROM StringIds'))
        result = []
        for row in db.execute('SELECT * FROM '+kt):
            r = dict(row)
            name = strings.get(r['demangledName'])
            if not name: raise ValueError('unresolved GPU kernel name')
            result.append(dict(start=r['start'],end=r['end'],kind='kernel',name=name,
                symbol=strings.get(r.get('mangledName')),grid=[r['grid'+d] for d in 'XYZ'],
                block=[r['block'+d] for d in 'XYZ'],shared_bytes=r['staticSharedMemory']+r['dynamicSharedMemory'],
                device=r['deviceId'],context=r['contextId'],stream=r['streamId'],pid=r.get('globalPid')))
        for prefix in ('HGPTI','CUPTI'):
            mt = prefix+'_ACTIVITY_KIND_MEMSET'
            for row in db.execute('SELECT * FROM '+mt) if mt in tables else ():
                r = dict(row)
                result.append(dict(start=r['start'],end=r['end'],kind='memset',name='MEMSET',
                    value=r['value'],bytes=r['bytes'],device=r['deviceId'],context=r['contextId'],
                    stream=r['streamId'],pid=r.get('globalPid')))
            memcpy = prefix+'_ACTIVITY_KIND_MEMCPY'
            if memcpy in tables and db.execute('SELECT count(*) FROM '+memcpy).fetchone()[0]:
                raise ValueError('captured H2D/D2H: profiler range leaked setup or correctness work')
    if not result or any(r['end'] <= r['start'] for r in result): raise ValueError('empty/invalid GPU intervals')
    if len({(r['device'],r['context'],r['pid']) for r in result}) != 1:
        raise ValueError('trace includes more than one device/context/process')
    return sorted(result,key=lambda r:(r['start'],r['end']))


def helper_role(row, arm):
    if row['kind'] == 'memset': return None
    n = row['name'].replace(' ', '')
    if arm == 'native-fp16':
        if 'dequantize_block_' in n: return 'expansion'
        if ('convert_cont_ppu_kernel<float,__half>' in n or
                'fp32_to_fp16' in n or 'convert_f32_to_f16' in n): return 'input'
        if ('convert_cont_ppu_kernel<__half,float>' in n or
                'fp16_to_fp32' in n or 'convert_f16_to_f32' in n): return 'output'
    elif arm == 'full-bf16':
        if '::gather_bf16(' in n: return 'input'
        if '::finish_bf16(' in n: return 'output'
        if re.search(r'\bfull_(direct|wide|transpose|shared|packed_exchange)<',n): return 'expansion'
    else:
        if 'convert_rows<true>' in n: return 'input'
        if 'convert_rows<false>' in n: return 'output'
        if any(s in n for s in ('sf_unit16<','sf_columns<','prepass_unit_kernel<','sf_bfloat_units<')):
            return 'expansion'
    return None


def signature(row):
    return {k:v for k,v in row.items() if k not in ('start','end','stream','pid','device','context','phase')}


def one_call(rows, arm, stages, split):
    helpers = [(i,helper_role(r,arm)) for i,r in enumerate(rows) if helper_role(r,arm)]
    expected = [s for s in stages if s not in ('main','error_reset')]
    if [r for _,r in helpers] != expected or helpers[-1][0] != len(rows)-1:
        raise ValueError('GPU call helper inventory/order differs: '+repr(helpers))
    first_main = helpers[-2][0]+1
    last_main = helpers[-1][0]
    if last_main <= first_main: raise ValueError('missing GPU main kernel')
    phases = {i:role for i,role in helpers}
    if arm == 'full-bf16':
        if helpers[0][0] != 1 or not (rows[0]['kind'] == 'memset' and
                rows[0]['bytes'] == 4 and rows[0]['value'] == 0):
            raise ValueError('full BF16 reset activity missing or differs')
        phases[0] = 'error_reset'
    elif helpers[0][0] != 0: raise ValueError('unaccounted work before input/expansion')
    for i in range(first_main,last_main):
        k = kernel_role(rows[i]['name'])
        phases[i] = 'reducer' if k in ('reducer','reducer_finish') else 'main'
    if set(phases) != set(range(len(rows))): raise ValueError('unaccounted GPU activity within call')
    if split > 1 and 'reducer' not in phases.values(): raise ValueError('Split-K has no reducer activity')
    times = defaultdict(float)
    for i,r in enumerate(rows):
        r['phase'] = phases[i]
        times[phases[i]] += (r['end']-r['start'])/1000
    if not any(r['kind'] == 'kernel' and phases[i] == 'main' for i,r in enumerate(rows)):
        raise ValueError('provider has no GPU compute')
    cursor, union = rows[0]['start'], 0
    for r in rows:
        union += max(0,r['end']-max(cursor,r['start']))
        cursor = max(cursor,r['end'])
    span = (max(r['end'] for r in rows)-rows[0]['start'])/1000
    total = sum(times.values())
    return dict(phases_us=dict(times),gpu_sum_us=total,gpu_span_us=span,
                idle_gap_us=span-union/1000,overlap_us=total-union/1000,
                fingerprint=digest([(r['phase'],signature(r)) for r in rows]))


def parse_rows(rows, cases):
    """Each API capture is a fixed number of complete calls. Ordered JSON
    receipts bind captures to configs; no Python clocks or CPU ranges used.
    """
    offset, result = 0, []
    for case in cases:
        arm, contract = case['arm'], case['trace']
        calls, raw = [], []
        split = dict(case.get('recipe',{}).get('parameters',[])).get('split',1)
        for _ in range(contract['captured_calls']):
            start = offset
            while offset < len(rows) and helper_role(rows[offset],arm) != 'output': offset += 1
            if offset == len(rows): raise ValueError('Asys complete-call coverage missing: '+case['case_id'])
            offset += 1
            block = [dict(r) for r in rows[start:offset]]
            calls.append(one_call(block,arm,case['stages'],split)); raw.append(block)
        cut = contract['discarded_calls']; kept = calls[cut:]
        if len(kept) != contract['measured_calls'] or len(kept) < 3:
            raise ValueError('incorrect primer/sample count')
        if len({r['fingerprint'] for r in kept}) != 1:
            raise ValueError('post-warmup GPU kernel/launch inventory changed; no config timing admitted')
        phases = sorted(kept[0]['phases_us'])
        kernels = []
        for i,node in enumerate(raw[cut]):
            samples = [(block[i]['end']-block[i]['start'])/1000 for block in raw[cut:]]
            kernels.append(dict(signature(node),phase=node['phase'],samples_us=samples,
                                median_us=statistics.median(samples)))
        row = dict(case_id=case['case_id'],arm=arm,status='PASS',calls=len(kept),
            kernel_inventory=kernels,samples=kept,
            phases_us={p:statistics.median(r['phases_us'][p] for r in kept) for p in phases},
            **{key:statistics.median(r[key] for r in kept)
               for key in ('gpu_sum_us','gpu_span_us','idle_gap_us','overlap_us')},
            measured_rounds_us=[statistics.median(r['gpu_sum_us'] for r in kept[j*cut:(j+1)*cut]) for j in range(3)],
            scope='GPU_ACTIVITIES_INSIDE_COMPLETE_CALL_NOT_API_INTERVAL_NOT_MODEL_LATENCY',
            timing_method='ASYS_GPU_ACTIVITY',cuda_graph=False,first_ring_excluded=True)
        result.append(row)
    if offset != len(rows): raise ValueError(f'unassigned GPU activities: {len(rows)-offset}')
    return result


def parse(path, cases):
    for case in cases:
        if case.get('status') != 'PASS':raise ValueError('numerical case did not pass')
        proof=case.get('correctness')
        if proof != 'PASS' and (not isinstance(proof,list) or not proof or any(
                p.get('nonfinite') != 0 or p.get('relative_l1_error',1) >= .005 for p in proof)):
            raise ValueError('case numerical proof missing or failed')
    result = parse_rows(activities(path),cases)
    return dict(status='PASS',cases=result,sqlite_sha256=sha(path),production_changed=False,
                performance_admitted=False)


if __name__=='__main__':
    import argparse
    import json
    from dev.strategy.journal import atomic_json
    p=argparse.ArgumentParser(description='Re-read saved Asys evidence; no GPU or new timings')
    p.add_argument('sqlite',type=Path);p.add_argument('cases',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('output must be new; original receipts are immutable')
    atomic_json(a.output,parse(a.sqlite,json.loads(a.cases.read_text())))
    print('PREFILL_ASYS_REVIEW '+str(a.output))
