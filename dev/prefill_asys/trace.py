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
                device=r['deviceId'],context=r['contextId'],stream=r['streamId'],pid=r.get('globalPid'),
                correlation_id=r.get('correlationId')))
        for prefix in ('HGPTI','CUPTI'):
            mt = prefix+'_ACTIVITY_KIND_MEMSET'
            for row in db.execute('SELECT * FROM '+mt) if mt in tables else ():
                r = dict(row)
                result.append(dict(start=r['start'],end=r['end'],kind='memset',name='MEMSET',
                    value=r['value'],bytes=r['bytes'],device=r['deviceId'],context=r['contextId'],
                    stream=r['streamId'],pid=r.get('globalPid'),correlation_id=r.get('correlationId')))
            memcpy = prefix+'_ACTIVITY_KIND_MEMCPY'
            if memcpy in tables and db.execute('SELECT count(*) FROM '+memcpy).fetchone()[0]:
                raise ValueError('captured H2D/D2H: profiler range leaked setup or correctness work')
    if not result or any(r['end'] <= r['start'] for r in result): raise ValueError('empty/invalid GPU intervals')
    if len({(r['device'],r['context'],r['pid']) for r in result}) != 1:
        raise ValueError('trace includes more than one device/context/process')
    return sorted(result,key=lambda r:(r['start'],r['end']))


def runtime_activities(path):
    """API evidence can explain a discarded primer, never supply GPU timing."""
    with sqlite3.connect(Path(path).resolve(strict=True).as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        table = next((t for t in ('HGPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_RUNTIME') if t in tables),None)
        if table is None: return []
        required={'start','end','nameId','globalTid','correlationId','returnValue'}
        if not required<={r[1] for r in db.execute('PRAGMA table_info('+table+')')}: return []
        strings = dict(db.execute('SELECT id,value FROM StringIds'))
        return [dict(start=r['start'],end=r['end'],name=strings.get(r['nameId']),
                     tid=r['globalTid'],correlation_id=r['correlationId'],return_value=r['returnValue'])
                for r in db.execute('SELECT * FROM '+table+' ORDER BY start,end')]


def primer_reset_evidence(first_input, rows, apis):
    cid = first_input.get('correlation_id')
    launch = [r for r in apis if cid is not None and r['correlation_id']==cid]
    if (len(launch)!=1 or launch[0]['return_value']!=0 or
            launch[0]['name'] not in ('hggcLaunchKernel','cudaLaunchKernel')):
        raise ValueError('missing primer reset has no unique successful input launch API')
    launch = launch[0]
    prior = sorted((r for r in apis if r['tid']==launch['tid'] and r['end']<=launch['start']),
                   key=lambda r:(r['start'],r['end']))
    if (len(prior)<2 or prior[-2]['name'] not in ('hggcProfilerStart','cudaProfilerStart') or
            prior[-1]['name'] not in ('hggcMemsetAsync','cudaMemsetAsync') or
            any(r['return_value']!=0 for r in prior[-2:])):
        raise ValueError('missing primer reset is not the first successful API after capture start')
    reset = prior[-1]
    if (sum(r['correlation_id']==reset['correlation_id'] for r in apis)!=1 or
            any(r.get('correlation_id')==reset['correlation_id'] for r in rows)):
        raise ValueError('primer reset API correlation is ambiguous or already has GPU activity')
    return dict(call_index=0,reason='MISSING_FIRST_PRIMER_RESET_ACTIVITY',
                reset_api=reset['name'],reset_correlation_id=reset['correlation_id'],
                input_correlation_id=cid,api_return_value=0,timing_used=False)


def helper_role(row, arm):
    if row['kind'] == 'memset': return None
    n = row['name'].replace(' ', '')
    if arm != 'native-fp16':
        typed=re.search(r'dense_convert_(?:vec8|stripe4|full4|scalar|cont)<(?:true|false),(true|false)(?:,|>)',n)
        if typed:return 'input' if typed[1]=='true' else 'output'
        if re.search(r'dense_dequant_(?:direct|packed|unsigned|pair|cooperative|q8|q8_exchange|q8_tuned|q8_half2|q8_paired|plane_pair)<',n):return 'expansion'
    if arm == 'native-fp16':
        if 'dequantize_block_' in n: return 'expansion'
        if ('convert_cont_ppu_kernel<float,__half>' in n or
                'fp32_to_fp16' in n or 'convert_f32_to_f16' in n): return 'input'
        if ('convert_cont_ppu_kernel<__half,float>' in n or
                'fp16_to_fp32' in n or 'convert_f16_to_f32' in n): return 'output'
    elif arm in ('full-bf16','full-fp16','full-bf16-typed'):
        if '::gather_bf16(' in n: return 'input'
        if '::finish_bf16(' in n: return 'output'
        if re.search(r'\bfull_(direct|wide|transpose|shared|packed_exchange)<',n): return 'expansion'
    else:
        if 'convert_rows<true>' in n or 'convert_flat<true>' in n: return 'input'
        if 'convert_rows<false>' in n or 'convert_flat<false>' in n: return 'output'
        if any(s in n for s in ('sf_unit16<','sf_columns<','prepass_unit_kernel<','sf_bfloat_units<')):
            return 'expansion'
    return None


def signature(row):
    return {k:v for k,v in row.items() if k not in
            ('start','end','stream','pid','device','context','phase','correlation_id')}


def one_call(rows, arm, stages, split, *, primer_without_reset=False):
    helpers = [(i,helper_role(r,arm)) for i,r in enumerate(rows) if helper_role(r,arm)]
    expected = [s for s in stages if s not in ('main','error_reset')]
    if [r for _,r in helpers] != expected or helpers[-1][0] != len(rows)-1:
        raise ValueError('GPU call helper inventory/order differs: '+repr(helpers))
    for row in rows:
        name=row['name'].replace(' ','')
        cast=re.search(r'dense_convert_(?:vec8|stripe4|full4|scalar|cont)<(true|false),',name)
        dq=re.search(r'dense_dequant_(?:direct|packed|unsigned|pair|cooperative|plane_pair)<[^,]+,(true|false)',name)
        if dq is None: dq=re.search(r'dense_dequant_q8(?:_exchange|_tuned)?<(true|false)(?:,|>)',name)
        if re.search(r'dense_dequant_q8_(half2|paired)<',name) and arm!='full-fp16':
            raise ValueError('half2/paired dequant is F16-only: '+row['name'])
        for match in (cast,dq):
            if match and (match[1]=='true') != (arm in ('full-bf16','full-bf16-typed')):
                raise ValueError('typed helper precision differs from arm: '+row['name'])
    first_main = helpers[-2][0]+1
    last_main = helpers[-1][0]
    if last_main <= first_main: raise ValueError('missing GPU main kernel')
    phases = {i:role for i,role in helpers}
    if arm in ('full-bf16','full-fp16','full-bf16-typed'):
        if primer_without_reset:
            if helpers[0] != (0,expected[0]) or expected[0] not in ('input','expansion') or 'error_reset' not in stages:
                raise ValueError('invalid incomplete primer inventory')
        elif helpers[0][0] != 1 or not (rows[0]['kind'] == 'memset' and
                rows[0]['bytes'] == 4 and rows[0]['value'] == 0):
            raise ValueError('full-dequant reset activity missing or differs')
        else: phases[0] = 'error_reset'
    elif helpers[0][0] != 0: raise ValueError('unaccounted work before input/expansion')
    for i in range(first_main,last_main):
        k = kernel_role(rows[i]['name'])
        phases[i] = 'reducer' if k in ('reducer','reducer_finish') else 'main'
    if set(phases) != set(range(len(rows))): raise ValueError('unaccounted GPU activity within call')
    if split > 1 and 'reducer' not in phases.values(): raise ValueError('Split-K has no reducer activity')
    for i,r in enumerate(rows):
        r['phase'] = phases[i]
    if not any(r['kind'] == 'kernel' and phases[i] == 'main' for i,r in enumerate(rows)):
        raise ValueError('provider has no GPU compute')
    # The exception is only an inventory check. Do not synthesize a reset or
    # even return a partial duration for this already-discarded call.
    if primer_without_reset: return None
    times = defaultdict(float)
    for i,r in enumerate(rows): times[phases[i]] += (r['end']-r['start'])/1000
    cursor, union = rows[0]['start'], 0
    for r in rows:
        union += max(0,r['end']-max(cursor,r['start']))
        cursor = max(cursor,r['end'])
    span = (max(r['end'] for r in rows)-rows[0]['start'])/1000
    total = sum(times.values())
    return dict(phases_us=dict(times),gpu_sum_us=total,gpu_span_us=span,
                idle_gap_us=span-union/1000,overlap_us=total-union/1000,
                fingerprint=digest([(r['phase'],signature(r)) for r in rows]))


def parse_rows(rows, cases, *, apis=()):
    """Each API capture is a fixed number of complete calls. Ordered JSON
    receipts bind captures to configs; no Python clocks or CPU ranges used.
    """
    offset, result = 0, []
    for case in cases:
        arm, contract = case['arm'], case['trace']
        calls, raw, exceptions = [], [], []
        cut = contract['discarded_calls']
        if cut<=0 or contract['measured_calls']!=3*cut or contract['captured_calls']!=4*cut:
            raise ValueError('incorrect primer/sample count')
        parameters = dict(case.get('recipe',{}).get('parameters',[]))
        split = parameters.get('split',1)
        for call in range(contract['captured_calls']):
            start = offset
            while offset < len(rows) and helper_role(rows[offset],arm) != 'output': offset += 1
            if offset == len(rows): raise ValueError('Asys complete-call coverage missing: '+case['case_id'])
            offset += 1
            block = [dict(r) for r in rows[start:offset]]
            missing = (call==0 and arm in ('full-bf16','full-fp16','full-bf16-typed') and
                       helper_role(block[0],arm) in ('input','expansion'))
            if missing: exceptions.append(primer_reset_evidence(block[0],rows,apis))
            try:
                calls.append(one_call(block,arm,case['stages'],split,primer_without_reset=missing))
            except ValueError as exc:
                # Keep the strict completeness check. A coordinator's compact
                # failure must identify the exact call, not just an output
                # helper with no way to tell primer loss from missing samples.
                names=[dict(name=r['name'][:240],correlation_id=r.get('correlation_id')) for r in block]
                raise ValueError(f'{exc}; case={case["case_id"]} arm={arm} call={call} '
                                 f'discarded_calls={cut} activities={names!r}') from exc
            if parameters.get('dense_reducer'):
                elements=parameters['dense_reducer']
                w=case['workload'];reducers=[r for r in block if r['phase']=='reducer']
                name=f'PpuMixedInputSplitKParallelM1FastReductionKernel<{elements},{split}>'
                if (elements not in (2,4,8) or w['m']*w['n']%(32*elements) or split not in (2,4,8) or len(reducers)!=1 or
                        name not in reducers[0]['name'].replace(' ','') or
                        reducers[0]['grid']!=[w['m']*w['n']//(32*elements),1,1] or
                        reducers[0]['block']!=[32,1,1] or reducers[0]['shared_bytes']!=0):
                    raise ValueError('compact reducer did not execute the required native body/launch')
            raw.append(block)
        kept = calls[cut:]
        if len(kept) != contract['measured_calls'] or len(kept) < 3:
            raise ValueError('incorrect primer/sample count')
        if len({r['fingerprint'] for r in kept}) != 1:
            raise ValueError('post-warmup GPU kernel/launch inventory changed; no config timing admitted')
        if exceptions and [signature(r) for r in raw[0]] != [signature(r) for r in raw[cut][1:]]:
            raise ValueError('incomplete primer kernel inventory differs from measured calls')
        phases = sorted(kept[0]['phases_us'])
        kernels = []
        for i,node in enumerate(raw[cut]):
            samples = [(block[i]['end']-block[i]['start'])/1000 for block in raw[cut:]]
            kernels.append(dict(signature(node),phase=node['phase'],samples_us=samples,
                                median_us=statistics.median(samples)))
        row = dict(case_id=case['case_id'],arm=arm,status='PASS',calls=len(kept),
            helper_policy=case.get('helper_policy'),helper_mode=case.get('helper_mode'),
            arithmetic=case.get('receipt',{}).get('arithmetic'),
            kernel_inventory=kernels,samples=kept,
            phases_us={p:statistics.median(r['phases_us'][p] for r in kept) for p in phases},
            **{key:statistics.median(r[key] for r in kept)
               for key in ('gpu_sum_us','gpu_span_us','idle_gap_us','overlap_us')},
            measured_rounds_us=[statistics.median(r['gpu_sum_us'] for r in kept[j*cut:(j+1)*cut]) for j in range(3)],
            scope='GPU_ACTIVITIES_INSIDE_COMPLETE_CALL_NOT_API_INTERVAL_NOT_MODEL_LATENCY',
            timing_method='ASYS_GPU_ACTIVITY',cuda_graph=False,first_ring_excluded=True)
        if exceptions: row['primer_exceptions'] = exceptions
        result.append(row)
    if offset != len(rows): raise ValueError(f'unassigned GPU activities: {len(rows)-offset}')
    return result


def parse(path, cases):
    for case in cases:
        if case.get('status') != 'PASS':raise ValueError('numerical case did not pass')
        proof=case.get('correctness')
        if proof != 'PASS' and (not isinstance(proof,list) or not proof or any(
                p.get('nonfinite') != 0 or not 0<=p.get('relative_l1_error',1)<.005 for p in proof)):
            raise ValueError('case numerical proof missing or failed')
    result = parse_rows(activities(path),cases,apis=runtime_activities(path))
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
