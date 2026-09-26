"""Validate an entire steady-clock cohort and export the measured comparison."""
import argparse
import csv
import json
from pathlib import Path

from compare import sha, validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    cohort=json.loads((a.cohort/'summary.json').read_text())
    if (not cohort['complete'] or len(cohort['results'])!=6 or
            cohort.get('contract')!='GPTQ_SYM_INT4_G128_F16_V1'):
        raise ValueError('complete six-point GPTQ cohort required; GGUF-subset results are not admissible')
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError('use a fresh report directory')
    seen=set()
    all_rows=[]
    best=[]
    authorities=set()
    for point in cohort['results']:
        dtype,m,n,k=point['point']
        key=(dtype,m,n,k)
        if key in seen: raise ValueError('duplicate shape')
        seen.add(key)
        path=a.cohort/f'd{dtype}-m{m}-n{n}-k{k}'/'summary.json'
        receipt=json.loads(path.read_text())
        if receipt['status']!='PASS': raise ValueError('unadmitted point')
        data=receipt['result'][0]
        validate(data)
        if data['shape']!=[m,n,k] or data['dtype']!=dtype or data['cache']!='rotating':
            raise ValueError('point identity differs')
        authorities.add((receipt['reference_manifest_sha256'],receipt['importer_sha256']))
        counts = data.get('config_counts', {'machete':9, 'marlin':3, 'quactlize':6})
        if counts not in ({'machete':9, 'marlin':3, 'quactlize':6},
                          {'machete':9, 'marlin':3, 'quactlize':8},
                          {'machete':9, 'marlin':3, 'quactlize':9},
                          {'machete':9, 'marlin':3, 'quactlize':12}):
            raise ValueError('unknown config inventory')
        for arm,count in counts.items():
            candidates=[r for r in data['rows'] if r['arm']==arm]
            if len(candidates)!=count or {x['config'] for x in candidates}!=set(range(count)):
                raise ValueError('configuration denominator differs')
            winner=min(candidates,key=lambda x:x['median_us'])
            best.append(dict(dtype=dtype,m=m,n=n,k=k,arm=arm,config=winner['config'],us=winner['median_us']))
            for row in candidates:
                if row.get('warmup_min_device_ms')!=250 or row['clocks_before']!=row['clocks_after']:
                    raise ValueError('steady-clock evidence missing')
                if any(x['guards']!='PASS' for x in row['correctness']):
                    raise ValueError('guard proof missing')
                all_rows.append(dict(dtype=dtype,m=m,n=n,k=k,arm=arm,config=row['config'],us=row['median_us'],
                    sm_mhz=row['clocks_after']['sm_mhz'],memory_mhz=row['clocks_after']['memory_mhz'],
                    receipt_sha256=sha(path)))
    expected={(0,m,n,k) for m in (1,8,128) for n,k in ((512,2048),(4096,4096))}
    if seen!=expected or len(authorities)!=1: raise ValueError('cohort/source coverage differs')
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'benchmark.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        writer.writeheader();writer.writerows(all_rows)
    verdict=[]
    for key in sorted(seen):
        values={r['arm']:r for r in best if (r['dtype'],r['m'],r['n'],r['k'])==key}
        ref=min(values[x]['us'] for x in ('machete','marlin'))
        q=values['quactlize']['us']
        verdict.append({'point':key,'best':values,'delta_pct':100*(q/ref-1),
                        'status':'SCREEN_WIN_REQUIRES_CONFIRMATION' if q<ref else 'TARGET_OPEN'})
    (a.output/'summary.json').write_text(json.dumps({'scope':'SCREEN_NOT_PRODUCTION_ADMISSION',
        'configurations':len(all_rows),'points':verdict,'cohort_sha256':sha(a.cohort/'summary.json')},indent=2)+'\n')
    print(f'PPU17_REFERENCE_REPORT points=6 configurations={len(all_rows)} production_admission=PENDING',flush=True)


if __name__=='__main__':
    main()
