#!/usr/bin/env python3
"""Compact complete H800 results without promoting a production selector."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.h800_smallm.build_gemv import SHAPES
from dev.h800_smallm.select import load_case


def gemv_rows(folder):
    summary=json.loads((folder/'summary.json').read_text())
    expected={f'{scope}-m{m}-{n}x{k}-{mode}' for n,k in SHAPES for mode in ('warm','rotating')
        for scope in ('dense','indexed-shared-a','indexed-independent-a')
        for m in (range(1,8) if scope=='dense' else (1,))}
    if set(summary['complete'])!=expected or summary['failures']: raise ValueError('incomplete GEMV matrix')
    rows=[]
    for key in sorted(expected):
        row=load_case(folder/(key+'.json'))
        if row['rounds']!=6 or row['samples']!=15: raise ValueError('confirmation requires 6x15')
        if row['scope']!='dense' and row['experts']!=256: raise ValueError('indexed confirmation must use E256')
        if row['scope']!='dense' and len(set(row['ids']))!=8: raise ValueError('eight distinct GPU IDs required')
        for arm,best in row['best'].items():
            all_configs={tuple(r['recipe']) for r in row['records'] if r['arm']==arm}
            medians={}
            for cfg in all_configs:
                records=[r for r in row['records'] if r['arm']==arm and tuple(r['recipe'])==cfg]
                if sorted(r['round'] for r in records)!=list(range(6)) or any(len(r['samples_us'])!=15 for r in records):
                    raise ValueError('recipe/round/sample denominator differs')
                medians[cfg]=statistics.median(r['median_us'] for r in records)
            if abs(min(medians.values())-best['median_us'])>1e-8: raise ValueError('reported winner differs')
        delta={a:100*(row['best']['kpack']['median_us']/row['best'][a]['median_us']-1) for a in ('xplane','reference')}
        if delta!=row['delta_pct']: raise ValueError('reported regret differs')
        parity='WITHIN_5PCT' if max(delta.values())<=5 else 'OPEN'
        if row['parity']!=parity: raise ValueError('reported parity differs')
        rows.append(dict(row,key=key,source_sha256=hashlib.sha256((folder/(key+'.json')).read_bytes()).hexdigest()))
    return rows


def moe_rows(folder):
    grouped={};inputs={}
    expected={(merged,tokens,router,k) for merged in (0,1) for tokens in (1,2,3,4)
              for router in (-1,0,1,2) for k in (512,2048,3072)}
    for arm in ('baseline','selected'):
        for turn in range(4):
            path=folder/f'model-{turn}-{arm}.log';text=path.read_text();seen=set()
            if text.count('KPACK_MOE_CHAIN_CUDA PASS cells=96')!=1 or 'ids_and_weights=RAW_BITS bad=0' not in text:
                raise ValueError('MoE oracle incomplete')
            if len(re.findall(r'^KPACK_MOE_CHAIN_CUDA .* replays=7 bad=0 ',text,re.M))!=96:
                raise ValueError('MoE graph-replay denominator differs')
            for line in text.splitlines():
                if not line.startswith('KPACK_MOE_PREPARE_PERF '): continue
                pairs=dict(re.findall(r'(\w+)=([^ ]+)',line))
                key=tuple(int(pairs[x]) for x in ('merged','tokens','router','k'))
                samples=json.loads(pairs['samples'])
                if (key in seen or len(samples)!=15 or any(not math.isfinite(x) or x<=0 for x in samples)
                    or abs(statistics.median(samples)-float(pairs['median_us']))>2e-6):
                    raise ValueError('invalid MoE samples')
                seen.add(key);grouped.setdefault(key,{}).setdefault(arm,[]).append(samples)
            if seen!=expected: raise ValueError('MoE shape matrix differs')
            inputs[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    rows=[]
    for key,arms in sorted(grouped.items()):
        med={arm:statistics.median(statistics.median(s) for s in rounds) for arm,rounds in arms.items()}
        rows.append(dict(zip(('merged','tokens','router','k'),key),median_us=med,
            delta_pct=100*(med['selected']/med['baseline']-1),samples_us=arms))
    return rows,inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--moe',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();g=gemv_rows(a.results);m,inputs=moe_rows(a.moe)
    a.output.mkdir(parents=True,exist_ok=False)
    raw=dict(gemv=g,moe=m,moe_log_hashes=inputs)
    compressed=gzip.compress(json.dumps(raw,separators=(',',':')).encode(),mtime=0)
    (a.output/'samples.json.gz').write_bytes(compressed)
    compact_g=[{k:v for k,v in row.items() if k!='records'} for row in g]
    compact_m=[{k:v for k,v in row.items() if k!='samples_us'} for row in m]
    summary=dict(scope='H800_Q4_DENSE_M1_TO_7_INDEXED_M1_AND_MOE_PREPARE',
        production_changed=False,PPU_admission='PENDING',NCU_counters='UNAVAILABLE_HOST_PERMISSION',
        gemv_cells=len(g),gemv_numeric_pass=len(g),gemv_within_5pct=sum(r['parity']=='WITHIN_5PCT' for r in g),
        gemv_open=[r['key'] for r in g if r['parity']!='WITHIN_5PCT'],
        moe_cells=len(m),moe_model_splits=[2,2,1],
        sample_receipt_sha256=hashlib.sha256(compressed).hexdigest(),gemv=compact_g,moe=compact_m)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.output/'bundle.json').write_text((a.bundle/'manifest.json').read_text())
    (a.output/'selection.json').write_text((a.bundle/'selection.json').read_text())
    print(f'H800_SMALLM_REPORT numeric={len(g)}/108 parity={summary["gemv_within_5pct"]}/108 moe={len(m)}/96 output={a.output}')


if __name__=='__main__': main()
