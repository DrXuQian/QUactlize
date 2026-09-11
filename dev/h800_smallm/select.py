#!/usr/bin/env python3
"""Freeze bounded screen winners and assemble only their existing libraries."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


def load_case(path):
    row=json.loads(path.read_text())
    if row['status']!='PASS': raise ValueError('failed screen case: '+str(path))
    for r in row['records']:
        if (not r['samples_us'] or any(not math.isfinite(x) or x<=0 for x in r['samples_us']) or
            not math.isfinite(r['error']) or r['error']>=.005 or
            abs(statistics.median(r['samples_us'])-r['median_us'])>1e-5):
            raise ValueError('invalid screen record: '+str(path))
    return row


def rank(row,arm):
    values={}
    for r in row['records']:
        if r['arm']==arm: values.setdefault(tuple(r['recipe']),[]).append(r['median_us'])
    return sorted(values,key=lambda x:statistics.median(values[x]))


def select(screens):
    options={}
    for folder in screens:
        for path in sorted(folder.glob('*.json')):
            if path.name=='summary.json': continue
            row=load_case(path);options.setdefault(path.stem,[]).append(row)
    selected={}
    for key,rows in options.items():
        # Use normalized within-run regret when comparing distinct kernel
        # bodies measured in separate screens. Final comparison is same-run.
        winner=min(rows,key=lambda r:max(r['delta_pct'].values()))
        recipes={arm:[list(x) for x in rank(winner,arm)[:2]] for arm in ('xplane','reference','kpack')}
        # Challenge both controls with winners seen in other screens as well.
        for arm in ('xplane','reference'):
            other={tuple(rank(r,arm)[0]) for r in rows}
            recipes[arm]=[list(x) for x in sorted(other|{tuple(x) for x in recipes[arm]})]
        recipes['_implementation']=winner.get('kpack_implementation',
            'small' if winner['n']==512 else 'medium' if winner['n']==1024 else 'large')
        selected[key]=recipes
    return selected


def assemble(bundles,selected,output):
    needed={'xplane','reference'}|{r['_implementation'] for r in selected.values()}
    output.mkdir(parents=True,exist_ok=False);arms={}
    for folder in reversed(bundles):
        manifest=json.loads((folder/'manifest.json').read_text())
        for arm,row in manifest['arms'].items():
            if arm not in needed or arm in arms: continue
            source=folder/arm
            if hashlib.sha256((source/'kernel.so').read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('changed payload: '+str(source))
            shutil.copytree(source,output/arm)
            arms[arm]=dict(row,origin=str(source),origin_manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest())
    if set(arms)!=needed: raise ValueError('missing implementations: '+str(needed-set(arms)))
    (output/'manifest.json').write_text(json.dumps(dict(arms=arms,production_changed=False,
        scope='CUDA_H800_FROZEN_SMALLM_CONFIRMATION'),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screens',type=Path,nargs='+',required=True)
    p.add_argument('--bundles',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();choices=select(a.screens)
    if len(choices)!=108: raise ValueError(f'expected 108 workload/cache cells, got {len(choices)}')
    assemble(a.bundles,choices,a.output)
    (a.output/'selection.json').write_text(json.dumps(choices,indent=2)+'\n')
    print('H800_SMALLM_SELECTION FROZEN cells=108 output='+str(a.output),flush=True)


if __name__=='__main__': main()
