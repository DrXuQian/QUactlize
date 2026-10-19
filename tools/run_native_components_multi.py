#!/usr/bin/env python3
"""Coordinate disjoint native component shards; never time two jobs on one GPU."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.strategy.journal import atomic_json
from quactlize.runtime.tuning import digest


def pack(root, destinations):
    target=root/('review-'+str(time.time_ns())+'.tgz')
    with tarfile.open(target,'w:gz')as archive:
        paths=[root/'summary.json',root/'run.json']
        for output in destinations:
            paths += [output/n for n in ('authority.json','summary.json','comparison.json','comparison.csv')]
            paths += list((output/'results').glob('*.json'))+list((output/'helpers').glob('*.json'))
            paths += [output/'build/adapter.json',output/'build/carrier.json',output/'console.log']
            if not (output/'summary.json').is_file():
                paths += list((output/'build').glob('*build.log'))
            # Failure logs only. Successful raw component samples are already
            # in result JSON. Never package weights, objects, DSOs or requests.
            for base in ('results','helpers'):
                for p in (output/base).glob('*.json'):
                    data=json.loads(p.read_text())
                    if data.get('status')=='FAIL'and data.get('log'):
                        log=Path(data['log'])
                        if log.is_file() and log.resolve().is_relative_to(root):paths.append(log)
            paths += list((output/'attempts').glob('*/*.inventory.json'))
        for p in sorted(set(paths)):
            if p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root):
                archive.add(p,arcname=str(p.relative_to(root)),recursive=False)
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--devices',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--llama-source',type=Path,required=True)
    p.add_argument('--llama-build',type=Path,required=True)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--l2-bytes',type=int,default=0)
    p.add_argument('--phase',choices=('all','pilot'),default='all')
    p.add_argument('--confirm-idle',action='store_true')
    a=p.parse_args();cards=a.devices.split(',')
    if not cards or any(not x.isdigit()for x in cards)or len(set(cards))!=len(cards):p.error('devices must be distinct ordinals')
    if not a.confirm_idle:p.error('--confirm-idle is required')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u',str(ROOT/'tools/run_native_components.py'),
        '--llama-source',str(a.llama_source.resolve()),'--llama-build',str(a.llama_build.resolve()),
        '--sdk',str(a.sdk.resolve()),'--l2-bytes',str(a.l2_bytes),'--phase',a.phase,'--confirm-idle']
    binding=dict(devices=cards,command=command,source_root=str(ROOT),mode='NATIVE_COMPONENTS_NO_CALLER_PATCH')
    receipt=a.output/'run.json'
    if receipt.exists() and json.loads(receipt.read_text())!=binding:raise ValueError('resume coordinator inputs changed')
    if not receipt.exists():atomic_json(receipt,binding)
    probes=[]
    for card in cards:
        directory=a.output/('probe-'+card);directory.mkdir(exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=card)
        with (directory/'probe.log').open('a')as log:
            subprocess.run([*command,'--probe-only','--output',str(directory)],env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        probes.append(json.loads((directory/'device.json').read_text()))
    physical=[(x['device']['host'],x['device']['pci'])for x in probes]
    if len(set(physical))!=len(cards):raise ValueError('requested workers alias the same physical GPU')
    cohorts={(x['sdk'],x['device']['compute_units'],x['device']['warp_size'],x['l2']['bytes'])for x in probes}
    if len(cohorts)!=1:raise ValueError('mixed device/SDK cohort; use separate comparisons')
    children=[];outputs=[];start=time.monotonic()
    try:
        for shard,card in enumerate(cards):
            output=a.output/('gpu-'+card);output.mkdir(exist_ok=True);outputs.append(output)
            log=(output/'console.log').open('a')
            child=subprocess.Popen([*command,'--output',str(output),'--shard',str(shard),'--shards',str(len(cards))],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=card),stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True)
            children.append((child,log))
        last=0
        while any(child.poll()is None for child,_ in children):
            now=time.monotonic()
            if now-last>=30:
                completed=0
                for output in outputs:
                    completed += sum(json.loads(path.read_text()).get('status')=='MEASURED'for path in (output/'results').glob('*.json'))
                print(f'NATIVE_COMPONENT_MULTI_WAIT alive={sum(c.poll()is None for c,_ in children)}/{len(cards)} '
                      f'measured={completed} elapsed_minutes={(now-start)/60:.1f} logs={a.output}/gpu-*/console.log',flush=True)
                last=now
            time.sleep(1)
    except BaseException:
        for child,_ in children:
            if child.poll()is None:
                # Stop only this coordinator's own worker process group,
                # including its in-flight fixture child, never another job.
                os.killpg(child.pid,signal.SIGTERM)
        for child,_ in children:child.wait()
        raise
    finally:
        for _,log in children:log.close()
    rc=[child.returncode for child,_ in children]
    summaries=[json.loads((output/'summary.json').read_text())if (output/'summary.json').is_file()else
               dict(status='INCOMPLETE',expected=None,measured=0)for output in outputs]
    result=dict(status='COMPLETE'if not any(rc)and all(s['status']=='COMPLETE'for s in summaries)else'INCOMPLETE',
                returncodes=rc,measured=sum(s['measured']for s in summaries),
                scope='NATIVE_AND_SUPPLEMENT_COMPONENTS_HISTORICAL_COMPARISON_NOT_MODEL_TPOT',
                no_regression_admitted=False,production_changed=False)
    atomic_json(a.output/'summary.json',result)
    archive=pack(a.output,outputs)
    print('NATIVE_COMPONENT_MULTI_DONE '+json.dumps(result),flush=True)
    print('results='+str(archive),flush=True)
    return 0 if result['status']=='COMPLETE'else 1


if __name__=='__main__':raise SystemExit(main())
