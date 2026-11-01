#!/usr/bin/env python3
"""Model-driven shape inventory, Asys sweeps and exact/bucket policy export.

No model-name registry or production writes. plan reads GGUF headers only.
sweep prepares/executes existing backends and exports complete confirmed cells.
An unavailable fused carrier remains a reported gap, not a plain GEMM timing.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.model_tuning import inventory, plans, policy
from dev.strategy.journal import atomic_json


def read(path):
    return json.loads(Path(path).read_bytes())


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('plan','prepare','sweep','export'))
    source=p.add_mutually_exclusive_group()
    source.add_argument('--model',type=Path,help='exact GGUF first-shard file; no recursive search')
    source.add_argument('--models',type=Path,help='JSON list of {name,path,tp}; any model names')
    p.add_argument('--tp',type=int,default=1)
    p.add_argument('--m',type=int,nargs='+',help='default M1..8,16,32,64,128,256,512,1024,2048,4096,5120')
    p.add_argument('--forms',help='default projection,concat,swiglu; fusion has a separate capability boundary')
    p.add_argument('--routers',help='default spread,cluster; evidence profiles, not runtime selection inputs')
    p.add_argument('--headers-snapshot',type=Path,help='host-only replay of saved {GGUF path: header rows}')
    p.add_argument('--llama-source',type=Path,default=Path(os.environ['LLAMA_CI_DIR']) if os.environ.get('LLAMA_CI_DIR') else None)
    p.add_argument('--output',type=Path,help='new run directory; defaults to a retained /tmp/quactlize-tune-*')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--sdk',type=Path,default=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK')))
    p.add_argument('--devices',default='0')
    p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--compute-units',type=int,default=72,help='proposal estimate only; launches use real resource queries')
    p.add_argument('--l2-bytes',type=int,help='verified device L2 capacity, required for sweep')
    p.add_argument('--configs-per-batch',type=int,default=16)
    p.add_argument('--budget-hours',type=float,default=8,help='indexed-prefill worker budget; dense/small-M run to completion')
    p.add_argument('--asys',type=Path);p.add_argument('--smi',type=Path)
    p.add_argument('--confirm-idle',action='store_true')
    p.add_argument('--skip-unavailable',action='store_true',help='run supported forms; overall export still reports missing points')
    p.add_argument('--bucket-max-regret-pct',type=float,default=1.0,
        help='bucket cost relative to measured champion; cannot regress against measured incumbent')
    return p


def model_list(a):
    if a.models:
        result=read(a.models)
        if not isinstance(result,list):raise ValueError('--models must contain a JSON list')
        return result
    if a.model:
        return [dict(name=a.model.stem,path=str(a.model),tp=a.tp)]
    raise ValueError('supply --model or --models for a new run')


def load_inventory(a):
    path=a.output/'inventory.json'
    if path.exists():
        result=inventory.validate(read(path))
        for supplied,field in ((sorted(a.m) if a.m else None,'ms'),
                               (a.forms.split(',') if a.forms else None,'forms'),
                               (a.routers.split(',') if a.routers else None,'routers')):
            if supplied is not None and supplied!=result[field]:
                raise ValueError('resume '+field+' differs from frozen inventory')
        if a.model or a.models:
            supplied={(m['name'],m.get('tp',1),str(Path(m['path']).absolute())) for m in model_list(a)}
            old={(m['name'],m['tp'],str(Path(m['files'][0]).absolute())) for m in result['models']}
            if supplied!=old:raise ValueError('resume model paths/TP differ from frozen inventory')
        return result
    models=model_list(a)
    if a.headers_snapshot:
        snapshots=read(a.headers_snapshot)
        def loader(path):
            rows=snapshots[str(path)]
            return rows,[t for r in rows for t in r['header']['tensors']]
    else:loader=None
    splitter=None
    if any(m.get('tp',1)>1 for m in models):
        if not a.llama_source:raise ValueError('TP inventory needs --llama-source for CPU split rules')
        splitter=inventory.Splitter(a.llama_source,a.output/'tp-headers')
    result=inventory.collect(models,ms=a.m or inventory.MS,forms=(a.forms or ','.join(inventory.FORMS)).split(','),
                             routers=(a.routers or 'spread,cluster').split(','),
                             loader=loader,partitioner=splitter)
    atomic_json(path,result)
    return result


def main(argv=None):
    p=parser();a=p.parse_args(argv)
    cards=a.devices.split(',')
    if (not cards or len(set(cards))!=len(cards) or any(not c.isdecimal() for c in cards) or
            not 1<=a.jobs<=64 or not 1<=a.configs_per_batch<=32 or
            a.compute_units<=0 or not 0<a.budget_hours<=24):
        p.error('invalid device, jobs, batch or budget arguments')
    if a.output is None:
        if a.resume or a.stage=='export':p.error('--resume/export requires --output')
        a.output=Path(tempfile.mkdtemp(prefix='quactlize-tune-',dir=os.environ.get('RUN_ROOT')))
    else:
        a.output=a.output.resolve()
        if a.stage=='export' or a.resume:
            if not (a.output/'inventory.json').is_file():p.error('resume/export needs the retained inventory.json')
        else:
            try:a.output.mkdir(parents=True,exist_ok=False)
            except FileExistsError:p.error('output exists; use --resume or a new directory')
    with (a.output/'coordinator.lock').open('a') as held:
        try:fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('another tuning coordinator owns '+str(a.output)) from None
        return execute(a)


def execute(a):
    print('MODEL_TUNING_RUN output='+str(a.output),flush=True)
    value=load_inventory(a);parts=plans.partition(value)
    summary=dict(status=value['status'],counts=value['counts'],points=len(value['points']),
        executable_points=sum(len(v) for v in parts['groups'].values()),
        unavailable=len(parts['pending']),excluded=len(value['excluded']),blocked=len(value['blocked']),
        gpu_work=False,production_changed=False)
    atomic_json(a.output/'coverage.json',dict(summary,pending=parts['pending'],excluded=value['excluded'],blocked=value['blocked']))
    print('MODEL_TUNING_INVENTORY '+json.dumps(summary),flush=True)
    if a.stage=='plan':return int(value['status']!='READY')
    campaign_path=a.output/'plans/campaign.json'
    if a.stage=='export':
        campaign=read(campaign_path)
    else:
        if a.stage=='sweep' and parts['pending'] and not a.skip_unavailable:
            raise ValueError('some requested forms have no connected Asys carrier; see coverage.json; no GPU work started')
        campaign=plans.prepare(value,a.output/'plans',compute_units=a.compute_units)
    if a.stage=='prepare':
        print('MODEL_TUNING_PREPARED '+json.dumps(dict(status=campaign['status'],jobs=len(campaign['jobs']))),flush=True)
        return int(campaign['status']!='READY')
    if a.stage=='sweep':
        from dev.model_tuning.runner import sweep
        sweep(campaign,a,a.output/'runs')
    from dev.model_tuning.evidence import collect
    records,issues=collect(value,campaign,a.output/'runs')
    result=policy.export(records,expected=[r['id'] for r in value['points']],pending=issues,
                         max_regret_pct=a.bucket_max_regret_pct)
    atomic_json(a.output/'policy.json',result)
    print('MODEL_TUNING_EXPORT '+json.dumps(dict(status=result['status'],exact=len(result['exact']),
        validated_buckets=sum(b['holdout_validated'] for b in result['buckets'].values()),
        missing=len(result['missing']),issues=len(issues),policy=str(a.output/'policy.json'),
        production_changed=False,native_no_regression_admitted=False)),flush=True)
    return int(result['status']!='READY_FOR_INTEGRATION')


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,KeyError) as exc:
        print('MODEL_TUNING_FAIL '+str(exc),file=sys.stderr,flush=True)
        raise SystemExit(1)
