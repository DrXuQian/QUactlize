"""One coordinator; all paired configurations of a point stay on one GPU."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

from dev.strategy.journal import atomic_json
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[2]


def commands(job, args, root):
    backend = job['backend']; directory = Path(root)/backend
    plan = json.loads(Path(job['plan']).read_bytes())
    if (plan['sha256'] != job['sha256'] or
            digest({k:v for k,v in plan.items() if k!='sha256'})!=plan['sha256']):
        raise ValueError('campaign/engine plan changed')
    available = args.devices.split(',')
    capacity = len(plan['points'])
    if backend == 'dense-prefill':
        capacity = len({(p['workload']['qtype'],p['workload']['n'],p['workload']['k']) for p in plan['points']})
    cards = available[:min(capacity, len(available))]
    base = [sys.executable, '-u']
    common = ['--sdk',str(args.sdk),'--jobs',str(args.jobs),'--l2-bytes',str(args.l2_bytes),'--confirm-idle']
    for name in ('asys','smi'):
        if getattr(args,name,None):
            common += ['--'+name, str(getattr(args,name))]
    if backend.startswith('smallm-'):
        command = base+[str(ROOT/'tools/run_smallm_regressions_asys.py'), '--external-plan',job['plan'],
            '--output',str(directory),'--devices',','.join(cards),*common]
        if (directory/'plan.json').exists():command.append('--resume')
        return [(command, directory.with_suffix('.log'), None)]
    if backend == 'moe-prefill':
        command = base+[str(ROOT/'tools/run_moe_confirmation.py'),'--external-plan',job['plan'],
            '--mode','refine-prefill','--output',str(directory),'--devices',','.join(cards),
            '--budget-hours',str(args.budget_hours),*common]
        if (directory/'run.json').exists():command.append('--resume')
        return [(command, directory.with_suffix('.log'), None)]
    result=[]
    for index,card in enumerate(cards):
        out = directory/('gpu-'+card)/'results'
        command=base+[str(ROOT/'tools/run_prefill_model_asys.py'),'--finalist-plan',job['plan'],
            '--output',str(out),'--device',card,'--shard',str(index),'--shards',str(len(cards)),
            '--no-native','--configs-per-batch',str(args.configs_per_batch),*common]
        if args.llama_source:command+=['--llama-source',str(args.llama_source)]
        if out.exists():command.append('--resume')
        result.append((command,out.parent/'console.log',dict(os.environ,CUDA_VISIBLE_DEVICES=card)))
    return result


def sweep(campaign, args, root):
    if not args.confirm_idle:
        raise ValueError('confirm assigned devices are idle before timing')
    if not args.l2_bytes or args.l2_bytes <= 0:
        raise ValueError('--l2-bytes must be the verified device L2 capacity')
    if campaign['pending'] and not args.skip_unavailable:
        raise ValueError('inventory has unsupported measurement contracts; choose explicit forms or --skip-unavailable')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    binding=dict(campaign=campaign['sha256'],devices=args.devices,sdk=str(args.sdk),l2_bytes=args.l2_bytes)
    config=root/'coordinator.json'
    if config.exists() and json.loads(config.read_bytes())!=binding:
        raise ValueError('resume campaign/device/SDK changed; use a new run')
    atomic_json(config,binding)
    statuses=[]
    for job in campaign['jobs']:
        active=[];started=time.monotonic()
        try:
            for argv,log,env in commands(job,args,root):
                log.parent.mkdir(parents=True,exist_ok=True)
                atomic_json(log.with_suffix('.command.json'),argv)
                with log.open('a') as stream:
                    child=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                active.append((child,log))
            while any(p.poll() is None for p,_ in active):
                print('MODEL_TUNING_WAIT '+json.dumps(dict(backend=job['backend'],
                    alive=sum(p.poll() is None for p,_ in active),minutes=round((time.monotonic()-started)/60,1),
                    logs=[str(log) for _,log in active])),flush=True)
                for child,_ in active:
                    if child.poll() is None:
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:pass
                        break
        finally:
            from tools.run_smallm_regressions_asys import stop_process
            for child,_ in active:stop_process(child)
        rows=[]
        for child,log in active:
            row=dict(rc=child.returncode,log=str(log))
            if child.returncode:
                row['diagnostic']='\n'.join(log.read_text(errors='replace').splitlines()[-25:])
                print('MODEL_TUNING_WORKER_FAIL '+json.dumps(row),flush=True)
            rows.append(row)
        statuses.append(dict(backend=job['backend'],workers=rows,seconds=time.monotonic()-started))
        atomic_json(root/'status.json',statuses)
    return statuses
