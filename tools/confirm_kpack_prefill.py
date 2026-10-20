#!/usr/bin/env python3
"""Confirm completed-screen finalists; no sweep, online tuner or default policy write."""
import argparse
import fcntl
import io
import json
import lzma
import os
from pathlib import Path
import subprocess
import signal
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.strategy.journal import atomic_json
from dev.strategy.prefill_closure import require, export, REVIEW_SHA256
from dev.strategy.prefill_confirmation import bind_point, verdict, ROUNDS
from dev.strategy.prefill_plan import family
from dev.strategy.provenance import harness_identity
from dev.strategy.resource_pilot import verified_plan
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest


def read(path):
    value = json.loads(Path(path).read_text())
    if 'record_sha256' in value:
        require(value['record_sha256'] == digest({k:v for k,v in value.items() if k != 'record_sha256'}),
                'changed receipt: '+str(path))
    return value


def save_once(path, value):
    if path.exists():
        require(read(path) == value, 'resume identity changed: '+str(path))
    else:
        atomic_json(path, value)


def bind_shard(previous, plan, shard):
    old = read(previous/'execution-plan.json'); verified_plan(old)
    points = {p['id']:p for p in old['points']}
    selected = [p for p in plan['points'] if p['shard'] == shard]
    require(set(points) == {p['id'] for p in selected}, 'prior shard workload denominator differs')
    return [bind_point(p, points[p['id']]) for p in selected]


def setup_worker(args):
    from dev.strategy.prefill_build import build, verify
    from tools.run_kpack_strategy_prefill import stable_build, wait
    from tools.run_kpack_strategy_decode import dequant_checked
    from tools.build_kpack_dequant import build as build_dequant
    from tools.probe_kpack_strategy import device_probe
    from tools.run_kpack_strategy_pilot import oracle_identity
    from dev.strategy.isolated import check_sdk_graph_abi
    from quactlize.runtime.native import sdk_identity

    plan = read(args.plan); verified_plan(plan)
    previous = args.previous/f'shard-{args.shard}'/'measurement'
    old = read(previous/'authority.json')
    shard = next(s for s in plan['shards'] if s['shard'] == args.shard)
    require(shard['authority'] == digest(old) and old['resource'] == shard['resource'],
            'previous raw run differs from reviewed archive')
    device, sdk = device_probe(args.sdk), sdk_identity(args.sdk)
    require(device == old['resource']['device'] and sdk == old['resource']['sdk'],
            'confirmation must use its original physical card and SDK')
    oracle = oracle_identity()
    require(oracle == old['oracle'], 'independent oracle/runtime packages differ from reviewed screen')
    graph_abi = check_sdk_graph_abi(args.sdk)
    lock = Path('/tmp')/('quactlize-components-'+digest(device['pci'])[:24]+'.lock')
    with lock.open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True, exist_ok=True)
        # Narrow helper build only. No TensorCore recompilation or new parent.
        hp = stable_build(args, 'helpers', build, lambda path,sdk:verify(sdk,path))
        dq = stable_build(args, 'dequant', build_dequant, dequant_checked)
        points = bind_shard(previous, plan, args.shard)
        authority = dict(plan=plan['plan_sha256'], device=device, sdk=sdk,
            cache=old['resource']['cache'], harness=harness_identity(ROOT),
            helpers_path=str(args.output/'helpers'), dequant_path=str(args.output/'dequant'),
            helpers=hp['sha256'], dequant=dq['sha256'], l2=old['l2'],
            providers=old['providers'],
            oracle=oracle, graph_abi=graph_abi, idle='OPERATOR_CONFIRMED',
            body_source=digest([dq['source_hashes'],hp['identity']]),
            previous_authority=digest(old), protocol=plan['protocol'],
            previous_timings_reused=False, production_changed=False)
        save_once(args.output/'authority.json', authority)
        families = {}
        for point in points: families.setdefault(family(point), []).append(point)
        for key, rows in families.items():
            old_manifest = read(previous/'families'/(key+'.json'))
            require(old_manifest['sha256'] == digest({k:v for k,v in old_manifest.items() if k != 'sha256'}) and
                    old_manifest['authority_sha256'] == digest(old), 'prior resource family manifest changed')
            arms = [j['arm'] for p in rows for j in p['jobs'].values() if j['role'] == 'tc']
            mids = {a['module'] for a in arms}
            queries = {a['binding']['query'] for a in arms}
            value = dict(points=rows, authority=digest(authority),
                modules={m:old_manifest['modules'][m] for m in mids},
                specs={m:old_manifest['specs'][m] for m in mids},
                queries={q:old_manifest['queries'][q] for q in queries})
            require(all(old['tc'][m] == value['modules'][m]['sha256'] for m in mids),
                    'prior image differs from reviewed binary')
            value['sha256'] = digest(value)
            save_once(args.output/'families'/(key+'.json'), value)
        # Each family keeps its original card. Every round is a fresh paired
        # comparison; completed round receipts survive interruption unchanged.
        started = time.monotonic(); completed = 0
        for key, rows in sorted(families.items()):
            skipped = set()
            while any(p['id'] not in skipped and not all(
                    (args.output/'points'/p['id']/f'round-{i}.json').exists() for i in range(ROUNDS))
                    for p in rows):
                attempt = key+'-'+str(time.time_ns())
                failure = args.output/'failures'/(attempt+'.json')
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--child',
                    '--sdk', str(args.sdk), '--output', str(args.output), '--family', key,
                    '--failure-report', str(failure)]
                for pid in sorted(skipped): command += ['--skip-point', pid]
                log = args.output/f'family-{attempt}.log'
                rc = wait(command, log, held=held, phase='finalists-'+key[:12])
                print(f'STRATEGY_CONFIRM_FAMILY family={key} rc={rc} remaining_continue=1', flush=True)
                if not rc or not failure.is_file(): break
                failed_point = read(failure).get('point')
                if failed_point not in {p['id'] for p in rows} or failed_point in skipped: break
                # A failed point gets no automatic retry. Fresh-process peers
                # continue; only a later --resume retries its unfinished rounds.
                skipped.add(failed_point)
            completed += len(rows)
            print(f'STRATEGY_CONFIRM_PROGRESS points={completed}/{len(points)} '
                  f'minutes={(time.monotonic()-started)/60:.1f} sweep=0', flush=True)
        results = []
        for point in points:
            paths = [args.output/'points'/point['id']/f'round-{i}.json' for i in range(ROUNDS)]
            if not all(p.exists() for p in paths):
                results.append(dict(point=point['id'], status='INCOMPLETE')); continue
            rows = [read(p) for p in paths]
            require(all(r['authority']['sha256'] == digest(authority) for r in rows),
                    'completed round uses another run authority')
            value = verdict(point, rows)
            atomic_json(args.output/'results'/(point['id']+'.json'), value)
            results.append(value)
        summary = dict(status='PASS' if all(r['status']=='PASS' for r in results) else 'INCOMPLETE',
            expected=len(points), confirmed=sum(r['status']=='PASS' for r in results),
            failures=[dict(point=r['point'], status=r['status']) for r in results if r['status']!='PASS'],
            plan=plan['plan_sha256'], authority=digest(authority), production_changed=False,
            performance_admitted=False, scope='COMPONENT_CONFIRMATION_NOT_MODEL_AB')
        atomic_json(args.output/'summary.json', summary)
        print('STRATEGY_CONFIRM_DONE '+json.dumps(summary), flush=True)
        return int(summary['status'] != 'PASS')


def package(output):
    """Small review only: selected results, receipts and bounded failure logs."""
    archive = output/('confirmation-'+str(time.time_ns())+'.tgz')
    selected = [output/'request.json', output/'summary.json', output/'finalists.json',
                output/'devices.json', output/'images.json', output/'source.json']
    selected += sorted(output.glob('clock-lock-*.json'))
    selected += sorted(output.glob('setup-error-*.json'))
    selected += sorted((output/'build-attempts').glob('*/*.log'))
    selected += sorted((output/'cache/tc').glob('*/build.log'))
    selected += sorted(output.glob('*.log'))
    for shard in sorted(output.glob('shard-*')):
        selected += [shard/'authority.json',shard/'summary.json',shard/'console.log',
                     shard/'resources.json',shard/'command.json']
        selected += sorted((shard/'results').glob('*.json'))
        selected += sorted((shard/'failures').glob('*.json'))
        selected += sorted(shard.glob('family-*.log'))
        selected += sorted((shard/'build-attempts').glob('*/*.log'))
        if (shard/'authority.json').exists():
            from tools.pack_kpack_strategy_review import Review
            compact=shard/('review-'+str(time.time_ns())+'.jsonl.xz')
            with lzma.open(compact,'xb',preset=6) as stream:
                review=Review(shard,stream)
                review.emit('confirmation_authority',value=read(shard/'authority.json'))
                for path in sorted((shard/'points').glob('*/round-*.json')):
                    row=review.read(path)
                    extra = {k: row[k] for k in ('clock_observations', 'wall_seconds') if k in row}
                    review.emit('prefill_confirmation_round',point=row['point'],round=row['round'],
                        authority=row['authority'],record_sha256=row['record_sha256'],
                        order=row['order'],jobs={k:review.proof(v) for k,v in row['jobs'].items()}, **extra)
                    for cid,c in row['components'].items():
                        review.component(cid,c,'confirmation',row['point'])
                review.emit('end',counts=dict(review.counts),prior_lines_sha256=review.content.hexdigest(),
                    export_status='COMPLETE',production_changed=False,performance_admitted=False)
            selected.append(compact)
    with tarfile.open(archive,'x:gz') as tar:
        for path in selected:
            require(not path.is_symlink() and path.resolve().is_relative_to(output), 'unsafe review path')
            if not path.is_file(): continue
            with path.open('rb') as stream:
                if path.suffix == '.log': stream.seek(max(0,path.stat().st_size-16384))
                data = stream.read()
            entry = tarfile.TarInfo(str(path.relative_to(output)))
            entry.size=len(data); tar.addfile(entry,io.BytesIO(data))
    print('results='+str(archive),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path)
    p.add_argument('--review',type=Path)
    p.add_argument('--plan',type=Path)
    p.add_argument('--sdk',type=Path,default=Path('/usr/local/PPU_SDK'))
    p.add_argument('--devices',default='1,2,3,4,5,6,7')
    p.add_argument('--output',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--idle-confirmed',action='store_true')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--run-root',type=Path,default=Path('/workspace'))
    p.add_argument('--shard',type=int,help=argparse.SUPPRESS)
    p.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--family',help=argparse.SUPPRESS)
    p.add_argument('--skip-point',action='append',default=[],help=argparse.SUPPRESS)
    p.add_argument('--failure-report',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args(); args.source=ROOT; args.sdk=args.sdk.resolve(strict=True)
    if args.output: args.output=args.output.resolve()
    if args.child:
        if not args.failure_report or not args.family:
            p.error('child requires family and failure report')
        require(args.failure_report.parent == args.output/'failures', 'invalid child failure report')
        from dev.strategy.prefill_confirm_worker import child
        return child(args)
    if args.shard is not None: return setup_worker(args)
    if not args.previous: p.error('--previous completed supplement directory required')
    args.previous=args.previous.resolve(strict=True)
    if args.resume and not args.output: p.error('--resume requires --output')
    output=args.output or Path(tempfile.mkdtemp(prefix='kpack-prefill-confirm-',dir=args.run_root.resolve(strict=True)))
    require(output!=args.previous and not output.is_relative_to(args.previous),'output must not change previous run')
    output.mkdir(parents=True,exist_ok=args.resume or args.output is None)
    request=dict(previous=str(args.previous),sdk=str(args.sdk),harness=harness_identity(ROOT),devices=args.devices)
    save_once(output/'request.json',request)
    path=output/'finalists.json'
    if not path.exists():
        if args.plan:
            plan=read(args.plan); verified_plan(plan)
            require(plan['archive_sha256']==REVIEW_SHA256,'unreviewed finalist archive')
        else:
            from tools.export_kpack_strategy import load_sources
            review=args.review or args.previous/'review-1792311707684248419.tgz'
            plan=export(review,load_sources()[0])
        atomic_json(path,plan)
    plan=read(path);verified_plan(plan)
    require(plan['counts']['rule_gaps']==0,'blind rule gap remains; do not start confirmation')
    print('STRATEGY_CONFIRM_PLAN '+json.dumps(plan['counts']|dict(run=str(output),
        rounds=ROUNDS,new_tc_compiles=0,full_sweep=False,duration_guaranteed=False)),flush=True)
    if args.plan_only:return 0
    if not args.idle_confirmed:p.error('confirm all selected GPUs idle with --idle-confirmed')
    selectors=args.devices.split(',')
    require(all(s.isdigit() for s in selectors) and len(set(selectors))==len(selectors)==len(plan['shards']),
            'supply all original physical cards once')
    code=('import json,sys;from pathlib import Path;from tools.probe_kpack_strategy import device_probe;'
          'print(json.dumps(device_probe(Path(sys.argv[1]))))')
    visible=[]
    for s in selectors:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=s)
        d=json.loads(subprocess.check_output([sys.executable,'-c',code,str(args.sdk)],cwd=ROOT,env=env,text=True))
        visible.append((s,d))
    require(len({(d['host'],d['pci']) for s,d in visible})==len(visible),'duplicate physical GPU')
    bindings=[]
    for shard in plan['shards']:
        matches=[s for s,d in visible if d==shard['resource']['device']]
        require(len(matches)==1,'original device missing for shard '+str(shard['shard']))
        bindings.append((shard,matches[0]))
    active,statuses={},{}
    totals={s['shard']:sum(p['shard']==s['shard'] for p in plan['points'])*ROUNDS for s in plan['shards']}
    initial={s:len(list((output/f'shard-{s}'/'points').glob('*/round-*.json'))) for s in totals}
    start=time.monotonic()
    try:
        for shard,selector in bindings:
            folder=output/f'shard-{shard["shard"]}';folder.mkdir(exist_ok=True)
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--shard',str(shard['shard']),
                 '--previous',str(args.previous),'--plan',str(path),'--sdk',str(args.sdk),'--output',str(folder)]
            save_once(folder/'command.json',dict(argv=cmd,selector=selector))
            with (folder/'console.log').open('ab') as log:
                active[shard['shard']]=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=selector,OMP_NUM_THREADS='1',
                             OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        while active:
            for shard,proc in list(active.items()):
                if proc.poll() is not None:
                    statuses[str(shard)]=proc.returncode;del active[shard]
            atomic_json(output/'summary.json',dict(status='RUNNING' if active else
                'PASS' if all(r==0 for r in statuses.values()) else 'INCOMPLETE',processes=statuses,
                elapsed_seconds=time.monotonic()-start,production_changed=False,performance_admitted=False))
            if active:
                completed={s:len(list((output/f'shard-{s}'/'points').glob('*/round-*.json'))) for s in totals}
                elapsed=time.monotonic()-start
                remaining=[elapsed/(completed[s]-initial[s])*(totals[s]-completed[s])/60
                           for s in active if completed[s]-initial[s]>=ROUNDS]
                eta=(f'{max(remaining):.1f}' if len(remaining)==len(active) else 'UNKNOWN')
                print(f'STRATEGY_CONFIRM_WAIT alive={len(active)}/{len(selectors)} '
                      f'rounds={sum(completed.values())}/{sum(totals.values())} '
                      f'minutes={elapsed/60:.1f} remaining_minutes={eta} '
                      f'eta=OBSERVED_ROUND_RATE_NOT_GUARANTEE logs={output}/shard-*/console.log',flush=True)
                time.sleep(30)
    finally:
        if active:
            atomic_json(output/'summary.json',dict(status='INTERRUPTED',processes=statuses,
                active=list(active),production_changed=False,performance_admitted=False))
            for proc in active.values():
                if proc.poll() is None and os.getpgid(proc.pid)==proc.pid:
                    os.killpg(proc.pid,signal.SIGTERM)
            for proc in active.values():
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if proc.poll() is None and os.getpgid(proc.pid)==proc.pid:
                        os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait()
        package(output)
    return int(any(r!=0 for r in statuses.values()))


if __name__=='__main__':
    raise SystemExit(main())
