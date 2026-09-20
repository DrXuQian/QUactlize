#!/usr/bin/env python3
"""Bounded paired-chain device gate, complete-call confirmation and ACU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_model.run import verify
from dev.gemv_model.access import access
from dev.gemv_simt.native import Runtime, Graph, checked
from dev.gemv_simt.q8_vector_run import l2_identity
from dev.tp2_chain.plan import POINTS, ARMS, TOKEN_CONTROLS, ROUNDS, SAMPLES, candidates, stages
from dev.tp2_chain.bench import Bench
from quactlize.runtime.compiler import sha, LIBRARIES
from tools.run_kpack_pack_gate import device_identity
from tools.profile_kpack_gpu_compact import AcuRange, acu_launch_command


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def fresh_path(directory, name):
    """A profile is a family of files, not a file named only by its stem."""
    path = directory/name
    if path.exists() or any(directory.glob(name+'.*')):
        path = directory/(name+'.attempt-'+str(time.time_ns()))
    return path


def summarize(samples):
    if len(samples) != ROUNDS or any(len(r) != SAMPLES for r in samples):
        raise ValueError('six complete rounds of fifteen samples required')
    if any(not math.isfinite(x) or x <= 0 for r in samples for x in r):
        raise ValueError('invalid complete-chain timing')
    medians = [statistics.median(r) for r in samples]
    return dict(median_us=statistics.median(medians), round_medians_us=medians, samples_us=samples)


def profiles(text, point, arm):
    offset = text.find('"ID"')
    if offset < 0:
        raise ValueError('ACU raw kernel export missing')
    rows = [r for r in csv.DictReader(io.StringIO(text[offset:])) if r.get('Kernel Name')]
    names = [re.sub(r'\s+', '', r['Kernel Name']) for r in rows]
    if len(rows) != len(stages(arm)):
        raise ValueError('profile does not contain one complete chain')
    if 'prepare_detail::once_with_finish<' not in names[0] or 'bfloat16_t' not in names[0]:
        raise ValueError('production BF16 snapshot prepare missing')
    if 'moe_weighted_finish<true,cutlass::bfloat16_t>' not in names[-1]:
        raise ValueError('production BF16 weighted finish missing')
    if not any(s in names[-2] for s in ('measured_decode_kernel<13,', 'register_reuse_model<13,')):
        raise ValueError('retained Q5 down producer missing')
    if arm == 'unfused':
        if not any(s in names[1] for s in ('measured_decode_kernel<12,', 'register_reuse<12,')):
            raise ValueError('selected Q4 projection missing')
        if 'moe_chain_swiglu_compute<cutlass::bfloat16_t>' not in names[2]:
            raise ValueError('standalone BF16 activation missing')
    elif arm == 'paired-shipping':
        if not any(s in names[1] for s in ('simt_gate_up<12,1,1,8>', 'simt_gate_up_model<12,1,1,8,')):
            raise ValueError('shipped paired body missing')
    else:
        index = int(arm.rsplit('-', 1)[1]); cfg = candidates(point)[index]
        if f'point_12_{point.physical_n}_{point.k}_arm_{index}(' not in names[1]:
            raise ValueError('paired candidate implementation differs')
        if rows[1]['Block Size'].replace(' ', '') != f'({cfg.warps*32},1,1)' or \
                rows[1]['Grid Size'].replace(' ', '') != f'({8*point.physical_n//cfg.tile_n},1,1)':
            raise ValueError('paired candidate geometry differs')
    return [dict(kernel=r['Kernel Name'], grid=r['Grid Size'], block=r['Block Size']) for r in rows]


def numerical(bench, arm):
    records = []
    rt = bench.rt
    for tokens in TOKEN_CONTROLS:
        mark = len(rt.allocations)
        graph = None
        try:
            bench.configure(tokens)
            bench.update(0)
            graph = Graph(rt, [bench.launch(arm)])
            for repeat, large in ((0, False), (1, False), (1, True)):
                bench.update(repeat, large=large)
                graph.sample()
                records.append(dict(tokens=tokens, repeat=repeat, large=large,
                                    router=True, alias=True, **bench.check()))
        finally:
            if graph:
                graph.close()
            rt.release_after(mark); bench.scratch.clear()
    # Snapshot contract without recomputing IDs/weights; direct aliases and
    # invalid expert sentinels are independent from the router implementation.
    for router, alias in ((True, False), (False, True), (False, False)):
        mark = len(rt.allocations)
        graph = None
        try:
            bench.configure(1, router=router, alias=alias); bench.update(0)
            graph = Graph(rt, [bench.launch(arm)])
            for repeat in (0, 1):
                bench.update(repeat); graph.sample()
                records.append(dict(tokens=1, repeat=repeat, large=False,
                                    router=router, alias=alias, **bench.check()))
            if not router:
                for invalid in (-1, 256):
                    bench.update(0, invalid=invalid); graph.sample(); bench.check(invalid=True)
        finally:
            if graph:
                graph.close()
            rt.release_after(mark); bench.scratch.clear()
    return records


def child(args):
    manifest = verify(args.bundle, 'tp2-chain')
    point = next(p for p in POINTS if p.name == args.point)
    result = dict(status='FAIL', point=asdict(point), manifest_sha256=sha(args.bundle/'manifest.json'),
                  numerics={}, failures={}, production_selection_changed=False, scope='FULL_MOE_CHAIN_NOT_MODEL_TPOT')
    rt = None
    graphs = {}
    try:
        rt = Runtime(args.sdk, 'ppu')
        identity = device_identity(rt)
        l2 = l2_identity(dict(l2_bytes=rt.attribute(38)), args.l2_bytes)
        if l2['l2_bytes'] <= 0:
            raise ValueError('verified positive L2 size required')
        result.update(device=identity, l2=l2, runtime={f'lib{x}.so':sha(args.sdk/'lib'/f'lib{x}.so') for x in LIBRARIES},
                      timing_idle_admission='OPERATOR_IDLE_DEVICE_REQUIRED_NO_INDEPENDENT_AUDIT')
        bench = Bench(rt, args.bundle, manifest, point, l2['l2_bytes'], bool(args.profile_arm))
        result['fixture'] = bench.record
        if args.profile_arm:
            arm = args.profile_arm
            bench.configure(1); bench.update(0)
            call = bench.launch(arm)
            for _ in range(5):
                checked(call(), 'excluded profile warmup')
            rt.sync(); bench.check()
            with AcuRange(rt):
                checked(call(), 'profile complete chain'); rt.sync()
            result.update(status='PASS', arm=arm, numerics=bench.check(),
                          scope='ACU_KERNEL_REPLAY_NOT_UNPROFILED_CHAIN_TIMING')
            return 0
        for arm in ARMS:
            try:
                result['numerics'][arm] = numerical(bench, arm)
            except ValueError as error:
                # A finite-value/guard mismatch rejects this candidate, not
                # the other independent arms. A device runtime error instead
                # escapes to the point boundary; never reuse a poisoned GPU.
                result['failures'][arm] = str(error)
                traceback.print_exc()
            save(args.output, result)
            print(f'PAIRED_CHAIN_NUMERIC point={point.name} arm={arm} controls={len(result["numerics"].get(arm, []))} '
                  f'status={"PASS" if arm in result["numerics"] else "FAIL"} remaining_continue=1', flush=True)
        if 'unfused' not in result['numerics'] or not any(a in result['numerics'] for a in ARMS[1:]):
            raise ValueError('no valid unfused/paired numerical comparison')
        eligible = tuple(a for a in ARMS if a in result['numerics'])
        bench.configure(1); bench.update(0)
        for arm in eligible:
            calls = [bench.launch(arm, i) for i in range(bench.copies)]
            graphs[arm] = Graph(rt, calls*max(1, (32+len(calls)-1)//len(calls)))
            bench.check()
        samples = {a:[] for a in eligible}
        start = time.monotonic()
        for r in range(ROUNDS):
            bench.update(r)
            for arm in (eligible if r % 2 == 0 else tuple(reversed(eligible))):
                samples[arm].append([graphs[arm].sample() for _ in range(SAMPLES)])
                bench.check()
            elapsed = time.monotonic()-start
            print(f'PAIRED_CHAIN_CONFIRM point={point.name} round={r+1}/{ROUNDS} elapsed_s={elapsed:.1f} '
                  f'remaining_s={elapsed/(r+1)*(ROUNDS-r-1):.1f} scope=CONFIRMATION_ONLY', flush=True)
        summary = {arm:summarize(s) for arm, s in samples.items()}
        baseline = summary['unfused']['median_us']
        for arm, row in summary.items():
            row.update(delta_pct=100*(row['median_us']/baseline-1), stages=stages(arm),
                       effective_weight_MBU_pct=100*bench.weight_bytes/row['median_us']/2.7e6)
        best_paired = min(eligible[1:], key=lambda a:summary[a]['median_us'])
        base_ptrs = bench.rings['paired'].at(0)
        bases = {'A':bench.input.ptr % 128, **{k:(v or 0) % 128 for k, v in base_ptrs.items()}}
        result.update(status='PARTIAL' if result['failures'] else 'PASS', summary=summary, best_paired=best_paired,
                      best_arm=min(eligible, key=lambda a:summary[a]['median_us']),
                      access=[access(point, c, bases) for c in candidates(point)])
        print('PAIRED_CHAIN_RESULT '+json.dumps(dict(point=point.name, best=result['best_arm'],
              paired=best_paired, unfused_us=baseline, paired_us=summary[best_paired]['median_us'],
              delta_pct=summary[best_paired]['delta_pct'])), flush=True)
        return int(bool(result['failures']))
    except Exception as error:
        result['error'] = str(error); traceback.print_exc(); return 1
    finally:
        try:
            for graph in graphs.values():
                graph.close()
            if rt:
                rt.close()
        except Exception as error:
            result.update(status='FAIL', cleanup_error=str(error))
            traceback.print_exc()
        save(args.output, result)


def source_receipt():
    paths = ('dev/tp2_chain/run.py', 'dev/tp2_chain/bench.py', 'dev/tp2_chain/plan.py',
             'dev/gemv_model/fixture.py', 'dev/gemv_model/run.py', 'dev/gemv_model/access.py',
             'dev/gemv_simt/native.py', 'dev/gemv_simt/production.py',
             'dev/bf16_compute/fixture.py', 'dev/bf16_compute/native.py',
             'dev/gate_up_perf/bench.py', 'dev/gate_up_perf/plan.py',
             'quactlize/execution/native.py', 'quactlize/execution/simt_codegen.py',
             'quactlize/fusion/native.py', 'quactlize/dispatch/native.py',
             'tools/run_kpack_pack_gate.py', 'tools/kpack_warmup_fixture.py',
             'tools/profile_kpack_gpu_compact.py', 'reference/gguf_kpack.py')
    return {p:sha(ROOT/p) for p in paths}


def run_point(args, point, device, device_receipt):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
    base = [sys.executable, '-u', str(Path(__file__).resolve()), '--sdk', str(args.sdk), '--bundle', str(args.bundle),
            '--point', point.name, '--l2-bytes', str(args.l2_bytes)]
    dest = args.output/(point.name+'.json')
    identity_file = args.output/(point.name+'.identity.json')
    identity = dict(manifest=sha(args.bundle/'manifest.json'), device=device_receipt, l2=args.l2_bytes,
                    runner=source_receipt(), runtime={f'lib{x}.so':sha(args.sdk/'lib'/f'lib{x}.so') for x in LIBRARIES})
    if identity_file.exists():
        if not args.resume or json.loads(identity_file.read_text()) != identity:
            raise ValueError('point resume identity differs')
    else:
        save(identity_file, identity)
    old = json.loads(dest.read_text()) if dest.exists() else {}
    def run(command, log):
        # A separate environment per subprocess; never mutate process-global
        # CUDA_VISIBLE_DEVICES while the other point is running.
        start = time.monotonic()
        with log.open('x') as out:
            process = subprocess.Popen(command, env=env, stdout=out, stderr=subprocess.STDOUT)
            offset = 0
            while process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                with log.open(errors='replace') as src:
                    src.seek(offset)
                    for line in src:
                        if line.startswith('PAIRED_CHAIN_'):
                            print(line.rstrip(), flush=True)
                    offset = src.tell()
                if process.poll() is None:
                    print(f'PAIRED_CHAIN_WAIT point={point.name} device={device} elapsed_s={time.monotonic()-start:.0f} log={log}', flush=True)
            return process.returncode
    def fresh(suffix):
        return fresh_path(args.output, point.name+suffix)
    if args.resume and old.get('status') == 'PASS':
        rc = 0
    else:
        if dest.exists():
            dest.rename(fresh('.previous.json'))
        rc = run(base+['--output', str(dest)], fresh('.log'))
    data = json.loads(dest.read_text()) if dest.exists() else {}
    row = dict(point=point.name, device=device, status=data.get('status','FAIL'), rc=rc, profiles=[])
    if row['status'] not in ('PASS','PARTIAL') or 'summary' not in data:
        return row
    for arm in ('unfused', data['best_paired']):
        receipt_path = args.output/(point.name+'-'+arm+'.profile.json')
        if args.resume and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['status']=='PASS' and sha(args.output/receipt['report'])==receipt['sha256']:
                profiles((args.output/receipt['csv']).read_text(), point, arm)
                row['profiles'].append(receipt); continue
        stem = fresh('-'+arm+'.acu')
        report, numeric, raw = (Path(str(stem)+s) for s in ('.acurep','.json','.csv'))
        entry = dict(arm=arm, status='FAIL', report=report.name)
        try:
            command = acu_launch_command(args.sdk/'asight/bin/acu', stem, base+['--output',str(numeric),'--profile-arm',arm])
            save(Path(str(stem)+'.command.json'), command)
            code = run(command, Path(str(stem)+'.log'))
            proof = json.loads(numeric.read_text())
            if code or proof['status']!='PASS' or proof['manifest_sha256']!=data['manifest_sha256'] or proof['device']!=data['device']:
                raise ValueError('profile result/identity differs')
            if proof['fixture']['raw_sha256'] != data['fixture']['raw_sha256']:
                raise ValueError('profile does not use the timed GGUF fixture')
            checked(run([str(args.sdk/'asight/bin/acu'),'--import',str(report),'--page','raw','--csv'], raw), 'ACU export')
            entry.update(status='PASS', kernels=profiles(raw.read_text(),point,arm), sha256=sha(report), csv=raw.name)
        except Exception as error:
            entry['error']=str(error); row['status']='PARTIAL'; traceback.print_exc()
        save(receipt_path, entry); row['profiles'].append(entry)
    row.update(summary=data['summary'], best_arm=data['best_arm'])
    return row


def collect(args):
    verify(args.bundle, 'tp2-chain')
    devices = args.devices.split(',')
    if len(devices)!=2 or len(set(devices))!=2 or any(not re.fullmatch(r'\d+', d) for d in devices):
        raise ValueError('two distinct single-device ordinals required; use 6,7')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    # Confirm physical identity in fresh children before assigning timed work.
    identities=[]
    for device in devices:
        command=[sys.executable,str(Path(__file__).resolve()),'--sdk',str(args.sdk),'--probe']
        text=subprocess.check_output(command, env=dict(os.environ,CUDA_VISIBLE_DEVICES=device), text=True)
        found=[json.loads(l.split(' ',1)[1]) for l in text.splitlines() if l.startswith('PAIRED_CHAIN_DEVICE ')]
        if len(found)!=1:raise ValueError('device identity probe missing')
        identities.append(found[0])
    if identities[0]['pci']==identities[1]['pci']:
        raise ValueError('workers resolve to the same physical device')
    save(args.output/'devices.json', identities)
    records=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_point,args,p,d,i) for p,d,i in zip(POINTS,devices,identities)]
        for point, future in zip(POINTS,futures):
            try:row=future.result()
            except Exception as error:row=dict(point=point.name,status='FAIL',error=str(error));traceback.print_exc()
            records.append(row)
    passed=all(r['status']=='PASS' for r in records)
    save(args.output/'summary.json', dict(status='PASS' if passed else 'INCOMPLETE', records=records,
         manifest_sha256=sha(args.bundle/'manifest.json'), scope='FULL_MOE_CHAIN_NOT_MODEL_TPOT',
         production_selection_changed=False, device_admission='PENDING_REVIEW'))
    print(f'PAIRED_CHAIN_DONE status={"PASS" if passed else "INCOMPLETE"} results={args.output}',flush=True)
    return 0 if passed else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True);p.add_argument('--bundle',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--point',choices=[p.name for p in POINTS])
    p.add_argument('--profile-arm',choices=ARMS);p.add_argument('--devices',default='6,7')
    p.add_argument('--l2-bytes',type=int,default=67108864);p.add_argument('--resume',action='store_true')
    p.add_argument('--probe',action='store_true')
    a=p.parse_args()
    if a.probe:
        rt=Runtime(a.sdk,'ppu')
        try:print('PAIRED_CHAIN_DEVICE '+json.dumps(device_identity(rt)),flush=True)
        finally:rt.close()
        return 0
    if not a.bundle or not a.output:p.error('--bundle and --output required')
    return child(a) if a.point else collect(a)


if __name__=='__main__':raise SystemExit(main())
