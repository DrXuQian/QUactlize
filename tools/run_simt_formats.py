#!/usr/bin/env python3
"""Bounded all-format SIMT timing cohort; no JIT, model run or policy update."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.gemv_simt import spec

FORMATS = (10, 11, 13, 14, 8)
FAMILIES = (
    ('dense-small', 'dense', 512, 2048, 1),
    ('dense-medium', 'dense', 1024, 5120, 1),
    ('dense-large', 'dense', 5120, 8192, 1),
    ('moe-gate-up', 'indexed', 1024, 2048, 1),
    ('moe-down', 'indexed', 2048, 512, 8),
)
SCHEMA = 'quactlize.simt-format-sweep.v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def cases(qtypes, tokens):
    if (not qtypes or len(set(qtypes)) != len(qtypes) or not set(qtypes) <= set(FORMATS) or
            not tokens or len(set(tokens)) != len(tokens) or not set(tokens) <= set(range(1, 9))):
        raise ValueError('format/token inventory differs')
    return [dict(id=f'q{q}-{name}-t{t}', qtype=q, mode=mode, n=n, k=k, tokens=t,
                 channels=ch, experts=1 if mode=='dense' else 256)
            for q in qtypes for name, mode, n, k, ch in FAMILIES for t in tokens]


def verify(bundle):
    m = json.loads((bundle/'manifest.json').read_text())
    if m.get('schema') != spec.SCHEMA or m.get('platform') != 'ppu' or m.get('profile') != 'full':
        raise ValueError('requires the full prebuilt PPU SIMT inventory')
    for q in spec.QTYPES:
        if m['configs'][str(q)] != [c.record() for c in spec.runtime_inventory(q)]:
            raise ValueError('compiled configuration inventory differs')
    for name, expected in ((m['library'], m['library_sha256']),
                           (m['baseline']['library'], m['baseline']['sha256'])):
        p = bundle/name
        if Path(name).name != name or p.is_symlink() or sha(p) != expected:
            raise ValueError('prebuilt image differs: '+name)
    for name, expected in m['source_hashes'].items():
        if not (ROOT/name).resolve().is_relative_to(ROOT) or sha(ROOT/name) != expected:
            raise ValueError('compiled source receipt differs: '+name)
    return m


PROBE = '''
import ctypes as C,json,sys
from dev.gemv_simt.native import Library,Runtime,checked
rt=Runtime(sys.argv[1],'ppu'); lib=Library(__import__('pathlib').Path(sys.argv[2]))
identity=lib.probe()
fn=rt.lib.hggcDeviceGetPCIBusId;fn.argtypes=[C.c_char_p,C.c_int,C.c_int];fn.restype=C.c_int
pci=C.create_string_buffer(64);checked(fn(pci,len(pci),identity['ordinal']),'PCI identity')
identity['pci']=pci.value.decode();rt.close()
print('SIMT_PROBE '+json.dumps(identity))
'''


def candidate_keys(q):
    new = {c.key for c in spec.runtime_inventory(q)}
    old = {f'old-pair-c{c}-w{w}-s{s}'
           for c, w, s in product((16, 32), (2, 4, 8), spec.SPLITS)}
    if q != 8:
        old |= {f'old-scalar-c{c}-w{w}-s{s}'
                for c, w, s in product((16, 32), (4, 8), (1, 4))}
    return {'new': new, 'old': old}


def active_experts(point):
    if point['mode'] == 'dense':
        return 1
    return len({(slot*3+token*5) % point['experts']
                for token in range(point['tokens']) for slot in range(8)})


def validate_devices(devices):
    if not devices or len({p['pci'] for p in devices}) != len(devices):
        raise ValueError('workers resolved to the same physical device')
    if any(not p['pci'] or p['marker'] != 'PASS' for p in devices):
        raise ValueError('missing physical device or same-image launch proof')


def result_valid(path, point, authority):
    try:
        d=json.loads(path.read_text()); r=d['result']
        valid = (d['status']=='PASS' and r['status']=='PASS' and d['phase']=='perf' and
                d['manifest_sha256']==authority['bundle_sha256'] and
                d['qtype']==point['qtype'] and r['production_admitted'] is False and
                all(r[k]==point[k] for k in ('qtype','mode','n','k','tokens','channels','experts')) and
                r['cache']=='rotating' and r['l2_bytes']==authority['l2_bytes'] and
                r['active_experts']==active_experts(point) and r['ring_copies']>0 and
                r['weight_bytes']>0 and
                r['weight_bytes']*r['experts']==r['resident_weight_bytes']*r['active_experts'] and
                r['ring_bytes']==r['weight_bytes']*r['ring_copies'] and
                r['allocated_ring_bytes']==r['resident_weight_bytes']*r['ring_copies'] and
                r['ring_bytes']>=2.25*r['l2_bytes'] and r['calls_per_graph']>=32 and
                r['calls_per_graph']%r['ring_copies']==0 and
                r['replay_proof']['replays']==3 and
                r['replay_proof']['negative']=='ZERO_A_REJECTED' and
                len(r['replay_proof']['errors'])==3 and
                all(math.isfinite(e) and 0<=e<0.005 for e in r['replay_proof']['errors']))
        if not valid:
            return False
        keys = candidate_keys(point['qtype'])
        expected = {(arm, key) for arm, values in keys.items() for key in values}
        if (len(r['screen']) != len(expected) or
                {(x['arm'], x['key']) for x in r['screen']} != expected):
            return False
        finalists = {(arm, x['key']) for arm in keys
                     for x in sorted((x for x in r['screen'] if x['arm']==arm),
                                     key=lambda x:x['median_us'])[:2]}
        confirmations = {(arm, key, repeat) for arm, key in finalists
                         for repeat in range(authority['rounds'])}
        if (len(r['confirmation']) != len(confirmations) or
                {(x['arm'], x['key'], x['round']) for x in r['confirmation']} != confirmations):
            return False
        for phase, count in (('screen', 3), ('confirmation', authority['samples'])):
            for row in r[phase]:
                samples = row['samples_us']
                if (len(samples)!=count or not all(math.isfinite(t) and t>0 for t in samples) or
                        not math.isfinite(row['error']) or not 0<=row['error']<0.005 or
                        not math.isclose(row['median_us'], statistics.median(samples), rel_tol=1e-12)):
                    return False
        if set(r['best']) != set(keys):
            return False
        for arm in keys:
            medians = {key: statistics.median(t for x in r['confirmation']
                        if (x['arm'],x['key'])==(arm,key) for t in x['samples_us'])
                       for a, key in finalists if a==arm}
            best=r['best'][arm]
            if (best['key'] not in medians or
                    not math.isclose(best['median_us'], medians[best['key']], rel_tol=1e-12) or
                    not math.isclose(best['median_us'], min(medians.values()), rel_tol=1e-12)):
                return False
        return True
    except (OSError, KeyError, ValueError, TypeError, ZeroDivisionError):
        return False


def run(a):
    a.bundle=a.bundle.resolve(strict=True); a.output=a.output.resolve()
    manifest=verify(a.bundle); plan=cases(a.qtypes,a.tokens)
    if len(set(a.devices))!=len(a.devices) or any(not d.isdigit() for d in a.devices):
        raise ValueError('devices must be distinct numeric ordinals')
    if a.plan_only:
        print(json.dumps(dict(cases=plan, configs={str(q):len(spec.runtime_inventory(q)) for q in a.qtypes}),indent=2))
        return 0
    a.output.mkdir(parents=True,exist_ok=True)
    identity=[]
    for device in a.devices:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=device)
        p=subprocess.run([sys.executable,'-c',PROBE,str(a.sdk),str(a.bundle)],cwd=ROOT,env=env,
                         text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (a.output/f'probe-{device}.log').write_text(p.stdout)
        lines=[s.removeprefix('SIMT_PROBE ') for s in p.stdout.splitlines() if s.startswith('SIMT_PROBE ')]
        if p.returncode or len(lines)!=1: raise ValueError(f'device {device} probe failed; see probe-{device}.log')
        info=json.loads(lines[0]);info['visible']=device;identity.append(info)
    validate_devices(identity)
    authority=dict(schema=SCHEMA,plan=plan,devices=identity,rounds=a.rounds,samples=a.samples,
        l2_bytes=a.l2_bytes,bundle_sha256=sha(a.bundle/'manifest.json'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        runner_sha256=sha(__file__),runtime_sha256=sha(a.sdk/'lib/libhggc_wrapper.so'),
        scope='F32_ENDPOINT_SIMT_COMPLETE_CALL_NOT_TC_NOT_MODEL',precision='F16_ACTIVATIONS_F32_ACCUMULATOR',
        policy_changed=False)
    receipt=a.output/'authority.json'
    if receipt.exists():
        if json.loads(receipt.read_text())!=authority: raise ValueError('resume authority differs; use a new run')
    else: write(receipt,authority)
    print(f'SIMT_SWEEP_START cases={len(plan)} workers={len(a.devices)} tokens={a.tokens} compile=NONE jit=NONE',flush=True)

    def worker(index):
        assigned=plan[index::len(a.devices)];device=a.devices[index]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=device)
        durations=[];failed=0
        for done,point in enumerate(assigned):
            case=a.output/point['id'];case.mkdir(exist_ok=True)
            passes=[p for p in case.glob('attempt-*.json') if result_valid(p,point,authority)]
            if passes:
                print(f"SIMT_SWEEP_REUSE worker={device} case={point['id']}",flush=True);continue
            attempt=1
            while (case/f'attempt-{attempt}.json').exists() or (case/f'attempt-{attempt}.log').exists(): attempt+=1
            output=case/f'attempt-{attempt}.json';log=case/f'attempt-{attempt}.log'
            command=[sys.executable,'-u',str(ROOT/'dev/gemv_simt/run.py'),'--sdk',str(a.sdk),
                '--bundle',str(a.bundle),'--output',str(output),'--qtype',str(point['qtype']),
                '--phase','perf','--cache','rotating','--n',str(point['n']),'--k',str(point['k']),
                '--tokens',str(point['tokens']),'--mode',point['mode'],'--channels',str(point['channels']),
                '--experts',str(max(8,point['experts'])),'--l2-bytes',str(a.l2_bytes),
                '--rounds',str(a.rounds),'--samples',str(a.samples)]
            write(case/f'attempt-{attempt}.command.json',dict(argv=command,device=device))
            start=time.monotonic();last=start
            print(f"SIMT_SWEEP_CASE worker={device} case={point['id']} start={done+1}/{len(assigned)} log={log}",flush=True)
            with log.open('w') as stream:
                p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                try:
                    while p.poll() is None:
                        time.sleep(1)
                        now=time.monotonic()
                        if now-last>=30:
                            tail=log.read_text(errors='replace').splitlines()[-1:]
                            eta=sum(durations[-8:])/len(durations[-8:])*(len(assigned)-done)/60 if durations else None
                            print('SIMT_SWEEP_PROGRESS '+json.dumps(dict(worker=device,completed=done,total=len(assigned),
                                current=point['id'],case_minutes=(now-start)/60,
                                remaining_worker_minutes=eta,eta_advisory=True,last=tail)),flush=True)
                            last=now
                except BaseException:
                    p.terminate();p.wait();raise
            durations.append(time.monotonic()-start)
            ok=p.returncode==0 and result_valid(output,point,authority)
            failed+=not ok
            write(case/f'attempt-{attempt}.process.json',dict(rc=p.returncode,seconds=durations[-1],valid=ok))
            print(f"SIMT_SWEEP_CASE worker={device} case={point['id']} status={'PASS' if ok else 'FAIL'} seconds={durations[-1]:.1f} remaining_continue=1",flush=True)
        return failed

    with ThreadPoolExecutor(max_workers=len(a.devices)) as pool:
        list(pool.map(worker,range(len(a.devices))))
    rows=[];missing=[]
    for point in plan:
        paths=sorted((a.output/point['id']).glob('attempt-*.json'))
        good=[p for p in paths if result_valid(p,point,authority)]
        if not good: missing.append(point['id']);continue
        p=good[-1];r=json.loads(p.read_text())['result']
        for arm,v in r['best'].items():
            rows.append(point|dict(arm=arm,config=v['key'],median_us=v['median_us'],
                effective_weight_GBs=r['weight_bytes']/v['median_us']/1000,
                modeled_MBU_pct=r['weight_bytes']/v['median_us']/1000/2700*100,
                active_experts=r['active_experts'],result=str(p.relative_to(a.output)),sha256=sha(p)))
    if rows:
        with (a.output/'summary.tsv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    write(a.output/'summary.json',dict(status='PASS' if not missing else 'INCOMPLETE',cases=len(plan),
        passed=len(plan)-len(missing),failed=missing,policy_changed=False,MBU='WEIGHT_BYTES_MODEL_AT_2700_GBPS_NOT_ACU_COUNTER'))
    print(f'SIMT_SWEEP_DONE passed={len(plan)-len(missing)}/{len(plan)} results={a.output}',flush=True)
    return int(bool(missing))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--devices',nargs='+',default=['0'])
    p.add_argument('--qtypes',nargs='+',type=int,default=list(FORMATS));p.add_argument('--tokens',nargs='+',type=int,default=list(range(1,9)))
    p.add_argument('--rounds',type=int,default=4);p.add_argument('--samples',type=int,default=11)
    p.add_argument('--l2-bytes',type=int,required=True);p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    if a.rounds<2 or a.samples<3 or a.l2_bytes<=0:p.error('positive L2 and >=2 rounds, >=3 samples required')
    raise SystemExit(run(a))
