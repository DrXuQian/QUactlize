#!/usr/bin/env python3
"""Execute the prebuilt indexed Q4 S1 gate; successful cases are resumable."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_cuda.build import digest
from dev.gemv_ppu import moe_s1 as spec
from dev.gemv_ppu.moe_s1_bench import Bench,a_model
from tools.profile_kpack_gpu_compact import acu_launch_command


def cases():
    return [(t,ch,False) for t in spec.TOKENS for ch in (1,8)]+[(8,ch,True) for ch in (1,8)]


def case_id(n,k,t,ch,cluster): return f'n{n}-k{k}-t{t}-ch{ch}-'+('cluster' if cluster else 'spread')


def fingerprint(args,manifest):
    files=['dev/gemv_ppu/run_moe_s1.py','dev/gemv_ppu/moe_s1_bench.py',
           'tools/kpack_execution_fixture.py','tools/kpack_warmup_fixture.py',
           'tools/run_kpack_gemv_gate.py','tools/run_kpack_grouped_device_gate.py',
           'tools/run_kpack_grouped_decode_probe.py','dev/gemv_ppu/run.py',
           'quactlize/execution/native.py','quactlize/runtime/native.py',
           'reference/gguf_kpack.py','tools/profile_kpack_gpu_compact.py',
           'dev/gemv_ppu/small_latency.py','dev/gemv_ppu/reader_reuse.py']
    return dict(schema=spec.SCHEMA,manifest_sha256=digest(args.bundle/'manifest.json'),
        harness_sha256={p:digest(ROOT/p) for p in files},samples=15,rounds=6,screen_samples=5,
        l2_override=args.l2_bytes,cases_per_shape=len(cases()),production_changed=False)


def validate(row,identity,n,k,t,ch,cluster):
    if row.get('identity')!=identity or row.get('case')!=case_id(n,k,t,ch,cluster) or row.get('status')!='PASS':
        raise ValueError('case identity/status differs')
    if (row.get('shape')!=[t*8,n,k] or row.get('tokens')!=t or row.get('channels')!=ch or
        row.get('topk')!=8 or row.get('experts')!=256 or row.get('router')!=('cluster' if cluster else 'spread')):
        raise ValueError('row/token/expert case denominator differs')
    expected={r.key for r in spec.inventory()}
    if len(row['screen'])!=len(expected) or {x['key'] for x in row['screen']}!=expected:
        raise ValueError('screen recipe denominator differs')
    for x in row['screen']:
        proof=x['correctness']
        if not (0<=proof['error']<.005): raise ValueError('numerical admission missing')
        for p in ('fp16_fp32_bits','indexed_grouped_bits','indexed_dense_bits','zero_a','zero_codes','invalid_ids','output_guard','gpu_ids_replay'):
            if proof.get(p)!='PASS': raise ValueError('missing numerical control: '+p)
        if (n,k)==(512,2048) and spec.lookup(x['key']).reader==0:
            frozen=json.loads((ROOT/'prebuilt/ppu0010/q4-smallm-v1/manifest.json').read_text())['payloads']['libq4_smallm_n512_k2048.so']['sha256']
            if proof.get('immutable_dense')!=frozen: raise ValueError('immutable dense comparison missing')
        if len(x['samples_us'])!=5: raise ValueError('screen sample denominator differs')
    import math
    for x in row['screen']+row['confirmation']:
        samples=x['samples_us']
        if not samples or any(not math.isfinite(v) or v<=0 for v in samples): raise ValueError('invalid timing')
        if x['median_us']!=statistics.median(samples): raise ValueError('median differs')
    finalists=[x['key'] for x in sorted(row['screen'],key=lambda x:(x['median_us'],x['key']))[:2]]
    if len(row['confirmation'])!=12 or {(x['round'],x['key']) for x in row['confirmation']}!={(r,key) for r in range(6) for key in finalists}:
        raise ValueError('confirmation denominator differs')
    for x in row['confirmation']:
        if len(x['samples_us'])!=15: raise ValueError('confirmation samples differ')
    medians={key:statistics.median(x['median_us'] for x in row['confirmation'] if x['key']==key) for key in finalists}
    if row['selected']!=min(medians,key=lambda key:(medians[key],key)) or row['medians_us']!=medians:
        raise ValueError('winner differs')
    if row['copies']*8*(n*k*9//16)<2.25*row['device']['l2_bytes'] or row['calls_per_graph']%row['copies']:
        raise ValueError('active-expert cold ring incomplete')
    if row.get('launches_per_call')!=1 or row.get('scope')!='INDEXED_S1_NO_GATHER_SCATTER_NOT_CHAIN':
        raise ValueError('timing scope differs')
    return row


def child(args,manifest):
    import numpy as np
    n,k=args.shape
    identity=fingerprint(args,manifest)
    bench=None
    try:
        bench=Bench(args,n,k)
        for index,(t,ch,cluster) in enumerate(cases()):
            key=case_id(n,k,t,ch,cluster); path=args.output/(key+'.json')
            if args.profile and (t,ch,cluster) not in ((1,1,False),(8,8,True)): continue
            if path.exists():
                old=validate(json.loads(path.read_text()),identity,n,k,t,ch,cluster)
                if old['device']!=bench.device: raise ValueError('resume physical device differs')
                if not args.profile:
                    print(f'Q4_MOE_S1_RESUME case={key} status=PASS',flush=True); continue
            elif args.profile: raise ValueError('profile needs a completed numeric/timing case')
            bench.setup(t,ch,cluster)
            if args.profile:
                bench.case_r.fill(bench.out_base,0xa5,bench.output_bytes+32)
                bench.profile(old['selected'])
                print('Q4_MOE_S1_PROFILE '+json.dumps(dict(case=key,recipe=old['selected'],status='PASS',scope='ACU_FORCED_COLD')),flush=True)
                continue
            started=time.monotonic(); screen=[]
            for c in spec.inventory():
                proof=bench.correctness(c.key)
                samples=bench.measure(c.key,5)
                screen.append(dict(key=c.key,correctness=proof,samples_us=samples,median_us=statistics.median(samples)))
                print(f'Q4_MOE_S1_SCREEN case={key} recipes={len(screen)}/{len(spec.inventory())} key={c.key} median_us={statistics.median(samples):.6f}',flush=True)
            winners=[x['key'] for x in sorted(screen,key=lambda x:(x['median_us'],x['key']))[:2]]
            confirm=[]
            for rr in range(6):
                for recipe in winners[::1 if rr%2==0 else -1]:
                    values=bench.measure(recipe,15)
                    confirm.append(dict(key=recipe,round=rr,samples_us=values,median_us=statistics.median(values)))
            medians={key:statistics.median(x['median_us'] for x in confirm if x['key']==key) for key in winners}
            selected=min(medians,key=lambda key:(medians[key],key))
            row=dict(identity=identity,case=key,shape=[t*8,n,k],tokens=t,topk=8,channels=ch,experts=256,
                router='cluster' if cluster else 'spread',active_experts=int(np.unique(bench.data['expert']).size),
                max_rows=int(np.bincount(bench.data['expert']).max()),status='PASS',screen=screen,confirmation=confirm,
                medians_us=medians,selected=selected,device=bench.device,copies=bench.copies,calls_per_graph=bench.calls_per_graph,
                weight_sha256=bench.weight_sha256,a_sha256=hashlib.sha256(bench.data['a']).hexdigest(),
                ids_sha256=hashlib.sha256(bench.data['ids']).hexdigest(),elapsed_seconds=time.monotonic()-started,
                launches_per_call=1,scope='INDEXED_S1_NO_GATHER_SCATTER_NOT_CHAIN',input_type='F32_ROUNDED_TO_F16',output_type='F32',
                cache='ROTATING_ACTIVE_EXPERT_WEIGHTS_AT_LEAST_2_25_L2',first_launch='EXCLUDED',
                access=[dict(key=c.key,model=spec.access(c,n,k,t*8,1,dict(A=bench.a%128,B=bench.base.low%128,metadata=bench.base.units%128)),
                            indexed_a_models=[a_model(c,1,base) for base in sorted({
                                (bench.a+4*((rr//8)*bench.base.a_token_stride+(rr%8%ch)*bench.base.a_row_stride))%128
                                for rr in range(t*8)})]) for c in spec.inventory()])
            validate(row,identity,n,k,t,ch,cluster)
            path.write_text(json.dumps(row,indent=2)+'\n')
            print('Q4_MOE_S1_RESULT '+json.dumps(dict(case=key,selected=selected,median_us=medians[selected],
                completed=index+1,total=len(cases()),seconds=row['elapsed_seconds'])),flush=True)
        return 0
    finally:
        if bench: bench.close()


def main(args):
    manifest=spec.verify(args.bundle)
    # Execution needs the runtime, not the exact compiler/inspector executable.
    # Keep the previous SDK packaging fix: compare actual runtime library bytes.
    for name,sha in manifest['runtime'].items():
        if digest(args.sdk/'lib'/name)!=sha: raise ValueError('runtime differs: '+name)
    if args.child: return child(args,manifest)
    args.output.mkdir(parents=True,exist_ok=True)
    authority=fingerprint(args,manifest)
    ap=args.output/'authority.json'
    if ap.exists() and json.loads(ap.read_text())!=authority: raise ValueError('resume authority differs')
    ap.write_text(json.dumps(authority,indent=2)+'\n')
    failures=[]; profiles=[]; began=time.monotonic()
    for n,k in spec.SHAPES:
        label=f'n{n}-k{k}'
        cmd=[sys.executable,'-u',__file__,'--child','--shape',str(n),str(k),'--bundle',str(args.bundle),
             '--sdk',str(args.sdk),'--output',str(args.output),'--l2-bytes',str(args.l2_bytes)]
        log=args.output/(label+'.log')
        with log.open('a') as f:
            p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in p.stdout: f.write(line); f.flush(); print(line,end='',flush=True)
            rc=p.wait()
        if rc: failures.append(dict(shape=[n,k],phase='gate',rc=rc,log=log.name)); continue
        if args.acu:
            receipt=args.output/(label+'.acu.json')
            prior=json.loads(receipt.read_text()) if receipt.exists() else None
            if prior and prior.get('identity')==authority and prior.get('status')=='PASS' and all(
                    digest(args.output/prior[x])==prior[x+'_sha256'] for x in ('report','log')):
                profiles.append(prior)
                print(f'Q4_MOE_S1_ACU existing={label} verified=1',flush=True)
            else:
                stamp=str(time.time_ns())
                log=args.output/(label+'.acu-'+stamp+'.log'); report=args.output/(label+'.acu-'+stamp+'.acurep')
                with log.open('w') as f:
                    rc=subprocess.run(acu_launch_command(args.acu,report,cmd+['--profile']),stdout=f,stderr=subprocess.STDOUT).returncode
                reports=[p for p in args.output.glob(report.name+'*') if p.is_file() and p.stat().st_size]
                entries=[json.loads(line.split(' ',1)[1]) for line in log.read_text(errors='replace').splitlines() if line.startswith('Q4_MOE_S1_PROFILE ')]
                expected={case_id(n,k,1,1,False),case_id(n,k,8,8,True)}
                if rc or len(reports)!=1 or len(entries)!=2 or {x['case'] for x in entries}!=expected or any(x['status']!='PASS' for x in entries):
                    failures.append(dict(shape=[n,k],phase='acu',rc=rc,log=log.name,error='profile process/report/receipts incomplete'))
                else:
                    data=dict(status='PASS',identity=authority,shape=[n,k],entries=entries,
                        report=reports[0].name,report_sha256=digest(reports[0]),log=log.name,log_sha256=digest(log))
                    receipt.write_text(json.dumps(data,indent=2)+'\n'); profiles.append(data)
    rows=[]
    for n,k in spec.SHAPES:
        for t,ch,cluster in cases():
            path=args.output/(case_id(n,k,t,ch,cluster)+'.json')
            if not path.exists(): continue
            rows.append(validate(json.loads(path.read_text()),authority,n,k,t,ch,cluster))
    with (args.output/'summary.tsv').open('w') as f:
        fields=['case','tokens','channels','active_experts','max_rows','selected','median_us','status']
        w=csv.DictWriter(f,fields,delimiter='\t'); w.writeheader()
        for row in rows: w.writerow({k:row[k] for k in fields if k!='median_us'}|dict(median_us=row['medians_us'][row['selected']]))
    result=dict(status='PASS' if not failures and len(rows)==len(spec.SHAPES)*len(cases()) else 'INCOMPLETE',
        cases=len(rows),expected=len(spec.SHAPES)*len(cases()),failures=failures,profiles=profiles,
        seconds=time.monotonic()-began,device_admission='PENDING_REVIEW',production_changed=False,
        files={p.name:digest(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name not in ('result.json','console.log')})
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Q4_MOE_S1_DONE '+json.dumps(result|dict(files='IN_RESULT_JSON')),flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--bundle',type=Path,default=ROOT/'prebuilt/ppu0010/q4-moe-s1-v1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--l2-bytes',type=int,default=0)
    p.add_argument('--acu',type=Path)
    p.add_argument('--child',action='store_true')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--shape',nargs=2,type=int)
    a=p.parse_args()
    if a.child and tuple(a.shape or ()) not in spec.SHAPES: p.error('child requires a declared shape')
    try: raise SystemExit(main(a))
    except Exception:
        traceback.print_exc(); raise SystemExit(1)
