#!/usr/bin/env python3
"""One bounded follow-up to the completed model Asys, with immutable controls."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.gemv_simt.model_followup import POINTS
from dev.gemv_simt.run_model_followup import verify_candidate
from quactlize.runtime.compiler import sha
from tools.profile_kpack_model_decode import profile_plan
from tools.verify_kpack_dispatch import verify


def save(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def option(argv,key):
    if argv.count(key)!=1 or argv.index(key)+1==len(argv):
        raise ValueError('previous trace has no unique '+key)
    return Path(argv[argv.index(key)+1]).resolve(strict=True)


def validate_plan(rows):
    simt=[r for r in rows if r['point']['kind'] in ('simt','q4')]
    expected=[]
    for p in POINTS:
        expected.append(dict(q=p.q,n=p.n,k=p.k,mode=p.mode,channels=p.channels,compute=p.compute,
            tokens=1,experts=256 if p.mode else 1,topk=8 if p.mode else 1,kind='simt',**asdict(p.config)))
    keys=tuple(expected[0])
    actual=[{k:r['point'][k] for k in keys} for r in simt]
    if sorted(actual,key=str)!=sorted(expected,key=str):
        raise ValueError('current Asys SIMT shape/compute/config union differs from the five compiled points')
    tc=[r for r in rows if r['point']['kind']=='tc']
    if {(r['point']['q'],r['point']['n'],r['point']['k'],r['point']['split'],r['point']['compute']) for r in tc} != {
            (8,8192,2048,8,0),(8,4096,2048,8,0),(14,248320,2048,1,0)} or len(tc)!=3:
        raise ValueError('current Asys TC denominator differs from the three missing profiles')


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    argv=json.loads((a.previous/'results/trace.command.json').read_text())['argv']
    bundle=option(argv,'--bundle');cache=option(argv,'--jit-cache')
    # Do not substitute the live checkout's latest bundle or old all-BF16 run.
    continuation=json.loads((a.previous/'results/continuation.json').read_text())
    if sha(bundle/'manifest.json')!=continuation['runtime_manifest_sha256']:
        raise ValueError('previous model runtime receipt differs')
    m=verify(bundle,sdk=a.sdk)
    candidate=verify_candidate(a.candidate,bundle)
    save(a.output/'runtime-manifest.json',m)
    save(a.output/'candidate-manifest.json',candidate)
    sys.path.insert(0,str(a.llama/'tests'))
    from quactlize_native import simt_symbol_recipe,q4_symbol_recipe,q4_symbol_matches_plan
    trace=a.previous/'results/trace'
    native=trace/'qwen35-35b-q4km/native'
    proof=json.loads((native/'proof.json').read_text())
    if proof.get('missing_ops') or proof.get('kernel_execution')!='PASS_SHORT_REQUEST' or \
            proof.get('capture_scope')!='SECOND_REQUEST_SAME_PROCESS_FIRST_USE_EXCLUDED':
        raise ValueError('previous model trace did not cover a complete warmed native request')
    files=[native/'proof-request/0-native.selection.json',native/'kernel-times.json',
           native/'proof-request/0-native.log',native/'proof.json']
    rows=profile_plan(json.loads(files[0].read_text()),json.loads(files[1].read_text()),files[2].read_text(),
        (simt_symbol_recipe,q4_symbol_recipe,q4_symbol_matches_plan),proof['matched'])
    validate_plan(rows)
    save(a.output/'plan.json',dict(previous=str(a.previous),bundle=str(bundle),jit_cache=str(cache),
        jobs=rows,trace_receipts_sha256={str(p.relative_to(a.previous)):sha(p) for p in files},
        execution_sha256=m['execution_sha256'],candidate_manifest_sha256=sha(a.candidate/'manifest.json'),
        candidate_records=candidate['records'],production_selection_changed=False,
        targets=dict(small_weight_bytes_lt=2*1024**2,small_MBU_pct=40,large_MBU_pct=60,peak_GBs=2700),
        scope='M1_CURRENT_MODEL_FIVE_SIMT_AB_AND_THREE_TC_PROFILES'))
    print('MODEL_MBU_PLAN PASS tc=3 simt=5 candidate_kernels=14 production=UNCHANGED',flush=True)
    if a.plan_only:return 0
    phases=[('tc-acu',[sys.executable,'-u',str(ROOT/'tools/profile_kpack_model_decode.py'),
        '--bundle',str(bundle),'--sdk',str(a.sdk),'--llama',str(a.llama),'--trace',str(trace),
        '--jit-cache',str(cache),'--acu',str(a.acu),'--kind','tc','--output',str(a.output/'tc-acu')]),
        ('simt-ab',[sys.executable,'-u',str(ROOT/'dev/gemv_simt/run_model_followup.py'),
        '--bundle',str(a.candidate),'--shipping',str(bundle),'--sdk',str(a.sdk),'--acu',str(a.acu),
        '--l2-bytes',str(a.l2_bytes),'--output',str(a.output/'simt-ab')])]
    records=[]
    for name,command in phases:
        save(a.output/(name+'.command.json'),command)
        start=time.monotonic()
        print(f'MODEL_MBU_PHASE phase={name} START',flush=True)
        # Inherit stdout so both child progress and errors remain visible.
        rc=subprocess.run(command).returncode
        records.append(dict(phase=name,rc=rc,seconds=time.monotonic()-start))
        save(a.output/'status.json',dict(complete=False,phases=records))
        print(f'MODEL_MBU_PHASE phase={name} rc={rc} remaining_continue=1',flush=True)
    status='PASS' if all(r['rc']==0 for r in records) else 'FAIL'
    save(a.output/'status.json',dict(status=status,complete=True,phases=records,
        meaning='DIAGNOSTIC_COMPLETE_NOT_PERFORMANCE_TARGET_ADMISSION',production_selection_changed=False))
    print(f'MODEL_MBU_DONE status={status} results={a.output}',flush=True)
    return int(status!='PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','sdk','llama','candidate','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--acu',type=Path)
    p.add_argument('--l2-bytes',type=int,default=67108864,help='operator-verified PPU-ZW810 L2')
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    if not a.plan_only and (a.acu is None or not a.acu.is_file()):p.error('ACU is required')
    raise SystemExit(main(a))
