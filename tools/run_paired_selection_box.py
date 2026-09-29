#!/usr/bin/env python3
"""Prebuilt paired recipes, incremental caller CI, model correctness and warmed Asys."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quactlize.runtime.compiler import sha
from tools.resolve_kpack_batched_models import resolve_plan
from tools.verify_kpack_dispatch import verify
from tools.build_kpack_model_ci import local_source_state


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def fetch(pin,root):
    if (not re.fullmatch(r'[0-9a-f]{40}',pin['commit']) or
        pin['branch']!='artifacts/kpack-paired-recipes-v2' or
        pin['path']!='prebuilt/ppu0010/kpack-model-runtime-v1'):
        raise ValueError('invalid paired artifact pin')
    target=root/('quactlize-paired-artifact-'+pin['commit'][:12])
    subprocess.run(['git','fetch','origin',pin['branch']],cwd=ROOT,check=True)
    if not target.exists():
        subprocess.run(['git','worktree','add','--detach',target,pin['commit']],cwd=ROOT,
            env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1'),check=True)
    if subprocess.check_output(['git','-C',target,'rev-parse','HEAD'],text=True).strip()!=pin['commit']:
        raise ValueError('artifact worktree identity differs')
    subprocess.run(['git','-C',target,'lfs','pull','origin','--include='+pin['path']+'/**','--exclude='],check=True)
    bundle=target/pin['path']
    if sha(bundle/'manifest.json')!=pin['manifest_sha256']:raise ValueError('paired runtime manifest differs')
    return bundle


class Stages:
    def __init__(self,root,identity,resume):
        self.root=root;self.identity=identity;self.resume=resume

    def run(self,name,command,*,work=False,resume_partial=False):
        receipt=self.root/(name+'.stage.json')
        output=None
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if not self.resume or old['identity']!=self.identity:raise ValueError('stage identity differs: '+name)
            if old['rc']==0:
                if not Path(old['output']).is_dir():raise ValueError('completed stage output missing: '+name)
                print('PAIRED_MODEL_REUSE stage='+name+' output='+old['output'],flush=True)
                return Path(old['output'])
            if resume_partial and Path(old['output']).is_dir():output=Path(old['output'])
        attempt=name+'-'+str(time.time_ns())
        destination=self.root.parent/'ci' if work else self.root
        destination.mkdir(exist_ok=True)
        output=output or destination/attempt
        argv=list(map(str,command(output)))
        log=self.root/(attempt+'.log')
        save(self.root/(attempt+'.command.json'),dict(argv=argv))
        print('PAIRED_MODEL_START stage='+name+' log='+str(log),flush=True)
        start=time.monotonic()
        with log.open('x') as stream:
            process=subprocess.Popen(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        print(f'PAIRED_MODEL_WAIT stage={name} elapsed_minutes={(time.monotonic()-start)/60:.1f} log={log}',flush=True)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
        save(receipt,dict(identity=self.identity,output=str(output),log=str(log),rc=process.returncode,
                          wall_seconds=time.monotonic()-start))
        print(f'PAIRED_MODEL_STAGE stage={name} rc={process.returncode} output={output}',flush=True)
        if process.returncode:
            print('\n'.join(log.read_text(errors='replace').splitlines()[-20:]),flush=True)
            raise ValueError('failed stage '+name+'; see '+str(log))
        return output


def check_trace(output,model,manifest):
    path=output/model/'native/proof.json'
    proof=json.loads(path.read_text())
    name='q4_paired16_fixed' if model=='qwen35-122b-q4km' else 'q4_paired32_fixed'
    recipe=next(r for r in manifest['paired_gate_up']['selection']['recipes'] if r['name']==name)
    observed={int(r,16) for r in proof.get('paired_observed_recipes',[])}
    if (proof.get('kernel_execution')!='PASS_SHORT_REQUEST' or proof.get('paired_missing_recipes') or
        proof.get('capture_scope')!='SECOND_REQUEST_SAME_PROCESS_FIRST_USE_EXCLUDED' or
        int(recipe['id'],16) not in observed):
        raise ValueError('expected paired recipe not observed in warmed trace: '+str(path))
    return dict(recipe=recipe['id'],proof=str(path),sha256=sha(path))


def model_plan(root,devices):
    template=json.loads((ROOT/'tools/kpack_batched_tp2_122b.json').read_text())
    control=json.loads((ROOT/'tools/kpack_batched_int4_2048.json').read_text())['models'][0]
    template['models'][0]['devices']=','.join(devices);control['devices']=devices[0]
    template['models'].append(control)
    return resolve_plan(template,None,root)


def main(args):
    sdk=args.sdk.resolve(strict=True);result_root=args.result_root.resolve(strict=True)
    devices=args.devices.split(',')
    if len(devices)!=2 or len(set(devices))!=2 or not all(d.isdigit() for d in devices):
        raise ValueError('select two distinct devices, normally 6,7')
    pin=json.loads((ROOT/'tools/kpack_paired_selection_artifact.json').read_text())
    bundle=args.bundle.resolve(strict=True) if args.bundle else fetch(pin,result_root)
    if sha(bundle/'manifest.json')!=pin['manifest_sha256']:raise ValueError('explicit bundle differs from pin')
    manifest=verify(bundle,sdk=sdk)
    plan=None;compatibility=None;corpus=None
    if args.phases=='all':
        llama=args.llama.resolve(strict=True);ncp=args.ncp.resolve(strict=True)
        compatibility=Path(os.environ.get('QUACTLIZE_PPU_BUNDLE',''))
        if not (compatibility/'manifest.json').is_file():
            raise ValueError('set QUACTLIZE_PPU_BUNDLE to the retained six-library/TP2 compatibility bundle')
        compatibility=compatibility.resolve(strict=True)
        corpus=Path(os.environ.get('GSM8K_FILE','/sim/eec/shared/AI_workspace/llm-models/datasets/gsm8k/main/test-00000-of-000001.parquet'))
        if not corpus.is_file():raise ValueError('local GSM8K corpus missing: '+str(corpus))
        for tool in ('asight/bin/asys','bin/hgobjdump'):
            if not (sdk/tool).is_file():raise ValueError('SDK tool missing: '+tool)
        plan=model_plan(args.model_root,devices)
    args.output.mkdir(parents=True,exist_ok=True)
    results=args.output/'results';results.mkdir(exist_ok=args.resume)
    sources=[*sorted((ROOT/'dev/tp2_chain').glob('*.py')),
        *(ROOT/'tools'/p for p in ('run_paired_selection_box.py','run_kpack_batched_bench.py',
          'run_kpack_model_validation.py','build_kpack_model_ci.py','resolve_kpack_batched_models.py',
          'kpack_asys_scope.py','kpack_batched_tp2_122b.json','kpack_batched_int4_2048.json'))]
    identity=dict(bundle=pin['manifest_sha256'],caller=pin['llama_commit'],devices=devices,
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        sdk_runtime=manifest['execution_receipt']['runtime'],phases=args.phases,
        model_root=str(args.model_root),jobs=args.jobs,plan=plan,
        corpus=sha(corpus) if corpus else None,
        compatibility=sha(compatibility/'manifest.json') if compatibility else None,
        reused_caller=local_source_state(llama) if os.environ.get('LLAMA_CI_BUILD_DIR') and plan else None,
        build_paths={k:os.environ.get(k) for k in ('LLAMA_CI_BUILD_DIR','NCP_CI_DIR')},
        model_files={f:dict(size=Path(f).stat().st_size,mtime_ns=Path(f).stat().st_mtime_ns)
                     for m in (plan or {}).get('models',[]) for f in m['files']})
    stages=Stages(results,identity,args.resume)
    save(results/'bundle-manifest.json',manifest)
    if args.preflight_only:
        print('PAIRED_MODEL_PREFLIGHT PASS device_admission=PENDING',flush=True);return 0
    stages.run('components',lambda out:[sys.executable,'-u',ROOT/'dev/tp2_chain/production.py',
        '--sdk',sdk,'--bundle',bundle,'--output',out,'--devices',args.devices,'--l2-bytes','67108864']+
        (['--resume'] if out.exists() else []),resume_partial=True)
    if args.phases=='components':return 0
    # Fetch only the owner's fork. Never switch or overwrite the shared caller checkout.
    subprocess.run(['git','-C',llama,'fetch','https://github.com/DrXuQian/llama.cpp.git',pin['llama_branch']],check=True)
    subprocess.run(['git','-C',llama,'cat-file','-e',pin['llama_commit']+'^{commit}'],check=True)
    def build(out):
        cmd=[sys.executable,ROOT/'tools/build_kpack_model_ci.py','--llama',llama,'--ncp',ncp,
             '--sdk',sdk,'--output',out,'--receipt',out.with_suffix('.receipt.json'),'--jobs',str(args.jobs)]
        if os.environ.get('LLAMA_CI_BUILD_DIR'):
            if subprocess.check_output(['git','-C',llama,'rev-parse','HEAD'],text=True).strip()!=pin['llama_commit']:
                raise ValueError('local caller reuse requires the pinned caller commit; omit LLAMA_CI_BUILD_DIR for isolated CI')
            cmd+=['--local-llama','--reuse-llama-build',os.environ['LLAMA_CI_BUILD_DIR']]
        else:cmd+=['--llama-revision',pin['llama_commit']]
        if os.environ.get('NCP_CI_DIR'):cmd+=['--reuse-ncp-build',os.environ['NCP_CI_DIR']]
        return cmd
    ci=stages.run('caller-ci',build,work=True)
    receipt=json.loads(ci.with_suffix('.receipt.json').read_text());build_dir=Path(receipt['build'])
    if receipt['llama_source_commit']!=pin['llama_commit'] or any(
        not (build_dir/name).is_file() or sha(build_dir/name)!=digest for name,digest in receipt['files'].items()):
        raise ValueError('caller CI receipt/binaries changed; start a new run')
    save(results/'caller-ci.receipt.json',receipt)
    caller=Path(receipt['llama_worktree']['directory'])
    os.environ.update(LD_LIBRARY_PATH=str(build_dir/'bin')+os.pathsep+os.environ.get('LD_LIBRARY_PATH',''),
        QUACTLIZE_PPU_PACK_LIBRARY=str(bundle/'pack/libquactlize_ppu_pack.so'),
        QUACTLIZE_KPACK_EXECUTION=str(bundle),QUACTLIZE_KPACK_ROUTE='auto',QUACTLIZE_KPACK_COMPUTE='bf16',
        QUACTLIZE_KPACK_GATE_UP='1',QUACTLIZE_KPACK_PAIR_WEIGHTS='1',
        QUACTLIZE_KPACK_JIT_HELPER=str(ROOT/'tools/kpack_jit.py'),QUACTLIZE_KPACK_JIT_PYTHON=sys.executable,
        QUACTLIZE_KPACK_JIT_CACHE=str(args.jit_cache),
        QUACTLIZE_KPACK_DEEPGEMM_HELPER=str(bundle/'kpack_deepgemm_prewarm.py'))
    for key in ('QUACTLIZE_KPACK_PREFILL_POLICY','QUACTLIZE_KPACK_GEMV_POLICY','GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_DISABLE_FUSION'):
        os.environ.pop(key,None)
    model_results=[]
    for model in plan['models']:
        path=results/(model['name']+'.plan.json');save(path,dict(plan,models=[model]))
        common=[sys.executable,'-u',ROOT/'tools/run_kpack_model_validation.py','--llama',caller,
            '--build',build_dir,'--bundle',bundle,'--plan',path,'--cache',args.cache,
            '--jit-cache',args.jit_cache,'--logits',args.output/'logits','--corpus',corpus,
            '--asys',sdk/'asight/bin/asys','--inspector',sdk/'bin/hgobjdump']
        try:
            stages.run(model['name']+'-numerical',lambda out:common+['--phase','numerical','--output',out])
            stages.run(model['name']+'-benchmark',lambda out:[sys.executable,'-u',ROOT/'tools/run_kpack_batched_bench.py',
                '--binary',build_dir/'bin/llama-batched-bench','--llama-dir',caller,'--bundle',bundle,
                '--jit-cache',args.jit_cache,'--cache-root',args.cache,'--output-root',result_root,
                '--output',out,'--plan',path,'--order','abba','--require-selected','--repeats','2'])
            trace=stages.run(model['name']+'-trace',lambda out:common+['--phase','trace','--profiler-scope','private','--output',out])
            proof=check_trace(trace,model['name'],manifest)
            model_results.append(dict(model=model['name'],status='PASS',paired=proof))
        except ValueError as error:
            model_results.append(dict(model=model['name'],status='FAIL',error=str(error)))
            print(f'PAIRED_MODEL_FAIL model={model["name"]} remaining_models_continue=1 error={error}',flush=True)
    complete=all(r['status']=='PASS' for r in model_results)
    save(results/'summary.json',dict(status='PASS' if complete else 'INCOMPLETE',models=model_results,
        first_use_excluded=True,device_admission='PENDING_RESULT_REVIEW'))
    return 0 if complete else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('sdk','output','result-root','llama','ncp','cache','jit-cache','model-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--devices',default='6,7');parser.add_argument('--jobs',type=int,default=192)
    parser.add_argument('--bundle',type=Path);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--phases',choices=('all','components'),default='all')
    parser.add_argument('--preflight-only',action='store_true')
    raise SystemExit(main(parser.parse_args()))
