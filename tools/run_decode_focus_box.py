#!/usr/bin/env python3
"""Fetch three prebuilt low-MBU experiments; never replace production policy."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quactlize.runtime.compiler import sha
from dev.gemv_model.run import verify


def fetch(pin, result_root, cohort_name='tp2-focus'):
    paths = {'tp2-focus':'prebuilt/ppu0010/tp2-decode-focus-v1',
             'tp2-chain':'prebuilt/ppu0010/tp2-paired-chain-v1'}
    if cohort_name not in paths:
        raise ValueError('unknown prebuilt experiment')
    if (not re.fullmatch(r'[0-9a-f]{40}',pin['commit']) or
            pin['path']!=paths[cohort_name] or
            not re.fullmatch(r'artifacts/[A-Za-z0-9][A-Za-z0-9._/-]*',pin['branch']) or
            not re.fullmatch(r'[0-9a-f]{64}',pin['manifest_sha256'])):
        raise ValueError('invalid focus artifact pin')
    prefix = 'quactlize-focus-artifact-' if cohort_name=='tp2-focus' else 'quactlize-chain-artifact-'
    target=result_root/(prefix+pin['commit'][:10])
    subprocess.run(['git','fetch','origin',pin['branch']],cwd=ROOT,check=True)
    if not target.exists():
        subprocess.run(['git','worktree','add','--detach',target,pin['commit']],cwd=ROOT,
                       env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1'),check=True)
    if subprocess.check_output(['git','-C',target,'rev-parse','HEAD'],text=True).strip()!=pin['commit']:
        raise ValueError('focus artifact worktree differs')
    subprocess.run(['git','-C',target,'lfs','pull','origin','--include='+pin['path']+'/**','--exclude='],check=True)
    bundle=target/pin['path']
    if sha(bundle/'manifest.json')!=pin['manifest_sha256']:raise ValueError('focus manifest differs')
    verify(bundle,cohort_name)
    return bundle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('sdk','output','result-root'):p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not re.fullmatch(r'[0-9]+',os.environ.get('CUDA_VISIBLE_DEVICES','')):
        raise ValueError('one physical device is required; this handoff uses CUDA_VISIBLE_DEVICES=6')
    bundle=fetch(json.loads((ROOT/'tools/kpack_decode_focus_artifact.json').read_text()),a.result_root.resolve(strict=True))
    acu=a.sdk/'asight/bin/acu'
    if not acu.is_file():raise ValueError('missing ACU: '+str(acu))
    args=[sys.executable,'-u',ROOT/'dev/gemv_model/run.py','--cohort','tp2-focus',
          '--sdk',a.sdk,'--bundle',bundle,'--output',a.output,'--l2-bytes','67108864','--acu',acu]
    if a.resume:args.append('--resume')
    print('DECODE_FOCUS points=3 configs=22 production_selection_changed=0 cache=ROTATING_L2_64MiB',flush=True)
    return subprocess.run(args,cwd=ROOT).returncode


if __name__=='__main__':raise SystemExit(main())
