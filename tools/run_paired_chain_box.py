#!/usr/bin/env python3
"""Fetch the narrow paired-chain package; no JIT or caller rebuild on box."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_decode_focus_box import fetch
from dev.gemv_model.run import verify
from dev.tp2_chain.plan import POINTS
from quactlize.runtime.compiler import LIBRARIES, sha


def preflight(bundle, sdk):
    manifest = verify(bundle, 'tp2-chain')
    expected = {p.name for p in POINTS}
    if {r['point']['name'] for r in manifest['records']} != expected:
        raise ValueError('both compiled chain points are required')
    required = [sdk/'asight/bin/acu', *[sdk/'lib'/f'lib{x}.so' for x in LIBRARIES]]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ValueError('SDK root must be PPU_SDK, not CUDA_SDK; missing: '+', '.join(missing))
    return dict(manifest_sha256=sha(bundle/'manifest.json'), points=len(expected),
                runtime={p.name:sha(p) for p in required}, build_runtime=manifest['runtime'],
                runtime_matches_build=all(sha(sdk/'lib'/name)==digest for name,digest in manifest['runtime'].items()),
                production_selection_changed=False, device_admission='PENDING')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('sdk','output','result-root'):
        p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--devices',default='6,7')
    p.add_argument('--l2-bytes',type=int,default=67108864)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--bundle',type=Path,help='explicit already-fetched package for local preflight')
    a=p.parse_args()
    devices=a.devices.split(',')
    if len(devices)!=2 or len(set(devices))!=2 or any(not re.fullmatch(r'\d+',d) for d in devices):
        p.error('--devices requires two different ordinals, for example 6,7')
    if a.l2_bytes<=0:
        p.error('--l2-bytes must be a verified positive size')
    result_root=a.result_root.resolve(strict=True)
    sdk=a.sdk.resolve(strict=True)
    bundle=a.bundle.resolve(strict=True) if a.bundle else fetch(
        json.loads((ROOT/'tools/kpack_paired_chain_artifact.json').read_text()),result_root,'tp2-chain')
    receipt=preflight(bundle,sdk)
    print('PAIRED_CHAIN_PREFLIGHT '+json.dumps(receipt),flush=True)
    if a.preflight_only:
        return 0
    args=[sys.executable,'-u',ROOT/'dev/tp2_chain/run.py','--sdk',sdk,'--bundle',bundle,
          '--output',a.output,'--devices',a.devices,'--l2-bytes',str(a.l2_bytes)]
    if a.resume:
        args.append('--resume')
    print('PAIRED_CHAIN_BOX points=2 arms=4 correctness_tokens=1..8 timing_tokens=1 '
          'rounds=6 samples=15 acu_after_timing=1 production_selection_changed=0 compile_on_box=0',flush=True)
    return subprocess.run(args,cwd=ROOT).returncode


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,OSError,subprocess.CalledProcessError) as error:
        print('PAIRED_CHAIN_BOX FAIL: '+str(error),file=sys.stderr)
        raise SystemExit(1)
