#!/usr/bin/env python3
"""A1/A2 sparse metadata confirmation; default one point and one ACU capture."""
import argparse
from pathlib import Path
import sys
import tarfile
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.gemv_simt.build_q6_sparse import build, inputs
from dev.gemv_simt.q6_sparse import ARMS
from dev.gemv_simt.q6_sparse_access import pattern
from dev.strategy.journal import atomic_json
from tools.run_q6_metadata import child, collect, workloads


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sdk','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--l2-bytes',type=int,default=67108864)
    p.add_argument('--points',default='0',help='0 for focus; all for the existing eight points')
    p.add_argument('--confirm-idle',action='store_true')
    p.add_argument('--acu',type=Path)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--child',type=int,choices=range(len(workloads())),help=argparse.SUPPRESS)
    p.add_argument('--result',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args();a.sdk=a.sdk.resolve();a.output=a.output.resolve();a.profile_arm=None
    if a.points not in ('0','all'):p.error('--points must be 0 or all')
    if a.child is not None:
        if a.result is None:p.error('child needs a result path')
        return child(a,source_inputs=inputs,pattern_model=pattern)
    if a.output.exists():p.error('--output must be a new directory')
    print('Q6_SPARSE_START control=A1_REGISTER_FIELDS candidate=A2_SPARSE_METADATA '
          'profile_weight_copies=1 profile_graphs=0 production_changed=0',flush=True)
    try:
        return collect(a,builder=build,arms=ARMS,driver=Path(__file__),
                       point_indices=range(len(workloads())) if a.points=='all' else (0,),profile_arms=(1,))
    except Exception:
        error=traceback.format_exc()
        if a.output.is_dir():atomic_json(a.output/'failure.json',dict(status='FAIL',error=error))
        print(error,file=sys.stderr,flush=True)
        return 1
    finally:
        if a.output.is_dir():
            archive=a.output/'review.tgz'
            with tarfile.open(archive,'w:gz') as tar:
                for path in sorted(a.output.rglob('*')):
                    if path.is_file() and not path.is_symlink() and (path.suffix in ('.json','.log','.csv','.txt','.acurep')
                            or path.name=='profile-1'):
                        tar.add(path,arcname=str(path.relative_to(a.output)),recursive=False)
            print('results='+str(archive),flush=True)


if __name__=='__main__':
    raise SystemExit(main())
