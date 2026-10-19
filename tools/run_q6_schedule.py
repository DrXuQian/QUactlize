#!/usr/bin/env python3
"""Six Q6 recipes, one focus point, three ACU rounds; CE cycles decide."""
import argparse
import json
from pathlib import Path
import sys
import tarfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.gemv_simt.build_q6_schedule import build, inputs
from dev.gemv_simt.q6_schedule import ARMS
from dev.gemv_simt.q6_schedule_access import pattern
from dev.gemv_simt.q6_schedule_cycles import comparisons
from dev.strategy.journal import atomic_json
from tools.run_q6_metadata import child, collect, workloads


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('sdk', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--l2-bytes', type=int, default=67108864)
    p.add_argument('--points', choices=('0','all'), default='0')
    p.add_argument('--confirm-idle', action='store_true')
    p.add_argument('--acu', type=Path)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--child', type=int, choices=range(len(workloads())), help=argparse.SUPPRESS)
    p.add_argument('--result', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    a.sdk, a.output, a.profile_arm = a.sdk.resolve(), a.output.resolve(), None
    if a.child is not None:
        if a.result is None:
            p.error('child needs a result path')
        return child(a, source_inputs=inputs, pattern_model=pattern)
    if a.output.exists():
        p.error('--output must be a new directory')
    if not a.plan_only and a.acu is None:
        p.error('--acu is required for cycle comparisons')
    print('Q6_SCHEDULE_START arms=6 warp_counts=2,4,8 pipeline_depth=2 '
          'score=CE_CYCLES profiles=18 event_time=SECONDARY_ONLY production_changed=0', flush=True)
    try:
        rc = collect(a, builder=build, arms=ARMS, driver=Path(__file__),
                     point_indices=range(len(workloads())) if a.points == 'all' else (0,),
                     profile_arms=tuple(range(6)), profile_rounds=3)
        if not a.plan_only:
            summary = json.loads((a.output/'summary.json').read_text())
            cycles = comparisons(summary['profiles'], summary['points'][0])
            summary['cycles'] = cycles
            atomic_json(a.output/'summary.json', summary)
            atomic_json(a.output/'cycles.json', cycles)
            for pair in cycles['comparisons']:
                print('Q6_SCHEDULE_PAIR '+json.dumps({k:pair[k] for k in
                      ('axis','names','median','paired_delta_pct','median_delta_pct','verdict')}), flush=True)
            print('Q6_SCHEDULE_RANKING '+json.dumps(cycles['ranking']), flush=True)
        return rc
    except Exception:
        error = traceback.format_exc()
        if a.output.is_dir():
            atomic_json(a.output/'failure.json', dict(status='FAIL', error=error))
        print(error, file=sys.stderr, flush=True)
        return 1
    finally:
        if a.output.is_dir():
            archive = a.output/'review.tgz'
            with tarfile.open(archive, 'w:gz') as tar:
                for path in sorted(a.output.rglob('*')):
                    if path.is_file() and not path.is_symlink() and (
                            path.suffix in ('.json','.log','.csv','.txt','.acurep') or
                            path.name.startswith('profile-r') and not path.suffix):
                        tar.add(path, arcname=str(path.relative_to(a.output)), recursive=False)
            print('results='+str(archive), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
