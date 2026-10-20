#!/usr/bin/env python3
"""Reproduce the bounded M8 profile-gap review on the host; no GPU work."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.strategy.production import serialize
from dev.strategy.refresh_gap_review import close_supplement, review
from dev.strategy.refresh_gaps import TABLE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('--output', type=Path, help='new file, never overwrite raw evidence')
    p.add_argument('--check', type=Path, help='compare the complete checked-in review')
    a = p.parse_args()
    if a.output and a.output.exists():
        p.error('--output must be a new file')
    result = review(a.archive)
    pending = close_supplement(json.loads((ROOT/TABLE).read_bytes()), result)
    if a.check and result != json.loads(a.check.read_bytes()):
        p.error('checked-in review differs from raw evidence')
    if a.output:
        with a.output.open('x') as output:
            output.write(serialize(result))
    print('DECODE_PROFILE_GAPS_REVIEW '+json.dumps(dict(status=result['status'],
        **result['counts'], pending=pending['counts'], production_changed=False, kernel_launches=0)))


if __name__ == '__main__':
    main()
