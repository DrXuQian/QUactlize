#!/usr/bin/env python3
"""Audit the three reviewed Box archives and fit the small-M component policy."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.strategy.production import fit, load, serialize
from tools.generate_decode_selection import COMPONENTS, EVIDENCE, effective


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('screen', 'retry', 'q8'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    points, runs = load(args.screen, args.retry, args.q8)
    baseline = effective(json.loads(EVIDENCE.read_text()), include_components=False)
    evidence = fit(points, runs, baseline)
    if args.check:
        if json.loads(COMPONENTS.read_text()) != evidence:
            raise ValueError('component policy differs from reviewed evidence')
    else:
        COMPONENTS.write_text(serialize(evidence))
    print('SMALLM_COMPONENT_POLICY '+json.dumps(evidence['summary']))


if __name__ == '__main__':
    main()
