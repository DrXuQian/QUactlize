#!/usr/bin/env python3
"""Replay blind rules over the reviewed completed screen; no GPU or routing writes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.strategy.prefill_closure import export
from tools.export_kpack_strategy import load_sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error('output must be new')
    table, _ = load_sources()
    plan = export(a.review, table)
    with a.output.open('x') as out:
        json.dump(plan, out, sort_keys=True, indent=2, allow_nan=False)
        out.write('\n')
    print('STRATEGY_CLOSURE_PLAN '+json.dumps(plan['counts']), flush=True)
    return int(plan['counts']['rule_gaps'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
