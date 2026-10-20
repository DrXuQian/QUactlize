#!/usr/bin/env python3
"""Build an inactive, reviewed small-M selection table. No GPU work or tuning."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.strategy.production import serialize
from dev.strategy.refresh_selection import fit, load, readiness, require, supplement_plan, verify
from quactlize.dispatch.planning import plan_smallm
from quactlize.execution.q8_refresh import capability, check_cpp_transition
from tools.generate_decode_selection import EVIDENCE, effective


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--retry', type=Path, required=True)
    p.add_argument('--prior-pass', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='new review directory, never a runtime overwrite')
    p.add_argument('--check', type=Path, help='compare a checked-in candidate table with reconstructed evidence')
    p.add_argument('--table-output', type=Path, help='generate an inactive candidate table for source control')
    a = p.parse_args()
    if a.output.exists():
        p.error('--output must be a new directory')
    previous = None
    if a.table_output and a.table_output.exists():
        previous = json.loads(a.table_output.read_text())
        verify(previous)
    points, originals, sources = load(a.parent, a.retry, a.prior_pass)
    baseline = effective(json.loads(EVIDENCE.read_text()), include_refresh=False)
    table = fit(points, originals, sources, baseline)
    verify(table)
    if previous:
        require(previous['baseline_sha256'] == table['baseline_sha256'] and
                previous['archives'] == table['archives'],
                'refusing to replace a candidate table from another measured cohort')
    if a.check:
        require(json.loads(a.check.read_text()) == table, 'checked-in table differs from raw measurements')
    # Host compilation only. Compare the actual C++ choices, not a Python
    # approximation of the caller's dispatch or the historical old table.
    a.output.mkdir(parents=True, exist_ok=False)
    current = plan_smallm(a.output, [r['request'] for r in originals])
    bound = check_cpp_transition(originals, current['requests'])
    capabilities = readiness(table, baseline, q8_bindings=capability()['keys'])
    supplement = supplement_plan(table)
    (a.output/'selection.json').write_text(serialize(table))
    (a.output/'readiness.json').write_text(json.dumps(capabilities, indent=2, sort_keys=True)+'\n')
    (a.output/'pending.json').write_text(json.dumps(table['pending'], indent=2, sort_keys=True)+'\n')
    (a.output/'supplement-plan.json').write_text(json.dumps(supplement, indent=2, sort_keys=True)+'\n')
    if a.table_output:
        a.table_output.write_text(serialize(table))
    print('DECODE_REFRESH_SELECTION '+json.dumps(dict(
        **table['summary'], evidence_sha256=table['evidence_sha256'],
        readiness=capabilities['status'], pending_implementation=len(capabilities['pending']),
        candidate_profile_gaps=supplement['counts'],
        original_cpp_choices_checked=len(originals), kernel_launches=0,
        current_exact_q8_bindings_checked=bound,
        production_changed=False, output=str(a.output)), sort_keys=True))


if __name__ == '__main__':
    main()
