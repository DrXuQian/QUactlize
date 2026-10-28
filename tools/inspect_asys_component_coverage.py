#!/usr/bin/env python3
"""Read an existing capture/SQLite pair; no GPU launch or result admission."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.prefill_asys.trace import activities, runtime_activities
from dev.strategy.asys_components import parse_component_rows


def inspect(attempt):
    attempt = Path(attempt).resolve(strict=True)
    captured = json.loads((attempt / 'capture.json').read_text())
    if 'component_records' not in captured and 'data' in captured:
        captured = captured['data']
    schedule = captured['trace_schedule']
    records = captured['component_records']
    rows = activities(attempt / 'trace.sqlite', launch_shared=True)
    kernels = [r for r in rows if r['kind'] == 'kernel']
    apis = runtime_activities(attempt / 'trace.sqlite')
    expected = Counter()
    for item in schedule:
        expected[records[item['component']]['identity']['node']['symbol']] += item['kernels']
    actual = Counter(r['symbol'] for r in kernels)
    protocol = captured.get('protocol', dict(rounds=6, samples=5))
    result = dict(attempt=str(attempt), point=captured.get('point'), phase=captured.get('phase'),
        expected_kernels=sum(expected.values()), actual_kernels=len(kernels),
        non_kernel_activities=len(rows)-len(kernels), schedule_samples=len(schedule),
        components=len(records), runtime_api_counts=dict(Counter(a['name'] for a in apis)),
        symbols=[dict(symbol=s, expected=expected[s], actual=actual[s],
                      difference=actual[s]-expected[s]) for s in sorted(set(expected) | set(actual), key=str)],
        gpu_work=False, performance_admitted=False, production_changed=False,
        scope='EXISTING_TRACE_COMPONENT_COVERAGE_ONLY_NOT_NUMERICAL_OR_TIMING_ADMISSION')
    try:
        parse_component_rows(records, schedule, kernels, {}, protocol['rounds'], protocol['samples'])
    except (KeyError, TypeError, ValueError) as error:
        result.update(status='COVERAGE_OR_CONTRACT_FAIL', error=str(error))
    else:
        result.update(status='COMPONENT_COVERAGE_MATCHES')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = inspect(args.attempt)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == 'COMPONENT_COVERAGE_MATCHES' else 1


if __name__ == '__main__':
    raise SystemExit(main())
