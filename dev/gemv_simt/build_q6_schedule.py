#!/usr/bin/env python3
"""Build six bounded Q6 scheduling recipes, never a production replacement."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_simt import build_q6_metadata as base
from dev.gemv_simt.build_q6_index import inputs as parent_inputs
from dev.gemv_simt.q6_schedule import source, KERNEL_NAMES, CONFIGS, EQUAL_GROUPS
from dev.gemv_simt.q6_schedule_audit import audit
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import sha


def inputs():
    result = parent_inputs()
    for name in ('q6_schedule.py', 'build_q6_schedule.py', 'q6_schedule_audit.py'):
        relative = 'dev/gemv_simt/'+name
        result[relative] = sha(ROOT/relative)
    return result


def build(sdk, output, object_only=False):
    receipt = base.build(sdk, output, object_only, generator=source, source_inputs=inputs,
                         schema='q6-schedule-v1', kernel_names=KERNEL_NAMES,
                         record_contract=dict(configs=CONFIGS, equal_groups=EQUAL_GROUPS))
    atomic_json(output/'native-audit.json', [audit(output,c) for c in ('f16','bf16')])
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--object-only', action='store_true')
    a = p.parse_args()
    build(a.sdk, a.output.resolve(), a.object_only)
