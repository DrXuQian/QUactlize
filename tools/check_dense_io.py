#!/usr/bin/env python3
"""Check scalar/control/native-style dense casts in a built production library."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.dense_helpers.io_gate import run, GateFailure, SCOPE
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('--output already exists')
    library_sha=sha(a.library)
    try:
        value=run(a.sdk,a.library)
    except GateFailure as e:
        value=e.report
    except Exception as e:
        value=dict(status='FAIL',error=f'{type(e).__name__}: {e}',scope=SCOPE,performance_admitted=False)
    value.update(library=str(a.library.resolve()),library_sha256=library_sha,
                 gate_sha256=sha(ROOT/'dev/dense_helpers/io_gate.py'))
    if sha(a.library)!=library_sha:value.update(status='FAIL',error='library changed during the gate')
    atomic_json(a.output,value)
    if value.get('first_failure'):
        print('DENSE_IO_GATE_FIRST_FAILURE '+json.dumps(value['first_failure'],allow_nan=False),flush=True)
    if value.get('error'):print('DENSE_IO_GATE_ERROR '+value['error'],flush=True)
    print(f'DENSE_IO_GATE {value["status"]} cases={value.get("cases",0)} output={a.output}',flush=True)
    return int(value['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
