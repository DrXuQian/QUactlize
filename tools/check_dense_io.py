#!/usr/bin/env python3
"""Check scalar/control/native-style dense casts in a built production library."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.dense_helpers.io_gate import run
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('--output already exists')
    value=run(a.sdk,a.library)
    value['library_sha256']=sha(a.library)
    atomic_json(a.output,value)
    print(f'DENSE_IO_GATE PASS cases={value["cases"]} output={a.output}',flush=True)


if __name__=='__main__':main()
