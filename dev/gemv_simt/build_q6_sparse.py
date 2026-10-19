#!/usr/bin/env python3
"""Narrow Q6 sparse-metadata module, with the unchanged A1 body as control."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_simt import build_q6_metadata as base
from dev.gemv_simt.q6_sparse import source, KERNEL_NAMES
from quactlize.runtime.compiler import sha


def inputs():
    result=base.inputs()
    for name in ('q6_sparse.cuh','q6_sparse.py','build_q6_sparse.py'):
        relative='dev/gemv_simt/'+name
        result[relative]=sha(ROOT/relative)
    return result


def build(sdk,output,object_only=False):
    return base.build(sdk,output,object_only,generator=source,source_inputs=inputs,
                      schema='q6-sparse-v1',kernel_names=KERNEL_NAMES)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--object-only',action='store_true')
    a=p.parse_args()
    build(a.sdk,a.output.resolve(),a.object_only)
