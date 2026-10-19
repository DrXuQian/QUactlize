#!/usr/bin/env python3
"""Build the narrow unsigned-coordinate experiment; production is unchanged."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_simt import build_q6_metadata as base
from dev.gemv_simt.build_q6_sparse import inputs as parent_inputs
from dev.gemv_simt.q6_index import source, KERNEL_NAMES
from quactlize.runtime.compiler import sha


def inputs():
    result=parent_inputs()
    for name in ('q6_unsigned_map.hpp','q6_index.py','build_q6_index.py'):
        relative='dev/gemv_simt/'+name
        result[relative]=sha(ROOT/relative)
    return result


def build(sdk,output,object_only=False):
    return base.build(sdk,output,object_only,generator=source,source_inputs=inputs,
                      schema='q6-index-v1',kernel_names=KERNEL_NAMES)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--object-only',action='store_true')
    a=p.parse_args();build(a.sdk,a.output.resolve(),a.object_only)
