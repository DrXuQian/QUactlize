"""Build a bounded explicit-compute decode parent; no device is needed."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quactlize.decode.compiler import DecodeCompiler
from quactlize.decode.grouped_compiler import GroupedComputeCompiler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--qtype", type=int, choices=(8, 10, 11, 12, 13, 14), required=True)
    parser.add_argument("--kind", choices=("dense", "grouped"), required=True)
    parser.add_argument("--quant-route", choices=("fq", "sf"), default="fq")
    parser.add_argument("--persistent", type=int, choices=(0,1), default=0)
    parser.add_argument("--compute", choices=("f16", "bf16"), default="bf16")
    parser.add_argument("--tile-m", type=int, choices=(8, 16, 64), default=8)
    args = parser.parse_args()
    route = ("sf" if args.qtype == 8 else args.quant_route) + "-" + args.kind
    parent = dict(qtype=args.qtype, route=route, tm=args.tile_m, tn=64, tk=256,
                  wm=min(args.tile_m,16), wn=16, stages=2, ap=0, dn=16,
                  persistent=args.persistent if route == "fq-grouped" else -1,
                  symbol=f"bf16_compute_q{args.qtype}_{route.replace('-','_')}_tm{args.tile_m}_p{args.persistent}")
    compiler = DecodeCompiler if args.kind == "dense" else GroupedComputeCompiler
    result = compiler(args.sdk, args.cache, compute_type=args.compute).build(parent)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
