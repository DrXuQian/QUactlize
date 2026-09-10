#!/usr/bin/env python3
"""Compile a selected PPU parent, or prewarm a policy-selected parent plan.

CPU only: this program never loads a kernel, creates a device context, measures
candidates or imports torch. Use the same checkout/SDK for prewarm and resolve.
"""

import argparse
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import Compiler, validate_parent, source_contract, sha
from quactlize.runtime.tuning import ROUTES, digest


def parent_tuple(symbol, values):
    fields = ("qtype", "route", "tm", "tn", "tk", "wm", "wn", "stages", "ap", "dn", "persistent")
    parent = dict(zip(fields, values), symbol=symbol)
    if parent["route"] not in range(len(ROUTES)):
        raise ValueError("invalid parent route")
    parent["route"] = ROUTES[parent["route"]]
    validate_parent(parent)
    return parent


def inspect_modules(cache, keys, contract):
    """Read published cache entries for execution evidence, without compiling."""
    cache = cache.resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{64}", contract):
        raise ValueError("invalid dispatcher source contract")
    records = []
    for key in sorted(set(keys)):
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("invalid module key")
        directory = cache / key
        receipt, library = directory / "manifest.json", directory / "kernel.so"
        if any(p.is_symlink() for p in (directory, receipt, library)) or directory.resolve().parent != cache:
            raise ValueError("module cache path escapes entry")
        r = json.loads(receipt.read_text())
        if r.get("key") != key or source_contract(r["identity"]) != contract:
            raise ValueError("module source/receipt identity differs")
        source = Compiler.source(None, r["parent"], "")
        if digest(dict(identity=r["identity"], parent=r["parent"], source=source)) != key:
            raise ValueError("module compiler key differs")
        if sha(library) != r.get("sha256"):
            raise ValueError("module payload differs")
        records.append(r | dict(path=str(library), receipt_sha256=sha(receipt)))
    return records


def model_requests(path, tokens, include_sf=False, tensor_pattern=None):
    from tools.gguf_internal_shape_inventory import read_gguf_header
    from quactlize.gguf_roles import match_role
    if not tokens or any(type(t) is not int or t <= 0 for t in tokens):
        raise ValueError("tokens must be positive")
    with path.open("rb") as stream:
        header = read_gguf_header(stream, str(path))
    metadata = header["metadata"]
    if metadata.get("split.count", 1) != 1:
        raise ValueError("model prewarm needs an unsplit GGUF; use explicit requests for sharded models")
    arch = metadata.get("general.architecture", "")
    requests, omitted = set(), []
    for tensor in header["tensors"]:
        q, dims, name = tensor["qtype"], tensor["dims_gguf"], tensor["name"]
        if tensor_pattern and not re.search(tensor_pattern, name):
            omitted.append(dict(name=name, qtype=q, reason="OUTSIDE_CONSUMER_TENSOR_PATTERN"))
            continue
        role = match_role(name, len(dims)) if q in (8,10,11,12,13,14) else None
        if not role or role[0].route_class not in ("dense", "grouped"):
            omitted.append(dict(name=name, qtype=q, reason="NOT_A_SUPPORTED_KQUANT_MATMUL"))
            continue
        grouped = role[0].route_class == "grouped"
        k, n = dims[:2]
        experts = dims[2] if grouped else 1
        topk = metadata.get(f"{arch}.expert_used_count") if grouped else 1
        if type(topk) is not int or not 1 <= topk <= experts:
            raise ValueError(f"{name}: missing/invalid GGUF expert_used_count")
        if grouped and metadata.get(f"{arch}.expert_count") != experts:
            raise ValueError(f"{name}: tensor expert axis and GGUF expert_count differ")
        for t in tokens:
            for sf in ((1,) if q == 8 else (0, 1) if include_sf else (0,)):
                requests.add((q, 2*int(grouped)+sf, t*topk, n, k, experts, t))
    return sorted(requests), dict(model=str(path), header_identity=digest(header), omitted=omitted,
                                  tensor_pattern=tensor_pattern,
                                  scope="HEADER_ONLY_UNSPLIT_MODEL_NO_TENSOR_BYTES_NO_TP")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    resolve = sub.add_parser("resolve", help="dispatcher protocol: compile exactly one parent")
    prewarm = sub.add_parser("prewarm", help="compile distinct parents from a build plan, no tuning")
    inspect = sub.add_parser("inspect", help="validate selected disk-cache entries without loading or compiling")
    inspect.add_argument("--cache", type=Path, required=True)
    inspect.add_argument("--source-contract", required=True)
    inspect.add_argument("--keys", nargs="+", required=True)
    plan = sub.add_parser("plan", help="use the actual C++ heuristic for explicit model requests")
    inputs = plan.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--request", nargs=7, type=int, action="append",
                      metavar=("Q", "ROUTE", "M", "N", "K", "EXPERTS", "MAX_ROWS"))
    inputs.add_argument("--model", type=Path, help="read only GGUF headers; preserve name/op authority")
    plan.add_argument("--tokens", type=int, nargs="+", default=[1, 128, 512])
    plan.add_argument("--include-sf", action="store_true", help="also prewarm explicit SF routes, not route selection")
    plan.add_argument("--tensor-pattern", help="consumer weight-name regex; model mode only")
    plan.add_argument("--allow-misses", action="store_true", help="record uncovered families without inventing tactics")
    plan.add_argument("--output", type=Path, required=True)
    for p in (resolve, prewarm):
        p.add_argument("--sdk", type=Path, required=True)
        p.add_argument("--cache", type=Path, required=True)
    resolve.add_argument("--parent", required=True)
    resolve.add_argument("--source-contract", help="dispatcher source identity; check before compiling")
    resolve.add_argument("--tuple", type=int, nargs=11, required=True, dest="values")
    prewarm.add_argument("--plan", type=Path, required=True)
    prewarm.add_argument("--jobs", type=int, default=4)
    prewarm.add_argument("--receipt", type=Path)
    prewarm.add_argument("--source-contract", help="reject a foreign dispatcher before prewarm")
    args = parser.parse_args()
    start = time.monotonic()
    if args.command == "inspect":
        print(json.dumps(dict(modules=inspect_modules(args.cache, args.keys, args.source_contract))))
    elif args.command == "plan":
        from tools.build_kpack_dispatch import plan as select_plan
        if args.tensor_pattern and not args.model:
            raise ValueError("tensor-pattern requires a model")
        requests, authority = model_requests(args.model, args.tokens, args.include_sf, args.tensor_pattern) if args.model else (args.request, {})
        args.output.mkdir(parents=True, exist_ok=False)
        parents, selected = select_plan(args.output, requests)
        (args.output / "plan.json").write_text(json.dumps(dict(parents=parents, requests=selected,
                                                             authority=authority), indent=2)+"\n")
        misses = sum(r["status"] != "SELECTED" for r in selected)
        print(f"KPACK_JIT_PLAN parents={len(parents)} requests={len(selected)} misses={misses}")
        if misses and not args.allow_misses:
            raise ValueError("model requests outside heuristic coverage; no tactic was invented")
    elif args.command == "resolve":
        parent = parent_tuple(args.parent, args.values)
        print(f"KPACK_JIT_RESOLVE parent={parent['symbol']} cache={args.cache}", file=sys.stderr, flush=True)
        compiler = Compiler(args.sdk, args.cache)
        if args.source_contract and args.source_contract != source_contract(compiler.identity):
            raise ValueError("JIT helper source differs from dispatcher; rebuild the small dispatcher")
        record = compiler.build(parent)
        print(f"KPACK_JIT parent={parent['symbol']} cache_hit={int(record['cache_hit'])} "
              f"seconds={time.monotonic()-start:.3f} key={record['key']}", file=sys.stderr)
        print("QK_JIT_V1", record["key"], digest(record["identity"]), source_contract(record["identity"]))
    else:
        parents = json.loads(args.plan.read_text())["parents"]
        print(f"KPACK_JIT_PREWARM start parents={len(parents)} jobs={args.jobs} cache={args.cache}", flush=True)
        compiler = Compiler(args.sdk, args.cache, args.jobs)
        if args.source_contract and args.source_contract != source_contract(compiler.identity):
            raise ValueError("JIT helper source differs from dispatcher; rebuild the small dispatcher")
        records = compiler.compile_only(parents, progress=lambda n, total: print(
            f"KPACK_JIT_PREWARM completed={n}/{total} seconds={time.monotonic()-start:.1f}", flush=True))
        if args.receipt:
            args.receipt.write_text(json.dumps(dict(modules=records, seconds=time.monotonic()-start,
                                                   device_validated=False), indent=2)+"\n")
        print(f"KPACK_JIT_PREWARM PASS parents={len(records)} hits={sum(r['cache_hit'] for r in records)} "
              f"seconds={time.monotonic()-start:.3f} device_validation=PENDING")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"KPACK_JIT FAIL: {error}", file=sys.stderr)
        sys.exit(1)
