#!/usr/bin/env python3
"""Build the C++ selector and only its selected PPU parent closure."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import Compiler, sha, validate_parent
from quactlize.runtime.tuning import ROUTES, digest


def requests():
    result = []
    dense = [
        (1024, 5120),
        (5120, 8192),
        (5120, 25600),
        (8192, 5120),
        (25600, 5120),
        (512, 2048),
        (2048, 512),
        (3072, 512),
    ]
    grouped = [(512, 2048), (2048, 512), (512, 3072), (3072, 512)]
    for q in range(10, 15):
        for route in (0, 1):
            for n, k in dense:
                for m in (1, 4, 8, 64, 128, 256, 512, 1024, 2048, 4096):
                    result.append((q, route, m, n, k, 1, m))
        for route in (2, 3):
            for n, k in grouped:
                for tokens in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
                    result.append((q, route, tokens * 8, n, k, 256, tokens))
    return result


def plan(output):
    executable = output / "policy-query"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            f"-I{ROOT}",
            str(ROOT / "tools/kpack_native_policy.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    inputs = requests()
    lines = subprocess.check_output(
        [str(executable)],
        text=True,
        input="".join(" ".join(map(str, r)) + "\n" for r in inputs),
    ).splitlines()
    if len(lines) != len(inputs):
        raise ValueError("native policy omitted requests")
    parents = {}
    selected = []
    for req, line in zip(inputs, lines):
        if line == "MISS":
            selected.append(dict(request=req, status="FALLBACK_REQUIRED"))
            continue
        p = line.split()
        values = list(map(int, p[1:]))
        (
            q,
            route,
            tm,
            tn,
            tk,
            wm,
            wn,
            stages,
            ap,
            dn,
            persistent,
            split,
            grid_mode,
            grid_b,
            kind,
        ) = values
        parent = dict(
            symbol=p[0],
            qtype=q,
            route=ROUTES[route],
            tm=tm,
            tn=tn,
            tk=tk,
            wm=wm,
            wn=wn,
            stages=stages,
            ap=ap,
            dn=dn,
            persistent=persistent,
        )
        validate_parent(parent)
        if p[0] in parents and parents[p[0]] != parent:
            raise ValueError("symbol aliases different parent tuples")
        parents[p[0]] = parent
        selected.append(
            dict(
                request=req,
                status="SELECTED",
                parent=p[0],
                split=split,
                grid_mode=grid_mode,
                grid_b=grid_b,
                policy=kind,
            )
        )
    return sorted(parents.values(), key=lambda p: p["symbol"]), selected


def catalog(records):
    rows = []
    for r in records:
        p = r["parent"]
        strings = [p["symbol"], r["key"], digest(r["identity"])]
        values = [p["qtype"], ROUTES.index(p["route"])] + [
            p[f] for f in ("tm", "tn", "tk", "wm", "wn", "stages", "ap", "dn")
        ]
        rows.append(
            "  {"
            + ",".join([json.dumps(s) for s in strings] + list(map(str, values)))
            + "},"
        )
    return "static Image const kImages[] = {\n" + "\n".join(rows) + "\n};\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument(
        "--execution-bundle",
        type=Path,
        default=ROOT / "prebuilt/ppu0010/kpack-execution-v1",
    )
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    parents, selected = plan(output)
    (output / "plan.json").write_text(
        json.dumps(dict(parents=parents, requests=selected), indent=2) + "\n"
    )
    print(
        f"KPACK_DISPATCH_PLAN requests={len(selected)} parents={len(parents)} "
        f"misses={sum(r['status']=='FALLBACK_REQUIRED' for r in selected)}",
        flush=True,
    )
    if args.plan_only:
        return
    compiler = Compiler(args.sdk, args.cache or output / "modules", args.jobs)
    records = compiler.compile_only(
        parents, progress=lambda *x: print("KPACK_DISPATCH_BUILD", *x, flush=True)
    )
    for r in records:
        target = output / "modules" / r["key"] / "kernel.so"
        target.parent.mkdir(parents=True, exist_ok=True)
        if Path(r["path"]).resolve() != target:
            shutil.copy2(r["path"], target)
        r["path"] = str(target.relative_to(output))
    (output / "catalog.inc").write_text(catalog(records))
    host = output / "libquactlize_kpack_dispatch.so"
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        "-pthread",
        "-Wl,-Bsymbolic",
        f"-I{output}",
        str(ROOT / "quactlize/dispatch/binding.cpp"),
        "-ldl",
        "-o",
        str(host),
    ]
    subprocess.run(command, check=True)
    execution = args.execution_bundle.resolve() / "libquactlize_ppu_execution.so"
    receipt = json.loads((args.execution_bundle / "manifest.json").read_text())
    if receipt.get("schema") != "quactlize.kpack-execution-build.v1" or sha(
        execution
    ) != receipt.get("sha256"):
        raise ValueError("execution payload identity differs")
    for name, value in receipt["runtime"].items():
        if sha(args.sdk / "lib" / name) != value:
            raise ValueError(f"execution SDK runtime differs: {name}")
    shutil.copy2(execution, output / execution.name)
    source_paths = [
        *sorted((ROOT / "quactlize/dispatch").glob("*")),
        Path(__file__),
        ROOT / "tools/kpack_native_policy.cpp",
        ROOT / "policies/kpack_zw810_heuristic_v1.hpp",
        ROOT / "policies/kpack_zw810_runtime_v1.hpp",
    ]
    manifest = dict(
        schema="quactlize.kpack-native-dispatch.v1",
        modules=records,
        policy_hashes={
            str(f.relative_to(ROOT)): sha(f) for f in source_paths if f.is_file()
        },
        dispatch_sha256=sha(host),
        execution_sha256=sha(execution),
        compiler_identity=compiler.identity,
        execution_receipt=receipt,
        host_command=command,
        device_validated=False,
        heuristic_admitted=False,
        grouped_profile="device-bounds-proposal-no-router-readback",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"KPACK_DISPATCH_BUILD COMPILED modules={len(records)} root={output} device_validated=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
