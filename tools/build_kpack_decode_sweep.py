#!/usr/bin/env python3
"""Build a bounded grouped-decode experiment, without changing runtime bundles."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import Compiler, sha
from quactlize.runtime.candidates import PARENT_FIELDS
from tools.build_kpack_execution import build as build_execution


def parents():
    # Take actual measured parent records, including the two deployed winners
    # and TM8. Split is a runtime axis, not another compile-time Cartesian axis.
    inventory = json.loads((ROOT / "policies/kpack_zw810_tactics.json").read_text())[
        "parents"
    ]
    geometries = {
        12: [
            (16, 16, 16),
            (16, 64, 64),
            (8, 32, 32),
            (8, 64, 16),
            (8, 64, 32),
            (8, 64, 64),
        ],
        13: [(16, 32, 32), (16, 64, 64), (8, 16, 16), (8, 32, 32), (8, 64, 64)],
    }
    result = []
    for q, rows in geometries.items():
        for tm, tn, dn in rows:
            found = [
                p
                for p in inventory.values()
                if (
                    p["qtype"],
                    p["route"],
                    p["persistent"],
                    p["tm"],
                    p["tn"],
                    p["tk"],
                    p["wm"],
                    p["wn"],
                    p["stages"],
                    p["ap"],
                    p["dn"],
                )
                == (q, "fq-grouped", 0, tm, tn, 256, tm, 16, 2, 0, dn)
            ]
            if len(found) != 1:
                raise ValueError(f"measured parent is absent/ambiguous: {q,tm,tn,dn}")
            result.append({k: found[0][k] for k in PARENT_FIELDS})
    return result


def inspect_simt(sdk, library):
    result = []
    for q in (12, 13):
        for pair in (0, 1):
            symbol = f"_ZN9kpack_q{q}10kpack_gemvILi16ELi8ELb{pair}EEEv11qkg_call_v1i"
            resource = subprocess.check_output(
                [
                    str(sdk / "bin/hgobjdump"),
                    "--dump-resource-usage=" + symbol,
                    str(library),
                ],
                text=True,
            )
            assembly = subprocess.check_output(
                [str(sdk / "bin/hgobjdump"), "--dump-function=" + symbol, str(library)],
                text=True,
            )
            ops = re.findall(
                r"^\s*[0-9a-f]+:(?:\s+[0-9a-f]{2}){8}\s+(\S+)", assembly, re.M
            )
            if not ops:
                raise ValueError("inspector did not decode the exact SIMT symbol")
            fields = {
                k: int(re.search(re.escape(k) + r":(\d+)", resource)[1])
                for k in ("vreg_number", "sreg_number", "mma_en", "STACK SIZE")
            }
            if fields["mma_en"] != 0:
                raise ValueError("SIMT subject unexpectedly enables MMA")
            result.append(
                dict(
                    q=q,
                    pair=bool(pair),
                    symbol=symbol,
                    resource=fields,
                    static_instructions=len(ops),
                    static_opcodes=dict(Counter(ops)),
                )
            )
    return dict(
        scope="STATIC_FUNCTION_COUNTS_NOT_DYNAMIC_INSTRUCTIONS_OR_TIMING", rows=result
    )


def build(args):
    start = time.monotonic()
    compiler = Compiler(args.sdk, args.cache, args.jobs)
    records = compiler.compile_only(
        parents(),
        progress=lambda done, total: print(
            f"KPACK_DECODE_BUILD modules={done}/{total}", flush=True
        ),
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.execution:
        execution_dir = args.execution.resolve()
        execution = json.loads((execution_dir / "manifest.json").read_text())
        if execution["sha256"] != sha(execution_dir / execution["library"]) or any(
            sha(ROOT / name) != value
            for name, value in execution["source_hashes"].items()
        ):
            raise ValueError("execution payload/source receipt differs")
    else:
        execution_dir = output / "execution-build"
        execution = build_execution(args.sdk, execution_dir, args.jobs)
    shutil.copy2(
        execution_dir / execution["library"], output / "libquactlize_ppu_execution.so"
    )
    packed = []
    for record in records:
        relative = f"modules/{record['key']}.so"
        (output / "modules").mkdir(exist_ok=True)
        shutil.copy2(record["path"], output / relative)
        packed.append(
            {k: v for k, v in record.items() if k not in ("path", "cache_hit")}
            | {"path": relative}
        )
    manifest = dict(
        schema="quactlize.kpack-decode-sweep.v1",
        modules=packed,
        source=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        kernel_identity=compiler.kernel_identity,
        execution=execution,
        policy_sha256=sha(ROOT / "policies/kpack_zw810_tactics.json"),
        compile_seconds=time.monotonic() - start,
        device_validated=False,
        production_selection_changed=False,
        codegen=inspect_simt(
            args.sdk.resolve(), output / "libquactlize_ppu_execution.so"
        ),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"KPACK_DECODE_BUILD COMPILED modules={len(records)} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execution", type=Path)
    p.add_argument("--jobs", type=int, default=8)
    build(p.parse_args())
