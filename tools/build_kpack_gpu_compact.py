#!/usr/bin/env python3
"""Package bounded GPU-compact / persistent comparisons against immutable modules."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from quactlize.runtime.compiler import Compiler, sha
from quactlize.runtime.candidates import PARENT_FIELDS
from gen_fully_quantized_grouped_kpack_units import symbol as grouped_symbol
from tools.run_kpack_decode_sweep import verify as verify_baseline

BASELINE = ROOT / "prebuilt/ppu0010/kpack-decode-sweep-v1"
SCHEMA = "quactlize.kpack-gpu-compact.v1"


def plan():
    inventory = json.loads((ROOT / "policies/kpack_zw810_tactics.json").read_text())[
        "parents"
    ]
    rows = []
    for route, q, tm in (
        [("fq-grouped", q, 16) for q in range(10, 15)]
        + [("fq-grouped", q, 8) for q in (12, 13)]
        + [("sf-grouped", q, 16) for q in (12, 13)]
    ):
        matches = [
            p
            for p in inventory.values()
            if (
                p["route"],
                p["qtype"],
                p["tm"],
                p["tn"],
                p["tk"],
                p["wm"],
                p["wn"],
                p["stages"],
                p["dn"],
                p["persistent"],
            )
            == (
                route,
                q,
                tm,
                64,
                256,
                tm,
                16,
                2,
                64,
                0 if route == "fq-grouped" else -1,
            )
        ]
        if len(matches) != 1:
            raise ValueError(f"missing/duplicate frozen parent: {route,q,tm}")
        parent = {k: matches[0][k] for k in PARENT_FIELDS}
        persistent = parent
        if route == "fq-grouped":
            row = SimpleNamespace(
                tile_m=tm, tile_n=64, tactic_tile_k=256, warp_m=tm, warp_n=16, stages=2
            )
            persistent = {
                **parent,
                "persistent": 1,
                "symbol": grouped_symbol(q, row, 64, 1),
            }
        rows.append(
            dict(
                job=f'{"fq" if route=="fq-grouped" else "sf"}-q{q}-tm{tm}',
                ordinary=parent,
                persistent=persistent,
                persistent_origin="GENERATED_SCHEDULE_VARIANT_NOT_HISTORICAL_TIMING",
                model=(route == "fq-grouped" and q in (12, 13)),
            )
        )
    return rows


def build(args):
    baseline = verify_baseline(BASELINE.resolve())
    rows = plan()
    parents = {
        p["symbol"]: p for row in rows for p in (row["ordinary"], row["persistent"])
    }
    compiler = Compiler(args.sdk, args.cache, args.jobs)
    built = compiler.compile_only(
        list(parents.values()),
        progress=lambda done, total: print(
            f"GPU_COMPACT_BUILD modules={done}/{total}", flush=True
        ),
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    modules = {}

    def package(record, base):
        relative = f'modules/{record["key"]}.so'
        source = Path(record["path"])
        if not source.is_absolute():
            source = base / source
        if sha(source) != record["sha256"]:
            raise ValueError("payload differs")
        (output / "modules").mkdir(exist_ok=True)
        shutil.copy2(source, output / relative)
        modules[record["key"]] = {
            k: v for k, v in record.items() if k not in ("path", "cache_hit")
        }
        modules[record["key"]]["path"] = relative
        return record["key"]

    by_parent = {r["parent"]["symbol"]: r for r in built}
    for row in rows:
        row["ordinary_key"] = package(by_parent[row["ordinary"]["symbol"]], ROOT)
        row["persistent_key"] = package(by_parent[row["persistent"]["symbol"]], ROOT)
        if row["model"]:
            old = [r for r in baseline["modules"] if r["parent"] == row["ordinary"]]
            if len(old) != 1:
                raise ValueError("missing immutable baseline")
            row["baseline_key"] = package(old[0], BASELINE)
    execution = baseline["execution"]
    shutil.copy2(
        BASELINE / "libquactlize_ppu_execution.so", output / execution["library"]
    )
    manifest = dict(
        schema=SCHEMA,
        source=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        kernel_identity=compiler.kernel_identity,
        groups=rows,
        modules=list(modules.values()),
        baseline_manifest_sha256=sha(BASELINE / "manifest.json"),
        execution=execution,
        device_validated=False,
        production_selection_changed=False,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"GPU_COMPACT_BUILD COMPILED new={len(built)} total_modules={len(modules)} output={output}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=8)
    build(p.parse_args())
