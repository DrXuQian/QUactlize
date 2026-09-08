#!/usr/bin/env python3
"""Compile ten measured grouped parents with the additive device-only ABI."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import Compiler
from quactlize.runtime.candidates import PARENT_FIELDS


def parents():
    model = json.loads((ROOT / "policies/kpack_zw810_heuristic_v1.json").read_text())
    out = []
    for q in range(10, 15):
        for route in ("fq-grouped", "sf-grouped"):
            candidates = [
                e
                for e in model["entries"]
                if e["context"]["route"] == route
                and e["context"]["problem"]["qtype"] == q
                and e["context"]["problem"]["max_rows"] == 129
            ]
            selected = sorted(candidates, key=lambda e: e["config_id"])[0]
            c = model["configurations"][selected["config_id"]]
            out.append(
                dict(
                    parent={f: c[f] for f in PARENT_FIELDS},
                    provenance=selected,
                    config=c,
                )
            )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selected = parents()
    compiler = Compiler(args.sdk, args.output / "modules", args.jobs)
    records = compiler.compile_only(
        [v["parent"] for v in selected],
        progress=lambda *x: print("KPACK_GROUPED_DEVICE_BUILD", *x, flush=True),
    )
    # Relocatable paths; no copying or rebuilding of the old six-library bundle.
    for record in records:
        record["path"] = str(Path(record["path"]).relative_to(args.output.resolve()))
    manifest = dict(
        schema="quactlize.kpack-grouped-device-build.v2",
        selected=selected,
        modules=records,
        device_validated=False,
        heuristic_admitted=False,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"KPACK_GROUPED_DEVICE_BUILD status=COMPILED modules={len(records)} device_validated=0 output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
