#!/usr/bin/env python3
"""Build only grouped post-operation candidates; keep immutable compact controls."""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import Compiler, sha
from tools.build_kpack_gpu_compact import plan as compact_plan
from tools.run_kpack_gpu_compact import verify as verify_compact

BASELINE = ROOT / "prebuilt/ppu0010/kpack-gpu-compact-v1"
SCHEMA = "quactlize.kpack-grouped-postops.v1"


def plan():
    groups = []
    for item in compact_plan():
        if not item["ordinary"]["route"].startswith("fq"):
            continue
        for schedule in (
            ("ordinary", "persistent") if item["ordinary"]["tm"] == 8 else ("ordinary",)
        ):
            groups.append(
                dict(
                    job=item["job"] + "-" + schedule,
                    source_job=item["job"],
                    schedule=schedule,
                    parent=item[schedule],
                    model=item["ordinary"]["tm"] == 8,
                )
            )
    return groups


def build(args):
    old = verify_compact(BASELINE)
    groups = plan()
    compiler = Compiler(args.sdk, args.cache, args.jobs)
    built = compiler.compile_only(
        [g["parent"] for g in groups],
        progress=lambda d, t: print(
            f"GROUPED_POSTOPS_BUILD modules={d}/{t}", flush=True
        ),
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "modules").mkdir()
    modules = {}

    def package(record, base):
        source = Path(record["path"])
        if not source.is_absolute():
            source = base / source
        if sha(source) != record["sha256"]:
            raise ValueError("payload differs")
        path = f"modules/{record['key']}.so"
        shutil.copy2(source, output / path)
        modules[record["key"]] = {
            k: v for k, v in record.items() if k not in ("path", "cache_hit")
        }
        modules[record["key"]]["path"] = path
        return record["key"]

    for g in groups:
        source = next(x for x in old["groups"] if x["job"] == g["source_job"])
        key = source[g["schedule"] + "_key"]
        g["baseline"] = package(
            next(x for x in old["modules"] if x["key"] == key), BASELINE
        )
        g["candidate"] = package(
            next(x for x in built if x["parent"] == g["parent"]), ROOT
        )
    manifest = dict(
        schema=SCHEMA,
        groups=groups,
        modules=list(modules.values()),
        baseline_manifest_sha256=sha(BASELINE / "manifest.json"),
        kernel_identity=compiler.kernel_identity,
        device_validated=False,
        production_selection_changed=False,
        changes=["DIRECT_FP32_PARTIAL", "COMPACT_FIXED_S_REDUCER"],
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"GROUPED_POSTOPS_BUILD COMPLETE modules={len(modules)} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=8)
    build(p.parse_args())
