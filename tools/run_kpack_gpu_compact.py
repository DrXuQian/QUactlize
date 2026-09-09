#!/usr/bin/env python3
"""Prebuilt-only GPU compact/persistent admission; immutable baseline included."""

import argparse
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import sha
from quactlize.runtime.native import SDK, Module, sdk_identity
from tools.kpack_execution_fixture import IndexedWeights
from tools.kpack_warmup_fixture import activation_values
from tools.run_kpack_grouped_device_gate import graph_bind
from tools.run_kpack_pack_gate import device_identity
from tools.run_kpack_decode_sweep import (
    grouped_data,
    fixture,
    gemm_cell,
    pair_bind,
    simt_cell,
)

SCHEMA = "quactlize.kpack-gpu-compact.v1"
CASES = {12: (512, 2048), 13: (2048, 512)}


def verify(bundle):
    bundle = bundle.resolve(strict=True)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("production_selection_changed") is not False
    ):
        raise ValueError("not the GPU compact experiment")
    records = {}
    for r in manifest["modules"]:
        path = (bundle / r["path"]).resolve(strict=True)
        if (
            not path.is_relative_to(bundle)
            or r["key"] in records
            or sha(path) != r["sha256"]
        ):
            raise ValueError("duplicate/unbound module payload")
        records[r["key"]] = r
    from tools.build_kpack_gpu_compact import plan

    expected = {g["job"]: g for g in plan()}
    groups = manifest["groups"]
    if len(groups) != len(expected) or {g["job"] for g in groups} != set(expected):
        raise ValueError("missing/duplicate experiment group")
    for group in manifest["groups"]:
        if any(group.get(k) != v for k, v in expected[group["job"]].items()):
            raise ValueError("experiment group differs from frozen plan")
        for name in ("ordinary", "persistent"):
            if records[group[name + "_key"]]["parent"] != group[name]:
                raise ValueError("group parent differs from module")
        if (
            group["model"]
            and records[group["baseline_key"]]["parent"] != group["ordinary"]
        ):
            raise ValueError("baseline does not name the exact same parent")
    referenced = {
        g[name]
        for g in groups
        for name in ("ordinary_key", "persistent_key", "baseline_key")
        if name in g
    }
    if referenced != set(records):
        raise ValueError("module union differs from exact plan")
    library = (bundle / manifest["execution"]["library"]).resolve(strict=True)
    if (
        not library.is_relative_to(bundle)
        or sha(library) != manifest["execution"]["sha256"]
    ):
        raise ValueError("unchanged SIMT payload differs")
    return manifest


def jobs(manifest):
    return [g["job"] for g in manifest["groups"]] + ["simt-12", "simt-13"]


def split_counts(k):
    return [s for s in (1, 2, 4, 8) if k % (256 * s) == 0]


def expected_cells(job, manifest):
    if job.startswith("simt-"):
        return 2
    group = next(g for g in manifest["groups"] if g["job"] == job)
    if not group["model"]:
        return 4 * 3
    return (4 + len(split_counts(CASES[group["ordinary"]["qtype"]][1]))) * 5


def variants(group):
    out = []
    if group["model"]:
        out += [
            ("baseline-rect", group["baseline_key"], "device-only", 0, False),
            ("baseline-host-compact", group["baseline_key"], "host-compact", 0, False),
        ]
    return out + [
        ("gpu-compact", group["ordinary_key"], "device-only", 0, True),
        ("persistent-cu1", group["persistent_key"], "device-only", 1, True),
        ("persistent-cu2", group["persistent_key"], "device-only", 2, True),
    ]


def run_group(args, sdk, manifest, group, cells):
    records = {r["key"]: r for r in manifest["modules"]}
    p = group["ordinary"]
    geometries = [(256, 2048, 4)]
    if group["model"]:
        geometries.append((*CASES[p["qtype"]], 256))
    for n, k, e in geometries:
        ids = np.arange(8) * 17
        splits = split_counts(k)
        w = IndexedWeights(
            p["qtype"],
            n,
            k,
            e,
            partial_specs=[(256, s) for s in splits if s > 1],
            partial_experts=range(4) if e == 4 else np.r_[ids, (ids + 1) % e],
        )
        if e == 4:
            # The third profile changes the number of real M tiles, not just
            # expert IDs. Replayed graphs must discard the previous directory.
            profiles = [
                grouped_data(w, r) for r in ([9, 0, 3, 1], [0, 3, 1, 9], [0, 13, 0, 0])
            ]
        else:
            first = fixture(
                w, dict(mode=2, rows=8, topk=8, channels=1 if w.q == 12 else 8)
            )
            first["values"] = activation_values(np.arange(len(first["a"])))
            second = {**first, "expert": (first["expert"] + 1) % e}
            second["golden"] = np.stack(
                [
                    first["values"][first["arows"][i]] @ w.sums[expert]
                    for i, expert in enumerate(second["expert"])
                ]
            )
            second["denom"] = np.stack(
                [
                    np.abs(first["values"][first["arows"][i]]) @ w.abs_sums[expert]
                    for i, expert in enumerate(second["expert"])
                ]
            )
            profiles = [first, second]
        for s in splits:
            # Alternate the baseline/candidate order. All images and all
            # repeated measurements are bound to this run, not old timings.
            arms = variants(group)
            if s in (2, 8):
                arms = list(reversed(arms))
            for name, key, arm, b, directory in arms:
                print(
                    f"GPU_COMPACT_PROGRESS job={group['job']} E={e} split={s} arm={name}",
                    flush=True,
                )
                record = records[key]
                module = Module(record | {"path": str(args.bundle / record["path"])})
                result = gemm_cell(
                    args,
                    sdk,
                    module,
                    w,
                    profiles,
                    s,
                    arm,
                    grid_b=b,
                    gpu_directory=directory,
                )
                result.update(
                    variant=name,
                    build_key=key,
                    metadata=(
                        "RESIDENT_FP16"
                        if p["route"] == "sf-grouped"
                        else "PACKED_UNITS"
                    ),
                )
                cells.append(result)
                print("GPU_COMPACT_CELL " + json.dumps(result), flush=True)


def run_simt(args, sdk, manifest, q, cells):
    library = C.CDLL(
        str(args.bundle / manifest["execution"]["library"]), mode=C.RTLD_LOCAL
    )
    functions = pair_bind(library)
    n, k = CASES[q]
    w = IndexedWeights(q, n, k, 256)
    case = dict(mode=2, rows=8, topk=8, channels=1 if q == 12 else 8)
    recipes = (
        [("scalar", (16, 8, 1)), ("pair", (16, 8, 1))]
        if q == 12
        else [("scalar", (32, 8, 1)), ("pair", (16, 2, 1))]
    )
    for variant, (c, warps, s) in recipes:
        result = simt_cell(
            args,
            sdk,
            functions,
            w,
            case,
            (c, warps, s),
            variant,
        )
        cells.append(result)
        print("GPU_COMPACT_CELL " + json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument(
        "--bundle", type=Path, default=ROOT / "prebuilt/ppu0010/kpack-gpu-compact-v1"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--job")
    p.add_argument("--samples", type=int, default=11)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--graph-repeats", type=int, default=16)
    p.add_argument("--correctness-repeats", type=int, default=7)
    args = p.parse_args()
    counts = {
        k: getattr(args, k)
        for k in (
            "samples",
            "rounds",
            "warmups",
            "graph_repeats",
            "correctness_repeats",
        )
    }
    if min(counts.values()) < 1:
        p.error("counts must be positive")
    args.bundle = args.bundle.resolve(strict=True)
    manifest = verify(args.bundle)
    if args.job and args.job not in jobs(manifest):
        p.error("unknown job")
    sdk = SDK(args.sdk)
    graph_bind(sdk)
    authority = dict(
        manifest=sha(args.bundle / "manifest.json"),
        counts=counts,
        sdk=sdk_identity(args.sdk),
        device=device_identity(sdk),
        source={
            str(path.relative_to(ROOT)): sha(path)
            for path in (
                Path(__file__),
                ROOT / "tools/run_kpack_decode_sweep.py",
                ROOT / "tools/run_kpack_grouped_decode_probe.py",
                ROOT / "tools/run_kpack_grouped_device_gate.py",
                ROOT / "tools/kpack_execution_fixture.py",
                ROOT / "tools/kpack_warmup_fixture.py",
                ROOT / "tools/run_kpack_gemv_gate.py",
                ROOT / "quactlize/runtime/native.py",
                ROOT / "quactlize/execution/native.py",
                ROOT / "reference/gguf_kpack.py",
            )
        },
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.job:
        result = dict(job=args.job, status="FAIL", authority=authority, cells=[])
        start = time.monotonic()
        try:
            if args.job.startswith("simt-"):
                run_simt(args, sdk, manifest, int(args.job[5:]), result["cells"])
            else:
                group = next(g for g in manifest["groups"] if g["job"] == args.job)
                run_group(args, sdk, manifest, group, result["cells"])
            if len(result["cells"]) != expected_cells(args.job, manifest) or any(
                c["status"] != "PASS" for c in result["cells"]
            ):
                raise ValueError("job denominator/correctness differs")
            result["status"] = "PASS"
        except Exception as error:
            result["error"] = str(error)
            traceback.print_exc()
        result["seconds"] = time.monotonic() - start
        (args.output / f"{args.job}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return int(result["status"] != "PASS")
    receipt = args.output / "authority.json"
    if args.resume:
        if json.loads(receipt.read_text()) != authority:
            raise ValueError("resume authority differs")
    elif receipt.exists():
        raise ValueError("output already has a run; use --resume")
    else:
        receipt.write_text(json.dumps(authority, indent=2) + "\n")
    start = time.monotonic()
    results = []
    for job in jobs(manifest):
        output = args.output / f"{job}.json"
        old = (
            json.loads(output.read_text()) if args.resume and output.exists() else None
        )
        if (
            old
            and old.get("job") == job
            and old.get("status") == "PASS"
            and old.get("authority") == authority
            and len(old["cells"]) == expected_cells(job, manifest)
            and all(c.get("status") == "PASS" for c in old["cells"])
        ):
            results.append(old)
            print(f"GPU_COMPACT_REUSE job={job}", flush=True)
            continue
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--sdk",
            str(args.sdk),
            "--bundle",
            str(args.bundle),
            "--output",
            str(args.output),
            "--job",
            job,
        ]
        for name, count in counts.items():
            command += ["--" + name.replace("_", "-"), str(count)]
        log = args.output / f"{job}.log"
        print(f"GPU_COMPACT_START job={job} log={log}", flush=True)
        with log.open("w") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            while True:
                try:
                    rc = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(
                        f"GPU_COMPACT_PROGRESS completed={len(results)}/{len(jobs(manifest))} current={job} elapsed_minutes={(time.monotonic()-start)/60:.1f}",
                        flush=True,
                    )
        row = (
            json.loads(output.read_text())
            if output.exists()
            else dict(job=job, status="FAIL", cells=[], error=f"process rc={rc}")
        )
        if (
            rc
            or row.get("authority") != authority
            or row.get("job") != job
            or len(row.get("cells", [])) != expected_cells(job, manifest)
            or any(c.get("status") != "PASS" for c in row.get("cells", []))
        ):
            row["status"] = "FAIL"
        results.append(row)
        print(
            f"GPU_COMPACT_PROGRESS completed={len(results)}/{len(jobs(manifest))} job={job} status={row['status']}",
            flush=True,
        )
    failed = [r["job"] for r in results if r["status"] != "PASS"]
    measured = sum(len(r["cells"]) for r in results if r["status"] == "PASS")
    expected = sum(expected_cells(job, manifest) for job in jobs(manifest))
    summary = dict(
        status="FAIL" if failed or measured != expected else "PASS",
        failed_jobs=failed,
        measured_cells=measured,
        expected_cells=expected,
        authority=authority,
        jobs=results,
        elapsed_seconds=time.monotonic() - start,
        performance_admission="PENDING_REVIEW",
        production_selection_changed=False,
        scope="INCLUDES_GPU_DIRECTORY_AND_REDUCER_EXCLUDES_LLAMA_ADAPTERS",
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"GPU_COMPACT_DONE status={summary['status']} cells={measured}/{expected} failed={failed} results={args.output}",
        flush=True,
    )
    return int(summary["status"] != "PASS")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
