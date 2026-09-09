#!/usr/bin/env python3
"""Numerical and same-parent timing A/B for grouped direct store + reducer."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import Compiler, sha
from quactlize.runtime.tuning import digest
from quactlize.runtime.native import SDK, Module, sdk_identity
from tools.build_kpack_grouped_postops import SCHEMA, plan
from tools.kpack_execution_fixture import IndexedWeights
from tools.kpack_warmup_fixture import activation_values
from tools.run_kpack_decode_sweep import fixture, grouped_data, gemm_cell
from tools.run_kpack_grouped_device_gate import graph_bind
from tools.run_kpack_pack_gate import device_identity
from tools.profile_kpack_gpu_compact import AcuRange, acu_launch_command
from tools.run_kpack_grouped_decode_probe import timing_summary


def verify(bundle):
    bundle = bundle.resolve(strict=True)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("production_selection_changed") is not False
    ):
        raise ValueError("wrong postops manifest")
    expected = {g["job"]: g for g in plan()}
    if len(manifest["groups"]) != len(expected):
        raise ValueError("group denominator differs")
    records = {}
    for r in manifest["modules"]:
        path = (bundle / r["path"]).resolve(strict=True)
        if (
            r["key"] in records
            or not path.is_relative_to(bundle)
            or sha(path) != r["sha256"]
        ):
            raise ValueError("module identity/path/hash differs")
        source = Compiler.source(None, r["parent"], "")
        if (
            digest(dict(identity=r["identity"], parent=r["parent"], source=source))
            != r["key"]
        ):
            raise ValueError("parent/source identity differs")
        records[r["key"]] = r | {"path": str(path)}
    used = set()
    for g in manifest["groups"]:
        if g["job"] not in expected or any(
            g[k] != v for k, v in expected.pop(g["job"]).items()
        ):
            raise ValueError("unexpected/duplicate group")
        if g["baseline"] == g["candidate"]:
            raise ValueError("baseline is candidate")
        for arm in ("baseline", "candidate"):
            if records[g[arm]]["parent"] != g["parent"]:
                raise ValueError("A/B parents differ")
            used.add(g[arm])
        if records[g["candidate"]]["identity"]["kernel"] != manifest["kernel_identity"]:
            raise ValueError("candidate kernel identity differs")
    if used != set(records):
        raise ValueError("module union differs")
    return manifest, records


def cases_for(group):
    q = group["parent"]["qtype"]
    cases = [("ragged", 256, 2048, 4)]
    if group["model"]:
        cases.append(("model", 512 if q == 12 else 2048, 2048 if q == 12 else 512, 256))
    return cases


def expected_keys(group, rounds):
    return {
        (case, s, r, a)
        for case, _, k, _ in cases_for(group)
        for s in (1, 2, 4, 8)
        if k % (256 * s) == 0
        for r in range(rounds)
        for a in ("baseline", "candidate")
    }


def result_complete(result, group, rounds, authority):
    if not isinstance(result, dict):
        return False
    cells = result.get("cells", [])
    if not isinstance(cells, list) or not all(isinstance(c, dict) for c in cells):
        return False
    keys = [
        (c.get("case"), c.get("split"), c.get("round"), c.get("variant")) for c in cells
    ]
    samples = authority["counts"]["samples"]
    repeats = authority["counts"]["correctness_repeats"]

    def cell_complete(cell):
        elapsed = cell.get("graph_elapsed_samples_us")
        return (
            cell.get("status") == "PASS"
            and isinstance(elapsed, list)
            and len(elapsed) == 1
            and isinstance(elapsed[0], list)
            and len(elapsed[0]) == samples
            and all(
                isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                for v in elapsed[0]
            )
            and cell.get("correctness_checks")
            == repeats * (3 if cell.get("case") == "ragged" else 1)
        )

    return (
        result.get("status") == "PASS"
        and result.get("authority") == authority
        and result.get("job") == group["job"]
        and len(keys) == len(set(keys))
        and set(keys) == expected_keys(group, rounds)
        and all(cell_complete(c) for c in cells)
    )


def authority_for(args, sdk):
    return dict(
        manifest=sha(args.bundle / "manifest.json"),
        sdk=sdk_identity(args.sdk),
        device=device_identity(sdk),
        counts=dict(
            rounds=args.rounds,
            samples=args.samples,
            correctness_repeats=args.correctness_repeats,
        ),
        source={
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                Path(__file__),
                ROOT / "tools/run_kpack_decode_sweep.py",
                ROOT / "tools/kpack_execution_fixture.py",
                ROOT / "tools/kpack_warmup_fixture.py",
                ROOT / "tools/run_kpack_gemv_gate.py",
                ROOT / "tools/run_kpack_grouped_decode_probe.py",
                ROOT / "tools/run_kpack_grouped_device_gate.py",
                ROOT / "quactlize/runtime/native.py",
                ROOT / "quactlize/execution/native.py",
                ROOT / "reference/gguf_kpack.py",
            )
        },
    )


def run_job(args, manifest, records, group):
    sdk = SDK(args.sdk)
    graph_bind(sdk)
    result = dict(
        status="FAIL",
        job=group["job"],
        cells=[],
        authority=authority_for(args, sdk),
    )
    q = group["parent"]["qtype"]
    cases = cases_for(group)
    if args.profile:
        cases = [c for c in cases if c[0] == "model"]
    started = time.monotonic()
    try:
        for case, n, k, e in cases:
            splits = [s for s in (1, 2, 4, 8) if k % (256 * s) == 0]
            if args.profile:
                splits = [args.split]
            w = IndexedWeights(
                q,
                n,
                k,
                e,
                partial_specs=[(256, s) for s in splits if s > 1],
                partial_experts=range(4) if e == 4 else np.arange(8) * 17,
            )
            if e == 4:
                profiles = [
                    grouped_data(w, r)
                    for r in ([17, 0, 9, 3], [0, 3, 9, 17], [0, 29, 0, 0])
                ]
            else:
                data = fixture(
                    w, dict(mode=2, rows=8, topk=8, channels=1 if q == 12 else 8)
                )
                data["values"] = activation_values(np.arange(len(data["a"])))
                profiles = [data]
            for s in splits:
                for round_ in range(1 if args.profile else args.rounds):
                    arms = (
                        [args.profile]
                        if args.profile
                        else list(
                            ("baseline", "candidate")[:: 1 if round_ % 2 == 0 else -1]
                        )
                    )
                    outputs, resources = {}, {}
                    for arm in arms:
                        print(
                            f'GROUPED_POSTOPS_PROGRESS job={group["job"]} case={case} S={s} round={round_+1} arm={arm}',
                            flush=True,
                        )
                        opts = SimpleNamespace(
                            correctness_repeats=args.correctness_repeats,
                            warmups=3,
                            graph_repeats=1 if args.profile else 16,
                            rounds=1,
                            samples=1 if args.profile else args.samples,
                        )
                        cell = gemm_cell(
                            opts,
                            sdk,
                            Module(records[group[arm]]),
                            w,
                            profiles,
                            s,
                            "device-only",
                            grid_b=int(group["schedule"] == "persistent"),
                            gpu_directory=True,
                            capture_output=True,
                            profile_context=AcuRange(sdk) if args.profile else None,
                        )
                        outputs[arm] = cell.pop("output_fp16_bits")
                        resources[arm] = (
                            cell["shared_bytes"],
                            cell["workspace_bytes"],
                            cell["valid_ctas"],
                        )
                        cell.update(
                            variant=arm, case=case, round=round_, build_key=group[arm]
                        )
                        if args.profile:
                            cell = {
                                k: v
                                for k, v in cell.items()
                                if k
                                not in (
                                    "median_us",
                                    "min_us",
                                    "max_us",
                                    "graph_elapsed_samples_us",
                                    "effective_GBs",
                                )
                            }
                        result["cells"].append(cell)
                        args.output.write_text(json.dumps(result, indent=2) + "\n")
                    if not args.profile and outputs["baseline"] != outputs["candidate"]:
                        raise ValueError(
                            "same-parent baseline/candidate FP16 output bits differ"
                        )
                    if (
                        not args.profile
                        and resources["baseline"] != resources["candidate"]
                    ):
                        raise ValueError(
                            "shared/workspace/producer-CTA envelope changed"
                        )
        expected = (
            sum(
                2 * len([s for s in (1, 2, 4, 8) if k % (256 * s) == 0]) * args.rounds
                for _, _, k, _ in cases
            )
            if not args.profile
            else 1
        )
        if len(result["cells"]) != expected:
            raise ValueError("cell denominator differs")
        result.update(status="PASS", seconds=time.monotonic() - started)
    except Exception as error:
        result["error"] = str(error)
        traceback.print_exc()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result["status"] == "PASS"


def summarize(results):
    rows = []
    for result in results:
        groups = {}
        for cell in result["cells"]:
            groups.setdefault((cell["case"], cell["split"]), {}).setdefault(
                cell["variant"], []
            ).append(cell)
        for (case, split), arms in groups.items():
            row = dict(
                job=result["job"], case=case, split=split, status=result["status"]
            )
            for arm, cells in arms.items():
                row[arm + "_us"] = timing_summary(
                    [v for c in cells for v in c["graph_elapsed_samples_us"]], 16, 0
                )["median_us"]
            if all(k + "_us" in row for k in ("baseline", "candidate")):
                row["delta_pct"] = (row["candidate_us"] / row["baseline_us"] - 1) * 100
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument(
        "--bundle",
        type=Path,
        default=ROOT / "prebuilt/ppu0010/kpack-grouped-postops-v1",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--job")
    p.add_argument("--profile", choices=("baseline", "candidate"))
    p.add_argument("--split", type=int, choices=(2, 4, 8), default=4)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--samples", type=int, default=11)
    p.add_argument("--correctness-repeats", type=int, default=7)
    p.add_argument("--skip-acu", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if min(args.rounds, args.samples, args.correctness_repeats) < 1:
        p.error("counts must be positive")
    manifest, records = verify(args.bundle)
    if args.profile and not args.job:
        p.error("--profile requires --job")
    if args.job:
        matches = [g for g in manifest["groups"] if g["job"] == args.job]
        if len(matches) != 1:
            p.error("unknown job")
        if args.profile and (
            not matches[0]["model"] or matches[0]["parent"]["qtype"] != 12
        ):
            p.error("profile is Q4 model only")
        return 0 if run_job(args, manifest, records, matches[0]) else 1
    args.output.mkdir(parents=True, exist_ok=args.resume)
    authority = authority_for(args, SDK(args.sdk))
    receipt = args.output / "authority.json"
    if args.resume:
        if not receipt.is_file() or json.loads(receipt.read_text()) != authority:
            raise ValueError("resume authority changed: do not reuse old timings")
    else:
        receipt.write_text(json.dumps(authority, indent=2) + "\n")
    results = []
    for group in manifest["groups"]:
        output = args.output / (group["job"] + ".json")
        if args.resume and output.is_file():
            try:
                previous = json.loads(output.read_text())
            except json.JSONDecodeError:
                previous = {}
            if result_complete(previous, group, args.rounds, authority):
                results.append(previous)
                print("GROUPED_POSTOPS_REUSE job=" + group["job"], flush=True)
                continue
            stamp = f".failed.{time.time_ns()}"
            output.rename(output.with_name(output.name + stamp))
            log = args.output / (group["job"] + ".log")
            if log.is_file():
                log.rename(log.with_name(log.name + stamp))
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--sdk",
            str(args.sdk),
            "--bundle",
            str(args.bundle),
            "--job",
            group["job"],
            "--output",
            str(output),
            "--rounds",
            str(args.rounds),
            "--samples",
            str(args.samples),
            "--correctness-repeats",
            str(args.correctness_repeats),
        ]
        with (args.output / (group["job"] + ".log")).open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print("GROUPED_POSTOPS_RUNNING job=" + group["job"], flush=True)
        try:
            result = json.loads(output.read_text())
        except (OSError, json.JSONDecodeError) as error:
            result = dict(status="FAIL", job=group["job"], cells=[], error=str(error))
        if not isinstance(result, dict) or not isinstance(result.get("cells"), list):
            result = dict(status="FAIL", job=group["job"], cells=[], error="invalid job receipt")
        if process.returncode or not result_complete(
            result, group, args.rounds, authority
        ):
            result["status"] = "FAIL"
        results.append(result)
        print(
            f'GROUPED_POSTOPS_JOB job={group["job"]} status={result["status"]} cells={len(result["cells"])}',
            flush=True,
        )
    rows = summarize(results)
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print("GROUPED_POSTOPS_RESULT " + json.dumps(row), flush=True)
    passed = all(r["status"] == "PASS" for r in results)
    if passed and not args.skip_acu:
        for arm in ("baseline", "candidate"):
            stem = args.output / ("q4-up-s4-" + arm)
            if args.resume:
                for suffix in (".json", ".log", ".acurep"):
                    old = Path(str(stem) + suffix)
                    if old.is_file():
                        old.rename(
                            old.with_name(old.name + f".previous.{time.time_ns()}")
                        )
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--sdk",
                str(args.sdk),
                "--bundle",
                str(args.bundle),
                "--job",
                "fq-q12-tm8-ordinary",
                "--profile",
                arm,
                "--split",
                "4",
                "--output",
                str(stem) + ".json",
            ]
            with Path(str(stem) + ".log").open("w") as log:
                process = subprocess.Popen(
                    acu_launch_command(args.sdk / "bin/acu", stem, command),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                while process.poll() is None:
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        print("GROUPED_POSTOPS_ACU_RUNNING arm=" + arm, flush=True)
            report = Path(str(stem) + ".acurep")
            passed &= (
                process.returncode == 0
                and report.is_file()
                and report.stat().st_size > 0
            )
    print(
        "GROUPED_POSTOPS_COMPLETE status=" + ("PASS" if passed else "FAIL"), flush=True
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
