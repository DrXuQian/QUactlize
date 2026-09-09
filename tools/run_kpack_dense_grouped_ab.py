#!/usr/bin/env python3
"""Reproduce Q4 dense M1/N4096/K2048 beside eight N512 grouped projections.

Only existing, explicitly named modules run. The selected expert weights are
concatenated along logical N before timing; no online repack is introduced.
Warm graph timings and ACU replay metrics are separate evidence.
"""

import argparse
import ctypes as C
import hashlib
import json
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
from quactlize.runtime.native import SDK, Module, sdk_identity
from quactlize.runtime.tuning import digest
from tools.kpack_execution_fixture import IndexedWeights
from tools.kpack_warmup_fixture import activation_values
from tools.profile_kpack_gpu_compact import AcuRange, acu_launch_command
from tools.run_kpack_decode_sweep import admit, fixture, gemm_cell
from tools.run_kpack_grouped_decode_probe import timing_summary
from tools.run_kpack_grouped_device_gate import graph_bind
from tools.run_kpack_pack_gate import device_identity

NATIVE = ROOT / "prebuilt/ppu0010/kpack-native-v1"
COMPACT = ROOT / "prebuilt/ppu0010/kpack-gpu-compact-v1"
DENSE_BEST = "fqk_tc_q12_l1_a0_tm8_tn128_tk256_wm8_wn32_s2_bc0_ap1_dn64"
DENSE_MATCHED = "fqk_tc_q12_l1_a0_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap0_dn64"
GROUPED = "fqg_q12_l1_tm8_tn64_tk256_wm8_wn16_s2_ap0_dn64_nonpersistent"
ARMS = {
    "dense-historical-s8": (DENSE_BEST, 8, "dense"),
    "dense-matched-s2": (DENSE_MATCHED, 2, "dense"),
    "dense-matched-s4": (DENSE_MATCHED, 4, "dense"),
    "grouped-compact-s2": (GROUPED, 2, "device-only"),
    "grouped-compact-s4": (GROUPED, 4, "device-only"),
}
PROFILE_ARMS = ("dense-historical-s8", "grouped-compact-s2", "grouped-compact-s4")
WEIGHT_BYTES = 4718592
PEAK_GBS = 2766.0


def selected_modules(native=NATIVE, compact=COMPACT):
    records = {}
    for root, symbols, schema in (
        (native, (DENSE_BEST, DENSE_MATCHED), "quactlize.kpack-native-dispatch.v1"),
        (compact, (GROUPED,), "quactlize.kpack-gpu-compact.v1"),
    ):
        root = root.resolve(strict=True)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("schema") != schema:
            raise ValueError("unexpected module manifest")
        for symbol in symbols:
            candidates = [
                r for r in manifest["modules"] if r["parent"]["symbol"] == symbol
            ]
            if symbol == GROUPED:
                groups = [g for g in manifest["groups"] if g["job"] == "fq-q12-tm8"]
                if len(groups) != 1:
                    raise ValueError("missing/duplicate TM8 compact group")
                candidates = [
                    r for r in candidates if r["key"] == groups[0]["ordinary_key"]
                ]
            if len(candidates) != 1:
                raise ValueError("missing/duplicate exact comparison parent: " + symbol)
            r = candidates[0]
            path = (root / r["path"]).resolve(strict=True)
            source = Compiler.source(None, r["parent"], "")
            if (
                not path.is_relative_to(root)
                or sha(path) != r["sha256"]
                or digest(
                    dict(identity=r["identity"], parent=r["parent"], source=source)
                )
                != r["key"]
            ):
                raise ValueError("selected module payload/identity differs")
            records[symbol] = {
                **r,
                "path": str(path),
                "manifest_sha256": sha(root / "manifest.json"),
            }
    for symbol, r in records.items():
        p = r["parent"]
        expected = (
            "fq-grouped" if symbol == GROUPED else "fq-dense",
            12,
            8,
            128 if symbol == DENSE_BEST else 64,
            256,
            8,
            32 if symbol == DENSE_BEST else 16,
            2,
            1 if symbol == DENSE_BEST else 0,
            64,
            0 if symbol == GROUPED else -1,
        )
        if (
            tuple(
                p[k]
                for k in (
                    "route",
                    "qtype",
                    "tm",
                    "tn",
                    "tk",
                    "wm",
                    "wn",
                    "stages",
                    "ap",
                    "dn",
                    "persistent",
                )
            )
            != expected
        ):
            raise ValueError("comparison parent geometry differs")
    return records


def concatenate_q4_experts(w, ids):
    """Logical N concatenation, not concatenation of flattened packed buffers."""
    ids = tuple(int(e) for e in ids)
    if (
        w.q != 12
        or not ids
        or len(ids) != len(set(ids))
        or any(e < 0 or e >= w.experts for e in ids)
    ):
        raise ValueError("requires distinct in-range Q4 experts")
    if not all(np.array_equal(w.categories[e], w.categories[ids[0]]) for e in ids):
        raise ValueError("dense equivalence requires shared activation K categories")
    planes = {
        name: np.concatenate([a[e] for e in ids], axis=1)[None] if a.size else a.copy()
        for name, a in w.planes.items()
    }
    partials = {
        key: {
            0: tuple(
                np.concatenate([entries[e][i] for e in ids], axis=2) for i in (0, 1)
            )
        }
        for key, entries in w.partial_sums.items()
    }
    return SimpleNamespace(
        q=12,
        n=w.n * len(ids),
        k=w.k,
        experts=1,
        planes=planes,
        categories=w.categories[ids[0] : ids[0] + 1].copy(),
        sums=np.concatenate([w.sums[e] for e in ids], axis=1)[None],
        abs_sums=np.concatenate([w.abs_sums[e] for e in ids], axis=1)[None],
        partial_sums=partials,
    )


def paired_fixture():
    ids = tuple(range(0, 8 * 17, 17))
    grouped = IndexedWeights(
        12,
        512,
        2048,
        256,
        partial_specs=[(256, s) for s in (2, 4, 8)],
        partial_experts=ids,
        progress=lambda done, total: print(
            f"Q4_DENSE_GROUPED_FIXTURE experts={done}/{total}", flush=True
        ),
    )
    dense = concatenate_q4_experts(grouped, ids)
    profiles = {}
    for name, w, case in (
        ("dense", dense, dict(mode=0, rows=1, channels=1, topk=1)),
        ("device-only", grouped, dict(mode=2, rows=8, channels=1, topk=8)),
    ):
        data = fixture(w, case)
        data["values"] = activation_values(range(len(data["a"])))
        profiles[name] = data
        try:
            admit(np.zeros_like(data["golden"]), data, "zero-output negative")
        except ValueError:
            pass
        else:
            raise ValueError("paired fixture cannot reject a zero output")
    if not np.array_equal(
        profiles["dense"]["a"], profiles["device-only"]["a"]
    ) or not np.array_equal(
        profiles["dense"]["golden"].reshape(-1),
        profiles["device-only"]["golden"].reshape(-1),
    ):
        raise ValueError("dense/grouped logical inputs or oracle differ")
    for w in (dense, grouped):
        active = 1 if w.experts == 1 else 8
        if (
            sum(w.planes[k].nbytes for k in ("low", "high", "units"))
            // w.experts
            * active
            != WEIGHT_BYTES
        ):
            raise ValueError("active weight bytes differ")
    receipt = dict(
        dense_shape=[1, 4096, 2048],
        grouped_shape=[8, 512, 2048],
        experts=256,
        active_experts=list(ids),
        per_expert_m=1,
        active_weight_bytes=WEIGHT_BYTES,
        identical_activation=True,
        identical_logical_weights=True,
        packing="CONCATENATE_N_IN_EACH_KPACK_PLANE_OUTSIDE_TIMING",
        input_kind="SYNTHETIC_OFFICIAL_GGUF_NOT_MODEL_TENSOR_DUMP",
        dense_plane_sha256={
            k: hashlib.sha256(a.tobytes()).hexdigest() for k, a in dense.planes.items()
        },
    )
    return {"dense": dense, "device-only": grouped}, profiles, receipt


def sample_arm(args, sdk, records, weights, profiles, name, profiling=False):
    symbol, split, mode = ARMS[name]
    module = Module(records[symbol])
    options = SimpleNamespace(
        correctness_repeats=args.correctness_repeats,
        warmups=args.warmups,
        rounds=1,
        samples=1 if profiling else args.samples,
        graph_repeats=1 if profiling else args.graph_repeats,
    )
    result = gemm_cell(
        options,
        sdk,
        module,
        weights[mode],
        [profiles[mode]],
        split,
        mode,
        gpu_directory=mode == "device-only",
        capture_output=True,
        profile_context=AcuRange(sdk) if profiling else None,
    )
    result.update(
        arm_name=name,
        build_key=module.record["key"],
        module_sha256=module.record["sha256"],
    )
    return result


def summarize(samples):
    rows = {}
    for name in ARMS:
        values = [r for r in samples if r["arm_name"] == name]
        if not values:
            continue
        base = {k: v for k, v in values[-1].items() if k != "output_fp16_bits"}
        calls = {r["calls_per_graph"] for r in values}
        if len(calls) != 1:
            raise ValueError("graph repeat count changed across rounds")
        base.update(
            timing_summary(
                [r["graph_elapsed_samples_us"][0] for r in values],
                calls.pop(),
                WEIGHT_BYTES,
            )
        )
        base["rounds"] = len(values)
        base["effective_weight_MBU_pct"] = (
            base["effective_weight_GB_per_s"] / PEAK_GBS * 100
        )
        rows[name] = base
    if "dense-historical-s8" in rows:
        baseline = rows["dense-historical-s8"]["median_us"]
        for row in rows.values():
            row["delta_vs_dense_historical_pct"] = (
                row["median_us"] / baseline - 1
            ) * 100
    return rows


def benchmark(args, sdk, records, weights, profiles, receipt):
    samples, failures = [], []
    target = args.output / "timing.json"
    result = dict(
        status="INCOMPLETE",
        fixture=receipt,
        samples=samples,
        failures=failures,
        expected_blocks=len(ARMS) * args.rounds,
        peak_GB_per_s=PEAK_GBS,
        cache_scope="WARM_FIXED_RESIDENT_WEIGHTS_NO_CACHE_FLUSH",
        comparison_scope="SAME_LOGICAL_OPERANDS_NOT_IDENTICAL_COLLECTIVES_OR_METADATA_WORK",
        production_selection_changed=False,
    )
    for round_ in range(args.rounds):
        order = list(ARMS)
        if round_ % 2:
            order.reverse()
        for name in order:
            print(
                f"Q4_DENSE_GROUPED_PROGRESS round={round_+1}/{args.rounds} arm={name}",
                flush=True,
            )
            try:
                row = sample_arm(args, sdk, records, weights, profiles, name)
                row["round"] = round_ + 1
                samples.append(row)
            except Exception as error:
                traceback.print_exc()
                failures.append(dict(round=round_ + 1, arm=name, error=str(error)))
            target.write_text(json.dumps(result, indent=2) + "\n")
    result["summary"] = summarize(samples)
    result["status"] = (
        "PASS" if not failures and len(samples) == result["expected_blocks"] else "FAIL"
    )
    target.write_text(json.dumps(result, indent=2) + "\n")
    fields = (
        "arm",
        "median_us",
        "effective_weight_GB_per_s",
        "effective_weight_MBU_pct",
        "delta_vs_dense_historical_pct",
        "rounds",
        "timing_scope",
    )
    with (args.output / "summary.tsv").open("w") as f:
        f.write("\t".join(fields) + "\n")
        for name, row in result["summary"].items():
            data = {**row, "arm": name}
            f.write("\t".join(str(data.get(k, "NA")) for k in fields) + "\n")
            print(
                "Q4_DENSE_GROUPED_RESULT "
                + json.dumps({k: data.get(k) for k in fields}),
                flush=True,
            )
    return result


def capture(args, records):
    captures = []
    for name in PROFILE_ARMS:
        report = args.output / name
        output = args.output / f"{name}.json"
        log = args.output / f"{name}.acu.log"
        command = acu_launch_command(
            args.acu,
            report,
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--sdk",
                str(args.sdk),
                "--native",
                str(args.native),
                "--compact",
                str(args.compact),
                "--output",
                str(output),
                "--profile-arm",
                name,
            ],
        )
        started = time.monotonic()
        print(f"Q4_DENSE_GROUPED_ACU start={name} log={log}", flush=True)
        with log.open("x") as f:
            proc = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT)
            while True:
                try:
                    rc = proc.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(
                        f"Q4_DENSE_GROUPED_ACU running={name} seconds={time.monotonic()-started:.0f}",
                        flush=True,
                    )
        status = "FAIL"
        try:
            receipt = json.loads(output.read_text())
            expected = records[ARMS[name][0]]
            if (
                rc == 0
                and report.with_suffix(".acurep").stat().st_size > 0
                and "No kernels were profiled" not in log.read_text()
                and receipt["status"] == "PASS"
                and receipt["arm_name"] == name
                and receipt["result"]["build_key"] == expected["key"]
                and receipt["result"]["module_sha256"] == expected["sha256"]
            ):
                status = "CAPTURED"
        except (OSError, ValueError, KeyError):
            pass
        captures.append(
            dict(
                arm=name,
                status=status,
                rc=rc,
                report=str(report.with_suffix(".acurep")),
                log=str(log),
            )
        )
        print(
            f"Q4_DENSE_GROUPED_ACU completed={len(captures)}/{len(PROFILE_ARMS)} status={status}",
            flush=True,
        )
    (args.output / "acu-index.json").write_text(json.dumps(captures, indent=2) + "\n")
    return all(r["status"] == "CAPTURED" for r in captures)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--native", type=Path, default=NATIVE)
    p.add_argument("--compact", type=Path, default=COMPACT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=11)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--graph-repeats", type=int, default=16)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--correctness-repeats", type=int, default=2)
    p.add_argument("--skip-acu", action="store_true")
    p.add_argument("--acu", type=Path)
    p.add_argument("--profile-arm", choices=ARMS)
    args = p.parse_args()
    if (
        min(
            args.samples,
            args.rounds,
            args.graph_repeats,
            args.warmups,
            args.correctness_repeats,
        )
        < 1
    ):
        p.error("measurement counts must be positive")
    for name in ("sdk", "native", "compact"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output = args.output.resolve()
    if args.output.exists():
        p.error("output already exists; preserve it and use a fresh directory")
    if not args.profile_arm and not args.skip_acu:
        args.acu = (args.acu or args.sdk / "asight/bin/acu").resolve(strict=True)
    records = selected_modules(args.native, args.compact)
    print("Q4_DENSE_GROUPED_PREBUILT modules=3 compile=NONE", flush=True)
    sdk = SDK(args.sdk)
    graph_bind(sdk)
    weights, profiles, receipt = paired_fixture()
    if args.profile_arm:
        row = sample_arm(
            args, sdk, records, weights, profiles, args.profile_arm, profiling=True
        )
        for field in (
            "median_us",
            "min_us",
            "max_us",
            "graph_elapsed_samples_us",
            "effective_weight_GB_per_s",
        ):
            row.pop(field, None)
        args.output.write_text(
            json.dumps(
                dict(
                    status="PASS",
                    arm_name=args.profile_arm,
                    fixture=receipt,
                    result=row,
                ),
                indent=2,
            )
            + "\n"
        )
        return 0
    args.output.mkdir(parents=True)
    (args.output / "authority.json").write_text(
        json.dumps(
            dict(
                modules=records,
                sdk=sdk_identity(args.sdk),
                device=device_identity(sdk),
                sources={
                    str(path.relative_to(ROOT)): sha(path)
                    for path in (
                        Path(__file__),
                        ROOT / "tools/run_kpack_decode_sweep.py",
                        ROOT / "tools/kpack_execution_fixture.py",
                        ROOT / "tools/kpack_warmup_fixture.py",
                        ROOT / "tools/profile_kpack_gpu_compact.py",
                    )
                },
            ),
            indent=2,
        )
        + "\n"
    )
    result = benchmark(args, sdk, records, weights, profiles, receipt)
    acu_ok = True
    if not args.skip_acu:
        args.acu = (args.acu or args.sdk / "asight/bin/acu").resolve(strict=True)
        acu_ok = capture(args, records)
    ok = result["status"] == "PASS" and acu_ok
    print(
        f"Q4_DENSE_GROUPED_DONE status={'PASS' if ok else 'FAIL'} timing_blocks={len(result['samples'])}/{result['expected_blocks']} acu={'SKIPPED' if args.skip_acu else 'PASS' if acu_ok else 'FAIL'} output={args.output}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
