#!/usr/bin/env python3
"""One warmed, numerically checked grouped call for native ACU reports.

This is the admitted standalone .so path, not a llama.cpp model trace.
No compilation or production selector change occurs here.
"""

import argparse
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import sha
from quactlize.runtime.native import SDK, Module, checked, sdk_identity
from tools.kpack_execution_fixture import IndexedWeights
from tools.kpack_warmup_fixture import activation_values
from tools.run_kpack_decode_sweep import fixture, gemm_cell
from tools.run_kpack_gpu_compact import verify, variants
from tools.run_kpack_grouped_device_gate import graph_bind
from tools.run_kpack_pack_gate import device_identity

CASES = {"q4-up": (12, 512, 2048), "q5-down": (13, 2048, 512)}
ARMS = {"baseline": "baseline-rect", "compact": "gpu-compact"}
TILE_M = 8
CAPTURES = (
    ("q4-up", "baseline", 1),
    ("q4-up", "compact", 2),
    ("q4-up", "compact", 4),
    ("q5-down", "baseline", 1),
    ("q5-down", "compact", 1),
)


def capture_plan(case=None, arm=None, split=None):
    selected = [
        row
        for row in CAPTURES
        if all(
            value is None or value == field
            for value, field in zip((case, arm, split), row)
        )
    ]
    if not selected:
        raise ValueError("requested case/arm/split is outside the focused capture plan")
    return selected


def selection(manifest, case, arm, split=None):
    if split is None:
        split = 2 if case == "q4-up" and arm == "compact" else 1
    capture_plan(case, arm, split)
    q, n, k = CASES[case]
    groups = [g for g in manifest["groups"] if g["job"] == f"fq-q{q}-tm{TILE_M}"]
    if len(groups) != 1:
        raise ValueError("missing/duplicate TM8 profile parent; no TM16 fallback")
    group = groups[0]
    variant = next(v for v in variants(group) if v[0] == ARMS[arm])
    name, key, mode, grid_b, directory = variant
    module = next(r for r in manifest["modules"] if r["key"] == key)
    if module["parent"]["tm"] != TILE_M or module["parent"]["wm"] != TILE_M:
        raise ValueError("single-token decode profiling requires TM8/WM8")
    return dict(
        case=case,
        arm=arm,
        tile_m=TILE_M,
        q=q,
        n=n,
        k=k,
        split=split,
        variant=name,
        module=module,
        mode=mode,
        grid_b=grid_b,
        directory=directory,
        expected_components=["metadata"]
        + (["directory"] if directory else [])
        + ["gemm"]
        + (["reducer"] if split > 1 else []),
    )


def report_name(choice):
    return f"{choice['case']}-tm{choice['tile_m']}-{choice['arm']}-s{choice['split']}"


class AcuRange:
    def __init__(self, sdk):
        self.start, self.stop = sdk.lib.hggcProfilerStart, sdk.lib.hggcProfilerStop
        for fn in (self.start, self.stop):
            fn.argtypes, fn.restype = [], C.c_int

    def __enter__(self):
        checked(self.start(), "ACU profiler start")
        print("GPU_COMPACT_ACU_RANGE begin warmed_graph_calls=1", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        rc = self.stop()
        print(f"GPU_COMPACT_ACU_RANGE end stop_rc={rc}", flush=True)
        if exc_type is None:
            checked(rc, "ACU profiler stop")
        return False


def acu_command(acu, report, python, sdk, bundle, output, case, arm, split=None):
    # These flags are from SDK 2.1.1 acu --help. 'none' cache control still
    # clears L1/L2 in this SDK; use explicit 'all' and do not call it warm timing.
    command = [
        str(acu),
        "--set",
        "full",
        "--profile-from-start",
        "no",
        "--graph-profiling",
        "node",
        "--replay-mode",
        "kernel",
        "--cache-control",
        "all",
        "--kill",
        "no",
        "--check-exit-code",
        "yes",
        "--export",
        str(report),
        str(python),
        "-u",
        str(Path(__file__).resolve()),
        "--sdk",
        str(sdk),
        "--bundle",
        str(bundle),
        "--output",
        str(output),
        "--case",
        case,
        "--arm",
        arm,
    ]
    if split is not None:
        command += ["--split", str(split)]
    return command


def collect(args):
    planned = capture_plan(
        getattr(args, "case", None),
        getattr(args, "arm", None),
        getattr(args, "split", None),
    )
    total = len(planned)
    manifest = verify(args.bundle)
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.acu.is_file():
        raise ValueError(f"missing ACU executable: {args.acu}")
    records = []
    print(f"GPU_COMPACT_ACU_PLAN reports={total} selections={planned}", flush=True)
    for case in CASES:
        for arm, split in [(a, s) for c, a, s in planned if c == case]:
            choice = selection(manifest, case, arm, split)
            name = report_name(choice)
            report = args.output / name
            receipt = args.output / f"{name}.json"
            log = args.output / f"{name}.log"
            command = acu_command(
                args.acu,
                report,
                sys.executable,
                args.sdk,
                args.bundle,
                receipt,
                case,
                arm,
                split,
            )
            print(
                f"GPU_COMPACT_ACU_CAPTURE start={name} completed={len(records)}/{total} log={log}",
                flush=True,
            )
            started = time.monotonic()
            with log.open("x") as stream:
                proc = subprocess.Popen(
                    command, stdout=stream, stderr=subprocess.STDOUT
                )
                while True:
                    try:
                        rc = proc.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        print(
                            f"GPU_COMPACT_ACU_CAPTURE current={name} elapsed_seconds={time.monotonic()-started:.0f}",
                            flush=True,
                        )
            row = dict(
                case=case,
                arm=arm,
                tile_m=choice["tile_m"],
                split=choice["split"],
                command=command,
                status="FAIL",
                process_rc=rc,
                log=log.name,
                report=None,
            )
            try:
                candidates = [
                    p
                    for p in (report, Path(str(report) + ".acurep"))
                    if p.is_file() and p.stat().st_size
                ]
                if rc or len(candidates) != 1 or not receipt.is_file():
                    raise ValueError("capture/receipt missing or process failed")
                data = json.loads(receipt.read_text())
                if (
                    data.get("status") != "PASS"
                    or data.get("selection") != choice
                    or data.get("manifest_sha256") != sha(args.bundle / "manifest.json")
                ):
                    raise ValueError("capture receipt identity/correctness differs")
                if (
                    "no kernels were profiled"
                    in log.read_text(errors="replace").lower()
                ):
                    raise ValueError("ACU recorded no kernels")
                row.update(
                    status="CAPTURED",
                    report=candidates[0].name,
                    report_sha256=sha(candidates[0]),
                    receipt=receipt.name,
                )
            except (OSError, ValueError) as error:
                row["error"] = str(error)
                print(
                    f"GPU_COMPACT_ACU_CAPTURE FAIL name={name} error={error} log={log}",
                    flush=True,
                )
            records.append(row)
            print(
                f"GPU_COMPACT_ACU_CAPTURE completed={len(records)}/{total} status={row['status']} report={row['report']}",
                flush=True,
            )
    summary = dict(
        status=(
            "CAPTURED" if all(r["status"] == "CAPTURED" for r in records) else "FAIL"
        ),
        captures=records,
        expected_reports=total,
        scope="STANDALONE_NOT_LLAMA_CPP",
        performance_admission=False,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "acu-index.tsv").open("w") as f:
        f.write("case\ttile_m\tarm\tsplit\tstatus\treport\n")
        for r in records:
            f.write(
                "\t".join(
                    str(r[k])
                    for k in ("case", "tile_m", "arm", "split", "status", "report")
                )
                + "\n"
            )
    print(
        f"GPU_COMPACT_ACU_DONE status={summary['status']} reports={sum(r['status']=='CAPTURED' for r in records)}/{total} output={args.output}",
        flush=True,
    )
    return int(summary["status"] != "CAPTURED")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument(
        "--bundle", type=Path, default=ROOT / "prebuilt/ppu0010/kpack-gpu-compact-v1"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", choices=CASES)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--split", type=int, choices=(1, 2, 4))
    p.add_argument(
        "--collect",
        action="store_true",
        help="Collect the focused ACU reports, optionally filtered by case/arm/split",
    )
    p.add_argument("--acu", type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error("output exists; select a fresh report directory")
    args.bundle = args.bundle.resolve(strict=True)
    args.sdk = args.sdk.resolve(strict=True)
    args.output = args.output.resolve()
    if args.collect:
        try:
            capture_plan(args.case, args.arm, args.split)
        except ValueError as error:
            p.error(str(error))
        args.acu = (args.acu or args.sdk / "asight/bin/acu").resolve(strict=True)
        return collect(args)
    if not args.case or not args.arm:
        p.error("a profiled child requires both --case and --arm")
    manifest = verify(args.bundle)
    choice = selection(manifest, args.case, args.arm, args.split)
    sdk = SDK(args.sdk)
    graph_bind(sdk)
    record = choice["module"]
    module = Module(record | {"path": str(args.bundle / record["path"])})
    q, n, k, split = (choice[x] for x in ("q", "n", "k", "split"))
    ids = tuple(int(x) for x in range(0, 8 * 17, 17))
    print(
        f"GPU_COMPACT_ACU fixture case={args.case} arm={args.arm} q={q} N={n} K={k} E=256",
        flush=True,
    )
    w = IndexedWeights(
        q,
        n,
        k,
        256,
        partial_specs=[(256, split)] if split > 1 else [],
        partial_experts=ids,
        progress=lambda done, total: print(
            f"GPU_COMPACT_ACU fixture_experts={done}/{total}", flush=True
        ),
    )
    profile = fixture(w, dict(mode=2, rows=8, topk=8, channels=1 if q == 12 else 8))
    profile["values"] = activation_values(range(len(profile["a"])))
    options = SimpleNamespace(
        correctness_repeats=1, warmups=3, rounds=1, samples=1, graph_repeats=1
    )
    print(
        f"GPU_COMPACT_ACU selected parent={record['parent']['symbol']} split={split} build_key={record['key']} components={','.join(choice['expected_components'])}",
        flush=True,
    )
    result = gemm_cell(
        options,
        sdk,
        module,
        w,
        [profile],
        split,
        choice["mode"],
        grid_b=choice["grid_b"],
        gpu_directory=choice["directory"],
        profile_context=AcuRange(sdk),
    )
    # ACU replay/cache management changes duration. Do not publish the event
    # timer from this capture as another performance or bandwidth measurement.
    for field in (
        "median_us",
        "min_us",
        "max_us",
        "graph_elapsed_samples_us",
        "effective_weight_GB_per_s",
    ):
        result.pop(field, None)
    receipt = dict(
        status="PASS",
        case=args.case,
        arm=args.arm,
        selection=choice,
        manifest_sha256=sha(args.bundle / "manifest.json"),
        sdk=sdk_identity(args.sdk),
        device=device_identity(sdk),
        result=result,
        capture="ONE_WARMED_GRAPH_INDIVIDUAL_KERNEL_NODES",
        cache_control="ACU_ALL_NOT_THE_UNPROFILED_BENCHMARK",
        scope="STANDALONE_GROUPED_CALL_NOT_LLAMA_ADAPTERS_OR_MODEL",
        sources={
            str(path.relative_to(ROOT)): sha(path)
            for path in (Path(__file__), ROOT / "tools/run_kpack_decode_sweep.py")
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")
    print(
        f"GPU_COMPACT_ACU correctness=PASS case={args.case} arm={args.arm} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
