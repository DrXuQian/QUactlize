#!/usr/bin/env python3
"""Fixed-recipe Xplane/K-pack/shared-A replay. No FQ JIT or config sweep."""
import argparse
import csv
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_cuda.build import digest
from dev.gemv_ppu.astage_source import CASES
from dev.gemv_ppu.build_astage import verify
from dev.gemv_ppu.campaign import parse_cells
from dev.gemv_ppu.run import Bench, Config, Sizes, checked, SDK, device_identity, load_library
from tools.profile_kpack_gpu_compact import acu_launch_command

VARIANTS = ("xplane", "baseline", "shared-a")
ROUNDS = 6


def bind_pair(library, control_query, control_launch):
    query = library.quactlize_kpack_gemv_pair_query_v1
    query.argtypes, query.restype = control_query.argtypes, control_query.restype
    launch = library.quactlize_kpack_gemv_pair_run_v1
    launch.argtypes, launch.restype = control_launch.argtypes, control_launch.restype
    return query, launch


def check_exact(control, candidate):
    if not control or len(control) != len(candidate) or len(control) % 4:
        raise ValueError("raw FP32 comparison extent differs")
    a, b = np.frombuffer(control, dtype="<f4"), np.frombuffer(candidate, dtype="<f4")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("raw comparison contains nonfinite/unwritten output")
    mismatch = np.flatnonzero(a.view("<u4") != b.view("<u4"))
    if mismatch.size:
        i = int(mismatch[0])
        raise ValueError(f"shared-A differs from fixed K-pack: bad={mismatch.size}/{a.size} "
                         f"first={i} want=0x{a.view('<u4')[i]:08x} got=0x{b.view('<u4')[i]:08x}")
    return hashlib.sha256(control).hexdigest()


class AStageBench(Bench):
    """Reuse the original fixture, oracle, allocations, graph and cache protocol."""
    def __init__(self, args):
        super().__init__(args)
        self.control_query, self.control_launch = self.query, self.launch
        self.candidate = C.CDLL(str(args.candidate / "libq4_ppu_astage.so"), mode=C.RTLD_LOCAL)
        self.query, self.launch = bind_pair(self.candidate, self.control_query, self.control_launch)
        marker = self.candidate.q4_ppu_probe
        marker.argtypes = [C.POINTER(C.c_int)] * 3 + [C.c_char_p]
        marker.restype = C.c_int
        l2, sm, warp, name = C.c_int(), C.c_int(), C.c_int(), C.create_string_buffer(256)
        checked(marker(C.byref(l2), C.byref(sm), C.byref(warp), name), "candidate image/marker")
        if (sm.value, warp.value, name.value.decode()) != (self.device["sm"], self.device["warp"], self.device["name"]):
            raise ValueError("candidate reports another device geometry")

    def direct(self, launch, recipe):
        cfg = Config(*recipe)
        self.call.low, self.call.units = self.weight_pointers[0]
        self.r.fill(self.output_base, 0xff, self.output_bytes + 32)
        checked(launch(C.byref(self.call), C.byref(cfg), C.byref(self.arr)), "raw A/B launch")
        self.sdk.synchronize(self.r.stream)
        raw = self.sdk.download(self.output_base, self.output_bytes + 32)
        if raw[:16] != b"\xff" * 16 or raw[-16:] != b"\xff" * 16:
            raise ValueError("raw A/B output guard changed")
        return raw[16:-16]

    def measure(self, recipe):
        # These launches and the changed-A negative happen before all timing
        # and profiler ranges. Matching only the loose dequant oracle is not
        # sufficient for an unchanged-FP32-order A-only experiment.
        cfg = Config(*recipe)
        sizes = Sizes()
        checked(self.query(C.byref(self.call), C.byref(cfg), C.byref(self.arr), C.byref(sizes)),
                "shared-A admitted recipe")
        if sizes.workspace_bytes:
            raise ValueError("A-stage experiment unexpectedly requires a reducer")
        control = self.direct(self.control_launch, recipe)
        raw_hash = check_exact(control, self.direct(self.launch, recipe))
        if self.error() >= .005:
            raise ValueError("pre-timing independent GGUF dot failed")
        check_exact(control, self.direct(self.launch, recipe))
        if not np.any(np.frombuffer(control, dtype="<f4") != 0):
            raise ValueError("A-control fixture has no nonzero output")
        self.r.fill(self.a, 0, self.data["a"].nbytes)
        changed = self.direct(self.launch, recipe)
        if not np.all(np.frombuffer(changed, dtype="<f4") == 0):
            raise ValueError("candidate does not produce zero for zero A")
        checked(self.sdk.lib.hggcMemcpy(self.a, self.data["a"].ctypes.data, self.data["a"].nbytes, 1),
                "restore A")
        self.sdk.synchronize(None)
        row = super().measure(recipe)
        check_exact(control, self.sdk.download(self.output, self.output_bytes))
        return row | dict(raw_control="EXACT_FP32", raw_output_sha256=raw_hash, zero_a_check="PASS")


def parse_result(text, variant, recipe, shape, mode, samples):
    arm = "xplane" if variant == "xplane" else "new"
    rows = parse_cells(text, arm, [recipe], shape, mode, samples)
    row = rows[0]
    if row.get("variant") != variant or row.get("output_type") != "F32" or recipe[2] != 1:
        raise ValueError("wrong experiment arm or precision")
    if variant == "shared-a":
        if (row.get("raw_control") != "EXACT_FP32" or row.get("zero_a_check") != "PASS"
                or not isinstance(row.get("raw_output_sha256"), str)
                or len(row["raw_output_sha256"]) != 64):
            raise ValueError("missing bitwise A/B or zero-A validation")
    return row


def child(args):
    verify(args.candidate, args.bundle, sources=False)
    args.arm = "xplane" if args.variant == "xplane" else "new"
    bench = None
    try:
        bench = AStageBench(args) if args.variant == "shared-a" else Bench(args)
        selected = [r for r in CASES if (r[0], r[1], r[2]) == (bench.n, bench.k, args.mode)]
        if len(selected) != 1:
            raise ValueError("fixture/cache row outside the fixed experiment")
        recipe = selected[0][4 if args.variant == "xplane" else 3]
        row = bench.measure(recipe) | dict(variant=args.variant)
        print("Q4_PPU_CELL " + json.dumps(row), flush=True)
        return 0
    finally:
        if bench:
            bench.close()


def run(args):
    manifest = verify(args.candidate, args.bundle)
    if args.fixtures is None or args.output is None:
        raise ValueError("--fixtures and --output are required")
    cases = [c for c in CASES if not args.anchor_only or c[:2] == (5120, 8192)]
    sdk = SDK(args.sdk)
    device = device_identity(sdk)
    _, geometry = load_library(args, "new")
    device.update(geometry)
    inputs = dict(schema="quactlize.q4-astage-result-authority.v1", cases=cases,
        baseline=digest(args.bundle / "manifest.json"), candidate=digest(args.candidate / "manifest.json"),
        runner=digest(Path(__file__)), protocol=digest(ROOT / "dev/gemv_ppu/run.py"),
        parser=digest(ROOT / "dev/gemv_ppu/campaign.py"), device=device, samples=args.samples, rounds=ROUNDS,
        runtime={name: digest(args.sdk / "lib" / name) for name in manifest["runtime"]},
        fixtures={f"{n}x{k}": digest(args.fixtures / f"q12-n{n}-k{k}-e1-c1.npz") for n, k, *_ in cases})
    differences = [name for name, value in inputs["runtime"].items() if value != manifest["runtime"][name]]
    if differences:
        print("Q4_ASTAGE_SDK_DIFFERENCE recorded=" + ",".join(differences) + " marker_and_numeric_required=1", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    authority = args.output / "authority.json"
    if authority.exists() and json.loads(authority.read_text()) != json.loads(json.dumps(inputs)):
        raise ValueError("resume inputs differ; use a new result directory")
    authority.write_text(json.dumps(inputs, indent=2) + "\n")
    started = time.monotonic()
    report = dict(status="RUNNING", scope="FIXED_RECIPES_A_ONLY_NOT_RETUNED_NOT_PRODUCTION",
                  authority=inputs, cases=[], failures=[], profiles=[])

    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    def command(n, k, mode, variant, profile=False):
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--child", "--sdk", str(args.sdk),
               "--bundle", str(args.bundle), "--candidate", str(args.candidate), "--fixture",
               str(args.fixtures / f"q12-n{n}-k{k}-e1-c1.npz"), "--mode", mode,
               "--variant", variant, "--samples", str(args.samples), "--l2-bytes", str(args.l2_bytes)]
        return cmd + (["--profile"] if profile else [])

    def batch(n, k, mode, variant, recipe, turn):
        cmd = command(n, k, mode, variant)
        log = args.output / f"n{n}-k{k}-{mode}-{variant}-r{turn}.log"
        receipt = log.with_suffix(".json")
        if log.is_file() and receipt.is_file():
            prior = json.loads(receipt.read_text())
            if prior.get("rc") == 0 and prior.get("command") == cmd and prior.get("log_sha256") == digest(log):
                row = parse_result(log.read_text(), variant, recipe, [1, n, k], mode, args.samples)
                if row["device"] != device:
                    raise ValueError("cached device differs")
                return row
        if log.exists():
            log.rename(log.with_name(log.name + f".previous.{time.time_ns()}"))
        print(f"Q4_ASTAGE_PROGRESS shape=1x{n}x{k} mode={mode} arm={variant} "
              f"round={turn+1}/{ROUNDS} elapsed_s={time.monotonic()-started:.1f}", flush=True)
        with log.open("w") as stream:
            process = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        receipt.write_text(json.dumps(dict(command=cmd, rc=process.returncode, log_sha256=digest(log)), indent=2) + "\n")
        if process.returncode:
            raise ValueError(f"child rc={process.returncode}; log={log}")
        row = parse_result(log.read_text(), variant, recipe, [1, n, k], mode, args.samples)
        if row["device"] != device:
            raise ValueError("child device differs")
        return row

    for n, k, mode, kpack_recipe, xplane_recipe in cases:
        records = {variant: [] for variant in VARIANTS}
        failed = set()
        for turn in range(ROUNDS):
            for variant in (VARIANTS if turn % 2 == 0 else tuple(reversed(VARIANTS))):
                if variant in failed:
                    continue
                recipe = xplane_recipe if variant == "xplane" else kpack_recipe
                try:
                    records[variant].append(batch(n, k, mode, variant, recipe, turn))
                except Exception as exc:
                    failed.add(variant)
                    failure = dict(shape=[1, n, k], mode=mode, arm=variant, round=turn, error=str(exc))
                    report["failures"].append(failure)
                    print("Q4_ASTAGE_FAILURE " + json.dumps(failure), flush=True)
                save()
        medians = {variant: statistics.median(r["median_us"] for r in rows)
                   for variant, rows in records.items() if len(rows) == ROUNDS}
        row = dict(shape=[1, n, k], mode=mode, recipes=dict(kpack=kpack_recipe, xplane=xplane_recipe),
                   median_us=medians, records=records, verdict="INCOMPLETE")
        if len(medians) == len(VARIANTS):
            row.update(shared_vs_baseline_pct=100*(medians["shared-a"]/medians["baseline"]-1),
                       shared_vs_xplane_pct=100*(medians["shared-a"]/medians["xplane"]-1))
            row["verdict"] = "WITHIN_5_PERCENT" if row["shared_vs_xplane_pct"] <= 5 else "PARITY_OPEN"
        report["cases"].append(row)
        print("Q4_ASTAGE_RESULT " + json.dumps({key: value for key, value in row.items() if key != "records"}), flush=True)
        save()

    # Forced-cold counters are separate from warm/rotating event timing.
    if not args.skip_acu:
        for variant in VARIANTS:
            prefix = args.output / f"n5120-k8192-{variant}.acu"
            log = prefix.with_suffix(".acu.log")
            cmd = list(map(str, acu_launch_command(args.acu or args.sdk / "asight/bin/acu", prefix,
                                                  command(5120, 8192, "warm", variant, True))))
            receipt = prefix.with_suffix(".acu.json")
            info = dict(arm=variant, shape=[1, 5120, 8192], cache="FORCED_COLD", timing_authority=False)
            recipe = (2, 8, 1) if variant == "xplane" else (4, 8, 1)
            try:
                if receipt.is_file() and log.is_file():
                    cached = json.loads(receipt.read_text())
                    files = cached.get("files", [])
                    if (cached.get("status") == "PASS" and cached.get("command") == cmd
                            and cached.get("log_sha256") == digest(log) and files
                            and all(Path(f["file"]).name == f["file"] and
                                    digest(args.output / f["file"]) == f["sha256"] for f in files)):
                        parse_result(log.read_text(), variant, recipe, [1, 5120, 8192], "warm", 0)
                        report["profiles"].append(cached)
                        continue
                print(f"Q4_ASTAGE_ACU arm={variant} shape=1x5120x8192 cache=FORCED_COLD", flush=True)
                for previous in args.output.glob(prefix.name + "*"):
                    if previous.is_file():
                        previous.rename(previous.with_name(previous.name + f".previous.{time.time_ns()}"))
                with log.open("w") as stream:
                    process = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
                if process.returncode:
                    raise ValueError(f"ACU rc={process.returncode}; log={log}")
                parse_result(log.read_text(), variant, recipe, [1, 5120, 8192], "warm", 0)
                files = [p for p in args.output.glob(prefix.name + "*") if p.suffix in (".acurep", ".report")]
                if not files:
                    raise ValueError("ACU produced no report")
                info.update(status="PASS", command=cmd, log_sha256=digest(log),
                            files=[dict(file=p.name, sha256=digest(p)) for p in files])
            except Exception as exc:
                info.update(status="FAIL", error=str(exc))
            receipt.write_text(json.dumps(info, indent=2) + "\n")
            report["profiles"].append(info)
            save()

    verify(args.candidate, args.bundle)
    if digest(args.candidate / "manifest.json") != inputs["candidate"]:
        raise ValueError("candidate changed while measuring")
    if digest(args.bundle / "manifest.json") != inputs["baseline"]:
        raise ValueError("control changed while measuring")
    if any(digest(args.fixtures / f"q12-n{n}-k{k}-e1-c1.npz") != inputs["fixtures"][f"{n}x{k}"]
           for n, k, *_ in cases):
        raise ValueError("fixture changed while measuring")
    if any(digest(args.sdk / "lib" / name) != value for name, value in inputs["runtime"].items()):
        raise ValueError("runtime changed while measuring")
    report.update(status="PASS" if not report["failures"] and all(p["status"] == "PASS" for p in report["profiles"])
                  else "INCOMPLETE", elapsed_seconds=time.monotonic()-started)
    with (args.output / "summary.tsv").open("w") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["M", "N", "K", "cache", *[a+"_us" for a in VARIANTS],
                         "shared_vs_baseline_pct", "shared_vs_xplane_pct", "verdict"])
        for row in report["cases"]:
            writer.writerow([*row["shape"], row["mode"], *[row["median_us"].get(a, "NA") for a in VARIANTS],
                             row.get("shared_vs_baseline_pct", "NA"), row.get("shared_vs_xplane_pct", "NA"), row["verdict"]])
    save()
    print(f"Q4_ASTAGE_COMPLETE status={report['status']} cells={len(cases)} results={args.output}", flush=True)
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT / "prebuilt/ppu0010/q4-simt-ab-v1")
    parser.add_argument("--candidate", type=Path, default=ROOT / "prebuilt/ppu0010/q4-astage-v1")
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--mode", choices=("warm", "rotating"))
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--l2-bytes", type=int, default=0)
    parser.add_argument("--acu", type=Path)
    parser.add_argument("--skip-acu", action="store_true")
    parser.add_argument("--anchor-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    for name in ("sdk", "bundle", "candidate", "fixtures", "fixture"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve(strict=True))
    if args.output is not None:
        args.output = args.output.resolve()
    if args.samples < 3 or args.l2_bytes < 0:
        parser.error("invalid sample count or L2 capacity")
    if args.child and (args.fixture is None or args.variant is None or args.mode is None):
        parser.error("child needs --fixture, --variant and --mode")
    return child(args) if args.child else run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
