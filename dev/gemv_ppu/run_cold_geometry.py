#!/usr/bin/env python3
"""Single-shape, rotating-weight C4/C8 A/B plus the two measured controls."""
import argparse
import csv
import ctypes as C
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_cuda.build import digest
from dev.gemv_ppu.cold_geometry import PAYLOAD, RECIPES, SHAPE, geometry, verify
from dev.gemv_ppu.campaign import parse_cells
from dev.gemv_ppu.run import SDK, checked, device_identity, load_library
from dev.gemv_ppu.run_bload import ExperimentBench
from dev.gemv_ppu.run_h800_port import PortBench
from tools.profile_kpack_gpu_compact import acu_launch_command

ARMS = tuple(RECIPES)
ROUNDS = 6
MODE = "rotating"
PREFIX = "Q4_COLD_GEOMETRY_CELL "


class GeometryBench(ExperimentBench):
    def __init__(self, args):
        harness = SimpleNamespace(**(vars(args) | dict(variant="kpack-current", arm="new")))
        super().__init__(harness)
        self.is_c8 = args.variant == "kpack-c8"
        if not self.is_c8:
            return
        self.c8_library = C.CDLL(str(args.candidate / PAYLOAD), mode=C.RTLD_LOCAL)
        probe = self.c8_library.q4_ppu_probe
        probe.argtypes, probe.restype = [C.POINTER(C.c_int)]*3 + [C.c_char_p], C.c_int
        l2, sm, warp, name = C.c_int(), C.c_int(), C.c_int(), C.create_string_buffer(256)
        checked(probe(C.byref(l2), C.byref(sm), C.byref(warp), name), "C8 native image/marker")
        if (sm.value, warp.value, name.value.decode()) != (self.device["sm"], self.device["warp"], self.device["name"]):
            raise ValueError("C8 image/device differs")
        self.c8_launch = self.c8_library.q4_cold_c8_run
        self.c8_launch.argtypes, self.c8_launch.restype = [C.c_int]*2 + [C.c_void_p]*5, C.c_int

    def invoke(self, recipe, index=0, force_aiu=None):
        if not self.is_c8:
            return super().invoke(recipe, index, force_aiu)
        low, units = self.weight_pointers[index % self.copies]
        return self.c8_launch(self.n, self.k, self.a, low, units, self.output, self.r.stream)


def child(args):
    verify(args.candidate, args.controls, args.bundle, sources=False)
    bench = None
    try:
        if args.variant.startswith("kpack-"):
            bench = GeometryBench(args)
        else:
            harness = SimpleNamespace(**(vars(args) | dict(candidate=args.controls)))
            bench = PortBench(harness)
        if (1, bench.n, bench.k) != SHAPE or args.mode != MODE:
            raise ValueError("only M1/N8192/K5120 rotating weights are in scope")
        row = bench.measure(list(RECIPES[args.variant]))
        row.update(arm=args.variant, variant=args.variant, implementation="affine4-early-fast-bare"
                   if args.variant.startswith("kpack-") else args.variant,
                   launches_per_call=1, inter_cta_split=1,
                   timing_scope="RESIDENT_FULL_CALL_NO_HOST_PACK_NO_JIT",
                   cache_scope="ACU_FORCED_COLD" if args.profile else "ROTATING_GT_2_25_L2",
                   weight_arithmetic="FP32_GROUP_AFFINE" if args.variant.startswith("kpack-") else "PER_WEIGHT_FP16")
        if args.variant.startswith("kpack-"):
            row["geometry"] = geometry(RECIPES[args.variant][0])
        print(PREFIX + json.dumps(row), flush=True)
        return 0
    finally:
        if bench:
            bench.close()


def parse_result(text, arm, samples, *, profile=False):
    lines = [line for line in text.splitlines() if line.startswith(PREFIX)]
    converted = "\n".join("Q4_PPU_CELL " + line[len(PREFIX):] for line in lines)
    row = parse_cells(converted, arm, [RECIPES[arm]], list(SHAPE), MODE, samples)[0]
    is_kpack = arm.startswith("kpack-")
    if (row.get("variant") != arm or row.get("output_type") != "F32" or row.get("zero_a_check") != "PASS"
            or row.get("inter_cta_split") != 1 or row.get("launches_per_call") != 1
            or row.get("timing_scope") != "RESIDENT_FULL_CALL_NO_HOST_PACK_NO_JIT"
            or row.get("cache_scope") != ("ACU_FORCED_COLD" if profile else "ROTATING_GT_2_25_L2")
            or row.get("implementation") != ("affine4-early-fast-bare" if is_kpack else arm)
            or row.get("weight_arithmetic") != ("FP32_GROUP_AFFINE" if is_kpack else "PER_WEIGHT_FP16")
            or row.get("storage") != ("CANONICAL_KPACK4" if is_kpack else "RAW_GGUF" if arm == "raw-reference" else "XPLANE")):
        raise ValueError("cold geometry receipt precision/route/cache scope differs")
    if is_kpack and row.get("geometry") != geometry(RECIPES[arm][0]):
        raise ValueError("C4/C8 work distribution differs")
    return row


def summarize(records):
    medians = {arm: statistics.median(r["median_us"] for r in rows)
               for arm, rows in records.items() if len(rows) == ROUNDS}
    complete = set(medians) == set(ARMS)
    result = dict(shape=list(SHAPE), cache=MODE, status="PASS" if complete else "INCOMPLETE",
                  median_us=medians, c8_delta_pct={}, parity_verdict="INCOMPLETE")
    if "kpack-c8" in medians:
        result["c8_delta_pct"] = {arm: 100*(medians["kpack-c8"]/us-1)
                                 for arm, us in medians.items() if arm != "kpack-c8"}
    if complete:
        result["target_us"] = 1.05*min(medians[arm] for arm in ("xplane", "raw-reference"))
        result["parity_verdict"] = "WITHIN_5_PERCENT" if medians["kpack-c8"] <= result["target_us"] else "PARITY_OPEN"
    return result


def probe_device(args):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--probe", "--sdk", str(args.sdk),
               "--bundle", str(args.bundle), "--l2-bytes", str(args.l2_bytes)]
    proc = subprocess.run(command, capture_output=True, text=True)
    lines = [s.split(" ",1)[1] for s in proc.stdout.splitlines() if s.startswith("Q4_COLD_GEOMETRY_DEVICE ")]
    if proc.returncode or len(lines) != 1:
        raise ValueError(f"device probe failed rc={proc.returncode}: {proc.stdout}\n{proc.stderr}")
    return json.loads(lines[0])


def validate_entry(output, entry, arm, samples, device, *, profile=False):
    log = output / entry["log"]
    if log.parent != output or digest(log) != entry["log_sha256"]:
        raise ValueError("cached log path/hash differs")
    row = parse_result(log.read_text(), arm, samples, profile=profile)
    if row != entry["row"] or row["device"] != device:
        raise ValueError("cached row/device differs")
    if profile:
        if len(entry.get("reports", [])) != 1:
            raise ValueError("missing ACU report")
        for file in entry["reports"]:
            path = output / file["file"]
            if path.parent != output or digest(path) != file["sha256"]:
                raise ValueError("ACU report path/hash differs")
    return row


def run(args):
    manifest = verify(args.candidate, args.controls, args.bundle)
    device = probe_device(args)  # its process exits; parent never holds a GPU context
    inputs = [Path(__file__), ROOT / "dev/gemv_ppu/cold_geometry.py", ROOT / "dev/gemv_ppu/run.py",
              ROOT / "dev/gemv_ppu/campaign.py", ROOT / "dev/gemv_ppu/run_bload.py",
              ROOT / "dev/gemv_ppu/run_h800_port.py", ROOT / "tools/run_kpack_gemv_gate.py",
              ROOT / "tools/run_kpack_grouped_decode_probe.py", ROOT / "tools/profile_kpack_gpu_compact.py"]
    authority = dict(schema="quactlize.q4-cold-geometry-run.v1", shape=SHAPE, mode=MODE, rounds=ROUNDS,
        samples=args.samples, recipes=RECIPES, device=device, fixture_sha256=digest(args.fixture),
        candidate=digest(args.candidate / "manifest.json"), controls=digest(args.controls / "manifest.json"),
        baseline=digest(args.bundle / "manifest.json"),
        runtime={name:digest(args.sdk / "lib" / name) for name in manifest["runtime"]},
        sources={str(p.relative_to(ROOT)):digest(p) for p in inputs})
    authority = json.loads(json.dumps(authority))
    args.output.mkdir(parents=True, exist_ok=True)
    identity = args.output / "authority.json"
    if identity.exists() and json.loads(identity.read_text()) != authority:
        raise ValueError("resume source/device/runtime/fixture differs; use a fresh run")
    identity.write_text(json.dumps(authority, indent=2)+"\n")
    differences = [key for key,value in authority["runtime"].items() if manifest["runtime"][key] != value]
    if differences:
        print("Q4_COLD_GEOMETRY_SDK_DIFFERENCE recorded="+",".join(differences)+" real_marker_and_numeric_required=1", flush=True)
    report = dict(status="RUNNING", records={arm:[] for arm in ARMS}, failures=[], profiles=[],
                  production_changed=False, shape=list(SHAPE), cache=MODE)
    started = time.monotonic()
    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    def command(arm, profile=False):
        result = [sys.executable, "-u", str(Path(__file__).resolve()), "--child", "--sdk", str(args.sdk),
            "--candidate", str(args.candidate), "--controls", str(args.controls), "--bundle", str(args.bundle),
            "--fixture", str(args.fixture), "--variant", arm, "--mode", MODE,
            "--samples", str(args.samples), "--l2-bytes", str(args.l2_bytes)]
        return result + (["--profile"] if profile else [])
    def batch(arm, label, *, profile=False):
        receipt = args.output / f"{arm}-{label}.json"
        samples = 0 if profile else args.samples
        if receipt.exists():
            entry = json.loads(receipt.read_text())
            validate_entry(args.output, entry, arm, samples, device, profile=profile)
            return entry
        prefix = args.output / f"{arm}-{label}.{time.time_ns()}"
        log = prefix.with_name(prefix.name+(".acu.log" if profile else ".log"))
        cmd = acu_launch_command(args.acu, prefix, command(arm, True)) if profile else command(arm)
        print(f"Q4_COLD_GEOMETRY_PROGRESS arm={arm} phase={label} elapsed_s={time.monotonic()-started:.1f}", flush=True)
        with log.open("x") as stream:
            rc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT).returncode
        if rc:
            raise ValueError(f"child rc={rc}; log={log}")
        row = parse_result(log.read_text(), arm, samples, profile=profile)
        if row["device"] != device:
            raise ValueError("child physical device differs")
        entry = dict(row=row, log=log.name, log_sha256=digest(log))
        if profile:
            files = list(args.output.glob(prefix.name+"*.acurep"))
            if len(files) != 1:
                raise ValueError("expected exactly one ACU report")
            entry["reports"] = [dict(file=p.name, sha256=digest(p)) for p in files]
        receipt.write_text(json.dumps(entry, indent=2)+"\n")
        return entry
    for turn in range(ROUNDS):
        for arm in ARMS if turn%2 == 0 else ARMS[::-1]:
            try:
                report["records"][arm].append(batch(arm, f"r{turn}")["row"])
            except Exception as exc:
                report["failures"].append(dict(arm=arm, round=turn, error=str(exc)))
                print(f"Q4_COLD_GEOMETRY_FAILURE arm={arm} remaining_continue=1 error={exc}", flush=True)
            save()
    report["comparison"] = summarize(report["records"])
    print("Q4_COLD_GEOMETRY_RESULT "+json.dumps(report["comparison"]), flush=True)
    if not args.skip_acu:
        for arm in ARMS:
            try:
                entry = batch(arm, "profile", profile=True)
                report["profiles"].append(dict(arm=arm, status="PASS", cache="FORCED_COLD", timing_authority=False, **entry))
            except Exception as exc:
                report["profiles"].append(dict(arm=arm, status="FAIL", error=str(exc)))
                print(f"Q4_COLD_GEOMETRY_ACU_FAIL arm={arm} error={exc}", flush=True)
            save()
    verify(args.candidate, args.controls, args.bundle)
    if any(digest(ROOT/name) != sha for name,sha in authority["sources"].items()) or any(
            digest(args.sdk/"lib"/name) != sha for name,sha in authority["runtime"].items()):
        raise ValueError("runtime or orchestration changed during the run")
    ok = (not report["failures"] and report["comparison"]["status"] == "PASS"
          and all(p["status"] == "PASS" for p in report["profiles"]))
    report.update(status="PASS" if ok else "INCOMPLETE", seconds=time.monotonic()-started)
    save()
    with (args.output / "summary.tsv").open("w") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["arm","N","K","cache","recipe","median_us","rounds","status"])
        for arm in ARMS:
            writer.writerow([arm, *SHAPE[1:], MODE, RECIPES[arm],
                report["comparison"]["median_us"].get(arm,"NA"), len(report["records"][arm]), report["status"]])
    print(f"Q4_COLD_GEOMETRY_COMPLETE status={report['status']} results={args.output}", flush=True)
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--candidate", type=Path, default=ROOT / "prebuilt/ppu0010/q4-cold-geometry-v1")
    p.add_argument("--controls", type=Path, default=ROOT / "prebuilt/ppu0010/q4-h800-port-v1")
    p.add_argument("--bundle", type=Path, default=ROOT / "prebuilt/ppu0010/q4-simt-ab-v1")
    p.add_argument("--fixture", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--variant", choices=ARMS)
    p.add_argument("--mode", choices=(MODE,), default=MODE)
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--l2-bytes", type=int, default=0)
    p.add_argument("--child", action="store_true")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--skip-acu", action="store_true")
    p.add_argument("--acu", type=Path)
    a = p.parse_args()
    if a.samples<3 or a.l2_bytes<0:
        p.error("invalid samples/L2 capacity")
    for name in ("sdk", "candidate", "controls", "bundle"):
        setattr(a, name, getattr(a, name).resolve(strict=True))
    if a.probe:
        sdk = SDK(a.sdk)
        device = device_identity(sdk)
        _,g = load_library(a, "new"); device.update(g)
        print("Q4_COLD_GEOMETRY_DEVICE "+json.dumps(device), flush=True)
        return 0
    if a.fixture is None:
        p.error("--fixture is required")
    a.fixture = a.fixture.resolve(strict=True)
    if a.child:
        if a.variant is None: p.error("--variant is required for a child")
        return child(a)
    if a.output is None: p.error("--output is required")
    a.output = a.output.resolve()
    a.acu = a.acu or a.sdk / "asight/bin/acu"
    return run(a)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc(); raise SystemExit(1)
