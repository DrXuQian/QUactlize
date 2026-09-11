#!/usr/bin/env python3
"""Replay frozen H800 Q4 implementations against PPU-retuned controls.

Native prebuilt DSOs only. Each child owns one arm/shape/cache context; valid
cells are resumable and a failed child never discards other arms' results.
"""
import argparse
import csv
import ctypes as C
import json
import math
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
from dev.gemv_ppu.campaign import parse_cells
from dev.gemv_ppu.h800_port import IMPLEMENTATIONS, POLICY, REFERENCE_RECIPES, XPLANE_RECIPES, selection, verify
from dev.gemv_ppu.run import Bench, SDK, checked, device_identity, load_library
from dev.gemv_ppu.run_bload import exact_bits
from tools.run_kpack_grouped_decode_probe import Replay
from tools.profile_kpack_gpu_compact import AcuRange, acu_launch_command

VARIANTS = ("xplane", "raw-reference", "kpack")
ROUNDS = 6


def recipes(variant, n, k):
    if variant == "xplane":
        return list(XPLANE_RECIPES)
    if variant == "raw-reference":
        return list(REFERENCE_RECIPES)
    if variant == "kpack":
        return [tuple(selection(n, k)[1])]
    raise ValueError("unknown experiment arm")


class PortBench(Bench):
    def __init__(self, args):
        args.arm = "xplane" if args.variant == "xplane" else "new"
        super().__init__(args)
        self.variant = args.variant
        self.raw_weight = None
        self.implementation = self.variant
        if self.variant == "xplane":
            return
        family = selection(self.n, self.k)[0] if self.variant == "kpack" else "reference"
        self.extra = C.CDLL(str(args.candidate / f"libq4_ppu_port_{family}.so"), mode=C.RTLD_LOCAL)
        probe = self.extra.q4_ppu_probe
        probe.argtypes, probe.restype = [C.POINTER(C.c_int)] * 3 + [C.c_char_p], C.c_int
        l2, sm, warp, name = C.c_int(), C.c_int(), C.c_int(), C.create_string_buffer(256)
        checked(probe(C.byref(l2), C.byref(sm), C.byref(warp), name), "ported native image/marker")
        if (sm.value, warp.value, name.value.decode()) != (self.device["sm"], self.device["warp"], self.device["name"]):
            raise ValueError("ported DSO device differs")
        if self.variant == "kpack":
            self.implementation = IMPLEMENTATIONS[family]
            self.port_launch = self.extra.q4_h800_port_run
            self.port_launch.argtypes = [C.c_int] * 2 + [C.c_void_p] * 5
            self.port_launch.restype = C.c_int
        else:
            self.port_launch = self.extra.q4_ref_fp32_run_v2
            self.port_launch.argtypes = [C.c_int] * 5 + [C.c_void_p] * 4
            self.port_launch.restype = C.c_int
            self.raw_weight = np.ascontiguousarray(self.data["raw"]).view("u1").reshape(-1)
            if self.raw_weight.nbytes != self.weight_bytes:
                raise ValueError("raw GGUF and placed weight byte counts differ")
            base = self.r.alloc(self.weight_bytes * self.copies)
            self.weight_pointers = []
            for i in range(self.copies):
                ptr = base + i * self.weight_bytes
                checked(self.sdk.lib.hggcMemcpy(ptr, self.raw_weight.ctypes.data, self.weight_bytes, 1), "raw GGUF upload")
                self.weight_pointers.append((ptr, 0))
            self.sdk.synchronize(None)

    def invoke(self, recipe, index=0):
        low, units = self.weight_pointers[index % self.copies]
        if self.variant == "xplane":
            return self.launch(recipe[0], recipe[1], self.n, self.k, self.a, low, units, self.output, self.r.stream)
        if self.variant == "raw-reference":
            return self.port_launch(*recipe, self.n, self.k, self.a, low, self.output, self.r.stream)
        return self.port_launch(self.n, self.k, self.a, low, units, self.output, self.r.stream)

    def measure(self, recipe):
        if tuple(recipe) not in recipes(self.variant, self.n, self.k):
            raise ValueError("uncompiled or non-frozen recipe")
        self.r.fill(self.output_base, 0xff, self.output_bytes + 32)
        self.r.fill(self.workspace_base, 0xff, self.workspace_bytes + 256)
        checked(self.invoke(recipe), "independent-oracle launch")
        error = self.error()
        if not math.isfinite(error) or error >= .005:
            raise ValueError(f"independent GGUF FP64 dot failed: {error:.8g}")
        original = self.sdk.download(self.output, self.output_bytes)
        source = self.raw_weight if self.raw_weight is not None else self.host_low
        fault = source.copy()
        if self.raw_weight is not None:
            fault.reshape(-1, 144)[:, 16:] = 0  # preserve raw scale/min headers
        else:
            fault.fill(0)
        ptr = self.weight_pointers[0][0]
        checked(self.sdk.lib.hggcMemcpy(ptr, fault.ctypes.data, fault.nbytes, 1), "zero-code upload")
        self.sdk.synchronize(None)
        checked(self.invoke(recipe), "zero-code negative")
        if self.error() <= .005:
            raise ValueError("zero-code negative escaped")
        checked(self.sdk.lib.hggcMemcpy(ptr, source.ctypes.data, source.nbytes, 1), "restore codes")
        self.r.fill(self.a, 0, self.data["a"].nbytes)
        self.sdk.synchronize(None)
        checked(self.invoke(recipe), "zero-A check")
        self.sdk.synchronize(self.r.stream)
        zero = np.frombuffer(self.sdk.download(self.output, self.output_bytes), dtype="<f4")
        if not np.any(np.frombuffer(original, dtype="<f4") != 0) or not np.all(zero == 0):
            raise ValueError("zero-A check failed")
        checked(self.sdk.lib.hggcMemcpy(self.a, self.data["a"].ctypes.data, self.data["a"].nbytes, 1), "restore A")
        self.sdk.synchronize(None)
        for _ in range(5):
            for i in range(self.copies):
                checked(self.invoke(recipe, i), "cache setup")
        self.sdk.synchronize(self.r.stream)
        calls = max(2, (32 + self.copies - 1) // self.copies) * self.copies
        counter = 0
        def launch():
            nonlocal counter
            rc = self.invoke(recipe, counter)
            counter += 1
            return rc
        samples = []
        if self.args.profile:
            with AcuRange(self.sdk):
                checked(self.invoke(recipe), "ACU selected call")
                self.sdk.synchronize(self.r.stream)
        else:
            graph = Replay(self.sdk, self.r.stream, launch, calls)
            try:
                self.r.samples(graph, 5)  # graph upload/first launches excluded
                samples = [t / calls for t in self.r.samples(graph, self.args.samples)]
            finally:
                graph.close()
        error = max(error, self.error())
        exact_bits(original, self.sdk.download(self.output, self.output_bytes), "post-replay deterministic FP32")
        if error >= .005 or self.sdk.download(self.workspace_base, self.workspace_bytes + 256) != b"\xff" * (self.workspace_bytes + 256):
            raise ValueError("post-replay oracle or S1 workspace guard failed")
        return dict(status="PASS", arm=self.variant, implementation=self.implementation,
                    shape=[1, self.n, self.k], mode=self.args.mode, recipe=list(recipe),
                    error=error, zero_code_negative="PASS", zero_a_check="PASS", device=self.device,
                    copies=self.copies, weight_bytes=self.weight_bytes, calls_per_graph=calls,
                    samples_us=samples, median_us=statistics.median(samples) if samples else None,
                    output_type="F32", launches_per_call=1, inter_cta_split=1,
                    weight_arithmetic="FP32_GROUP_AFFINE" if self.implementation.startswith("affine") else "PER_WEIGHT_FP16",
                    storage="RAW_GGUF" if self.raw_weight is not None else "XPLANE" if self.variant == "xplane" else "CANONICAL_KPACK4",
                    timing_scope="RESIDENT_FULL_CALL_NO_HOST_PACK_NO_JIT")


def parse_result(text, variant, recipe, shape, mode, samples):
    row = parse_cells(text, variant, [recipe], shape, mode, samples)[0]
    implementation = IMPLEMENTATIONS[selection(*shape[1:])[0]] if variant == "kpack" else variant
    arithmetic = "FP32_GROUP_AFFINE" if implementation.startswith("affine") else "PER_WEIGHT_FP16"
    storage = "CANONICAL_KPACK4" if variant == "kpack" else "RAW_GGUF" if variant == "raw-reference" else "XPLANE"
    if (row.get("implementation") != implementation or row.get("weight_arithmetic") != arithmetic
            or row.get("output_type") != "F32" or row.get("zero_a_check") != "PASS"
            or row.get("launches_per_call") != 1 or row.get("inter_cta_split") != 1
            or row.get("storage") != storage or row.get("timing_scope") != "RESIDENT_FULL_CALL_NO_HOST_PACK_NO_JIT"
            or tuple(recipe) not in recipes(variant, *shape[1:])):
        raise ValueError("ported implementation, precision or recipe differs")
    return row


def child(args):
    verify(args.candidate, args.bundle, sources=False)
    bench = None
    try:
        bench = PortBench(args)
        for recipe in json.loads(args.recipes):
            row = bench.measure(recipe)
            print("Q4_PPU_CELL " + json.dumps(row), flush=True)
        return 0
    finally:
        if bench:
            bench.close()


def summarize(n, k, mode, records):
    winners = {}
    for variant, rounds in records.items():
        if len(rounds) != ROUNDS:
            continue
        keys = {tuple(r["recipe"]) for r in rounds[0]}
        if any({tuple(r["recipe"]) for r in batch} != keys for batch in rounds):
            continue
        candidates = []
        for recipe in keys:
            rows = [next(r for r in batch if tuple(r["recipe"]) == recipe) for batch in rounds]
            candidates.append(dict(recipe=list(recipe), median_us=statistics.median(r["median_us"] for r in rows), rounds=rows))
        if candidates:
            winners[variant] = min(candidates, key=lambda r: r["median_us"])
    full = set(winners) == set(VARIANTS)
    deltas = {v: 100 * (winners["kpack"]["median_us"] / winners[v]["median_us"] - 1)
              for v in VARIANTS[:2]} if full else {}
    return dict(shape=[1, n, k], mode=mode, winners=winners, delta_pct=deltas,
                verdict="INCOMPLETE" if not full else "WITHIN_5_PERCENT" if all(x < 5 for x in deltas.values()) else "PARITY_OPEN")


def run(args):
    manifest = verify(args.candidate, args.bundle)
    sdk = SDK(args.sdk)
    device = device_identity(sdk)
    _, geometry = load_library(args, "new")
    device.update(geometry)
    shapes = [shape[1:] for shape in POLICY]
    authority = dict(candidate=digest(args.candidate / "manifest.json"), baseline=digest(args.bundle / "manifest.json"),
        runner=digest(Path(__file__)), protocol=digest(ROOT / "dev/gemv_ppu/run.py"),
        parser=digest(ROOT / "dev/gemv_ppu/campaign.py"), device=device, rounds=ROUNDS, samples=args.samples,
        runtime={name: digest(args.sdk / "lib" / name) for name in manifest["runtime"]},
        fixtures={f"{n}x{k}": digest(args.fixtures / f"q12-n{n}-k{k}-e1-c1.npz") for n, k in shapes})
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "authority.json"
    if path.exists() and json.loads(path.read_text()) != authority:
        raise ValueError("resume authority differs; use a fresh result directory")
    path.write_text(json.dumps(authority, indent=2) + "\n")
    changes = [name for name, value in authority["runtime"].items() if value != manifest["runtime"][name]]
    if changes:
        print("Q4_H800_PORT_SDK_DIFFERENCE recorded=" + ",".join(changes) + " real_marker_and_numeric_required=1", flush=True)
    started = time.monotonic()
    report = dict(status="RUNNING", cases=[], failures=[], profiles=[], production_changed=False,
                  scope="FROZEN_KPACK_H800_RECIPES_PPU_RETUNED_CONTROLS", authority=authority)
    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    def command(n, k, mode, variant, wanted, samples, profile=False):
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--child", "--sdk", str(args.sdk),
               "--bundle", str(args.bundle), "--candidate", str(args.candidate), "--fixture",
               str(args.fixtures / f"q12-n{n}-k{k}-e1-c1.npz"), "--variant", variant, "--mode", mode,
               "--recipes", json.dumps(wanted), "--samples", str(samples), "--l2-bytes", str(args.l2_bytes)]
        return cmd + (["--profile"] if profile else [])
    def batch(n, k, mode, variant, wanted, label, samples):
        receipt = args.output / f"n{n}-k{k}-{mode}-{variant}-{label}.json"
        cached = json.loads(receipt.read_text()) if receipt.exists() else {}
        wanted_keys = {json.dumps(list(r)) for r in wanted}
        if not set(cached) <= wanted_keys:
            raise ValueError("cached receipt contains an unrequested recipe")
        for key, entry in cached.items():
            log = args.output / entry["log"]
            if log.parent != args.output or digest(log) != entry["log_sha256"]:
                raise ValueError("cached child log differs")
            text = "Q4_PPU_CELL " + json.dumps(entry["row"])
            row = parse_result(text, variant, json.loads(key), [1, n, k], mode, samples)
            if row["device"] != device or text not in log.read_text():
                raise ValueError("cached result/device not present in device log")
        missing = [list(r) for r in wanted if json.dumps(list(r)) not in cached]
        if missing:
            log = receipt.with_name(receipt.stem + f".{time.time_ns()}.log")
            print(f"Q4_H800_PORT_PROGRESS shape=1x{n}x{k} cache={mode} arm={variant} phase={label} recipes={len(missing)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
            with log.open("w") as out:
                rc = subprocess.run(command(n, k, mode, variant, missing, samples), stdout=out, stderr=subprocess.STDOUT).returncode
            seen = set()
            for text in log.read_text().splitlines():
                if not text.startswith("Q4_PPU_CELL "):
                    continue
                raw = json.loads(text.split(" ", 1)[1])
                key = json.dumps(raw["recipe"])
                if key in seen or raw["recipe"] not in missing:
                    raise ValueError("unexpected or duplicate child recipe")
                seen.add(key)
                row = parse_result(text, variant, raw["recipe"], [1, n, k], mode, samples)
                if row["device"] != device:
                    raise ValueError("child device differs")
                cached[key] = dict(row=row, log=log.name, log_sha256=digest(log))
            receipt.write_text(json.dumps(cached, indent=2) + "\n")
            if rc or len(seen) != len(missing):
                raise ValueError(f"child rc={rc}; preserved valid cells; log={log}")
        return [cached[json.dumps(list(r))]["row"] for r in wanted]
    for n, k in shapes:
        for mode in ("warm", "rotating"):
            shortlist, records = {}, {}
            for variant in VARIANTS:
                try:
                    rows = batch(n, k, mode, variant, recipes(variant, n, k), "screen", 5)
                    shortlist[variant] = [r["recipe"] for r in sorted(rows, key=lambda r: r["median_us"])[:2]]
                    records[variant] = []
                except Exception as exc:
                    report["failures"].append(dict(shape=[1, n, k], mode=mode, variant=variant, phase="screen", error=str(exc)))
                    print("Q4_H800_PORT_FAILURE " + str(exc), flush=True)
            for turn in range(ROUNDS):
                for variant in VARIANTS if turn % 2 == 0 else VARIANTS[::-1]:
                    if variant not in shortlist:
                        continue
                    try:
                        records[variant].append(batch(n, k, mode, variant, shortlist[variant], f"confirm{turn}", args.samples))
                    except Exception as exc:
                        report["failures"].append(dict(shape=[1, n, k], mode=mode, variant=variant, phase=f"confirm{turn}", error=str(exc)))
                        print("Q4_H800_PORT_FAILURE " + str(exc), flush=True)
                        del shortlist[variant]
                save()
            result = summarize(n, k, mode, records)
            report["cases"].append(result)
            print("Q4_H800_PORT_RESULT " + json.dumps(dict(shape=result["shape"], mode=mode,
                  median_us={v: r["median_us"] for v, r in result["winners"].items()}, delta_pct=result["delta_pct"], verdict=result["verdict"])), flush=True)
            save()
    if not args.skip_acu:
        for case in report["cases"]:
            if case["mode"] != "rotating" or tuple(case["shape"][1:]) not in ((512, 2048), (1024, 5120), (5120, 8192)):
                continue
            _, n, k = case["shape"]
            for variant, winner in case["winners"].items():
                prefix = args.output / f"n{n}-k{k}-{variant}.{time.time_ns()}.acu"
                log = prefix.with_suffix(".acu.log")
                record = dict(shape=case["shape"], variant=variant, recipe=winner["recipe"],
                              timing_authority=False, cache="FORCED_COLD", log=log.name)
                try:
                    cmd = acu_launch_command(args.acu, prefix, command(n, k, "warm", variant, [winner["recipe"]], args.samples, True))
                    print(f"Q4_H800_PORT_ACU shape=1x{n}x{k} arm={variant} cache=FORCED_COLD", flush=True)
                    with log.open("w") as stream:
                        rc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT).returncode
                    if rc:
                        raise ValueError(f"acu rc={rc}")
                    parse_result(log.read_text(), variant, winner["recipe"], [1, n, k], "warm", 0)
                    files = [p for p in args.output.glob(prefix.name + "*") if p.suffix in (".acurep", ".report")]
                    if not files:
                        raise ValueError("ACU report missing")
                    record.update(status="PASS", files=[dict(file=p.name, sha256=digest(p)) for p in files])
                except Exception as exc:
                    record.update(status="FAIL", error=str(exc))
                report["profiles"].append(record)
                save()
    verify(args.candidate, args.bundle)
    if digest(args.candidate / "manifest.json") != authority["candidate"] or any(
            digest(args.sdk / "lib" / name) != value for name, value in authority["runtime"].items()):
        raise ValueError("package or runtime changed during the run")
    complete = not report["failures"] and all(c["verdict"] != "INCOMPLETE" for c in report["cases"])
    report.update(status="PASS" if complete and all(p["status"] == "PASS" for p in report["profiles"]) else "INCOMPLETE",
                  parity_verdict="WITHIN_5_PERCENT" if complete and all(c["verdict"] == "WITHIN_5_PERCENT" for c in report["cases"]) else "OPEN",
                  seconds=time.monotonic() - started)
    with (args.output / "summary.tsv").open("w") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["N", "K", "cache", *[v + "_us" for v in VARIANTS], "vs_xplane_pct", "vs_ref_pct", "verdict"])
        for case in report["cases"]:
            writer.writerow([*case["shape"][1:], case["mode"], *[case["winners"].get(v, {}).get("median_us", "NA") for v in VARIANTS],
                             case["delta_pct"].get("xplane", "NA"), case["delta_pct"].get("raw-reference", "NA"), case["verdict"]])
    save()
    print(f"Q4_H800_PORT_COMPLETE status={report['status']} parity={report['parity_verdict']} results={args.output}", flush=True)
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT / "prebuilt/ppu0010/q4-simt-ab-v1")
    parser.add_argument("--candidate", type=Path, default=ROOT / "prebuilt/ppu0010/q4-h800-port-v1")
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--recipes")
    parser.add_argument("--mode", choices=("warm", "rotating"))
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--l2-bytes", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--skip-acu", action="store_true")
    parser.add_argument("--acu", type=Path)
    args = parser.parse_args()
    args.sdk = args.sdk.resolve(strict=True)
    args.bundle = args.bundle.resolve(strict=True)
    args.candidate = args.candidate.resolve(strict=True)
    if args.samples < 3 or args.l2_bytes < 0:
        parser.error("invalid samples/L2 capacity")
    if args.child:
        return child(args)
    if args.fixtures is None or args.output is None:
        parser.error("--fixtures and --output required")
    args.fixtures = args.fixtures.resolve(strict=True)
    args.output = args.output.resolve()
    args.acu = args.acu or args.sdk / "asight/bin/acu"
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
