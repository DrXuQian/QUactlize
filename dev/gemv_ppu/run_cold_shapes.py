#!/usr/bin/env python3
"""Cold-only Q4 shape sweep: existing best K-pack versus C8 and both controls."""
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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_cuda.build import digest
from dev.gemv_ppu.cold_shapes import PAYLOAD, RECIPE, SHAPES, geometry, verify
from dev.gemv_ppu.campaign import parse_cells
from dev.gemv_ppu.h800_port import IMPLEMENTATIONS, selection
from dev.gemv_ppu.run import checked
from dev.gemv_ppu.run_bload import CONTROL_RECEIPT, ExperimentBench, control_recipes
from dev.gemv_ppu.run_cold_geometry import probe_device
from dev.gemv_ppu.run_h800_port import PortBench
from tools.profile_kpack_gpu_compact import acu_launch_command

ARMS = ("xplane", "raw-reference", "kpack-current", "kpack-c8")
ROUNDS = 6
MODE = "rotating"
PREFIX = "Q4_COLD_SHAPES_CELL "


def recipe(arm, n, k):
    if arm == "kpack-c8":
        geometry(n, k)
        return RECIPE
    key = "kpack" if arm == "kpack-current" else arm
    return control_recipes()[(n, k, MODE)][key]


def implementation(arm, n, k):
    if arm == "kpack-c8":
        return "affine4-early-fast-bare"
    return IMPLEMENTATIONS[selection(n, k)[0]] if arm == "kpack-current" else arm


class C8Bench(ExperimentBench):
    def __init__(self, args):
        super().__init__(SimpleNamespace(**(vars(args) | dict(variant="kpack-current", arm="new"))))
        self.c8_library = C.CDLL(str(args.candidate / PAYLOAD), mode=C.RTLD_LOCAL)
        probe = self.c8_library.q4_ppu_probe
        probe.argtypes, probe.restype = [C.POINTER(C.c_int)]*3 + [C.c_char_p], C.c_int
        l2, sm, warp, name = C.c_int(), C.c_int(), C.c_int(), C.create_string_buffer(256)
        checked(probe(C.byref(l2), C.byref(sm), C.byref(warp), name), "C8 native image/marker")
        if (sm.value, warp.value, name.value.decode()) != (self.device["sm"], self.device["warp"], self.device["name"]):
            raise ValueError("C8 image/device differs")
        self.c8_launch = self.c8_library.q4_cold_shapes_run
        self.c8_launch.argtypes, self.c8_launch.restype = [C.c_int]*2 + [C.c_void_p]*5, C.c_int

    def invoke(self, wanted, index=0, force_aiu=None):
        if tuple(wanted) != RECIPE or force_aiu is not None:
            raise ValueError("C8 invocation recipe differs")
        low, units = self.weight_pointers[index % self.copies]
        return self.c8_launch(self.n, self.k, self.a, low, units, self.output, self.r.stream)


def child(args):
    verify(args.candidate, args.controls, args.bundle, sources=False)
    bench = None
    try:
        if args.variant == "kpack-c8":
            bench = C8Bench(args)
        else:
            variant = "kpack" if args.variant == "kpack-current" else args.variant
            bench = PortBench(SimpleNamespace(**(vars(args) | dict(variant=variant, candidate=args.controls))))
        n, k = bench.n, bench.k
        row = bench.measure(list(recipe(args.variant, n, k)))
        impl = implementation(args.variant, n, k)
        row.update(arm=args.variant, variant=args.variant, implementation=impl,
                   launches_per_call=1, inter_cta_split=1,
                   timing_scope="RESIDENT_FULL_CALL_NO_HOST_PACK_NO_JIT",
                   cache_scope="ACU_FORCED_COLD" if args.profile else "ROTATING_GT_2_25_L2",
                   weight_arithmetic="FP32_GROUP_AFFINE" if impl.startswith("affine") else "PER_WEIGHT_FP16")
        if args.variant == "kpack-c8":
            row["geometry"] = geometry(n, k)
        print(PREFIX + json.dumps(row), flush=True)
        return 0
    finally:
        if bench:
            bench.close()


def parse_result(text, arm, n, k, samples, *, profile=False):
    lines = [line for line in text.splitlines() if line.startswith(PREFIX)]
    converted = "\n".join("Q4_PPU_CELL " + line[len(PREFIX):] for line in lines)
    row = parse_cells(converted, arm, [recipe(arm, n, k)], [1, n, k], MODE, samples)[0]
    impl = implementation(arm, n, k)
    if (row.get("variant") != arm or row.get("output_type") != "F32" or row.get("zero_a_check") != "PASS"
            or row.get("inter_cta_split") != 1 or row.get("launches_per_call") != 1
            or row.get("timing_scope") != "RESIDENT_FULL_CALL_NO_HOST_PACK_NO_JIT"
            or row.get("cache_scope") != ("ACU_FORCED_COLD" if profile else "ROTATING_GT_2_25_L2")
            or row.get("implementation") != impl
            or row.get("weight_arithmetic") != ("FP32_GROUP_AFFINE" if impl.startswith("affine") else "PER_WEIGHT_FP16")
            or row.get("storage") != ("CANONICAL_KPACK4" if arm.startswith("kpack-") else
                                      "RAW_GGUF" if arm == "raw-reference" else "XPLANE")):
        raise ValueError("cold-shapes receipt precision/route/cache scope differs")
    if arm == "kpack-c8" and row.get("geometry") != geometry(n, k):
        raise ValueError("C8 work distribution differs")
    if (profile and row.get("median_us") is not None) or (
            not profile and not math.isfinite(row["median_us"])):
        raise ValueError("profile or nonfinite duration is not event timing")
    return row


def validate_entry(output, entry, arm, n, k, samples, device, *, profile=False):
    log = output / entry["log"]
    if log.parent != output or digest(log) != entry["log_sha256"]:
        raise ValueError("cached log path/hash differs")
    row = parse_result(log.read_text(), arm, n, k, samples, profile=profile)
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


def summarize(n, k, records):
    medians = {arm: statistics.median(r["median_us"] for r in rows)
               for arm, rows in records.items() if len(rows) == ROUNDS}
    complete = set(medians) == set(ARMS)
    result = dict(shape=[1, n, k], cache=MODE, status="PASS" if complete else "INCOMPLETE",
                  median_us=medians, c8_delta_pct={}, parity_verdict="INCOMPLETE", selected_kpack=None)
    if "kpack-c8" in medians:
        result["c8_delta_pct"] = {arm: 100*(medians["kpack-c8"]/us-1)
                                 for arm, us in medians.items() if arm != "kpack-c8"}
    if complete:
        target = 1.05*min(medians[arm] for arm in ARMS[:2])
        selected = min(ARMS[2:], key=medians.get)
        result.update(target_us=target, selected_kpack=selected,
                      selected_delta_pct={arm:100*(medians[selected]/medians[arm]-1) for arm in ARMS[:2]},
                      c8_parity_verdict="WITHIN_5_PERCENT" if medians["kpack-c8"] <= target else "PARITY_OPEN",
                      parity_verdict="WITHIN_5_PERCENT" if medians[selected] <= target else "PARITY_OPEN")
    return result


def run(args):
    manifest = verify(args.candidate, args.controls, args.bundle)
    selected_controls = json.loads(CONTROL_RECEIPT.read_text())["authority"]
    if (selected_controls["candidate"] != digest(args.controls/"manifest.json") or
            selected_controls["baseline"] != digest(args.bundle/"manifest.json")):
        raise ValueError("historical recipe receipt belongs to different control images")
    device = probe_device(args)  # exits before the parent launches any benchmark child
    inputs = ["dev/gemv_ppu/run_cold_shapes.py", "dev/gemv_ppu/cold_shapes.py",
              "dev/gemv_ppu/run_cold_geometry.py", "dev/gemv_ppu/run.py", "dev/gemv_ppu/campaign.py",
              "dev/gemv_ppu/run_bload.py", "dev/gemv_ppu/run_h800_port.py", "dev/gemv_ppu/h800_port.py",
              "tools/run_kpack_gemv_gate.py", "tools/run_kpack_grouped_decode_probe.py", "tools/profile_kpack_gpu_compact.py"]
    authority = dict(schema="quactlize.q4-cold-shapes-run.v1", shapes=SHAPES, mode=MODE, rounds=ROUNDS,
        samples=args.samples, device=device, control_receipt_sha256=digest(CONTROL_RECEIPT),
        fixtures={f"{n}x{k}":digest(args.fixtures/f"q12-n{n}-k{k}-e1-c1.npz") for n,k in SHAPES},
        recipes={f"{n}x{k}":{arm:recipe(arm,n,k) for arm in ARMS} for n,k in SHAPES},
        candidate=digest(args.candidate/"manifest.json"), controls=digest(args.controls/"manifest.json"),
        baseline=digest(args.bundle/"manifest.json"),
        runtime={name:digest(args.sdk/"lib"/name) for name in manifest["runtime"]},
        sources={name:digest(ROOT/name) for name in inputs})
    authority = json.loads(json.dumps(authority))
    args.output.mkdir(parents=True, exist_ok=True)
    identity = args.output/"authority.json"
    if identity.exists() and json.loads(identity.read_text()) != authority:
        raise ValueError("resume source/device/runtime/fixture differs; use a fresh run")
    identity.write_text(json.dumps(authority, indent=2)+"\n")
    changes = [name for name,sha in authority["runtime"].items() if manifest["runtime"][name] != sha]
    if changes:
        print("Q4_COLD_SHAPES_SDK_DIFFERENCE recorded="+",".join(changes)+" real_marker_and_numeric_required=1", flush=True)
    report = dict(status="RUNNING", cases=[], profiles=[], failures=[], production_changed=False, cache=MODE)
    started = time.monotonic()
    def save():
        (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    def command(arm,n,k,profile):
        cmd = [sys.executable,"-u",str(Path(__file__).resolve()),"--child","--sdk",str(args.sdk),
            "--candidate",str(args.candidate),"--controls",str(args.controls),"--bundle",str(args.bundle),
            "--fixture",str(args.fixtures/f"q12-n{n}-k{k}-e1-c1.npz"),"--variant",arm,
            "--samples",str(args.samples),"--l2-bytes",str(args.l2_bytes)]
        return cmd + (["--profile"] if profile else [])
    def batch(arm,n,k,label,*,profile=False):
        receipt = args.output/f"n{n}-k{k}-{arm}-{label}.json"
        samples = 0 if profile else args.samples
        if receipt.exists():
            entry = json.loads(receipt.read_text())
            validate_entry(args.output,entry,arm,n,k,samples,device,profile=profile)
            return entry
        prefix = args.output/f"n{n}-k{k}-{arm}-{label}.{time.time_ns()}"
        log = prefix.with_name(prefix.name+(".acu.log" if profile else ".log"))
        cmd = acu_launch_command(args.acu,prefix,command(arm,n,k,True)) if profile else command(arm,n,k,False)
        print(f"Q4_COLD_SHAPES_PROGRESS shape=1x{n}x{k} arm={arm} phase={label} elapsed_s={time.monotonic()-started:.1f}",flush=True)
        with log.open("x") as stream:
            rc = subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT).returncode
        if rc:
            raise ValueError(f"child rc={rc}; log={log}")
        row = parse_result(log.read_text(),arm,n,k,samples,profile=profile)
        if row["device"] != device:
            raise ValueError("child physical device differs")
        entry = dict(row=row,log=log.name,log_sha256=digest(log))
        if profile:
            files = list(args.output.glob(prefix.name+"*.acurep"))
            if len(files) != 1:
                raise ValueError("expected exactly one ACU report")
            entry["reports"] = [dict(file=p.name,sha256=digest(p)) for p in files]
        receipt.write_text(json.dumps(entry,indent=2)+"\n")
        return entry
    for n,k in SHAPES:
        case = dict(shape=[1,n,k],records={arm:[] for arm in ARMS})
        report["cases"].append(case)
        for turn in range(ROUNDS):
            for arm in ARMS if turn%2 == 0 else ARMS[::-1]:
                try:
                    case["records"][arm].append(batch(arm,n,k,f"r{turn}")["row"])
                except Exception as exc:
                    report["failures"].append(dict(shape=[1,n,k],arm=arm,round=turn,error=str(exc)))
                    print(f"Q4_COLD_SHAPES_FAILURE shape=1x{n}x{k} arm={arm} remaining_continue=1 error={exc}",flush=True)
                save()
        case["comparison"] = summarize(n,k,case["records"])
        print("Q4_COLD_SHAPES_RESULT "+json.dumps(case["comparison"]),flush=True)
        if not args.skip_acu:
            for arm in ARMS:
                try:
                    entry = batch(arm,n,k,"profile",profile=True)
                    report["profiles"].append(dict(shape=[1,n,k],arm=arm,status="PASS",timing_authority=False,**entry))
                except Exception as exc:
                    report["profiles"].append(dict(shape=[1,n,k],arm=arm,status="FAIL",error=str(exc)))
                    print(f"Q4_COLD_SHAPES_ACU_FAIL shape=1x{n}x{k} arm={arm} error={exc}",flush=True)
                save()
    verify(args.candidate,args.controls,args.bundle)
    if (any(digest(ROOT/name) != sha for name,sha in authority["sources"].items()) or
            any(digest(args.sdk/"lib"/name) != sha for name,sha in authority["runtime"].items()) or
            digest(CONTROL_RECEIPT) != authority["control_receipt_sha256"]):
        raise ValueError("runtime or orchestration changed during the run")
    ok = (not report["failures"] and all(c["comparison"]["status"]=="PASS" for c in report["cases"])
          and all(p["status"]=="PASS" for p in report["profiles"]))
    report.update(status="PASS" if ok else "INCOMPLETE",seconds=time.monotonic()-started,
                  within_5pct=sum(c["comparison"]["parity_verdict"]=="WITHIN_5_PERCENT" for c in report["cases"]))
    save()
    with (args.output/"summary.tsv").open("w") as stream:
        writer = csv.writer(stream,delimiter="\t")
        writer.writerow(["N","K",*ARMS,"selected_kpack","vs_xplane_pct","vs_reference_pct","verdict"])
        for c in report["cases"]:
            r = c["comparison"];d = r.get("selected_delta_pct",{})
            writer.writerow([*c["shape"][1:],*[r["median_us"].get(a,"NA") for a in ARMS],
                r["selected_kpack"],d.get("xplane","NA"),d.get("raw-reference","NA"),r["parity_verdict"]])
    print(f"Q4_COLD_SHAPES_COMPLETE status={report['status']} shapes=6 within_5pct={report['within_5pct']}/6 results={args.output}",flush=True)
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk",type=Path,required=True)
    p.add_argument("--candidate",type=Path,default=ROOT/"prebuilt/ppu0010/q4-cold-shapes-v1")
    p.add_argument("--controls",type=Path,default=ROOT/"prebuilt/ppu0010/q4-h800-port-v1")
    p.add_argument("--bundle",type=Path,default=ROOT/"prebuilt/ppu0010/q4-simt-ab-v1")
    p.add_argument("--fixtures",type=Path)
    p.add_argument("--fixture",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--variant",choices=ARMS)
    p.add_argument("--samples",type=int,default=15)
    p.add_argument("--l2-bytes",type=int,default=0)
    p.add_argument("--child",action="store_true")
    p.add_argument("--profile",action="store_true")
    p.add_argument("--skip-acu",action="store_true")
    p.add_argument("--acu",type=Path)
    a = p.parse_args()
    if a.samples<3 or a.l2_bytes<0:
        p.error("invalid samples/L2 capacity")
    a.mode = MODE  # a warm-cache invocation is intentionally not offered
    for name in ("sdk","candidate","controls","bundle"):
        setattr(a,name,getattr(a,name).resolve(strict=True))
    if a.child:
        if a.variant is None or a.fixture is None:
            p.error("--variant and --fixture are required for a child")
        a.fixture = a.fixture.resolve(strict=True)
        return child(a)
    if a.fixtures is None or a.output is None:
        p.error("--fixtures and --output are required")
    a.fixtures = a.fixtures.resolve(strict=True)
    a.output = a.output.resolve()
    a.acu = a.acu or a.sdk/"asight/bin/acu"
    return run(a)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
