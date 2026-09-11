#!/usr/bin/env python3
"""Bounded, source-bound NCU comparison of the two 5090-selected Q4 readers.

Uses application replay with cache control disabled. The runner re-creates
warm/rotating state before cudaProfilerStart on every replay pass.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


SECTIONS = ("SpeedOfLight", "ComputeWorkloadAnalysis", "MemoryWorkloadAnalysis",
            "LaunchStats", "Occupancy", "SchedulerStats", "WarpStateStats", "SourceCounters")
METRICS = ("dram__bytes.sum", "dram__bytes.sum.per_second",
           "dram__bytes.sum.pct_of_peak_sustained_elapsed", "lts__t_sectors.sum",
           "smsp__inst_executed.sum")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jobs(receipt):
    for case in receipt["cases"]:
        m, n, k = case["shape"]
        if m != 1 or n != 4096 or k not in (2048, 4096):
            raise ValueError("outside the matched Q4 profiling scope")
        for mode in ("warm", "rotating"):
            for arm, key in (("xplane", "xplane_best"), ("kpack", "kpack_best")):
                recipe = case["modes"][mode]["confirmed"][key]["recipe"]
                if recipe["arm"] != arm:
                    raise ValueError("recipe arm mismatch")
                yield dict(key=f"k{k}-{mode}-{arm}", arm=arm, mode=mode, n=n, k=k,
                           recipe=[recipe[x] for x in ("columns", "warps", "split")])


def retuned_jobs(receipt, small=False):
    if receipt.get("status") != "PASS" or receipt.get("arithmetic") != "BOTH_FP32_DOT_AND_REDUCTION_FP16_WEIGHT_AND_A_BOUNDARY":
        raise ValueError("expected a complete same-GPU FP32 retune")
    shapes=([1,512,2048],[1,1024,5120]) if small else ([1,4096,2048],[1,4096,4096])
    seen=set()
    for case in receipt["cases"]:
        if case["shape"] not in shapes:
            continue
        m,n,k=case["shape"]
        mode=case["mode"]
        if mode not in ("warm","rotating") or (k,mode) in seen:
            raise ValueError("duplicate or unknown anchor cache regime")
        seen.add((k,mode))
        for arm in ("xplane","kpack"):
            recipe=case["winners"][arm]["recipe"]
            if len(recipe)!=4 or recipe[0]!=arm:
                raise ValueError("retuned recipe arm differs")
            yield dict(key=f"k{k}-{mode}-{arm}",arm=arm,mode=mode,n=n,k=k,recipe=recipe[1:])
    if len(seen)!=4:
        raise ValueError("missing retuned anchor regimes")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runner", "xplane-library", "kpack-library", "fixtures", "recipes", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--ncu", type=Path, default=Path("/usr/local/cuda-12.8/bin/ncu"))
    p.add_argument("--retuned-fp32", action="store_true",
                   help="use this GPU's independently retuned FP32 winners for the two anchors")
    p.add_argument("--small-shapes",action="store_true",
                   help="profile the N512/K2048 and N1024/K5120 families instead of the large anchors")
    args = p.parse_args()
    if args.small_shapes and not args.retuned_fp32:
        p.error("small shapes require a same-GPU FP32 retune")
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads(args.recipes.read_text())
    for name in ("xplane", "kpack"):
        expected=(receipt["authority"]["xplane" if name=="xplane" else "candidate"]["sha256"]
                  if args.retuned_fp32 else receipt[name+"_library_sha256"])
        if sha(getattr(args, name + "_library")) != expected:
            raise ValueError(name + " library differs from measured comparison")
    planned = list(retuned_jobs(receipt,args.small_shapes) if args.retuned_fp32 else jobs(receipt))
    if len(planned) != 8 or len({x["key"] for x in planned}) != 8:
        raise ValueError("expected eight distinct profile arms")
    paths = dict(runner=args.runner, xplane=args.xplane_library, kpack=args.kpack_library,
                 recipes=args.recipes)
    result = dict(status="RUNNING", recipe_scope=("SAME_GPU_RETUNED_FP32" if args.retuned_fp32 else
                                                 "FIXED_5090_WINNERS_NOT_5070_RETUNING"),
                  replay_mode="application", cache_control="none", clock_control="none",
                  shape_scope="SMALL_N_512_1024" if args.small_shapes else "ANCHOR_N4096",
                  authority={name:dict(path=str(path.resolve()),sha256=sha(path)) for name,path in paths.items()},
                  ncu_version=subprocess.check_output([str(args.ncu), "--version"], text=True),
                  profiles=[])
    started = time.monotonic()
    for job in planned:
        fixture = args.fixtures / f"q12-n{job['n']}-k{job['k']}-e1-c1.bin"
        base = [str(args.runner.resolve()), str(fixture.resolve()),
                str(args.xplane_library.resolve()), str(args.kpack_library.resolve()),
                job["arm"], *map(str, job["recipe"]), job["mode"]]
        row = dict(**job, fixture_sha256=sha(fixture), commands={})
        try:
            # Separate unprofiled graph timing; never substitute replay durations.
            for phase in ("timing", "profile"):
                prefix = []
                if phase == "profile":
                    prefix = [str(args.ncu), "--replay-mode", "application",
                              "--profile-from-start", "off", "--cache-control", "none",
                              "--clock-control", "none", "--export", str((args.output/job["key"]).resolve())]
                    for section in SECTIONS:
                        prefix += ["--section", section]
                    prefix += ["--metrics", ",".join(METRICS)]
                command = prefix + base + [phase]
                row["commands"][phase] = command
                log = args.output / f"{job['key']}.{phase}.log"
                with log.open("w") as out:
                    proc = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, timeout=900)
                if proc.returncode:
                    raise RuntimeError(f"{phase} rc={proc.returncode}, log={log}")
                if "status=PASS" not in log.read_text():
                    raise ValueError(f"{phase} lacks runner correctness admission")
            report = args.output / f"{job['key']}.ncu-rep"
            if not report.is_file():
                raise ValueError("profiler did not write a report")
            for page in ("raw", "details"):
                command = [str(args.ncu), "--import", str(report.resolve()), "--page", page, "--csv", "--print-units", "base"]
                row["commands"][page] = command
                with (args.output/f"{job['key']}.{page}.csv").open("w") as out:
                    subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=60)
            row.update(status="PASS", report_sha256=sha(report))
        except (RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            row.update(status="FAIL", error=str(exc))
        result["profiles"].append(row)
        result["elapsed_seconds"] = time.monotonic() - started
        (args.output/"manifest.json").write_text(json.dumps(result, indent=2)+"\n")
        print(f"Q4_LAYOUT_NCU_PROGRESS completed={len(result['profiles'])}/8 key={job['key']} status={row['status']} elapsed_s={result['elapsed_seconds']:.1f}", flush=True)
    result["status"] = "PASS" if all(x["status"] == "PASS" for x in result["profiles"]) else "FAIL"
    if any(sha(path) != result["authority"][name]["sha256"] for name,path in paths.items()):
        raise ValueError("input changed during profiling")
    (args.output/"manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    print("Q4_LAYOUT_NCU_COMPLETE status="+result["status"], flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
