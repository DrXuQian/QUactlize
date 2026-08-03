#!/usr/bin/env python3
"""Build and measure native-vs-scale-first dense/MoE CUDA-core GEMV at N=131072 and shipping N=2048."""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--peak", type=float, default=1.792e12, help="peak DRAM B/s; default RTX 5090")
    ap.add_argument("--binary", type=Path, default=Path("/tmp/quactlize_decode_routes_bench"))
    a = ap.parse_args()
    import torch
    major, minor = torch.cuda.get_device_capability()
    cmd = ["nvcc", "-std=c++17", "-O3", f"-arch=sm_{major}{minor}", "--expt-relaxed-constexpr",
           f"-I{ROOT/'quactlize'/'include'}", f"-I{ROOT/'third_party'/'cutlass'/'include'}",
           f"-I{ROOT/'third_party'/'actlize'/'include'}", "-o", str(a.binary),
           str(ROOT/"benchmarks"/"decode_routes_cuda_probe.cu")]
    built = subprocess.run(cmd, capture_output=True, text=True)
    if built.returncode:
        sys.exit(built.stdout + built.stderr)
    run = subprocess.run([str(a.binary), str(a.reps), str(a.experts)], capture_output=True, text=True)
    if run.returncode:
        sys.exit(run.stdout + run.stderr)
    rows = list(csv.DictReader(run.stdout.splitlines()))
    print(f"RTX 5090 peak={a.peak/1e12:.3f} TB/s; warm samples batch up to 128 launches")
    print(f"{'N':>7} {'format':6} {'route':12} {'cold us':>9} {'c Gelem/s':>10} {'c peak':>7} "
          f"{'warm us':>9} {'w Gelem/s':>10} {'w peak':>7}")
    for r in rows:
        elems, traffic = float(r["elements"]), float(r["bytes"])
        cold, warm = float(r["cold_us"]), float(r["warm_us"])
        ge = lambda us: elems / (us * 1e-6) / 1e9
        peak = lambda us: traffic / (us * 1e-6) / a.peak * 100
        print(f"{int(r['rows']):7} {r['format']:6} {r['route']:12} {cold:9.3f} {ge(cold):10.1f} "
              f"{peak(cold):6.1f}% {warm:9.3f} {ge(warm):10.1f} {peak(warm):6.1f}%")
    raw = a.binary.with_suffix(".csv")
    raw.write_text(run.stdout)
    print(f"raw CSV: {raw}")


if __name__ == "__main__":
    main()
