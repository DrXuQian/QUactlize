#!/usr/bin/env python3
"""THE CUDA-CORE GEMV's MEASUREMENT, as a command rather than someone's shell history.

Every number quoted for vecdot_rows_kernel came out of an ad-hoc script. acu_capture.sh's header records why that
is a defect in itself -- "the repo recorded acu's OUTPUT fields but never the command, so every profiling round
started by asking for it again" -- and this file is the same fix for the GEMV.

  ./benchmarks/gemv_bench.py               the shipped config at the tuning shape AND at a real layer
  ./benchmarks/gemv_bench.py --sizes       %peak against problem size: where saturation begins
  ./benchmarks/gemv_bench.py --rpw         rows-per-warp sweep at both shapes, which is how the mis-tuning was found
  ./benchmarks/gemv_bench.py --rpw --bpr 4 repeat the sweep at K=1024 for the rows/bpr dispatcher
  ./benchmarks/gemv_bench.py --define X=1  add a -D, e.g. GGUF_VECDOT_FP32_ACTIVATION=1

TWO THINGS THAT WILL MISLEAD YOU IF NOBODY SAYS THEM.

1. THE EVENT TIMER QUANTISES AT 2.048 us on this 5090 -- 31 of 31 distinct timings ever observed here are exact
   multiples. Every number below is printed with its tick count for that reason. A one-tick difference is NOT a
   result: at the 131072-row shape one tick is ~1%, but at a real layer it is 8%. More reps do not help; the median
   of quantised samples is quantised. Cold cannot batch launches inside one event pair either, because only the
   first launch after an L2 flush is cold -- so the only lever on cold resolution is a bigger PROBLEM.

2. `rows` IS THE OUTPUT DIMENSION N, and the tuning shape is not the shipping shape. rows=131072 at bpr=8 is
   N=131072, K=2048: sixty-four times a real dense layer at decode, which is rows=2048. Q4 reads 65.5% of peak at
   the first and 9.3% at the second. The launcher therefore dispatches rows-per-warp from both rows and bpr; the
   fixed-rpw sweep is the witness used to set that policy. The default run prints BOTH shapes.

Warm launches are batched under one event pair so small-shape results are not quantised to whole 2.048-us ticks.
At rows=2048 the operand fits in L2 and cold/warm is meaningful. At rows=131072 Q3..Q6 exceed L2, so their ratio
near one means both readings are DRAM-fed, not that the kernel is insensitive to memory.
"""
import argparse, ctypes, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUANTUM_US = 2.048          # measured, not documented: see the module docstring
FORMATS = [("Q2_K", 10), ("Q3_K", 11), ("Q4_K", 12), ("Q5_K", 13), ("Q6_K", 14)]


def build(defines, tmp):
    import torch
    if not torch.cuda.is_available():
        sys.exit("no CUDA device")
    major, minor = torch.cuda.get_device_capability()
    so = tmp / ("probe_" + ("_".join(defines) or "default").replace("=", "") + ".so")
    # TWO CUTLASS TREES, AND THE ORDER IS LOAD-BEARING. third_party/cutlass is NVIDIA's and is what a local nvcc
    # build resolves `cutlass/...` against; third_party/actlize is the PPU fork and is the only place the shared
    # mixed-input converter lives. gguf_vecdot.hpp needs the converter and must also compile under plain nvcc, so
    # both are on the path with NVIDIA first -- actlize supplies only what NVIDIA's does not have.
    # KEEP THIS LIST IN STEP WITH tests/test_gguf_golden.py's probe build; they compile the same file and a
    # divergence shows up as "works in the test, fails here" or the reverse.
    cmd = ["nvcc", "-std=c++17", "-O3", f"-arch=sm_{major}{minor}", "-shared", "-Xcompiler=-fPIC",
           "--expt-relaxed-constexpr",
           f"-I{ROOT/'quactlize'/'include'}", f"-I{ROOT/'third_party'/'cutlass'/'include'}",
           f"-I{ROOT/'third_party'/'actlize'/'include'}"]
    cmd += [f"-D{d}" for d in defines]
    cmd += ["-o", str(so), str(ROOT / "tests" / "gguf_cuda_probe.cu")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stdout + r.stderr)
    lib = ctypes.CDLL(str(so))
    fp, dp = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double)
    lib.quactlize_cuda_vecdot_bench.argtypes = [ctypes.c_int] * 4 + [fp, fp, dp]
    lib.quactlize_cuda_vecdot_bench_config.argtypes = [ctypes.c_int] * 5 + [fp, fp, dp]
    lib.quactlize_cuda_vecdot_rows_per_warp.argtypes = [ctypes.c_int] * 3
    return lib


def run(lib, qtype, rows, bpr, reps, rpw=None):
    c, w, b = ctypes.c_float(), ctypes.c_float(), ctypes.c_double()
    args = [qtype, rows, bpr, reps] if rpw is None else [qtype, rpw, rows, bpr, reps]
    fn = lib.quactlize_cuda_vecdot_bench if rpw is None else lib.quactlize_cuda_vecdot_bench_config
    rc = fn(*args, ctypes.byref(c), ctypes.byref(w), ctypes.byref(b))
    return None if rc else (c.value, w.value, b.value)


def show(name, rows, bpr, r, peak, rpw="-"):
    cold, warm, byts = r
    els = rows * bpr * 256
    cold_gelem = els / (cold * 1e-6) / 1e9
    warm_gelem = els / (warm * 1e-6) / 1e9
    cold_peak = byts / (cold * 1e-6) / peak * 100
    warm_peak = byts / (warm * 1e-6) / peak * 100
    return (f"{name:6} {str(rpw):>3} {cold:9.2f} {cold/QUANTUM_US:6.0f}t {cold_gelem:8.0f} {cold_peak:6.1f}% "
            f"{warm:9.2f} {warm_gelem:8.0f} {warm_peak:6.1f}% {cold/warm:5.2f}")


HEAD = (f"{'fmt':6} {'rpw':>3} {'cold us':>9} {'ticks':>7} {'c Gel/s':>8} {'c peak':>7} "
        f"{'warm us':>9} {'w Gel/s':>8} {'w peak':>7} {'c/w':>5}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", action="store_true", help="%%peak against problem size, Q4_K")
    ap.add_argument("--rpw", action="store_true", help="rows-per-warp sweep at both shapes")
    ap.add_argument("--define", action="append", default=[], help="extra -D for the probe build")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--bpr", type=int, default=8, help="256-element blocks per output row (default: 8, K=2048)")
    ap.add_argument("--peak", type=float, default=1.792e12, help="DRAM peak in B/s (default: RTX 5090)")
    ap.add_argument("--tmp", default="/tmp", help="where to put the built probe")
    a = ap.parse_args()
    tmp = Path(a.tmp)
    lib = build(a.define, tmp)

    if a.sizes:
        print(f"Q4_K, runtime rows/bpr dispatch, sweeping only problem size at bpr={a.bpr}.")
        print(f"{'rows':>8} {'launch':>7} | " + HEAD)
        for rows in (2048, 8192, 16384, 32768, 65536, 131072, 262144):
            r = run(lib, 12, rows, a.bpr, a.reps)
            if r:
                # Ask the policy for a timing, not a guessed CTA count; the latter depends on its selected rpw and is
                # printed only for the default bpr=8 sweep documented in the handoff.
                rpw = lib.quactlize_cuda_vecdot_rows_per_warp(12, rows, a.bpr)
                print(f"{rows:8} {'policy':>7} | " + show("Q4_K", rows, a.bpr, r, a.peak, rpw))
        return

    if a.rpw:
        for rows, label in ((2048, f"N=2048, K={a.bpr * 256}, a decode layer when K matches"),
                            (131072, "the shape the defaults were tuned at")):
            print(f"\n-- rows={rows}: {label}")
            print(HEAD)
            for name, q in FORMATS:
                for rpw in (1, 2, 4, 8, 16):
                    r = run(lib, q, rows, a.bpr, a.reps, rpw)
                    if r:
                        print(show(name, rows, a.bpr, r, a.peak, rpw))
        return

    for rows, label in ((131072, "the throughput tuning shape"),
                        (2048, f"N=2048, K={a.bpr * 256}, the decode layer shape when K matches")):
        print(f"\n-- rows={rows}, bpr={a.bpr}: {label}")
        print(HEAD)
        for name, q in FORMATS:
            r = run(lib, q, rows, a.bpr, a.reps)
            if r:
                rpw = lib.quactlize_cuda_vecdot_rows_per_warp(q, rows, a.bpr)
                print(show(name, rows, a.bpr, r, a.peak, rpw))


if __name__ == "__main__":
    main()
