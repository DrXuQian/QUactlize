#!/usr/bin/env python3
"""THE CUDA-CORE GEMV's MEASUREMENT, as a command rather than someone's shell history.

Every number quoted for vecdot_rows_kernel came out of an ad-hoc script. acu_capture.sh's header records why that
is a defect in itself -- "the repo recorded acu's OUTPUT fields but never the command, so every profiling round
started by asking for it again" -- and this file is the same fix for the GEMV.

  ./benchmarks/gemv_bench.py               the shipped config at the tuning shape AND at a real layer
  ./benchmarks/gemv_bench.py --sizes       %peak against problem size: where saturation begins
  ./benchmarks/gemv_bench.py --rpw         rows-per-warp sweep at both shapes, which is how the mis-tuning was found
  ./benchmarks/gemv_bench.py --define X=1  add a -D to the probe build, e.g. GGUF_VECDOT_Q45_PAIRED=1

TWO THINGS THAT WILL MISLEAD YOU IF NOBODY SAYS THEM.

1. THE EVENT TIMER QUANTISES AT 2.048 us on this 5090 -- 31 of 31 distinct timings ever observed here are exact
   multiples. Every number below is printed with its tick count for that reason. A one-tick difference is NOT a
   result: at the 131072-row shape one tick is ~1%, but at a real layer it is 8%. More reps do not help; the median
   of quantised samples is quantised. Cold cannot batch launches inside one event pair either, because only the
   first launch after an L2 flush is cold -- so the only lever on cold resolution is a bigger PROBLEM.

2. `rows` IS THE OUTPUT DIMENSION N, and the tuning shape is not the shipping shape. rows=131072 at bpr=8 is
   N=131072, K=2048: sixty-four times a real dense layer at decode, which is rows=2048. The kernel reads 60.7% of
   peak at the first and 5.0% at the second, and the per-format rows-per-warp defaults chosen at the first are
   wrong at the second. That is why the default run below prints BOTH.
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
    return lib


def run(lib, qtype, rows, bpr, reps, rpw=None):
    c, w, b = ctypes.c_float(), ctypes.c_float(), ctypes.c_double()
    args = [qtype, rows, bpr, reps] if rpw is None else [qtype, rpw, rows, bpr, reps]
    fn = lib.quactlize_cuda_vecdot_bench if rpw is None else lib.quactlize_cuda_vecdot_bench_config
    rc = fn(*args, ctypes.byref(c), ctypes.byref(w), ctypes.byref(b))
    return None if rc else (c.value, w.value, b.value)


def show(name, rows, bpr, r, peak):
    cold, warm, byts = r
    els = rows * bpr * 256
    return (f"{name:6} {cold:9.2f} {cold/QUANTUM_US:6.0f}t {warm:9.2f} {cold/warm:5.2f} "
            f"{els/(cold*1e-6)/1e9:8.0f} {byts/(cold*1e-6)/peak*100:6.1f}%")


HEAD = f"{'fmt':6} {'cold us':>9} {'ticks':>7} {'warm us':>9} {'c/w':>5} {'Gelem/s':>8} {'%peak':>7}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", action="store_true", help="%%peak against problem size, Q4_K")
    ap.add_argument("--rpw", action="store_true", help="rows-per-warp sweep at both shapes")
    ap.add_argument("--define", action="append", default=[], help="extra -D for the probe build")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--peak", type=float, default=1.792e12, help="DRAM peak in B/s (default: RTX 5090)")
    ap.add_argument("--tmp", default="/tmp", help="where to put the built probe")
    a = ap.parse_args()
    tmp = Path(a.tmp)
    lib = build(a.define, tmp)

    if a.sizes:
        print("Q4_K, shipped config, sweeping only problem size. 16 rows per CTA at rpw=4.")
        print(f"{'rows':>8} {'CTAs':>7} | " + HEAD)
        for rows in (2048, 8192, 16384, 32768, 65536, 131072, 262144):
            r = run(lib, 12, rows, 8, a.reps)
            if r:
                print(f"{rows:8} {(rows+15)//16:7} | " + show("Q4_K", rows, 8, r, a.peak))
        return

    if a.rpw:
        for rows, label in ((2048, "N=K=2048, a real dense layer at decode"),
                            (131072, "the shape the defaults were tuned at")):
            print(f"\n-- rows={rows}: {label}")
            print(f"{'fmt':6} {'rpw':>4} | " + HEAD)
            for name, q in FORMATS:
                for rpw in (1, 2, 4, 8, 16):
                    r = run(lib, q, rows, 8, a.reps, rpw)
                    if r:
                        print(f"{'':6} {rpw:4} | " + show(name, rows, 8, r, a.peak))
        return

    for rows, label in ((131072, "the tuning shape"), (2048, "N=K=2048, what actually ships")):
        print(f"\n-- rows={rows}, bpr=8: {label}")
        print(HEAD)
        for name, q in FORMATS:
            r = run(lib, q, rows, 8, a.reps)
            if r:
                print(show(name, rows, 8, r, a.peak))


if __name__ == "__main__":
    main()
