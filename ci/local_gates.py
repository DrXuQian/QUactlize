#!/usr/bin/env python3
"""THE TIER THAT NEEDS NO DEVICE. Runs every check that can falsify something without a PPU, and reports one table.

Why this exists and why it is separate from the device tier: nearly every wrong result this project produced was
wrong in a way a local check could have caught -- a layout claimed rather than printed, a hypothesis never fed back
through the fixture, a bench comparing two kernels that computed different numbers, a gate that measured nothing.
Those need no hardware. Keeping them in one runner with one exit code makes "the local tier is green" a fact instead
of a memory of having run some of them.

Three kinds of check:

  gate     an l9x program: compiles with nvcc against the stub headers and asserts something about the real types,
           layouts or arithmetic. Its exit code is the verdict.
  syntax   nvcc's front end over a device source, diffed against a recorded baseline of accepted noise. Catches
           template instantiation failures that only appear under a macro combination.
  registry the coverage declarations against the source (see registry.py).
  asan     the host preprocessing chain compiled with -fsanitize=address and swept over shapes. It found two
           out-of-bounds accesses that no assertion could have located: both corrupted the heap silently and
           surfaced as an intermittent Bus error or SIGSEGV in an unrelated test, several tests later.
  pytest   the torch-op tests, if the extension is built and torch is importable. Skipped, not failed, otherwise --
           this tier must stay runnable on a machine with neither.

  ./ci/local_gates.py            run everything
  ./ci/local_gates.py --list     show what would run
  ./ci/local_gates.py -k q4k     run only matching names
"""
import argparse, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEV = ROOT / "dev" / "fold_derivation"
STUB = DEV / "stub_inc"
ACT = ROOT / "third_party" / "actlize" / "include"
OUT = Path(os.environ.get("QUACTLIZE_CI_OUT", "/tmp/quactlize_ci"))

# l9x gates. `args` is passed to the built binary; a fixture path is given relative to the repo root so a gate that
# needs data says so here rather than defaulting to a path that happens to exist.
# name -> (argv, extra nvcc flags). l95 asserts TYPE IDENTITY against the collective and needs -D__HGGCCC__, or
# CUTLASS_DEVICE degrades to host `inline` and every __syncthreads lands in host code -- its own header says so, and
# the first version of this runner reported it as a build failure of the tree rather than of the invocation.
GATES = [
    ("l91_gguf_scale_gate",   ["tests/data/scale_blocks_q4k.bin"]),
    ("l92_kbias_general",     []),
    ("l93_scale_decode",      ["tests/data/scale_blocks_q4k.bin"]),
    ("l94_native_scale_path", []),
    ("l95_stub_vs_real",      []),
    ("l96_packed_pair",       ["tests/data/scale_blocks_q4k.bin"]),
    ("l97_packed_g2s_threads",[]),
    ("l98_scale_swizzle",     []),
    ("l99_bench_like_for_like", []),
]

# (source, extra defines). A macro that changes types needs its own entry: the point of the front-end check is that
# ordinary parsing does not instantiate the combination someone will build on the box.
SYNTAX = [
    ("tests/test_q4k_packed_gemm.cu", ""),
    ("tests/test_q4k_packed_gemm.cu", "-DPPU_PACKED_SCALE=1"),
    ("tests/test_q4k_packed_gemm.cu", "-DPPU_PACKED_SCALE=1 -DPPU_PACKED_SPLIT_GROUPS=1"),
    ("tests/test_q4k_packed_gemm.cu", "-DPPU_PACKED_SCALE=1 -DPPU_SCALE_SWIZZLE=1"),
    ("tests/test_q4k_packed_gemm.cu", "-DPPU_SCALE_PAD=8"),
    ("tests/test_q4k_packed_gemm.cu", "-DPPU_B_DEQUANT_NOP=1"),
    ("tests/test_moe_grouped_verify.cu", ""),
    ("tests/test_moe_grouped_real.cu", ""),
    ("benchmarks/test_moe_splitk_bench.cu", ""),
    ("benchmarks/test_moe_splitk_bench.cu", "-DPPU_PACKED_SCALE=1"),
]

NVCC = ["nvcc", "-std=c++17", "-x", "cu", "-arch=sm_80", "-w"]

# The ASAN probe: an ordinary host C++ build, so it needs no CUDA toolchain beyond cuda_fp16.h for the `half` type.
ASAN_SRC = ROOT / "ci" / "asan_preprocess_probe.cpp"
PREPROC = ROOT / "quactlize" / "csrc" / "preprocess"


def asan():
    if not ASAN_SRC.exists():
        return "MISSING", f"{ASAN_SRC} not found", 0.0
    OUT.mkdir(parents=True, exist_ok=True)
    exe = OUT / "asan_preprocess_probe"
    cuda_inc = [p for p in ("/usr/local/cuda/include",) if Path(p).exists()]
    rc, log, dt = run(["g++", "-std=c++17", "-O1", "-g", "-fsanitize=address", "-fno-omit-frame-pointer",
                       "-DUSE_AIU=1", "-w", "-I", str(PREPROC), "-I", str(ROOT / "third_party/cutlass/include")]
                      + sum([["-I", p] for p in cuda_inc], [])
                      + ["-o", str(exe), str(ASAN_SRC), str(PREPROC / "cutlass_kernels/cutlass_preprocessors.cpp")])
    if rc != 0:
        first = next((l for l in log.splitlines() if " error" in l), "g++ failed")
        return "BUILD", first, dt
    rc, log, dt2 = run([str(exe)])
    # ASAN aborts on the first violation, so the exit status is the verdict; the summary line only appears on success.
    last = [l for l in log.splitlines() if "no violation" in l or "ERROR: AddressSanitizer" in l]
    return ("PASS" if rc == 0 else "FAIL"), (last[-1] if last else f"exit {rc}"), dt + dt2


def pytests():
    try:
        import torch  # noqa: F401
    except ImportError:
        return "SKIP", "torch not importable -- the host half cannot be exercised here", 0.0
    if not list((ROOT / "quactlize").glob("_C*.so")):
        return "SKIP", "extension not built (python setup.py build_ext --inplace)", 0.0
    rc, log, dt = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                       str(ROOT / "tests" / "test_preprocess_ops.py")], cwd=str(ROOT))
    summary = [l for l in log.splitlines() if "passed" in l or "failed" in l or "error" in l.lower()]
    return ("PASS" if rc == 0 else "FAIL"), (summary[-1] if summary else f"exit {rc}"), dt


def run(cmd, **kw):
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return p.returncode, (p.stdout + p.stderr), time.time() - t


GATE_FLAGS = {"l95_stub_vs_real": ["-D__HGGCCC__", "--expt-relaxed-constexpr"]}


def gate(name, args):
    src = DEV / f"{name}.cu"
    if not src.exists():
        return "MISSING", f"{src} not found", 0.0
    OUT.mkdir(parents=True, exist_ok=True)
    exe = OUT / name
    rc, log, dt = run(NVCC + GATE_FLAGS.get(name, []) +
                      ["-I", str(STUB), "-I", str(ACT), "-I", str(ROOT / "quactlize/include"),
                       "-I", str(ROOT / "tests"), "-o", str(exe), str(src)])
    if rc != 0:
        return "BUILD", log.strip().splitlines()[0] if log.strip() else "nvcc failed", dt
    rc, log, dt2 = run([str(exe)] + [str(ROOT / a) for a in args])
    tail = [l for l in log.splitlines() if l.strip()]
    return ("PASS" if rc == 0 else "FAIL"), (tail[-1] if tail else ""), dt + dt2


def syntax(src, defs):
    sc = DEV / "syntax_check.sh"
    if not sc.exists():
        return "MISSING", "syntax_check.sh not found", 0.0
    env = dict(os.environ, EXTRA_DEFS=defs, GEN_INC=str(DEV / "gen_stub"))
    rc, log, dt = run(["bash", str(sc), str(ROOT / src)], cwd=str(DEV), env=env)
    last = [l for l in log.splitlines() if l.strip()]
    return ("PASS" if rc == 0 else "FAIL"), (last[-1] if last else ""), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("-k", default="", help="only run checks whose name contains this")
    a = ap.parse_args()

    items = ([("gate", n, args) for n, args in GATES]
             + [("syntax", f"{Path(s).name} {d}".strip(), (s, d)) for s, d in SYNTAX]
             + [("asan", "preprocessing chain under ASAN", None),
                ("pytest", "torch op tests", None),
                ("registry", "declarations vs source", None)])
    items = [i for i in items if a.k in i[1]]
    if a.list:
        for kind, name, _ in items:
            print(f"  {kind:<9} {name}")
        return 0

    print(f"== quactlize local tier: {len(items)} checks, no device needed ==")
    fails = []
    for kind, name, payload in items:
        if kind == "gate":
            st, msg, dt = gate(name, payload)
        elif kind == "syntax":
            st, msg, dt = syntax(*payload)
        elif kind == "asan":
            st, msg, dt = asan()
        elif kind == "pytest":
            st, msg, dt = pytests()
        else:
            sys.path.insert(0, str(ROOT / "ci"))
            import registry
            probs = registry.check()
            st, msg, dt = ("PASS" if not probs else "FAIL"), (probs[0] if probs else "0 problems"), 0.0
        mark = {"PASS": "ok  ", "FAIL": "FAIL", "BUILD": "BLD!", "MISSING": "MISS", "SKIP": "skip"}[st]
        print(f"  [{mark}] {kind:<8} {name:<44} {dt:5.1f}s  {msg[:88]}")
        if st not in ("PASS", "SKIP"):
            fails.append(f"{kind}/{name}: {msg}")

    print(f"\n== {len(items) - len(fails)}/{len(items)} passed or skipped ==")
    for f in fails:
        print(f"   FAILED  {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
