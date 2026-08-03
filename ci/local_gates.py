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
  boxdry   build.sh itself, against a stub PPU SDK, as far as the compile. The only check that goes through
           actlize's example registration -- everything else verifies a piece of the build in isolation.
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
from concurrent.futures import ThreadPoolExecutor
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
    ("l100_fused_active",     []),
    # One entry per format that CAN activate. A format added to the packed path without an entry here is a format
    # whose decoder nobody has instantiated.
    # ONE ROW PER FORMAT. 0=Q4_K 1=Q5_K 2=Q2_K 3=Q3_K 4=Q6_K. A single row hardwired to format 2 is what this
    # gate had, which made the per-format activation check a one-format activation check.
    *[(f"l103_packed_format_active@fmt{f}", []) for f in (0, 1, 2, 3, 4)],
]

# (source, extra defines). A macro that changes types needs its own entry: the point of the front-end check is that
# ordinary parsing does not instantiate the combination someone will build on the box.
SYNTAX = [
    ("tests/test_q4k_packed_gemm.cu", ""),
    ("tests/test_q4k_packed_gemm.cu", "-DPPU_PACKED_SCALE=1"),
    # THE CONFIGURATION THAT SHIPPED BROKEN. kFusedScaleZero's definition referenced KernelConversionMode from a
    # point in the class where it is not yet declared, and the offending conjunct lives inside
    # `#if defined(PPU_PACKED_SCALE_FUSED)` -- so every build WITHOUT the macro preprocessed it away and compiled,
    # and the one build that used it was the only one that could not. Nothing here covered that combination, so the
    # define reached the box, was reported as a WARNING, applied to nothing, and produced a green correctness run and
    # an acu capture identical to pack. A macro that changes types needs its own row; this is why.
    ("tests/test_q4k_packed_gemm.cu", "-DPPU_PACKED_SCALE=1 -DPPU_PACKED_SCALE_FUSED=1"),
    # EVERY FORMAT THE PACKED PATH NOW CLAIMS. The generalisation is only real if each one instantiates, and the
    # limits it hit on the way -- a unit that is 2 mod 4 bytes against a cp.async that takes 4, 8 or 16, and a
    # second construction of the copy that spelled uint128 out while its declared type derived it -- were both
    # invisible until a format other than Q4_K was compiled. 0=Q4_K 1=Q5_K 2=Q2_K 3=Q3_K 4=Q6_K.
    # ONLY THE FORMATS THAT ACTUALLY ACTIVATE. Compiling the other two here proved the branch parses and nothing
    # more: every row of this fixture has Scale_TileK of 8 or 2, and the 16-group formats need 16, so their decoder
    # was never instantiated as live code while the gate reported clean. l103 is what checks activation; this row
    # covers the formats l103 says can be active. 0=Q4_K 1=Q5_K 2=Q2_K 3=Q3_K 4=Q6_K -- all five now, since the
    # two-plane collective gained the shared packed-scale channel and Q3/Q6 gained paired-unit staging.
    *[("tests/test_q4k_packed_gemm.cu", f"-DPPU_PACKED_SCALE=1 -DPPU_PACKED_FORMAT={f}") for f in (0, 1, 2, 3, 4)],
    ("tests/test_q4k_packed_gemm.cu", "-DPPU_PACKED_SCALE=1 -DPPU_PACKED_SPLIT_GROUPS=1"),
    ("tests/test_q4k_packed_gemm.cu", "-DPPU_PACKED_SCALE=1 -DPPU_SCALE_SWIZZLE=1"),
    ("tests/test_q4k_packed_gemm.cu", "-DPPU_SCALE_PAD=8"),
    ("tests/test_q4k_packed_gemm.cu", "-DPPU_B_DEQUANT_NOP=1"),
    ("tests/test_moe_grouped_verify.cu", ""),
    ("tests/test_moe_grouped_real.cu", ""),
    ("benchmarks/test_moe_splitk_bench.cu", ""),
    ("benchmarks/test_moe_splitk_bench.cu", "-DPPU_PACKED_SCALE=1"),
    # dev/'s top-level probes. They are DEVICE probes -- swzl_ldmatrix_probe reads the hardware swizzle, the
    # ablations and sweeps run on the accelerator -- so build.sh overlays them onto the box, and anything that
    # reaches the box belongs in the tier whose whole purpose is catching box-only compile failures locally.
    # They were absent from this list while they were also absent from the overlay, and the two facts hid each
    # other: cmake failed on a missing source before any of them could fail to compile.
    ("dev/swzl_ldmatrix_probe.cu", ""),
    ("dev/test_fold_int2.cu", ""),
    ("dev/test_int1_sweep.cu", ""),
    ("dev/test_moe_grouped_dataslice.cu", ""),
    ("dev/test_moe_grouped_probe.cu", ""),
    ("dev/test_q3_bconcat_ablate.cu", ""),
    ("dev/test_q3_bconcat_probe.cu", ""),
    ("dev/test_width_acu.cu", ""),
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
    """The whole tests/ directory -- and SKIPS ARE NOT PASSES.

    pytest exits 0 when tests skip, so this gate once reported the tier green over a run of "64 passed, 42 skipped":
    the built extension had been deleted and never rebuilt, so two fifths of the suite never executed and the summary
    line said 31/31. A skip is legitimate when the machine genuinely cannot run something -- no torch, no nvcc, no
    g++ -- and is a FAILURE when the reason is that something buildable here was not built. The two are told apart by
    the skip reason, because that is the only place the difference is recorded.

    The counts go in the message either way. A gate that hides how much it ran is a gate nobody can size."""
    rc, log, dt = run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider",
                       str(ROOT / "tests")], cwd=str(ROOT))
    summary = [l for l in log.splitlines() if "passed" in l or "failed" in l or "error" in l.lower()]
    msg = summary[-1] if summary else f"exit {rc}"
    if rc != 0:
        return "FAIL", msg, dt

    # "not built" is the actionable reason: the sources are here and so is the toolchain, so a skip means the tier is
    # reporting on code it did not run.
    fixable = [l for l in log.splitlines() if l.startswith("SKIPPED") and "not built" in l]
    if fixable:
        return "FAIL", (f"{msg} -- {len(fixable)} skipped because the extension is not built; "
                        f"run `python setup.py build_ext --inplace`"), dt
    return "PASS", msg, dt


def run(cmd, **kw):
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return p.returncode, (p.stdout + p.stderr), time.time() - t


GATE_FLAGS = {"l95_stub_vs_real": ["-D__HGGCCC__", "--expt-relaxed-constexpr"],
              # THE MACROS ARE THE POINT. This gate asserts the fused path is ON, so it has to be built the way the
              # box builds packfuse -- without these two it would assert about a configuration nobody runs and pass
              # for the wrong reason, which is the failure it exists to catch.
              **{f"l103_packed_format_active@fmt{f}":
                     ["-D__HGGCCC__", "--expt-relaxed-constexpr",
                      "-DPPU_PACKED_SCALE=1", f"-DPPU_PACKED_FORMAT={f}"] for f in (0, 1, 2, 3, 4)},
              "l100_fused_active": ["-D__HGGCCC__", "--expt-relaxed-constexpr",
                                    "-DPPU_PACKED_SCALE=1", "-DPPU_PACKED_SCALE_FUSED=1"]}


def lint_unroll():
    """Two unroll directives on one loop. hgcc rejects it; nvcc does not.

    WHY A LINT AND NOT A COMPILE. The syntax gate runs nvcc's front end, so it only catches what BOTH compilers
    dislike -- and hgcc is stricter here: `error: duplicate directives '#pragma unroll' and '#pragma unroll'` at
    ppu_mma_aiu_multistage_mixed_input.hpp, from an edit that added CUTLASS_PRAGMA_UNROLL above a loop that already
    had one. Every local check passed and the box did not build.

    The shape is why it survives reading: the two directives are separated by a COMMENT, so they are not adjacent
    lines and a plain grep for repeats finds nothing. This skips comments the way the compiler does.

    It does not close the general gap -- anything else where hgcc is stricter than nvcc is still invisible here --
    but it closes this one, which has now cost a box round trip.
    """
    import pathlib
    directives = {"CUTLASS_PRAGMA_UNROLL", "#pragma unroll", "CUTE_UNROLL", "CUTLASS_PRAGMA_NO_UNROLL"}
    roots = [ROOT / "third_party/actlize/include", ROOT / "quactlize/include"]
    hits = []
    for root in roots:
        if not root.exists():
            continue
        for f in list(root.rglob("*.h")) + list(root.rglob("*.hpp")) + list(root.rglob("*.cuh")):
            try:
                lines = f.read_text().split("\n")
            except Exception:
                continue
            prev = False
            for i, line in enumerate(lines, 1):
                t = line.strip()
                is_d = t in directives
                if is_d and prev:
                    hits.append(f"{f.relative_to(ROOT)}:{i}")
                if t and not t.startswith("//"):
                    prev = is_d
    if hits:
        return "FAIL", "two unroll directives on one loop (hgcc rejects this, nvcc does not): " + ", ".join(hits[:4]), 0.0
    return "PASS", "no duplicate unroll directives", 0.0


def lint_ppu_asm_device_guard():
    """Reject the compiler-pass guard that previously exposed PPU asm to hgcc's host pass.

    `__HGGCCC__` is the analogue of `__CUDACC__`: it is true in both host and device passes. PPU instructions must
    instead sit behind `__HGGC_ARCH__`, the analogue of `__CUDA_ARCH__`. The exact compiler-only spelling below was
    used by the native GGUF byte-permute trial and cannot be falsified by nvcc on a machine without hgcc.
    """
    roots = [ROOT / "quactlize/include", ROOT / "third_party/actlize/include"]
    bad = []
    # This existing arm is box-proven: test_q4k_packed_gemm rowC passed 5/5 on ppu001 and necessarily traverses
    # ppu_f16x2_sub/fma. Replacing its working compiler guard with an architecture guard cannot be falsified without
    # hgcc and could silently select the slower scalar fallback, so only NEW uses are rejected locally.
    proven_exemptions = {"third_party/actlize/include/cutlass/gguf_packed_scale.h"}
    compiler_only = re.compile(r"^\s*#(?:if|elif).*__HGGCCC__.*!\s*defined\s*\(\s*__NVCC__\s*\)")
    for root in roots:
        if not root.exists():
            continue
        for f in list(root.rglob("*.h")) + list(root.rglob("*.hpp")) + list(root.rglob("*.cuh")):
            try:
                lines = f.read_text().splitlines()
            except Exception:
                continue
            for i, line in enumerate(lines, 1):
                rel = str(f.relative_to(ROOT))
                if compiler_only.search(line) and rel not in proven_exemptions:
                    bad.append(f"{rel}:{i}")
    if bad:
        return ("FAIL", "PPU device asm/compiler branch must use __HGGC_ARCH__, not "
                "__HGGCCC__ && !__NVCC__: " + ", ".join(bad[:4]), 0.0)
    return "PASS", "no new PPU device branch uses the hgcc compiler-pass guard (1 box-proven exemption)", 0.0


def gate(name, args):
    """`name` may carry an @variant suffix: one source, several configurations, distinct rows.

    l103 asks whether a format's packed path is ACTIVE, and it was registered once, hardwired to format 2. So the
    gate whose entire purpose is per-format activation covered one format -- the same shape as the bug it had just
    been rewritten to catch. Five entries need five names, and five names for one file need this.
    """
    base = name.split("@", 1)[0]
    src = DEV / f"{base}.cu"
    if not src.exists():
        return "MISSING", f"{src} not found", 0.0
    OUT.mkdir(parents=True, exist_ok=True)
    exe = OUT / name.replace("@", "_")
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


def lint_stale_repo_path():
    """An absolute path naming a repo directory that is not this one.

    THE FAILURE THIS COMES FROM. The benchmark binaries printed their profile hint as `$BIN/test_gemv_perf ...`,
    and the single document that defined $BIN still read

        BIN=/sim/eec/shared/junfu.qx/<the-pre-rename-repo-dir>/third_party/actlize/build_w4a16_compare/...

    long after the repo was renamed to quactlize. Copying the printed hint therefore produced "No such file or
    directory" on a path the operator never typed -- and the hint itself looked correct, because the stale half was
    in a different file. Nothing could catch it: it is not code, it does not compile, and no test reads a doc.

    The structural fix was to print argv[0] and derive BIN from $PWD, so this only has to refuse REINTRODUCTION.
    It checks the shared-filesystem prefix rather than a list of old names, because the next rename will invent a
    name this file has never heard of.
    """
    import re as _re
    root_name = ROOT.name
    # Two slots: /sim/eec/shared/<user>/<dir>. The model store sits at <AI_workspace>/<llm-models>, i.e. it
    # occupies the USER slot -- an allowlist keyed only on the second component never fires for it.
    pat = _re.compile(r"/sim/eec/shared/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")
    hits = []
    for f in sorted(ROOT.rglob("*")):
        if not f.is_file() or "third_party" in f.parts or ".git" in f.parts:
            continue
        if f.suffix not in (".md", ".sh", ".py", ".cu", ".cuh", ".hpp", ".h", ".cpp", ".txt", ".in"):
            continue
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for m in pat.finditer(line):
                user, d = m.group(1), m.group(2)
                # PPU_SDK and the shared model store are genuinely elsewhere on the box; only repo dirs are the
                # hazard, because only a repo dir gets renamed out from under a written-down path.
                if user == "AI_workspace" or d in (root_name, "PPU_SDK"):
                    continue
                hits.append(f"{f.relative_to(ROOT)}:{i} -> .../{d}/")
    if hits:
        return ("FAIL",
                f"absolute path names a repo dir that is not '{root_name}' (a rename left it behind): "
                + ", ".join(hits[:4]), 0.0)
    return "PASS", f"no absolute path names a repo dir other than '{root_name}'", 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("-k", default="", help="only run checks whose name contains this")
    ap.add_argument("--strict", action="store_true",
                    help="treat SKIP as failure. The local tier tolerates a skip -- a machine without nvcc or gcc "
                         "genuinely cannot answer -- but a pre-commit or CI use must not, or the one check that "
                         "covers actlize's example registration can be absent and the exit status still zero, "
                         "which is how the deleted registration reached the box.")
    a = ap.parse_args()

    items = ([("gate", n, args) for n, args in GATES]
             + [("syntax", f"{Path(s).name} {d}".strip(), (s, d)) for s, d in SYNTAX]
             + [("overlay", "CMake targets vs the overlay", None),
                ("boxdry", "build.sh through actlize, stub SDK: test_moe_splitk_bench", "test_moe_splitk_bench"),
                ("boxdry", "build.sh through actlize, stub SDK: test_q4k_packed_gemm",
                 "test_q4k_packed_gemm PPU_PACKED_SCALE=1"),
                ("asan", "preprocessing chain under ASAN", None),
                ("pytest", "torch op tests", None),
                ("lint", "duplicate unroll directives (hgcc-only error)", lint_unroll),
                ("lint", "PPU asm uses device-pass architecture guard", lint_ppu_asm_device_guard),
                ("lint", "absolute paths name this repo dir, not a renamed one", lint_stale_repo_path),
    ("registry", "declarations vs source", None)])
    items = [i for i in items if a.k in i[1]]
    if a.list:
        for kind, name, _ in items:
            print(f"  {kind:<9} {name}")
        return 0

    print(f"== quactlize local tier: {len(items)} checks, no device needed ==")

    # RUN THEM CONCURRENTLY. Every check here is an independent process -- an nvcc front end, a g++ build, a cmake
    # configure -- and running twenty of them one after another took ten minutes, which is long enough that the tier
    # stopped being run after small edits. That is the failure mode: a gate nobody waits for is a gate nobody runs.
    # Results are collected first and printed in declaration order, so the output is still a stable table.
    def run_one(item):
        kind, name, payload = item
        if kind == "gate":
            return gate(name, payload)
        if kind == "syntax":
            return syntax(*payload)
        if kind == "boxdry":
            args = payload.split()
            rc, log, dt = run(["bash", str(ROOT / "ci/box_build_dryrun.sh"), args[0]]
                              + ([" ".join(args[1:])] if len(args) > 1 else []))
            last = [l for l in log.splitlines() if l.strip()]
            st = {0: "PASS", 2: "SKIP"}.get(rc, "FAIL")
            msg = next((l.strip() for l in last if "[ok]" in l or "[FAIL]" in l or "[SKIP]" in l),
                       last[-1].strip() if last else f"exit {rc}")
            return st, msg, dt
        if kind == "overlay":
            rc, log, dt = run([sys.executable, str(ROOT / "dev/fold_derivation/overlay_targets_check.py")])
            last = [l for l in log.splitlines() if l.strip()]
            return {0: "PASS", 2: "SKIP"}.get(rc, "FAIL"), (last[-1].strip() if last else f"exit {rc}"), dt
        if kind == "lint":
            return payload()
        if kind == "asan":
            return asan()
        if kind == "pytest":
            return pytests()
        sys.path.insert(0, str(ROOT / "ci"))
        import registry
        probs = registry.check()
        return ("PASS" if not probs else "FAIL"), (probs[0] if probs else "0 problems"), 0.0

    t0 = time.time()
    # NOT EVERYTHING IS INDEPENDENT. The boxdry checks each run the real build.sh, which REGISTERS our example in
    # actlize's examples/CMakeLists.txt and restores that file on exit -- two of them at once means one restores the
    # file while the other still needs the registration, and the second reports "our CMakeLists was not reached".
    # Concurrency found that immediately, which is the right outcome; they run one at a time and everything else
    # concurrently. Bounded workers because each nvcc front end takes GBs.
    exclusive = [i for i in items if i[0] == "boxdry"]
    parallel = [i for i in items if i[0] != "boxdry"]
    got = {}
    with ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4))) as pool:
        futures = {id(i): pool.submit(run_one, i) for i in parallel}
        for i in exclusive:                       # serialised, and overlapped with the pool's work
            got[id(i)] = run_one(i)
        for k, f in futures.items():
            got[k] = f.result()
    results = [got[id(i)] for i in items]

    fails = []
    for (kind, name, _), (st, msg, dt) in zip(items, results):
        mark = {"PASS": "ok  ", "FAIL": "FAIL", "BUILD": "BLD!", "MISSING": "MISS", "SKIP": "skip"}[st]
        print(f"  [{mark}] {kind:<8} {name:<44} {dt:5.1f}s  {msg[:88]}")
        if st in ("FAIL", "BUILD", "MISSING") or (a.strict and st == "SKIP"):
            fails.append(f"{kind}/{name}: {msg}" + (" (skipped, and --strict was given)" if st == "SKIP" else ""))
    print(f"  wall clock {time.time() - t0:.0f}s")

    print(f"\n== {len(items) - len(fails)}/{len(items)} passed or skipped ==")
    for f in fails:
        print(f"   FAILED  {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
