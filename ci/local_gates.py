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
    # Does a different configuration actually give a different LOW-plane layout? Answers a contradiction
    # between layouts.py xplane() and l61, and produces the prune table.
    ("l105_low_plane_config_classes", []),
    ("l106_compact_a_rows", []),
    ("l107_moe_router_fixture", []),
    ("l108_rt_error_contract", []),
    ("l109_rt_hggc_parse", []),
    ("l110_unit_pack_abi", []),
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
    # The VALUE is the compact A row capacity. All three small-M specialisations must instantiate the full
    # collective; testing only the historical boolean value 1 would leave the new hierarchical layout dead.
    *[(src, f"-DPPU_A_CPASYNC={r}")
      for src in ("tests/test_moe_grouped_verify.cu", "tests/test_fpA_intB_ppu.cu")
      for r in (1, 2, 4)],
    ("tests/test_moe_grouped_real.cu", ""),
    ("benchmarks/test_moe_splitk_bench.cu", ""),
    # THE BENCH THE WHOLE SWEEP RUNS THROUGH, and it had no local gate at all until 2026-08-04 -- which is how
    # a selection rewrite (median over interleaved repeats, tie reporting) got committed without ever being
    # compiled. It is also the file with the most host-side logic of any bench, so it is the one most likely to
    # break in a way nvcc catches and review does not.
    ("benchmarks/test_lowbit_moe_bench.cu", ""),
    # The DENSE bench, likewise ungated until 2026-08-04. It now carries a generated config table and a
    # static_assert tying that table to the binary's (bits, TileK); this row is what makes a stale table fail
    # here instead of producing a sweep over tactics the binary cannot select.
    ("benchmarks/test_lowbit_dense_bench.cu", ""),
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
              "l109_rt_hggc_parse": ["-D__HGGCCC__"],
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
                       "-I", str(ROOT / "tests"), "-I", str(ROOT / "benchmarks"),
                       "-o", str(exe), str(src)])
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


def lint_ppu_portability():
    """THE BOX-BUILT SOURCES MUST BE IN THE PORTABLE SUBSET -- checked here, not only inside build.sh.

    dev/fold_derivation/ppu_portability_check.py already existed and runs in under a second, but it was only
    invoked from build.sh (line 138). The local tier reaches build.sh solely through the `boxdry` gate, and that
    gate hangs on a googletest clone -- so in practice this check was unreachable from the tier whose entire
    purpose is catching box failures before the box.

    On 2026-08-04 that cost a round trip: benchmarks/bench_floor.cuh used cudaMalloc/cudaSuccess, the local
    syntax gate compiled it happily (nvcc accepts the NVIDIA runtime by definition), and the box rejected it.
    The check that would have caught it in zero seconds was sitting one directory away.
    """
    script = ROOT / "dev" / "fold_derivation" / "ppu_portability_check.py"
    if not script.is_file():
        return "SKIP", "no ppu_portability_check.py", 0.0
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=ROOT)
    line = next((l.strip() for l in (r.stdout + r.stderr).splitlines() if l.strip()), f"exit {r.returncode}")
    return ("PASS" if r.returncode == 0 else "FAIL"), line, 0.0


def lint_fixture_flags():
    """THE EMITTED INVOCATIONS MUST NAME OPTIONS THE BENCH ACTUALLY PARSES.

    benchmarks/fixtures.py exists so nobody types `"$BIN" 256 128 512 2048 32 2` by hand. That removes one class
    of error and introduces another: cutlass' CommandLine does not complain about an unknown --flag, so a
    renamed option turns into a silently ignored argument and the bench runs its DEFAULT shape while the log
    header says otherwise. Grep the parser for each name the emitter uses.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("fx", ROOT / "benchmarks" / "fixtures.py")
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    src = (ROOT / "benchmarks" / "test_lowbit_dense_bench.cu").read_text()
    missing = [f for f in fx.DENSE_FLAGS
               if f'get_cmd_line_argument("{f}"' not in src and f'check_cmd_line_flag("{f}"' not in src]
    if missing:
        return "FAIL", ("fixtures.py emits --" + ", --".join(missing)
                        + " but test_lowbit_dense_bench.cu does not parse them; an unknown flag is IGNORED, so the "
                          "bench would run its default shape and the log would not say so"), 0.0
    return "PASS", f"all {len(fx.DENSE_FLAGS)} emitted dense options are parsed by the bench", 0.0


def lint_tactic_cannot_change_offline_layout():
    """SELECTING A TACTIC MUST NOT CHANGE THE BYTES ON DISK, and ONE registry must decide what they are.

    The offline packer arranges weights for a specific (qtype, TileK); the weights are written once. So no
    runtime choice may imply a different TileK, or the kernel reads bytes laid out for something else -- a
    failure that produces numbers, not an error.

    THIS GATE'S OWN HISTORY IS THE ARGUMENT FOR ITS CURRENT SHAPE. Its first version compared the packer's
    hardcoded _tile_k against the library's hardcoded dispatch and PASSED -- while both said 256 and every
    measurement said 64 for int4. Two copies agreeing is not a reference. quactlize/include/ppu_format_config.inc
    is now the one source (INBOX 056), so the checks below are "does each consumer READ it" and "does it obey the
    rule it states", not "do the copies agree".

    FOUR CHECKS:
      1. No compiled config row carries TileK. Rows are (ID, NAME, TM, TN, WM, WN, STAGES); TileK comes from the
         enclosing template, so no config CAN move the layout. The emitter is parameterised on tile_k, so a sixth
         numeric field would be an easy and invisible way to break that.
      2. The registry obeys its own stated rule. Its header says SCALE_FIRST TileK is "the canonical minimum
         32-byte run for the narrowest code plane: 256/bits", using the high plane for two-plane formats. A rule
         written in a comment beside rows that violate it is worse than no rule.
      3. The bench's per-width TileK defaults equal that same 256/bits. This is the comparison the first version
         could not make and the one that would have caught the wrong value.
      4. The packer READS the registry rather than restating it, and schemes.CODE_PLANE agrees with the
         registry's plane widths -- that was a third spelling of "which planes does this format have".
    """
    import ast as _ast, re as _re
    inc = ROOT / "quactlize" / "include" / "ppu_format_config.inc"
    bench = ROOT / "benchmarks" / "test_lowbit_dense_bench.cu"
    pk = ROOT / "tools" / "pack_gguf.py"
    for f in (inc, bench, pk):
        if not f.is_file():
            return "FAIL", f"{f.name} is missing", 0.0

    # 1 -- config rows carry no TileK.
    for cfg in sorted((ROOT / "quactlize" / "include").glob("ppu_*_configs.inc")):
        for line in cfg.read_text().splitlines():
            m = _re.match(r"\s*X\((.*?)\)\s*\\?\s*$", line)
            if m and len(m.group(1).split(",")) != 7:
                return "FAIL", (f"{cfg.name}: '{line.strip()[:60]}' has {len(m.group(1).split(','))} args, "
                                f"expected 7. A TileK field here would let a tactic choice move the offline "
                                f"layout"), 0.0

    # 2 -- the registry against the rule its own header states.
    rows = []
    for m in _re.finditer(r"^\s*X\((.*?)\)\s*\\?\s*$", inc.read_text(), _re.M):
        a = [x.strip().strip('"') for x in m.group(1).split(",")]
        if len(a) != 9:
            return "FAIL", f"{inc.name}: row has {len(a)} args, expected 9: {m.group(1)[:60]}", 0.0
        rows.append(dict(name=a[1], qtype=int(a[2]), low=int(a[3]), high=int(a[4]),
                         gs=int(a[5]), scale_first=int(a[6]), fq=int(a[7])))
    if not rows:
        return "FAIL", f"{inc.name}: no rows parsed; the parser is wrong, not the registry", 0.0
    for r in rows:
        narrowest = r["high"] if r["high"] else r["low"]
        want = 256 // narrowest
        if r["scale_first"] != want:
            return "FAIL", (f"{inc.name}: {r['name']} declares scale_first_tile_k={r['scale_first']} but its own "
                            f"header's rule -- 256/bits over the narrowest plane ({narrowest}) -- gives {want}"), 0.0

    # 3 -- the bench's per-width defaults against the same rule.
    src = bench.read_text()
    want_bench = {1: 256, 2: 128, 4: 64}
    got = {}
    m1 = _re.search(r"BENCH_UINT1.*?#define BENCH_TSK (\d+)", src, _re.S)
    m2 = _re.search(r"BENCH_UINT2.*?#define BENCH_TSK (\d+)", src, _re.S)
    m4 = _re.search(r"constexpr int TileShapeK = 128 \* 8 / sizeof_bits<MmaType>::value", src)
    if m1: got[1] = int(m1.group(1))
    if m2: got[2] = int(m2.group(1))
    if m4: got[4] = 128 * 8 // 16          # the expression, evaluated: MmaType is half_t
    if set(got) != {1, 2, 4}:
        return "FAIL", (f"could not read the bench's per-width TileK defaults (found {sorted(got)}); the parser "
                        f"is wrong, not the bench, and a partial read would compare fewer widths silently"), 0.0
    for bits, v in sorted(got.items()):
        if v != want_bench[bits]:
            return "FAIL", (f"the bench's int{bits} TileK default is {v}; the registry's rule (256/bits) gives "
                            f"{want_bench[bits]}. This is the comparison that would have caught the wrong "
                            f"shipping value"), 0.0

    # 4 -- the packer reads rather than restates, and CODE_PLANE agrees.
    tree = _ast.parse(pk.read_text())
    for fn_name in ("_tile_k", "_low_bits", "_high_bits"):
        fn = next((n for n in tree.body if isinstance(n, _ast.FunctionDef) and n.name == fn_name), None)
        if fn is None:
            return "FAIL", f"tools/pack_gguf.py has no {fn_name}; the packer's arrangement source moved", 0.0
        literals = [n.value for n in _ast.walk(fn) if isinstance(n, _ast.Constant) and isinstance(n.value, int)]
        if literals:
            return "FAIL", (f"{fn_name} contains integer literal(s) {literals} -- it should READ "
                            f"ppu_format_config.inc, not restate it. That restatement is what made the packer "
                            f"and the library agree on a value nothing measured"), 0.0
    import importlib
    sys.path.insert(0, str(ROOT))
    try:
        schemes = importlib.import_module("quactlize.schemes")
        QuantType = importlib.import_module("quactlize.formats").QuantType
    except Exception as e:                                          # noqa: BLE001
        return "SKIP", f"quactlize not importable ({e}); the CODE_PLANE cross-check needs it", 0.0
    LO = {"i2": 2, "i2+i1": 2, "i4": 4, "i4+i1": 4, "i4+i2": 4}
    HI = {"i2": 0, "i2+i1": 1, "i4": 0, "i4+i1": 1, "i4+i2": 2}
    for r in rows:
        cp = schemes.CODE_PLANE[QuantType(r["qtype"])]
        if (LO[cp], HI[cp]) != (r["low"], r["high"]):
            return "FAIL", (f"{r['name']}: schemes.CODE_PLANE says {cp} = ({LO[cp]},{HI[cp]}) bits, the registry "
                            f"says ({r['low']},{r['high']}). Two spellings of which planes a format has"), 0.0

    return "PASS", (f"no config row carries TileK; all {len(rows)} registry rows obey 256/narrowest-plane; the "
                    f"bench's int1/2/4 defaults match it; the packer reads the registry and CODE_PLANE agrees"), 0.0


def lint_config_abi_matches_header():
    """THE ctypes MIRROR OF quactlize_ppu_config_v1 MUST MATCH THE HEADER, FIELD FOR FIELD, IN ORDER.

    tools/list_shipped.py reads the library's config inventory through ctypes. That Structure is a SECOND COPY of
    a layout defined in quactlize/include/quactlize_ppu_config.h, and the two going out of step does not fail
    loudly: ctypes reads whatever bytes are at the offsets it believes in and returns plausible integers. A
    reordered or inserted field yields a shipped-config list that is wrong in a way no downstream check can see,
    because every value still looks like a tile dimension.

    So this parses the header's struct body and compares names, order and type class against the ctypes
    _fields_. The header is the definition; the mirror follows it.
    """
    import re as _re
    hdr = ROOT / "quactlize" / "include" / "quactlize_ppu_config.h"
    tool = ROOT / "tools" / "list_shipped.py"
    if not hdr.is_file():
        return "SKIP", "quactlize_ppu_config.h not present yet", 0.0
    if not tool.is_file():
        return "FAIL", "tools/list_shipped.py is missing but the header it mirrors exists", 0.0

    body = _re.search(r"typedef\s+struct\s+quactlize_ppu_config_v1\s*\{(.*?)\}", hdr.read_text(), _re.S)
    if not body:
        return "FAIL", "could not find the quactlize_ppu_config_v1 struct body in the header", 0.0
    C_TO_CLASS = {"bool": "bool", "char const*": "str", "const char*": "str",
                  "int32_t": "int", "int64_t": "int", "uint32_t": "int", "float": "float"}
    hdr_fields = []
    for line in body.group(1).splitlines():
        line = _re.sub(r"//.*$", "", line).strip().rstrip(";")
        if not line or line.startswith("/*") or line.startswith("*"):
            continue
        m = _re.match(r"^(.*?)\s*\*?\s*(\w+)$", line)
        if not m:
            continue
        ctype, name = m.group(1).strip(), m.group(2)
        if "char" in ctype:
            ctype = "char const*"
        hdr_fields.append((name, C_TO_CLASS.get(ctype, ctype)))
    if not hdr_fields:
        return "FAIL", "parsed the struct body but found no fields; the parser, not the header, is wrong", 0.0

    # PARSED, NOT IMPORTED, and this is not a style preference. The first version imported list_shipped.py via
    # importlib and read Config._fields_ -- and reported on code that was NOT on disk. A .pyc is revalidated on
    # the source's mtime at ONE-SECOND resolution plus its byte SIZE, and the change that was being tested
    # (swapping two adjacent field lines) leaves the size identical; the restore landed in the same second. Both
    # validity checks passed and the loader served the stale bytecode. A gate that can report on code that does
    # not exist is worse than no gate, and nothing about its output said which code it read.
    #
    # ast.parse reads the file every time and has no cache to be stale.
    import ast as _ast
    PY_TO_CLASS = {"c_bool": "bool", "c_char_p": "str", "c_int32": "int",
                   "c_int64": "int", "c_uint32": "int", "c_float": "float"}
    tree = _ast.parse(tool.read_text())
    fields_node = None
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Assign) and any(
                getattr(t, "id", None) == "_fields_" for t in node.targets):
            fields_node = node.value
            break
    if fields_node is None:
        return "FAIL", "no _fields_ assignment found in list_shipped.py; the mirror cannot be checked", 0.0
    py_fields = []
    for elt in fields_node.elts:
        name = elt.elts[0].value
        ctype = elt.elts[1].attr if isinstance(elt.elts[1], _ast.Attribute) else getattr(elt.elts[1], "id", "?")
        py_fields.append((name, PY_TO_CLASS.get(ctype, ctype)))

    if py_fields != hdr_fields:
        n = min(len(py_fields), len(hdr_fields))
        first = next((i for i in range(n) if py_fields[i] != hdr_fields[i]), n)
        return "FAIL", (f"ctypes mirror and header disagree at field {first}: header has "
                        f"{hdr_fields[first] if first < len(hdr_fields) else '<end>'}, list_shipped.py has "
                        f"{py_fields[first] if first < len(py_fields) else '<end>'}. ctypes will read the wrong "
                        f"offsets and return plausible numbers rather than fail"), 0.0
    return "PASS", f"ctypes mirror matches the header's {len(hdr_fields)} fields in order and type class", 0.0


def lint_tactic_spaces_agree():
    """DENSE AND GROUPED MUST SEARCH THE SAME SET, or a comparison between them measures the sets.

    ppu_tactic_space.hpp keeps DenseSpace and GroupedSpace as separate wrappers over one implementation and says
    why at its line 185 -- so that future divergence stays visible instead of being hidden by sharing. It names
    the mechanism that makes that work: "The emitter asks each launcher for its own answer and a comparator
    checks them." The comparator is emit_tactic_configs.cpp --space=compare, and this runs it.

    WHAT MAKES THIS A CHECK RATHER THAN A RITUAL. It is asserted to FIRE, not merely to pass: the same source is
    compiled a second time against a header copy whose GroupedSpace demands eight warps instead of four, and a
    run that does not then report a disagreement fails this gate. A comparator that has only ever agreed cannot
    be distinguished from one that compares nothing -- and this repo has shipped that shape before (a selection
    test whose planted JSON never parsed, so the C++ read zero samples and "agreed").

    The divergence being reported is not itself a defect. The two operators may legitimately differ one day; the
    property this defends is that the difference is announced rather than absorbed into a table.
    """
    import subprocess, tempfile
    src = ROOT / "benchmarks" / "emit_tactic_configs.cpp"
    hdr = ROOT / "quactlize" / "include" / "ppu_tactic_space.hpp"
    if not src.is_file() or not hdr.is_file():
        return "FAIL", f"missing {src.name if not src.is_file() else hdr.name}", 0.0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        real = td / "emit_real"
        cc = ["c++", "-std=c++17", f"-I{ROOT/'quactlize'/'include'}", str(src), "-o"]
        r = subprocess.run(cc + [str(real)], capture_output=True, text=True)
        if r.returncode != 0:
            return "FAIL", f"emitter does not compile: {r.stderr.strip().splitlines()[-1][:120]}", 0.0
        args = ["4", "64", "--space=compare", "2", "3", "4", "6", "8", "12"]
        got = subprocess.run([str(real)] + args, capture_output=True, text=True)
        if got.returncode != 0:
            tail = [l for l in got.stdout.splitlines() if "disagreement" in l]
            return "FAIL", ("DenseSpace and GroupedSpace disagree: " + (tail[-1] if tail else "see --space=compare")
                            + " -- if intended, the emitter's consumers must be told which space they use"), 0.0

        # THE PLANTED DIVERGENCE. Patch only the SECOND wrapper (GroupedSpace); assert two were found first, so a
        # header refactor that collapses them cannot turn this into a silent no-op.
        text = hdr.read_text()
        old = "  static constexpr Exclusion kernel_exclusion(Candidate c) { return common_kernel_exclusion(c); }"
        i = text.find(old)
        j = text.find(old, i + 1) if i >= 0 else -1
        if j <= i:
            return "FAIL", ("could not find two identical kernel_exclusion wrappers to plant a divergence into; "
                            "the probe would be a no-op and the real run's PASS would prove nothing"), 0.0
        new = ("  static constexpr Exclusion kernel_exclusion(Candidate c) { "
               "if ((c.tm/c.wm)*(c.tn/c.wn) < 8) return Exclusion::PpuWarpGroupThreads; "
               "return common_kernel_exclusion(c); }")
        (td / "ppu_tactic_space.hpp").write_text(text[:j] + new + text[j + len(old):])
        probe = td / "emit_probe"
        r = subprocess.run(["c++", "-std=c++17", f"-I{td}", f"-I{ROOT/'quactlize'/'include'}", str(src),
                            "-o", str(probe)], capture_output=True, text=True)
        if r.returncode != 0:
            return "FAIL", f"planted-divergence build failed: {r.stderr.strip().splitlines()[-1][:120]}", 0.0
        bad = subprocess.run([str(probe)] + args, capture_output=True, text=True)
        if bad.returncode == 0:
            return "FAIL", ("the comparator reported NO disagreement against a GroupedSpace whose warp minimum "
                            "was raised to eight -- it is not comparing what it claims to"), 0.0
    n = next((l for l in bad.stdout.splitlines() if "disagreement" in l), "?")
    return "PASS", f"spaces agree, and the comparator fires when they do not ({n.strip()} when planted)", 0.0


def lint_inbox_delivered():
    """AN UNREAD INBOX ITEM IS AN UNDELIVERED ONE, and writing the file is not sending it.

    codex is REQUEST/RESPONSE: its process exists only for the duration of a call. The collaboration protocol's
    rule 1 -- reread INBOX at every checkpoint -- covers what is appended WHILE it runs. Anything appended after
    its turn ends sits in a file that nothing will open until the next call is made.

    On 2026-08-04 four items (035-038) were written and reported to the user as dispatched when only the file
    had changed. That is not a protocol gap, it is a category error about what the channel is: the channel is
    the call, and INBOX is the durable record the call refers to. This check makes the gap visible instead of
    relying on remembering, by comparing the highest INBOX number against STATUS's inbox-consumed.

    It cannot be a hard failure -- a gap is NORMAL for a few minutes while codex works -- so it reports the
    number of undelivered items and only fails when there is no call in flight to explain them. It has no way
    to see the call, so it states the gap and leaves the judgement visible rather than silently passing.
    """
    inbox = ROOT / ".coord" / "INBOX.md"
    status = ROOT / ".coord" / "STATUS.md"
    if not inbox.is_file() or not status.is_file():
        return "SKIP", "no .coord/{INBOX,STATUS}.md", 0.0
    nums = [int(m.group(1)) for m in re.finditer(r"^##\s+(\d{3})\s", inbox.read_text(), re.M)]
    if not nums:
        return "SKIP", "no numbered INBOX items", 0.0
    m = re.search(r"inbox-consumed:\s*(\d+)", status.read_text())
    if not m:
        return "FAIL", "STATUS.md has no inbox-consumed line -- 'has it read this' becomes a guess again", 0.0
    top, seen = max(nums), int(m.group(1))
    if seen >= top:
        return "PASS", f"INBOX {top:03d} consumed through {seen:03d}: nothing undelivered", 0.0
    gap = [n for n in sorted(nums) if n > seen]
    return "FAIL", (f"{len(gap)} INBOX item(s) NOT consumed: {', '.join(f'{n:03d}' for n in gap)} "
                    f"(STATUS says {seen:03d}). If no codex call is in flight, these were written and not sent "
                    f"-- writing the file is not dispatching it."), 0.0


def lint_selection_agrees():
    """THE PRECONDITION FOR DELETING THE C++ SELECTION (docs/BENCH_DESIGN.md step 3).

    benchmarks/bench_select.hpp decides inside the bench; benchmarks/analyse.py decides outside it. Until they
    are shown to agree on data that exercises the boundary -- a candidate whose band just overlaps the leader's
    and whose MEDIAN is worse -- removing either one replaces an unverified procedure with another unverified
    one. Needs no device: both halves are host code.
    """
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rfE", str(ROOT / "tests" / "test_bench_selection.py")],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode == 0:
        return "PASS", "C++ and Python verdicts match on the planted boundary fixture", 0.0
    tail = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()][-3:]
    return "FAIL", " | ".join(tail), 0.0


def lint_gguf_coverage():
    """A GGUF TYPE THAT NOTHING HAS CLASSIFIED IS THE DEFECT -- not an unsupported one.

    formats.py can only report on the types it enumerates, so the nine IQ types were not "unsupported", they
    were invisible; the user found that by asking and nothing here would have. tools/coverage.py takes the
    universe from ggml.h, so this gate fires when upstream adds a format and we have not even decided it is out
    of scope. Deciding "no" is fine; not noticing is not.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("cov", ROOT / "tools" / "coverage.py")
    cov = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cov)
    if not Path(cov.DEFAULT_GGML).is_file():
        return "SKIP", f"no ggml.h at {cov.DEFAULT_GGML}", 0.0   # the authority is absent, so nothing is known
    rows, unknown = cov.classify()
    if unknown:
        return "FAIL", ("ggml.h has types nothing has classified: "
                        + ", ".join(f"{n}={v}" for n, v in unknown)
                        + " -- add them to tools/coverage.py FAMILY, even if the answer is out-of-scope"), 0.0
    return "PASS", f"all {len(rows)} ggml types classified; {sum(1 for r in rows if r[2] == 'kquant')} supported", 0.0


def lint_undefined_names():
    """Names used before they exist. THE ONLY CLASS OF PYFLAKES FINDING THIS ASSERTS ON, and deliberately.

    THE FAILURE THIS COMES FROM. tests/test_gguf_routes.py's merge-premise oracle used `sf` two lines above the
    assignment that creates it. Trivial, and it cost a full ppu001 round trip -- because that test SKIPS without a
    device, so its first execution anywhere was on the box. Device-only tests get zero local runs by construction,
    which makes them the one place where an ordering mistake survives to the slowest possible feedback loop.

    Scoped to undefined names because that class is never a style opinion: if pyflakes says a name is undefined,
    the code raises when it reaches that line. Unused imports and the rest are left alone -- a lint that also
    reports taste is one people learn to skim, and this needs to be read.

    pyflakes on the planted bug: "undefined name 'sf'". On the tree as it stands: nothing.
    """
    try:
        import pyflakes  # noqa: F401
    except ImportError:
        return "SKIP", "pyflakes not installed (pip install pyflakes) -- device-only tests get no other flow check", 0.0
    rc, log, dt = run([sys.executable, "-m", "pyflakes",
                       str(ROOT / "tests"), str(ROOT / "quactlize"), str(ROOT / "ci"), str(ROOT / "benchmarks")])
    del rc
    hits = [l for l in log.splitlines()
            if "undefined name" in l or "referenced before assignment" in l]
    if hits:
        return "FAIL", ("names used before they exist -- these RAISE when reached: "
                        + "; ".join(h.strip() for h in hits[:3])), dt
    return "PASS", "no undefined names", dt


def lint_stale_repo_path():
    """An absolute path naming a repo directory that is not this one.

    THE FAILURE THIS COMES FROM. The benchmark binaries printed their profile hint as `$BIN/test_gemv_perf ...`,
    and the single document that defined $BIN still read

        BIN=/sim/eec/shared/junfu.qx/<the-pre-rename-repo-dir>/build_ppu/...

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
                ("lint", "names used before they exist (device-only tests get no other flow check)", lint_undefined_names),
                ("lint", "every ggml.h quant type is classified, in scope or out", lint_gguf_coverage),
                ("lint", "the C++ and Python selection procedures agree on planted data", lint_selection_agrees),
                ("lint", "box-built sources stay in the PPU-portable subset", lint_ppu_portability),
                ("lint", "emitted bench flags are ones the bench parses", lint_fixture_flags),
                ("lint", "every INBOX item is consumed, or is explained by a call in flight", lint_inbox_delivered),
                ("lint", "dense and grouped tactic spaces agree, and the comparator fires", lint_tactic_spaces_agree),
                ("lint", "the ctypes config mirror matches its C header field for field", lint_config_abi_matches_header),
                ("lint", "no tactic choice can change the offline layout", lint_tactic_cannot_change_offline_layout),
    ("registry", "declarations vs source", None)])
    # MATCH THE KIND AS WELL AS THE NAME. `-k lint` matched NOTHING, because a lint's name is its description
    # ("duplicate unroll directives") and the word "lint" only appears in the kind. The run then printed
    # "0 checks ... 0/0 passed" and exited zero -- a green result that establishes nothing, which is the exact
    # failure shape this file exists to catch elsewhere. So: match either field, and refuse an empty selection.
    items = [i for i in items if a.k in i[1] or a.k in i[0]]
    if a.k and not items:
        print(f"-k {a.k!r} selected NO checks. Nothing ran; this is a failure, not a pass. --list shows the names.")
        return 2
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
        if kind == "registry":
            sys.path.insert(0, str(ROOT / "ci"))
            import registry
            probs = registry.check()
            return ("PASS" if not probs else "FAIL"), (probs[0] if probs else "0 problems"), 0.0
        # AN UNKNOWN KIND IS A FAILURE, NOT THE REGISTRY CHECK. This used to fall through to registry, so any
        # row registered under a new kind silently ran a DIFFERENT check and reported its result -- two gates
        # added on 2026-08-04 ("coverage", "select") passed for days' worth of runs without ever executing.
        # A dispatch whose default is "run something else" cannot be caught by reading the output: it is green,
        # it is fast, and it names the check that did not run.
        return "FAIL", f"unknown check kind {kind!r} -- register it in run_one or use an existing kind", 0.0

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
