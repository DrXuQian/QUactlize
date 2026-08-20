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
  boxdry   build.sh itself, against a stub PPU SDK, through generated device objects and a real host link. The only
           check that goes through actlize's example registration -- everything else verifies a piece in isolation.
  asan     the host preprocessing chain compiled with -fsanitize=address and swept over shapes. It found two
           out-of-bounds accesses that no assertion could have located: both corrupted the heap silently and
           surfaced as an intermittent Bus error or SIGSEGV in an unrelated test, several tests later.
  pytest   the torch-op tests, if the extension is built and torch is importable. Skipped, not failed, otherwise --
           this tier must stay runnable on a machine with neither.

  ./ci/local_gates.py            run everything
  ./ci/local_gates.py --list     show what would run
  ./ci/local_gates.py -k q4k     run only matching names
"""
import argparse, json, os, re, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEV = ROOT / "dev" / "fold_derivation"
STUB = DEV / "stub_inc"
ACT = ROOT / "third_party" / "actlize" / "include"
ACT_UTIL = ROOT / "third_party" / "actlize" / "tools" / "util" / "include"
OUT = Path(os.environ.get("QUACTLIZE_CI_OUT", "/tmp/quactlize_ci"))


_NVCC_DEVICE_PROBE = None


def nvcc_can_compile_device_cuda():
    """(ok, why) -- can this machine's `nvcc` actually compile NVIDIA device code?

    THE NAME IS NOT THE ANSWER. On the box `which nvcc` is NVIDIA's own driver (PPU_SDK/CUDA_SDK/bin/nvcc,
    "NVIDIA (R) Cuda compiler driver"), but it hands DEVICE preprocessing to ppu_clang++. So `nvcc --version`
    says yes while `threadIdx` is undeclared and cutlass/float8.h's `#include <hggc_fp8.h>` fires. Checking the
    version string would have produced exactly the false green this probe exists to stop.

    So: compile something that needs a device compiler, and let the compiler answer. Cached; ~1 s once.

    AND THE PROBE MUST NOT BE WEAKER THAN THE PRECONDITION. Version one compiled three lines of bare device
    code. Run on ppu001 it answered CAPABLE -- correctly, that much does compile there -- and all five guarded
    rows still went red, with the same misleading reason the probe was written to remove. The break is one
    layer up: `#include <cuda_fp16.h>` pulls `crt/device_functions.h`, and the PPU SDK ships a MODIFIED copy
    that calls `__assert` (stock CUDA 12.8 has no such symbol anywhere in that header) which nothing on our
    command line declares. So the probe includes cuda_fp16.h and uses a __half intrinsic: that is what every
    CUTLASS translation unit needs, and therefore the capability actually worth asking about.

    WHY IT MATTERS. The `gate` tier's oracles (l120/l121/l122, ...) are HOST-ONLY by content -- zero __global__,
    zero __device__, zero launches -- but they include the CUTLASS/CuTe stack, whose device bodies only survive
    a device compile. On 2026-08-11 that made l121/l122 fail on the box while passing on a dev container, and
    the failure read as "the grouped Stream-K contract is broken" rather than "this machine cannot run this
    check". A SKIP naming the reason is the honest verdict; --strict turns it into a failure wherever a green
    tier is being claimed as evidence.
    """
    global _NVCC_DEVICE_PROBE
    if _NVCC_DEVICE_PROBE is None:
        src = OUT / "nvcc_device_probe.cu"
        OUT.mkdir(parents=True, exist_ok=True)
        src.write_text("#include <cuda_fp16.h>\n"
                       "__global__ void k(__half* p){ *p = __hadd(p[threadIdx.x], p[blockIdx.x]); }\n"
                       "int main(){ return 0; }\n")
        rc, log, _ = run(NVCC + ["-o", str(OUT / "nvcc_device_probe"), str(src)])
        if rc == 0:
            _NVCC_DEVICE_PROBE = (True, "")
        else:
            first = next((l.strip() for l in log.splitlines() if ": error:" in l or ": fatal error:" in l),
                         (log.strip().splitlines() or ["nvcc failed"])[-1])
            _NVCC_DEVICE_PROBE = (False, first[:160])
    return _NVCC_DEVICE_PROBE


def _sdk_target_includes() -> list:
    """The PPU SDK's own target include dir, WHEN THERE IS ONE. Empty off the box, which is the point.

    WHY THIS EXISTS, and it is a case where the local tier was green on the wrong compiler. actlize's
    cutlass/float8.h:88 includes <hggc_fp8.h> under `#ifdef PPU_FP8_ENABLED`, and :55 defines that macro from
    `__HGGCCC_VER_MAJOR__ >= 12`. NVIDIA nvcc does not define __HGGCCC_VER_MAJOR__, so on a dev container that
    include is NEVER EXPANDED and the stub set never needed an fp8 header. On the box the same `nvcc` (real
    NVIDIA nvcc, under PPU_SDK/CUDA_SDK/bin) hands device preprocessing to ppu_clang++, which DOES define it --
    and that invocation's include path does not carry the SDK's targets/<triple>/include, where hggc_fp8.h
    actually lives. Result on 2026-08-11: l121 and l122 failed to build on the box with
    `fatal error: hggc_fp8.h: No such file or directory` while passing here.

    NOT a stub. Putting a fake hggc_fp8.h in stub_inc would SHADOW the real header on the box -- stub_inc is
    first on the -I list -- so the gate would compile against an invented fp8 type while every device build uses
    the SDK's. This appends the real directory, LAST, so nothing already on the path is displaced.

    The SDK root follows build.sh's precedence exactly (PPU_SDK, then PPU_HOME, then PPU_SDK_SITE_DEFAULT) so
    the two cannot disagree about which SDK is in play.
    """
    root = (os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME")
            or os.environ.get("PPU_SDK_SITE_DEFAULT") or "")
    if not root:
        return []
    out = []
    for d in sorted(Path(root).glob("targets/*/include")):
        if d.is_dir():
            out += ["-I", str(d)]
    return out

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
    ("l107_moe_router_fixture", []),
    ("l108_rt_error_contract", []),
    ("l109_rt_hggc_parse", []),
    ("l110_unit_pack_abi", []),
    ("l112_mixed_policy_parity", []),
    ("l113_mixed_metadata_policy", []),
    ("l114_scale_copy_coverage@host", []),
    ("l114_scale_copy_coverage@ordinary", []),
    ("l114_scale_copy_coverage@fold", []),
    ("l114_scale_copy_coverage@two_plane", []),
    ("l120_streamk_min_iters_policy", []),
    ("l121_grouped_streamk_wrapper", []),
    ("l122_streamk_fixup_cohort", []),
    ("l146_q4k_pdf_ab_fixture", []),
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
    ("tests/test_moe_grouped_streamk.cu", ""),
    # L>1 is the unique coverage here: each expert gets different low/high weights and metadata, then the grouped
    # result is checked against per-expert L=1 launches.  Both optional collective headers must therefore instantiate,
    # and PPU_B_CHUNK changes the converter pipeline rather than merely changing host code.
    ("tests/test_lowbit_grouped.cu", ""),
    ("tests/test_lowbit_grouped.cu", "-DPPU_B_CHUNK=1"),
    # These registered targets all had baseline files but no SYNTAX row.  Keep the boundaries that still provide
    # distinct evidence: all five placed k-quant formats, the standalone GEMV converter/kernel matrix and perf main,
    # and the native-scale device decoder.  test_fpA_intB_ppu and test_moe_grouped_ppu are superseded perf-only
    # fossils (no oracle, obsolete timing/traffic); an empty baseline must not promote either into evidence.
    ("tests/test_fpA_kquant_dense.cu", ""),
    ("tests/test_gemv_lowbit.cu", ""),
    ("benchmarks/test_gemv_perf.cu", ""),
    ("tests/test_q4k_native_scale.cu", ""),
    # test_ppu_f16x2_probe.cu intentionally has no local SYNTAX row: its alias arm deliberately exposes raw PPU
    # f16x2 instructions, so nvcc reaches ptxas, rejects `ppu.fma/sub` as unknown and exits 255 without this gate's
    # completion witness.  The source remains a box gate; unlike the old empty baseline, its absence here does not
    # pretend that nvcc checked it.
    # #112's collective gate is a new device path, not merely a host harness:
    # compile the raw-mainloop G3 arm and both TM8/TM16 production G4 arms
    # locally so a dependent-template error cannot wait for the ppu001 box.
    ("tests/test_ppu_m8n16_collective.cu", ""),
    # INBOX 097 changes scale-copy ownership, so its two independent-golden box targets must at least instantiate
    # locally. Q65 once omitted the optional two-plane specialization include and otherwise reached the box with
    # CollectiveMma's failing primary template; keeping the real source here makes that omission non-repeatable.
    ("tests/test_q3_bconcat_real.cu", ""),
    ("tests/test_q65_bconcat_real.cu", ""),
    ("benchmarks/test_moe_splitk_bench.cu", ""),
    # THE BENCH THE WHOLE SWEEP RUNS THROUGH, and it had no local gate at all until 2026-08-04 -- which is how
    # a selection rewrite (median over interleaved repeats, tie reporting) got committed without ever being
    # compiled. It is also the file with the most host-side logic of any bench, so it is the one most likely to
    # break in a way nvcc catches and review does not.
    ("benchmarks/test_lowbit_moe_bench.cu", ""),
    # ScaleOnly selects a different compiled traffic-model branch and halves its metadata planes. Compile that mode
    # explicitly; the default ScaleZero row cannot validate the ScaleOnly macro path.
    ("benchmarks/test_lowbit_moe_bench.cu", "-DLOWBIT_QMODE=1"),
    # MOE_STAGES turns six preprocessor arms into a selected subset. The CMake/box gates prove the flags arrive;
    # this row proves a multi-value selection still parses the real bench header.
    ("benchmarks/test_lowbit_moe_bench.cu", "-DMOE_STAGES_2 -DMOE_STAGES_12"),
    # The DENSE bench, likewise ungated until 2026-08-04. It now carries a generated config table and a
    # static_assert tying that table to the binary's (bits, TileK); this row is what makes a stale table fail
    # here instead of producing a sweep over tactics the binary cannot select.
    # Fixed Split-K is an independent target and instantiates the M==1 packed-A
    # provider plus its FP32 partial producer/reducer.  The ordinary dense rows
    # below cannot make a dependent error in that new kernel body visible.
    ("benchmarks/test_lowbit_dense_splitk_parallel.cu", ""),
    ("benchmarks/test_lowbit_dense_bench.cu", ""),
    ("benchmarks/test_lowbit_dense_bench.cu", "-DBENCH_UINT2"),
    ("benchmarks/test_lowbit_dense_bench.cu", "-DBENCH_UINT1"),
    ("benchmarks/test_lowbit_dense_bench.cu", "-DPPU_B_CHUNK=1"),
    # 107a's dedicated main has a one-row registry; compile that preprocessor identity separately from the full
    # tactic table. The unit row below is what instantiates the named persistent kernel and its scheduler loop.
    ("benchmarks/test_lowbit_dense_bench.cu",
     "-DDENSE_PERSISTENT_AB=1 -DDENSE_AB_BITS=4 -DDENSE_AB_TM=64 -DDENSE_AB_TN=64 "
     "-DDENSE_AB_TK=64 -DDENSE_AB_WM=64 -DDENSE_AB_WN=32 -DDENSE_AB_ST=3 -DDENSE_AB_BC=0 -DBENCH_GS=32"),
    # 107b is deliberately a different target from 107a.  Its named kernel
    # instantiates fixup(), whose barrier cohort is fixed at 128 threads, and
    # its host path carries the decomposition/witness/per-launch lock reset.
    ("benchmarks/test_lowbit_dense_bench.cu",
     "-DDENSE_STREAMK_AB=1 -DDENSE_AB_BITS=4 -DDENSE_AB_TM=64 -DDENSE_AB_TN=128 "
     "-DDENSE_AB_TK=64 -DDENSE_AB_WM=64 -DDENSE_AB_WN=32 -DDENSE_AB_ST=2 "
     "-DDENSE_AB_BC=0 -DBENCH_GS=128 -DTILE_M=64 -DTILE_N=128 -DWARP_M=64 "
     "-DWARP_N=32 -DSTAGES=2"),
    # Independent Marlin scheduler target: artifact TK64 and tactic TK128 are
    # deliberately different, and the same TU owns DP/Stream-K/Marlin arms.
    ("benchmarks/test_lowbit_dense_bench.cu",
     "-DDENSE_MARLIN_AB=1 -DDENSE_STREAMK_AB=1 -DBENCH_GS=128 -DBENCH_TSK=64 "
     "-DDENSE_AB_BITS=4 -DDENSE_AB_ARTIFACT_TK=64 -DDENSE_AB_TM=16 -DDENSE_AB_TN=128 "
     "-DDENSE_AB_TK=128 -DDENSE_AB_WM=16 -DDENSE_AB_WN=32 -DDENSE_AB_ST=3 "
     "-DDENSE_AB_BC=0 -DTILE_M=16 -DTILE_N=128 -DWARP_M=16 -DWARP_N=32 -DSTAGES=3"),
    # Classic-aligned 2N x 4K has 256 compute threads but only a 64-thread K0
    # output cohort.  Compile the real host verifier under that exact identity;
    # the historical WK1 row above cannot expose a return to CTA-wide owners.
    ("benchmarks/test_lowbit_dense_bench.cu",
     "-DDENSE_MARLIN_WK4_AB=1 -DDENSE_MARLIN_AB=1 -DDENSE_STREAMK_AB=1 "
     "-DBENCH_GS=128 -DBENCH_TSK=64 -DDENSE_AB_BITS=4 -DDENSE_AB_ARTIFACT_TK=64 "
     "-DDENSE_AB_TM=16 -DDENSE_AB_TN=128 -DDENSE_AB_TK=128 -DDENSE_AB_WM=16 "
     "-DDENSE_AB_WN=64 -DDENSE_AB_WARP_K=32 -DDENSE_AB_ST=4 -DDENSE_AB_BC=0 "
     "-DTILE_M=16 -DTILE_N=128 -DWARP_M=16 -DWARP_N=64 -DSTAGES=4"),
    # Independent multirow standalone-Marlin registry: unlike the fixed A/B
    # target, this must parse the eight-field generated table and search it.
    ("benchmarks/test_lowbit_dense_bench.cu",
     "-DDENSE_MARLIN_STANDALONE_SWEEP=1 -DDENSE_MARLIN_AB=1 -DDENSE_STREAMK_AB=1 "
     "-DBENCH_GS=128 -DBENCH_TSK=64 -DDENSE_AB_BITS=4 -DDENSE_AB_ARTIFACT_TK=64 "
     "-DDENSE_AB_TM=16 -DDENSE_AB_TN=128 -DDENSE_AB_TK=128 -DDENSE_AB_WM=16 "
     "-DDENSE_AB_WN=64 -DDENSE_AB_WARP_K=32 -DDENSE_AB_ST=4 -DDENSE_AB_BC=0 "
     "-DTILE_M=16 -DTILE_N=128 -DWARP_M=16 -DWARP_N=64 -DSTAGES=4"),
    # Main mode only declares generated wrappers. This is one real unit-mode row, so shared tag/metric plumbing in
    # lowbit_dense_unit.inc is instantiated locally instead of waiting for hgcc on the box.
    ("dev/fold_derivation/test_lowbit_dense_unit.cu", ""),
    ("dev/fold_derivation/test_lowbit_dense_unit.cu", "-DPPU_B_CHUNK=1"),
    ("dev/fold_derivation/test_lowbit_dense_unit.cu", "-DDENSE_PERSISTENT_AB=1 -DBENCH_GS=32"),
    ("dev/fold_derivation/test_lowbit_dense_unit.cu",
     "-DDENSE_STREAMK_AB=1 -DBENCH_GS=128 -DTILE_M=64 -DTILE_N=128 "
     "-DWARP_M=64 -DWARP_N=32 -DSTAGES=2"),
    ("dev/fold_derivation/test_lowbit_dense_unit.cu",
     "-DDENSE_MARLIN_AB=1 -DDENSE_STREAMK_AB=1 -DBENCH_GS=128 -DBENCH_TSK=64 "
     "-DDENSE_AB_BITS=4 -DDENSE_AB_ARTIFACT_TK=64 -DDENSE_AB_TM=16 -DDENSE_AB_TN=128 "
     "-DDENSE_AB_TK=128 -DDENSE_AB_WM=16 -DDENSE_AB_WN=32 -DDENSE_AB_ST=3 "
     "-DDENSE_AB_BC=0 -DTILE_M=16 -DTILE_N=128 -DWARP_M=16 -DWARP_N=32 -DSTAGES=3"),
    # The full-table executable has a distinct generated-unit preprocessor arm:
    # every row is unconditionally Marlin, with no DP/runtime-switch fallback.
    # Compile one real table row so that arm is not merely source-linted.
    ("dev/fold_derivation/test_lowbit_dense_unit.cu",
     "-DDENSE_MARLIN_SWEEP=1 -DBENCH_GS=128 -DBENCH_TSK=64 "
     "-DTILE_M=16 -DTILE_N=128 -DWARP_M=16 -DWARP_N=32 -DSTAGES=3"),
    ("dev/fold_derivation/test_lowbit_dense_unit.cu",
     "-DDENSE_MARLIN_STANDALONE_SWEEP=1 -DDENSE_MARLIN_AB=1 -DDENSE_STREAMK_AB=1 "
     "-DBENCH_GS=128 -DBENCH_TSK=64 -DDENSE_AB_BITS=4 -DDENSE_AB_ARTIFACT_TK=64 "
     "-DDENSE_AB_TM=16 -DDENSE_AB_TN=128 -DDENSE_AB_TK=128 -DDENSE_AB_WM=16 "
     "-DDENSE_AB_WN=64 -DDENSE_AB_WARP_K=32 -DDENSE_AB_ST=4 -DDENSE_AB_BC=0 "
     "-DTILE_M=16 -DTILE_N=128 -DWARP_M=16 -DWARP_N=64 -DSTAGES=4"),
    # One real committed row from every CTA-warp cohort released by A2.  The
    # ordinary DP and Marlin arms must both compile; run_l131 also requires an
    # intentionally inexact explicit cohort to fail at the authored binding.
    ("dev/fold_derivation/l131_marlin_rejected_cohorts.cu", ""),
    # THE SHIPPING .so BOUNDARY. The benches compiled the grouped collective for years while the product wrapper
    # did not expose it; compiling this translation unit is what covers the six-entry ABI and every qtype dispatch.
    ("quactlize/csrc/device/ppu_dense_backend.cu", ""),
    # True folded-reader control: Q3 ArtifactTileK=64 is F_low/F_high=2/4 beneath a TK256 tensor tactic.  This
    # flag-on row proves the arrangement-aware ABI instantiates the packed two-plane collective rather than merely
    # accepting the descriptor in host arithmetic.  Single-plane F>1 remains explicitly fail-closed (l138).
    ("quactlize/csrc/device/ppu_dense_backend.cu",
     "-DPPU_PACKED_SCALE=1 -DPPU_PACKED_FORMAT=3 -DQUACTLIZE_DENSE_ONLY=11"),
    ("benchmarks/test_moe_splitk_bench.cu", "-DPPU_PACKED_SCALE=1"),
    # dev/'s top-level probes. They are DEVICE probes -- swzl_ldmatrix_probe reads the hardware swizzle, the
    # ablations and sweeps run on the accelerator -- so build.sh overlays them onto the box, and anything that
    # reaches the box belongs in the tier whose whole purpose is catching box-only compile failures locally.
    # They were absent from this list while they were also absent from the overlay, and the two facts hid each
    # other: cmake failed on a missing source before any of them could fail to compile.
    ("dev/swzl_ldmatrix_probe.cu", ""),
    ("dev/test_fold_int2.cu", ""),
    ("dev/test_fold_int2.cu", "-DFOLD_ARTIFACT_TILEK=64"),
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


# EXTRA TRANSLATION UNITS a probe must LINK, not merely include. A probe that calls an exported op needs its
# definition; without this l110_unit_pack_abi built and then failed at link, so the registered row reported BUILD
# while the check itself was correct -- a gate red for a reason that has nothing to do with what it tests, which
# is the fastest way to teach people to ignore it. Sources, not flags, because the missing thing is a definition.
GATE_SRCS = {
    "l110_unit_pack_abi": ["quactlize/csrc/device/ppu_dense_layout.cu"],
}

GATE_FLAGS = {"l95_stub_vs_real": ["-D__HGGCCC__", "--expt-relaxed-constexpr"],
              "l112_mixed_policy_parity": ["-D__HGGCCC__", "--expt-relaxed-constexpr"],
              "l113_mixed_metadata_policy": ["-D__HGGCCC__", "--expt-relaxed-constexpr"],
              "l114_scale_copy_coverage@ordinary":
                  ["-D__HGGCCC__", "--expt-relaxed-constexpr", "-DL114_PROVIDER=1"],
              "l114_scale_copy_coverage@fold":
                  ["-D__HGGCCC__", "--expt-relaxed-constexpr", "-DL114_PROVIDER=2"],
              "l114_scale_copy_coverage@two_plane":
                  ["-D__HGGCCC__", "--expt-relaxed-constexpr", "-DL114_PROVIDER=3"],
              "l109_rt_hggc_parse": ["-D__HGGCCC__"],
              "l120_streamk_min_iters_policy": ["-D__HGGCCC__", "--expt-relaxed-constexpr"],
              "l121_grouped_streamk_wrapper": ["-D__HGGCCC__", "--expt-relaxed-constexpr"],
              "l122_streamk_fixup_cohort": ["-D__HGGCCC__", "--expt-relaxed-constexpr"],
              # THE MACROS ARE THE POINT. This gate asserts the fused path is ON, so it has to be built the way the
              # box builds packfuse -- without these two it would assert about a configuration nobody runs and pass
              # for the wrong reason, which is the failure it exists to catch.
              **{f"l103_packed_format_active@fmt{f}":
                     ["-D__HGGCCC__", "--expt-relaxed-constexpr",
                      "-DPPU_PACKED_SCALE=1", f"-DPPU_PACKED_FORMAT={f}"] for f in (0, 1, 2, 3, 4)},
              "l100_fused_active": ["-D__HGGCCC__", "--expt-relaxed-constexpr",
                                    "-DPPU_PACKED_SCALE=1", "-DPPU_PACKED_SCALE_FUSED=1"]}


def lint_mixed_policy_parity_fires():
    """The descriptor equality witness must reject an operator-local mainloop change."""
    src = DEV / "l112_mixed_policy_parity.cu"
    if not src.is_file():
        return "FAIL", f"missing {src.name}", 0.0
    OUT.mkdir(parents=True, exist_ok=True)
    planted = OUT / "l112_mixed_policy_parity_planted"
    rc, log, dt = run(NVCC + ["-D__HGGCCC__", "--expt-relaxed-constexpr",
                              "-DPPU_PLANT_MIXED_POLICY_DRIFT=1",
                              "-I", str(STUB), "-I", str(ACT), "-I", str(ACT_UTIL),
                              "-I", str(ROOT / "quactlize/include"),
                              "-I", str(ROOT / "tests"), "-I", str(ROOT / "benchmarks"),
                              "-o", str(planted), str(src)])
    expected = "dense/grouped mixed policy descriptors diverged"
    if rc == 0:
        return "FAIL", "mixed-policy parity accepted a planted grouped-only B-layout change", dt
    if expected not in log:
        first = next((line for line in log.splitlines() if "error:" in line), "no compiler diagnostic")
        return "FAIL", f"planted build failed for the wrong reason: {first[:140]}", dt
    return "PASS", "descriptor parity rejects a planted grouped-only B-layout change", dt


def lint_scale_copy_coverage_fires():
    """The shared witness must reject the exact uncapped layout that truncated Q3/Q5 scale loads."""
    src = DEV / "l114_scale_copy_coverage.cu"
    if not src.is_file():
        return "FAIL", f"missing {src.name}", 0.0
    OUT.mkdir(parents=True, exist_ok=True)
    planted = OUT / "l114_scale_copy_coverage_uncapped"
    rc, log, dt = run(NVCC + ["-DL114_PROVIDER=0", "-DL114_PLANT_UNCAPPED_SCALE_COPY=1",
                              "-I", str(STUB), "-I", str(ACT), "-I", str(ACT_UTIL),
                              "-I", str(ROOT / "quactlize/include"),
                              "-I", str(ROOT / "tests"), "-I", str(ROOT / "benchmarks"),
                              "-o", str(planted), str(src)])
    expected = "scale copy asks for more thread slots than the CTA has"
    if rc == 0:
        return "FAIL", "ScaleCopyCoverage accepted the planted uncapped 128-slot layout for a 64-thread CTA", dt
    if expected not in log:
        first = next((line for line in log.splitlines() if "error:" in line), "no compiler diagnostic")
        return "FAIL", f"uncapped scale-copy build failed for the wrong reason: {first[:140]}", dt
    return "PASS", "shared witness rejects the old uncapped Q3/Q5 scale-copy layout", dt


def lint_fold_metadata_single_owner():
    """The folded shipping body must use ScaleCopyPlan's exact owner set."""
    path = (ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/"
            "ppu_mma_aiu_fold.hpp")
    if not path.is_file():
        return "FAIL", f"missing {path}", 0.0

    def violations(text):
        required = [
            (r"ScaleCopyPlan::owns_physical_thread\s*\(\s*thread_idx\s*\)", 1,
             "one physical-owner decision"),
            (r"ScaleCopyPlan::logical_slot\s*\(\s*thread_idx\s*\)", 1,
             "one logical-slot mapping"),
            (r"copy_async_extra_info\s*\([^;]*metadata_copy_owner\s*\)\s*;", 2,
             "owner flag at preload and steady-state async issue"),
            (r"if\s*\(\s*!metadata_copy_owner\s*\)\s*return\s*;", 1,
             "non-owner async early return"),
            (r"if\s*\(\s*metadata_copy_owner\s*\)\s*clear\s*\(\s*tSsS\s*\)\s*;", 1,
             "owner-only scale clear"),
            (r"if\s*\(\s*metadata_copy_owner\s*\)\s*clear\s*\(\s*tZsZ\s*\)\s*;", 1,
             "owner-only zero clear"),
        ]
        hits = []
        for pattern, expected, label in required:
            actual = len(re.findall(pattern, text, re.S))
            if actual != expected:
                hits.append(f"{label}: {actual} != {expected}")
        if re.search(r"thread_idx\s*%\s*\(\s*Scale_GmemCopyThrLayoutH", text):
            hits.append("legacy modulo-replayed publisher is live")
        return hits

    source = path.read_text()
    hits = violations(source)
    if hits:
        return "FAIL", "; ".join(hits[:4]), 0.0

    planted = source.replace(
        "ScaleCopyPlan::owns_physical_thread(thread_idx)", "true", 1)
    if not violations(planted):
        return "FAIL", "single-owner guard accepted a planted all-thread publisher", 0.0
    return ("PASS",
            "fold clear and both async issue points use one proved owner; all-thread plant red",
            0.0)


def lint_device_probe_scope():
    """A device-compiler SKIP may only guard a check that actually invokes a device compiler."""
    # THE GUARD IS ITSELF A WAY TO LOSE A CHECK. `nvcc_can_compile_device_cuda()` exists so an environment that
    # cannot compile device code says so instead of blaming the source. Applied one function too wide it does the
    # opposite: it silently skips a check that would have run and passed. That happened the hour it was written --
    # lint_dense_streamk_contract runs ci/check_dense_streamk_contract.py, which starts no subprocess at all, and
    # the guard turned it into a SKIP on every box.
    #
    # So the rule is a property, not a list: a function that returns SKIP on the probe must reach a compiler.
    # Resolved one level through _run_ci_script and through an explicitly named shell runner.  L124 is the latter:
    # the Python gate launches run_l124_fp32_residue_mask.sh, and that script invokes nvcc.  Ignoring that one
    # seam misclassifies a real compiler gate as a source-only check, just as ignoring _run_ci_script once did.
    # ast.parse rather than import -- a stale .pyc can serve bytecode the disk no longer has (see the memory on
    # verification failure shapes), and this check exists precisely to be trusted about the file's current text.
    import ast
    src = (ROOT / "ci" / "local_gates.py").read_text()
    tree = ast.parse(src)
    bad = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        body = ast.get_source_segment(src, fn) or ""
        if "nvcc_can_compile_device_cuda()" not in body or fn.name == "nvcc_can_compile_device_cuda":
            continue
        reaches = "NVCC" in body
        for script in re.findall(r'_run_ci_script\(\s*"([^"]+)"', body):
            f = ROOT / "ci" / script
            if f.exists() and re.search(r'"nvcc"|\bnvcc\b', f.read_text()):
                reaches = True
        for script in re.findall(r'["\']([^"\']+\.sh)["\']', body):
            candidates = (DEV / script, ROOT / script)
            f = next((candidate for candidate in candidates if candidate.is_file()), None)
            shell_code = "\n".join(
                line for line in f.read_text().splitlines()
                if not line.lstrip().startswith("#")
            ) if f is not None else ""
            if re.search(r'\bnvcc\b', shell_code):
                reaches = True
        if not reaches:
            bad.append(fn.name)
    if bad:
        return "FAIL", ("these guard on the device-compiler probe but never reach a compiler, so the guard can only "
                        "skip a check that would have run: " + ", ".join(bad)), 0.0
    return "PASS", "every device-compiler SKIP guards a check that reaches a compiler", 0.0

def lint_streamk_min_zero_fires():
    """The default-compatible Stream-K policy seam must reject a zero-sized stripe at its type boundary."""
    ok, why = nvcc_can_compile_device_cuda()
    if not ok:
        return "SKIP", ("needs an NVIDIA device compiler for the CUTLASS stack its oracle includes: "
                        + why), 0.0
    src = DEV / "l120_streamk_min_iters_policy.cu"
    if not src.is_file():
        return "FAIL", f"missing {src.name}", 0.0
    OUT.mkdir(parents=True, exist_ok=True)
    planted = OUT / "l120_streamk_min_zero.o"
    rc, log, dt = run(NVCC + ["-D__HGGCCC__", "--expt-relaxed-constexpr",
                              "-DL120_SELECTED_MIN=0", "-I", str(STUB), "-I", str(ACT),
                              "-I", str(ACT_UTIL), "-I", str(ROOT / "quactlize/include"),
                              "-I", str(ROOT / "benchmarks"),
                              "-c", "-o", str(planted), str(src)])
    expected = "Stream-K requires at least one K tile per work unit"
    if rc == 0:
        return "FAIL", "ParamsT<0> compiled, so the minimum-stripe type admits an empty work unit", dt
    if expected not in log:
        first = next((line for line in log.splitlines() if "error:" in line), "no compiler diagnostic")
        return "FAIL", f"ParamsT<0> failed for the wrong reason: {first[:140]}", dt
    return "PASS", "ParamsT<0> is rejected at the policy type boundary", dt


def lint_mixed_pipeline_shared():
    """Every shipping mixed collective must delegate its stage ring to the one shared driver.

    The descriptor witness makes dense/grouped policy drift a compile error, but a collective could otherwise keep
    that witness while copying the cadence back into its operator(). Refuse both ways to bypass the seam: omitting
    the driver call, or reintroducing the stage counters/waits in a provider body.
    """
    rels = [
        "quactlize_mma_mixed_input.hpp",
        "ppu_mma_aiu_fold.hpp",
        "ppu_mma_aiu_mixed_input_2plane.hpp",
    ]
    # THE COLLECTIVES LEFT actlize ON 2026-08-06. This used to read them out of the submodule, where two of the
    # three no longer exist and the third is upstream's, which has no shared driver at all. It failed loudly
    # rather than passing, but a gate pointed at the wrong tree tests nothing either way.
    base = ROOT / "quactlize" / "include" / "quactlize_extensions" / "cutlass" / "gemm" / "collective"
    sources = {}
    for rel in rels:
        path = base / rel
        if not path.is_file():
            return "FAIL", f"missing mixed collective {rel}", 0.0
        sources[rel] = path.read_text()

    def violations(texts):
        hits = []
        for rel, text in texts.items():
            if text.count('#include "quactlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_pipeline.hpp"') != 1:
                hits.append(f"{rel}: shared-driver include count != 1")
            if text.count("using PipelineDriver = detail::MixedPipelineDriver;") != 1:
                hits.append(f"{rel}: pipeline type witness count != 1")
            if text.count("detail::run_mixed_pipeline<") != 1:
                hits.append(f"{rel}: shared-driver call count != 1")
            for local in ("smem_pipe_read", "smem_pipe_write"):
                if local in text:
                    hits.append(f"{rel}: reintroduced local stage state {local}")
            if "cp_async_wait<" in text:
                hits.append(f"{rel}: reintroduced local pipeline wait")
            if re.search(r"for\s*\(\s*;\s*k_tile_count\s*>", text):
                hits.append(f"{rel}: reintroduced local K-tile pipeline loop")
        return hits

    hits = violations(sources)
    if hits:
        return "FAIL", "; ".join(hits[:4]), 0.0

    # Prove the gate observes an actual bypass rather than merely seeing the three files. This plant exists only in
    # memory: delete one delegation call and require the same inspection above to reject it.
    planted = dict(sources)
    victim = rels[0]
    planted[victim] = planted[victim].replace("detail::run_mixed_pipeline<", "detail::local_mixed_pipeline<", 1)
    if not violations(planted):
        return "FAIL", "shared-pipeline guard accepted a planted collective-local bypass", 0.0
    return "PASS", "three collectives use one stage-ring driver; planted local bypass rejected", 0.0


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
    # The file moved out of actlize on 2026-08-06; the exemption follows it, or its one box-proven arm reads as
    # a new violation.
    proven_exemptions = {"quactlize/include/quactlize_extensions/cutlass/gguf_packed_scale.h"}
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
    ok, why = nvcc_can_compile_device_cuda()
    if not ok:
        return "SKIP", (f"this nvcc cannot compile NVIDIA device code, so the CUTLASS stack these host-only "
                        f"oracles include will not build: {why}. Run this tier where nvcc is a full CUDA "
                        f"toolchain; --strict makes this a failure."), 0.0
    OUT.mkdir(parents=True, exist_ok=True)
    exe = OUT / name.replace("@", "_")
    rc, log, dt = run(NVCC + GATE_FLAGS.get(name, []) +
                      ["-I", str(STUB), "-I", str(ACT), "-I", str(ACT_UTIL),
                       "-I", str(ROOT / "quactlize/include"),
                       "-I", str(ROOT / "tests"), "-I", str(ROOT / "benchmarks")] +
                      ["-o", str(exe), str(src)] +
                      [str(ROOT / s) for s in GATE_SRCS.get(name, GATE_SRCS.get(base, []))])
    if rc != 0:
        # THE FIRST LINE OF AN nvcc FAILURE CARRIES NO INFORMATION. It is the head of the
        # "In file included from a.h:1,\n from b.h:2,\n ... : error: ..." cascade, so this
        # used to report `In file included from .../cutlass/half.h:68,` and nothing else --
        # a build failure whose message names an #include. Observed 2026-08-11 on the box for
        # l121/l122, where it cost a round trip to learn that the gate had truncated the error
        # rather than that the compiler had produced one.
        #
        # The ASAN gate in this same file already did the right thing (`next(l for l in log if
        # " error" in l)`); gate() simply did not follow it. Report the first real diagnostic,
        # say how many there were, and keep the last include-chain line for context because
        # that is where the instantiation came from.
        errs = [l for l in log.splitlines() if ": error:" in l or ": fatal error:" in l]
        chain = [l for l in log.splitlines() if l.lstrip().startswith(("In file included from", "from "))]
        if errs:
            where = f"  [via {chain[-1].strip()}]" if chain else ""
            more = f"  (+{len(errs) - 1} more)" if len(errs) > 1 else ""
            return "BUILD", errs[0].strip() + more + where, dt
        return "BUILD", (log.strip().splitlines()[-1] if log.strip() else "nvcc failed"), dt
    rc, log, dt2 = run([str(exe)] + [str(ROOT / a) for a in args])
    tail = [l for l in log.splitlines() if l.strip()]
    return ("PASS" if rc == 0 else "FAIL"), (tail[-1] if tail else ""), dt + dt2


def syntax(src, defs):
    sc = DEV / "syntax_check.sh"
    if not sc.exists():
        return "MISSING", "syntax_check.sh not found", 0.0
    env = dict(os.environ, EXTRA_DEFS=defs, GEN_INC=str(DEV / "gen_stub"))
    rc, log, dt = run(["bash", str(sc), str(ROOT / src)], cwd=str(DEV), env=env)
    lines = [l for l in log.splitlines() if l.strip()]
    # syntax_check.sh SEPARATES "cannot run" FROM "found errors" -- 3=no device compiler, 2=no compiler at all,
    # 1=real errors -- and that distinction is worth nothing unless the caller reads it. It did not, at first:
    # exit 3 landed in the `else` and was reported as FAIL with the tail of the skip message as its reason
    # ("is a full CUDA toolchain. Exiting 3 so a caller can tell..."), which is both wrong and absurd. An exit
    # code invented for a distinction has to be decoded somewhere or it is just a number.
    if rc in (2, 3):
        why = next((l.strip() for l in lines if "SKIP" in l or "not on PATH" in l), lines[0] if lines else "")
        return "SKIP", why, dt
    return ("PASS" if rc == 0 else "FAIL"), (lines[-1] if lines else ""), dt


def lint_syntax_inventory():
    """A baseline is coverage metadata, so it must correspond to a live SYNTAX source in both directions.

    Empty baseline files are valid (they mean zero accepted diagnostics), which made eight orphan files look exactly
    like eight clean, exercised sources.  Compare names instead of contents so that state cannot recur silently.
    """
    listed = {Path(src).name for src, _ in SYNTAX}
    baseline_dir = DEV / "syntax_baseline"
    baselined = {p.name[:-4] for p in baseline_dir.glob("*.txt")}
    orphan = sorted(baselined - listed)
    missing = sorted(listed - baselined)
    if orphan or missing:
        parts = []
        if orphan:
            parts.append("baseline without SYNTAX row: " + ", ".join(orphan))
        if missing:
            parts.append("SYNTAX row without baseline: " + ", ".join(missing))
        return "FAIL", "; ".join(parts), 0.0
    return "PASS", f"{len(listed)} syntax sources and baseline files match", 0.0


def lint_ppu_portability():
    """Only the CMake-evaluated PPU source graph is subject to PPU portability.

    Directory membership is not a compile edge.  The checker runs the repository's real target-registration CMake,
    follows project-owned includes from those opt-in translation units, and tests a fresh NVIDIA-only TU both before
    and after registration.  No SDK is required for branch liveness, but without CMake the graph cannot be known and
    the result is an explicit SKIP rather than a false PASS.
    """
    script = ROOT / "dev" / "fold_derivation" / "ppu_portability_check.py"
    if not script.is_file():
        return "FAIL", "ppu_portability_check.py is a repository gate and is missing", 0.0
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=ROOT)
    line = next((l.strip() for l in (r.stdout + r.stderr).splitlines() if l.strip()), f"exit {r.returncode}")
    status = {0: "PASS", 2: "SKIP"}.get(r.returncode, "FAIL")
    return status, line, 0.0


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
    """DENSE AND GROUPED MUST BE TWO NAMES FOR ONE GENERATOR, with route-only emitter differences.

    A sampled comparison between two wrappers made divergence easy to express and merely hoped the sample would
    notice later. The source invariant is stronger: exactly one TacticSpace implementation exists and DenseSpace /
    GroupedSpace are aliases of that type. The two emitter entry points remain because table labels, filenames and
    macro prefixes are distinct public ABIs; after normalising only those route names, their complete output must be
    byte-identical for every argument set declared by the thirteen shipping tables.

    WHAT MAKES THIS A CHECK RATHER THAN A RITUAL. Two planted controls must fire. Replacing GroupedSpace with a
    behaviour-identical derived type must fail the header's type-identity static_assert. Separately, changing only
    the grouped emitter entry to retain stage 2 must make the full-output comparison disagree. Thus neither an alias
    check that accepts two implementations nor an output comparison that compares nothing can report PASS.
    """
    import hashlib, importlib.util, subprocess, tempfile
    src = ROOT / "benchmarks" / "emit_tactic_configs.cpp"
    hdr = ROOT / "quactlize" / "include" / "ppu_tactic_space.hpp"
    if not src.is_file() or not hdr.is_file():
        return "FAIL", f"missing {src.name if not src.is_file() else hdr.name}", 0.0

    header_text = hdr.read_text()
    old_layers = ("dense_kernel_exclusion", "dense_non_smem_exclusion", "dense_topology_exclusion",
                  "dense_static_sweep_exclusion", "dense_sweep_exclusion")
    leftover = [name for name in old_layers
                if re.search(rf"\bconstexpr\s+Exclusion\s+{name}\s*\(", header_text)]
    if leftover:
        return "FAIL", f"second dense tactic chain still exists: {', '.join(leftover)}", 0.0
    structural = {
        "one TacticSpace implementation": header_text.count("struct TacticSpace {") == 1,
        "no DenseSpace implementation": "struct DenseSpace" not in header_text,
        "no GroupedSpace implementation": "struct GroupedSpace" not in header_text,
        "DenseSpace aliases TacticSpace": header_text.count("using DenseSpace = TacticSpace;") == 1,
        "GroupedSpace aliases TacticSpace": header_text.count("using GroupedSpace = TacticSpace;") == 1,
        "type identity is asserted": "std::is_same_v<DenseSpace, GroupedSpace>" in header_text,
    }
    missing = [claim for claim, ok in structural.items() if not ok]
    if missing:
        return "FAIL", "single-generator structure is absent: " + "; ".join(missing), 0.0

    checker_path = ROOT / "ci" / "check_dense_tactic_table.py"
    spec = importlib.util.spec_from_file_location("tactic_table_args", checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    # Share the exact-regeneration gate's inventory rule. A literal thirteen-file list would make the fourteenth
    # shipping table exact-regenerate elsewhere while silently escaping route parity here.
    tables = sorted((ROOT / "benchmarks").glob("lowbit_*_configs.inc"))
    if len(tables) < 13:
        return "FAIL", f"expected at least the 13 shipping tactic tables, found {len(tables)}", 0.0
    table_args = []
    for table in tables:
        name = table.name
        argv, why = checker.declared_args(table.read_text())
        if argv is None:
            return "FAIL", f"cannot recover emitter arguments from {name}: {why}", 0.0
        table_args.append((name, argv))

    def routed(argv, route):
        return [f"--space={route}" if arg.startswith("--space=") else arg for arg in argv]

    def neutral(output):
        # These are the ONLY intended route differences: durable table identity, not legality or row ordering.
        for before, after in ((b"LOWBIT_DENSE", b"LOWBIT_ROUTE"),
                              (b"LOWBIT_GROUPED", b"LOWBIT_ROUTE"),
                              (b"lowbit_dense", b"lowbit_route"),
                              (b"lowbit_grouped", b"lowbit_route"),
                              (b"--space=dense", b"--space=route"),
                              (b"--space=grouped", b"--space=route"),
                              (b"space=dense", b"space=route"),
                              (b"space=grouped", b"space=route")):
            output = output.replace(before, after)
        return output

    def run_pair(binary, argv):
        dense = subprocess.run([str(binary), *routed(argv, "dense")], cwd=ROOT, capture_output=True)
        grouped = subprocess.run([str(binary), *routed(argv, "grouped")], cwd=ROOT, capture_output=True)
        return dense, grouped

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        real = td / "emit_real"
        cc = ["c++", "-std=c++17", f"-I{ROOT/'quactlize'/'include'}", str(src), "-o"]
        r = subprocess.run(cc + [str(real)], cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            return "FAIL", f"emitter does not compile: {r.stderr.strip().splitlines()[-1][:120]}", 0.0

        row_total = 0
        for name, argv in table_args:
            dense, grouped = run_pair(real, argv)
            if dense.returncode or grouped.returncode:
                bad = dense if dense.returncode else grouped
                detail = bad.stderr.decode(errors="replace").strip().splitlines()
                return "FAIL", (f"emitter rejected {name}'s declared args on "
                                f"{'dense' if dense.returncode else 'grouped'} route: "
                                f"{detail[-1] if detail else f'exit {bad.returncode}'}"), 0.0
            dnorm, gnorm = neutral(dense.stdout), neutral(grouped.stdout)
            if dnorm != gnorm:
                return "FAIL", (f"dense/grouped output differs for {name} after route-name normalisation: "
                                f"dense={hashlib.sha256(dnorm).hexdigest()[:12]} "
                                f"grouped={hashlib.sha256(gnorm).hexdigest()[:12]}"), 0.0
            row_total += len(re.findall(rb"^\s*X\(", dense.stdout, re.M))

        # PLANTED CONTROL 1: aliases with identical behaviour are still two implementation types and must be
        # rejected structurally. The diagnostic is required so an unrelated compile failure cannot satisfy it.
        alias_old = "using GroupedSpace = TacticSpace;"
        alias_new = "struct PlantedGroupedSpace : TacticSpace {};\nusing GroupedSpace = PlantedGroupedSpace;"
        if header_text.count(alias_old) != 1:
            return "FAIL", "cannot plant the GroupedSpace type-identity control", 0.0
        alias_dir = td / "alias_control"
        alias_dir.mkdir()
        (alias_dir / hdr.name).write_text(header_text.replace(alias_old, alias_new))
        alias_probe = td / "emit_alias_probe"
        planted_alias = subprocess.run(
            ["c++", "-std=c++17", f"-I{alias_dir}", f"-I{ROOT/'quactlize'/'include'}", str(src),
             "-o", str(alias_probe)], cwd=ROOT, capture_output=True, text=True)
        identity_diag = "dense and grouped must remain aliases of one tactic-space generator"
        if planted_alias.returncode == 0 or identity_diag not in planted_alias.stderr:
            return "FAIL", ("planted independent GroupedSpace did not fail through the type-identity assertion; "
                            "the single-generator guard is not proving its claim"), 0.0

        # PLANTED CONTROL 2: keep only stage 2 at the grouped route. The normalised full-output check must see the
        # missing rows/header coverage even though both routes still call the same generator implementation.
        source_text = src.read_text()
        route_old = ('  if (std::strcmp(space, "grouped") == 0) {\n'
                     '    return emit(*spec, bits, tk, g_tactic_tks, "grouped", prefix("LOWBIT_GROUPED"));\n'
                     '  }')
        route_new = ('  if (std::strcmp(space, "grouped") == 0) {\n'
                     '    g_stages.assign(1, 2);  // PLANTED gate control\n'
                     '    return emit(*spec, bits, tk, g_tactic_tks, "grouped", prefix("LOWBIT_GROUPED"));\n'
                     '  }')
        if source_text.count(route_old) != 1:
            return "FAIL", "cannot plant the grouped emitter-route control", 0.0
        route_src = td / src.name
        route_src.write_text(source_text.replace(route_old, route_new))
        route_probe = td / "emit_route_probe"
        planted_route = subprocess.run(
            ["c++", "-std=c++17", f"-I{ROOT/'quactlize'/'include'}", str(route_src), "-o", str(route_probe)],
            cwd=ROOT, capture_output=True, text=True)
        if planted_route.returncode:
            detail = planted_route.stderr.strip().splitlines()
            return "FAIL", f"planted route-control build failed: {detail[-1] if detail else 'no diagnostic'}", 0.0
        dense, grouped = run_pair(route_probe, table_args[0][1])
        if dense.returncode or grouped.returncode or neutral(dense.stdout) == neutral(grouped.stdout):
            return "FAIL", ("full-output comparison did not detect the planted grouped stage restriction; "
                            "it is not comparing what it claims to"), 0.0

    return "PASS", (f"one TacticSpace type; {len(table_args)} shipping argv sets / {row_total} rows agree after "
                    "route-only names; alias and grouped-route controls both fire"), 0.0


def lint_attempt_record_roundtrip():
    """The C++ that writes attempt/sample records and the Python that reads them must agree, byte for byte.

    Both sides have self-tests and neither can catch a drift between them: analyse.py's fixtures are JSON strings
    written by hand in analyse.py, so they check the reader against its author's belief about the writer. This
    compiles bench_samples.hpp for real, runs it, and parses its actual output -- and it is registered HERE
    rather than only existing as a script, because codex pointed out that calling something a gate while nothing
    invokes it is worse than not having it.
    """
    checker = ROOT / "ci" / "check_attempt_roundtrip.py"
    if not checker.is_file():
        return "FAIL", "ci/check_attempt_roundtrip.py is missing", 0.0
    r = subprocess.run([sys.executable, str(checker)], cwd=ROOT, capture_output=True, text=True)
    line = next((l.strip() for l in (r.stdout + r.stderr).splitlines() if l.strip()), f"exit {r.returncode}")
    return ("PASS" if r.returncode == 0 else "FAIL"), line, 0.0


def lint_route_admits():
    """Ask the compiler whether the dense route admits the geometries we claim, with controls that must fail.

    REGISTERED BECAUSE IT WAS NOT. ci/check_route_admits.py was written, described as a gate, cited as evidence
    for deleting the sub-four-warp exclusion -- and never invoked by anything. codex caught that: "either
    register it or stop calling it a tier gate". Its own three controls (two configurations that must be
    rejected, one static_assert planted inside the mainloop's device body that must fire) are what make a green
    verdict here mean anything.
    """
    checker = ROOT / "ci" / "check_route_admits.py"
    if not checker.is_file():
        return "FAIL", "ci/check_route_admits.py is missing", 0.0
    r = subprocess.run([sys.executable, str(checker)], cwd=ROOT, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    if r.returncode == 0:
        return "PASS", next((l.strip() for l in reversed(out) if l.strip()), "passed"), 0.0
    return "FAIL", next((l.strip() for l in out if "FAIL" in l or "!!" in l), "\n".join(out[-2:])), 0.0


def lint_switch_macros():
    """A BUILD SWITCH WITH NO RECORDED WAY IN IS NOT A USABLE SWITCH.

    Registered the day it was written, because the last checker that was not (ci/check_route_admits.py) was cited
    as evidence for deleting an exclusion while nothing invoked it. Its six in-memory classifier controls cover
    unresolved, temporarily allowed, newly wired, deleted, setter-resolved, and definer-resolved states.

    Motivated by a real cost rather than tidiness. Three macros in this tree mean "shrink A's padding at small M"
    -- PPU_A_PACK, PPU_A_CPASYNC and the tactic table's ACR column (the last two DELETED 2026-08-07 with the
    feature) -- and on 2026-08-06 a measurement was filed
    as "compact A at capacity 1 is 45% slower" that could not afterwards be attributed to any of them. Two of the
    three are gone with the feature (task #42); PPU_A_PACK is a different, separately controlled A path.
    """
    return _run_ci_script("check_switch_macros.py", "every owned build switch has a recorded route")


def lint_build_advice():
    """Every NAME printed or written as a build input must have an implemented route through build.sh/CMake."""
    return _run_ci_script("check_build_advice.py", "advertised build inputs are routed")


def lint_moe_build_knobs():
    """The MoE format/tile/stage restrictions must change real generated sources or compile flags."""
    return _run_ci_script("check_moe_build_knobs.py", "MoE build restrictions affect the generator")


def lint_bench_measurement_shared():
    """Dense and MoE must consume the same constants, tag, repetitions, MFU, and two-ended traffic model."""
    return _run_ci_script("check_bench_measurement.py", "dense/MoE measurement fields share one implementation")


def lint_moe_event_timing():
    """The MoE primary timer must exclude setup without serialising its 20-launch batch."""
    return _run_ci_script("check_moe_event_timing.py", "MoE event interval and batching protocol are pinned")


def lint_dense_streamk_contract():
    """107b must share worker decomposition, use absolute K, and reset locks outside each event."""
    return _run_ci_script(
        "check_dense_streamk_contract.py",
        "dense Stream-K worker/K/fixup/timing seams and the exact fixture are pinned")


def lint_dense_streamk_q4k65_target():
    """The historical gs32 row must remain an isolated, normal-first scheduler A/B."""
    return _run_ci_script(
        "check_dense_streamk_q4k65_target.py",
        "Q4_K65 target, exact fixture, admission order, and 107b isolation are pinned")


def lint_dense_persistent_grid_contract():
    """Absolute persistent-DP grids must preserve exact grid-stride ownership."""
    return _run_ci_script(
        "check_dense_persistent_grid_contract.py",
        "persistent DP absolute-grid CLI, lowering, exhaustive ownership, and controls are pinned")


def lint_dense_streamk_sweep_target():
    """The complete gs32 sweep must retain its independently derived denominator and named kernel."""
    return _run_ci_script(
        "check_dense_streamk_sweep_target.py",
        "full dense Stream-K census, private target, direct wrappers, and box runner are pinned")


def lint_dense_shipping_tm8():
    """The complete m8 family must share exact compiled legality and one shape-default authority."""
    return _run_ci_script(
        "check_dense_shipping_tm8.py",
        "dense TM8 ships six stages, closes exact 51/9 cells, and owns the M<8 empty-config default")


def lint_fixed_splitk_last_arriver():
    """Actual-last completion must close both its exhaustive protocol and production source sequence."""
    script = DEV / "run_l196_fixed_splitk_last_arriver.sh"
    if not script.is_file():
        return "FAIL", f"missing {script.name}", 0.0
    rc, log, dt = run(["bash", str(script)], cwd=str(ROOT))
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    host = next((line for line in lines if line.startswith("[l196] PASS")), "")
    source = next((line for line in lines if line.startswith("[l196:source] PASS")), "")
    if rc != 0 or not host or not source:
        return "FAIL", (lines[-1] if lines else f"l196 exited {rc}"), dt
    return "PASS", host.removeprefix("[l196] PASS ") + "; source-sequence=PASS", dt


def lint_fixed_splitk_production_type():
    """The separate and fused completion policies must reach one real shipping mainloop body."""
    ok, why = nvcc_can_compile_device_cuda()
    if not ok:
        return "SKIP", f"fixed Split-K device-body proof unavailable: {why}", 0.0
    script = DEV / "run_l190_dense_splitk_parallel_type.sh"
    if not script.is_file():
        return "FAIL", f"missing {script.name}", 0.0
    rc, log, dt = run(["bash", str(script)], cwd=str(ROOT))
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    verdict = next((line for line in reversed(lines) if line.startswith("[l190] PASS")), "")
    return ("PASS" if rc == 0 and verdict else "FAIL"), (
        verdict or (lines[-1] if lines else f"l190 exited {rc}")), dt


def lint_dense_splitk_sweep_contract():
    """The generated sweep and exact fused canary must share one fail-closed output contract."""
    return _run_ci_script(
        "check_dense_splitk_sweep_contract.py",
        "dense Split-K sweep binds its denominator, two-launch oracle and actual-last canary")


def _run_fixed_splitk_runner(script_name: str, out_env: str,
                             verdict_prefix: str):
    """Run one real fixed-SplitK compiler runner and require its final verdict."""
    script = DEV / script_name
    if not script.is_file():
        return "FAIL", f"missing {script.name}", 0.0
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env[out_env] = str(OUT / script.stem)
    rc, log, dt = run(["bash", str(script)], cwd=str(ROOT), env=env)
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    verdict = next((line for line in reversed(lines)
                    if line.startswith(verdict_prefix)), "")
    if rc != 0 or not verdict:
        return "FAIL", (lines[-1] if lines else f"{script.name} exited {rc}"), dt
    return "PASS", verdict, dt


def lint_dense_splitk_shipping_selector():
    """W4 policy and exported C ABI must fail closed around one production type."""
    if shutil.which(NVCC[0]) is None:
        return "SKIP", "W4 fixed Split-K selector proof needs nvcc, which is not on PATH", 0.0
    ok, why = nvcc_can_compile_device_cuda()
    if not ok:
        return "SKIP", f"W4 fixed Split-K selector proof needs a CUDA device compiler: {why}", 0.0
    # Each checker runs its named concrete compiler runner.  Require both the
    # reusable L197 policy/type contract and L200's exported backend C-ABI edge;
    # neither verdict may stand in for the other.
    policy_runner = DEV / "run_l197_dense_splitk_shipping_selector.sh"
    policy_checker = ROOT / "ci" / "check_dense_splitk_shipping_selector.py"
    production_runner = DEV / "run_l200_dense_w4_splitk_production.sh"
    production_checker = ROOT / "ci" / "check_dense_w4_splitk_production.py"
    required = (policy_runner, policy_checker, production_runner,
                production_checker)
    missing = next((path for path in required if not path.is_file()), None)
    if missing is not None:
        return "FAIL", f"missing {missing.relative_to(ROOT)}", 0.0
    policy_rc, policy_log, policy_dt = run(
        [sys.executable, str(policy_checker)], cwd=str(ROOT))
    policy_lines = [line.strip() for line in policy_log.splitlines()
                    if line.strip()]
    policy_verdict = next((line for line in reversed(policy_lines)
                           if line.startswith(
                               "[dense-splitk-shipping-selector] PASS")), "")
    if policy_rc != 0 or not policy_verdict:
        return "FAIL", (policy_lines[-1] if policy_lines
                         else f"{policy_checker.name} exited {policy_rc}"), policy_dt
    production_rc, production_log, production_dt = run(
        [sys.executable, str(production_checker)], cwd=str(ROOT))
    production_lines = [line.strip() for line in production_log.splitlines()
                        if line.strip()]
    production_verdict = next((line for line in reversed(production_lines)
                               if line.startswith(
                                   "[dense-w4-splitk-production] PASS")), "")
    if production_rc != 0 or not production_verdict:
        return "FAIL", (production_lines[-1] if production_lines
                         else f"{production_checker.name} exited {production_rc}"), (
                             policy_dt + production_dt)
    return ("PASS", policy_verdict + " | " + production_verdict,
            policy_dt + production_dt)


def lint_dense_splitk_oneplane_formats():
    """Every shipping one-plane row must retain its exact type and FP32 partial ABI."""
    if shutil.which(NVCC[0]) is None:
        return "SKIP", "one-plane fixed Split-K type proof needs nvcc, which is not on PATH", 0.0
    ok, why = nvcc_can_compile_device_cuda()
    if not ok:
        return "SKIP", f"one-plane fixed Split-K type proof needs a CUDA device compiler: {why}", 0.0
    checker = ROOT / "ci" / "check_dense_splitk_oneplane.py"
    if not checker.is_file():
        return "FAIL", f"missing {checker.relative_to(ROOT)}", 0.0
    source_rc, source_log, source_dt = run(
        [sys.executable, str(checker)], cwd=str(ROOT))
    source_lines = [line.strip() for line in source_log.splitlines() if line.strip()]
    source_verdict = next((line for line in reversed(source_lines)
                           if line.startswith("[dense-splitk-oneplane] PASS")), "")
    if source_rc != 0 or not source_verdict:
        return "FAIL", (source_lines[-1] if source_lines
                         else f"{checker.name} exited {source_rc}"), source_dt
    status, runner_verdict, runner_dt = _run_fixed_splitk_runner(
        "run_l198_dense_splitk_oneplane.sh", "QUACTLIZE_L198_OUT",
        "[l198:runner] PASS")
    if status != "PASS":
        return status, runner_verdict, source_dt + runner_dt
    return "PASS", source_verdict + " | " + runner_verdict, source_dt + runner_dt


def lint_dense_splitk_multiformat_types():
    """All qtypes, metadata ABIs and BChunk arms must bind real shipping collectives."""
    if shutil.which(NVCC[0]) is None:
        return "SKIP", "multiformat fixed Split-K type proof needs nvcc, which is not on PATH", 0.0
    ok, why = nvcc_can_compile_device_cuda()
    if not ok:
        return "SKIP", f"multiformat fixed Split-K type proof needs a CUDA device compiler: {why}", 0.0
    return _run_fixed_splitk_runner(
        "run_l199_dense_splitk_multiformat_type.sh", "QUACTLIZE_L199_OUT",
        "[l199] PASS:")


def lint_streamk_tail_plan():
    """INBOX 122's scan must include attributed zero, medium, and extreme last waves."""
    return _run_ci_script(
        "check_streamk_tail_plan.py",
        "Stream-K scan shapes derive from runtime workers and print Q/W/tail per row")


def lint_streamk_tail_oracle():
    """The committed dense domain must admit an exact, nonempty DP-major tail partition.

    L201 is a Python/boundary oracle, not one of the nvcc-compiled ``GATES``.
    Registering it as a lint keeps the device-free exhaustive proof in the full
    local tier.  Require its unique terminal witness as well as rc=0: otherwise
    a runner that stops after printing only an anchor would look green.
    """
    script = DEV / "run_l201_streamk_tail_oracle.sh"
    if not script.is_file():
        return "FAIL", f"missing {script.name}", 0.0
    rc, log, dt = run(["bash", str(script)], cwd=str(ROOT))
    witness = (
        "[l201] PASS: 1772=577+1195 authority; 4616=4212 preferred+404 "
        "exact-divisor fallback; every admitted (q,k) cell exact-once and "
        "nonempty; min-peer lower bound attained on 4616/4616; legacy two-wave "
        "anchor unchanged; negative-controls=5/5_RED"
    )
    if rc != 0:
        lines = [line.strip() for line in log.splitlines() if line.strip()]
        return "FAIL", (lines[-1] if lines else f"{script.name} exited {rc}"), dt
    if log.count(witness) != 1:
        return "FAIL", "L201 exited zero without its unique exhaustive PASS witness", dt
    return "PASS", "4616 tail partitions exact/nonempty/min-peer; five planted policies red", dt


def lint_ppu_chunked_gdn_oracle():
    """The PPU GDN chunk algebra must equal an independent token recurrence.

    L203 is deliberately host-only.  It exhausts the admitted chunk/tail grid,
    binds one full 64x128x128 production specialization, proves the blocked
    unit-lower inverse, and requires four semantic plants to turn red.  The
    device compiler gate is separate: absence of a PPU SDK cannot weaken this
    algebraic claim.
    """
    script = DEV / "run_l203_chunked_gdn_oracle.sh"
    if not script.is_file():
        return "FAIL", f"missing {script.name}", 0.0
    rc, log, dt = run(["bash", str(script)], cwd=str(ROOT))
    witness = (
        "[L203] PASS: pure-C++ GDN recurrence == chunk/WY; "
        "inverse/scheduler/tails/plants closed"
    )
    if rc != 0:
        lines = [line.strip() for line in log.splitlines() if line.strip()]
        return "FAIL", (lines[-1] if lines else f"{script.name} exited {rc}"), dt
    if log.count(witness) != 1:
        return "FAIL", "L203 exited zero without its unique complete PASS witness", dt
    if log.count("EXPECTED_RED/PASS") != 4:
        return "FAIL", "L203 did not exercise all four predeclared semantic plants", dt
    return "PASS", "85 chunk cases + production 64x128x128 equal token recurrence; four plants red", dt


def lint_ppu_chunked_gdn_device_compile():
    """Instantiate the exact CUDA/CUTLASS device body and its admission reds."""
    script = DEV / "run_l204_chunked_gdn_device_compile.sh"
    if not script.is_file():
        return "FAIL", f"missing {script.name}", 0.0
    rc, log, dt = run(["bash", str(script)], cwd=str(ROOT))
    if rc == 3:
        lines = [line.strip() for line in log.splitlines() if line.strip()]
        return "SKIP", (lines[-1] if lines else "L204 needs nvcc"), dt
    witness = "[l204] PASS: exact device type compiled; C/head negatives red;"
    if rc != 0 or log.count(witness) != 1:
        lines = [line.strip() for line in log.splitlines() if line.strip()]
        return "FAIL", (lines[-1] if lines else f"{script.name} exited {rc}"), dt
    if log.count("EXPECTED_RED/PASS") != 2:
        return "FAIL", "L204 did not exercise both compile-time admission negatives", dt
    return "PASS", "shipping C64/K128/V128 device body instantiated; chunk/head plants red", dt


def lint_ppu_chunked_gdn_abi_harness():
    """Bind the box correctness harness to the public ABI and exact fixture."""
    script = DEV / "run_l205_ppu_chunked_gdn_abi.sh"
    if not script.is_file():
        return "FAIL", f"missing {script.name}", 0.0
    rc, log, dt = run(["bash", str(script), "--local"], cwd=str(ROOT))
    contract = "[L205 contract] PASS: public ABI only; T65/C64/KV128/GVA1:2;"
    compile_witness = "[L205 local] PASS: public-header/runtime compile contract;"
    cuda_witness = "[L205 CUDA] PASS: exact scalar collective body launched with global test scratch"
    honest_scope = "[L205 device] SKIP: --local selected; PPU execution requires --box"
    if rc != 0:
        lines = [line.strip() for line in log.splitlines() if line.strip()]
        return "FAIL", (lines[-1] if lines else f"{script.name} exited {rc}"), dt
    if log.count(contract) != 1 or log.count(compile_witness) != 1:
        return "FAIL", "L205 lost its unique public-ABI contract or host-compile witness", dt
    if log.count("[GDN fixture exactness plant] non-bf16-H EXPECTED_RED/PASS") != 1:
        return "FAIL", "L205 BF16-boundary proof did not reject its non-representable H plant", dt
    admission_plants = [
        line for line in log.splitlines()
        if line.startswith("[GDN admission] plant=") and line.endswith(" EXPECTED_RED/PASS")
    ]
    if len(admission_plants) != 7 or not any("plant=misaligned-q " in line for line in admission_plants) or not any(
            "plant=grid-overflow " in line for line in admission_plants) or not any(
            "plant=extent-overflow " in line for line in admission_plants):
        return "FAIL", "L205 did not reject all seven admission plants, including alignment/overflow", dt
    paired_coverage = (
        "[GDN WY coverage] pattern=paired strict_lower_nonzero=64 "
        "inverse_offdiag_nonzero=64 causal_offdiag_nonzero=64 expected=64 EXACT/PASS"
    )
    if log.count(paired_coverage) != 2:
        return "FAIL", "L205 paired fixture did not exercise the exact 64/64/64 WY edges", dt
    paired_device = [
        line for line in log.splitlines()
        if line.startswith("[GDN device] pattern=paired ") and line.endswith(" RAW-BIT/PASS")
    ]
    if len(paired_device) != 2 or not any("state=zero " in line for line in paired_device) or not any(
            "state=nonzero " in line for line in paired_device):
        return "FAIL", "L205 paired WY zero/nonzero device arms were not both raw-bit exact", dt
    if log.count(honest_scope) != 1:
        return "FAIL", "L205 local arm did not state that device execution remains a box postcondition", dt

    # Same-file, one-variable negative control: only MODE changes.  /bin/false
    # stands in for the box's misleading command named nvcc; if the --box arm
    # consults it at all, `set -e` turns this preflight red before its witness.
    # This is the exact failure that previously prevented the shipping hgcc
    # build from being reached on PPU boxes.
    box_env = os.environ.copy()
    box_env.update({
        "NVCC": "/bin/false",
        "NVIDIA_SMI": "/bin/true",
        "QZ_GDN_BOX_PREFLIGHT_ONLY": "1",
        "OUT": "/workspace/quactlize-l205-box-mode-preflight",
    })
    box_rc, box_log, box_dt = run(
        ["bash", str(script), "--box"], cwd=str(ROOT), env=box_env)
    dt += box_dt
    box_skip = (
        "[L205 CUDA] SKIP: --box selects shipping PPU execution; "
        "NVIDIA reference belongs to --local"
    )
    box_witness = "[L205 box preflight] PASS: local CUDA tools were not consulted"
    if box_rc != 0 or box_log.count(box_skip) != 1 or box_log.count(box_witness) != 1:
        lines = [line.strip() for line in box_log.splitlines() if line.strip()]
        return "FAIL", (lines[-1] if lines else "L205 box mode consulted local CUDA tools"), dt

    if log.count(cuda_witness) == 1:
        return "PASS", "public ABI harness exact; full scalar body executed on local CUDA", dt
    cuda_skip = next((line.strip() for line in log.splitlines()
                      if line.startswith("[L205 CUDA] SKIP:")), "")
    if cuda_skip:
        return "SKIP", f"public ABI compile contract passed; {cuda_skip}", dt
    return "FAIL", "L205 neither executed nor honestly skipped its local CUDA body", dt


def lint_ppu_chunked_gdn_global_dot_ownership():
    """Exhaust the shipping PPU TiledMma accumulator destination map."""
    script = DEV / "run_l206_chunked_gdn_global_dot_ownership.sh"
    if not script.is_file():
        return "FAIL", f"missing {script.name}", 0.0
    rc, log, dt = run(["bash", str(script)], cwd=str(ROOT))
    if rc == 3:
        lines = [line.strip() for line in log.splitlines() if line.strip()]
        return "SKIP", (lines[-1] if lines else "L206 needs nvcc"), dt
    witness = (
        "[l206] PASS: production accumulator map exact-once; "
        "two negatives red;"
    )
    ownership = (
        "threads=128 coordinate_stride=1 tile=64x64 visits=4096 "
        "holes=0 duplicate_coordinates=0 duplicate_visits=0 oob=0 min=1 max=1"
    )
    if rc != 0 or log.count(witness) != 1 or log.count(ownership) != 1:
        lines = [line.strip() for line in log.splitlines() if line.strip()]
        return "FAIL", (lines[-1] if lines else f"{script.name} exited {rc}"), dt
    if log.count("EXPECTED_RED/PASS") != 2:
        return "FAIL", "L206 did not reject both ownership-map plants", dt
    return "PASS", "real PPU TiledMma owns every 64x64 destination exactly once", dt


def lint_dense_marlin_contract():
    """Marlin must remain an additive K-fast scheduler with its own peer protocol and launch guard."""
    return _run_ci_script(
        "check_dense_marlin_contract.py",
        "dense Marlin decomposition/cooperative and same-event DP/SK/Marlin route are pinned")


def lint_dense_marlin_exhaustive():
    """The declared Marlin deployment domain must close exact-once without sampling."""
    return _run_ci_script(
        "check_dense_marlin_exhaustive.py",
        "dense Marlin exhausts 656230 raw tuples and every production (q,K-tile) cell")


def lint_dense_marlin_codegen():
    """The real dense Cfg must emit the same raw-shape/K/workspace/lock arithmetic."""
    ok, why = nvcc_can_compile_device_cuda()
    if not ok:
        return "SKIP", f"Marlin generated-code proof unavailable: {why}", 0.0
    script = DEV / "run_l134_marlin_codegen.sh"
    if not script.is_file():
        return "FAIL", f"missing {script.name}", 0.0
    rc, log, dt = run(["bash", str(script)], cwd=str(ROOT))
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    verdict = lines[-1] if lines else "l134 produced no output"
    return ("PASS" if rc == 0 else "FAIL"), verdict, dt


def lint_dense_marlin_sweep_contract():
    """The full-table target must emit every structurally capable exact-cohort Marlin wrapper."""
    return _run_ci_script(
        "check_dense_marlin_sweep_contract.py",
        "dense Marlin sweep has private sources, exact capable cohorts, distinct identity and no DP/cache fallback")


def lint_dense_marlin_rejection_census():
    """The A2 capability delta must recover every row formerly cut by the 2/4-warp whitelist."""
    return _run_ci_script(
        "check_dense_marlin_rejection_census.py",
        "Marlin A2 recovery closes exactly by cohort against the committed legal rows")


def lint_dense_marlin_rejected_cohorts():
    """Compile one released Marlin row per A2 cohort and reject an inexact explicit cohort."""
    script = DEV / "run_l131_marlin_rejected_cohorts.sh"
    if not script.is_file():
        return "FAIL", f"missing {script.name}", 0.0
    rc, log, dt = run(["bash", str(script)], cwd=str(ROOT))
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    verdict = lines[-1] if lines else "l131 produced no output"
    if rc in (2, 3):
        why = next((line for line in lines if "SKIP" in line), verdict)
        return "SKIP", why, dt
    return ("PASS" if rc == 0 else "FAIL"), verdict, dt


def lint_dense_marlin_output_owners():
    """WK4 verification must use the exact K0 output cohort and reject false owners."""
    ok, why = nvcc_can_compile_device_cuda()
    if not ok:
        return "SKIP", f"Marlin output-owner proof unavailable: {why}", 0.0
    script = DEV / "run_l139_marlin_warpk_reduce.sh"
    if not script.is_file():
        return "FAIL", f"missing {script.name}", 0.0
    rc, log, dt = run(["bash", str(script)], cwd=str(ROOT))
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    verdict = lines[-1] if lines else "l139 Marlin owner proof produced no output"
    return ("PASS" if rc == 0 else "FAIL"), verdict, dt


def lint_dense_arrangement_abi():
    """A folded artifact descriptor must select its exact reader and fail closed on every ABI mismatch."""
    script = DEV / "run_l138_dense_arrangement_abi.sh"
    if not script.is_file():
        return "FAIL", f"missing {script.name}", 0.0
    rc, log, dt = run(["bash", str(script)], cwd=str(ROOT))
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    verdict = lines[-1] if lines else "l138 produced no output"
    return ("PASS" if rc == 0 else "FAIL"), verdict, dt


def lint_bc_arrangement_layout():
    """BC's arrangement reader must equal the production xplane writer over every physical code slot."""
    script = DEV / "run_l137_bc_arrangement_layout.sh"
    if not script.is_file():
        return "FAIL", f"missing {script.name}", 0.0
    rc, log, dt = run(["bash", str(script)], cwd=str(ROOT))
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    verdict = lines[-1] if lines else "l137 produced no output"
    return ("PASS" if rc == 0 else "FAIL"), verdict, dt


def lint_format_loader():
    """Every qtype must open its own packed-format binary under the documented path precedence."""
    script = DEV / "run_l140_format_loader.sh"
    if not script.is_file():
        return "FAIL", f"missing {script.name}", 0.0
    rc, log, dt = run(["bash", str(script)], cwd=str(ROOT))
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    verdict = lines[-1] if lines else "l140 produced no output"
    if rc != 0:
        return "FAIL", verdict, dt
    # l140 proves loader semantics, but the actual device recipe once supplied only the generic base path.  That
    # made load_format(fmt) splice a nonexistent `_fmtN.so` even though the just-built library was valid.  Bind the
    # integration line and its independent qtype mapping here so this exact failure cannot recur outside the oracle.
    batch = (ROOT / "benchmarks" / "run_batch.sh").read_text()
    required = (
        '12) _packed_fmt=0', '13) _packed_fmt=1', '10) _packed_fmt=2',
        '11) _packed_fmt=3', '14) _packed_fmt=4',
        '"QUACTLIZE_PPU_LIB_FMT${_packed_fmt}=$_fmtso"',
    )
    missing = [needle for needle in required if needle not in batch]
    if missing:
        return "FAIL", f"run_batch format-loader binding missing {missing}", dt
    return "PASS", verdict + "; run_batch FMT binding=PASS", dt


def lint_q36_zero_redundancy():
    """Q3/Q6 external fp16 zero must remain a bitwise function of the scale returned by each producer."""
    script = DEV / "run_l139_q36_zero_redundancy.sh"
    if not script.is_file():
        return "FAIL", f"missing {script.name}", 0.0
    rc, log, dt = run(["bash", str(script), "--json"], cwd=str(ROOT))
    if rc != 0:
        lines = [line.strip() for line in log.splitlines() if line.strip()]
        return "FAIL", (lines[-1] if lines else f"l139 exited {rc}"), dt

    payload = None
    for line in log.splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "formats" in candidate:
            payload = candidate
    if payload is None:
        return "FAIL", "l139 emitted no JSON result", dt

    expected_claim = (
        "external-fp16-zero-is-structurally-derived; "
        "packed-unit-already-has-no-external-zero; physical-plane-removal-not-implemented"
    )
    expected_scope = {
        "structural_formulas": True,
        "bitwise_witness_elements_per_arm": 8192,
        "sample_is_not_exhaustive": True,
        "packed_collective_kPackedZMul_proved": False,
    }
    if payload.get("claim") != expected_claim or payload.get("scope") != expected_scope:
        return "FAIL", "l139 claim/scope drifted from the proved boundary", dt
    rows = payload.get("formats")
    if not isinstance(rows, list) or [row.get("name") for row in rows] != ["Q3_K", "Q6_K"]:
        return "FAIL", "l139 did not report exactly Q3_K and Q6_K", dt
    for row in rows:
        name = row["name"]
        arm_counts = tuple(row.get(k) for k in (
            "scale_first_elements", "dense_elements", "packed_decode_elements"))
        if arm_counts != (8192, 8192, 8192):
            return "FAIL", f"{name} sf/dense/packed arms are not each 8192 elements: {arm_counts}", dt
        if tuple(row.get(k) for k in ("scale_first_bad", "dense_bad", "packed_decode_bad")) != (0, 0, 0):
            return "FAIL", f"{name} reconstructed zero differs from the producer", dt
        if not (0.0 <= row.get("official_max_block_relative_error", 1.0) < 1.0e-3):
            return "FAIL", f"{name} lost the independent official-GGUF anchor", dt
        if row.get("wrong_bias_witnesses", 0) <= 0 or row.get("packed_perturbation_witnesses") != 1:
            return "FAIL", f"{name} negative controls did not fire exactly", dt
        want_rounding = 1 if name == "Q6_K" else 0
        if row.get("targeted_dense_actual_vs_staged_bad") != 0:
            return "FAIL", f"{name} actual producer disagrees with the staged formula", dt
        if row.get("targeted_dense_rounding_witnesses") != want_rounding:
            return "FAIL", f"{name} staged-rounding control expected {want_rounding}", dt

    # The two meta-controls guard the proof machinery itself.  The first plants a stale extension without touching
    # source mtimes; the second proves Python -O cannot erase the assertion-backed structural checks into a green.
    stale_env = dict(os.environ, L139_PLANT_STALE_EXTENSION="1")
    stale_rc, stale_log, stale_dt = run(["bash", str(script), "--json"], cwd=str(ROOT), env=stale_env)
    if stale_rc == 0 or "_C is older than its source inputs" not in stale_log:
        return "FAIL", "l139 stale-extension plant did not fail closed", dt + stale_dt
    optimized_rc, optimized_log, optimized_dt = run(
        [sys.executable, "-O", str(DEV / "q36_zero_redundancy.py"), "--json"], cwd=str(ROOT))
    if optimized_rc == 0 or "assertions are disabled" not in optimized_log:
        return "FAIL", "l139 Python -O plant did not fail closed", dt + stale_dt + optimized_dt
    return ("PASS", "Q3/Q6 external zero=derived bitwise (8192/arm/format); official anchor and all negatives PASS",
            dt + stale_dt + optimized_dt)


def lint_grouped_marlin_contract():
    """Grouped Marlin must flatten ragged experts to global q without changing collectives."""
    return _run_ci_script(
        "check_grouped_marlin_contract.py",
        "grouped Marlin preserves ragged-prefix/global-q seams and instantiates all collective families")


def lint_grouped_marlin_exhaustive():
    """Every committed grouped table/shape tuple must cover each (q,K-tile) once."""
    return _run_ci_script(
        "check_grouped_marlin_exhaustive.py",
        "grouped Marlin exhausts committed formats and ragged MoE routes without sampling")


def lint_gemv_perf_authority():
    """The GEMV bench must derive real shapes/routing and detect expert-pitch regressions."""
    return _run_ci_script(
        "check_gemv_perf_authority.py",
        "GEMV S068-S079 fixtures are ragged, expert-distinct, poisoned and pitch-sensitive")


def lint_gemv_tactic_space():
    """The finite GEMV axes, legality census and generated 540-unit view must agree."""
    return _run_ci_script(
        "check_gemv_tactic_space.py",
        "GEMV finite axes/prune census and the committed compile-unit authority agree")


def lint_gemv_exact_ctam():
    """The benchmark-only exact CtaM route must not mutate adaptive shipping dispatch."""
    return _run_ci_script(
        "check_gemv_exact_ctam.py",
        "GEMV exact dense/grouped CtaM domains compile while adaptive production stays unchanged")


def lint_gemv_event_protocol():
    """Each GEMV sample must have one independent device-event pair."""
    return _run_ci_script(
        "check_gemv_event_protocol.py",
        "GEMV raw events preserve one warmup and one pair per measured launch")


def lint_gemv_sweep_driver():
    """Bounded GEMV runs must resume without path aliases or poisoned raw prefixes."""
    return _run_ci_script(
        "check_gemv_sweep_driver.py",
        "GEMV driver dry-run, deadline, slash-ID, resume and exact coverage controls pass")


def lint_gemv_sweep_integration():
    """Manifest identities, generated exact-CtaM units and raw writer are one graph."""
    return _run_ci_script(
        "check_gemv_sweep_integration.py",
        "GEMV full/partial manifests share the exact-CtaM and raw-event runtime identity")


def lint_box_run_adjudicator():
    """Preregistered dense/GEMV box verdicts must be mechanical and planted-red."""
    return _run_ci_script(
        "check_box_run_adjudicator.py",
        "box results use the sealed preregistration; convergence/partial/VOID plants pass")


def lint_box_runner_bundle_contract():
    """Frozen box runners must emit the exact evidence their adjudicator consumes."""
    return _run_ci_script(
        "check_box_runner_bundle_contract.py",
        "dense/GEMV runners own provenance, command journals and authoritative census")


def lint_box_identity_probe():
    """Box provenance must measure one device or fail without guessing."""
    return _run_ci_script(
        "check_box_identity_probe.py",
        "box identity is atomic, source-labelled, and 0/multiple/empty probes fail closed")


def lint_l143_wk4_committed_evidence():
    """The result-SHA WK1 admission must be reproducible locally with planted reds."""
    return _run_ci_script(
        "check_l143_wk4_committed_evidence.py",
        "L143 committed evidence regenerates exactly and seventeen structural plants red")


def lint_gemv_lop3_codegen():
    """The real sm_120 shipping GEMV must report normalized extraction codegen, not mere LOP3 presence."""
    script = ROOT / "ci" / "check_gemv_lop3_codegen.py"
    if not script.is_file():
        return "FAIL", f"missing {script.name}", 0.0
    rc, log, dt = run([sys.executable, str(script)], cwd=str(ROOT))
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    verdict = lines[-1] if lines else "l145 produced no output"
    if rc == 3:
        return "SKIP", verdict, dt
    return ("PASS" if rc == 0 else "FAIL"), verdict, dt


def lint_q4k_pdf_ab_contract():
    """The reconstructed PDF comparison must preserve source, packer and raw-timing boundaries."""
    script = ROOT / "ci" / "check_q4k_pdf_ab_contract.py"
    if not script.is_file():
        return "FAIL", f"missing {script.name}", 0.0
    rc, log, dt = run([sys.executable, str(script)], cwd=str(ROOT))
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    verdict = lines[-1] if lines else "Q4_K PDF A/B contract produced no output"
    if rc == 3:
        return "SKIP", verdict, dt
    return ("PASS" if rc == 0 else "FAIL"), verdict, dt


def lint_grouped_streamk_contract():
    """Grouped Stream-K must preserve global q for locks while decoding expert-local compute coordinates."""
    ok, why = nvcc_can_compile_device_cuda()
    if not ok:
        return "SKIP", ("needs an NVIDIA device compiler for the CUTLASS stack its oracle includes: "
                        + why), 0.0
    return _run_ci_script(
        "check_grouped_streamk_contract.py",
        "grouped Stream-K q/worker/K/fixup/timing seams and both decode controls are pinned")


def lint_streamk_fixup_cohort():
    """Stream-K fixup must use the exact 64/128-thread CTA as barrier and scratch cohort."""
    ok, why = nvcc_can_compile_device_cuda()
    if not ok:
        return "SKIP", ("needs an NVIDIA device compiler for the CUTLASS stack its oracle includes: "
                        + why), 0.0
    return _run_ci_script(
        "check_streamk_fixup_cohort.py",
        "Stream-K exact CTA cohort and real-fragment workspace coverage are pinned")


def lint_fp32_residue_fixup():
    """Every tactic layout must predicate the same scalar FP32 fixup slots."""
    ok, why = nvcc_can_compile_device_cuda()
    if not ok:
        return "SKIP", ("needs an NVIDIA device compiler for the CUTLASS stack its oracle includes: "
                        + why), 0.0
    script = DEV / "run_l124_fp32_residue_mask.sh"
    if not script.is_file():
        return "FAIL", f"missing {script.name}", 0.0
    out = OUT / "l124_fp32_residue_mask"
    env = dict(os.environ, QUACTLIZE_L124_OUT=str(out))
    rc, log, dt = run(["bash", str(script)], cwd=str(ROOT), env=env)
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    verdict = next((line for line in reversed(lines) if "L124 planted-" in line),
                   lines[-1] if lines else "l124 produced no output")
    return ("PASS" if rc == 0 else "FAIL"), verdict, dt


def lint_fp32_residue_contract():
    """Predicates guard scalar workspace accesses, never Stream-K lock progress."""
    return _run_ci_script(
        "check_fp32_residue_fixup_contract.py",
        "FP32 residue mask is shared by store/reduce/load-add while barriers remain unconditional")


def lint_m8n16_g2_contract():
    """G2 must replay the historical provider index on one production x4 payload."""
    return _run_ci_script(
        "check_m8n16_g2_contract.py",
        "m8n16 G2 maps one 16-row x4 payload with get_i/get_j and the historical NVIDIA provider index")


def lint_grouped_metadata_layout():
    """L125 exhausts the exact G5 zero-plane layout without asking a device."""
    script = DEV / "run_l125_grouped_metadata_layout.sh"
    if not script.is_file():
        return "FAIL", f"missing {script.name}", 0.0
    out = OUT / "l125_grouped_metadata_layout"
    env = dict(os.environ, QUACTLIZE_L125_OUT=str(out))
    rc, log, dt = run(["bash", str(script)], cwd=str(ROOT), env=env)
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    verdict = " | ".join(lines[-2:]) if lines else "l125 produced no output"
    return ("PASS" if rc == 0 else "FAIL"), verdict, dt


def lint_grouped_metadata_layout_contract():
    """Production G5 and L125 share one typed helper while independent anchors remain load-bearing."""
    return _run_ci_script(
        "check_grouped_metadata_layout_contract.py",
        "G5 metadata host-algebra contract")


def lint_grouped_b_idprobe_contract():
    """L130 exhausts G5 B expert identity with independent byte-map anchors."""
    return _run_ci_script(
        "check_grouped_b_idprobe_contract.py",
        "G5 B-side exhaustive identity contract")


def lint_g5_harness_slot_contract():
    """G5 must translate caller slots to real experts before naming rows and oracles."""
    return _run_ci_script(
        "check_g5_harness_slot_contract.py",
        "G5 caller-slot, real-expert, and output-row index spaces remain distinct")


def lint_metadata_stride_contract():
    """Caller dS must survive lowering and change the shared S/Z CuTe address map."""
    return _run_ci_script(
        "check_metadata_stride_contract.py",
        "three mixed-input collectives consume caller dS and L127 rejects compact substitution")


def lint_mixed_argument_contract():
    """Outer A bases and metadata residues must use caller/logical coordinates."""
    return _run_ci_script(
        "check_mixed_argument_contract.py",
        "three mixed-input collectives share dA outer-base and logical-N residue seams")


def lint_mixed_logical_m_residue():
    """Output M/N residues must use logical tiles even when m8 widens physical A."""
    return _run_ci_script(
        "check_mixed_logical_m_residue.py",
        "nine mixed-input kernels use logical M/N residues and loaded-A K")


def lint_subbyte_units():
    """Physical bytes, logical codes and scheduler K tiles must not alias units."""
    return _run_ci_script(
        "check_subbyte_units.py",
        "sub-byte allocation/copy and expert-pitch unit seams are explicit")


def lint_plain_ldsm_failclosed():
    """The twelve dormant ppu001 plain-LDSM entries must fail before assembly when called."""
    return _run_ci_script(
        "check_plain_ldsm_failclosed.py",
        "ppu001 plain LDSM fails at C++ call sites while ppu0015 tc02 remains intact")


def _run_ci_script(name: str, label: str):
    """Shared shim for the ci/ checkers that are complete programs already; report their last meaningful line."""
    checker = ROOT / "ci" / name
    if not checker.is_file():
        return "FAIL", f"ci/{name} is missing", 0.0
    r = subprocess.run([sys.executable, str(checker)], cwd=ROOT, capture_output=True, text=True)
    out = [l.strip() for l in (r.stdout + r.stderr).splitlines() if l.strip()]
    if r.returncode == 0:
        return "PASS", " | ".join(out[-2:]) if out else label, 0.0
    return "FAIL", next((l for l in out if "FAIL" in l or "ERROR" in l), "\n".join(out[-2:])), 0.0


def lint_actlize_pristine():
    """actlize must carry none of quactlize's work: no owned symbol, no file changed outside the allow-list.

    REGISTERED WITH THE CHECK, not after it. The extraction on 2026-08-06 moved five whole files, a dispatch
    policy block and a converter's worth of specialisations out of the vendor fork; nothing but this notices when
    one drifts back, and a leak back into actlize is invisible precisely because it keeps building.
    """
    return _run_ci_script("check_actlize_pristine.py", "actlize carries no quactlize work")


def lint_extension_additive():
    """quactlize_extensions must ADD to actlize, never redefine it or overlap its specialisations.

    The dangerous half is the overlap: a partial specialisation whose constraint intersects a vendor one is
    ambiguous only for the argument lists a build instantiates, so a table that never reaches the overlap
    compiles green. quactlize's builder claimed six of actlize's schedule tags for months without a diagnostic,
    because its copy REPLACED actlize's in the include list rather than joining it.
    """
    return _run_ci_script("check_extension_additive.py", "quactlize_extensions adds rather than redefines")


def lint_owned_symbol_includes():
    """A file naming a quactlize_extensions type must reach its defining header, per a real include closure.

    WRITTEN AFTER THE BOX CAUGHT IT, which is the whole point. The extraction repointed the 17 files that
    included actlize's umbrella and missed the ones reaching the same types through DIRECT includes of specific
    cutlass headers. Those kept compiling until actlize stopped defining the type, and the box then reported two
    of ten -- because a compiler stops at the first. Ten round trips at minutes each is the cost this replaces.
    """
    return _run_ci_script("check_owned_symbol_includes.py", "owned types reach their defining header")


def lint_generated_include_edges():
    """An #include the CMake generator writes into a generated source must resolve.

    The closure tools read what a compiler reads; this build generates sources in between, so no closure over
    the repository can see `#include "moe_splitk_unit.inc"` -- the file carrying it does not exist until cmake
    runs. Assembling main on 2026-08-06 lost moe_bench_unit.inc and gemv_perf_unit.inc that way, twice.
    """
    return _run_ci_script("check_generated_include_edges.py", "generated includes resolve")


def lint_dense_unit_generator():
    """The dense CMake parser must batch the committed table and reject every ambiguous input shape."""
    return _run_ci_script("check_dense_unit_generator.py", "dense table parser and batcher fail closed")


def lint_format_table_buildable():
    """A listed format must have the collective its own row implies, so a feature revert cannot leave a claim.

    ppu_format_config.inc and formats.py are cross-checked against EACH OTHER; neither is checked against what
    the tree can build. Remove the two-plane collective and both still agree that Q3/Q5/Q6 exist.
    """
    return _run_ci_script("check_format_table_buildable.py", "listed formats have their collectives")


def lint_dense_tactic_table_current():
    """EVERY committed dense/grouped X-macro must be exact output from the current emitter.

    THIS GATE COVERED ONE TABLE OUT OF SIX until 2026-08-07, and the five it skipped are the ones that drifted:
    each grouped table had been emitted with a SINGLE --tactic-tk while dense carried three, so grouped i4 searched
    564 rows where the same space offers 1164. Nothing was wrong with the tables' provenance -- it was never read.
    Discovering that took a user asking why a sweep winner looked wrong; a gate that runs on one member of a set is
    not a gate on the set, and "the checker is named check_DENSE_tactic_table" is not a reason for it to be.
    """
    checker = ROOT / "ci" / "check_dense_tactic_table.py"
    if not checker.is_file():
        return "FAIL", "ci/check_dense_tactic_table.py is missing", 0.0
    tables = sorted((ROOT / "benchmarks").glob("lowbit_*_configs.inc"))
    if len(tables) < 2:
        return "FAIL", f"expected the dense table and the grouped set, found {len(tables)}", 0.0
    verified = []
    for tbl in tables:
        r = subprocess.run([sys.executable, str(checker), "--table", str(tbl)], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode:
            detail = next((ln for ln in r.stdout.splitlines() if "ERROR:" in ln), r.stdout.strip())
            return "FAIL", detail or r.stderr.strip(), 0.0
        verified.append(tbl.name)
    checked = subprocess.run([sys.executable, str(checker)], cwd=ROOT, capture_output=True, text=True)
    if checked.returncode:
        detail = next((line for line in checked.stdout.splitlines() if "ERROR:" in line), checked.stdout.strip())
        return "FAIL", detail or checked.stderr.strip(), 0.0

    # Assert that exact comparison, rather than only the self-reported metadata, is live. Keep all provenance
    # unchanged and mutate one row: a metadata-only checker would accept this, while regeneration must reject it.
    import tempfile
    source = ROOT / "benchmarks" / "lowbit_dense_configs.inc"
    with tempfile.TemporaryDirectory() as td:
        planted = Path(td) / source.name
        text = source.read_text()
        # THE ANCHOR IS FOUND, NOT HARDCODED. It was the literal "  X(16,64,16,16,2,B)" until TacticTileK became a
        # row field and rows grew a seventh argument -- at which point this control could not plant and the gate
        # said so rather than passing, which is what it should do. But a probe that breaks whenever the row FORMAT
        # changes is a probe that gets edited under time pressure, so take the first row and mutate its last
        # numeric field instead of naming one.
        rows = re.findall(r"^  X\([^)]*\)", text, re.M)
        if not rows:
            return "FAIL", "cannot plant the dense-table drift probe: no X( rows in the table", 0.0
        old = rows[0]
        if text.count(old) != 1:
            return "FAIL", f"cannot plant the dense-table drift probe: {old} is not unique", 0.0
        # Mutate the LAST numeric field (PPU_B_CHUNK) to a value no table emits, derived from the row rather than
        # typed: a literal replacement row has to be edited every time the row format changes, and the version
        # that was there emitted a 6-field row into a 7-field table -- which the checker would have rejected for
        # the wrong reason, reporting drift where the probe itself was malformed.
        # The row ends `,B)` -- B is the dispatch-body placeholder, not a field -- so the last NUMERIC field is
        # the one before it. Anchoring on `,B)` rather than on `)` is the difference between mutating stages and
        # matching nothing, which is what the first version of this did.
        mutated = re.sub(r"(\d+)(,B\)$)", r"99\2", old)
        if mutated == old:
            return "FAIL", f"cannot plant the dense-table drift probe: no numeric field in {old}", 0.0
        planted.write_text(text.replace(old, mutated))
        rejected = subprocess.run([sys.executable, str(checker), "--table", str(planted)], cwd=ROOT,
                                  capture_output=True, text=True)
        if rejected.returncode == 0:
            return "FAIL", "checker accepted a table whose first tactic row was changed without changing provenance", 0.0
        # A decode table used to advertise a command that overwrote its full table. Exact regeneration catches a
        # hand edit, but would accept the bad target again after both emitter and tables were regenerated. Require
        # the checker to bind the advertised sink to the file being checked, and require that exact diagnostic so
        # an unrelated parse failure cannot make this negative control green.
        decode = ROOT / "benchmarks" / "lowbit_grouped_Q3_K_decode_configs.inc"
        decode_text = decode.read_text()
        wrong_target = "benchmarks/lowbit_grouped_Q3_K_configs.inc"
        right_target = "benchmarks/lowbit_grouped_Q3_K_decode_configs.inc"
        if decode_text.count(right_target) != 1:
            return "FAIL", "cannot plant decode output-target drift: expected one regeneration target", 0.0
        planted_decode = Path(td) / decode.name
        planted_decode.write_text(decode_text.replace(right_target, wrong_target))
        rejected_target = subprocess.run([sys.executable, str(checker), "--table", str(planted_decode)], cwd=ROOT,
                                         capture_output=True, text=True)
        target_diagnostic = f"regeneration command writes {wrong_target}, not {decode.name}"
        if rejected_target.returncode == 0 or target_diagnostic not in rejected_target.stdout:
            return "FAIL", "checker did not reject a decode command that overwrites its full table", 0.0
    summary = checked.stdout.strip().removeprefix("[dense-table] ")
    return "PASS", (f"{len(verified)} table(s) exact: {', '.join(verified)}; "
                    "planted row and decode-output drift rejected"), 0.0


def lint_tactic_buckets_do_not_extrapolate():
    """The offline writer stores only measured power-of-two buckets; it never opens the final range."""
    import types
    source = ROOT / "tools" / "tune.py"
    try:
        namespace = {"__file__": str(source), "__name__": "quactlize_tune_bucket_check"}
        exec(compile(source.read_text(), str(source), "exec"), namespace)
        module = types.SimpleNamespace(**namespace)
    except (OSError, SyntaxError):
        return "FAIL", "cannot load tools/tune.py", 0.0
    expected = {1: 1, 2: 2, 3: 2, 4: 4, 63: 32, 64: 64, 4097: 4096}
    got = {m: module.m_bucket(m) for m in expected}
    if got != expected:
        return "FAIL", f"power-of-two bucket mapping drifted: {got}", 0.0
    rows = module.measured_buckets({1: "a", 2: "b", 4: "c", 64: "d", 2048: "e", 4096: "f"})
    keys = [row.get("m_bucket") for row in rows]
    if keys != [1, 2, 4, 64, 2048, 4096] or any("m_max" in row for row in rows):
        return "FAIL", f"writer extrapolated or changed measured buckets: {rows}", 0.0
    if any(row.get("m_bucket") == 8 for row in rows):
        return "FAIL", "writer invented an unmeasured M=8 bucket", 0.0
    if module.fnv_hex(module.BUCKET_POLICY.encode()) != "96fb357ad3662e26" or \
            module.fnv_hex(module.ROUTE_SCHEMA.encode()) != "7de8d745e8971595":
        return "FAIL", "runtime-cache policy hashes drifted from the ggml reader contract", 0.0

    import tempfile
    from benchmarks.workloads import MODELS
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake_so = td / "libquactlize_ppu.so"
        fake_so.write_bytes(b"loaded-ELF-provenance-control")
        cache = td / "tactics.cache"
        inventory = [dict(enable_cuda_kernel=False, name="64x64:64x32:s3",
                          tile_m=64, tile_n=64, warp_m=64, warp_n=32, stages=3)]
        table = [dict(n=8192, k=5120, gs=32,
                      buckets=[dict(m_bucket=4, measured_m=4, config="64x64:64x32:s3")])]

        class Admit:
            def __call__(self, *_args):
                return 1

        class ValidityLib:
            quactlize_ppu_dense_lowbit_config_valid_v1 = Admit()
            quactlize_ppu_gemv_lowbit_config_valid_v1 = Admit()
            quactlize_ppu_dense_fully_quantized_config_valid_v1 = Admit()
            quactlize_ppu_grouped_lowbit_config_valid_v1 = Admit()
            quactlize_ppu_grouped_fully_quantized_config_valid_v1 = Admit()
            quactlize_ppu_vecdot_moe_config_valid_v1 = Admit()

        module.write_runtime_cache(cache, fake_so, "ppu001-test", "Qwen3-32B", MODELS["Qwen3-32B"],
                                   "dense_lowbit", 12, 32, table, inventory, "0123456789abcdef", ValidityLib())
        text = cache.read_text()
        required = ("schema=quactlize-ppu-tactic-cache-v1", "workload=Qwen3-32B",
                    "m_bucket=4", "op=attn_q", "config=64x64:64x32:s3")
        if any(field not in text for field in required) or "m_max" in text or "unresolved" in text:
            return "FAIL", "strict runtime cache omitted provenance or included non-winner state", 0.0

        moe_model = "Qwen3.5-35B-A3B"
        moe_table = [dict(n=512, k=2048, gs=32,
                          buckets=[dict(m_bucket=4, measured_m=4, config="64x64:64x32:s3")])]
        moe_cache = td / "grouped-lowbit.cache"
        module.write_runtime_cache(
            moe_cache, fake_so, "ppu001-test", moe_model, MODELS[moe_model], "grouped_lowbit",
            12, 32, moe_table, inventory, "0123456789abcdef", ValidityLib())
        moe_text = moe_cache.read_text()
        moe_required = ("route=grouped_lowbit", "op=ffn_moe_gate", "experts=256", "top_k=8",
                        "cuda=0", "config=64x64:64x32:s3")
        if any(field not in moe_text for field in moe_required):
            return "FAIL", "grouped-lowbit cache branch omitted its route or MoE key axes", 0.0

        class Decline(Admit):
            def __call__(self, *_args):
                return 0

        declined = ValidityLib()
        declined.quactlize_ppu_dense_lowbit_config_valid_v1 = Decline()
        try:
            module.write_runtime_cache(td / "invalid.cache", fake_so, "ppu001-test", "Qwen3-32B",
                                       MODELS["Qwen3-32B"], "dense_lowbit", 12, 32, table, inventory,
                                       "0123456789abcdef", declined)
        except ValueError:
            pass
        else:
            return "FAIL", "runtime-cache writer accepted a winner rejected by the library validity query", 0.0
    return "PASS", "measured buckets and winner-only runtime cache; unmeasured M=8 stays absent", 0.0


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


def lint_moe_only_filter():
    """A complete MOE_ONLY tag must cross the cheap shape gate and select its row.

    This is deliberately separate from the winner-selection test above: the regression happens before timing and
    leaves the selection procedure no samples to rank. The fixture compiles the production formatter/matcher and
    includes the old bc-bearing shape as a negative control, so a green result proves it exercises that failure.
    """
    test = ROOT / "tests" / "test_moe_only_filter.py"
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rfE", str(test)],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode == 0:
        return "PASS", "exact and stage-bearing MOE_ONLY tags cross both production filter gates", 0.0
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
                ("boxdry", "build.sh forwards restricted MoE axes and stage list",
                 ("test_lowbit_moe_decode_bench", "SK_QUANT=2", "MOE_FORMATS=i4", "MOE_TM_LIST=16",
                  "MOE_TN_LIST=16", "MOE_WM_LIST=16", "MOE_STAGES=2;12")),
                ("boxdry", "dense persistent A/B target reaches its one-row object graph and host link",
                 ("test_lowbit_dense_persistent_ab", "DENSE_PERSISTENT_AB=1", "TILE_M=64", "TILE_N=128",
                  "WARP_M=32", "WARP_N=32", "STAGES=2", "BENCH_GS=32")),
                ("boxdry", "dense Stream-K target reaches its isolated object graph and host link",
                 ("test_lowbit_dense_streamk_ab", "DENSE_STREAMK_AB=1")),
                ("boxdry", "Q4_K65 Stream-K A/B reaches its isolated object graph and host link",
                 ("test_lowbit_dense_streamk_q4k65_ab", "DENSE_STREAMK_AB=1",
                  "DENSE_STREAMK_Q4K65_AB=1", "BENCH_GS=32")),
                ("boxdry", "full dense Stream-K sweep links all 167 private row units",
                 ("test_lowbit_dense_streamk_sweep", "BENCH_GS=32")),
                ("boxdry", "dense Marlin target reaches its isolated DP/Stream-K/Marlin object graph and host link",
                 ("test_lowbit_dense_marlin_ab", "DENSE_MARLIN_AB=1")),
                ("boxdry", "dense standalone Marlin m8 target reaches its generated object graph and host link",
                 ("test_lowbit_dense_marlin_m8_ab", "DENSE_MARLIN_WK4_AB=1",
                  "DENSE_MARLIN_M8_AB=1", "DENSE_MARLIN_AB=1")),
                ("boxdry", "dense standalone Marlin sweep links every admitted private row unit",
                 ("test_lowbit_dense_marlin_standalone_sweep",
                  "DENSE_MARLIN_STANDALONE_SWEEP=1", "BENCH_GS=128")),
                ("boxdry", "generated-unit undefined reference is rejected by the real host link",
                 ("test_lowbit_dense_streamk_ab", "DENSE_STREAMK_AB=1",
                  "BOX_DRYRUN_PLANT_LINK_FAILURE=1", "BOX_DRYRUN_EXPECT_LINK_FAILURE=1")),
                ("boxdry-negative", "a real generated-unit link defect is FAIL, never SKIP",
                 ("test_lowbit_dense_streamk_ab", "DENSE_STREAMK_AB=1",
                  "BOX_DRYRUN_PLANT_LINK_FAILURE=1")),
                ("boxdry", "dense Marlin full table links every private generated device unit",
                 ("test_lowbit_dense_marlin_sweep", "DENSE_MARLIN_SWEEP=1", "BENCH_GS=128")),
                ("boxdry", "grouped Stream-K target reaches its isolated object graph and host link",
                 "test_moe_grouped_streamk"),
                ("asan", "preprocessing chain under ASAN", None),
                ("pytest", "torch op tests", None),
                ("lint", "duplicate unroll directives (hgcc-only error)", lint_unroll),
                ("lint", "PPU asm uses device-pass architecture guard", lint_ppu_asm_device_guard),
                ("lint", "absolute paths name this repo dir, not a renamed one", lint_stale_repo_path),
                ("lint", "names used before they exist (device-only tests get no other flow check)", lint_undefined_names),
                ("lint", "every ggml.h quant type is classified, in scope or out", lint_gguf_coverage),
                ("lint", "the C++ and Python selection procedures agree on planted data", lint_selection_agrees),
                ("lint", "an exact MOE_ONLY tag crosses both shape and row filters", lint_moe_only_filter),
                ("lint", "box-built sources stay in the PPU-portable subset", lint_ppu_portability),
                ("lint", "emitted bench flags are ones the bench parses", lint_fixture_flags),
                ("lint", "every INBOX item is consumed, or is explained by a call in flight", lint_inbox_delivered),
                ("lint", "dense/grouped tactic names alias one generator and route output agrees", lint_tactic_spaces_agree),
                ("lint", "dense/grouped mixed policy descriptor parity fires on planted drift", lint_mixed_policy_parity_fires),
                ("lint", "l114_scale_copy_coverage: uncapped layout fails the shared witness", lint_scale_copy_coverage_fires),
                ("lint", "fold metadata publication uses one proved physical owner", lint_fold_metadata_single_owner),
                ("lint", "a device-compiler SKIP only guards checks that reach a compiler", lint_device_probe_scope),
                ("lint", "Stream-K minimum policy rejects an empty K stripe", lint_streamk_min_zero_fires),
                ("lint", "all mixed collectives use one stage-ring driver", lint_mixed_pipeline_shared),
                ("lint", "the committed dense tactic table exactly regenerates from its stamped sources", lint_dense_tactic_table_current),
                ("lint", "offline tactic buckets never extrapolate beyond measured M", lint_tactic_buckets_do_not_extrapolate),
                ("lint", "the sample writer and the sample reader agree on the bytes", lint_attempt_record_roundtrip),
                ("lint", "the dense route admits the geometries we claim, and its controls still fail", lint_route_admits),
                ("lint", "the ctypes config mirror matches its C header field for field", lint_config_abi_matches_header),
                ("lint", "no tactic choice can change the offline layout", lint_tactic_cannot_change_offline_layout),
                ("lint", "actlize carries no quactlize symbol and no unlisted file change", lint_actlize_pristine),
                ("lint", "every owned build switch has a recorded build route", lint_switch_macros),
                ("lint", "advertised build inputs have a build.sh/CMake route", lint_build_advice),
                ("lint", "advertised MoE restrictions change generated code", lint_moe_build_knobs),
                ("lint", "dense and MoE consume one named measurement layer", lint_bench_measurement_shared),
                ("lint", "MoE events bracket only gemm.run and retain the host-wall audit", lint_moe_event_timing),
                ("lint", "dense Stream-K shares worker/K decomposition and resets locks before timing", lint_dense_streamk_contract),
                ("lint", "Q4_K65 normal admission precedes an exact same-row forced Stream-K A/B", lint_dense_streamk_q4k65_target),
                ("lint", "persistent DP absolute grids retain exact whole-tile ownership", lint_dense_persistent_grid_contract),
                ("lint", "full dense Stream-K sweep has an exact denominator and no DP fallback", lint_dense_streamk_sweep_target),
                ("lint", "dense TM8 family and M<8 default share one exhaustive shipping authority", lint_dense_shipping_tm8),
                ("lint", "fixed Split-K actual-last/publish diagnostic is ordered, reusable, and source-bound", lint_fixed_splitk_last_arriver),
                ("lint", "fixed Split-K separate and completion modes reach the same production mainloop", lint_fixed_splitk_production_type),
                ("lint", "dense Split-K sweep binds its oracle and fused exact canary", lint_dense_splitk_sweep_contract),
                ("lint", "fixed Split-K W4 selector and exported C ABI fail closed", lint_dense_splitk_shipping_selector),
                ("lint", "fixed Split-K one-plane shipping formats retain exact types", lint_dense_splitk_oneplane_formats),
                ("lint", "fixed Split-K multiformat metadata ABIs retain exact types", lint_dense_splitk_multiformat_types),
                ("lint", "Stream-K tail scan covers attributed zero, medium, and extreme waves", lint_streamk_tail_plan),
                ("lint", "dense Stream-K tail partitions exhaust the committed BPC domain", lint_streamk_tail_oracle),
                ("lint", "PPU chunked GDN recurrence and WY algebra agree on every admitted tail", lint_ppu_chunked_gdn_oracle),
                ("lint", "PPU chunked GDN exact device body instantiates with fail-closed geometry", lint_ppu_chunked_gdn_device_compile),
                ("lint", "PPU chunked GDN public-ABI box harness is exactness-bound and locally compilable", lint_ppu_chunked_gdn_abi_harness),
                ("lint", "PPU chunked GDN global-dot accumulator destinations are exact-once", lint_ppu_chunked_gdn_global_dot_ownership),
                ("lint", "dense Marlin keeps K-fast stripes, reverse q locks, and the scheduler-owned grid", lint_dense_marlin_contract),
                ("lint", "dense Marlin exhausts the declared deployment domain without sampling", lint_dense_marlin_exhaustive),
                ("lint", "the real dense Marlin Cfg emits the proved raw-shape and unit seams", lint_dense_marlin_codegen),
                ("lint", "dense Marlin sweep has private sources, exact capable cohorts, and distinct provenance", lint_dense_marlin_sweep_contract),
                ("lint", "Marlin A2 recovers every formerly filtered row with an exact cohort census", lint_dense_marlin_rejection_census),
                ("lint", "each released Marlin cohort compiles and an inexact explicit cohort reds", lint_dense_marlin_rejected_cohorts),
                ("lint", "dense Marlin verification uses the exact K0 output cohort", lint_dense_marlin_output_owners),
                ("lint", "folded dense artifacts select an exact versioned reader and F2-to-F1 reds", lint_dense_arrangement_abi),
                ("lint", "BC arrangement maps exhaustively equal the production writer and F2-to-F1 reds", lint_bc_arrangement_layout),
                ("lint", "packed qtypes select distinct format binaries under fail-closed path precedence", lint_format_loader),
                ("lint", "Q3/Q6 external fp16 zero is scale-derived with independent anchors", lint_q36_zero_redundancy),
                ("lint", "grouped Marlin preserves ragged q and all mixed-input collective families", lint_grouped_marlin_contract),
                ("lint", "grouped Marlin exhausts every committed format/shape tuple", lint_grouped_marlin_exhaustive),
                ("lint", "GEMV reference fixtures expose expert routing and packed pitch", lint_gemv_perf_authority),
                ("lint", "GEMV finite tactic axes and generated units share one authority", lint_gemv_tactic_space),
                ("lint", "GEMV exact-CtaM sweep leaves adaptive production dispatch unchanged", lint_gemv_exact_ctam),
                ("lint", "GEMV timing records one raw device-event pair per launch", lint_gemv_event_protocol),
                ("lint", "GEMV bounded driver resumes without path or raw-prefix poisoning", lint_gemv_sweep_driver),
                ("lint", "GEMV manifest, exact units and raw writer preserve one identity", lint_gemv_sweep_integration),
                ("lint", "box results are judged only by the sealed preregistration", lint_box_run_adjudicator),
                ("lint", "frozen box runners produce their adjudicator evidence", lint_box_runner_bundle_contract),
                ("lint", "box identity is measured atomically and never guessed", lint_box_identity_probe),
                ("lint", "L143 WK1 admission is executable committed evidence with planted reds", lint_l143_wk4_committed_evidence),
                ("lint", "GEMV production converter reports normalized sm_120 extraction codegen", lint_gemv_lop3_codegen),
                ("lint", "Q4_K PDF reconstruction preserves pack, topology and raw-event evidence", lint_q4k_pdf_ab_contract),
                ("lint", "grouped Stream-K preserves q locks, worker/K decomposition, and timing", lint_grouped_streamk_contract),
                ("lint", "l122_streamk_fixup_cohort contract pins the exact 64/128-thread CTA cohort", lint_streamk_fixup_cohort),
                ("lint", "l124 predicates every shipped FP32 accumulator residue and preserves S1-4", lint_fp32_residue_fixup),
                ("lint", "FP32 residue predicates scalar fixup accesses without predicating locks", lint_fp32_residue_contract),
                ("lint", "syntax baselines and live SYNTAX sources match", lint_syntax_inventory),
                ("lint", "m8n16 G2 replays the historical bad index on the production x4 payload", lint_m8n16_g2_contract),
                ("lint", "l125 exhausts all 256 G5 zero-plane addresses through the production CuTe map", lint_grouped_metadata_layout),
                ("lint", "G5 production and l125 share one exact typed metadata-layout seam", lint_grouped_metadata_layout_contract),
                ("lint", "l130 exhausts all 256 G5 B experts with independent byte-map anchors", lint_grouped_b_idprobe_contract),
                ("lint", "G5 maps caller slots through real experts before naming oracles and output rows", lint_g5_harness_slot_contract),
                ("lint", "caller dS changes all three shipping S/Z metadata address maps", lint_metadata_stride_contract),
                ("lint", "mixed-input outer bases and residues honor caller/logical coordinates", lint_mixed_argument_contract),
                ("lint", "all mixed-input kernels derive output M/N residue from logical CTA tiles", lint_mixed_logical_m_residue),
                ("lint", "sub-byte logical codes and packed bytes never share an unlabeled owner", lint_subbyte_units),
                ("lint", "ppu001 plain LDSM fails in C++ before its unproved assembler path", lint_plain_ldsm_failclosed),
                ("lint", "quactlize_extensions adds to actlize rather than redefining it", lint_extension_additive),
                ("lint", "every file naming a quactlize type reaches its defining header", lint_owned_symbol_includes),
                ("lint", "every listed GGUF format has the collective its row implies", lint_format_table_buildable),
                ("lint", "includes the CMake generator writes into generated sources resolve", lint_generated_include_edges),
                ("lint", "dense table parser batches the committed rows and fails closed", lint_dense_unit_generator),
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
            if isinstance(payload, tuple):
                args = list(payload)
            else:
                legacy = payload.split()
                args = [legacy[0]] + ([" ".join(legacy[1:])] if len(legacy) > 1 else [])
            rc, log, dt = run(["bash", str(ROOT / "ci/box_build_dryrun.sh")] + args)
            last = [l for l in log.splitlines() if l.strip()]
            st = {0: "PASS", 2: "SKIP"}.get(rc, "FAIL")
            msg = next((l.strip() for l in last if "[ok]" in l or "[FAIL]" in l or "[SKIP]" in l),
                       last[-1].strip() if last else f"exit {rc}")
            return st, msg, dt
        if kind == "boxdry-negative":
            args = list(payload)
            rc, log, dt = run(["bash", str(ROOT / "ci/box_build_dryrun.sh")] + args)
            lines = [line.strip() for line in log.splitlines() if line.strip()]
            if rc == 2:
                why = next((line for line in lines if "[SKIP]" in line), "environment cannot run boxdry")
                return "SKIP", why, dt
            if rc != 1:
                return "FAIL", f"planted link defect returned {rc}, expected the raw FAIL status 1", dt
            if any("[SKIP]" in line for line in lines):
                return "FAIL", "planted link defect was reported as an environment SKIP", dt
            if not any("[FAIL]" in line for line in lines):
                return "FAIL", "planted link defect returned 1 without an explicit FAIL verdict", dt
            if not any("qz_boxdry_generated_unit_anchor" in line for line in lines):
                return "FAIL", "negative arm failed for a reason other than the planted cross-TU symbol", dt
            return "PASS", "planted cross-TU undefined symbol was classified as FAIL (raw rc=1), never SKIP", dt
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
    exclusive = [i for i in items if i[0] in ("boxdry", "boxdry-negative")]
    parallel = [i for i in items if i[0] not in ("boxdry", "boxdry-negative")]
    got = {}
    with ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4))) as pool:
        futures = {id(i): pool.submit(run_one, i) for i in parallel}
        for i in exclusive:                       # serialised, and overlapped with the pool's work
            got[id(i)] = run_one(i)
        for k, f in futures.items():
            got[k] = f.result()
    results = [got[id(i)] for i in items]

    failures = []
    skips = []
    passes = 0
    for (kind, name, _), (st, msg, dt) in zip(items, results):
        mark = {"PASS": "ok  ", "FAIL": "FAIL", "BUILD": "BLD!", "MISSING": "MISS", "SKIP": "skip"}[st]
        print(f"  [{mark}] {kind:<8} {name:<44} {dt:5.1f}s  {msg[:88]}")
        if st == "PASS":
            passes += 1
        elif st == "SKIP":
            skips.append(f"{kind}/{name}: {msg}")
        else:
            failures.append(f"{kind}/{name}: {msg}")
    print(f"  wall clock {time.time() - t0:.0f}s")

    print(f"\n== PASS {passes} / SKIP {len(skips)} / FAIL {len(failures)} / TOTAL {len(items)} ==")
    for skipped in skips:
        print(f"   SKIPPED {skipped}")
    for f in failures:
        print(f"   FAILED  {f}")
    if a.strict and skips:
        print(f"   STRICT  {len(skips)} SKIP verdict(s) make this invocation non-green; classifications stay SKIP")
    return 1 if failures or (a.strict and skips) else 0


if __name__ == "__main__":
    sys.exit(main())
