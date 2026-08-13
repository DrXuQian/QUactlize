#!/usr/bin/env python3
"""Contract for the isolated classic-aligned Marlin PPU stack.

The original version of this gate authorized a WarpK compatibility branch in
the generic mixed-input collective.  That architecture has been retired: the
classic format, load/dequant cadence, scheduler and cooperative now live in
four standalone Marlin files.  This gate proves both halves of that boundary:
the standalone target is wired end to end, and the deleted compatibility seam
cannot silently grow back into generic code.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PATHS = {
    "cmake": ROOT / "quactlize/csrc/CMakeLists.txt.in",
    "unit": ROOT / "benchmarks/lowbit_dense_unit.inc",
    "bench": ROOT / "benchmarks/test_lowbit_dense_bench.cu",
    "format": ROOT / "quactlize/include/marlin_format_ppu.hpp",
    "standalone_tactic": ROOT / "quactlize/include/marlin_tactic_space_ppu.hpp",
    "collective": ROOT / (
        "quactlize/include/quactlize_extensions/cutlass/gemm/collective/"
        "marlin_collective_ppu.hpp"
    ),
    "scheduler": ROOT / (
        "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/"
        "marlin_scheduler_ppu.hpp"
    ),
    "kernel": ROOT / (
        "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/"
        "marlin_kernel_ppu.hpp"
    ),
    "builder": ROOT / (
        "quactlize/include/quactlize_extensions/cutlass/gemm/collective/"
        "builders/quactlize_mma_builder.inl"
    ),
    "generic_collective": ROOT / (
        "quactlize/include/quactlize_extensions/cutlass/gemm/collective/"
        "quactlize_mma_mixed_input.hpp"
    ),
    "generic_converter": ROOT / (
        "quactlize/include/quactlize_extensions/cutlass/"
        "quactlize_mix_gemm_convert.h"
    ),
    "generic_kernel": ROOT / (
        "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/"
        "ppu_aiu_gemm_mixed_input_marlin.hpp"
    ),
    "tactic": ROOT / "quactlize/include/ppu_tactic_space.hpp",
    "policy": ROOT / "quactlize/include/ppu_mixed_policy.hpp",
    "xplane": ROOT / "quactlize/include/xplane_offline.hpp",
    "aggregate": ROOT / "dev/fold_derivation/run_l143_dense_marlin_wk4_target.sh",
    "l167": ROOT / "dev/fold_derivation/run_l167_classic_marlin_format.sh",
    "l168": ROOT / "dev/fold_derivation/run_l168_marlin_pipeline_trace.sh",
    "l169": ROOT / "dev/fold_derivation/run_l169_standalone_marlin_unit.sh",
    "l170": ROOT / "dev/fold_derivation/run_l170_standalone_marlin_scheduler.sh",
    "box": ROOT / "tools/run_dense_marlin_wk4_box.sh",
}


def require(text: str, token: str, owner: str, bad: list[str]) -> None:
    if token not in text:
        bad.append(f"{owner}: missing {token!r}")


def forbid(text: str, token: str, owner: str, bad: list[str]) -> None:
    if token in text:
        bad.append(f"{owner}: retired compatibility token returned: {token!r}")


def audit(files: dict[str, str]) -> list[str]:
    bad: list[str] = []

    cmake = files["cmake"]
    old_begin = cmake.find("set(_DENSE_MARLIN_ARTIFACT_TK 64)")
    new_begin = cmake.find("# Classic-aligned Marlin decode target.")
    new_end = cmake.find("# Root-cause cross-check", new_begin)
    if old_begin < 0 or new_begin <= old_begin or new_end <= new_begin:
        bad.append("cmake: cannot isolate historical and standalone target blocks")
        old = new = ""
    else:
        old = cmake[old_begin:new_begin]
        new = cmake[new_begin:new_end]
    for token in (
        "set(_DENSE_MARLIN_WN 32)", "set(_DENSE_MARLIN_ST 3)",
        "test_lowbit_dense_marlin_ab",
    ):
        require(old, token, "historical target", bad)
    forbid(old, "DENSE_AB_WARP_K", "historical target", bad)
    for token in (
        "set(_DENSE_MARLIN_WK4_TM 16)",
        "set(_DENSE_MARLIN_WK4_TN 128)",
        "set(_DENSE_MARLIN_WK4_TK 128)",
        "set(_DENSE_MARLIN_WK4_WM 16)",
        "set(_DENSE_MARLIN_WK4_WN 64)",
        "set(_DENSE_MARLIN_WK4_WARP_K 32)",
        "set(_DENSE_MARLIN_WK4_ST 4)",
        "test_lowbit_dense_marlin_wk4_ab",
        "DENSE_MARLIN_WK4_AB=1",
    ):
        require(new, token, "standalone target", bad)

    unit = files["unit"]
    for token in (
        "using StandaloneCfg = StandaloneMarlinCfg<",
        "using G = typename StandaloneCfg::MarlinGemm;",
        "Kernel::IsStandaloneMarlin",
        "typename StandaloneCfg::MarlinMain",
        'return run<G>(options, dense_tactic(cfg), "marlin");',
    ):
        require(unit, token, "generated standalone wrapper", bad)
    standalone_arm = unit.split("#if defined(DENSE_MARLIN_WK4_AB)", 1)[-1].split(
        "#else", 1
    )[0]
    for token in ("StreamKGemm", "PersistentGemm", "::Gemm;"):
        forbid(standalone_arm, token, "generated standalone wrapper", bad)

    bench = files["bench"]
    for token in (
        "struct StandaloneMarlinCfg",
        '#include "marlin_tactic_space_ppu.hpp"',
        "marlin_tactics_ppu::MarlinTacticPPU Tactic",
        "marlin_tactics_ppu::admitted(Tactic)",
        "MarlinCollectivePPU<", "MarlinKernelPPU<",
        "quactlize::marlin::pack_biased_int4_bytes(",
        "quactlize::marlin::unpack_biased_int4_bytes(",
        "quactlize::marlin::permute_gs128_scales(",
        "artifact=classic-marlin-u32 scale=classic-gs128-permuted",
        "test_lowbit_dense_marlin_wk4_ab is Marlin-only: pass --marlin",
        "return final_result.passed ? 0 : 1;",
    ):
        require(bench, token, "standalone benchmark route", bad)
    for token in ("place_derived_warp_k", "recover_derived_warp_k"):
        forbid(bench, token, "standalone benchmark route", bad)

    fmt = files["format"]
    for token in (
        "pack_biased_int4_bytes", "unpack_biased_int4_bytes",
        "permute_gs128_scales", "unpermute_gs128_scales",
    ):
        require(fmt, token, "Marlin format", bad)

    standalone_tactic = files["standalone_tactic"]
    for token in (
        "struct MarlinTacticPPU", "kMarlinClassicReferencePPU",
        "constexpr bool admitted(", "static_assert(cartesian_size() == 60000",
    ):
        require(token=token, text=standalone_tactic,
                owner="standalone tactic authority", bad=bad)

    collective = files["collective"]
    for token in (
        "class MarlinCollectivePPU", "MarlinCpAsyncLoadPolicyPPU",
        "cute::is_same_v<LoadPolicy, MarlinCpAsyncLoadPolicyPPU>",
        "cute::Layout<cute::Shape<cute::_1, cute::_2, cute::_4>>",
        "WarpM == 16 && WarpN == 64 && WarpK == 32",
        "Stages == 4 && GroupSize == 128 && Threads == 256",
        "sizeof(SharedStorage) == 50176", "dequantize_biased_int4",
        "for (int pipe = 0; pipe < Stages;)",
    ):
        require(collective, token, "Marlin collective", bad)
    for token in (
        '#include "quactlize_mma_mixed_input', '#include "quactlize_mma_builder',
        '#include "quactlize_mix_gemm_convert', '#include "xplane_offline',
        "switch (compute_warp_k)",
    ):
        forbid(collective, token, "Marlin collective", bad)

    scheduler = files["scheduler"]
    for token in (
        "class MarlinSchedulerPPU", "uint32_t blocks_per_cu = 1",
        "sizeof(WorkTileInfo) == 20", "uint32_t peer_idx = 0",
        "work.has_flag(WorkFlag::Split)",
        "return work.is_valid() && work.N_idx >= 0 ? int(work.N_idx) : -1;",
        "p.iters_per_block_ <= p.k_tiles_per_output_",
        "Barrier::wait_eq", "Barrier::arrive_inc",
        "PeerReleaseAction::Reset", "p.locks_[lock] = BarrierType(0)",
        "sizeof(Params) == 40",
    ):
        require(scheduler, token, "Marlin scheduler", bad)
    for token in (
        "BlockStripedReduce", "Barrier::wait_eq_reset",
        "int32_t M_idx =", "int32_t L_idx =",
        "uint32_t k_tiles_per_output =", "uint32_t slice_idx =",
        "uint32_t output_tile_idx =", "uint32_t lock_idx =",
        "uint32_t block_idx =", "bool valid =",
    ):
        forbid(scheduler, token, "Marlin scheduler", bad)

    kernel = files["kernel"]
    for token in (
        "class MarlinKernelPPU", "IsStandaloneMarlin = true",
        "MaxThreadsPerBlock == 256 && WarpKCohorts == 4",
        "OutputThreads == 64", "for (int step = red_off; step > 0; step /= 2)",
        "thread_block_reduce(accum, shared);",
        "TileScheduler::acquire_peer_turn", "global_handoff(accum",
        "TileScheduler::release_peer_turn", "write_result(accum",
    ):
        require(kernel, token, "Marlin kernel", bad)
    for token in ("BlockStripedReduce", "TileScheduler::fixup"):
        forbid(kernel, token, "Marlin kernel", bad)

    generic_forbidden = {
        "builder": ("requestedWarpK", "WarpOnK"),
        "generic_collective": (
            "WarpKCohorts", "convert_int4_two_source",
            "convert_int4_shadow_source", "compute_warp_k",
        ),
        "generic_converter": ("emit_value",),
        "generic_kernel": ("ReductionScratchElements", "OutputThreads * WarpKCohorts"),
        "tactic": ("WarpKDoesNotDivideTile", "WarpKUnsupportedFormat", "int warp_k;"),
        "policy": ("cute::size<2>(WarpShape{})},",),
        "xplane": ("plane_map_warp_k", "place_derived_warp_k", "recover_derived_warp_k"),
    }
    for owner, tokens in generic_forbidden.items():
        for token in tokens:
            forbid(files[owner], token, owner, bad)

    aggregate = files["aggregate"]
    for token in (
        "run_l167_classic_marlin_format.sh",
        "run_l168_marlin_pipeline_trace.sh",
        "run_l169_standalone_marlin_unit.sh",
        "run_l170_standalone_marlin_scheduler.sh",
        "check_dense_marlin_wk4_target.py",
    ):
        require(aggregate, token, "standalone aggregate runner", bad)
    for token in ("run_l140_warpk_tactic_axis.sh", "run_l142_twosource_consumer_compile.sh",
                  "run_l155_wk4_indexed_converter.sh"):
        forbid(aggregate, token, "standalone aggregate runner", bad)

    l170 = files["l170"]
    for token in (
        "LEGACY_44_DESCRIPTOR RECOMPUTED_PREDICATE",
        "-DL170_PLANT_${compile_plant}=1",
        "hot=20B legacy-44B=RED local-lock=RED recomputed-predicate=RED",
    ):
        require(l170, token, "L170 descriptor runner", bad)

    return bad


def main() -> int:
    missing = [str(path) for path in PATHS.values() if not path.is_file()]
    if missing:
        for path in missing:
            print(f"[dense-marlin-wk4] FAIL: missing {path}", file=sys.stderr)
        return 1
    files = {name: path.read_text() for name, path in PATHS.items()}
    bad = audit(files)

    plants = (
        ("unit", "wrapper-back-to-generic", "using StandaloneCfg = StandaloneMarlinCfg<",
         "using StandaloneCfg = Cfg<"),
        ("bench", "format-back-to-xplane", "quactlize::marlin::pack_biased_int4_bytes(",
         "xplane::place_derived_warp_k("),
        ("bench", "cfg-bypasses-tactic-authority",
         "static_assert(marlin_tactics_ppu::admitted(Tactic),",
         "static_assert(true,"),
        ("collective", "layout-loses-k-cohorts",
         "cute::Layout<cute::Shape<cute::_1, cute::_2, cute::_4>>",
         "cute::Layout<cute::Shape<cute::_1, cute::_2, cute::_1>>"),
        ("collective", "load-policy-switches-to-aiu",
         "cute::is_same_v<LoadPolicy, MarlinCpAsyncLoadPolicyPPU>",
         "cute::is_same_v<LoadPolicy, MarlinAiuLoadPolicyPPU>"),
        ("kernel", "reduction-becomes-flat", "step > 0; step /= 2",
         "step > 0; step = 0"),
        ("kernel", "output-cohort-becomes-cta", "OutputThreads == 64",
         "OutputThreads == 256"),
        ("scheduler", "lock-becomes-local",
         "return work.is_valid() && work.N_idx >= 0 ? int(work.N_idx) : -1;",
         "return work.is_valid() && work.N_idx >= 0 ? int(work.N_idx & 15) : -1;"),
        ("scheduler", "legacy-44-byte-descriptor",
         "sizeof(WorkTileInfo) == 20", "sizeof(WorkTileInfo) == 44"),
        ("scheduler", "split-predicate-is-recomputed",
         "return work.is_valid() && work.has_flag(WorkFlag::Split);",
         "return work.is_valid() && !(is_first_peer(work) && is_final_peer(work));"),
        ("builder", "generic-k-cohort-returns",
         "cute::Layout<Shape<WarpOnM, WarpOnN, _1>>>,",
         "cute::Layout<Shape<WarpOnM, WarpOnN, WarpOnK>>>,"),
        ("generic_collective", "generic-two-source-returns", "/// Utilities to transform B.",
         "int convert_int4_two_source;\n  /// Utilities to transform B."),
        ("xplane", "generic-warpk-api-returns", "namespace xplane {",
         "namespace xplane {\nint plane_map_warp_k;"),
    )
    for owner, label, old, new in plants:
        if files[owner].count(old) != 1:
            bad.append(f"cannot plant {label}: anchor count={files[owner].count(old)}")
            continue
        planted = dict(files)
        planted[owner] = planted[owner].replace(old, new)
        if not audit(planted):
            bad.append(f"plant {label} was not rejected")

    if bad:
        for item in bad:
            print(f"[dense-marlin-wk4] FAIL: {item}", file=sys.stderr)
        return 1
    print(
        "[dense-marlin-wk4] PASS: standalone format/collective/scheduler/kernel "
        "wired; standalone tactic authority consumed; generic WK4 compatibility "
        "absent; thirteen structural plants rejected"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
