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
    "load": ROOT / (
        "quactlize/include/quactlize_extensions/cutlass/gemm/collective/"
        "marlin_load_ppu.hpp"
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
    "m8_box": ROOT / "tools/run_dense_marlin_m8_acu_box.sh",
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
        "test_lowbit_dense_marlin_m8_ab",
        "DENSE_MARLIN_M8_AB=1",
        "set(_DENSE_MARLIN_M8_SCAFFOLD_TM ${_DENSE_MARLIN_WK4_TM})",
        "set(_DENSE_MARLIN_M8_SCAFFOLD_WM ${_DENSE_MARLIN_WK4_WM})",
        "-DDENSE_AB_TM=${_DENSE_MARLIN_M8_TM}",
        "-DDENSE_AB_WM=${_DENSE_MARLIN_M8_WM}",
        "-DTILE_M=${_DENSE_MARLIN_M8_SCAFFOLD_TM}",
        "-DWARP_M=${_DENSE_MARLIN_M8_SCAFFOLD_WM}",
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
    require(
        standalone_arm,
        "using StandaloneCfg = StandaloneMarlinCfg<\n"
        "      GroupSize, TM, TN, TK, WM, WN, ST, DENSE_AB_WARP_K>;",
        "generated fixed-WK4 standalone wrapper",
        bad,
    )
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
        "standalone Marlin m8/m16 target is Marlin-only: pass --marlin",
        "if (options.marlin_profile_subject_only) {",
        "if (!host_exact_profile) {",
        "block_B.copy_to_host(tensor_B.host_data());",
        "[dense marlin ACU subject-only] instruction=m%dn16k16",
        "subject_launches=1 device_reference=0 lock_fingerprints=0",
        "return final_result.passed ? 0 : 1;",
    ):
        require(bench, token, "standalone benchmark route", bad)
    profile_begin = bench.find("if (options.marlin_profile_subject_only) {")
    profile_end = bench.find("\n#endif", profile_begin)
    if profile_begin < 0 or profile_end <= profile_begin:
        bad.append("standalone benchmark route: cannot isolate ACU subject-only arm")
        profile_arm = ""
    else:
        profile_arm = bench[profile_begin:profile_end]
    for token in (
        "if constexpr (kStandaloneMarlin) {",
        "Gemm::GemmKernel::InstructionM",
        "--marlin-profile-subject-only reached a non-standalone kernel",
        "result.passed = false;",
    ):
        require(profile_arm, token, "standalone ACU type guard", bad)
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
        "cute::Int<WarpOnN>, cute::Int<WarpOnK>",
        "(WarpN == 64 && WarpK == 32)",
        "(WarpN == 128 && WarpK == 16)",
        "TileM == 8 || TileM == 16",
        "static constexpr int AStoredRows = InstructionM == 8 ? 1 : TileM",
        ": ASharedStage == 16) &&",
        "Stages >= 2 && Stages <= 6 &&",
        "GroupSize == 128 && GroupSize % TileK == 0",
        "Threads == 128 || Threads == 256",
        "Stages != 4 ||",
        "sizeof(SharedStorage) == (InstructionM == 8 ? 34816 : 50176)",
        "bool const m_supported = InstructionM == 8 ? m == 1 : (m > 0 && m <= TileM);",
        "dequantize_biased_int4",
        "for (int pipe = 0; pipe < Stages;)",
    ):
        require(collective, token, "Marlin collective", bad)
    for token in (
        '#include "quactlize_mma_mixed_input', '#include "quactlize_mma_builder',
        '#include "quactlize_mix_gemm_convert', '#include "xplane_offline',
        "switch (compute_warp_k)",
    ):
        forbid(collective, token, "Marlin collective", bad)

    load = files["load"]
    for token in (
        "struct FragmentA {", "__half2 value[4];",
        "struct FragmentA8 {", "__half2 value[2];",
        "FragmentAFor = std::conditional_t<InstructionM == 8, FragmentA8, FragmentA>",
        "sizeof(FragmentA8) == 2 * sizeof(uint32_t)",
        "sizeof(FragmentA) == 4 * sizeof(uint32_t)",
    ):
        require(load, token, "Marlin load", bad)
    m16_begin = load.find("CUTLASS_DEVICE void ldmatrix_a_m16(")
    m8_begin = load.find("CUTLASS_DEVICE void ldmatrix_a_m8(")
    dispatch_begin = load.find("template <int InstructionM>", m8_begin)
    if not (0 <= m16_begin < m8_begin < dispatch_begin):
        bad.append("Marlin load: cannot isolate m16/x4 and m8/x2 bodies")
        m16_load = m8_load = ""
    else:
        m16_load = load[m16_begin:m8_begin]
        m8_load = load[m8_begin:dispatch_begin]
    x4 = "ppu.ldmatrix.sync.aligned.m8n8.x4.shared.b16"
    x2 = "ppu.ldmatrix.sync.aligned.m8n8.x2.shared.b16"
    if m16_load.count(x4) != 1 or x2 in m16_load:
        bad.append("Marlin load: m16 is not exactly the unchanged x4 path")
    if ': "=r"(a[0]), "=r"(a[2]), "=r"(a[1]), "=r"(a[3])' not in m16_load:
        bad.append("Marlin load: m16 x4 register permutation drifted")
    if m8_load.count(x2) != 1 or x4 in m8_load:
        bad.append("Marlin load: m8 is not exactly the plain x2 path")
    if ': "=r"(a[0]), "=r"(a[1])' not in m8_load:
        bad.append("Marlin load: m8 no longer publishes exactly two registers")
    if "discarded_" in m8_load:
        bad.append("Marlin load: m8 regained discarded x4 destinations")

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
        "MaxBlocksPerCu = InstructionM == 8 ? 3 : 2",
        "WarpKCohorts == 2 || WarpKCohorts == 4",
        "OutputThreads == uint32_t(CollectiveMainloop::WarpOnN * 32)",
        "for (int step = red_off; step > 0; step /= 2)",
        "thread_block_reduce(accum, shared);",
        "TileScheduler::acquire_peer_turn", "global_handoff(accum",
        "TileScheduler::release_peer_turn", "write_result(accum",
    ):
        require(kernel, token, "Marlin kernel", bad)
    for token in ("BlockStripedReduce", "TileScheduler::fixup"):
        forbid(kernel, token, "Marlin kernel", bad)

    m8_box = files["m8_box"]
    for token in (
        "for bpc in 1 2 3; do",
        "if [ \"$bpc\" -ne 1 ]; then",
        "--marlin-profile-subject-only",
        "report_candidates=()",
        '[ -s "${report_base}.acurep" ] && report_candidates+=("${report_base}.acurep")',
        '[ "${#report_candidates[@]}" -eq 1 ]',
        "subject_launches=1 device_reference=0 lock_fingerprints=0",
        "each report contains one m8 subject launch and no m16 GemmRef",
    ):
        require(m8_box, token, "m8 ACU runner", bad)

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
        ("unit", "wrapper-back-to-generic",
         "using StandaloneCfg = StandaloneMarlinCfg<\n"
         "      GroupSize, TM, TN, TK, WM, WN, ST, DENSE_AB_WARP_K>;",
         "using StandaloneCfg = Cfg<\n"
         "      GroupSize, TM, TN, TK, WM, WN, ST>;"),
        ("bench", "format-back-to-xplane", "quactlize::marlin::pack_biased_int4_bytes(",
         "xplane::place_derived_warp_k("),
        ("bench", "cfg-bypasses-tactic-authority",
         "static_assert(marlin_tactics_ppu::admitted(Tactic),",
         "static_assert(true,"),
        ("bench", "acu-member-lookup-escapes-standalone-type-guard",
         "if (options.marlin_profile_subject_only) {\n    if constexpr (kStandaloneMarlin) {",
         "if (options.marlin_profile_subject_only) {\n    if (kStandaloneMarlin) {"),
        ("collective", "layout-loses-k-cohorts",
         "cute::Int<WarpOnN>, cute::Int<WarpOnK>",
         "cute::Int<WarpOnN>, cute::_1"),
        ("collective", "load-policy-switches-to-aiu",
         "cute::is_same_v<LoadPolicy, MarlinCpAsyncLoadPolicyPPU>",
         "cute::is_same_v<LoadPolicy, MarlinAiuLoadPolicyPPU>"),
        ("collective", "m8-restores-padded-a",
         "static constexpr int AStoredRows = InstructionM == 8 ? 1 : TileM;",
         "static constexpr int AStoredRows = InstructionM == 8 ? 8 : TileM;"),
        ("collective", "m8-broadens-M-admission",
         "bool const m_supported = InstructionM == 8 ? m == 1 : (m > 0 && m <= TileM);",
         "bool const m_supported = m > 0 && m <= TileM;"),
        ("load", "m8-falls-back-to-x4",
         "ppu.ldmatrix.sync.aligned.m8n8.x2.shared.b16",
         "ppu.ldmatrix.sync.aligned.m8n8.x4.shared.b16"),
        ("load", "m8-regains-discarded-destinations",
         ': "=r"(a[0]), "=r"(a[1])\n      : "l"(smem_ptr));',
         ': "=r"(a[0]), "=r"(a[1]), "=r"(discarded_v2), "=r"(discarded_v3)\n      : "l"(smem_ptr));'),
        ("kernel", "reduction-becomes-flat", "step > 0; step /= 2",
         "step > 0; step = 0"),
        ("kernel", "output-cohort-becomes-cta",
         "OutputThreads == uint32_t(CollectiveMainloop::WarpOnN * 32)",
         "OutputThreads == uint32_t(MaxThreadsPerBlock)"),
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
        ("m8_box", "acu-report-resolution-accepts-ambiguity",
         '[ "${#report_candidates[@]}" -eq 1 ]',
         '[ "${#report_candidates[@]}" -ge 1 ]'),
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
        "absent; nineteen structural plants rejected"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
