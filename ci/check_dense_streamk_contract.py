#!/usr/bin/env python3
"""Device-free contract for #107b's dense mixed-input Stream-K seam.

The dangerous failures here all compile and often return plausible numbers: workspace
decomposition and launch can see different worker counts, K slices can restart at zero,
fixup can run with the wrong barrier cohort, or an old turnstile value can deadlock the
next launch.  The PPU box owns numerical proof; this checker pins the source
ordering and the isolated four-warp fixture before a box round trip.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_streamk.hpp"
BENCH = ROOT / "benchmarks/test_lowbit_dense_bench.cu"
UNIT = ROOT / "dev/fold_derivation/test_lowbit_dense_unit.cu"
DISPATCH = ROOT / "benchmarks/lowbit_dense_unit.inc"
CMAKE = ROOT / "quactlize/csrc/CMakeLists.txt.in"
BOX_GATE = ROOT / "tools/run_dense_streamk_107b_box.sh"


def section(text: str, begin: str, end: str) -> str:
    if text.count(begin) != 1 or text.count(end) < 1:
        raise ValueError(f"cannot isolate {begin!r} .. {end!r}")
    return text.split(begin, 1)[1].split(end, 1)[0]


def ordered_once(text: str, anchors: tuple[str, ...], label: str, bad: list[str]) -> None:
    if any(text.count(x) != 1 for x in anchors):
        bad.append(f"{label}: anchors are missing or duplicated")
    elif [text.index(x) for x in anchors] != sorted(text.index(x) for x in anchors):
        bad.append(f"{label}: source order is wrong")


def audit(header: str, bench: str, unit: str, dispatch: str, cmake: str,
          box_gate: str) -> list[str]:
    bad: list[str] = []

    for token in (
        "using TileSchedulerTag = StreamKScheduler;",
        "static_assert(!isGroupProblemShape_v<ProblemShape>",
        "static_assert(MaxThreadsPerBlock == 64u || MaxThreadsPerBlock == 128u",
        "static_assert(TileScheduler::FixupThreadCount == MaxThreadsPerBlock",
        "static_assert(DispatchPolicy::Stages - 1 <= 8",
        "args.scheduler.splits == 1",
        "TileSchedulerParams::ReductionMode::Deterministic",
        "TileSchedulerParams::DecompositionMode::StreamK",
    ):
        if header.count(token) != 1:
            bad.append(f"kernel must contain exactly one {token!r}")
    if "should_perform_separate_reduction" in header:
        bad.append("107b must not wire the disabled separate-reduction path")

    try:
        ws_size = section(header, "  static size_t scheduler_workspace_size", "\npublic:")
        lower = section(header, "  static Params to_underlying_arguments", "\n  static bool can_implement")
        init = section(header, "  static cutlass::Status initialize_workspace", "\n  static dim3 get_grid_shape")
        grid = section(header, "  static dim3 get_grid_shape", "\n  static dim3 get_block_shape")
        device = section(header, "  CUTLASS_DEVICE\n  void operator()", "\n};")
    except ValueError as e:
        return bad + [str(e)]

    # The same virtual worker population must own all four scheduler decisions.
    for label, body, token in (
        ("workspace-size", ws_size, "scheduler_hw_info(args)"),
        ("argument-lowering", lower, "KernelHardwareInfo workers = scheduler_hw_info(args);"),
        ("workspace-init", init, "KernelHardwareInfo workers = scheduler_hw_info(args);"),
        ("launch-grid", grid, "params.scheduler_hw_info"),
    ):
        if body.count(token) != 1:
            bad.append(f"{label} does not consume exactly one shared virtual-worker source")
    if lower.count("shape = scheduler_problem_shape(args.problem_shape)") != 1 or \
       init.count("shape = scheduler_problem_shape(args.problem_shape)") != 1 or \
       ws_size.count("shape = scheduler_problem_shape(args.problem_shape)") != 1:
        bad.append("workspace size/init/lowering do not share the SwapAB-adjusted scheduler shape")
    if "std::min" in grid or "min(" in grid:
        bad.append("Stream-K grid is truncated by logical output tiles")

    if device.count("uint32_t const k_tile_start =") != 1:
        bad.append("device loop does not have exactly one scheduler-owned K-start seam")
        k_path = ""
    else:
        # Ignore the earlier invalid-warpgroup skip, which has its own fetch.
        # From the real K slice onward there must be exactly one final fetch.
        k_path = device.split("uint32_t const k_tile_start =", 1)[1]
    ordered_once(k_path, (
        "TileScheduler::get_work_k_tile_start(work_tile_info)",
        "idx2crd(k_tile_start, shape<2>(gA))",
        "collective_mainloop(params.mainloop",
        "TileScheduler::fixup(",
        "TileScheduler::compute_epilogue(",
        "scheduler.fetch_next_work(work_tile_info)",
    ), "absolute-K mainloop/fixup/final-epilogue", bad)
    if device.count("TileScheduler::get_work_k_tile_count(") != 1:
        bad.append("mainloop does not use exactly one scheduler-owned K-tile count")

    # A0's old bool collapsed harmless half regrouping and a real ownership bug
    # into the same word.  The diagnostic must derive buckets from the lowered
    # scheduler, prove its own (q,k) coverage, and use the exact comparator that
    # produced the original disposition.  tile_peer_range() assumes one SK
    # group in this vendor revision and is therefore explicitly forbidden here.
    diagnostic_tokens = (
        "bool dense_classify_streamk_tiles(",
        "sk.big_groups_ != sk.sk_tiles_ % groups",
        "std::find_if(coverage.begin(), coverage.end(),",
        "[](uint16_t visits) { return visits != 1; }",
        "coverage=exact-once",
        "[dense verify bucket=%s] tiles=%llu outputs=%llu mismatches=%llu",
        "cutlass::relatively_equal(want, got, epsilon, non_zero_floor)",
        "bucket comparator disagrees with device comparator",
        "max_rel_sym=%.9g max_half_ulp=%u nonfinite=%llu",
    )
    for token in diagnostic_tokens:
        if bench.count(token) != 1:
            bad.append(f"A0 bucket diagnostic must contain exactly one {token!r}")
    if bench.count("dense_classify_streamk_tiles(") != 2:
        bad.append("the Stream-K classifier must have exactly one definition and one accepted-arm call")
    if "tile_peer_range(" in bench:
        bad.append("A0 bucket diagnostic uses tile_peer_range(), which is not group-general")
    if "likely the int4-oriented reference" in bench:
        bad.append("a failed Stream-K arm still blames the shared reference without evidence")
    try:
        run_body = section(bench, "template <typename Gemm>\nResult run(",
                           "\nResult run_scale_only")
        accepted = section(run_body, "  Result result;", "\n  // Correctness / Warmup iteration")
    except ValueError as e:
        bad.append(str(e))
    else:
        ordered_once(accepted, (
            "gemm.can_implement(arguments)",
            "gemm.initialize(arguments, workspace.get())",
            "Gemm::GemmKernel::to_underlying_arguments(arguments, workspace.get())",
            "dense_classify_streamk_tiles(verify_partition,",
        ), "only classify an accepted lowered Params", bad)

    try:
        timing = section(bench, "    // One distinct event pair per launch for every arm", "\n#else\n    PpuTimer timer;")
    except ValueError as e:
        bad.append(str(e))
    else:
        ordered_once(timing, (
            "Gemm::GemmKernel::initialize_workspace(",
            "hggcEventRecord(events.start, nullptr)",
            "CUTLASS_CHECK(gemm.run());",
            "hggcEventRecord(events.stop, nullptr)",
            "hggcDeviceSynchronize()",
            "hggcEventElapsedTime(&elapsed_ms, events.start, events.stop)",
        ), "per-launch lock-reset/event interval", bad)
        if timing.count("for (int iter = 0; iter < options.iterations; ++iter)") != 2:
            bad.append("Stream-K target must have one record loop and one post-sync query loop")
    try:
        warmup = section(bench, "  // Create the whole pool before warmup", "\n  // Check if output from kernel")
    except ValueError as e:
        bad.append(str(e))
    else:
        ordered_once(warmup, (
            "DenseKernelEventBatch streamk_events(",
            "hggcEventRecord(streamk_events.at(0).start, nullptr)",
            "CUTLASS_CHECK(gemm.run());",
            "hggcEventRecord(streamk_events.at(0).stop, nullptr)",
            "hggcDeviceSynchronize()",
        ), "precreated event pool/instrumented warmup", bad)
    for token in (
        "actual=%s real_cu=%d ctas_per_cu=%d",
        "params.scheduler_hw_info.cu_count == int(workers)",
        "witness[0] == 8 && witness[1] == 1 && witness[2] == 0",
        "outputs=%zu bad=%d bitdiff=%d",
        "[dense kernel-span-upper]",
        "spread=(max-min)/mean=",
        "distinct-event-pairs=%zu warmup-event-pairs=1 includes-launch-idle=1",
        "MBU=N/A",
        "StreamK partial-C traffic is per-tile and not yet surfaced",
        "return final_result.passed ? 0 : 1;",
    ):
        if bench.count(token) != 1:
            bad.append(f"bench must contain exactly one {token!r}")
    exact_shape = (
        "options.m != 64 || options.n != 128 || options.k != 4352 || options.l != 1 ||\n"
        "       options.g != 128 || std::abs(options.alpha - 0.75f) > 1.0e-7f ||\n"
        "       std::abs(options.beta - 0.5f) > 1.0e-7f"
    )
    if bench.count(exact_shape) != 1:
        bad.append("the dyadic 64x128x4352/gs128/alpha=.75/beta=.5 gate is not exact")

    unit_row = "X(lowbit_dense_streamk_probe,64,128,64,64,32,2,0)"
    if unit.count(unit_row) != 1:
        bad.append("local unit does not instantiate the isolated 128-thread Stream-K row")
    if dispatch.count("using G = typename Cfg<GroupSize, TM, TN, TK, WM, WN, ST>::StreamKGemm;") != 1:
        bad.append("generated unit dispatch does not select the Stream-K named kernel")
    for token in (
        "test_lowbit_dense_streamk_ab",
        "set(_DENSE_SK_TM 64)", "set(_DENSE_SK_TN 128)", "set(_DENSE_SK_TK 64)",
        "set(_DENSE_SK_WM 64)", "set(_DENSE_SK_WN 32)", "set(_DENSE_SK_ST 2)",
        "DENSE_STREAMK_AB=1 BENCH_GS=128",
    ):
        if cmake.count(token) < 1:
            bad.append(f"isolated CMake target is missing {token!r}")

    for token in (
        "run_diagnostic_case 'A0 Stream-K diagnostic'",
        "require_verify_buckets 'A0 non-persistent control'",
        "require_verify_buckets 'A0 serial-persistent control'",
        "A0 correctness is the printed disposition, not this script exit",
        "coverage=exact-once",
    ):
        if box_gate.count(token) != 1:
            bad.append(f"box gate must contain exactly one {token!r}")
    if 'if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]' not in box_gate:
        bad.append("box gate does not preserve a complete rc=1 numerical diagnostic")
    if "run_case 'A0 Stream-K'" in box_gate:
        bad.append("box gate still drops the failing A0 Stream-K diagnostic")
    return bad


def main() -> int:
    texts = [p.read_text() for p in (HEADER, BENCH, UNIT, DISPATCH, CMAKE, BOX_GATE)]
    bad = audit(*texts)
    if bad:
        print("[dense-streamk-contract] FAIL: " + "; ".join(bad))
        return 1

    # Each plant is valid-enough source text whose failure would otherwise be silent.
    plants = [
        (0, "static_assert(TileScheduler::FixupThreadCount == MaxThreadsPerBlock",
         "static_assert(TileScheduler::FixupThreadCount != MaxThreadsPerBlock",
         "exact fixup cohort"),
        (0, "params.scheduler_hw_info, args);", "params.real_hw_info, args);",
         "shared launch-worker source"),
        (0, "idx2crd(k_tile_start, shape<2>(gA))", "idx2crd(0, shape<2>(gA))",
         "absolute K start"),
        (1, "Gemm::GemmKernel::initialize_workspace(\n            arguments, workspace.get(), /*stream=*/nullptr));\n      }\n      auto& events = streamk_events.at(iter + 1);\n      CUTLASS_PPU_CHECK(hggcEventRecord(events.start, nullptr));",
         "auto& events = streamk_events.at(iter + 1);\n      CUTLASS_PPU_CHECK(hggcEventRecord(events.start, nullptr));\n      Gemm::GemmKernel::initialize_workspace(\n            arguments, workspace.get(), /*stream=*/nullptr));\n      }",
         "lock reset before event"),
        (1, "MBU=N/A", "MBU=0.0%", "unmodeled per-tile C traffic"),
        (1, "return final_result.passed ? 0 : 1;", "return 0;", "gate exit status"),
        (1, "[](uint16_t visits) { return visits != 1; }",
         "[](uint16_t visits) { return visits != 0; }", "exact (q,k) coverage"),
        (1, "cutlass::relatively_equal(want, got, epsilon, non_zero_floor)",
         "true /* planted comparator bypass */", "same comparator for bucket and disposition"),
        (2, "X(lowbit_dense_streamk_probe,64,128,64,64,32,2,0)",
         "X(lowbit_dense_streamk_probe,64,128,64,64,32,3,0)", "isolated fixture"),
        (5, "run_diagnostic_case 'A0 Stream-K diagnostic'",
         "run_case 'A0 Stream-K'", "preserve a failed A0 diagnostic"),
    ]
    for index, old, new, label in plants:
        planted = list(texts)
        if planted[index].count(old) != 1:
            print(f"[dense-streamk-contract] FAIL: cannot plant {label!r} control")
            return 1
        planted[index] = planted[index].replace(old, new, 1)
        if not audit(*planted):
            print(f"[dense-streamk-contract] FAIL: checker accepted planted {label} regression")
            return 1

    print("[dense-streamk-contract] PASS -- shared workers x4, absolute K, exact-CTA fixup, "
          "per-launch lock reset, exact seam fixture, exact DP/SK error buckets; "
          "ten planted regressions rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
