#!/usr/bin/env python3
"""Device-free contract for #107b's dense mixed-input Stream-K seam.

The dangerous failures here all compile and often return plausible numbers: workspace
decomposition and launch can see different worker counts, K slices can restart at zero,
fixup can run with the wrong barrier cohort, an old turnstile value can deadlock the
next launch, or replay can silently run only on a tiny fixture while fixed A0 prints an
unrelated whole-K verdict.  The PPU box owns numerical proof; this checker pins the
source ordering and makes every actual Stream-K arm fail closed on an empty seam and
bind its fixture exactness, whole-K reference, and fixup-only replay to one invocation
before deriving a disposition.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_streamk.hpp"
BENCH = ROOT / "benchmarks/test_lowbit_dense_bench.cu"
UNIT = ROOT / "dev/fold_derivation/test_lowbit_dense_unit.cu"
DISPATCH = ROOT / "benchmarks/lowbit_dense_unit.inc"
CMAKE = ROOT / "quactlize/csrc/CMakeLists.txt.in"
BOX_GATE = ROOT / "tools/run_dense_streamk_107b_box.sh"
BARRIER = ROOT / "third_party/actlize/include/cutlass/arch/barrier.h"
EXACT_FIXTURE = ROOT / "ci/check_exact_fixture.py"


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
          box_gate: str, barrier: str) -> list[str]:
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
    for token, count in (
        ("static constexpr uint32_t PreFixupCaptureMagic = 0x534b4650u;", 1),
        ("struct DiagnosticState {", 1),
        ("std::is_trivially_copyable_v<DiagnosticState>", 1),
        ("DiagnosticState* diagnostics = nullptr;", 2),
        ("params.diagnostics->witness", 3),
        ("TileScheduler::output_tile_index(", 1),
        ("map_idx = q * stride + k;", 1),
        ("CaptureStriped::store(capture_array, *accumulator_array, thread_idx);", 1),
        ("atomicAdd(diagnostics->pre_fixup_capture_slot_visits + slot, 1u);", 1),
        ("diagnostics->pre_fixup_capture_slot_k_counts[slot] =", 1),
        ("atomicAdd(&diagnostics->pre_fixup_capture_error_count, 1u);", 1),
    ):
        if header.count(token) != count:
            bad.append(f"gate-only partial capture requires {count} occurrence(s) of {token!r}")

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
        "bool const full_output_tile =",
        "TileScheduler::compute_epilogue(",
        "scheduler.fetch_next_work(work_tile_info)",
    ), "absolute-K mainloop/fixup/final-epilogue", bad)
    for token, count in (
        ("TileScheduler::fixup(", 2),
        ("detail::make_accumulator_residue_mask(", 1),
        ("if (!requires_fixup || full_output_tile)", 1),
        ("take<0, 2>(residue_mnk), thread_idx", 1),
    ):
        if device.count(token) != count:
            bad.append(f"residue-aware fixup requires {count} occurrence(s) of {token!r}")
    if device.count("TileScheduler::get_work_k_tile_count(") != 1:
        bad.append("mainloop does not use exactly one scheduler-owned K-tile count")
    ordered_once(device, (
        "collective_mainloop(params.mainloop",
        "bool const requires_fixup =",
        "diagnostics->pre_fixup_capture_magic == PreFixupCaptureMagic",
        "CaptureStriped::store(capture_array, *accumulator_array, thread_idx);",
        "bool const full_output_tile =",
        "if (!requires_fixup || full_output_tile)",
    ), "pre-fixup capture stays after mainloop and before production fixup", bad)

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
        "[dense verify partition] DP=%llu SK-whole=%llu SK-split=%llu",
        "[dense verify owners] tile=%dx%d cta_threads=%d ",
        "output_threads=%d K_cohorts=%d stripes/output_thread=%d ",
        "dense_map_accumulator_owners<Gemm>(verify_partition)",
        "right_inverse(accumulators.layout())",
        "final_visit[q] = uint16_t(last_local_tile - local);",
        "[dense verify bucket=%s] tiles=%llu outputs=%llu mismatches=%llu",
        "[dense verify mismatch] out=%zu tile=%zu q=%u",
        "[dense verify fingerprint] comparator_positions=%llu",
        "position_fnv1a=%016llx value_fnv1a=%016llx",
        "final_visit0=%llu final_visit_gt0=%llu",
        "cutlass::relatively_equal(want, got, epsilon, non_zero_floor)",
        "bucket comparator disagrees with device comparator",
        "max_rel_sym=%.9g max_half_ulp=%u nonfinite=%llu",
        "if ((split_tiles == 0) != (peer_excess == 0))",
        "partition.split_tiles = split_tiles;",
        "[dense streamk split gate] NOT EXERCISED real_cu=%d",
        "[dense streamk split gate] EXERCISED real_cu=%d ctas_per_cu=%d",
        "result.split_path_exercised = false;",
        "Disposition: NOT EXERCISED",
        "std::vector<std::vector<DenseVerifyPeerRange>> peer_ranges(sk.sk_tiles_);",
        "a.k_begin < b.k_begin",
        "range.k_begin != expected_k",
        "partition.capture_slot_by_qk[map_index] = int32_t(range.capture_slot);",
        "partition.split_peer_ranges.size() != split_tiles + peer_excess",
        "[dense verify interpretation] ORDER-INDEPENDENT fixture: raw_bitdiff=%llu;",
        "[dense verify interpretation] fixture rounds: ordinary-reference ULP is diagnostic only;",
        "capture_vs_normal_bitdiff +=",
        "for (uint32_t visits : capture_slot_visits) bad_slot_visits += visits != 1;",
        "capture_slot_k_counts[peer] !=",
        "if (bad_k_counts != 0)",
        "if (capture_vs_normal_bitdiff != 0)",
        "device_replay_bitdiff += replay_half.raw() != got.raw();",
        "non_split_reference_mismatches +=",
        "non_split_reference_bitdiff += got.raw() != ref.raw();",
        "typename Gemm::CollectiveEpilogue::ThreadEpilogueOp,",
        "ReplayEpilogue replay_epilogue(replay_params);",
        "volatile float workspace_replay = captured(first);",
        "float const replay = captured(last - 1) + float(workspace_replay);",
        "options.beta == 0.0f ? ElementC(0) : host_c[out];",
        "ElementD const replay_half = replay_epilogue(replay, replay_source);",
        "triangle_closed ? \"CLOSED\" : \"OPEN\"",
        "device_replay_bitdiff == 0 &&",
        "bool const fixup_closed = split_outputs > 0 && device_replay_bitdiff == 0 &&",
        "triangle_closed;",
        "host_diagnostics.pre_fixup_capture_magic =",
        "captured_diagnostics.pre_fixup_capture_error_count",
        "[streamk replay meaning] FIXUP-CLOSED: production fixup matches ",
        "partial correctness is not established.",
        "fixture=a0-exact shape=%dx%dx%d ",
        "kExactFixtureNonzerosPerRow = 32",
        "kExactFixtureScales[] = {1, 2, 4}",
        "kExactFixtureZeros[] = {0}",
        "int const sign = ((k >> 3) & 1) ? -1 : 1;",
        "int const q = ((k >> 3) & 1) ? (-8 + code) : code;",
    )
    for token in diagnostic_tokens:
        if bench.count(token) != 1:
            bad.append(f"A0 bucket diagnostic must contain exactly one {token!r}")
    if bench.count("DenseReplayEvidence verify_streamk_same_order_partial_replay(") != 2:
        bad.append("same-order replay must have one declaration and one definition")
    if bench.count("arguments.diagnostics = streamk_diagnostics.get();") != 2:
        bad.append("exact fixture and adaptive split gate must each wire the one-pointer diagnostic POD")
    if bench.count("dense_classify_streamk_tiles(") != 2:
        bad.append("the Stream-K classifier must have exactly one definition and one accepted-arm call")
    if "tile_peer_range(" in bench:
        bad.append("A0 bucket diagnostic uses tile_peer_range(), which is not group-general")
    if "likely the int4-oriented reference" in bench:
        bad.append("a failed Stream-K arm still blames the shared reference without evidence")
    try:
        common_map = section(
            bench, "    bool const common_map_ok =",
            "\n    // These fields describe K-peer capture")
        replay_map = section(
            bench, "    bool const replay_map_ok =",
            "\n    if (!common_map_ok || !replay_map_ok)")
    except ValueError as e:
        bad.append(str(e))
    else:
        # Ordinary DP/persistent arms have an output-tile ownership map but no
        # K-peer capture plan.  Requiring replay-only fields in common_map_ok is
        # the exact regression that made their bucket diagnostics silently
        # NOT CLASSIFIABLE.  Conversely, a Stream-K replay map must retain all
        # three pieces needed to address every captured (q,K_idx) peer.
        for token in (
            "k_tiles_per_output_tile",
            "split_peer_offsets",
            "capture_slot_by_qk",
        ):
            if token in common_map:
                bad.append(f"common DP tile map incorrectly requires replay-only {token}")
        for token in (
            "partition->k_tiles_per_output_tile > 0 &&",
            "partition->split_peer_offsets.size() >= 1 &&",
            "partition->capture_slot_by_qk.size() ==",
        ):
            if replay_map.count(token) != 1:
                bad.append(f"Stream-K replay map must contain exactly one {token!r}")
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
            "dense_classify_streamk_tiles(",
            "if (options.streamk && verify_partition.classification_closed)",
            "if (verify_partition.split_tiles == 0 ||",
            "result.split_path_exercised = false;",
        ), "only classify an accepted lowered Params", bad)
        # `--streamk_split_gate` is only a command-line request for a
        # correctness-only arm.  Once the named Stream-K kernel is selected,
        # every real `options.streamk` arm (fixed A0 included) must prove that
        # it has a peer seam and must replay that seam.  Reintroducing the old
        # flag guard silently moves both checks back onto the tiny/adaptive
        # fixture while A0 prints an unrelated ordinary-reference verdict.
        if "options.streamk_split_gate" in run_body:
            bad.append("run<Gemm> still limits split/replay evidence to --streamk_split_gate")
        for token in (
            "if (options.streamk && verify_partition.classification_closed) {\n      uint64_t const workers =",
            "if (options.streamk) {\n      if (ordinary_diagnostic_state == DenseVerifyState::NotClassifiable)",
            "else if (options.streamk) {\n    if (result.passed) {",
            "replay_evidence = verify_streamk_same_order_partial_replay<Gemm>(",
            "else if (fixture_evidence.order_independent) {",
            "reference_raw_bitdiff == 0;",
            "if (!result.verification_classified) {\n    std::cout << \"  Disposition: NOT CLASSIFIABLE \"",
        ):
            if run_body.count(token) != 1:
                bad.append(f"every actual Stream-K arm requires exactly one {token!r}")
        ordered_once(run_body, (
            "if (verify_partition.split_tiles == 0 ||",
            "std::cout << \"  Disposition: NOT EXERCISED\"",
            "// Correctness / Warmup iteration",
            "replay_evidence = verify_streamk_same_order_partial_replay<Gemm>(",
            "if (options.streamk && result.verification_classified)",
            "else if (options.streamk) {\n    if (result.passed) {",
        ), "nonempty seam -> same-order replay -> fixture-owned disposition", bad)

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
    try:
        streamk_metrics = section(
            bench,
            "#if defined(DENSE_STREAMK_AB)\n"
            "    if constexpr (dense_is_streamk_gemm<Gemm>::value) {",
            "    } else if constexpr (dense_is_marlin_gemm<Gemm>::value) {")
    except ValueError as e:
        bad.append(str(e))
        streamk_metrics = ""
    try:
        streamk_exit = section(
            bench,
            "#elif defined(DENSE_STREAMK_AB)\n"
            "  // The 107b target is a mechanism/numerical gate",
            "#elif defined(DENSE_MARLIN_SWEEP)")
    except ValueError as e:
        bad.append(str(e))
        streamk_exit = ""

    for token in (
        "actual=%s real_cu=%d ctas_per_cu=%d",
        "params.scheduler_hw_info.cu_count == int(workers)",
        "witness[0] == 8 && witness[1] == 1 && witness[2] == 0",
        "outputs=%zu bad=%d bitdiff=%d",
        "[dense kernel-span-upper]",
        "spread=(max-min)/mean=",
        "distinct-event-pairs=%zu warmup-event-pairs=1 includes-launch-idle=1",
        "StreamK-C valid_elements=%llu peer_excess=%llu",
        "[streamk sequential CPU-FP32 fixture] order=k-ascending ",
        "[streamk same-order replay] split_tiles=%llu peers=%zu ",
        "if (!final_result.split_path_exercised) return 2;",
        "if (!final_result.verification_classified) return 3;",
    ):
        if bench.count(token) != 1:
            bad.append(f"bench must contain exactly one {token!r}")
    for token in (
        "if (!final_result.split_path_exercised) return 2;",
        "if (!final_result.verification_classified) return 3;",
        "return final_result.passed ? 0 : 1;",
    ):
        if streamk_exit.count(token) != 1:
            bad.append(f"Stream-K exit arm must contain exactly one {token!r}")
    if streamk_metrics.count("MODEL-ONLY/not-a-DRAM-counter") != 1:
        bad.append("Stream-K reporting branch must contain exactly one MODEL-ONLY/not-a-DRAM-counter label")
    for token in (
        "valid_fixup_elements +=\n        (peers[q] - 1) * uint64_t(valid_m) * uint64_t(valid_n);",
        "partition.valid_fixup_elements = valid_fixup_elements;",
        "2.0 * sizeof(float) * double(verify_partition.valid_fixup_elements)",
    ):
        if bench.count(token) != 1:
            bad.append(f"dense per-q partial-C model must contain exactly one {token!r}")
    if streamk_metrics.count("bench_measure::make_traffic_with_output_bytes(") != 1:
        bad.append("Stream-K reporting branch must contain exactly one make_traffic_with_output_bytes call")
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

    try:
        control_case = section(
            box_gate, "run_control_case() {", "\n}\n\nrequire_exact_fixture()")
    except ValueError as e:
        bad.append(str(e))
    else:
        ordered_once(control_case, (
            'if [ "$rc" -eq 1 ]; then',
            'fail "$label reported a real numerical/invariant failure"',
            'if [ "$rc" -eq 0 ]; then',
            'elif [ "$rc" -eq 3 ]; then',
            'NOT CLASSIFIABLE; continuing to the Stream-K subject',
            'else',
            'fail "$label exited rc=$rc instead of CLASSIFIED(0/1) or NOT CLASSIFIABLE(3)"',
            'dense kernel-span-upper',
        ), "control rc1 fails while rc3 continues to the Stream-K subject", bad)

    for token in (
        "run_control_case 'A0 non-persistent control'",
        "run_control_case 'A0 serial-persistent control'",
        "COMMON_SHAPE=(--m=2048 --n=\"$SPLIT_N\" --k=4096 --l=1 --g=128 --mode=1)",
        "run_split_probe()",
        "if [ \"$rc\" -eq 2 ]",
        "return 2",
        "return \"$rc\"",
        "if (rc == 0) != passed_disposition or (rc == 1) != failed_disposition:",
        "if rc == 0 and not (",
        "if rc == 1 and upstream_failed and not (",
        "print(4096)                     # exact A0 is always tried first",
        "for n_tiles in range(32, 32 + 64)",
        "if n != 4096 and tiles > workers and tiles % workers:",
        "if run_split_probe \"A0 replay probe N=${candidate_n}\"",
        "if run_split_probe \"adaptive split-path repeat N=${SPLIT_N}\"",
        "if run_split_probe 'A0 Stream-K same-order replay and performance'",
        '"$SK_LOG" "${COMMON[@]}" --streamk',
        "[ \"$probe_rc\" -eq 2 ]",
        "split == gate_split > 0",
        "peers == gate_peers > 0",
        "SK-split>0 and peer_excess>0 were prerequisites",
        "correctness is the printed disposition, never an empty PASS",
        "[streamk same-order replay] split_tiles=1 peers=8 split_outputs=8192",
        "capture_holes=0 bad_slot_visits=0 bad_k_counts=0 capture_vs_normal_bitdiff=0 device_replay_bitdiff=0",
        "non_split_reference_mismatches=0 non_split_reference_bitdiff=0",
        "triangle=CLOSED FIXUP-CLOSED",
        "ci/check_exact_fixture.py",
        "STABLE_POSITIONS_AND_VALUES",
        "STABLE_POSITIONS_VALUE_DRIFT",
        "POSITION_DRIFT",
    ):
        if box_gate.count(token) != 1:
            bad.append(f"box gate must contain exactly one {token!r}")
    if 'if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]' not in box_gate:
        bad.append("box gate does not preserve a complete rc=1 numerical diagnostic")
    if box_gate.count('if [ "$rc" -eq 3 ]') != 2:
        bad.append("control and replay helpers must each fail closed on NOT CLASSIFIABLE rc=3")
    if "run_case 'A0 Stream-K'" in box_gate:
        bad.append("box gate still drops the failing A0 Stream-K diagnostic")
    if "run_diagnostic_case 'A0 Stream-K" in box_gate:
        bad.append("fixed A0 still bypasses the same-order replay parser")
    if box_gate.count("run_split_probe") != 5:
        bad.append("box gate must use one replay parser for exact gate, A0 selection/repeat, and A0 timing")
    if box_gate.count("--streamk_exact_fixture") != 3:
        bad.append("A0 selection, repeat, and shared DP/P/SK controls must all select the exact fixture")
    if box_gate.count('require_exact_fixture "$label" "$log"') != 2:
        bad.append("both control and Stream-K helpers must bind exactness to their own log")
    if box_gate.count("--iterations=0 --streamk") != 2:
        bad.append("box gate must run exactly two correctness-only A0 replay arms")
    if "--streamk_split_gate" in box_gate:
        bad.append("box gate still limits A0 replay to the obsolete split-only option")
    if "coverage=exact-once" not in box_gate:
        bad.append("box gate does not require the exact scheduler coverage witness")
    try:
        split_flow = section(box_gate,
                             "# Select A0 itself when its lowered scheduler contains a peer seam.",
                             "\n# Same binary, tactic, selected shape, event protocol")
    except ValueError as e:
        bad.append(str(e))
    else:
        ordered_once(split_flow, (
            "WORKERS=\"$(python3 - \"$GATE_LOG\"",
            "mapfile -t SPLIT_CANDIDATES",
            "print(4096)",
            "for candidate_n in \"${SPLIT_CANDIDATES[@]}\"",
            "A0 replay probe N=${candidate_n}",
            "SPLIT_LOG=\"$candidate_log\"",
            "adaptive split-path repeat N=${SPLIT_N}",
            "python3 - \"$SPLIT_LOG\" \"$SPLIT_REPEAT_LOG\"",
        ), "runtime A0 selection/replay before fingerprint comparison", bad)
    try:
        a0_flow = section(box_gate,
                          "# Same binary, tactic, selected shape, event protocol",
                          "\npython3 - \"$NP_LOG\"")
    except ValueError as e:
        bad.append(str(e))
    else:
        ordered_once(a0_flow, (
            "COMMON_SHAPE=(--m=2048 --n=\"$SPLIT_N\"",
            "run_control_case 'A0 non-persistent control'",
            "run_control_case 'A0 serial-persistent control'",
            "if run_split_probe 'A0 Stream-K same-order replay and performance'",
            '"$SK_LOG" "${COMMON[@]}" --streamk',
            "fail \"A0 Stream-K subject failed or was not exercised rc=$a0_rc\"",
        ), "selected-shape A0 controls and replay-owned Stream-K timing", bad)

    # This is a compiler-ordering contract, not a hardware-instruction check.
    # ppu.bar.sync and ppu.fence are opaque strings to C++; volatile preserves
    # the asm itself but only a memory clobber pins ordinary FP32 workspace
    # accesses to the two sides of the publish/acquire seam.  Keep all four
    # arms aligned with upstream CUTLASS, not just the one A0 happens to call.
    sync = ('asm volatile("ppu.bar.sync %0, %1;" : : "r"(barrier_id), '
            '"r"(num_threads) : "memory");')
    arrive = ('asm volatile("ppu.bar.arrive %0, %1;" : : "r"(barrier_id), '
              '"r"(num_threads) : "memory");')
    if barrier.count(sync) != 2 or barrier.count(arrive) != 2:
        bad.append("all four PPU named-barrier asm arms must carry a compiler memory clobber")
    legacy_sync = ('asm volatile("ppu.bar.sync %0, %1;" : : "r"(barrier_id), '
                   '"r"(num_threads));')
    legacy_arrive = ('asm volatile("ppu.bar.arrive %0, %1;" : : "r"(barrier_id), '
                     '"r"(num_threads));')
    if legacy_sync in barrier or legacy_arrive in barrier:
        bad.append("a PPU named-barrier arm retains the compiler-reorderable spelling")
    return bad


def main() -> int:
    texts = [p.read_text() for p in
             (HEADER, BENCH, UNIT, DISPATCH, CMAKE, BOX_GATE, BARRIER)]
    bad = audit(*texts)
    exact = subprocess.run(
        [sys.executable, str(EXACT_FIXTURE)], cwd=ROOT,
        capture_output=True, text=True)
    if exact.returncode != 0:
        detail = (exact.stdout + exact.stderr).strip().splitlines()
        bad.append("exact A0 fixture checker did not PASS: " +
                   (detail[-1] if detail else f"rc={exact.returncode}"))
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
        (0, "diagnostics->pre_fixup_capture_magic == PreFixupCaptureMagic",
         "diagnostics->pre_fixup_capture_magic != PreFixupCaptureMagic",
         "capture magic fail-close"),
        (0, "map_idx = q * stride + k;", "map_idx = q + k;",
         "capture map retains both q and absolute K"),
        (0, "CaptureStriped::store(capture_array, *accumulator_array, thread_idx);",
         "/* planted capture drop */ (void)capture_array;",
         "every peer fragment is captured before fixup"),
        (0, "atomicAdd(diagnostics->pre_fixup_capture_slot_visits + slot, 1u);",
         "/* planted visit-count drop */ (void)slot;",
         "duplicate or missing peer slots stay visible"),
        (0, "diagnostics->pre_fixup_capture_slot_k_counts[slot] =",
         "/* planted K-count capture drop */ (void)",
         "device peer lengths remain tied to the host range census"),
        (1, "Gemm::GemmKernel::initialize_workspace(\n            arguments, workspace.get(), /*stream=*/nullptr));\n      }\n      auto& events = streamk_events.at(iter + 1);\n      CUTLASS_PPU_CHECK(hggcEventRecord(events.start, nullptr));",
         "auto& events = streamk_events.at(iter + 1);\n      CUTLASS_PPU_CHECK(hggcEventRecord(events.start, nullptr));\n      Gemm::GemmKernel::initialize_workspace(\n            arguments, workspace.get(), /*stream=*/nullptr));\n      }",
         "lock reset before event"),
        (1, "2.0 * sizeof(float) * double(verify_partition.valid_fixup_elements)",
         "2.0 * sizeof(float) * double(verify_partition.peer_excess)",
         "partial-C traffic drops valid residue weighting"),
        (1, '          "%s | StreamK-C valid_elements=%llu peer_excess=%llu "\n'
            '          "logical_RW=%.0f MODEL-ONLY/not-a-DRAM-counter\\n",',
         '          "%s | StreamK-C valid_elements=%llu peer_excess=%llu "\n'
            '          "logical_RW=%.0f measured-DRAM-bytes\\n",',
         "logical partial-C model is mislabeled as a DRAM counter"),
        (1, "partition->local_stripe.size() ==\n"
         "            std::size_t(partition->tile_m) * partition->tile_n;",
         "partition->local_stripe.size() ==\n"
         "            std::size_t(partition->tile_m) * partition->tile_n &&\n"
         "        partition->k_tiles_per_output_tile > 0;",
         "common DP map cannot inherit a Stream-K replay requirement"),
        (1, "partition->k_tiles_per_output_tile > 0 &&",
         "true &&", "replay map retains K-tile stride"),
        (1, "partition->split_peer_offsets.size() >= 1 &&",
         "true &&", "replay map retains peer offsets"),
        (1, "partition->capture_slot_by_qk.size() ==",
         "std::size_t(0) ==", "replay map retains q/K capture slots"),
        (1, "if (!final_result.verification_classified) return 3;\n"
         "  return final_result.passed ? 0 : 1;",
         "if (!final_result.verification_classified) return 3;\n"
         "  return 0;", "gate exit status"),
        (1, "if (!final_result.split_path_exercised) return 2;",
         "if (!final_result.split_path_exercised) return 0;",
         "NOT EXERCISED has a distinct process status"),
        (1, "if (verify_partition.split_tiles == 0 ||\n"
         "          verify_partition.peer_excess == 0)",
         "if (false && (verify_partition.split_tiles == 0 ||\n"
         "          verify_partition.peer_excess == 0))",
         "empty split path fails closed in the benchmark"),
        (1, "if (options.streamk && verify_partition.classification_closed) {\n      uint64_t const workers =",
         "if (options.streamk_split_gate && verify_partition.classification_closed) {\n      uint64_t const workers =",
         "fixed A0 cannot bypass the nonempty-seam gate"),
        (1, "if (options.streamk) {\n      if (ordinary_diagnostic_state == DenseVerifyState::NotClassifiable)",
         "if (options.streamk_split_gate) {\n      if (ordinary_diagnostic_state == DenseVerifyState::NotClassifiable)",
         "fixed A0 cannot bypass same-order replay"),
        (1, "else if (options.streamk) {\n    if (result.passed) {",
         "else if (options.streamk_split_gate) {\n    if (result.passed) {",
         "every actual Stream-K disposition is replay-owned"),
        (1, "std::cout << \"  Disposition: NOT EXERCISED\"",
         "std::cout << \"  Disposition: Passed\"",
         "an empty seam cannot be reported as Passed"),
        (1, "a.k_begin < b.k_begin", "a.k_begin > b.k_begin",
         "peer replay is ordered by increasing K_idx"),
        (1, "range.k_begin != expected_k", "range.k_begin == expected_k",
         "peer ranges are contiguous exact-once"),
        (1, "if (capture_vs_normal_bitdiff != 0)", "if (false)",
         "instrumented launch must equal normal launch"),
        (1, "device_replay_bitdiff == 0 &&",
         "device_replay_bitdiff >= 0 &&", "same-order replay is bit exact"),
        (1, "else if (fixture_evidence.order_independent) {",
         "else if (false) {", "exactness must select the strict reference verdict"),
        (1, "reference_raw_bitdiff == 0;",
         "true; /* planted raw-reference bypass */",
         "an exact fixture cannot pass with a raw reference difference"),
        (1, "replay_evidence = verify_streamk_same_order_partial_replay<Gemm>(",
         "result.passed = verify_streamk_same_order_partial_replay<Gemm>(",
         "fixup replay cannot overwrite the ordinary-reference result"),
        (1, "int const sign = ((k >> 3) & 1) ? -1 : 1;",
         "int const sign = ((m + kg) & 1) ? -1 : 1;",
         "exact A0 avoids cancellation and signed-zero ambiguity"),
        (1, "bool const fixup_closed = split_outputs > 0 && device_replay_bitdiff == 0 &&",
         "bool const fixup_closed = split_outputs > 0 && device_replay_bitdiff >= 0 &&",
         "fixup closure requires bit-exact device replay"),
        (5, "  return \"$rc\"\n}", "  return 0\n}",
         "an exercised replay failure cannot become an adaptive-search success"),
        (1, "[](uint16_t visits) { return visits != 1; }",
         "[](uint16_t visits) { return visits != 0; }", "exact (q,k) coverage"),
        (1, "cutlass::relatively_equal(want, got, epsilon, non_zero_floor)",
         "true /* planted comparator bypass */", "same comparator for bucket and disposition"),
        (1, "dense_map_accumulator_owners<Gemm>(verify_partition)",
         "true /* planted owner-map bypass */", "real MMA lane/stripe inverse"),
        (1, "final_visit[q] = uint16_t(last_local_tile - local);",
         "final_visit[q] = uint16_t(0);", "persistent final-peer visit ordinal"),
        (2, "X(lowbit_dense_streamk_probe,64,128,64,64,32,2,0)",
         "X(lowbit_dense_streamk_probe,64,128,64,64,32,3,0)", "isolated fixture"),
        (5, "if run_split_probe 'A0 Stream-K same-order replay and performance'",
         "if run_control_case 'A0 Stream-K same-order replay and performance'",
         "A0 timing arm cannot drop same-order replay"),
        (5, "if run_split_probe \"A0 replay probe N=${candidate_n}\"",
         "if run_control_case \"A0 replay probe N=${candidate_n}\"",
         "adaptive A0 selection cannot drop same-order replay"),
        (5, 'fail "$label reported a real numerical/invariant failure"',
         'printf "%s\\n" "$label reported a real numerical/invariant failure"',
         "a classified control failure remains fatal"),
        (5, "printf '[107b][control] %s: NOT CLASSIFIABLE; continuing to the Stream-K subject\\n' \"$label\"",
         "fail \"$label blocked the Stream-K subject after NOT CLASSIFIABLE\"",
         "an unclassifiable optional control does not block Stream-K"),
        (5, "    return 2\n  fi",
         "    return 0\n  fi", "empty split probe cannot become evidence"),
        (5, "    if n != 4096 and tiles > workers and tiles % workers:",
         "    if n != 4096 and tiles > workers:", "adaptive candidates avoid exact worker waves"),
        (5, "split == gate_split > 0 and\n        peers == gate_peers > 0",
         "split == gate_split >= 0 and\n        peers == gate_peers >= 0",
         "split evidence must be non-empty"),
        (6, '    // rise above its acquire-side sync.\n'
            '    asm volatile("ppu.bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads) : "memory");',
         '    // rise above its acquire-side sync.\n'
            '    asm volatile("ppu.bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));',
         "named-barrier compiler memory fence"),
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
          "named-barrier compiler fence, per-launch lock reset, adaptive nonempty seam, "
          "per-run exact A0 evidence, compact pre-fixup capture and fixup-only replay; "
          f"{len(plants)} planted regressions rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
