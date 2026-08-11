#!/usr/bin/env python3
"""Device-free contract for grouped mixed-input Stream-K phase 2 / Min=2.

The dangerous regressions all compile into plausible kernels: host decomposition
and launch can use different worker populations, an expert-local coordinate can
alias a global lock, or the ragged prefix can overwrite reduction scratch.  This
checker pins those seams, compiles two negative controls for the vendor policy
type, and proves the owned handle rejects update() and raw Params launches; the
The live isolated specialization must select Min=2 rather than silently falling
back to the ABI-compatible Min=8 default.  The PPU box remains responsible for
numerical/fixup proof.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WRAPPER = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_group_streamk.hpp"
BUILDER = ROOT / "quactlize/include/moe_grouped_streamk_ppu.cuh"
TEST = ROOT / "tests/test_moe_grouped_streamk.cu"
L121 = ROOT / "dev/fold_derivation/l121_grouped_streamk_wrapper.cu"
CMAKE = ROOT / "quactlize/csrc/CMakeLists.txt.in"
BUILD = ROOT / "build.sh"


_CPP_TOKEN = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
    re.MULTILINE | re.DOTALL)


def without_cpp_comments(text: str) -> str:
    """Drop comments while retaining string/character literals and line structure."""
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.startswith(('"', "'")):
            return token
        return "\n" * token.count("\n")
    return _CPP_TOKEN.sub(replace, text)


def section(text: str, begin: str, end: str) -> str:
    if text.count(begin) != 1 or text.count(end) < 1:
        raise ValueError(f"cannot isolate {begin!r} .. {end!r}")
    return text.split(begin, 1)[1].split(end, 1)[0]


def ordered_once(text: str, anchors: tuple[str, ...], label: str, bad: list[str]) -> None:
    if any(text.count(x) != 1 for x in anchors):
        bad.append(f"{label}: anchors missing or duplicated")
    elif [text.index(x) for x in anchors] != sorted(text.index(x) for x in anchors):
        bad.append(f"{label}: source order is wrong")


def audit(wrapper: str, builder: str, test: str, cmake: str, build: str) -> list[str]:
    bad: list[str] = []
    # Structural evidence must live in executable source, not in a comment
    # preserving a checker anchor after the implementation changed.
    wrapper_code = without_cpp_comments(wrapper)
    builder_code = without_cpp_comments(builder)
    test_code = without_cpp_comments(test)
    constant_bypass = (r"\btrue\s*\|\||\bfalse\s*&&|\bfalse\s*\?|"
                       r"\bif\s*\(\s*false\s*\)")
    if re.search(constant_bypass, wrapper_code):
        bad.append("owned wrapper contains a constant short-circuit bypass")
    for token in (
        "class GroupStreamKMixedInputKernel",
        "TileShape, ClusterShape, MinSkIters, MaxThreadsPerBlock>;",
        "detail::PersistentTileSchedulerPPUStreamKParamsT<MinSkIters>",
        '"grouped Stream-K lost configured MinSkIters"',
        '"Stream-K stripe is shorter than pipeline startup"',
        "static_assert(MaxThreadsPerBlock == 64u || MaxThreadsPerBlock == 128u",
        "static_assert(TileScheduler::FixupThreadCount == MaxThreadsPerBlock",
        "static_assert(!CollectiveMainloop::SwapAB",
        "q += mt * nt;",
        "args.mainloop.group_row_offsets != nullptr",
        "auto expert_shape = params.host_shape_mirror[expert];",
        "args.census = census;",
        "ExpectedGroupSize =",
    ):
        hay = (builder_code if token in ("args.census = census;",
                                         "ExpectedGroupSize =")
               else wrapper_code)
        if hay.count(token) != 1:
            bad.append(f"expected exactly one {token!r}")

    for token in (
        "constexpr uint32_t kMinSkIters = 2;",
        "kStages, true, int4_t, void, kArtifactTileK, kMinSkIters>;",
        "Op<64>::TileSchedulerParams::min_iters_per_sk_unit_ == 2",
        "Op<256>::TileSchedulerParams::min_iters_per_sk_unit_ == 2",
        '"phase 2 grouped Stream-K must explicitly select Min=2"',
    ):
        if test_code.count(token) != 1:
            bad.append(f"live phase-2 specialization missing exactly one {token!r}")

    try:
        geometry = section(wrapper_code, "  static HostGeometry host_geometry", "\n  static SchedulerProblemShape")
        scheduler_workspace = section(
            wrapper_code, "  static size_t scheduler_workspace_size", "\n  static WorkspaceLayout")
        scheduler_hw = section(
            wrapper_code, "  static KernelHardwareInfo scheduler_hw_info", "\n  static size_t scheduler_workspace_size")
        workspace_size = section(
            wrapper_code, "  static size_t get_workspace_size", "\n\n  static cutlass::Status initialize_workspace")
        layout = section(wrapper_code, "  struct WorkspaceLayout", "\n  static HostGeometry")
        workspace_calc = section(
            wrapper_code, "  static WorkspaceLayout workspace_layout", "\n\n public:")
        lower = section(wrapper_code, "  static Params to_underlying_arguments", "\n  static bool can_implement")
        init = section(wrapper_code, "  static cutlass::Status initialize_workspace",
                       "\n  static cutlass::Status reset_scheduler_workspace_after_prefix_install")
        reset = section(wrapper_code,
                        "  static cutlass::Status reset_scheduler_workspace_after_prefix_install",
                        "\n  static dim3 get_grid_shape")
        grid = section(wrapper_code, "  static dim3 get_grid_shape", "\n  static dim3 get_block_shape")
        device = section(wrapper_code, "  CUTLASS_DEVICE\n  void operator()", "\n};")
        runtime = section(builder_code, "  static void configure_runtime", "\n  static Plan inspect")
        arm = section(test_code, "ArmResult run_streamk_arm", "\nint run_decode64_nonuniform")
        decode64 = section(test_code, "int run_decode64_nonuniform", "\nbool print_c_traffic")
        valid_traffic = section(test_code, "uint64_t valid_fixup_elements(",
                                "\nPhase2Expectation phase2_expectation")
        verify = section(test_code, "int verify_output", "\ntemplate <class Params>")
        policy = section(test_code, "bool host_policy_line", "\nstruct ArmResult")
        phase2_oracle = section(
            test_code, "Phase2Expectation phase2_expectation",
            "\ntemplate <int TK>\nArmResult run_streamk_arm")
        main = section(test_code, "int main()", "\n}")
    except ValueError as e:
        return bad + [str(e)]

    q_mutations = re.findall(r"(?<!\.)\bq\s*(?:\+=|-=|\*=|/=|=)", geometry)
    if q_mutations != ["q =", "q +="] or geometry.count("q += mt * nt;") != 1:
        bad.append("host q must be initialized once and changed only by q += mt * nt")
    if geometry.count("(*prefix)[size_t(e) + 1] = int(q);") != 1 or \
            geometry.count("out.q = int(q);") != 1:
        bad.append("host q is not written unchanged to both prefix and geometry")

    ordered_once(layout, (
        "scheduler_bytes", "prefix_offset", "prefix_bytes",
        "shape_offset", "shape_bytes", "epilogue_offset",
        "epilogue_bytes", "total_bytes",
    ), "workspace field order", bad)
    workspace_assignments = re.findall(
        r"out\.(scheduler_bytes|prefix_offset|prefix_bytes|shape_offset|"
        r"shape_bytes|epilogue_offset|epilogue_bytes|total_bytes)\s*=",
        workspace_calc)
    expected_workspace_assignments = [
        "scheduler_bytes", "prefix_offset", "prefix_bytes",
        "shape_offset", "shape_bytes", "epilogue_offset",
        "epilogue_bytes", "total_bytes"]
    if workspace_assignments != expected_workspace_assignments:
        bad.append("workspace fields must each be assigned once in non-overlapping order")
    for token in (
        "out.prefix_bytes = sizeof(int) * size_t(g.groups + 1);",
        "out.shape_bytes = sizeof(UnderlyingProblemShape) * size_t(g.groups);",
        "out.prefix_offset = round_nearest(out.scheduler_bytes,",
        "out.shape_offset = round_nearest(out.prefix_offset + out.prefix_bytes,",
        "out.epilogue_offset = round_nearest(out.shape_offset + out.shape_bytes,",
        "out.total_bytes = round_nearest(out.epilogue_offset + out.epilogue_bytes,",
    ):
        if workspace_calc.count(token) != 1:
            bad.append(f"workspace layout missing exact non-overlap expression {token!r}")
    if not re.search(r"return\s+out\s*;\s*}\s*$", workspace_calc):
        bad.append("workspace layout does not return the fully populated structure")
    workspace_size_flat = re.sub(r"\s+", " ", workspace_size)
    if "return workspace_layout(args, g).total_bytes;" not in workspace_size_flat:
        bad.append("public workspace size is not the complete layout total")
    scheduler_workspace_flat = re.sub(r"\s+", " ", scheduler_workspace)
    expected_scheduler_workspace = (
        "return TileScheduler::template get_workspace_size<"
        "SchedulerProblemShape, ElementAccumulator>( args.scheduler, shape, "
        "scheduler_hw_info(args), NumMmaWarpGroups);")
    if scheduler_workspace_flat.count(expected_scheduler_workspace) != 1:
        bad.append("scheduler workspace size is not the direct reviewed vendor result")
    ordered_once(lower, (
        "TileScheduler::to_underlying_arguments(",
        "workspace_ptr + layout.epilogue_offset",
        "workspace_ptr + layout.prefix_offset",
        "workspace_ptr + layout.shape_offset",
    ), "scheduler/prefix/epilogue lowering", bad)
    lower_flat = re.sub(r"\s+", " ", lower)
    expected_prefix_ptr = (
        "workspace_ptr ? reinterpret_cast<int const*>( workspace_ptr + "
        "layout.prefix_offset) : nullptr,")
    if expected_prefix_ptr not in lower_flat:
        bad.append("lowered prefix pointer is not the aligned post-scheduler segment")
    expected_shape_ptr = (
        "workspace_ptr ? reinterpret_cast<UnderlyingProblemShape const*>( "
        "workspace_ptr + layout.shape_offset) : nullptr,")
    if expected_shape_ptr not in lower_flat:
        bad.append("lowered shape mirror pointer is not the aligned post-prefix segment")
    if ("scheduler, workspace, layout.scheduler_bytes, barrier_offset, "
            "barrier_bytes,") not in lower_flat:
        bad.append("lowered Params do not bind the scheduler and barrier workspace")
    expected_barrier_size = (
        "TileSchedulerParams::get_barrier_workspace_size( scheduler.sk_tiles_, "
        "NumMmaWarpGroups, cutlass::sizeof_bits<typename "
        "TileScheduler::BarrierType>::value);")
    if lower_flat.count(expected_barrier_size) != 1:
        bad.append("lowered Params do not derive the barrier tail from the frozen scheduler")
    if ("void* epilogue_workspace = workspace_ptr ? workspace_ptr + "
            "layout.epilogue_offset : nullptr;") not in lower_flat:
        bad.append("lowered epilogue pointer is not the aligned post-prefix segment")
    # All scheduler decisions must consume the same virtual worker source.
    worker_formula = (
        "int64_t const workers = int64_t(real.cu_count) * "
        "int64_t(args.ctas_per_cu);")
    if wrapper_code.count(worker_formula) != 1:
        bad.append("physical workers are not exactly real_CU * occupancy_ctas_per_CU")
    if not re.search(r"return\s+KernelHardwareInfo\{real\.device_id, bounded\}\s*;\s*}\s*$",
                     scheduler_hw):
        bad.append("scheduler hardware info does not return the bounded worker count")
    for label, body, token in (
        ("workspace", scheduler_workspace, "scheduler_hw_info(args), NumMmaWarpGroups"),
        ("lowering", lower, "KernelHardwareInfo workers = scheduler_hw_info(args);"),
        ("initialization", init, "KernelHardwareInfo workers = scheduler_hw_info(args);"),
        ("grid", grid, "params.scheduler_hw_info"),
    ):
        if body.count(token) != 1:
            bad.append(f"{label} does not use exactly one shared worker source")
    if "real_hw_info(args)" in scheduler_workspace:
        bad.append("scheduler workspace sizing uses real CUs instead of physical workers")
    for label, body in (("lowering", lower), ("initialization", init)):
        if len(re.findall(r"\bworkers\s*=", body)) != 1 or \
                "KernelHardwareInfo workers = scheduler_hw_info(args);" not in body:
            bad.append(f"{label} may overwrite the shared physical-worker value")
    lower_flat = re.sub(r"\s+", " ", lower)
    init_flat = re.sub(r"\s+", " ", init)
    if "ClusterShape{}, workers, args.scheduler, workspace," not in lower_flat:
        bad.append("scheduler decomposition does not consume the shared worker value")
    if ("args.scheduler, workspace_ptr, stream, shape, workers, "
            "NumMmaWarpGroups") not in init_flat:
        bad.append("scheduler workspace initialization does not consume the shared worker value")
    expected_null_guard = (
        "if (layout.total_bytes > 0 && workspace == nullptr) { "
        "return Status::kErrorWorkspaceNull; }")
    if init_flat.count(expected_null_guard) != 1:
        bad.append("full initialization does not fail closed on a null workspace")
    expected_scheduler_init = (
        "Status status = TileScheduler::template initialize_workspace<"
        "SchedulerProblemShape, ElementAccumulator>( args.scheduler, "
        "workspace_ptr, stream, shape, workers, NumMmaWarpGroups, "
        "1, 1, host_adapter);")
    if init_flat.count(expected_scheduler_init) != 1:
        bad.append("full initialization does not execute the reviewed scheduler init exactly once")
    expected_status_guard = (
        "if (status != Status::kSuccess) { return status; }")
    if init_flat.count(expected_status_guard) != 1:
        bad.append("full initialization does not propagate scheduler init failure")
    if re.search(r"\btrue\s*\|\||\bfalse\s*&&|\bfalse\s*\?|"
                 r"\bif\s*\(\s*false\s*\)", init):
        bad.append("full initialization contains a constant short-circuit bypass")
    reset_flat = re.sub(r"\s+", " ", reset)
    for token in (
        "if (workspace == nullptr || workspace != params.workspace_base || "
        "params.output_tile_prefix == nullptr || params.host_shape_mirror == nullptr)",
        "if (params.scheduler.reduction_workspace_ != params.workspace_base || "
        "params.scheduler_barrier_bytes > params.scheduler_workspace_bytes || "
        "params.scheduler_barrier_offset != params.scheduler_workspace_bytes - "
        "params.scheduler_barrier_bytes || (params.scheduler.sk_units_ > "
        "params.scheduler.sk_tiles_ && params.scheduler_barrier_bytes == 0))",
        "if (params.scheduler_barrier_bytes == 0)",
        "hggcError_t const status = hggcMemsetAsync( "
        "static_cast<uint8_t*>(workspace) + params.scheduler_barrier_offset, 0, "
        "params.scheduler_barrier_bytes, stream);",
    ):
        if token not in reset_flat:
            bad.append(f"frozen timed reset missing {token!r}")
    if re.search(r"\btrue\s*\|\||\bfalse\s*&&|"
                 r"\bif\s*\(\s*false\s*\)", reset):
        bad.append("frozen timed reset contains a constant short-circuit bypass")
    if "hggcMemcpy" in reset or "CollectiveEpilogue::initialize_workspace" in reset:
        bad.append("timed reset touches installed metadata or epilogue setup")
    if "std::min" in grid or "min(" in grid:
        bad.append("physical Stream-K grid is truncated by logical tile count")
    grid_call = re.sub(r"\s+", " ", grid)
    expected_grid = (
        "return TileScheduler::get_grid_shape( params.scheduler, "
        "params.scheduler_problem_shape, TileShape{}, ClusterShape{}, "
        "params.scheduler_hw_info, args);")
    if expected_grid not in grid_call:
        bad.append("grid call does not consume params.scheduler_hw_info directly")
    for token in (
        "workspace_ptr + layout.prefix_offset, prefix.data()",
        "workspace_ptr + layout.shape_offset",
        "args.problem_shape.host_problem_shapes, layout.shape_bytes",
    ):
        if init.count(token) != 1:
            bad.append(f"one-shot prefix/shape install missing {token!r}")
    init_flat = re.sub(r"\s+", " ", init)
    exact_prefix_copy = (
        "hggcError_t copy_status = hggcMemcpy( workspace_ptr + "
        "layout.prefix_offset, prefix.data(), layout.prefix_bytes, "
        "hggcMemcpyHostToDevice);")
    exact_shape_copy = (
        "copy_status = hggcMemcpy( workspace_ptr + layout.shape_offset, "
        "args.problem_shape.host_problem_shapes, layout.shape_bytes, "
        "hggcMemcpyHostToDevice);")
    if init_flat.count(exact_prefix_copy) != 1 or \
            init_flat.count(exact_shape_copy) != 1 or init.count("hggcMemcpy(") != 2:
        bad.append("one-shot prefix/shape installs are not the two exact live blocking copies")
    if "prefix_ready" in wrapper_code or "prefix_ready" in test_code:
        bad.append("naked prefix_ready opt-out reintroduced")

    for token in (
        "params.output_tile_prefix[mid + 1] <= q",
        "int const expert = lo;",
        "int const local = q - params.output_tile_prefix[expert];",
        "int const m_idx = mt > 0 ? local % mt : 0;",
        "int const n_idx = mt > 0 ? local / mt : 0;",
        "auto const real_blk_coord = make_coord(m_idx, n_idx, _, expert);",
    ):
        if device.count(token) != 1:
            bad.append(f"device q decode missing {token!r}")
    if "get_problem_shape(expert)" in device:
        bad.append("device geometry bypasses the installed host-shape mirror")
    ordered_once(device, (
        "int const q = sched_work.M_idx;",
        "auto const real_blk_coord = make_coord(m_idx, n_idx, _, expert);",
        "TileScheduler::get_work_k_tile_start(sched_work)",
    ), "q/real-coordinate/K-start path", bad)
    device_flat = re.sub(r"\s+", " ", device)
    if ("uint32_t const k_tile_start = "
            "TileScheduler::get_work_k_tile_start(sched_work);") not in device_flat:
        bad.append("live absolute-K variable is not assigned directly from scheduler work")
    if device.count("idx2crd(k_tile_start, shape<2>(gA))") != 1 or \
            re.search(r"idx2crd\s*\((?!k_tile_start\s*,)", device):
        bad.append("mainloop K iterator is not the one exact absolute-k expression")
    k_path = device.split("uint32_t const k_tile_start =", 1)[-1]
    ordered_once(k_path, (
        "idx2crd(k_tile_start, shape<2>(gA))",
        "collective_mainloop(params.mainloop",
        "TileScheduler::requires_fixup(params.scheduler, sched_work)",
        "bool const full_output_tile =",
        "epilogue(real_problem_shape, blk_shape, real_blk_coord",
        "scheduler.fetch_next_work(sched_work)",
    ), "absolute-K/fixup/final-epilogue path", bad)
    for token, count in (
        ("TileScheduler::fixup(", 2),
        ("params.scheduler, sched_work, accumulators,", 2),
        ("detail::make_accumulator_residue_mask(", 1),
        ("if (!requires_fixup || full_output_tile)", 1),
        ("take<0, 2>(residue_mnk), thread_idx", 1),
    ):
        if device.count(token) != count:
            bad.append(f"residue-aware global-q fixup requires {count} occurrence(s) of {token!r}")
    if "should_perform_separate_reduction" in wrapper_code:
        bad.append("isolated grouped Stream-K must not wire the disabled separate-reduction path")

    phase2_oracle_flat = re.sub(r"\s+", " ", phase2_oracle)
    for token in (
        "workers < 2 * q",
        "uint64_t const total_k_tiles = uint64_t(q) * uint64_t(kt);",
        "uint64_t const units_at_min_stripe = total_k_tiles / kMinSkIters;",
        "std::min<uint64_t>(uint64_t(workers), units_at_min_stripe)",
        "units < 2ull * uint64_t(q)",
        "total_k_tiles % units != 0",
        "uint64_t const stripe_k_tiles = total_k_tiles / units;",
        "uint64_t(kt) % stripe_k_tiles != 0",
        "uint64_t const peers_per_tile = uint64_t(kt) / stripe_k_tiles;",
        "out.sk_tiles = uint32_t(q);",
        "out.peer_excess = uint64_t(q) * (peers_per_tile - 1);",
        "out.fixup_work_items = uint32_t(units);",
    ):
        if token not in phase2_oracle_flat:
            bad.append(f"independent phase-2 arithmetic oracle missing {token!r}")
    if "get_num_sk_" in phase2_oracle or "TileSchedulerParams" in phase2_oracle:
        bad.append("phase-2 expected decomposition mirrors the vendor policy instead of using an independent oracle")
    arm_flat = re.sub(r"\s+", " ", arm)
    for token in (
        "result.expected_supported = expected.supported;",
        "if (!expected.supported) {",
        '"Q=%d Kt=%d W=%d ORACLE-UNSUPPORTED/FAIL\\n"',
        "++result.errors; return result;",
        '"stripe_k_tiles=%u peers_per_tile=%u ORACLE-PASS\\n"',
    ):
        if token not in arm_flat:
            bad.append(f"phase-2 unsupported policy does not fail closed: missing {token!r}")
    if "requested=Heuristic" in test_code or "O::inspect(heuristic" in test_code:
        bad.append("box gate treats a Heuristic request rejected by can_implement as a lowered Params oracle")

    for token in (
        "args.scheduler.splits = 1;",
        "args.scheduler.max_swizzle_size = 1;",
        "RasterOrderOptions::AlongN",
        "ReductionMode::Deterministic",
        "DecompositionMode::StreamK",
    ):
        if runtime.count(token) != 1:
            bad.append(f"runtime policy missing {token!r}")
    runtime_assignments = re.findall(r"args\.([A-Za-z0-9_.]+)\s*=", runtime)
    if runtime_assignments != [
            "hw_info", "ctas_per_cu", "scheduler.splits",
            "scheduler.max_swizzle_size", "scheduler.raster_order",
            "scheduler.reduction_mode", "scheduler.decomposition_mode",
            "census"] or re.search(r"\bargs\s*=", runtime):
        bad.append("runtime configuration has an unreviewed overwrite")
    grid_assignments = re.findall(r"args\.([A-Za-z0-9_.]+)\s*=", grid)
    if grid_assignments != [
            "max_swizzle_size", "raster_order", "splits",
            "reduction_mode", "decomposition_mode"] or \
            re.search(r"\bargs\s*=", grid):
        bad.append("grid scheduler arguments have an unreviewed overwrite")
    if "int group_size" in builder_code:
        bad.append("owned builder reintroduced runtime gs beside its static schedule")
    for token in (
        "CollectiveMainloop::DispatchPolicy::StaticGroupSize",
        "args.domain_valid =",
        "!AiuInterleaved || (n % 256 == 0 && k % 256 == 0)",
        "grouped Stream-K epilogue M layout must match MMA atom and M warps",
        "class Gemm : private RawGemm",
        "return RawGemm::run(stream, host_adapter, launch_with_pdl);",
        "cutlass::Status update(Arguments const&, void* = nullptr) = delete;",
    ):
        if builder_code.count(token) != 1:
            bad.append(f"owned builder fail-close missing {token!r}")

    timed_region = section(
        arm,
        "    auto const wall_start = std::chrono::high_resolution_clock::now();",
        "    auto const wall_stop = std::chrono::high_resolution_clock::now();")
    loop_match = re.search(
        r"for \(int i = 0; i < kTimed; \+\+i\) \{(?P<body>.*?)\n    \}\n"
        r"    CUTLASS_PPU_CHECK\(hggcDeviceSynchronize\(\)\);\s*$",
        timed_region, re.DOTALL)
    if loop_match is None:
        bad.append("timed loop must end before the one final device synchronize")
        timed_body = timed_region
    else:
        timed_body = loop_match.group("body")
    ordered_once(timed_body, (
        "Kernel::reset_scheduler_workspace_after_prefix_install(",
        "hggcEventRecord(events[size_t(i) + 1].start, nullptr)",
        "gemm.run()",
        "hggcEventRecord(events[size_t(i) + 1].stop, nullptr)",
    ), "per-launch reset/event protocol", bad)
    timed_body_flat = re.sub(r"\s+", " ", timed_body)
    reset_call = (
        "Kernel::reset_scheduler_workspace_after_prefix_install( "
        "gemm.params(), workspace.get(), nullptr)")
    if arm_flat.count(reset_call) != 2 or timed_body_flat.count(reset_call) != 1:
        bad.append("warmup and every timed launch must reset the exact installed workspace via frozen Params")
    if "hggcDeviceSynchronize()" in timed_body or \
            timed_region.count("hggcDeviceSynchronize()") != 1:
        bad.append("timed batch must synchronize exactly once after all 20 stop events")
    if arm.count("constexpr int kTimed = 20;") != 1 or \
            arm.count("EventBatch events(kTimed + 1);") != 1:
        bad.append("timed batch is not 20 independent pairs plus one warmup pair")
    clean_config = "O::configure_runtime(args, device_id, real_cu, ctas_per_cu);"
    if arm.count(clean_config) != 2:
        bad.append("expected one initial and one post-census clean runtime configuration")
    else:
        clean_suffix = arm.rsplit(clean_config, 1)[1].split("if (time_arm)", 1)[0]
        if "census" in clean_suffix:
            bad.append("clean numeric/timing params reattach the census after relowering")
    ordered_once(arm, (
        "auto const wall_stop = std::chrono::high_resolution_clock::now();",
        "hggcEventElapsedTime(",
    ), "one final sync before event queries", bad)
    for token in (
        "O::configure_runtime(args, device_id, real_cu, ctas_per_cu);",
        "prefix-shape-copy-before-timing=1",
        "census-disabled=1",
        "split_tiles == int(ht[3])",
        "peer_excess == uint64_t(ht[0] - ht[3])",
        "missing_k == 0 && duplicate_k == 0",
        "[grouped streamk result] fixture=%s TK=%d Min=%u Q=%d Kt=%d W=%d",
        "errors += !host_policy_line<P8>(\"min8\", 128, 8, 432, 0, 128, 128);",
        "errors += !host_policy_line<P2>(\"min2\", 128, 8, 432, 128, 128, 432);",
        "s068.active == expected_active",
        "local_lock_collisions == 14",
        "decode_local_lock_collisions == 112",
        "errors += !router_ok;",
        "std::isfinite(us.front()) && std::isfinite(median)",
        "timing_ok ? \"TIMING-PASS\" : \"TIMING-FAIL\"",
        "if (!timing_ok) ++result.errors;",
        "return errors ? 1 : 0;",
    ):
        if test_code.count(token) < 1:
            bad.append(f"box gate missing {token!r}")
    for token in (
        "run_streamk_arm<256>",
        " Q=%d local_lock_collisions=%d ",
        "run_decode64_nonuniform(s068, ds068, device_id, real_cu)",
        "ragged-0,1,17,0,33",
    ):
        if test_code.count(token) != 1:
            bad.append(f"isolated fixture missing exactly one {token!r}")

    if test_code.count("run_streamk_arm<64>") != 2:
        bad.append("TK64 must run once on S068 and once on the ragged fixture")
    for token in (
        "using O = Decode64Op;",
        "constexpr int kExpectedQ = 128;",
        "constexpr int kExpectedKt = 8;",
        "constexpr int kExpectedWorkers = 432;",
        "constexpr size_t kBarrierBytes = 512;",
        "plan.sk_tiles == kExpectedQ && plan.sk_units == kExpectedWorkers",
        "!uniform.supported",
        "split_tiles == plan.q && holes == 0 && missing_k == 0",
        "ht[0] == peer_sum && ht[1] == uint32_t(plan.q)",
        "ht[2] == 0 && ht[3] == uint32_t(plan.q) && ht[4] == 0 && ht[5] == 0",
        '"streamk-min2-decode64-nonuniform"',
        "f, hp, kDecodeTM, kDecodeTN, &logical_exact);",
        '"MODEL-ONLY/not-a-DRAM-counter %s\\n"',
        "2ull * uint64_t(kDecodeTM) * kDecodeTN * sizeof(float) * peer_excess",
    ):
        if decode64.count(token) != 1:
            bad.append(f"decode64 nonuniform gate missing exactly one {token!r}")
    if "print_c_traffic" in decode64 or "expected_peer_excess" in decode64:
        bad.append("decode64 nonuniform arm fabricates the uniform C-traffic oracle")
    if test_code.count("Decode64Op::Kernel::MaxThreadsPerBlock == 64") != 1 or \
            test_code.count(
                "Decode64Op::Kernel::TileScheduler::FixupThreadCount == 64") != 1:
        bad.append("decode64 type gate does not pin both CTA and fixup cohort to 64")
    if test_code.count("decode_placement_diff == 0") != 1:
        bad.append("decode64 does not prove its resident artifact matches the control")
    c_traffic = section(test_code, "bool print_c_traffic", "\nint run_legacy_control")
    if re.search(r"\btrue\s*\|\||\bfalse\s*&&", test_code):
        bad.append("box oracle contains a constant short-circuit bypass")
    for token in (
        "2ull * accumulator_tile * arm.peer_excess",
        "2ull * arm.logical_fixup_elements * sizeof(float)",
        "2ull * arm.expected_logical_fixup_elements * sizeof(float)",
        "output_d + logical_workspace_rw",
        "output_d + expected_logical_workspace_rw",
        "MODEL-ONLY/not-a-DRAM-counter",
    ):
        if c_traffic.count(token) != 1:
            bad.append(f"C traffic missing exact logical/full-tile distinction {token!r}")
    if c_traffic.count(
            "arm.peer_excess == arm.expected_peer_excess") != 1:
        bad.append("C traffic does not compare measured and independently expected peers")
    if c_traffic.count("arm.expected_supported") != 1:
        bad.append("C traffic can pass after the phase-2 arithmetic oracle rejected its geometry")
    if c_traffic.count("gate_path = production_beta0 + gate_c_input") != 1:
        bad.append("nonzero-beta gate traffic does not distinguish its C input read")
    for token in (
        "for (int n_idx = 0; n_idx < nt; ++n_idx)",
        "for (int m_idx = 0; m_idx < mt; ++m_idx, ++q)",
        "peer_count[q] > 0 ? peer_count[q] - 1 : 0",
        "elements += excess * uint64_t(valid_m) * uint64_t(valid_n)",
        "ok = ok && q == peer_count.size()",
    ):
        if valid_traffic.count(token) != 1:
            bad.append(f"valid-residue traffic oracle missing exactly one {token!r}")
    verify_flat = re.sub(r"\s+", " ", verify)
    if ("return (bad == 0 && nonfinite == 0 && poison == 0 && absmax > 0.0) "
            "? 0 : 1;") not in verify_flat:
        bad.append("numeric verifier does not return its accumulated verdict")
    policy_flat = re.sub(r"\s+", " ", policy)
    if ("return ht == expected_heuristic_tiles && ft == "
            "expected_forced_tiles && fu == expected_forced_units;") not in policy_flat:
        bad.append("host policy helper does not return its exact-oracle verdict")
    for line in (
        "  if (!census_identity) ++result.errors;",
        "  if (!exact) ++result.errors;",
        "  result.errors += verify_output(",
        "    if (!timing_ok) ++result.errors;",
    ):
        if not re.search(r"^" + re.escape(line), arm, re.MULTILINE):
            bad.append(f"arm verdict is not live at its reviewed scope: {line.strip()!r}")
    if not re.search(r"^  return traffic_ok;", c_traffic, re.MULTILINE):
        bad.append("C-traffic helper does not return its oracle verdict")
    for line in (
        "  errors += !host_policy_line<P8>(\"min8\", 128, 8, 432, 0, 128, 128);",
        "  errors += !host_policy_line<P2>(\"min2\", 128, 8, 432, 128, 128, 432);",
        "  errors += !router_ok;",
        "  errors += !print_c_traffic(s068, tk256, 256);",
        "  errors += !print_c_traffic(s068, tk64, 64);",
        "  errors += !print_c_traffic(ragged, rag, 64);",
    ):
        if not re.search(r"^" + re.escape(line) + r"$", main, re.MULTILINE):
            bad.append(f"main verdict is not live at top level: {line.strip()!r}")
    for call in (
        "print_c_traffic(s068, tk256, 256)",
        "print_c_traffic(s068, tk64, 64)",
        "print_c_traffic(ragged, rag, 64)",
    ):
        if test_code.count(call) != 1:
            bad.append(f"missing independent C-traffic oracle {call!r}")
    if not re.search(r"return\s+errors\s*\?\s*1\s*:\s*0\s*;\s*$", main):
        bad.append("box main does not end in the accumulated-error verdict")
    if cmake.count("test_moe_grouped_streamk") != 4:
        bad.append("CMake target is missing or duplicated")
    if build.count("test_moe_grouped_streamk)") != 1:
        bad.append("build.sh has no unique grouped Stream-K run hint")
    return bad


def compile_l121(source: Path, include_override: Path | None = None) -> tuple[int, str]:
    with tempfile.TemporaryDirectory(prefix="qz-l121-out-") as td:
        exe = Path(td) / "l121"
        cmd = [
            "nvcc", "-std=c++17", "-x", "cu", "-arch=sm_80", "-w",
            "-D__HGGCCC__", "--expt-relaxed-constexpr",
            "-I", str(ROOT / "dev/fold_derivation/stub_inc"),
            "-I", str(ROOT / "third_party/actlize/include"),
            "-I", str(ROOT / "third_party/actlize/tools/util/include"),
        ]
        if include_override is not None:
            cmd += ["-I", str(include_override)]
        cmd += [
            "-I", str(ROOT / "quactlize/include"),
            "-I", str(ROOT / "tests"), "-I", str(ROOT / "benchmarks"),
            "-o", str(exe), str(source),
        ]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        log = p.stdout + p.stderr
        if p.returncode == 0:
            r = subprocess.run([str(exe)], capture_output=True, text=True)
            return r.returncode, log + r.stdout + r.stderr
        return p.returncode, log


def compiled_controls() -> list[str]:
    bad: list[str] = []
    rc, log = compile_l121(L121)
    if rc != 0 or "Params=exact PASS" not in log:
        bad.append("L121 positive wrapper specialization did not build/run")

    with tempfile.TemporaryDirectory(prefix="qz-l121-plant-") as td:
        td = Path(td)
        inc = td / "include"
        planted_wrapper = inc / WRAPPER.relative_to(ROOT / "quactlize/include")
        planted_wrapper.parent.mkdir(parents=True)
        text = WRAPPER.read_text()
        old = ("using TileScheduler = detail::PersistentTileSchedulerPPUStreamK<\n"
               "      TileShape, ClusterShape, MinSkIters, MaxThreadsPerBlock>;")
        new = ("using TileScheduler = detail::PersistentTileSchedulerPPUStreamK<\n"
               "      TileShape, ClusterShape>;")
        if text.count(old) != 1:
            bad.append("cannot plant default-Params scheduler regression")
        else:
            planted_wrapper.write_text(text.replace(old, new, 1))
            shutil.copy2(BUILDER, inc / BUILDER.name)
            rc, log = compile_l121(L121, inc)
            expected = "grouped Stream-K lost configured MinSkIters"
            if rc == 0 or expected not in log:
                bad.append("default Params plant did not fail with the exact owned-wrapper assertion")

        min1 = td / "l121_min1.cu"
        src = L121.read_text()
        old_min = "3, true, cutlass::int4b_t, void, 64, 2u>;"
        if src.count(old_min) != 1:
            bad.append("cannot plant MinSkIters=1 pipeline regression")
        else:
            min1.write_text(src.replace(old_min,
                                        "3, true, cutlass::int4b_t, void, 64, 1u>;", 1))
            rc, log = compile_l121(min1)
            expected = "Stream-K stripe is shorter than pipeline startup"
            if rc == 0 or expected not in log:
                bad.append("Min=1 plant did not fail with the pipeline-startup assertion")

        no_update = td / "l121_no_update.cu"
        update_site = "int main() {"
        update_call = (
            "int main() {\n"
            "  Operation::Gemm gemm;\n"
            "  Operation::Arguments args{};\n"
            "  (void)gemm.update(args, nullptr);")
        if src.count(update_site) != 1:
            bad.append("cannot compile the deleted-update control")
        else:
            no_update.write_text(src.replace(update_site, update_call, 1))
            rc, _ = compile_l121(no_update)
            if rc == 0:
                bad.append("owned grouped handle unexpectedly exposes update()")

        no_params_run = td / "l121_no_params_run.cu"
        params_run_call = (
            "int main() {\n"
            "  Operation::Gemm gemm;\n"
            "  Operation::Params params{};\n"
            "  (void)gemm.run(params);")
        if src.count(update_site) != 1:
            bad.append("cannot compile the Params-run negative control")
        else:
            no_params_run.write_text(src.replace(update_site, params_run_call, 1))
            rc, _ = compile_l121(no_params_run)
            if rc == 0:
                bad.append("owned grouped handle unexpectedly exposes run(Params&)")
    return bad


def main() -> int:
    paths = (WRAPPER, BUILDER, TEST, CMAKE, BUILD, L121)
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    if missing:
        print("[grouped-streamk-contract] FAIL: missing " + ", ".join(missing))
        return 1
    texts = [p.read_text() for p in (WRAPPER, BUILDER, TEST, CMAKE, BUILD)]
    bad = audit(*texts)

    # Source-level plants exercise the structural half of the checker itself.
    plants = [
        # Keep the old anchor present wherever possible: these controls must
        # reject changed semantics, not merely notice a deleted phrase.
        (0, "q += mt * nt;", "q += mt * nt; q = mt;",
         "prefix is overwritten after the right accumulation"),
        (0, "(*prefix)[size_t(e) + 1] = int(q);",
         "(*prefix)[size_t(e) + 1] = 0;",
         "prefix entries discard the accumulated q"),
        (0, "out.q = int(q);", "out.q = 1;",
         "scheduler geometry discards the accumulated q"),
        (0,
         "int64_t const workers = int64_t(real.cu_count) * int64_t(args.ctas_per_cu);",
         "int64_t const workers = int64_t(real.cu_count) + int64_t(args.ctas_per_cu);",
         "physical worker population is added instead of multiplied"),
        (0, "params.scheduler_hw_info, args);",
         "params.real_hw_info, args); (void)params.scheduler_hw_info;",
         "grid uses real CU while retaining the worker token"),
        (0, "int const expert = lo;", "int const expert = 0; (void)lo;",
         "device q decode pins every tile to expert zero"),
        (0, "idx2crd(k_tile_start, shape<2>(gA))",
         "(false ? idx2crd(k_tile_start, shape<2>(gA)) : "
         "idx2crd(0, shape<2>(gA)))", "live K slice restarts at zero"),
        (0,
         "uint32_t const k_tile_start =\n"
         "          TileScheduler::get_work_k_tile_start(sched_work);",
         "uint32_t const observed_k_tile_start =\n"
         "          TileScheduler::get_work_k_tile_start(sched_work);\n"
         "      uint32_t const k_tile_start = 0;\n"
         "      (void)observed_k_tile_start;",
         "scheduler K result is observed but the live iterator restarts at zero"),
        (0, "out.prefix_bytes = sizeof(int) * size_t(g.groups + 1);",
         "out.prefix_bytes = sizeof(int) * size_t(g.groups + 1); "
         "out.prefix_offset = 0;", "prefix overlaps scheduler scratch"),
        (0, "out.prefix_bytes = sizeof(int) * size_t(g.groups + 1);",
         "out.prefix_bytes = sizeof(int) * size_t(g.groups);",
         "prefix allocation drops its final sentinel"),
        (0, "out.shape_bytes = sizeof(UnderlyingProblemShape) * size_t(g.groups);",
         "out.shape_bytes = sizeof(UnderlyingProblemShape) * size_t(g.groups - 1);",
         "shape mirror drops the final expert"),
        (0, "args.scheduler, shape, scheduler_hw_info(args), NumMmaWarpGroups);",
         "args.scheduler, shape, false ? scheduler_hw_info(args) : "
         "real_hw_info(args), NumMmaWarpGroups);",
         "workspace sizing uses real CU while retaining worker code"),
        (0, "args.scheduler, shape, scheduler_hw_info(args), NumMmaWarpGroups);",
         "args.scheduler, shape, scheduler_hw_info(args), NumMmaWarpGroups) / 2;",
         "scheduler workspace size is silently halved"),
        (0,
         "return TileScheduler::template get_workspace_size<SchedulerProblemShape,\n"
         "                                                       ElementAccumulator>(\n"
         "        args.scheduler, shape, scheduler_hw_info(args), NumMmaWarpGroups);",
         "if (false) {\n"
         "      return TileScheduler::template get_workspace_size<SchedulerProblemShape,\n"
         "                                                       ElementAccumulator>(\n"
         "        args.scheduler, shape, scheduler_hw_info(args), NumMmaWarpGroups);\n"
         "    }\n"
         "    return 0;",
         "scheduler workspace sizing is retained only in a dead branch"),
        (0, "return workspace_layout(args, g).total_bytes;",
         "return workspace_layout(args, g).scheduler_bytes;",
         "public workspace size drops prefix, shape, and epilogue"),
        (0, "if (layout.total_bytes > 0 && workspace == nullptr) {",
         "if (false && layout.total_bytes > 0 && workspace == nullptr) {",
         "full initialization disables its null-workspace guard"),
        (0, "return Status::kErrorWorkspaceNull;",
         "return Status::kSuccess;",
         "null-workspace guard reports success"),
        (0,
         "Status status =\n"
         "        TileScheduler::template initialize_workspace<SchedulerProblemShape,\n"
         "                                                     ElementAccumulator>(\n"
         "            args.scheduler, workspace_ptr, stream, shape, workers,\n"
         "            NumMmaWarpGroups, /*epilogue_subtile=*/1,\n"
         "            /*num_accumulator_mtxs=*/1, host_adapter);",
         "Status status = false\n"
         "        ? TileScheduler::template initialize_workspace<SchedulerProblemShape,\n"
         "                                                     ElementAccumulator>(\n"
         "            args.scheduler, workspace_ptr, stream, shape, workers,\n"
         "            NumMmaWarpGroups, /*epilogue_subtile=*/1,\n"
         "            /*num_accumulator_mtxs=*/1, host_adapter)\n"
         "        : Status::kSuccess;",
         "full scheduler initialization is retained only in a dead branch"),
        (0, "if (status != Status::kSuccess) {",
         "if (status == Status::kSuccess) {",
         "scheduler initialization success exits before metadata install"),
        (0,
         "Status status =\n"
         "        TileScheduler::template initialize_workspace<SchedulerProblemShape,",
         "if (false) {\n"
         "    Status status =\n"
         "        TileScheduler::template initialize_workspace<SchedulerProblemShape,",
         "full scheduler initialization is enclosed in a dead branch"),
        (0, "workspace_ptr + layout.prefix_offset, prefix.data()",
         "workspace_ptr, prefix.data()",
         "prefix copy overwrites scheduler scratch"),
        (0,
         "hggcError_t copy_status = hggcMemcpy(\n"
         "        workspace_ptr + layout.prefix_offset, prefix.data(),\n"
         "        layout.prefix_bytes, hggcMemcpyHostToDevice);",
         "hggcError_t copy_status = false ? hggcMemcpy(\n"
         "        workspace_ptr + layout.prefix_offset, prefix.data(),\n"
         "        layout.prefix_bytes, hggcMemcpyHostToDevice) : hggcSuccess;",
         "prefix install is retained only in a dead branch"),
        (0,
         "copy_status = hggcMemcpy(\n"
         "        workspace_ptr + layout.shape_offset,\n"
         "        args.problem_shape.host_problem_shapes, layout.shape_bytes,\n"
         "        hggcMemcpyHostToDevice);",
         "copy_status = false ? hggcMemcpy(\n"
         "        workspace_ptr + layout.shape_offset,\n"
         "        args.problem_shape.host_problem_shapes, layout.shape_bytes,\n"
         "        hggcMemcpyHostToDevice) : hggcSuccess;",
         "shape install is retained only in a dead branch"),
        (0, "params.scheduler_workspace_bytes - params.scheduler_barrier_bytes ||",
         "params.scheduler_workspace_bytes - params.scheduler_barrier_bytes + 16 ||",
         "frozen barrier-tail boundary is shifted"),
        (0, "params.scheduler.reduction_workspace_ != params.workspace_base ||",
         "false && params.scheduler.reduction_workspace_ != params.workspace_base ||",
         "frozen scheduler scratch identity is disabled"),
        (0, "if (params.scheduler_barrier_bytes == 0) {",
         "if (true || params.scheduler_barrier_bytes == 0) {",
         "frozen barrier reset always skips the clear"),
        (0, "workspace != params.workspace_base ||",
         "false && workspace != params.workspace_base ||",
         "frozen reset disables workspace identity"),
        (0,
         "hggcError_t const status = hggcMemsetAsync(\n"
         "        static_cast<uint8_t*>(workspace) + params.scheduler_barrier_offset, 0,\n"
         "        params.scheduler_barrier_bytes, stream);",
         "hggcError_t const status = false ? hggcMemsetAsync(\n"
         "        static_cast<uint8_t*>(workspace) + params.scheduler_barrier_offset, 0,\n"
         "        params.scheduler_barrier_bytes, stream) : hggcSuccess;",
         "frozen barrier reset keeps the clear only in a dead branch"),
        (0, "hggcError_t const status = hggcMemsetAsync(\n",
         "if (false) {\n    hggcError_t const status = hggcMemsetAsync(\n",
         "frozen barrier reset is enclosed in a dead branch"),
        (0,
         "static_cast<uint8_t*>(workspace) + params.scheduler_barrier_offset, 0,\n"
         "        params.scheduler_barrier_bytes, stream);",
         "workspace, 0, params.scheduler_workspace_bytes, stream);",
         "timed reset clears reduction scratch as well as barriers"),
        (0,
         "TileScheduler::fixup(params.scheduler, sched_work, accumulators,\n"
         "                             NumMmaWarpGroups, 0);",
         "TileScheduler::fixup(params.scheduler, real_work, accumulators,\n"
         "                             NumMmaWarpGroups, 0);",
         "local work aliases locks"),
        (0, "static_assert(TileScheduler::FixupThreadCount == MaxThreadsPerBlock",
         "static_assert(TileScheduler::FixupThreadCount != MaxThreadsPerBlock",
         "wrong barrier cohort"),
        (2, "return errors ? 1 : 0;",
         "if (false) return errors ? 1 : 0; return 0;",
         "dead error verdict precedes an unconditional success"),
        (2,
         "return (bad == 0 && nonfinite == 0 && poison == 0 && absmax > 0.0) ? 0 : 1;",
         "return 0;", "numeric verifier ignores mismatches"),
        (2, "if (got[i].raw() != want[i].raw()) {",
         "if (false && got[i].raw() != want[i].raw()) {",
         "numeric comparator is permanently disabled"),
        (2, "bool const census_identity =",
         "bool const census_identity = true ||",
         "census oracle is permanently true"),
        (2, "bool const exact =", "bool const exact = true ||",
         "exact decomposition oracle is permanently true"),
        (2, "bool const timing_ok =", "bool const timing_ok = true ||",
         "timing validity oracle is permanently true"),
        (2, "bool const router_ok =", "bool const router_ok = true ||",
         "router oracle is permanently true"),
        (2, "bool const traffic_ok =", "bool const traffic_ok = true ||",
         "traffic oracle is permanently true"),
        (2, "constexpr uint32_t kMinSkIters = 2;",
         "constexpr uint32_t kMinSkIters = 8;",
         "live grouped specialization falls back to Min=8"),
        (2,
         "kStages, true, int4_t, void, kArtifactTileK, kMinSkIters>;",
         "kStages, true, int4_t, void, kArtifactTileK>;",
         "live grouped specialization omits Min and takes the default"),
        (2, "total_k_tiles / kMinSkIters",
         "total_k_tiles / 8u",
         "phase-2 oracle silently retains the phase-1 stripe"),
        (2, "out.peer_excess = uint64_t(q) * (peers_per_tile - 1);",
         "out.peer_excess = 0; (void)units;",
         "phase-2 oracle erases the expected peer excess"),
        (2, "arm.peer_excess == arm.expected_peer_excess",
         "arm.peer_excess == arm.peer_excess",
         "traffic oracle compares the census to itself"),
        (2, "!uniform.supported;", "uniform.supported;",
         "decode64 accepts the rejected uniform peer oracle"),
        (2,
         "split_tiles == plan.q && holes == 0 && missing_k == 0 &&",
         "split_tiles == plan.q && holes == 0 && missing_k >= 0 &&",
         "decode64 stops requiring exact K coverage"),
        (2, "decode_local_lock_collisions == 112",
         "decode_local_lock_collisions >= 0",
         "decode64 lock-alias witness becomes vacuous"),
        (2, "decode_placement_diff == 0",
         "decode_placement_diff == decode_placement_diff",
         "decode64 artifact comparison becomes self-referential"),
        (2,
         "f, hp, kDecodeTM, kDecodeTN, &logical_exact);",
         "uint64_t(kDecodeTM) * kDecodeTN * peer_excess",
         "decode64 replaces the measured per-q residue model with full tiles"),
        (2, "if (!census_identity) ++result.errors;",
         "if (false && !census_identity) ++result.errors;",
         "census failure is moved into a dead branch"),
        (2,
         "errors += !host_policy_line<P8>(\"min8\", 128, 8, 432, 0, 128, 128);",
         "if (false) errors += !host_policy_line<P8>(\"min8\", 128, 8, 432, 0, 128, 128);",
         "min8 policy failure is moved into a dead branch"),
        (2, "census-disabled=1", "census-disabled=0", "timing retains census atomics"),
        (1, "args.census = census;",
         "args.census = census; args.scheduler = {};",
         "runtime policy is cleared after the reviewed assignments"),
        (1, "class Gemm : private RawGemm",
         "class Gemm : public RawGemm",
         "owned handle permits upcasting around the deleted update seam"),
        (1, "cutlass::Status update(Arguments const&, void* = nullptr) = delete;",
         "cutlass::Status update(Arguments const&, void* = nullptr);",
         "owned handle no longer deletes update"),
        (1, "return RawGemm::run(stream, host_adapter, launch_with_pdl);",
         "return RawGemm::run(params(), stream, host_adapter, launch_with_pdl);",
         "owned handle regains a raw Params launch seam"),
        (2, "constexpr int kTimed = 20;", "constexpr int kTimed = 1;",
         "timing silently shrinks to one launch"),
        (2,
         "gemm.params(), workspace.get(), nullptr));\n"
         "      CUTLASS_PPU_CHECK(hggcEventRecord(events[size_t(i) + 1].start",
         "gemm.params(), workspace.get() + 16, nullptr));\n"
         "      CUTLASS_PPU_CHECK(hggcEventRecord(events[size_t(i) + 1].start",
         "timed reset shifts away from the installed workspace base"),
        (2, "2ull * arm.logical_fixup_elements * sizeof(float)",
         "2ull * accumulator_tile * arm.peer_excess",
         "C traffic replaces valid rows with full accumulator tiles"),
        (2, "elements += excess * uint64_t(valid_m) * uint64_t(valid_n)",
         "elements += excess * uint64_t(tile_m) * uint64_t(tile_n)",
         "per-q traffic oracle ignores ragged output residues"),
        (2, "gate_beta=%.3g gate_C_read=%llu gate_C_path=%llu \"\n"
            "              \"MODEL-ONLY/not-a-DRAM-counter %s\\n\",",
         "gate_beta=%.3g gate_C_read=%llu gate_C_path=%llu measured-DRAM-bytes %s\\n\",",
         "logical workspace accesses are mislabeled as a DRAM counter"),
        (2,
         "hggcEventRecord(events[size_t(i) + 1].stop, nullptr));\n    }\n"
         "    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());",
         "hggcEventRecord(events[size_t(i) + 1].stop, nullptr));\n"
         "      CUTLASS_PPU_CHECK(hggcDeviceSynchronize());\n    }",
         "each timed launch synchronizes separately"),
    ]
    for index, old, new, label in plants:
        planted = list(texts)
        if planted[index].count(old) != 1:
            bad.append(f"cannot plant {label}")
            continue
        planted[index] = planted[index].replace(old, new, 1)
        if not audit(*planted):
            bad.append(f"checker accepted planted {label}")

    bad.extend(compiled_controls())
    if bad:
        print("[grouped-streamk-contract] FAIL: " + "; ".join(bad))
        return 1
    print("[grouped-streamk-contract] PASS -- explicit Min2, independent worker "
          "oracle, q lock identity, shared workers, workspace/prefix separation, "
          "absolute K, exact-CTA fixup; "
          f"{len(plants)} semantic source plants + default-Params/Min1/"
          "deleted-update/no-Params-run compiled controls rejected; device body is "
          "covered separately by the registered SYNTAX target")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
