#!/usr/bin/env python3
"""Local contract for the additive dense Marlin scheduler/cooperative.

The device result belongs to ppu001.  This gate pins everything layout and
integer algebra can decide locally: K-fast exact-once stripes, scheduler-owned
Q-vs-CU launch protection, reverse peer order, global-q locks, exact fixup
cohort, the unchanged mixed-input collective, and the same-binary/event A/B/C
route.  Its red controls are structural mutations, not alternate expected
outputs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "third_party/actlize/include/cutlass/gemm/kernel/ppu_tile_scheduler_marlin_core.hpp"
SCHED = ROOT / "third_party/actlize/include/cutlass/gemm/kernel/ppu_tile_scheduler_marlin.hpp"
SELECTOR = ROOT / "third_party/actlize/include/cutlass/gemm/kernel/tile_scheduler.hpp"
KERNEL = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_marlin.hpp"
BENCH = ROOT / "benchmarks/test_lowbit_dense_bench.cu"
DISPATCH = ROOT / "benchmarks/lowbit_dense_unit.inc"
UNIT = ROOT / "dev/fold_derivation/test_lowbit_dense_unit.cu"
CMAKE = ROOT / "quactlize/csrc/CMakeLists.txt.in"
BUILD = ROOT / "build.sh"
BOX = ROOT / "tools/run_dense_marlin_box.sh"
L126 = ROOT / "dev/fold_derivation/run_l126_marlin_scheduler.sh"


def exact(text: str, token: str, count: int, bad: list[str], label: str) -> None:
    got = text.count(token)
    if got != count:
        bad.append(f"{label}: expected {count} occurrence(s) of {token!r}, got {got}")


def audit(files: dict[str, str]) -> list[str]:
    c, s, sel, k = (files[x] for x in ("core", "sched", "selector", "kernel"))
    b, d, u, cm, build, box = (
        files[x] for x in ("bench", "dispatch", "unit", "cmake", "build", "box"))
    bad: list[str] = []

    for token in (
        "p.grid_blocks_ = p.output_tiles_ >= cu_count ? p.output_tiles_ : cu_count;",
        "p.iters_per_block_ = ceil_div_u64(p.total_k_tiles_, p.grid_blocks_);",
        "uint64_t const q = cursor / p.k_tiles_per_output_;",
        "uint64_t const k = cursor % p.k_tiles_per_output_;",
        "out.slice_idx = uint32_t(last_peer - block_idx);",
        "out.lock_idx = q;",
        "out.N_idx = int32_t(q_mn % p.tiles_n_);",
        "out.M_idx = int32_t(q_mn % p.tiles_m_);",
        "out.L_idx = int32_t(q_mn / p.tiles_m_);",
    ):
        exact(c, token, 1, bad, "scheduler core")
    if "cu_count *" in c or "ctas_per_cu" in c:
        bad.append("scheduler core lets occupancy multiply its CU-owned grid")

    for token in (
        "fixup_thread_count_capable(",
        "thread_count >= uint32_t(cutlass::NumThreadsPerWarp)",
        "thread_count <= 32u * uint32_t(cutlass::NumThreadsPerWarp)",
        "thread_count % uint32_t(cutlass::NumThreadsPerWarp) == 0",
        "FixupThreadCount == 0 ? DerivedThreadCount : FixupThreadCount",
        "FixupThreadCount == 0 || FixupThreadCount == DerivedThreadCount",
        "Cohort == DerivedThreadCount",
        "using BarrierManager = NamedBarrierManager<\n        Cohort,",
        "BarrierManager::ThreadCount == Cohort",
        "using Striped = BlockStripedReduce<Cohort, AccumulatorArray>;",
        "uint32_t const thread = uint32_t(threadIdx.x);",
        "Striped::store(workspace_array, *accumulator_array, thread, predicate);",
        "BarrierManager::wait_eq(0, locks, thread, lock, work.slice_idx);",
        "Striped::load_add(*accumulator_array, workspace_array, thread, predicate);",
        "BarrierManager::wait_eq_reset(0, locks, thread, lock, work.slice_idx);",
        "work.output_tile_idx * tile_elements",
        "int const lock = int(work.lock_idx);",
    ):
        if token not in s:
            bad.append(f"cooperative is missing {token!r}")
    exact(s, "BarrierManager::arrive_inc(0, locks, thread, lock, 1);", 2, bad, "peer chain")
    if "threadIdx.x) % Cohort" in s:
        bad.append("Marlin fixup aliases surplus CTA threads through a cohort modulo")
    if "atomic_add" in s or "BlockStripedReduceT::reduce" in s:
        bad.append("Marlin cooperative silently became the Stream-K atomic protocol")

    for token in (
        "struct MarlinScheduler { };",
        "MarlinScheduler,",
        "PersistentTileSchedulerPPUMarlin<TileShape, ClusterShape>",
    ):
        exact(sel, token, 1, bad, "additive selector")

    for token in (
        "class MarlinMixedInputKernel",
        "using ArchTag = typename CollectiveMainloop::ArchTag;",
        "using TileSchedulerTag = MarlinScheduler;",
        "TileShape, ClusterShape, MaxThreadsPerBlock",
        "MaxThreadsPerBlock == uint32_t(cute::size(TiledMma{}))",
        "TileScheduler::fixup_thread_count_capable(MaxThreadsPerBlock)",
        "TileScheduler::FixupThreadCount == MaxThreadsPerBlock",
        "static constexpr bool IsDenseMarlin = true;",
        "TileScheduler::get_work_k_tile_start(work_tile_info)",
        "TileScheduler::get_work_k_tile_count(",
        "collective_mainloop(params.mainloop, load_inputs, accumulators,",
        "detail::make_accumulator_residue_mask(",
        "TileScheduler::compute_epilogue(work_tile_info, params.scheduler)",
    ):
        if token not in k:
            bad.append(f"named kernel is missing {token!r}")
    for forbidden in ("MixGemmEmit", "Converter", "dequant", "fold_for"):
        if forbidden in k:
            bad.append(f"scheduler kernel reached format logic {forbidden!r}")

    for token in (
        "#include \"quactlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_marlin.hpp\"",
        "using MarlinGemm = cutlass::gemm::device::GemmUniversalAdapter<MarlinKernel>;",
        "struct dense_is_marlin_gemm",
        "expected = logical_ctas >= uint64_t(cu_count)",
        "occupancy must not multiply it",
        "[dense marlin decomposition]",
        "Marlin-C valid_elements=%llu peer_excess=%llu",
        "MODEL-ONLY/not-a-DRAM-counter",
        "--marlin",
    ):
        if token not in b:
            bad.append(f"benchmark route is missing {token!r}")
    exact(d, "options.marlin", 1, bad, "generated dispatch")
    for token in (
        "Kernel::TileScheduler::fixup_thread_count_capable(",
        "Kernel::TileScheduler::FixupThreadCount ==",
        "Kernel::MaxThreadsPerBlock",
    ):
        if token not in d:
            bad.append(f"generated Marlin wrapper is missing {token!r}")
    exact(u, "X(lowbit_dense_marlin_probe,16,128,128,16,32,3,0)", 1, bad, "local unit")

    for token in (
        "set(_DENSE_MARLIN_ARTIFACT_TK 64)",
        "set(_DENSE_MARLIN_TK 128)",
        "set(_DENSE_MARLIN_TN 128)",
        "DENSE_AB_ARTIFACT_TK=${_DENSE_MARLIN_ARTIFACT_TK}",
        "DENSE_MARLIN_AB=1 DENSE_STREAMK_AB=1 BENCH_GS=128",
        "test_lowbit_dense_marlin_ab",
    ):
        if token not in cm:
            bad.append(f"CMake route is missing {token!r}")
    exact(build, '[ "$TARGET" = "test_lowbit_dense_marlin_ab" ]', 1, bad, "build route")

    for token in (
        "--m=1 --n=4096 --k=4096",
        "run_arm non-persistent",
        "run_arm streamk --streamk",
        "run_arm marlin --marlin",
        "distinct-event-pairs=20",
        "lock-reset-before-start=1",
        "lock-reset-before-start=0",
        "handoffs=[1-9][0-9]*",
    ):
        if token not in box:
            bad.append(f"box comparison is missing {token!r}")
    return bad


def main() -> int:
    files = {
        "core": CORE.read_text(), "sched": SCHED.read_text(),
        "selector": SELECTOR.read_text(), "kernel": KERNEL.read_text(),
        "bench": BENCH.read_text(), "dispatch": DISPATCH.read_text(),
        "unit": UNIT.read_text(), "cmake": CMAKE.read_text(),
        "build": BUILD.read_text(), "box": BOX.read_text(),
    }
    bad = audit(files)

    plants = (
        ("core", "grid-times-occupancy",
         "p.grid_blocks_ = p.output_tiles_ >= cu_count ? p.output_tiles_ : cu_count;",
         "p.grid_blocks_ = cu_count * 4;"),
        ("core", "iters-minus-one",
         "p.iters_per_block_ = ceil_div_u64(p.total_k_tiles_, p.grid_blocks_);",
         "p.iters_per_block_ = ceil_div_u64(p.total_k_tiles_, p.grid_blocks_) - 1;"),
        ("core", "local-lock",
         "out.lock_idx = q;", "out.lock_idx = uint64_t(out.N_idx);"),
        ("core", "natural-peer-order",
         "out.slice_idx = uint32_t(last_peer - block_idx);",
         "out.slice_idx = uint32_t(block_idx - first_peer);"),
        ("sched", "capability-over-cta-limit",
         "thread_count <= 32u * uint32_t(cutlass::NumThreadsPerWarp)",
         "thread_count <= 64u * uint32_t(cutlass::NumThreadsPerWarp)"),
        ("sched", "barrier-default-128",
         "using BarrierManager = NamedBarrierManager<\n        Cohort,",
         "using BarrierManager = NamedBarrierManager<\n        128,"),
        ("sched", "reducer-default-128",
         "using Striped = BlockStripedReduce<Cohort, AccumulatorArray>;",
         "using Striped = BlockStripedReduce<128, AccumulatorArray>;"),
        ("sched", "weak-derived-binding",
         "static_assert(Cohort == DerivedThreadCount,",
         "static_assert(Cohort % DerivedThreadCount == 0,"),
        ("sched", "thread-modulo-alias",
         "uint32_t const thread = uint32_t(threadIdx.x);",
         "uint32_t const thread = uint32_t(threadIdx.x) % Cohort;"),
        ("kernel", "kernel-drops-explicit-cohort",
         "TileShape, ClusterShape, MaxThreadsPerBlock>;",
         "TileShape, ClusterShape>;"),
    )
    for owner, label, old, new in plants:
        if files[owner].count(old) != 1:
            bad.append(f"cannot plant {label}: source anchor missing/duplicated")
            continue
        planted = dict(files)
        planted[owner] = planted[owner].replace(old, new)
        if not audit(planted):
            bad.append(f"contract accepted planted {label} regression")

    run = subprocess.run(
        ["bash", str(L126)], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    required = (
        "classic Q=16 Kt=16 CU=20 G=20 I=13 active=20 last=9 handoff=18 cross-CTA=14 -> PASS",
        "decode Q=128 Kt=8 CU=72 G=128 I=8 handoff=0 -> PASS",
        "grid*occupancy G=288 handoff=128 -> EXPECTED-RED",
        "I-1 holes=16 dup=0 owner=0 -> EXPECTED-RED",
        "natural-peer mismatches=32 -> EXPECTED-RED",
        "default persistent/StreamK selector types unchanged; result=PASS",
    )
    if run.returncode != 0:
        bad.append(f"l126 returned {run.returncode}: {run.stdout[-1000:]}")
    for token in required:
        if token not in run.stdout:
            bad.append(f"l126 output is missing {token!r}")

    if bad:
        print("[dense-marlin-contract] FAIL: " + "; ".join(bad))
        return 1
    print("[dense-marlin-contract] PASS -- additive K-fast scheduler, reverse q-lock cooperative, "
          "exact cohort, artifact/tactic split, and same-event DP/SK/Marlin route; "
          "ten structural plants rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
