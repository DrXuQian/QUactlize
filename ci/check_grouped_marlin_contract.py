#!/usr/bin/env python3
"""Pin grouped Marlin's ragged-q/lock/collective seams."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_group_marlin.hpp"
BUILDER = ROOT / "quactlize/include/moe_grouped_marlin_ppu.cuh"
GEOMETRY = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_grouped_ragged_geometry.hpp"
TYPES = ROOT / "dev/fold_derivation/l135_grouped_marlin_types.cu"


def validate(kernel: str, builder: str, geometry: str, types: str) -> list[str]:
    bad: list[str] = []
    required_kernel = (
        "class GroupMarlinMixedInputKernel",
        "PersistentTileSchedulerPPUMarlin<",
        "GroupedRaggedOutputTiles::append_group(",
        "GroupedRaggedOutputTiles::decode_expert(",
        "GroupedRaggedOutputTiles::decode_local_mn(",
        "make_shape(g.q * int(cute::size<0>(TileShape{})),",
        "TileScheduler::get_grid_shape(params.scheduler)",
        "TileScheduler::fixup(params.scheduler, sched_work, accumulators",
        "epilogue(real_problem_shape, blk_shape, real_blk_coord, accumulators",
        "args.mainloop.group_row_offsets != nullptr",
        "args.scheduler.blocks_per_cu == 1",
        "TileScheduler::fixup_thread_count_capable(MaxThreadsPerBlock)",
    )
    for token in required_kernel:
        if token not in kernel:
            bad.append("kernel missing " + token)
    forbidden_kernel = (
        "the first Marlin scheduler wiring is dense-only",
        "ctas_per_cu",
        "scheduler_hw_info",
        "MinSkIters",
        "PersistentTileSchedulerPPUStreamK",
    )
    for token in forbidden_kernel:
        if token in kernel:
            bad.append("kernel retained foreign/dense seam " + token)
    if kernel.count("params.scheduler, sched_work, accumulators") != 2:
        bad.append("both full/residue fixup arms must preserve sched_work/global q")
    if "real_blk_coord" not in kernel or "sched_work.M_idx" not in kernel:
        bad.append("global scheduler and expert-local coordinate systems not both explicit")

    required_geometry = (
        "out = mt * nt;",
        "if (prefix[mid + 1] <= q)",
        "return lo < groups ? lo : -1;",
    )
    for token in required_geometry:
        if token not in geometry:
            bad.append("geometry missing " + token)

    required_builder = (
        "GroupMarlinMixedInputKernel<",
        "kernel_policy_valid_v<",
        "ppu_tactics::GroupedSpace",
        "args.mainloop.ptr_B2 = B2;",
        "args.mainloop.dB2_valid = true;",
        "cutlass::make_cute_packed_stride(",
        "cutlass::gemm::GemmUniversalMode::kGrouped",
        "cutlass::Status update(Arguments const&, void* = nullptr) = delete;",
    )
    for token in required_builder:
        if token not in builder:
            bad.append("builder missing " + token)

    required_types = (
        "using Ordinary =",
        "using Folded =",
        "using TwoPlane =",
        "ArtifactLowFold == 1",
        "ArtifactLowFold == 2",
        "HighBits == 1",
        "ArtifactHighFold == 4",
    )
    for token in required_types:
        if token not in types:
            bad.append("type oracle missing " + token)
    return bad


def compile_types() -> tuple[int, str]:
    with tempfile.TemporaryDirectory(prefix="quactlize-l135-") as td:
        out = Path(td) / "l135.cu.cpp"
        cmd = [
            "nvcc", "-std=c++17", "-arch=sm_80", "--expt-relaxed-constexpr",
            "-D__HGGCCC__", "-DPPU_FORCE_INSTANTIATE=1",
            "-Xcudafe", "--error_limit=100000",
            f"-I{ROOT / 'dev/fold_derivation/stub_inc'}",
            f"-I{ROOT / 'third_party/actlize/include'}",
            f"-I{ROOT / 'third_party/actlize/tools/util/include'}",
            f"-I{ROOT / 'tests'}", f"-I{ROOT / 'benchmarks'}",
            f"-I{ROOT / 'quactlize/include'}", f"-I{ROOT / 'dev'}",
            "-cuda", "-o", str(out), "-x", "cu", str(TYPES),
            "-Wno-deprecated-gpu-targets",
        ]
        run = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        return run.returncode, run.stdout


def main() -> int:
    paths = (KERNEL, BUILDER, GEOMETRY, TYPES)
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    if missing:
        print("[grouped-marlin-contract] FAIL: missing " + ", ".join(missing))
        return 1
    source = [p.read_text() for p in paths]
    bad = validate(*source)
    if bad:
        print("[grouped-marlin-contract] FAIL: " + "; ".join(bad))
        return 1

    plants = (
        (0, "global-q-fixup", "params.scheduler, sched_work, accumulators",
         "params.scheduler, real_work, accumulators", 2),
        (0, "uniform-M-prefix", "GroupedRaggedOutputTiles::append_group(",
         "GroupedRaggedOutputTiles::group_tile_count(", 1),
        (0, "occupancy-workers", "TileScheduler::get_grid_shape(params.scheduler)",
         "TileScheduler::get_grid_shape(params.scheduler, ctas_per_cu)", 1),
        (0, "dense-B-leaks-into-grouped", "args.scheduler.blocks_per_cu == 1",
         "args.scheduler.blocks_per_cu > 0", 1),
        (2, "empty-expert-lower-bound", "if (prefix[mid + 1] <= q)",
         "if (prefix[mid + 1] < q)", 1),
        (1, "drop-two-plane", "args.mainloop.ptr_B2 = B2;", "(void)B2;", 1),
    )
    for idx, label, old, new, count in plants:
        if source[idx].count(old) != count:
            print(f"[grouped-marlin-contract] FAIL: cannot plant {label}")
            return 1
        planted = list(source)
        planted[idx] = planted[idx].replace(old, new, count)
        if not validate(*planted):
            print(f"[grouped-marlin-contract] FAIL: plant {label} stayed green")
            return 1

    rc, log = compile_types()
    if rc != 0:
        print("[grouped-marlin-contract] FAIL: ordinary/fold/two-plane type compile\n" + log[-3000:])
        return 1
    print("[grouped-marlin-contract] PASS: ragged prefix/global-q lock and default-B1 scheduler-owned G are pinned; ordinary/fold/two-plane instantiate one grouped Marlin kernel; six plants rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
