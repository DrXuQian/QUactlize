#!/usr/bin/env python3
"""Static and arithmetic closure for the fixed Split-K N-on-x experiment."""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "quactlize/include/actlize_extensions/cutlass/gemm/kernel/"
    "ppu_aiu_gemm_mixed_input_splitk_parallel.hpp")
BENCH = ROOT / "benchmarks/test_fully_quantized_internal_sweep.cu"
RUNNER = ROOT / "tools/run_fq_q4k_grid_order_ab_box.sh"
GUARD = "#if defined(PPU_FIXED_SPLITK_N_ON_X) && (PPU_FIXED_SPLITK_N_ON_X != 0)"


class CheckError(ValueError):
    pass


def check_source(text: str) -> None:
    required_once = (
        "int const m_coord = int(blockIdx.y);\n"
        "    int const n_coord = int(blockIdx.x);",
        "int const m_coord = int(blockIdx.x);\n"
        "    int const n_coord = int(blockIdx.y);",
        "uint64_t const q = uint64_t(m_coord) * n_tiles + uint64_t(n_coord);",
        "uint64_t const n_tiles = uint64_t(gridDim.x);",
        "uint64_t const n_tiles = uint64_t(gridDim.y);",
    )
    if text.count(GUARD) != 3:
        raise CheckError("N-on-x guard denominator differs")
    for needle in required_once:
        if text.count(needle) != 1:
            raise CheckError(f"source seam denominator differs: {needle}")
    # The two launch spellings must be bijections onto exactly the same logical
    # (m,n,s,q) set.  Include M>1 controls even though the measured target is M=1.
    for mt, nt, splits in ((1, 128, 4), (3, 5, 2), (7, 1, 8)):
        native = {
            (m, n, s, m * nt + n)
            for s in range(splits) for n in range(nt) for m in range(mt)
        }
        n_on_x = {
            (y, x, z, y * nt + x)
            for z in range(splits) for y in range(mt) for x in range(nt)
        }
        if native != n_on_x or len(native) != mt * nt * splits:
            raise CheckError("grid-axis swap changed logical work ownership")


def check_seams(bench: str, runner: str) -> None:
    bench_needles = (
        'constexpr char const* split_grid_order = "n-on-x";',
        'constexpr char const* split_grid_order = "native-grid";',
        'split_timing=ordered-close split_grid_order=%s ',
    )
    runner_needles = (
        'for schedule in native-grid n-on-x; do',
        'defs="PPU_FIXED_SPLITK_N_ON_X=1"',
        'PPU_DEFS="$defs" PPU_EXTRA_DEFS=',
        'for layout in xplane kpack4; do',
        'for ap in 0 1; do',
    )
    for needle in bench_needles:
        if bench.count(needle) != (2 if needle.endswith("=%s ") else 1):
            raise CheckError(f"benchmark identity seam differs: {needle}")
    for needle in runner_needles:
        if runner.count(needle) != 1:
            raise CheckError(f"runner factorial seam differs: {needle}")


def self_test() -> None:
    text = SOURCE.read_text()
    check_source(text)
    bench = BENCH.read_text()
    runner = RUNNER.read_text()
    check_seams(bench, runner)
    plants = (
        text.replace("int const n_coord = int(blockIdx.x);",
                     "int const n_coord = int(blockIdx.y);", 1),
        text.replace("uint64_t const n_tiles = uint64_t(gridDim.x);",
                     "uint64_t const n_tiles = uint64_t(gridDim.y);", 1),
        text.replace(GUARD, "#if 1", 1),
        text.replace("uint64_t const q = uint64_t(m_coord) * n_tiles + uint64_t(n_coord);",
                     "uint64_t const q = uint64_t(n_coord) * n_tiles + uint64_t(m_coord);", 1),
    )
    for plant in plants:
        try:
            check_source(plant)
        except CheckError:
            pass
        else:
            raise AssertionError("source negative plant stayed green")
    seam_plants = (
        (bench.replace('split_grid_order = "n-on-x"',
                       'split_grid_order = "native-grid"', 1), runner),
        (bench, runner.replace('PPU_DEFS="$defs"', 'PPU_DEFS=', 1)),
        (bench, runner.replace('for layout in xplane kpack4; do',
                               'for layout in kpack4; do', 1)),
    )
    for broken_bench, broken_runner in seam_plants:
        try:
            check_seams(broken_bench, broken_runner)
        except CheckError:
            pass
        else:
            raise AssertionError("runner/marker negative plant stayed green")
    print("[fq-fixed-splitk-n-on-x:self-test] PASS exact two-axis seam, "
          "logical-q bijection, runtime marker and 2x2x2 build factorial; "
          "seven negatives RED")


if __name__ == "__main__":
    try:
        self_test()
    except (AssertionError, CheckError, OSError) as exc:
        print(f"[fq-fixed-splitk-n-on-x:self-test] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
