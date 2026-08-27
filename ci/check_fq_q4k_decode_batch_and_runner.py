#!/usr/bin/env python3
"""Fail-closed local contract for Q4_K native SIMT batching and decode sweep."""

from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNEL = ROOT / "quactlize/include/gguf_bc_vecdot.hpp"
BACKEND = ROOT / "quactlize/csrc/device/ppu_backend.cu"
THOP = ROOT / "quactlize/csrc/preprocess/thop/gguf_prepass_ops.cpp"
BENCH = ROOT / "benchmarks/test_fully_quantized_internal_sweep.cu"
TC_BENCH = ROOT / "benchmarks/fully_quantized_splitk_producer_bench.hpp"
POLICY = ROOT / "benchmarks/fq_q4k_decode_real_shapes_policy.json"
PLAN = ROOT / "tools/plan_fq_q4k_decode_real_shapes.py"
ANALYZE = ROOT / "tools/analyze_fq_q4k_decode_real_shapes.py"
RUNNER = ROOT / "tools/run_fq_q4k_decode_real_shapes_box.sh"
GENERATOR = ROOT / "tools/gen_fully_quantized_splitk_producer_units.py"


class CheckError(ValueError):
    pass


def check(kernel: str, backend: str, thop: str, bench: str,
          policy: str, plan: str, analyze: str, runner: str,
          generator: str, tc_bench: str) -> None:
    kernel_needles = (
        "int const dense_batch_row = Grouped ? 0 : int(blockIdx.y);",
        "int const activation_row = active ? gathered_row : 0;",
        "float* row_out = out + (active ? int64_t(gathered_row) * rows : 0);",
        "dim3 grid(vecdot::vecdot_grid_size<T,RowsPerWarp>(n,Threads),max_rows,Grouped?experts:1);",
    )
    if any(token not in kernel for token in kernel_needles):
        raise CheckError("dense SIMT grid-y activation/output ownership is incomplete")
    if "for (int batch" in kernel or "for (int row_launch" in kernel:
        raise CheckError("dense SIMT batch regressed to a host launch loop")
    if backend.count("(experts == 0 && total_rows >= 8)") != 4 or \
            "nullptr, out, n, bpr, 1, total_rows, stream" not in backend:
        raise CheckError("device ABI does not admit exactly dense M=1..7")
    if "a.size(0) > 0 && a.size(0) < 8" not in thop or \
            "int experts = 0, max_rows = int(a.size(0));" not in thop:
        raise CheckError("THOP ABI does not forward the native dense batch")
    bench_needles = (
        "bool const supported = shape.m < 8;",
        "dBcOut(std::size_t(shape.m) * shape.n)",
        "nullptr, output, shape.n, bpr, 1, shape.m, nullptr",
        '"native-grid-y-m-lt8"',
        '"UNSUPPORTED_M_GE_8"',
        '"--tm8-max-m="',
        "cli.tm8_max_m",
    )
    if any(token not in bench for token in bench_needles):
        raise CheckError("benchmark does not measure native M<8 SIMT output")
    if '"decode_m": [1, 2, 4, 8]' not in policy or \
            '"bc_batch_policy": "native-grid-y-m-lt8"' not in policy:
        raise CheckError("policy lost the exact decode/SIMT denominator")
    if "DECODE_M = (1, 2, 4, 8)" not in plan:
        raise CheckError("planner lost one decode M")
    analyze_needles = (
        "m * n * split * int(model[\"partial_element_bytes\"])",
        "float(model[\"bandwidth_fraction\"])",
        '"SIMT_BC"',
        '"PRODUCER_PLUS_MODELED_REDUCER"',
        "if shape[0] >= 8:",
        "TC_SPLITS = (1, 2, 4, 8)",
        '(row["symbol"], row["S_int"]) for row in tc)',
        'row["S_int"] == expected_split',
        "selected_census = collections.Counter(",
        "observed not in (selected_census, full_census)",
        'unselected split produced a non-census row',
        "SCHEDULER_TERMINAL_STATES = frozenset(",
        'board_counts[f"S{split}"] = {\n                "status": "UNAVAILABLE"',
        "if not values:\n            if split == 1:",
        "has unknown terminal state",
        "Every Q4 artifact now has a typed TM8 tensor-core denominator",
        "typed_rows=12",
        '"unavailable_cell_count"',
        "def update_analysis_resume_audit(",
        '"schema": "quactlize.fq_q4k_decode_analysis_resume.v2"',
        "analysis-only resume audit is not append-only",
    )
    if any(token not in analyze for token in analyze_needles):
        raise CheckError("analysis mixes producer-only, reducer, or SIMT scope")
    if analyze.count('["status"] == "UNAVAILABLE":') != 2:
        raise CheckError("unavailable artifact cells are not skipped at both rankings")
    runner_needles = (
        "--only-split=1",
        "--bc-mode=all",
        "phase=scheduler",
        "--bc-mode=skip",
        "--bc-mode=only",
        "committed phase log changed",
        'atomic_text "$commit" "$actual"',
        ".uncommitted.",
        ".failed.",
        "analyze_fq_q4k_decode_real_shapes.py\" finalize",
        "ppu_packed_metadata_ownership.hpp",
        "ppu_mma_aiu_fold.hpp",
        "quactlize_mma_mixed_input.hpp",
        "--tile-m-filter 8",
        "--tm8-max-m=8",
        "tensor_core=TM8/WM8/M<=8/source-typed-denominator-retained",
        "analysis-only resume requires an empty original source.patch",
        "resume source authority changed outside analysis-only seam",
        'compiled_binary_identity="MUST_MATCH_FROZEN_BINARY_HASHES"',
        "measurement_source_state_sha256",
        "update_analysis_resume_audit(",
        '"merge-base","--is-ancestor"',
        "allowed_analysis_files=analysis_only",
        "audit_hops=",
    )
    if any(token not in runner for token in runner_needles):
        raise CheckError("runner does not execute screen/scheduler/TC/SIMT phases")
    if runner.count("--tile-m-filter 8") != 1 or \
            runner.count("--tm8-max-m=8") != 3:
        raise CheckError("decode TM8 selection/admission call denominator changed")
    generator_needles = (
        'parser.add_argument("--tile-m-filter"',
        "r.tile_m == tile_m_filter",
        '"source_typed_rows": len(source_eligible)',
        '"selection_reject_rows": len(selection_rejects)',
    )
    if any(token not in generator for token in generator_needles):
        raise CheckError("decode TileM filter lost its complete source denominator")
    if generator.count("r.tile_m == tile_m_filter") != 2:
        raise CheckError("decode TileM selection/count predicates diverged")
    tc_bench_needles = (
        "int tm8_max_m = ppu_dense_shipping::kDecodeDefaultExclusiveM - 1;",
        "in.m > options.tm8_max_m",
    )
    if any(token not in tc_bench for token in tc_bench_needles):
        raise CheckError("TM8 runtime admission is not bound to the decode M ceiling")


def main() -> int:
    texts = [path.read_text() for path in
             (KERNEL, BACKEND, THOP, BENCH, POLICY, PLAN, ANALYZE, RUNNER,
              GENERATOR, TC_BENCH)]
    check(*texts)
    plants = [
        (0, ",max_rows,Grouped?experts:1);", ",1,Grouped?experts:1);"),
        (0, "out + (active ? int64_t(gathered_row) * rows : 0)", "out"),
        (3, "shape.m < 8", "shape.m <= 8"),
        (4, '"decode_m": [1, 2, 4, 8]', '"decode_m": [1, 2, 4]'),
        (6, '"PRODUCER_PLUS_MODELED_REDUCER"', '"PRODUCER_ONLY"'),
        (6, "TC_SPLITS = (1, 2, 4, 8)", "TC_SPLITS = (1,)"),
        (6, 'row["S_int"] == expected_split', 'True'),
        (6, 'board_counts[f"S{split}"] = {\n                "status": "UNAVAILABLE"',
         'board_counts[f"S{split}"] = {\n                "status": "AVAILABLE"'),
        (6, "if not values:\n            if split == 1:",
         "if not values:\n            if False:"),
        (6, "typed_rows=12", "typed_rows=0"),
        (6, '["status"] == "UNAVAILABLE":', '["status"] == "AVAILABLE":'),
        (7, "--tile-m-filter 8", "--tile-m-filter 16"),
        (7, "--tm8-max-m=8", "--tm8-max-m=7"),
        (7, 'compiled_binary_identity="MUST_MATCH_FROZEN_BINARY_HASHES"',
         'compiled_binary_identity="UNBOUND"'),
        (7, "allowed_analysis_files=analysis_only",
         "allowed_analysis_files=set(authority_rel)"),
        (8, "r.tile_m == tile_m_filter", "True"),
    ]
    for index, old, new in plants:
        planted = list(texts)
        if old not in planted[index]:
            raise CheckError(f"negative-control seam missing: {old}")
        planted[index] = planted[index].replace(old, new, 1)
        try:
            check(*planted)
        except CheckError:
            pass
        else:
            raise CheckError(f"negative control stayed green: {old} -> {new}")
    print("[fq-q4k-decode:self-test] PASS: M=1/2/4/8, TM8-only TC, native "
          "one-launch M<8 SIMT, phase/census separation, analysis-only resume "
          "binding with append-only multi-hop audit, optional scheduler boards "
          "with mandatory S1, native A32/F2 TM8 coverage, and "
          "sixteen "
          "negative plants")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CheckError, OSError) as error:
        print(f"[fq-q4k-decode:self-test] FAIL: {error}")
        raise SystemExit(2)
