#!/usr/bin/env python3
"""Source contract for the five-format production layout performance closure."""

from __future__ import annotations

import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def require(source: str, markers: tuple[str, ...], label: str) -> None:
    for marker in markers:
        if marker not in source:
            raise AssertionError(f"{label} misses {marker!r}")


def main() -> int:
    bench = (ROOT / "benchmarks/test_fq_kquant_layout_perf.cu").read_text()
    runner = (ROOT / "tools/run_fq_kquant_kpack_perf_box.sh").read_text()
    cmake = (ROOT / "quactlize/csrc/fq_kquant_layout_perf.cmake.in").read_text()
    grouped_kernel = (ROOT / "quactlize/include/ppu_aiu_gemm_mixed_input_group.hpp").read_text()
    backend = (ROOT / "quactlize/csrc/device/ppu_dense_backend.cu").read_text()
    require(bench, (
        "quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v2",
        "quactlize_ppu_grouped_fully_quantized_dev_for_arrangement_v2",
        "ppu_arrangements::kquant_kpack_transpose_v1(kQtype)",
        "ppu_arrangements::q4_kpack4_transpose_v1()",
        "QUACTLIZE_PPU_LAYOUT_XPLANE_V1",
        "compare_raw_kernel<<<blocks, 256>>>",
        "hggcEventRecord(begin, nullptr)",
        "hggcEventRecord(end, nullptr)",
        "moe_router_fixture::route(",
        "nullptr, config.wire,",
        "&weights.descriptor);",
        "FQ_KQUANT_LAYOUT_DENSE",
        "FQ_KQUANT_LAYOUT_GROUPED",
        "FQ_KQUANT_LAYOUT_FAILURE",
        "if (!run(row, false, dx, cli) || !run(row, true, dk, cli)) return false;",
    ), "benchmark")
    require(grouped_kernel, (
        "bool const has_host_geometry =",
        "bool const has_device_geometry =",
        "args.problem_shape.problem_shapes != nullptr",
        "args.representative_m > 0",
        "args.representative_n > 0",
        "args.representative_k > 0",
        "args.mtiles_uniform > 0",
        "(!has_host_geometry && !has_device_geometry)",
    ), "grouped device admission")
    require(backend, (
        "arrangement->artifact_tile_k == tactic_tile_k",
        "quactlize_ppu_grouped_fully_quantized_config_v1(",
        "quactlize_ppu_grouped_fully_quantized_dev_v2(",
    ), "grouped Xplane arrangement control")
    require(runner, (
        "for q in 10 11 12 13 14",
        'format_defs="PPU_PACKED_SCALE=1 PPU_PACKED_FORMAT=$fmt QUACTLIZE_DENSE_ONLY=$q"',
        "--order=$order",
        'if [ "$q" = 12 ]',
        'run_args+=("${grouped_args[@]}")',
        '"${dense_args[@]}" "${grouped_args[@]}"',
        'grep -F -- "-DFQ_KQUANT_PERF_QTYPE=$q" "$target_make"',
        "python3 -B \"$analyzer\" analyze",
        "source-authority.sha256",
    ), "runner")
    require(cmake, (
        "test_fq_kquant_layout_perf",
        "target_link_libraries(test_fq_kquant_layout_perf PRIVATE quactlize_ppu)",
        "DEV_COMPILE_FLAGS\n      -DFQ_KQUANT_PERF_QTYPE=${FQ_KQUANT_PERF_QTYPE}",
        "FQ_KQUANT_PERF_QTYPE=${FQ_KQUANT_PERF_QTYPE}",
    ), "CMake")

    plants = (
        bench.replace("hggcEventRecord(end, nullptr)",
                      "hggcEventRecord(begin, nullptr)", 1),
        bench.replace("QUACTLIZE_PPU_LAYOUT_XPLANE_V1",
                      "QUACTLIZE_PPU_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1", 1),
        runner.replace('run_args+=("${grouped_args[@]}")',
                       'run_args+=("${dense_args[@]}")', 1),
        grouped_kernel.replace("bool const has_device_geometry =",
                               "bool const has_device_geometry = false &&", 1),
        backend.replace(
            "return arrangement->artifact_tile_k == tactic_tile_k &&\n"
            "        grouped_fully_quantized_config_valid(",
            "return false && arrangement->artifact_tile_k == tactic_tile_k &&\n"
            "        grouped_fully_quantized_config_valid(", 1),
    )
    assert plants[0] != bench and "hggcEventRecord(end, nullptr)" not in plants[0]
    assert plants[1] != bench and plants[1].count("QUACTLIZE_PPU_LAYOUT_XPLANE_V1") == 0
    assert plants[2] != runner and \
        'run_args+=("${grouped_args[@]}")' not in plants[2]
    assert plants[3] != grouped_kernel and \
        "bool const has_device_geometry = false &&" in plants[3]
    assert plants[4] != backend and \
        "return false && arrangement->artifact_tile_k == tactic_tile_k" in plants[4]

    for script in ("plan_fq_kquant_kpack_perf.py",
                   "analyze_fq_kquant_kpack_perf.py"):
        subprocess.run([sys.executable, "-B", str(ROOT / "tools" / script),
                        "self-test"], check=True, stdout=subprocess.DEVNULL)
    print("[fq-kquant-perf:self-test] PASS production dense/grouped C ABI, "
          "five-format same-binary events, device-only admission, exact "
          "grouped Xplane control, Q4 grouped and five source plants RED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, subprocess.CalledProcessError) as error:
        print(f"[fq-kquant-perf:self-test] FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)
