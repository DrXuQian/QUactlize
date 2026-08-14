#!/usr/bin/env python3
"""Bind host checked-lowering to the assume-valid standalone device path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
COLLECTIVE = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp"
KERNEL = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/marlin_kernel_ppu.hpp"
OUTPUT_MAP = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/marlin_output_map_ppu.hpp"
HANDLE = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/device/marlin_gemm_ppu.hpp"
BENCH = ROOT / "benchmarks/test_lowbit_dense_bench.cu"


def fail(plant: str, message: str) -> int:
    if plant == "none":
        print(f"[l179-source] FAIL: {message}", file=sys.stderr)
    else:
        print(f"[l179-source:red] plant={plant} caught=1 reason={message} result=RED", file=sys.stderr)
    return 1


def balanced_body(text: str, anchor: str) -> str:
    """Return one brace-balanced declaration body, including its braces."""
    start = text.find(anchor)
    if start < 0:
        raise ValueError(f"missing declaration anchor {anchor}")
    open_brace = text.find("{", start)
    if open_brace < 0:
        raise ValueError(f"missing opening brace after {anchor}")
    depth = 0
    for pos in range(open_brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace : pos + 1]
    raise ValueError(f"unterminated declaration after {anchor}")


def planted(
    plant: str, collective: str, kernel: str, output_map: str,
    handle: str, bench: str,
):
    if plant == "device-valid":
        collective = collective.replace(
            "int k_tiles_remaining = 0;", "int k_tiles_remaining = 0;\n    bool valid = false;", 1)
    elif plant == "unchecked-lowering":
        kernel = kernel.replace(
            "supported\n            ? TileScheduler::to_underlying_arguments(",
            "true\n            ? TileScheduler::to_underlying_arguments(", 1)
    elif plant == "noncanonical-batch-stride":
        kernel = kernel.replace(
            "(int64_t(cute::get<2>(args.epilogue.dD)) == 0 ||\n"
            "            int64_t(cute::get<2>(args.epilogue.dD)) == m * n) &&\n"
            "           l == 1;",
            "int64_t(cute::get<2>(args.epilogue.dD)) == m * n &&\n"
            "           l == 1;",
            1,
        )
    elif plant == "col-guard":
        kernel = kernel.replace("if (row < problem_m) {", "if (row < problem_m && col < problem_n) {", 1)
    elif plant == "unchecked-workspace":
        handle = handle.replace(
            "return can_implement(args) == Status::kSuccess\n"
            "        ? RawGemm::get_workspace_size(args)\n        : 0;",
            "return RawGemm::get_workspace_size(args);", 1)
    elif plant == "raw-adapter":
        bench = bench.replace(
            "cutlass::gemm::device::MarlinGemmPPU<MarlinKernel>",
            "cutlass::gemm::device::GemmUniversalAdapter<MarlinKernel>", 1)
    elif plant != "none":
        raise ValueError(f"unknown plant {plant}")
    return collective, kernel, output_map, handle, bench


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", default="none")
    args = parser.parse_args()
    plant = args.plant
    try:
        collective, kernel, output_map, handle, bench = (
            COLLECTIVE.read_text(), KERNEL.read_text(), OUTPUT_MAP.read_text(),
            HANDLE.read_text(), BENCH.read_text()
        )
        collective, kernel, output_map, handle, bench = planted(
            plant, collective, kernel, output_map, handle, bench)
    except (OSError, ValueError) as exc:
        return fail(plant, str(exc))

    errors: list[str] = []
    for token in ("address_arithmetic_supported(", "mul_add_fits_int", "args.group_size == GroupSize"):
        if token not in collective:
            errors.append(f"host mainloop check lacks {token}")
    try:
        device_state = "\n".join(
            balanced_body(collective, anchor)
            for anchor in ("struct CtaState", "struct SegmentState")
        )
    except ValueError as exc:
        errors.append(str(exc))
        device_state = collective
    for forbidden in ("bool valid", "int n_tiles =", "int k_tiles ="):
        if forbidden in device_state:
            errors.append(f"device address state retained {forbidden}")
    rebase = collective.split("static SegmentState rebase_segment(", 1)[1].split("template <class ProblemShape>", 1)[0]
    for forbidden in ("if (", "work.is_valid", "state.valid"):
        if forbidden in rebase:
            errors.append(f"assume-valid rebase retained {forbidden}")
    for token in (
        "bool const supported = arguments_supported(args, hw);",
        "supported\n            ? TileScheduler::to_underlying_arguments(",
        ": TileSchedulerParams{}",
        "(int64_t(cute::get<2>(args.epilogue.dD)) == 0 ||\n"
        "            int64_t(cute::get<2>(args.epilogue.dD)) == m * n) &&\n"
        "           l == 1;",
    ):
        if token not in kernel:
            errors.append(f"kernel checked-lowering lacks {token}")
    handoff = kernel.split("static void global_handoff(", 1)[1].split("static void write_result(", 1)[0]
    if "col < problem_n" in handoff:
        errors.append("handoff retained mathematically redundant col<N guard")
    for token in (
        "constexpr int output_row(", "constexpr int output_n_base(",
        "constexpr int output_col_offset(",
    ):
        if output_map.count(token) != 1:
            errors.append(f"authoritative output map lacks {token}")
    kernel_compact = "".join(kernel.split())
    for token, count in (
        ("marlin_ppu_detail::output_row<InstructionM>(", 2),
        ("marlin_ppu_detail::output_n_base<TileN,NBlocksPerWarp>(", 2),
        ("marlin_ppu_detail::output_col_offset<InstructionM>(", 2),
    ):
        if kernel_compact.count(token) != count:
            errors.append(
                f"production output path consumes {token} "
                f"{kernel_compact.count(token)} times, expected {count}"
            )
    for token in (
        "class MarlinCheckedHandlePPU", "using RawGemm = RawGemm_",
        "using MarlinGemmPPU = detail::MarlinCheckedHandlePPU<",
        "GemmUniversalAdapter<GemmKernel_>", "bool installed_ = false",
        "Status update(Arguments const&, void* = nullptr) = delete",
        "static dim3 get_grid_shape(Params const&) = delete",
        "Params const& params() const = delete", "if (!installed_)",
        "? RawGemm::get_workspace_size(args)",
    ):
        if token not in handle:
            errors.append(f"owned handle lacks {token}")
    try:
        handle_body = balanced_body(handle, "class MarlinCheckedHandlePPU")
        private_pos = handle_body.index("private:")
        raw_pos = handle_body.index("RawGemm raw_")
        public_pos = handle_body.index("public:")
        if not (private_pos < raw_pos < public_pos):
            errors.append("raw adapter storage is not privately composed")
    except (ValueError, IndexError) as exc:
        errors.append(f"owned handle structure invalid: {exc}")
    if "using MarlinGemm = cutlass::gemm::device::MarlinGemmPPU<MarlinKernel>;" not in bench:
        errors.append("shipping standalone Cfg does not use the owned handle")

    if errors:
        return fail(plant, "; ".join(errors))
    if plant != "none":
        print(
            f"[l179-source:red] plant={plant} caught=0 "
            "reason=named plant survived result=FAIL",
            file=sys.stderr,
        )
        return 1
    print("[l179-source] PASS: checked Args->Params lowering owns the zero-grid fail-close; device rebase is assume-valid; unsafe adapter seams are deleted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
