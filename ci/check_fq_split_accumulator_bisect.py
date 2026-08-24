#!/usr/bin/env python3
"""Source contract for the exact post-mainloop accumulator/epilogue bisection."""

from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNEL = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_splitk_parallel.hpp"
DIRECT = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/kernel/detail/ppu_splitk_direct_accumulator_store.hpp"
ORACLE = ROOT / "dev/fold_derivation/l222_fq_splitk_direct_accumulator_store.cu"
RUNNER = ROOT / "tools/run_fq_q4k_split_accumulator_bisect_box.sh"
CHECKER = ROOT / "tools/check_fq_split_accumulator_bisect.py"


class CheckError(ValueError):
    pass


def require(label: str, text: str, needles: tuple[str, ...]) -> None:
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise CheckError(f"{label} contract missing: {missing}")


def check(kernel: str, direct: str, oracle: str,
          runner: str, checker: str) -> None:
    require("kernel", kernel, (
        "PPU_SPLITK_DIRECT_ACCUMULATOR_STORE",
        "detail::store_splitk_accumulators_direct(",
        "CollectivePartialEpilogue partial_epilogue",
        "collective_mainloop(params.mainloop",
    ))
    if not (kernel.index("collective_mainloop(params.mainloop") <
            kernel.index("PPU_SPLITK_DIRECT_ACCUMULATOR_STORE") <
            kernel.index("CollectivePartialEpilogue partial_epilogue")):
        raise CheckError("direct-store seam moved before mainloop or after epilogue")
    require("direct", direct, (
        "get_thread_slice(thread_idx)",
        "partition_C(gD)",
        "partition_C(identity)",
        "tD(i) = accumulators(i)",
        "elem_less(coordinates(i)",
    ))
    require("oracle", oracle, (
        "#ifndef L222_BAD_THREAD_MODULO",
        "#ifndef L222_BAD_FRAGMENT_ROTATE",
        "store_splitk_accumulators_direct(",
        "holes=",
        "duplicates=",
        "value_bad=",
    ))
    require("runner", runner, (
        "shared-epilogue direct-accumulator",
        "PPU_PACKED_METADATA_OWNER_ONLY=1",
        'defs="$defs PPU_SPLITK_DIRECT_ACCUMULATOR_STORE=1"',
        "--split-workspace-probe",
        "check_fq_split_accumulator_bisect.py",
    ))
    require("checker", checker, (
        'verdict = "MAINLOOP_ACCUMULATOR_CORRUPTION_CONFIRMED"',
        'verdict = "PARTIAL_EPILOGUE_CORRUPTION_CONFIRMED"',
        'verdict = "DIRECT_STORE_NEGATIVE_CONTROL_FAILED"',
        'mainloop_prefix=IDENTICAL_BY_COMPILE_TIME_POST_MAINLOOP_SEAM',
    ))


def main() -> int:
    paths = (KERNEL, DIRECT, ORACLE, RUNNER, CHECKER)
    texts = [path.read_text() for path in paths]
    check(*texts)
    plants = (
        (0, "collective_mainloop(params.mainloop",
         "collective_mainloop_WRONG(params.mainloop"),
        (1, "partition_C(gD)", "partition_A(gD)"),
        (2, "#ifndef L222_BAD_THREAD_MODULO",
         "#ifndef L222_WRONG_THREAD_MODULO"),
        (3, 'defs="$defs PPU_SPLITK_DIRECT_ACCUMULATOR_STORE=1"',
         'defs="$defs PPU_SPLITK_DIRECT_ACCUMULATOR_STORE=0"'),
        (4, 'verdict = "PARTIAL_EPILOGUE_CORRUPTION_CONFIRMED"',
         'verdict = "PARTIAL_EPILOGUE_UNKNOWN"'),
    )
    for index, old, new in plants:
        changed = list(texts)
        changed[index] = changed[index].replace(old, new, 1)
        try:
            check(*changed)
        except CheckError:
            pass
        else:
            raise CheckError(f"negative stayed green: {old}")
    print("[fq-accumulator-bisect-source:self-test] PASS post-mainloop seam, "
          "actual CuTe direct map, two negatives, two-arm runner and verdicts")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CheckError, OSError) as error:
        print(f"[fq-accumulator-bisect-source:self-test] FAIL: {error}")
        raise SystemExit(2)
