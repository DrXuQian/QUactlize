#!/usr/bin/env python3
"""Source contract for the exact post-mainloop accumulator/epilogue bisection."""

from __future__ import annotations

import argparse
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNEL = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_splitk_parallel.hpp"
DIRECT = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/kernel/detail/ppu_splitk_direct_accumulator_store.hpp"
ORACLE = ROOT / "dev/fold_derivation/l222_fq_splitk_direct_accumulator_store.cu"
RUNNER = ROOT / "tools/run_fq_q4k_split_accumulator_bisect_box.sh"
CHECKER = ROOT / "tools/check_fq_split_accumulator_bisect.py"
EVIDENCE = ROOT / "dev/fold_derivation/l222_fq_splitk_direct_accumulator_store.expected.txt"

EXPECTED_EVIDENCE = (
    "L222_DIRECT_ACCUMULATOR_STORE visits=576 expected=576 holes=0 "
    "duplicates=0 value_bad=0 invalid_touched=0 verdict=PASS",
    "L222_DIRECT_ACCUMULATOR_STORE visits=576 expected=576 holes=432 "
    "duplicates=144 value_bad=432 invalid_touched=0 verdict=FAIL",
    "L222_DIRECT_ACCUMULATOR_STORE visits=576 expected=576 holes=0 "
    "duplicates=0 value_bad=576 invalid_touched=0 verdict=FAIL",
    "[l222] PASS: exact direct store plus duplicate-owner and "
    "register-coordinate negatives",
)


class CheckError(ValueError):
    pass


def require(label: str, text: str, needles: tuple[str, ...]) -> None:
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise CheckError(f"{label} contract missing: {missing}")


def check_evidence(evidence: str) -> None:
    lines = tuple(line for line in evidence.splitlines() if line)
    if lines != EXPECTED_EVIDENCE:
        raise CheckError("committed L222 evidence denominator or verdict differs")


def check(kernel: str, direct: str, oracle: str,
          runner: str, checker: str, evidence: str) -> None:
    require("kernel", kernel, (
        "PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE",
        "detail::store_splitk_accumulators_direct(",
        "CollectivePartialEpilogue partial_epilogue",
        "collective_mainloop(params.mainloop",
    ))
    mainloop_at = kernel.index("collective_mainloop(params.mainloop")
    legacy_at = kernel.index(
        "PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE", mainloop_at)
    shared_at = kernel.index("CollectivePartialEpilogue partial_epilogue",
                             legacy_at)
    direct_at = kernel.index("detail::store_splitk_accumulators_direct(",
                             shared_at)
    if not (mainloop_at < legacy_at < shared_at < direct_at):
        raise CheckError("production direct-store seam or legacy negative moved")
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
        'defs="PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE=1"',
        "retained diagnostic owner-only metadata",
        "l222_fq_splitk_direct_accumulator_store.expected.txt",
        "committed-local-oracle",
        "--split-workspace-probe",
        "--epilogue-performance",
        "--accumulator-performance",
        "PERF_CORRECTNESS_REPEATS",
        "PRODUCTION_CLOSURE_COMPLETE",
        "check_fq_split_accumulator_bisect.py",
    ))
    require("checker", checker, (
        'verdict = "MAINLOOP_ACCUMULATOR_CORRUPTION_CONFIRMED"',
        'verdict = "PARTIAL_EPILOGUE_CORRUPTION_CONFIRMED"',
        'verdict = "DIRECT_STORE_NEGATIVE_CONTROL_FAILED"',
        'mainloop_prefix=IDENTICAL_BY_COMPILE_TIME_POST_MAINLOOP_SEAM',
        "FQ_ACCUMULATOR_PRODUCTION_CLOSURE",
        "packed-row-S4-same-run",
        "PERF_REGRESSION_LIMIT",
    ))
    check_evidence(evidence)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--committed-only", action="store_true")
    parser.add_argument("--evidence", type=pathlib.Path)
    args = parser.parse_args()
    if args.committed_only:
        if args.evidence is None:
            raise CheckError("--committed-only requires --evidence")
        check_evidence(args.evidence.read_text())
        print("[fq-accumulator-bisect-source:evidence] PASS "
              "committed-local-oracle exact positive and two negatives")
        return 0

    paths = (KERNEL, DIRECT, ORACLE, RUNNER, CHECKER, EVIDENCE)
    texts = [path.read_text() for path in paths]
    check(*texts)
    plants = (
        (0, "collective_mainloop(params.mainloop",
         "collective_mainloop_WRONG(params.mainloop"),
        (1, "partition_C(gD)", "partition_A(gD)"),
        (2, "#ifndef L222_BAD_THREAD_MODULO",
         "#ifndef L222_WRONG_THREAD_MODULO"),
        (3, 'defs="PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE=1"',
         'defs="PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE=0"'),
        (4, 'verdict = "PARTIAL_EPILOGUE_CORRUPTION_CONFIRMED"',
         'verdict = "PARTIAL_EPILOGUE_UNKNOWN"'),
        (5, "holes=432 duplicates=144", "holes=0 duplicates=0"),
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
          "production-default CuTe direct map, legacy shared negative, "
          "two oracle negatives, committed evidence, two-arm runner and verdicts")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CheckError, OSError) as error:
        print(f"[fq-accumulator-bisect-source:self-test] FAIL: {error}")
        raise SystemExit(2)
