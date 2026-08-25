#!/usr/bin/env python3
"""Guard the compile-time first-prefix Split-K root experiment."""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
HELPER = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/kernel/detail/ppu_splitk_shared_prefix_probe.hpp"
KERNEL = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_splitk_parallel.hpp"
RUNNER = ROOT / "tools/run_fq_q4k_split_shared_prefix_root_box.sh"
CHECKER = ROOT / "tools/check_fq_split_shared_prefix_root.py"


POLICIES = {
    "ACCUMULATOR_OPAQUE": 1,
    "CLONE_OPAQUE": 2,
    "CTA_ONLY": 3,
    "FLAT_CONSTANT": 4,
    "FLAT_ACCUMULATOR": 5,
    "R2S_VECTOR": 6,
    "R2S_SCALAR": 7,
    "R2S_SCALAR_SNAPSHOT": 8,
    "R2S_S2R_VECTOR": 9,
    "R2S_S2R_SCALAR": 10,
}

ARMS = {
    "production-direct": "",
    "accumulator-opaque": "PPU_SPLITK_SHARED_PREFIX_POLICY=1",
    "clone-opaque": "PPU_SPLITK_SHARED_PREFIX_POLICY=2",
    "cta-only": "PPU_SPLITK_SHARED_PREFIX_POLICY=3",
    "flat-constant-disjoint": (
        "PPU_SPLITK_SHARED_PREFIX_POLICY=4 "
        "PPU_SPLITK_SHARED_PROBE_DISJOINT_STORAGE=1"),
    "flat-accumulator-disjoint": (
        "PPU_SPLITK_SHARED_PREFIX_POLICY=5 "
        "PPU_SPLITK_SHARED_PROBE_DISJOINT_STORAGE=1"),
    "r2s-vector-disjoint": (
        "PPU_SPLITK_SHARED_PREFIX_POLICY=6 "
        "PPU_SPLITK_SHARED_PROBE_DISJOINT_STORAGE=1"),
    "r2s-scalar-disjoint": (
        "PPU_SPLITK_SHARED_PREFIX_POLICY=7 "
        "PPU_SPLITK_SHARED_PROBE_DISJOINT_STORAGE=1"),
    "r2s-snapshot-disjoint": (
        "PPU_SPLITK_SHARED_PREFIX_POLICY=8 "
        "PPU_SPLITK_SHARED_PROBE_DISJOINT_STORAGE=1"),
    "r2s-s2r-vector-disjoint": (
        "PPU_SPLITK_SHARED_PREFIX_POLICY=9 "
        "PPU_SPLITK_SHARED_PROBE_DISJOINT_STORAGE=1"),
    "r2s-s2r-scalar-disjoint": (
        "PPU_SPLITK_SHARED_PREFIX_POLICY=10 "
        "PPU_SPLITK_SHARED_PROBE_DISJOINT_STORAGE=1"),
    "legacy-shared-output":
        "PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE=1",
    "full-discard": (
        "PPU_SPLITK_SHARED_SYNC_POLICY=3 "
        "PPU_SPLITK_SHARED_PROBE_DISCARD_GMEM=1"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_texts(helper: str, kernel: str, runner: str,
                checker: str) -> None:
    for name, value in POLICIES.items():
        token = f"#define PPU_SPLITK_SHARED_PREFIX_{name} {value}"
        require(helper.count(token) == 1,
                f"prefix policy binding changed: {token}")
    require(helper.count("asm volatile(\"\" : : \"r\"(bits) : \"memory\");") == 1,
            "opaque register-liveness seam changed")
    require(helper.count("copy(tiled_r2s, tCaC(_, mma_m, mma_n),") == 1,
            "vector R2S prefix changed")
    require(helper.count("copy(tiled_s2r, tDsC, tDrC);") == 1,
            "vector S2R prefix changed")
    require("destination(i) = source(i);" in helper,
            "scalar R2S control changed")
    require("flat[offset] = float(offset);" in helper and
            "flat[offset] = accumulators(i);" in helper,
            "flat constant/accumulator controls changed")
    require("ValuesPerThread * Threads == cosize_v<SmemLayout>" in helper,
            "flat exact-once shared denominator changed")
    require(helper.count("if constexpr") >= 10,
            "prefix policies are no longer compile-time selected")
    for forbidden in ("atomicAdd", "__threadfence", "ptr_D", "ptr_C"):
        require(forbidden not in helper,
                f"prefix probe gained an external side effect: {forbidden}")

    require(kernel.count(
        '#include "actlize_extensions/cutlass/gemm/kernel/detail/ppu_splitk_shared_prefix_probe.hpp"') == 1,
        "kernel prefix include changed")
    require(kernel.count("run_splitk_shared_prefix_probe<") == 1,
            "kernel prefix call changed")
    require(kernel.count("store_splitk_accumulators_direct(") == 3,
            "prefix/full/default direct-store denominator changed")
    prefix_branch = kernel.index("#if defined(PPU_SPLITK_SHARED_PREFIX_POLICY)",
                                 kernel.index("int const plane"))
    full_branch = kernel.index("#elif defined(PPU_SPLITK_SHARED_SYNC_POLICY)",
                               prefix_branch)
    legacy_branch = kernel.index(
        "#elif defined(PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE", full_branch)
    require(prefix_branch < full_branch < legacy_branch,
            "prefix/full/legacy/default branch order changed")
    require(kernel.count("select exactly one Split-K shared diagnostic family") == 1,
            "mutually exclusive diagnostic-family guard changed")

    require(runner.count("local -a arms=(") == 1,
            "runner arm authority changed")
    for arm, defs in ARMS.items():
        require(runner.count(f"      {arm}") >= 1,
                f"runner lost arm: {arm}")
        require(runner.count(f"defs='{defs}'") == 1,
                f"runner define binding changed for {arm}")
        require(checker.count(f'    "{arm}",') == 1,
                f"checker denominator changed for {arm}")
    require("PROBE_REPEATS:-512" in runner,
            "high-repeat default changed")
    require("hgobjdump\" \"-res-usage=$candidate\"" in runner,
            "resource-usage capture changed")
    require("source.before.sha256" in runner and
            "source.after.sha256" in runner,
            "source authority before/after guard changed")
    require("production/legacy controls" in checker and
            "non-admission" in checker,
            "checker self-test lost historical/control distinction")


def self_test(helper: str, kernel: str, runner: str,
              checker: str) -> None:
    check_texts(helper, kernel, runner, checker)
    plants = (
        (helper.replace(
            "flat[offset] = accumulators(i);",
            "flat[offset] = float(offset);", 1), kernel, runner, checker),
        (helper.replace(
            "copy(tiled_s2r, tDsC, tDrC);", "", 1),
         kernel, runner, checker),
        (helper, kernel.replace(
            "run_splitk_shared_prefix_probe<",
            "run_splitk_shared_prefix_removed<", 1), runner, checker),
        (helper, kernel, runner.replace(
            "PPU_SPLITK_SHARED_PREFIX_POLICY=7",
            "PPU_SPLITK_SHARED_PREFIX_POLICY=6", 1), checker),
        (helper, kernel, runner, checker.replace(
            '    "r2s-snapshot-disjoint",', "", 1)),
    )
    for plant in plants:
        try:
            check_texts(*plant)
        except ValueError:
            pass
        else:
            raise AssertionError("prefix source-seam negative stayed green")


def main() -> int:
    try:
        texts = tuple(path.read_text() for path in
                      (HELPER, KERNEL, RUNNER, CHECKER))
        self_test(*texts)
        print("[fq-shared-prefix-root-source:self-test] PASS exact "
              "compile-time prefixes, production isolation, resource/source "
              "authority and five negative plants")
        return 0
    except (AssertionError, OSError, ValueError) as error:
        print(f"[fq-shared-prefix-root-source] FAIL: {error}",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
