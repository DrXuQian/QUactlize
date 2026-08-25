#!/usr/bin/env python3
"""Guard the one-build extension of the Q4_K Split-K prefix bundle."""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_fq_q4k_split_shared_prefix_closure_box.sh"
REPORTER = ROOT / "tools/report_fq_split_shared_prefix_codegen.py"
CHECKER = ROOT / "tools/check_fq_split_shared_prefix_root.py"
SOURCE_SHA = "4288d8f651c5c8556e399bcf43392e621d692f7b"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_texts(runner: str, reporter: str, checker: str) -> None:
    require(runner.count(f"source_sha={SOURCE_SHA}") == 1,
            "parent device bundle SHA changed")
    require(runner.count("build-legacy-shared-output") >= 1,
            "closure lost the named legacy build directory")
    require(runner.count(
        "PPU_DEFS='PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE=1'") == 1,
        "true legacy-output negative define changed")
    require(runner.count("TARGET=test_fully_quantized_internal_sweep") == 1,
            "closure gained another build invocation")
    require("ANALYSIS-ONLY-RESUME" in runner and
            'case "$resume"' in runner and
            "finalize_closure" in runner,
            "preserved-artifact analysis resume disappeared")
    for arm in (
        "production-direct", "accumulator-opaque", "clone-opaque",
        "cta-only", "flat-constant-disjoint",
        "flat-accumulator-disjoint", "r2s-vector-disjoint",
        "r2s-scalar-disjoint", "r2s-snapshot-disjoint",
        "r2s-s2r-vector-disjoint", "r2s-s2r-scalar-disjoint",
        "full-discard",
    ):
        require(runner.count(arm) >= 1, f"reused arm missing: {arm}")
    require(r"sha256=\\([0-9a-f]\\{64\\}\\)" in runner,
            "reused binary hash parser changed")
    require("source.before.sha256" in runner and
            "source.after.sha256" in runner and
            "cat-file blob \"$source_sha:$path\"" in runner,
            "parent source authority changed")
    require('sha256sum -c "$source_out/results/authority.sha256"' in runner,
            "parent result authority check disappeared")
    require('"-res-usage=$candidate"' in runner and
            '-line "-func=$candidate"' in runner,
            "exact-symbol resource/line disassembly changed")
    require("[ \"$count\" -eq 2 ]" in runner,
            "AP0/AP1 exact kernel denominator changed")
    require("kernel=$kernel codegen denominator=" in runner and
            "for kernel in 1 2" in runner,
            "exact ELF-ordinal codegen denominator changed")
    require("legacy codegen binary authority is missing or non-unique" in runner and
            "correctness_repeats=$repeats" in runner,
            "preserved legacy binary/repeat authority changed")
    require("--legacy-shared-output-direct" in runner and
            "--legacy-shared-output-probe" in runner,
            "legacy logs no longer enter the adjudicator")

    require('for name in ("mma", "smem.st", "tsm.st"' in reporter,
            "shared-store opcode census changed")
    require("resource_hint_mode=" in reporter and
            "resource_hints=" in reporter and
            "shared_store_forms=" in reporter and
            '"KernelAiuPackedA<"' in reporter and
            'else "UNRESOLVED"' in reporter,
            "flexible raw resource evidence disappeared")
    require("shared-store negative" in reporter,
            "reporter negative control disappeared")

    require('    "legacy-shared-output",' in checker,
            "checker lost the true historical negative")
    require("full_discard=COUNTERFACTUAL-NOT-ADMISSION" in checker,
            "discard arm was promoted back into admission")
    require("UNADJUDICATED_LEGACY_SHARED_OUTPUT_DID_NOT_REPRODUCE" in checker,
            "legacy nonreproduction no longer fails closed")


def self_test(runner: str, reporter: str, checker: str) -> None:
    check_texts(runner, reporter, checker)
    plants = (
        (runner.replace(SOURCE_SHA, "0" * 40, 1), reporter, checker),
        (runner.replace(
            "PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE=1",
            "PPU_SPLITK_SHARED_PROBE_DISCARD_GMEM=1", 1),
         reporter, checker),
        (runner.replace('"-res-usage=$candidate"', '"-isa"', 1),
         reporter, checker),
        (runner.replace("ANALYSIS-ONLY-RESUME", "FRESH-REBUILD", 1),
         reporter, checker),
        (runner, reporter.replace(
            'for name in ("mma", "smem.st", "tsm.st"',
            'for name in ("mma", "removed.store", "tsm.st"', 1), checker),
        (runner, reporter, checker.replace(
            "full_discard=COUNTERFACTUAL-NOT-ADMISSION",
            "full_discard=ADMISSION", 1)),
    )
    for plant in plants:
        try:
            check_texts(*plant)
        except ValueError:
            pass
        else:
            raise AssertionError("closure source-seam negative stayed green")


def main() -> int:
    try:
        texts = tuple(path.read_text() for path in
                      (RUNNER, REPORTER, CHECKER))
        self_test(*texts)
        print("[fq-shared-prefix-closure-source:self-test] PASS parent/hash "
              "reuse, one legacy build, exact two-kernel codegen, corrected "
              "admission and six negative plants")
        return 0
    except (AssertionError, OSError, ValueError) as error:
        print(f"[fq-shared-prefix-closure-source] FAIL: {error}",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
