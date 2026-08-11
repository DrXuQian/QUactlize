#!/usr/bin/env python3
"""Fail-closed contract for the full-table dense Marlin sweep.

This is deliberately a source contract, not a device benchmark.  It proves the
parts that must be true before a box run is meaningful:

* the committed dense tables are filtered to the only CTA cohorts supported by
  the Marlin cooperative (two or four warps, i.e. 64/128 threads);
* the target has private main/unit source paths, because PPUToolchain keys a
  device object by source path rather than target plus flags;
* both host and device compiles see ``DENSE_MARLIN_SWEEP``;
* every generated wrapper is unconditionally a Marlin wrapper -- there is no
  runtime option which can silently turn the sweep back into DP;
* provenance/sample identity names Marlin, and ordinary dense tactic caches
  cannot be loaded or written by this binary.

The exact eligible-row census is recomputed from the committed seven-field
tables here.  It is not copied from CMake's status message.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CMAKE = ROOT / "quactlize/csrc/CMakeLists.txt.in"
BENCH = ROOT / "benchmarks/test_lowbit_dense_bench.cu"
DISPATCH = ROOT / "benchmarks/lowbit_dense_unit.inc"
BUILD = ROOT / "build.sh"
TABLES = (
    ROOT / "benchmarks/lowbit_dense_configs.inc",
    ROOT / "benchmarks/lowbit_dense_i1_configs.inc",
    ROOT / "benchmarks/lowbit_dense_i2_configs.inc",
)

ROW_RE = re.compile(r"^\s*X\((\d+(?:,\d+){6}),B\)\s*\\?\s*$", re.M)


def rows(path: Path) -> list[tuple[int, ...]]:
    parsed = [tuple(map(int, match.split(","))) for match in ROW_RE.findall(path.read_text())]
    if not parsed:
        raise RuntimeError(f"{path.name}: no seven-field committed rows")
    return parsed


def eligible(row: tuple[int, ...]) -> bool:
    tm, tn, _tk, wm, wn, _st, _bc = row
    if tm % wm or tn % wn:
        return False
    return (tm // wm) * (tn // wn) in (2, 4)


def target_call(cmake: str) -> str:
    match = re.search(
        r"quactlize_ppu_executable\(\s*test_lowbit_dense_marlin_sweep\b(.*?)\n\s*\)",
        cmake,
        re.S,
    )
    return match.group(1) if match else ""


def compile_def_call(cmake: str) -> str:
    match = re.search(
        r"target_compile_definitions\(\s*test_lowbit_dense_marlin_sweep\b(.*?)\n\s*\)",
        cmake,
        re.S,
    )
    return match.group(1) if match else ""


def marlin_dispatch_arm(dispatch: str) -> str:
    """Return the first preprocessor arm owned by DENSE_MARLIN_SWEEP."""
    match = re.search(
        r"#if\s+defined\(DENSE_MARLIN_SWEEP\)(.*?)(?=\n\s*#(?:elif|else|endif)\b)",
        dispatch,
        re.S,
    )
    return match.group(1) if match else ""


def audit(cmake: str, bench: str, dispatch: str, build: str) -> list[str]:
    bad: list[str] = []

    call = target_call(cmake)
    host_defs = compile_def_call(cmake)
    if not call:
        bad.append("missing test_lowbit_dense_marlin_sweep target")
    else:
        # These names intentionally describe paths, not merely variables.  Reusing
        # either ordinary source path recreates PPUToolchain's target/flag cache bug.
        if "MARLIN_SWEEP" not in call.upper() or "_MARLIN_SWEEP_MAIN" not in call or \
                "_MARLIN_SWEEP_UNIT_SRCS" not in call:
            bad.append("Marlin sweep target does not consume private main and unit source lists")
        if "test_lowbit_dense_bench.cu" in call or "_LOWBIT_DENSE_UNIT_SRCS" in call:
            bad.append("Marlin sweep target reuses an ordinary dense source path")
        if not re.search(r"DEV_COMPILE_FLAGS[^\n)]*\$\{[^}]*MARLIN_SWEEP[^}]*DEFS", call):
            bad.append("Marlin sweep device compile does not consume its dedicated definitions")

    if "DENSE_MARLIN_SWEEP=1" not in host_defs:
        bad.append("Marlin sweep host compile does not define DENSE_MARLIN_SWEEP=1")
    if not re.search(
            r"set\([^)]*MARLIN_SWEEP[^)]*DEFS.*?-DDENSE_MARLIN_SWEEP=1", cmake, re.S):
        bad.append("Marlin sweep device definitions do not contain -DDENSE_MARLIN_SWEEP=1")

    for token in (
        "lowbit_dense_marlin_sweep_units",
        "test_lowbit_dense_marlin_sweep_main.cu",
        "LOWBIT_DENSE_MARLIN_SWEEP_CONFIGS",
    ):
        if token not in cmake:
            bad.append(f"CMake generator is missing private sweep token {token!r}")

    # CMake must derive the cohort from each committed row.  The two divisions
    # are what distinguish CTA threads from a coincidentally equal TM or TN.
    if not re.search(r"math\(EXPR\s+[^\n]*[Mm][Aa][Rr][Ll][Ii][Nn][^\n]*[Tt][Mm][^\n]*/[^\n]*[Ww][Mm]", cmake):
        bad.append("CMake does not derive the Marlin CTA M-warp count from TM/WM")
    if not re.search(r"math\(EXPR\s+[^\n]*[Mm][Aa][Rr][Ll][Ii][Nn][^\n]*[Tt][Nn][^\n]*/[^\n]*[Ww][Nn]", cmake):
        bad.append("CMake does not derive the Marlin CTA N-warp count from TN/WN")
    if not re.search(r"(?:EQUAL|STREQUAL)\s+2.*?(?:OR|or).*?(?:EQUAL|STREQUAL)\s+4", cmake, re.S):
        bad.append("CMake filter is not the exact 2-or-4-warp Marlin cohort")
    for token in ("source_rows", "eligible_rows", "filtered_rows"):
        if token not in cmake.lower():
            bad.append(f"Marlin sweep does not expose its {token.replace('_', '-')} census")

    arm = marlin_dispatch_arm(dispatch)
    if not arm:
        bad.append("generated wrapper has no DENSE_MARLIN_SWEEP arm")
    else:
        if "MarlinGemm" not in arm or '"marlin"' not in arm:
            bad.append("Marlin sweep arm does not instantiate and label MarlinGemm")
        for forbidden in (
            "options.marlin",
            "typename Cfg<GroupSize, TM, TN, TK, WM, WN, ST>::Gemm",
            '"non-persistent"',
        ):
            if forbidden in arm:
                bad.append(f"Marlin sweep arm can silently fall back through {forbidden!r}")
        if "static_assert" not in arm or not re.search(r"(?:64|2).*?(?:128|4)", arm, re.S):
            bad.append("wrapper does not fail closed outside the 64/128-thread cohort")

    if not re.search(r"DENSE_MARLIN_SWEEP.*?scheduler=marlin", bench, re.S):
        bad.append("benchmark provenance does not bind DENSE_MARLIN_SWEEP to scheduler=marlin")
    if 'return "dense-marlin-v1";' not in bench:
        bad.append("Marlin samples can masquerade as ordinary dense-v1 samples")
    if "if (!options.tactic_file.empty() || !options.save_tactic_file.empty())" not in bench:
        bad.append("Marlin sweep cache rejection is not controlled by both load and save paths")
    if not re.search(
            r"DENSE_MARLIN_SWEEP.*?(?:tactic_file|save_tactic_file).*?(?:reject|unsupported|cannot|must not)",
            bench,
            re.S | re.I,
    ):
        bad.append("Marlin sweep does not explicitly reject ordinary tactic-cache load/save")

    if '"test_lowbit_dense_marlin_sweep"' not in build:
        bad.append("build.sh has no exact Marlin sweep target route")

    return bad


def replace_once(text: str, old: str, new: str, label: str, bad: list[str]) -> str | None:
    if text.count(old) != 1:
        bad.append(f"cannot plant {label}: expected one {old!r}, got {text.count(old)}")
        return None
    return text.replace(old, new, 1)


def main() -> int:
    source = {
        "cmake": CMAKE.read_text(),
        "bench": BENCH.read_text(),
        "dispatch": DISPATCH.read_text(),
        "build": BUILD.read_text(),
    }
    bad = audit(**source)

    census: dict[str, tuple[int, int]] = {}
    try:
        for table in TABLES:
            table_rows = rows(table)
            census[table.name] = (len(table_rows), sum(map(eligible, table_rows)))
    except RuntimeError as exc:
        bad.append(str(exc))

    # Each plant targets a distinct silent-failure class.  A green plant means
    # this checker has become ceremonial and must fail the tree.
    cmake_plants = (
        ("ordinary-unit-path", "lowbit_dense_marlin_sweep_units", "lowbit_dense_units"),
        ("ordinary-main-path", "test_lowbit_dense_marlin_sweep_main.cu", "test_lowbit_dense_bench.cu"),
        ("missing-device-define", "-DDENSE_MARLIN_SWEEP=1", "-DDENSE_MARLIN_SWEEP_REMOVED=1"),
        ("one-warp-cohort", "_DENSE_MARLIN_CTA_WARPS EQUAL 2",
         "_DENSE_MARLIN_CTA_WARPS EQUAL 1"),
    )
    for label, old, new in cmake_plants:
        planted_text = replace_once(source["cmake"], old, new, label, bad)
        if planted_text is None:
            continue
        planted = dict(source)
        planted["cmake"] = planted_text
        if not audit(**planted):
            bad.append(f"contract accepted planted {label}")

    dispatch_plants = (
        ("runtime-marlin-switch", "#if defined(DENSE_MARLIN_SWEEP)",
         "#if defined(DENSE_MARLIN_SWEEP)\n  if (options.marlin)"),
    )
    for label, old, new in dispatch_plants:
        planted_text = replace_once(source["dispatch"], old, new, label, bad)
        if planted_text is None:
            continue
        planted = dict(source)
        planted["dispatch"] = planted_text
        if not audit(**planted):
            bad.append(f"contract accepted planted {label}")

    # The same return spelling also belongs to the one-row A/B arm.  Mutate
    # only the full-sweep preprocessor arm so the control cannot accidentally
    # prove that an unrelated branch is inspected.
    arm = marlin_dispatch_arm(source["dispatch"])
    old = 'return run<G>(options, dense_tactic(cfg), "marlin");'
    if arm.count(old) != 1:
        bad.append(f"cannot plant dp-fallback in sweep arm: expected one return, got {arm.count(old)}")
    else:
        planted_arm = arm.replace(
            old,
            'return run<typename Cfg<GroupSize, TM, TN, TK, WM, WN, ST>::Gemm>('
            'options, dense_tactic(cfg), "non-persistent");',
            1,
        )
        planted = dict(source)
        planted["dispatch"] = source["dispatch"].replace(arm, planted_arm, 1)
        if not audit(**planted):
            bad.append("contract accepted planted dp-fallback")

    bench_plants = (
        ("ordinary-sample-identity", 'return "dense-marlin-v1";', 'return "dense-v1";'),
        ("cache-guard-disabled",
         "if (!options.tactic_file.empty() || !options.save_tactic_file.empty())",
         "if (false && (!options.tactic_file.empty() || !options.save_tactic_file.empty()))"),
    )
    for label, old, new in bench_plants:
        planted_text = replace_once(source["bench"], old, new, label, bad)
        if planted_text is None:
            continue
        planted = dict(source)
        planted["bench"] = planted_text
        if not audit(**planted):
            bad.append(f"contract accepted planted {label}")

    if bad:
        print("[dense-marlin-sweep-contract] FAIL: " + "; ".join(bad))
        return 1
    summary = "; ".join(
        f"{name}={eligible_count}/{total} eligible ({total - eligible_count} filtered)"
        for name, (total, eligible_count) in census.items()
    )
    print("[dense-marlin-sweep-contract] PASS: private host/device route, exact 64/128-thread "
          "filter, forced Marlin wrappers, distinct provenance and cache rejection; "
          f"eight source plants rejected; {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
