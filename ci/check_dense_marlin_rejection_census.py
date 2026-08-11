#!/usr/bin/env python3
"""Audit every row rejected by the second-stage dense Marlin filter.

The source universe is deliberately the *committed dense tactic tables* --
those rows have already survived the ordinary tactic-space exclusions.  This
checker must never quietly reinterpret the rejected count as a subtraction
from the raw Cartesian product (the TileK guard made that mistake expensive).

``--write`` refreshes the checked-in TSV.  The default mode regenerates it in
memory and requires byte-for-byte equality, so every rejected row keeps an
explicit, reviewable reason when either a table or a guard moves.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dev/fold_derivation/MARLIN_SWEEP_REJECTION_CENSUS.tsv"
DOC = ROOT / "dev/fold_derivation/MARLIN_SWEEP_REJECTION_CENSUS.md"
CMAKE = ROOT / "quactlize/csrc/CMakeLists.txt.in"
SCHED = ROOT / "third_party/actlize/include/cutlass/gemm/kernel/ppu_tile_scheduler_marlin.hpp"
KERNEL = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_marlin.hpp"
WRAPPER = ROOT / "benchmarks/lowbit_dense_unit.inc"
PROBE = ROOT / "dev/fold_derivation/l131_marlin_rejected_cohorts.cu"

TABLES = (
    ("i4", ROOT / "benchmarks/lowbit_dense_configs.inc"),
    ("i2", ROOT / "benchmarks/lowbit_dense_i2_configs.inc"),
    ("i1", ROOT / "benchmarks/lowbit_dense_i1_configs.inc"),
)
ROW_RE = re.compile(r"^\s*X\((\d+(?:,\d+){6}),B\)\s*\\?\s*$", re.M)
REASON = "MARLIN_FIXUP_COHORT_NOT_IN_SUPPORTED_SET"
CATEGORY = "CURRENT_IMPLEMENTATION"
DOC_BEGIN = "<!-- BEGIN GENERATED MARLIN REJECTION CENSUS -->"
DOC_END = "<!-- END GENERATED MARLIN REJECTION CENSUS -->"


def line_of(path: Path, needle: str) -> int:
    lines = path.read_text().splitlines()
    hits = [i for i, line in enumerate(lines, 1) if needle in line]
    if len(hits) != 1:
        raise RuntimeError(f"{path.relative_to(ROOT)}: expected one {needle!r}, got {hits}")
    return hits[0]


def rel_line(path: Path, needle: str) -> str:
    return f"{path.relative_to(ROOT)}:{line_of(path, needle)}"


def line_at(path: Path, offset: int) -> str:
    return f"{path.relative_to(ROOT)}:{path.read_text()[:offset].count(chr(10)) + 1}"


def parse_cmake_supported_cohorts() -> tuple[set[int], str]:
    """Read the actual CMake OR-of-EQUAL cohort guard, fail closed otherwise."""
    text = CMAKE.read_text()
    match = re.search(
        r"if\((?P<condition>\s*\(\s*_DENSE_MARLIN_CTA_WARPS\s+EQUAL.*)\)\s*\n"
        r"\s*list\(APPEND _LOWBIT_DENSE_MARLIN_SWEEP_ROWS",
        text,
        re.S,
    )
    if not match:
        raise RuntimeError("cannot locate the CMake Marlin cohort admission block")
    condition = match.group("condition")
    atom = re.compile(r"\(\s*_DENSE_MARLIN_CTA_WARPS\s+EQUAL\s+(\d+)\s*\)")
    values = [int(value) for value in atom.findall(condition)]
    normalized = atom.sub("ATOM", condition)
    if not values or len(values) != len(set(values)) or not re.fullmatch(
            r"\s*ATOM(?:\s+OR\s+ATOM)*\s*", normalized):
        raise RuntimeError(
            "CMake Marlin cohort guard is no longer a unique OR-of-EQUAL set: "
            + repr(condition)
        )
    if any(value <= 0 for value in values):
        raise RuntimeError(f"CMake Marlin cohort set contains a nonpositive value: {values}")
    guard_offset = match.start("condition") + condition.find("_DENSE_MARLIN_CTA_WARPS")
    return set(values), line_at(CMAKE, guard_offset)


def parse_cpp_supported_values(path: Path, variable: str, message: str) -> set[int]:
    """Parse one C++ static_assert that is strictly an OR-of-== set."""
    text = path.read_text()
    pattern = re.compile(
        r"static_assert\((?P<condition>\s*" + re.escape(variable) +
        r"\s*==.*?),\s*\"" + re.escape(message) + r"\"\s*\);",
        re.S,
    )
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(
            f"{path.relative_to(ROOT)}: expected one support assertion {message!r}, "
            f"got {len(matches)}"
        )
    condition = matches[0].group("condition")
    atom = re.compile(re.escape(variable) + r"\s*==\s*(\d+)u?")
    values = [int(value) for value in atom.findall(condition)]
    normalized = atom.sub("ATOM", condition)
    if not values or len(values) != len(set(values)) or not re.fullmatch(
            r"\s*ATOM(?:\s*\|\|\s*ATOM)*\s*", normalized):
        raise RuntimeError(
            f"{path.relative_to(ROOT)}: support assertion is not a unique OR-of-== set: "
            + repr(condition)
        )
    return set(values)


def supported_cohorts() -> tuple[set[int], str]:
    """Require CMake, scheduler, named kernel, wrapper and fixup to agree."""
    warps, first_guard = parse_cmake_supported_cohorts()
    threads = {warps_per_cta * 32 for warps_per_cta in warps}
    scheduler = parse_cpp_supported_values(
        SCHED,
        "FixupThreadCount",
        "Marlin cooperative supports only derived/exact 64/128-thread CTA cohorts",
    )
    kernel = parse_cpp_supported_values(
        KERNEL,
        "MaxThreadsPerBlock",
        "dense mixed-input Marlin supports exact 64/128-thread CTAs",
    )
    wrapper = parse_cpp_supported_values(
        WRAPPER,
        "G::GemmKernel::MaxThreadsPerBlock",
        "the dense Marlin sweep admits only 64/128-thread rows",
    )
    fixup = parse_cpp_supported_values(
        SCHED,
        "Cohort",
        "Marlin cooperative derived an unsupported CTA cohort",
    )
    expected = {
        "scheduler explicit cohort (plus derived=0)": scheduler - {0},
        "named kernel": kernel,
        "generated wrapper": wrapper,
        "fixup": fixup,
    }
    if 0 not in scheduler:
        raise RuntimeError("scheduler support assertion lost its derived-cohort zero arm")
    mismatches = {name: values for name, values in expected.items() if values != threads}
    if mismatches:
        detail = ", ".join(f"{name}={sorted(values)}" for name, values in mismatches.items())
        raise RuntimeError(
            f"CMake admits warp cohorts {sorted(warps)} / threads {sorted(threads)}, "
            f"but implementation guards disagree: {detail}"
        )
    return warps, first_guard


def parse_rows(path: Path) -> list[tuple[int, ...]]:
    rows = [tuple(map(int, text.split(","))) for text in ROW_RE.findall(path.read_text())]
    if not rows:
        raise RuntimeError(f"{path.relative_to(ROOT)}: no seven-field rows")
    if len(rows) != len(set(rows)):
        raise RuntimeError(f"{path.relative_to(ROOT)}: duplicate committed rows")
    return rows


def cohort(row: tuple[int, ...]) -> tuple[int, int, int]:
    tm, tn, _tk, wm, wn, _st, _bc = row
    if tm % wm or tn % wn:
        raise RuntimeError(f"committed dense row has non-integral warp topology: {row}")
    mw, nw = tm // wm, tn // wn
    return mw, nw, mw * nw


def generate() -> tuple[str, dict[str, collections.Counter[int]], set[int]]:
    admitted_warps, first_guard = supported_cohorts()
    admitted_text = "|".join(map(str, sorted(admitted_warps)))
    header = (
        "bits\ttable\tsource_index\ttm\ttn\ttk\twm\twn\tstages\tb_chunk\t"
        "m_warps\tn_warps\tcta_warps\tcta_threads\tsupported_cta_warps\t"
        "category\treason_id\tfirst_guard\n"
    )
    output = [header]
    census: dict[str, collections.Counter[int]] = {}
    for bits, path in TABLES:
        counter: collections.Counter[int] = collections.Counter()
        rows = parse_rows(path)
        for source_index, row in enumerate(rows, 1):
            tm, tn, tk, wm, wn, stages, bc = row
            mw, nw, warps = cohort(row)
            counter[warps] += 1
            if warps in admitted_warps:
                continue
            output.append(
                f"{bits}\t{path.name}\t{source_index}\t{tm}\t{tn}\t{tk}\t"
                f"{wm}\t{wn}\t{stages}\t{bc}\t{mw}\t{nw}\t{warps}\t{warps * 32}\t"
                f"{admitted_text}\t{CATEGORY}\t{REASON}\t{first_guard}\n"
            )
        census[bits] = counter
    return "".join(output), census, admitted_warps


def verify_probe(census: dict[str, collections.Counter[int]], admitted_warps: set[int]) -> None:
    text = PROBE.read_text()
    i4_rows = set(parse_rows(TABLES[0][1]))
    representatives = {
        1: (8, 16, 64, 8, 16, 2, 0),
        8: (8, 128, 64, 8, 16, 2, 0),
        16: (8, 256, 64, 8, 16, 2, 0),
        32: (32, 256, 64, 16, 16, 2, 0),
    }
    for warps, row in representatives.items():
        if row not in i4_rows:
            raise RuntimeError(f"L131 representative vanished from committed i4 table: {row}")
        if cohort(row)[2] != warps:
            raise RuntimeError(f"L131 representative has wrong cohort: {row}")
        compact = ",".join(map(str, row))
        if compact not in re.sub(r"\s+", "", text):
            raise RuntimeError(f"L131 source does not instantiate representative {row}")
    domains = {bits: set(counts) for bits, counts in census.items()}
    if len({frozenset(domain) for domain in domains.values()}) != 1:
        raise RuntimeError(f"committed table cohort domains disagree: {domains}")
    rejected_domain = domains["i4"] - admitted_warps
    if set(representatives) != rejected_domain:
        raise RuntimeError(
            f"L131 representatives {sorted(representatives)} do not cover every rejected "
            f"cohort {sorted(rejected_domain)}"
        )


def generated_doc_block(
        census: dict[str, collections.Counter[int]], admitted_warps: set[int]) -> str:
    warp_text = "|".join(map(str, sorted(admitted_warps)))
    thread_text = "|".join(str(warps * 32) for warps in sorted(admitted_warps))
    output = [
        DOC_BEGIN,
        f"Parsed supported CTA warp cohorts: `{warp_text}` (threads: `{thread_text}`).",
        "",
        "| format | committed legal source rows | Marlin rows (parsed supported set) | rejected | rejected by CTA warps |",
        "|---|---:|---:|---:|---|",
    ]
    total_source = total_accepted = total_rejected = 0
    for bits, _path in TABLES:
        counts = census[bits]
        source = sum(counts.values())
        accepted = sum(count for warps, count in counts.items() if warps in admitted_warps)
        rejected = source - accepted
        breakdown = ", ".join(
            f"w{warps}={count}" for warps, count in sorted(counts.items())
            if warps not in admitted_warps
        )
        output.append(f"| {bits} | {source} | {accepted} | {rejected} | {breakdown} |")
        total_source += source
        total_accepted += accepted
        total_rejected += rejected
    output.extend((
        f"| **total** | **{total_source}** | **{total_accepted}** | **{total_rejected}** | |",
        "",
        "All committed rows have integral `TM/WM` and `TN/WN`.  There is exactly one",
        "rejection reason:",
        "",
        "| reason id | category | rows | meaning |",
        "|---|---|---:|---|",
        f"| `{REASON}` | `{CATEGORY}` | {total_rejected} | The named Marlin cooperative is currently implemented and gated for the parsed set: exact {thread_text.replace('|', '/')}-thread ({warp_text.replace('|', '/')}-warp) CTA cohorts. |",
        "| hardware/ISA limitation | — | 0 | No rejected row reached a hardware/ISA diagnostic. |",
        "| accidental independent enumeration rule | — | 0 | The CMake rule mirrors four fail-closed implementation assertions; it is not an otherwise unexplained second filter. |",
        DOC_END,
    ))
    return "\n".join(output)


def checked_doc_text(block: str) -> tuple[str, str]:
    if not DOC.exists():
        raise RuntimeError(f"missing {DOC.relative_to(ROOT)}")
    text = DOC.read_text()
    pattern = re.compile(re.escape(DOC_BEGIN) + r".*?" + re.escape(DOC_END), re.S)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(
            f"{DOC.relative_to(ROOT)}: expected one generated census block, got {len(matches)}"
        )
    rewritten = text[:matches[0].start()] + block + text[matches[0].end():]
    return text, rewritten


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="refresh checked-in TSV")
    args = parser.parse_args()
    try:
        expected, census, admitted_warps = generate()
        verify_probe(census, admitted_warps)
        old_doc, expected_doc = checked_doc_text(generated_doc_block(census, admitted_warps))
    except RuntimeError as exc:
        print(f"[dense-marlin-rejection-census] FAIL: {exc}")
        return 1

    if args.write:
        OUT.write_text(expected)
        if old_doc != expected_doc:
            DOC.write_text(expected_doc)
    elif not OUT.exists():
        print(f"[dense-marlin-rejection-census] FAIL: missing {OUT.relative_to(ROOT)}")
        return 1
    elif OUT.read_text() != expected:
        print("[dense-marlin-rejection-census] FAIL: TSV is stale; run this checker with --write")
        return 1
    elif old_doc != expected_doc:
        print("[dense-marlin-rejection-census] FAIL: MD generated block is stale; run this checker with --write")
        return 1

    summaries = []
    rejected_total = 0
    for bits, counts in census.items():
        source = sum(counts.values())
        accepted = sum(count for warps, count in counts.items() if warps in admitted_warps)
        rejected = source - accepted
        rejected_total += rejected
        rejected_breakdown = ",".join(
            f"w{warps}={count}" for warps, count in sorted(counts.items())
            if warps not in admitted_warps
        )
        summaries.append(f"{bits}={rejected}/{source} rejected ({rejected_breakdown})")
    print(
        "[dense-marlin-rejection-census] PASS: "
        f"parsed CMake cohorts={sorted(admitted_warps)}; implementation guards agree; "
        f"{rejected_total} relative-to-committed-legal rows each carry {REASON}/"
        f"{CATEGORY}; " + "; ".join(summaries)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
