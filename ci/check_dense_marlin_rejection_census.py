#!/usr/bin/env python3
"""Audit A2's removal of the dense Marlin 2/4-warp whitelist.

The source universe is deliberately the *committed dense tactic tables* --
those rows have already survived the ordinary tactic-space exclusions.  This
checker must never quietly reinterpret the rejected count as a subtraction
from the raw Cartesian product (the TileK guard made that mistake expensive).

``--write`` refreshes the checked-in TSV.  The default mode regenerates it in
memory and requires byte-for-byte equality.  The TSV deliberately retains one
row for every tactic rejected by the historical PRE_A2={2,4} implementation
and records that the current structural capability admits it.  Thus a future
filter cannot erase the evidence by merely making the current rejected set
empty.
"""

from __future__ import annotations

import argparse
import collections
import re
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
WARP_THREADS = 32
CURRENT_MIN_THREADS = 32
CURRENT_MAX_THREADS = 1024
PRE_A2_WARPS = frozenset({2, 4})
EXPECTED_RECOVERED_BY_WARP = {1: 610, 8: 1012, 16: 713, 32: 353}
EXPECTED_SOURCE_ROWS = 4790
EXPECTED_PRE_A2_ROWS = 2102
EXPECTED_RECOVERED_ROWS = 2688
REASON = "A2_COHORT_CAPABILITY_RECOVERED"
CATEGORY = "CURRENT_IMPLEMENTATION_REMOVED"
DOC_BEGIN = "<!-- BEGIN GENERATED MARLIN REJECTION CENSUS -->"
DOC_END = "<!-- END GENERATED MARLIN REJECTION CENSUS -->"


def line_at(path: Path, offset: int) -> str:
    return f"{path.relative_to(ROOT)}:{path.read_text()[:offset].count(chr(10)) + 1}"


def parse_cmake_capability() -> tuple[int, int, str]:
    """Read CMake's thread-range capability, rejecting a cohort allow-list."""
    text = CMAKE.read_text()
    thread_expr = re.search(
        r'math\(EXPR\s+_DENSE_MARLIN_CTA_THREADS\s+'
        r'"\$\{_DENSE_MARLIN_CTA_WARPS\}\s*\*\s*(\d+)"\)',
        text,
    )
    if not thread_expr or int(thread_expr.group(1)) != WARP_THREADS:
        raise RuntimeError("CMake does not derive CTA threads as CTA warps * 32")
    match = re.search(
        r"if\((?P<condition>\s*\(\s*_DENSE_MARLIN_CTA_THREADS\s+GREATER_EQUAL.*)\)\s*\n"
        r"\s*list\(APPEND _LOWBIT_DENSE_MARLIN_SWEEP_ROWS",
        text,
        re.S,
    )
    if not match:
        raise RuntimeError("cannot locate the CMake Marlin thread-capability block")
    condition = match.group("condition")
    bounds = re.fullmatch(
        r"\s*\(\s*_DENSE_MARLIN_CTA_THREADS\s+GREATER_EQUAL\s+(\d+)\s*\)\s*"
        r"AND\s*\(\s*_DENSE_MARLIN_CTA_THREADS\s+LESS_EQUAL\s+(\d+)\s*\)\s*",
        condition,
    )
    if not bounds:
        raise RuntimeError(
            "CMake Marlin capability is no longer one inclusive thread range: "
            + repr(condition)
        )
    minimum, maximum = map(int, bounds.groups())
    guard_offset = match.start("condition") + condition.find("_DENSE_MARLIN_CTA_THREADS")
    return minimum, maximum, line_at(CMAKE, guard_offset)


def require_once(text: str, pattern: str, label: str) -> None:
    hits = re.findall(pattern, text, re.S)
    if len(hits) != 1:
        raise RuntimeError(f"expected one {label}, got {len(hits)}")


def current_capability() -> tuple[int, int, str]:
    """Require CMake, scheduler, named kernel and wrapper to share one ability."""
    minimum, maximum, first_guard = parse_cmake_capability()
    scheduler = SCHED.read_text()
    kernel = KERNEL.read_text()
    wrapper = WRAPPER.read_text()

    helper = re.search(
        r"fixup_thread_count_capable\(\s*uint32_t\s+thread_count\s*\)\s*\{"
        r"(?P<body>.*?)\}",
        scheduler,
        re.S,
    )
    if not helper:
        raise RuntimeError("scheduler lost fixup_thread_count_capable")
    body = re.sub(r"\s+", "", helper.group("body"))
    want_body = (
        "returnthread_count>=uint32_t(cutlass::NumThreadsPerWarp)&&"
        "thread_count<=32u*uint32_t(cutlass::NumThreadsPerWarp)&&"
        "thread_count%uint32_t(cutlass::NumThreadsPerWarp)==0;"
    )
    if body != want_body:
        raise RuntimeError("scheduler capability is not warp-aligned 32..1024: " + body)

    require_once(
        scheduler,
        r"static_assert\(\s*FixupThreadCount\s*==\s*0\s*\|\|\s*"
        r"fixup_thread_count_capable\(FixupThreadCount\)",
        "scheduler explicit-or-derived capability assertion",
    )
    require_once(
        scheduler,
        r"static_assert\(\s*fixup_thread_count_capable\(Cohort\)",
        "scheduler derived capability assertion",
    )
    require_once(
        scheduler,
        r"static_assert\(\s*FixupThreadCount\s*==\s*0\s*\|\|\s*"
        r"FixupThreadCount\s*==\s*DerivedThreadCount",
        "scheduler explicit/derived exact binding",
    )
    require_once(
        scheduler,
        r"static_assert\(\s*Cohort\s*==\s*DerivedThreadCount",
        "scheduler cohort/accumulator exact binding",
    )
    require_once(
        kernel,
        r"static_assert\(\s*TileScheduler::fixup_thread_count_capable\(MaxThreadsPerBlock\)",
        "named-kernel capability assertion",
    )
    require_once(
        kernel,
        r"static_assert\(\s*TileScheduler::FixupThreadCount\s*==\s*MaxThreadsPerBlock",
        "named-kernel exact cohort assertion",
    )
    require_once(
        wrapper,
        r"static_assert\(\s*Kernel::TileScheduler::fixup_thread_count_capable\(\s*"
        r"Kernel::MaxThreadsPerBlock\s*\)",
        "generated-wrapper capability assertion",
    )
    require_once(
        wrapper,
        r"static_assert\(\s*Kernel::TileScheduler::FixupThreadCount\s*==\s*"
        r"Kernel::MaxThreadsPerBlock",
        "generated-wrapper exact cohort assertion",
    )

    if (minimum, maximum) != (CURRENT_MIN_THREADS, CURRENT_MAX_THREADS):
        raise RuntimeError(
            f"CMake admits [{minimum},{maximum}] threads, scheduler admits "
            f"[{CURRENT_MIN_THREADS},{CURRENT_MAX_THREADS}]"
        )
    return minimum, maximum, first_guard


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


def capable(warps: int, minimum: int, maximum: int) -> bool:
    threads = warps * WARP_THREADS
    return minimum <= threads <= maximum and threads % WARP_THREADS == 0


def generate() -> tuple[str, dict[str, collections.Counter[int]], tuple[int, int]]:
    minimum, maximum, first_guard = current_capability()
    header = (
        "bits\ttable\tsource_index\ttm\ttn\ttk\twm\twn\tstages\tb_chunk\t"
        "m_warps\tn_warps\tcta_warps\tcta_threads\tpre_a2_supported_cta_warps\t"
        "current_capability\tpre_a2_status\tcurrent_status\tcategory\ttransition_id\t"
        "current_capability_guard\n"
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
            if warps in PRE_A2_WARPS:
                continue
            if not capable(warps, minimum, maximum):
                raise RuntimeError(
                    f"A2 capability still rejects PRE_A2 row {bits}:{source_index} cohort w{warps}"
                )
            output.append(
                f"{bits}\t{path.name}\t{source_index}\t{tm}\t{tn}\t{tk}\t"
                f"{wm}\t{wn}\t{stages}\t{bc}\t{mw}\t{nw}\t{warps}\t{warps * 32}\t"
                f"2|4\twarp-aligned-threads-{minimum}..{maximum}\tREJECTED\tADMITTED\t"
                f"{CATEGORY}\t{REASON}\t{first_guard}\n"
            )
        census[bits] = counter
    return "".join(output), census, (minimum, maximum)


def verify_probe(
        census: dict[str, collections.Counter[int]], capability: tuple[int, int]) -> None:
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
    recovered_domain = domains["i4"] - PRE_A2_WARPS
    if set(representatives) != recovered_domain:
        raise RuntimeError(
            f"L131 representatives {sorted(representatives)} do not cover every recovered "
            f"cohort {sorted(recovered_domain)}"
        )

    minimum, maximum = capability
    recovered = collections.Counter()
    source = pre_a2 = current = 0
    for counts in census.values():
        source += sum(counts.values())
        pre_a2 += sum(n for warps, n in counts.items() if warps in PRE_A2_WARPS)
        current += sum(n for warps, n in counts.items() if capable(warps, minimum, maximum))
        for warps, n in counts.items():
            if warps not in PRE_A2_WARPS and capable(warps, minimum, maximum):
                recovered[warps] += n
    if dict(sorted(recovered.items())) != EXPECTED_RECOVERED_BY_WARP:
        raise RuntimeError(
            f"A2 per-cohort recovery drifted: got {dict(sorted(recovered.items()))}, "
            f"expected {EXPECTED_RECOVERED_BY_WARP}"
        )
    if (source, pre_a2, current, current - pre_a2) != (
            EXPECTED_SOURCE_ROWS, EXPECTED_PRE_A2_ROWS,
            EXPECTED_SOURCE_ROWS, EXPECTED_RECOVERED_ROWS):
        raise RuntimeError(
            "A2 closure drifted: "
            f"source={source} PRE_A2={pre_a2} current={current} recovered={current - pre_a2}"
        )


def generated_doc_block(
        census: dict[str, collections.Counter[int]], capability: tuple[int, int]) -> str:
    minimum, maximum = capability
    output = [
        DOC_BEGIN,
        "Historical baseline: `PRE_A2={2,4}` CTA warps (threads `64|128`).",
        f"Current capability: warp-aligned CTA threads in `[{minimum},{maximum}]`.",
        "",
        "| format | committed legal rows | PRE_A2 admitted | PRE_A2 rejected | current admitted | current rejected | A2 recovered by cohort |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    total_source = total_pre = total_current = total_recovered = 0
    for bits, _path in TABLES:
        counts = census[bits]
        source = sum(counts.values())
        pre = sum(count for warps, count in counts.items() if warps in PRE_A2_WARPS)
        current = sum(
            count for warps, count in counts.items() if capable(warps, minimum, maximum))
        recovered = current - pre
        breakdown = ", ".join(
            f"w{warps}={count}" for warps, count in sorted(counts.items())
            if warps not in PRE_A2_WARPS and capable(warps, minimum, maximum)
        )
        output.append(
            f"| {bits} | {source} | {pre} | {source - pre} | {current} | "
            f"{source - current} | {breakdown} |"
        )
        total_source += source
        total_pre += pre
        total_current += current
        total_recovered += recovered
    output.extend((
        f"| **total** | **{total_source}** | **{total_pre}** | "
        f"**{total_source - total_pre}** | **{total_current}** | "
        f"**{total_source - total_current}** | **{total_recovered} recovered** |",
        "",
        "All committed rows have integral `TM/WM` and `TN/WN`.  A2 removes exactly",
        "the former current-implementation rejection:",
        "",
        "| transition id | category | rows | meaning |",
        "|---|---|---:|---|",
        f"| `{REASON}` | `{CATEGORY}` | {total_recovered} | Every row rejected only by PRE_A2's 2/4-warp whitelist is admitted by the current structural capability. |",
        "| current rejection | — | 0 | Current capability admits 4790/4790 committed-legal rows. |",
        "| hardware/ISA limitation | — | 0 | A2 adds no claim of device speed or correctness; those remain box gates. |",
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
        expected, census, capability = generate()
        verify_probe(census, capability)
        old_doc, expected_doc = checked_doc_text(generated_doc_block(census, capability))
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
    recovered_total = 0
    minimum, maximum = capability
    for bits, counts in census.items():
        source = sum(counts.values())
        pre = sum(count for warps, count in counts.items() if warps in PRE_A2_WARPS)
        current = sum(
            count for warps, count in counts.items() if capable(warps, minimum, maximum))
        recovered = current - pre
        recovered_total += recovered
        recovered_breakdown = ",".join(
            f"w{warps}={count}" for warps, count in sorted(counts.items())
            if warps not in PRE_A2_WARPS and capable(warps, minimum, maximum)
        )
        summaries.append(
            f"{bits}=PRE_A2:{pre}/{source}->current:{current}/{source} "
            f"(+{recovered}: {recovered_breakdown})"
        )
    print(
        "[dense-marlin-rejection-census] PASS: "
        f"PRE_A2={sorted(PRE_A2_WARPS)}; current=warp-aligned-{minimum}..{maximum}; "
        f"implementation guards agree; recovered={recovered_total}/{EXPECTED_RECOVERED_ROWS}; "
        "current=4790/4790; " + "; ".join(summaries)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
