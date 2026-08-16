#!/usr/bin/env python3
"""Fail-closed structural contract for the full dense Stream-K sweep.

The device run owns numerical correctness and performance.  This local gate
owns a different question: does the binary enumerate every currently
compilable int4/gs32 Stream-K row from the committed dense authority, and do
all generated wrappers instantiate StreamKGemm rather than silently falling
back to the ordinary DP kernel?

The denominator is derived here from ``lowbit_dense_configs.inc``.  It is not
copied from the generated registry or from the runner, since agreement between
two consumers of the same wrong generated count would otherwise look like
evidence.  Every negative control below mutates one of the files passed to the
same ``audit`` function used for the real tree.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PATHS = {
    "table": ROOT / "benchmarks/lowbit_dense_configs.inc",
    "cmake": ROOT / "quactlize/csrc/CMakeLists.txt.in",
    "bench": ROOT / "benchmarks/test_lowbit_dense_bench.cu",
    "unit": ROOT / "benchmarks/lowbit_dense_unit.inc",
    "build": ROOT / "build.sh",
    "runner": ROOT / "tools/run_dense_streamk_q4k65_sweep_box.sh",
}

ROW_RE = re.compile(
    r"^\s*X\((\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),B\)\s*\\?\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Census:
    source: int
    eligible: int
    filtered: int
    threads64: int
    threads128: int
    stage2: int
    stage3: int
    stage4: int
    stage6: int
    stage8: int


EXPECTED = Census(
    source=1772,
    eligible=665,
    filtered=1107,
    threads64=293,
    threads128=372,
    stage2=150,
    stage3=149,
    stage4=137,
    stage6=125,
    stage8=104,
)


def parse_rows(text: str) -> list[tuple[int, int, int, int, int, int, int]]:
    return [tuple(map(int, match.groups())) for match in ROW_RE.finditer(text)]


def cta_threads(row: tuple[int, ...]) -> int:
    tm, tn, _tk, wm, wn, _st, _bc = row
    if tm % wm != 0 or tn % wn != 0:
        raise ValueError(f"non-integral warp topology in committed row {row}")
    return 32 * (tm // wm) * (tn // wn)


def eligible_rows(rows: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    return [
        row for row in rows
        if cta_threads(row) in (64, 128) and row[5] - 1 <= 8
    ]


def census(rows: list[tuple[int, ...]]) -> Census:
    admitted = eligible_rows(rows)
    threads = Counter(cta_threads(row) for row in admitted)
    stages = Counter(row[5] for row in admitted)
    return Census(
        source=len(rows),
        eligible=len(admitted),
        filtered=len(rows) - len(admitted),
        threads64=threads[64],
        threads128=threads[128],
        stage2=stages[2],
        stage3=stages[3],
        stage4=stages[4],
        stage6=stages[6],
        stage8=stages[8],
    )


def unique_section(
    text: str, begin: str, end: str, owner: str, bad: list[str]
) -> str:
    if text.count(begin) != 1:
        bad.append(
            f"{owner}: expected one {begin!r}, got {text.count(begin)}"
        )
        return ""
    start = text.index(begin)
    try:
        stop = text.index(end, start + len(begin))
    except ValueError:
        bad.append(f"{owner}: cannot find following end marker {end!r}")
        return ""
    if stop <= start:
        bad.append(f"{owner}: end marker precedes begin marker")
        return ""
    return text[start:stop]


def require_once(text: str, token: str, owner: str, bad: list[str]) -> None:
    count = text.count(token)
    if count != 1:
        bad.append(f"{owner}: expected one {token!r}, found {count}")


def require(text: str, token: str, owner: str, bad: list[str]) -> None:
    if token not in text:
        bad.append(f"{owner}: missing {token!r}")


def forbid(text: str, token: str, owner: str, bad: list[str]) -> None:
    if token in text:
        bad.append(f"{owner}: forbidden {token!r}")


def audit(files: dict[str, str]) -> list[str]:
    bad: list[str] = []
    rows = parse_rows(files["table"])
    if len(set(rows)) != len(rows):
        bad.append("authority: committed dense table contains duplicate rows")
    try:
        got = census(rows)
    except ValueError as error:
        bad.append(f"authority: {error}")
        got = None
    if got != EXPECTED:
        bad.append(f"authority census: got {got}, expected {EXPECTED}")
    admitted = eligible_rows(rows) if got is not None else []
    if any(row[6] != 0 for row in admitted):
        bad.append("authority census: a Stream-K-admitted int4 row unexpectedly uses B-chunk")

    cmake = unique_section(
        files["cmake"],
        "# Full committed-table dense Stream-K sweep.",
        "# Full committed-table Marlin sweep.",
        "Stream-K CMake block",
        bad,
    )
    for token in (
        "qz_resolve_sources(_DENSE_STREAMK_SWEEP_TABLE lowbit_dense_configs.inc)",
        "qz_parse_tactic_xmacro(_DENSE_STREAMK_SWEEP_SOURCE_TABLE_ROWS",
        "LIST_MACRO LOWBIT_DENSE_CFG_LIST",
        "COUNT_MACRO LOWBIT_DENSE_CFG_ROWS",
        'math(EXPR _DENSE_STREAMK_CTA_THREADS\n       "${_DENSE_STREAMK_M_WARPS} * ${_DENSE_STREAMK_N_WARPS} * 32")',
        "(_DENSE_STREAMK_CTA_THREADS EQUAL 64)",
        "(_DENSE_STREAMK_CTA_THREADS EQUAL 128)",
        'math(EXPR _DENSE_STREAMK_STARTUP_STAGES "${_st} - 1")',
        "(_DENSE_STREAMK_STARTUP_STAGES LESS_EQUAL 8)",
        '"${_LOWBIT_DENSE_STREAMK_SWEEP_SOURCE_ROWS} - ${_LOWBIT_DENSE_STREAMK_SWEEP_ELIGIBLE_ROWS}"',
        "dense Stream-K sweep filtered every row",
        "LOWBIT_DENSE_STREAMK_SWEEP_SOURCE_ROWS",
        "LOWBIT_DENSE_STREAMK_SWEEP_ROWS",
        "LOWBIT_DENSE_STREAMK_SWEEP_FILTERED_ROWS",
        "LOWBIT_DENSE_STREAMK_SWEEP_CONFIGS(X,A)",
        "lowbit_dense_streamk_sweep_units",
        "lowbit_dense_streamk_sweep_unit_bc",
        "test_lowbit_dense_streamk_sweep_main.cu",
        "test_lowbit_dense_streamk_sweep\n  \"${_DENSE_STREAMK_SWEEP_MAIN}\"",
        "-DDENSE_STREAMK_SWEEP=1 -DBENCH_GS=32",
        "DENSE_STREAMK_SWEEP=1 BENCH_GS=32",
        "scheduler=streamk wrappers",
        "dense Stream-K generator emitted ${_DENSE_STREAMK_SWEEP_GENERATED_ROWS} wrappers",
        "source_rows=${_LOWBIT_DENSE_STREAMK_SWEEP_SOURCE_ROWS}",
        "eligible_rows=${_LOWBIT_DENSE_STREAMK_SWEEP_ELIGIBLE_ROWS}",
        "filtered_rows=${_LOWBIT_DENSE_STREAMK_SWEEP_FILTERED_ROWS}",
    ):
        require(cmake, token, "Stream-K CMake target", bad)
    for token in (
        "_LOWBIT_DENSE_UNIT_SRCS",
        "lowbit_dense_streamk_ab_unit.cu",
        "test_lowbit_dense_streamk_ab_main.cu",
        "DENSE_STREAMK_Q4K65_AB=1",
        "-DDENSE_STREAMK_AB=1",
    ):
        forbid(cmake, token, "private Stream-K sweep target", bad)
    require_once(
        cmake,
        "quactlize_ppu_executable(\n  test_lowbit_dense_streamk_sweep",
        "Stream-K CMake target",
        bad,
    )

    bench = files["bench"]
    registry = unique_section(
        bench,
        "#elif defined(DENSE_STREAMK_SWEEP)\n// A private registry",
        "#elif defined(DENSE_MARLIN_SWEEP)",
        "Stream-K benchmark registry",
        bad,
    )
    for token in (
        '#include "lowbit_dense_configs.inc"',
        '#include "lowbit_dense_streamk_sweep_configs.inc"',
        '"scheduler=streamk;source=lowbit_dense_configs.inc"',
        "LOWBIT_DENSE_STREAMK_SWEEP_ROWS",
        "LOWBIT_DENSE_STREAMK_SWEEP_CONFIGS",
    ):
        require(registry, token, "private Stream-K benchmark registry", bad)
    forbid(registry, "LOWBIT_DENSE_CFG_LIST", "filtered Stream-K registry", bad)
    provenance = unique_section(
        bench,
        "#elif defined(DENSE_STREAMK_SWEEP)\n  static_assert",
        "#elif defined(DENSE_MARLIN_SWEEP)",
        "Stream-K provenance printer",
        bad,
    )
    for token in (
        '"[dense-table] scheduler=streamk file=%s rows=%d source_rows=%d "',
        '"eligible_rows=%d filtered_rows=%d gs=32 "',
        '"cohort_capability=exact-cta-threads-64-or-128 "',
        '"startup_capability=Stages-1<=8 source_space_fnv1a64=%s "',
        "LOWBIT_DENSE_STREAMK_SWEEP_FILTERED_ROWS +",
        "LOWBIT_DENSE_STREAMK_SWEEP_ROWS ==",
        "LOWBIT_DENSE_STREAMK_SWEEP_SOURCE_ROWS",
    ):
        require(provenance, token, "Stream-K benchmark provenance", bad)
    forbid(provenance, "scheduler=non-persistent", "Stream-K provenance", bad)
    for token in (
        "#if defined(DENSE_STREAMK_AB) || defined(DENSE_STREAMK_SWEEP)",
        "#define DENSE_STREAMK_INSTRUMENTED 1",
        "defined(DENSE_MARLIN_STANDALONE_SWEEP) || defined(DENSE_STREAMK_SWEEP)",
        "#if defined(DENSE_STREAMK_Q4K65_AB) || defined(DENSE_STREAMK_SWEEP)",
        "fixture=streamk-sweep-gs32-exact",
    ):
        require(bench, token, "Stream-K semantic instrumentation", bad)

    unit = files["unit"]
    wrapper = unique_section(
        unit,
        "#if defined(DENSE_STREAMK_SWEEP)",
        "#elif defined(DENSE_MARLIN_SWEEP)",
        "direct Stream-K unit wrapper",
        bad,
    )
    for token in (
        "using G = typename Cfg<GroupSize, TM, TN, TK, WM, WN, ST>::StreamKGemm;",
        'return run<G>(options, dense_tactic(cfg), "streamk");',
    ):
        require(wrapper, token, "direct Stream-K unit wrapper", bad)
    for token in (
        "options.streamk",
        "::Gemm;",
        '"non-persistent"',
        "::PersistentGemm",
        "::MarlinGemm",
    ):
        forbid(wrapper, token, "direct Stream-K unit wrapper", bad)

    build = files["build"]
    target_guard = '[ "$TARGET" = "test_lowbit_dense_streamk_sweep" ]'
    if build.count(target_guard) != 2:
        bad.append(
            "build target must appear once in the dense-table allowlist and "
            "once in its fail-before-hgcc contract preflight"
        )
    require_once(
        build,
        'if [ "$TARGET" = "test_lowbit_dense_streamk_sweep" ]; then',
        "Stream-K sweep contract preflight guard",
        bad,
    )
    require_once(
        build,
        'python3 "$HERE/ci/check_dense_streamk_sweep_target.py" || exit 1',
        "fail-before-hgcc Stream-K sweep preflight",
        bad,
    )

    runner = files["runner"]
    for token in (
        "target=test_lowbit_dense_streamk_sweep",
        "scheduler=streamk",
        "source_rows=1772",
        "eligible_rows=665",
        "filtered_rows=1107",
        "threads=64:293,128:372",
        "stages=2:150,3:149,4:137,6:125,8:104",
        "--streamk_exact_fixture",
        "--m=2048 --n=4096 --k=4096 --l=1 --g=32 --mode=1",
        "--alpha=1 --beta=0",
    ):
        require(runner, token, "box sweep runner", bad)
    for token in ("/tmp/", "mktemp", "probe_box_identity"):
        forbid(runner, token, "workspace-only box sweep runner", bad)

    return bad


def mutate_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"negative control {label}: expected one mutation anchor {old!r}, found {count}"
        )
    return text.replace(old, new, 1)


def negative_controls(files: dict[str, str]) -> list[str]:
    failures: list[str] = []
    cases: list[tuple[str, str, str, str]] = [
        (
            "wrong CTA cohort",
            "cmake",
            "(_DENSE_STREAMK_CTA_THREADS EQUAL 128)",
            "(_DENSE_STREAMK_CTA_THREADS EQUAL 256)",
        ),
        (
            "permit stage-12",
            "cmake",
            "(_DENSE_STREAMK_STARTUP_STAGES LESS_EQUAL 8)",
            "(_DENSE_STREAMK_STARTUP_STAGES LESS_EQUAL 11)",
        ),
        (
            "lose filtered denominator",
            "bench",
            '"eligible_rows=%d filtered_rows=%d gs=32 "',
            '"eligible_rows=%d gs=32 "',
        ),
        (
            "relabel scheduler",
            "bench",
            '"[dense-table] scheduler=streamk file=%s rows=%d source_rows=%d "',
            '"[dense-table] scheduler=non-persistent file=%s rows=%d source_rows=%d "',
        ),
    ]
    for label, owner, old, new in cases:
        changed = dict(files)
        try:
            changed[owner] = mutate_once(files[owner], old, new, label)
        except RuntimeError as error:
            failures.append(str(error))
            continue
        if not audit(changed):
            failures.append(f"negative control {label}: same-source mutation stayed green")

    # The AB runtime branch intentionally names StreamKGemm too.  Mutate only
    # the sweep's compile-time branch, so this negative control proves that a
    # surviving unrelated token cannot make the direct-wrapper check green.
    section_bad: list[str] = []
    wrapper = unique_section(
        files["unit"],
        "#if defined(DENSE_STREAMK_SWEEP)",
        "#elif defined(DENSE_MARLIN_SWEEP)",
        "direct Stream-K unit wrapper negative control",
        section_bad,
    )
    if section_bad:
        failures.extend(section_bad)
    else:
        old = "using G = typename Cfg<GroupSize, TM, TN, TK, WM, WN, ST>::StreamKGemm;"
        new = "using G = typename Cfg<GroupSize, TM, TN, TK, WM, WN, ST>::Gemm;"
        if wrapper.count(old) != 1:
            failures.append(
                "negative control remove direct StreamKGemm wrapper: "
                f"expected one branch-local anchor, found {wrapper.count(old)}"
            )
        else:
            changed = dict(files)
            changed_wrapper = wrapper.replace(old, new, 1)
            changed["unit"] = files["unit"].replace(wrapper, changed_wrapper, 1)
            if not audit(changed):
                failures.append(
                    "negative control remove direct StreamKGemm wrapper: "
                    "same-source mutation stayed green"
                )
    return failures


def main() -> int:
    missing = [str(path) for path in PATHS.values() if not path.is_file()]
    if missing:
        for path in missing:
            print(f"[dense-streamk-sweep-contract] missing: {path}")
        return 1
    files = {name: path.read_text() for name, path in PATHS.items()}
    bad = audit(files)
    if not bad:
        bad.extend(negative_controls(files))
    if bad:
        for issue in bad:
            print(f"[dense-streamk-sweep-contract] FAIL: {issue}")
        return 1
    got = census(parse_rows(files["table"]))
    print(
        "[dense-streamk-sweep-contract] PASS: "
        f"source={got.source} eligible={got.eligible} filtered={got.filtered} "
        f"threads=64:{got.threads64}/128:{got.threads128} "
        "stages=2:150/3:149/4:137/6:125/8:104; "
        "private direct-StreamK target + provenance + runner bound; "
        "5/5 same-source negative controls red"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
