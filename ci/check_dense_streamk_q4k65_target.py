#!/usr/bin/env python3
"""Fail-closed local contract for the historical Q4_K65 Stream-K A/B.

This is not a numerical substitute for the PPU run.  It proves that the box
runner will compare the historical gs32 row with the *same* Main/Epi types
under normal and forced Stream-K scheduling, while leaving #107b's gs128
mechanism gate untouched.  It also derives the exact fixture's FP32/fp16
bounds from its published construction instead of trusting its log string.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PATHS = {
    "cmake": ROOT / "quactlize/csrc/CMakeLists.txt.in",
    "bench": ROOT / "benchmarks/test_lowbit_dense_bench.cu",
    "build": ROOT / "build.sh",
    "runner": ROOT / "tools/run_dense_streamk_q4k65_box.sh",
}

FP32_EXACT_INT = 1 << 24
FP16_EXACT_INT = 1 << 11


def unique_section(text: str, begin: str, end: str, owner: str,
                   bad: list[str]) -> str:
    if text.count(begin) != 1 or text.count(end) != 1:
        bad.append(f"{owner}: cannot isolate unique {begin!r} .. {end!r}")
        return ""
    start = text.index(begin)
    stop = text.index(end, start + len(begin))
    if stop <= start:
        bad.append(f"{owner}: section end precedes its begin")
        return ""
    return text[start:stop]


def require_once(text: str, token: str, owner: str, bad: list[str]) -> None:
    count = text.count(token)
    if count != 1:
        bad.append(f"{owner}: expected one {token!r}, found {count}")


def ordered_once(text: str, tokens: tuple[str, ...], owner: str,
                 bad: list[str]) -> None:
    counts = [text.count(token) for token in tokens]
    if any(count != 1 for count in counts):
        bad.append(f"{owner}: ordered anchors have counts {counts}, expected all one")
        return
    positions = [text.index(token) for token in tokens]
    if positions != sorted(positions):
        bad.append(f"{owner}: ordered anchors appear out of order")


def parse_cmake_ints(block: str, prefix: str, owner: str,
                     bad: list[str]) -> dict[str, int]:
    expected_names = ("TM", "TN", "TK", "WM", "WN", "ST")
    values: dict[str, int] = {}
    for name in expected_names:
        pattern = rf"set\({re.escape(prefix)}_{name}\s+(\d+)\)"
        found = re.findall(pattern, block)
        if len(found) != 1:
            bad.append(f"{owner}: expected one integer assignment for {prefix}_{name}")
        else:
            values[name] = int(found[0])
    return values


def parse_cpp_int(text: str, name: str, owner: str,
                  bad: list[str]) -> int | None:
    found = re.findall(rf"\b{re.escape(name)}\s*=\s*(\d+)\s*;", text)
    if len(found) != 1:
        bad.append(f"{owner}: expected one definition of {name}, found {len(found)}")
        return None
    return int(found[0])


def parse_cpp_array(text: str, name: str, owner: str,
                    bad: list[str]) -> list[int] | None:
    found = re.findall(
        rf"\b{re.escape(name)}\s*\[\s*\]\s*=\s*\{{([^}}]*)\}}\s*;", text)
    if len(found) != 1:
        bad.append(f"{owner}: expected one definition of {name}, found {len(found)}")
        return None
    values = [int(value) for value in re.findall(r"-?\d+", found[0])]
    if not values:
        bad.append(f"{owner}: {name} is empty")
        return None
    return values


def exact_bound(bench: str, bad: list[str]) -> tuple[int, int] | None:
    owner = "Q4_K65 exact fixture"
    nonzeros = parse_cpp_int(
        bench, "kQ4K65ExactFixtureNonzerosPerRow", owner, bad)
    scales = parse_cpp_array(bench, "kQ4K65ExactFixtureScales", owner, bad)
    zeros = parse_cpp_array(bench, "kExactFixtureZeros", owner, bad)
    if nonzeros is None or scales is None or zeros is None:
        return None

    # QuantType(q) is filled from the complete signed int4 set [-8,7].  The
    # source anchors below separately pin the K/N-varying code construction and
    # A-sign alignment, so this is an independent magnitude derivation rather
    # than a copy of the runtime max_output print.
    max_weight = max(
        abs(q * scale + zero)
        for q in range(-8, 8)
        for scale in scales
        for zero in zeros
    )
    max_output = nonzeros * max_weight
    if max_output > FP32_EXACT_INT:
        bad.append(
            f"{owner}: max|D|={max_output} exceeds FP32 exact integer bound "
            f"{FP32_EXACT_INT}")
    if max_output > FP16_EXACT_INT:
        bad.append(
            f"{owner}: max|D|={max_output} exceeds fp16 exact integer bound "
            f"{FP16_EXACT_INT}")
    if (nonzeros, scales, zeros, max_weight, max_output) != (
            128, [1, 2], [0], 16, 2048):
        bad.append(
            f"{owner}: construction drifted: nonzeros={nonzeros} scales={scales} "
            f"zeros={zeros} max|w|={max_weight} max|D|={max_output}")
    return max_weight, max_output


def audit(files: dict[str, str]) -> list[str]:
    bad: list[str] = []
    cmake = files["cmake"]
    bench = files["bench"]
    build = files["build"]
    runner = files["runner"]

    old = unique_section(
        cmake,
        "# 107b dense Stream-K mechanism gate.",
        "# Historical Q4_K-prefill anchor, isolated from 107b's gs128 mechanism gate.",
        "107b CMake block", bad)
    new = unique_section(
        cmake,
        "# Historical Q4_K-prefill anchor, isolated from 107b's gs128 mechanism gate.",
        "# Independent Marlin CTA-stripe scheduler/cooperative.",
        "Q4_K65 CMake block", bad)

    old_values = parse_cmake_ints(old, "_DENSE_SK", "107b target", bad)
    if old_values and old_values != {
            "TM": 64, "TN": 128, "TK": 64,
            "WM": 64, "WN": 32, "ST": 2}:
        bad.append(f"107b target changed: {old_values}")
    for token in (
        'set(_DENSE_SK_SRC "${CMAKE_CURRENT_BINARY_DIR}/lowbit_dense_streamk_ab_unit.cu")',
        'set(_DENSE_SK_MAIN "${CMAKE_CURRENT_BINARY_DIR}/test_lowbit_dense_streamk_ab_main.cu")',
        "test_lowbit_dense_streamk_ab\n  \"${_DENSE_SK_MAIN}\"\n  \"${_DENSE_SK_SRC}\"",
        "-DDENSE_STREAMK_AB=1 -DBENCH_GS=128",
        "DENSE_STREAMK_AB=1 BENCH_GS=128",
    ):
        require_once(old, token, "unchanged 107b target", bad)
    if "Q4K65" in old or "q4k65" in old:
        bad.append("107b target is contaminated by Q4_K65 wiring")

    new_values = parse_cmake_ints(new, "_DENSE_SK_Q4K65", "Q4_K65 target", bad)
    if new_values and new_values != {
            "TM": 64, "TN": 64, "TK": 64,
            "WM": 64, "WN": 32, "ST": 3}:
        bad.append(f"Q4_K65 target geometry changed: {new_values}")
    for token in (
        '"${CMAKE_CURRENT_BINARY_DIR}/lowbit_dense_streamk_q4k65_ab_unit.cu"',
        '"${CMAKE_CURRENT_BINARY_DIR}/test_lowbit_dense_streamk_q4k65_ab_main.cu"',
        "test_lowbit_dense_streamk_q4k65_ab\n  \"${_DENSE_SK_Q4K65_MAIN}\"\n  \"${_DENSE_SK_Q4K65_SRC}\"",
        "-DDENSE_STREAMK_AB=1 -DDENSE_STREAMK_Q4K65_AB=1 -DBENCH_GS=32",
        "DENSE_STREAMK_AB=1 DENSE_STREAMK_Q4K65_AB=1 BENCH_GS=32",
        '"  X(${_DENSE_SK_Q4K65_FN},${_DENSE_SK_Q4K65_TM},${_DENSE_SK_Q4K65_TN},${_DENSE_SK_Q4K65_TK},${_DENSE_SK_Q4K65_WM},${_DENSE_SK_Q4K65_WN},${_DENSE_SK_Q4K65_ST},0)\\n"',
    ):
        require_once(new, token, "isolated Q4_K65 target", bad)
    for old_path in (
        "lowbit_dense_streamk_ab_unit.cu",
        "test_lowbit_dense_streamk_ab_main.cu",
    ):
        if old_path in new:
            bad.append(f"Q4_K65 target aliases 107b source path {old_path!r}")
    if build.count('[ "$TARGET" = "test_lowbit_dense_streamk_q4k65_ab" ]') != 1:
        bad.append("build.sh does not admit exactly one Q4_K65 target")

    # Both adapters must be aliases over the same Cfg::Main and Cfg::Epi.  A
    # second collective would make a scheduler A/B incomparable even if the
    # runtime labels looked right.
    for token in (
        "using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>;",
        "using StreamKKernel = cutlass::gemm::kernel::StreamKMixedInputKernel<\n"
        "      Shape<int,int,int,int>, Main, Epi>;",
        "#if defined(DENSE_STREAMK_Q4K65_AB)\n"
        "#define LOWBIT_DENSE_TABLE_FILE                 \"streamk-q4k65-ab-single-row\"",
        'fixture=q4k65-exact shape=%dx%dx%d ',
        "#if defined(DENSE_STREAMK_Q4K65_AB)\n"
        "       options.m != 2048 || options.n != 4096 ||",
        "#if defined(DENSE_STREAMK_Q4K65_AB)\n"
        "       options.g != 32 || options.alpha != 1.0f || options.beta != 0.0f",
        "int const sign = ((k >> 3) & 1) ? -1 : 1;",
        "int const code = (5 * k + 3 * n) & 7;",
        "int const q = ((k >> 3) & 1) ? (-8 + code) : code;",
        "max_output <= std::ldexp(1.0, 24)",
        "max_output <= std::ldexp(1.0, 11)",
    ):
        require_once(bench, token, "Q4_K65 benchmark contract", bad)
    bound = exact_bound(bench, bad)
    if bound is not None and bound != (16, 2048):
        bad.append(f"Q4_K65 exact bound is {bound}, expected (16, 2048)")

    for token in (
        "TARGET=test_lowbit_dense_streamk_q4k65_ab",
        "QUANT=int4 BENCH_GS=32",
        "CONFIG='64x64x64:64x32:s3:bc0->0'",
        "CONFIG_LABEL='64x64:64 w64x32 s3 bc0->0'",
        "--m=2048 --n=4096 --k=4096 --l=1 --g=32 --mode=1",
        "--alpha=1 --beta=0 --config=\"$CONFIG\"",
        "tile 64x64x64  warp 64x32  stages 3  instruction=m16$",
        "fixture=q4k65-exact ",
        "max\\|D\\|=2048 ",
        "scheduler=non-persistent\\] logical_cta=(\\d+) cu=(\\d+) ",
        "if cu != 72:",
        "expected historical 72-CU box",
        "actual=(\\w+) real_cu=(\\d+)",
        "normal_workers=",
        "Disposition: Passed (whole-K reference bit-exact; fixup replay closed)",
    ):
        require_once(runner, token, "Q4_K65 box runner", bad)
    ordered_once(runner, (
        "== historical normal-scheduler admission ==",
        '"$BIN" "${COMMON[@]}" --iterations=100 2>&1 | tee "$BASELINE_LOG"',
        "== exact normal control ==",
        '"$BIN" "${COMMON[@]}" --iterations=20 --streamk_exact_fixture \\\n'
        '    2>&1 | tee "$NORMAL_LOG"',
        "== exact forced hybrid Stream-K subject ==",
        '"$BIN" "${COMMON[@]}" --iterations=20 --streamk_exact_fixture --streamk \\\n'
        '    2>&1 | tee "$STREAMK_LOG"',
        "Q4K65_VERDICT ",
    ), "normal-before-Stream-K admission", bad)
    if "/tmp/" in runner or "mktemp" in runner:
        bad.append("Q4_K65 runner writes outside its explicit /workspace bundle")

    return bad


def self_test(files: dict[str, str]) -> list[str]:
    """Same-source negative controls: vary only the property under test."""
    failures: list[str] = []

    def must_fail(label: str, key: str, old: str, new: str) -> None:
        mutated = dict(files)
        if mutated[key].count(old) != 1:
            failures.append(f"negative control {label}: mutation anchor is not unique")
            return
        mutated[key] = mutated[key].replace(old, new, 1)
        if not audit(mutated):
            failures.append(f"negative control {label}: planted defect was accepted")

    must_fail(
        "107b geometry drift", "cmake",
        "set(_DENSE_SK_TN 128)", "set(_DENSE_SK_TN 64)")
    must_fail(
        "Q4_K65 aliases 107b main TU", "cmake",
        "test_lowbit_dense_streamk_q4k65_ab_main.cu",
        "test_lowbit_dense_streamk_ab_main.cu")
    must_fail(
        "Stream-K runs before normal admission", "runner",
        "== historical normal-scheduler admission ==",
        "== zzz historical normal-scheduler admission ==")
    must_fail(
        "fp16 exact bound exceeded", "bench",
        "kQ4K65ExactFixtureScales[] = {1, 2};",
        "kQ4K65ExactFixtureScales[] = {1, 2, 4};")
    must_fail(
        "cross-device baseline admitted", "runner",
        "if cu != 72:", "if cu != 32:")
    must_fail(
        "space-bearing report tag used as selector", "runner",
        "CONFIG='64x64x64:64x32:s3:bc0->0'",
        "CONFIG='64x64:64 w64x32 s3 bc0->0'")
    return failures


def main() -> int:
    missing = [str(path) for path in PATHS.values() if not path.is_file()]
    if missing:
        print("[q4k65-streamk-contract] FAIL: missing " + ", ".join(missing),
              file=sys.stderr)
        return 1
    files = {name: path.read_text() for name, path in PATHS.items()}
    bad = audit(files)
    if not bad:
        bad.extend(self_test(files))
    if bad:
        for problem in bad:
            print(f"[q4k65-streamk-contract] FAIL: {problem}", file=sys.stderr)
        return 1
    _, max_output = exact_bound(files["bench"], [])  # audited above
    print(
        "[q4k65-streamk-contract] PASS: 107b unchanged; isolated gs32 "
        "64x64x64/w64x32/s3 m16 target shares Main/Epi; normal admission "
        f"precedes forced Stream-K; exact max|D|={max_output} <= 2^11; "
        "six same-source negative controls red")
    return 0


if __name__ == "__main__":
    sys.exit(main())
