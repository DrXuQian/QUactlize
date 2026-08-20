#!/usr/bin/env python3
"""Bind the Q4/A32 component fixtures before any PPU value is classified.

The checker derives output zero from the pinned sparse fixture algebra.  It
does not execute the collective and therefore cannot turn a host proof into a
device pass.  Its job is narrower: a shell label, host golden, and fixture
construction must not drift into the contradictory state observed at
7dad9ac, where a row labeled metadata-only carried code-only's golden.
"""

from fractions import Fraction
from pathlib import Path
import re
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks/test_scalefirst_internal_sweep.cu"
RUNNER = ROOT / "tools/run_scalefirst_q4_a32_exact_box.sh"
MODES = (
    "transport-only", "code-only", "scale-only", "zero-only",
    "metadata-only", "exact",
)
TAG_MODES = (
    "scale-group-tag", "scale-n-tag", "zero-group-tag", "zero-n-tag",
    "code-k0-tag", "code-k1-tag", "code-k2-tag", "code-k3-tag",
    "code-n0-tag", "code-n1-tag", "code-n2-tag",
)
TAG_ROUNDS = ("even", "odd", "stage")


def half_bits(value: Fraction) -> int:
    return struct.unpack("<H", struct.pack("<e", float(value)))[0]


def first_golden(mode: str) -> int:
    k = 5120
    active = [s * k // 8 + ((37 * s + 11) % (k // 8)) for s in range(8)]
    total = Fraction(0)
    for s, kk in enumerate(active):
        a = Fraction(1, 2) if mode == "transport-only" else (
            Fraction(-1, 2) if s & 1 else Fraction(1, 2))
        code = 1 if mode in {
            "transport-only", "scale-only", "zero-only", "metadata-only"
        } else ((7 * kk + 3) % 15) - 7
        group = kk // 32
        scale = 1 << ((17 * group + 1) % 3) if mode in {
            "scale-only", "metadata-only", "exact"
        } else 1
        zero = (((11 * group) % 3) - 1) * 3 if mode in {
            "zero-only", "metadata-only", "exact"
        } else 0
        total += a * (scale * code + zero)
    return half_bits(total)


def function_modes(source: str, name: str) -> set[str] | None:
    match = re.search(
        rf"constexpr bool {name}\(FixtureMode mode\) \{{(.*?)\n\}}",
        source, re.S)
    if not match:
        return None
    return set(re.findall(r"FixtureMode::([A-Za-z0-9]+)", match.group(1)))


def audit(bench: str, runner: str) -> list[str]:
    bad: list[str] = []
    expected = {mode: first_golden(mode) for mode in MODES}
    if len(expected) != 6:
        bad.append("fixture denominator is not six")

    scale_modes = function_modes(bench, "fixture_uses_varied_scale")
    if scale_modes != {
        "Exact", "ScaleOnly", "MetadataOnly", "ScaleGroupTag", "ScaleNTag"
    }:
        bad.append(f"varied-scale modes drifted: {scale_modes}")
    zero_modes = function_modes(bench, "fixture_uses_varied_zero")
    if zero_modes != {
        "Exact", "ZeroOnly", "MetadataOnly", "ZeroGroupTag", "ZeroNTag"
    }:
        bad.append(f"varied-zero modes drifted: {zero_modes}")
    constant_modes = function_modes(bench, "fixture_uses_constant_code")
    if constant_modes != {"Exact", "CodeOnly"} or \
            "mode != FixtureMode::Exact && mode != FixtureMode::CodeOnly" not in bench or \
            "!is_code_tag(mode)" not in bench:
        bad.append("constant-code complement drifted")

    for token in (
        "a_fnv=", "low_native_fnv=", "low_placed_fnv=", "scale_fnv=",
        "zero_fnv=", "golden_fnv=", "probe_fnv=", "tag_round=%s",
        "first_golden=0x%04x",
    ):
        if token not in bench:
            bad.append(f"prelaunch fixture binding lost {token}")
    if "mode == FixtureMode::TransportOnly" not in bench:
        bad.append("transport-only did not select its non-cancelling A fixture")
    if "if (cli.fixture_binding)" not in bench or \
            "--fixture-binding" not in runner:
        bad.append("diagnostic runner no longer opts into prelaunch binding")

    found = {
        mode: int(bits, 16)
        for mode, bits in re.findall(
            r"^\s+([a-z-]+)\) expected=(0x[0-9a-f]+) ;;$", runner, re.M)
    }
    if set(found) != set(MODES):
        bad.append(f"runner fixture denominator drifted: {sorted(found)}")
    for mode, want in expected.items():
        if found.get(mode) != want:
            bad.append(
                f"{mode} first golden is {found.get(mode)!r}, derived 0x{want:04x}")
    if "marker_count" not in runner or "[ \"$marker_count\" -eq 1 ]" not in runner:
        bad.append("runner no longer requires exactly one prelaunch identity marker")
    if "Q4_A32_COMPONENTS transport_only=" not in runner:
        bad.append("six-arm component verdict is not emitted")
    tag_specs = set(re.findall(
        r"\b(scale-(?:group|n)-tag|zero-(?:group|n)-tag|"
        r"code-[kn][0-3]-tag):(even|odd|stage)\b",
        runner,
    ))
    expected_tag_specs = {
        (mode, round_) for mode in TAG_MODES for round_ in TAG_ROUNDS
    }
    if tag_specs != expected_tag_specs:
        bad.append(
            "coordinate-tag runner denominator drifted "
            f"missing={sorted(expected_tag_specs - tag_specs)} "
            f"extra={sorted(tag_specs - expected_tag_specs)}"
        )
    for token in (
        "return (k >> 0) & 15", "return (k >> 4) & 15",
        "return (k >> 8) & 15", "return (k >> 12) & 15",
        "return (n >> 0) & 15", "return (n >> 4) & 15",
        "return (n >> 8) & 15", "scale = float(g + 1)",
        "scale = float(n + 1)", "zero = float(g + 1)",
        "zero = float(n + 1)", "SF_TAG_ROWS mode=%s",
        "SF_TAG_COLS mode=%s",
    ):
        if token not in bench:
            bad.append(f"coordinate-tag fixture lost {token}")
    return bad


def main() -> int:
    bench = BENCH.read_text()
    runner = RUNNER.read_text()
    bad = audit(bench, runner)
    if bad:
        print("[q4-a32-fixtures] FAIL: " + "; ".join(bad), file=sys.stderr)
        return 1

    plants = (
        (bench, runner.replace(
            "metadata-only) expected=0xc100", "metadata-only) expected=0x4000", 1),
         "code-golden metadata"),
        (bench.replace(
            " || mode == FixtureMode::ScaleOnly", "", 1), runner,
         "missing scale-only component"),
        (bench.replace(" zero_fnv=%016llx", "", 1), runner,
         "missing zero fingerprint"),
        (bench, runner.replace("code-k3-tag:stage", "", 1),
         "missing coordinate-tag denominator"),
        (bench.replace("return (k >> 12) & 15", "return (k >> 11) & 15", 1),
         runner, "wrong coordinate-tag bit"),
    )
    for planted_bench, planted_runner, label in plants:
        if not audit(planted_bench, planted_runner):
            print(f"[q4-a32-fixtures] FAIL: {label} plant was false-green",
                  file=sys.stderr)
            return 1

    rendered = ", ".join(
        f"{mode}=0x{first_golden(mode):04x}" for mode in MODES)
    print("[q4-a32-fixtures] PASS: " + rendered +
          "; wrong-golden/missing-component/missing-fingerprint/"
          "missing-tag/wrong-tag-bit plants red")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
