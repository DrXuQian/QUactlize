#!/usr/bin/env python3
"""Structural contract for the one-launch standalone classic ACU baseline."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "tools/classic_marlin_156_profile.cu"
RUNNER = ROOT / "tools/run_classic_marlin_156_box.sh"


def audit(harness: str, runner: str) -> list[str]:
    bad: list[str] = []
    for token in (
        "constexpr int kM = 1;", "constexpr int kN = 4096;",
        "constexpr int kK = 4096;", "constexpr int kGroupSize = 128;",
        "constexpr int kExpectedMma = 65536;",
        "identity=Marlin<256,1,8,8,4,8>",
        "int const rc = marlin_classic_ppu::marlin_cuda(",
        "MARLIN156 launch-count=1",
        "sync_code=%d sync=%s",
    ):
        if harness.count(token) != 1:
            bad.append(f"harness requires exactly one {token!r}")
    if harness.count("marlin_classic_ppu::marlin_cuda(") != 1:
        bad.append("harness must contain exactly one production launch call")
    for forbidden in ("for (int i", "cudaEventRecord", "bench_one("):
        if forbidden in harness:
            bad.append(f"one-launch harness contains {forbidden!r}")

    for token in (
        "probe_box_identity.py\" resolve", "source.before.sha256",
        "source.after.sha256", "cmp \"$OUT/source.before.sha256\"",
        "-DMARLIN_STAGES=4", "-DMARLIN_MAX_MB=2", "-DMARLIN_MIN_BLOCKS=2",
        "--set full", "--csv --page details", "classic.details.csv",
        "expected_vmma_per_launch", "sha256sum \"$BIN\"",
        "MARLIN156 launch-count=1 rc=0 sync_code=0 sync=",
    ):
        if token not in runner:
            bad.append(f"runner is missing {token!r}")
    if "bench_marlin" in runner:
        bad.append("runner regressed to the seven-launch benchmark")
    return bad


def main() -> int:
    harness = HARNESS.read_text()
    runner = RUNNER.read_text()
    bad = audit(harness, runner)
    plants = (
        ("shape", harness, "constexpr int kM = 1;", "constexpr int kM = 2;", runner),
        ("second-launch", harness,
         "int const rc = marlin_classic_ppu::marlin_cuda(",
         "int const rc = marlin_classic_ppu::marlin_cuda(\n"
         "      a, b, c, scales, kM, kN, kK, locks, kGroupSize, 0, 0, -1, -1, -1, kMaxPar);\n"
         "  marlin_classic_ppu::marlin_cuda(", runner),
        ("acu-full", harness, "", "", runner.replace("--set full", "--set basic")),
    )
    for label, h0, old, new, r0 in plants:
        planted_h = h0 if not old else h0.replace(old, new, 1)
        if not audit(planted_h, r0):
            bad.append(f"plant {label} was not rejected")
    if bad:
        for item in bad:
            print(f"[classic-156] FAIL: {item}", file=sys.stderr)
        return 1
    print("[classic-156] PASS: exact one-launch shape, source/tool/binary identity and full ACU capture are fail-closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
