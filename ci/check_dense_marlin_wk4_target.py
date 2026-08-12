#!/usr/bin/env python3
"""Contract for the isolated classic-aligned dense Marlin target.

This check owns no performance claim.  It pins the build/CLI/artifact seams
that can be falsified locally before the PPU run: the old 4N x 1K target stays
unchanged, the new target is exactly 1M x 2N x 4K, its generated wrapper can
only instantiate Marlin, and its bytes come from the explicit WK4 artifact
API rather than the shipping WK1 writer.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CMAKE = ROOT / "quactlize/csrc/CMakeLists.txt.in"
UNIT = ROOT / "benchmarks/lowbit_dense_unit.inc"
BENCH = ROOT / "benchmarks/test_lowbit_dense_bench.cu"
BUILD = ROOT / "build.sh"
BOX = ROOT / "tools/run_dense_marlin_wk4_box.sh"


def exact(text: str, token: str, count: int, bad: list[str], label: str) -> None:
    got = text.count(token)
    if got != count:
        bad.append(f"{label}: expected {count} occurrence(s) of {token!r}, got {got}")


def audit(files: dict[str, str]) -> list[str]:
    cm, unit, bench, build, box = (
        files[name] for name in ("cmake", "unit", "bench", "build", "box")
    )
    bad: list[str] = []

    # The historical target is the measured 4N x 1K/stages-3 control.  A new
    # target is evidence only if the control did not silently move with it.
    old_begin = cm.find("set(_DENSE_MARLIN_ARTIFACT_TK 64)")
    old_end = cm.find("# Classic-aligned Marlin decode target.")
    if old_begin < 0 or old_end <= old_begin:
        bad.append("cannot isolate the historical Marlin target block")
        old = ""
    else:
        old = cm[old_begin:old_end]
    for token in (
        "set(_DENSE_MARLIN_WN 32)",
        "set(_DENSE_MARLIN_ST 3)",
        "test_lowbit_dense_marlin_ab",
        "DENSE_MARLIN_AB=1 DENSE_STREAMK_AB=1",
    ):
        if token not in old:
            bad.append(f"historical target drifted: missing {token!r}")
    if "DENSE_AB_WARP_K" in old or "place_derived_warp_k" in old:
        bad.append("historical target acquired the WK4 artifact/type axis")

    new_begin = cm.find("# Classic-aligned Marlin decode target.")
    new_end = cm.find("# Root-cause cross-check", new_begin)
    if new_begin < 0 or new_end <= new_begin:
        bad.append("cannot isolate the classic-aligned target block")
        new = ""
    else:
        new = cm[new_begin:new_end]
    for token in (
        "set(_DENSE_MARLIN_WK4_ARTIFACT_TK 64)",
        "set(_DENSE_MARLIN_WK4_TM 16)",
        "set(_DENSE_MARLIN_WK4_TN 128)",
        "set(_DENSE_MARLIN_WK4_TK 128)",
        "set(_DENSE_MARLIN_WK4_WM 16)",
        "set(_DENSE_MARLIN_WK4_WN 64)",
        "set(_DENSE_MARLIN_WK4_WARP_K 32)",
        "set(_DENSE_MARLIN_WK4_ST 4)",
        "lowbit_dense_marlin_wk4_ab_unit.cu",
        "test_lowbit_dense_marlin_wk4_ab_main.cu",
        "test_lowbit_dense_marlin_wk4_ab",
        "DENSE_MARLIN_WK4_AB=1 DENSE_MARLIN_AB=1 DENSE_STREAMK_AB=1",
        "-DDENSE_AB_WARP_K=${_DENSE_MARLIN_WK4_WARP_K}",
        "DENSE_AB_WARP_K=${_DENSE_MARLIN_WK4_WARP_K}",
    ):
        if token not in new:
            bad.append(f"aligned target is missing {token!r}")
    exact(new, "DEV_COMPILE_FLAGS ${_DENSE_MARLIN_WK4_DEFS}", 1, bad,
          "aligned device definitions")
    exact(new, "target_compile_definitions(test_lowbit_dense_marlin_wk4_ab PRIVATE", 1,
          bad, "aligned host definitions")

    for token in (
        "#if defined(DENSE_MARLIN_WK4_AB)",
        "using G = typename Cfg<GroupSize, TM, TN, TK, WM, WN, ST,\n"
        "                         DENSE_AB_WARP_K>::MarlinGemm;",
        "Kernel::WarpKCohorts == 4",
        "Kernel::MaxThreadsPerBlock == 256",
        "Kernel::OutputThreads == 64",
        "Kernel::TileScheduler::FixupThreadCount == 64",
        "return run<G>(options, dense_tactic(cfg), \"marlin\");",
        "#define LOWBIT_DENSE_CFG_WARP_K , DENSE_AB_WARP_K",
    ):
        if token not in unit:
            bad.append(f"generated aligned wrapper is missing {token!r}")
    aligned_arm = unit.split("#if defined(DENSE_MARLIN_WK4_AB)", 1)[1].split(
        "#else", 1
    )[0]
    for forbidden in ("StreamKGemm", "PersistentGemm", "::Gemm;"):
        if forbidden in aligned_arm:
            bad.append(f"aligned wrapper can silently instantiate {forbidden!r}")

    for token in (
        '"marlin-wk4-aligned-single-row"',
        "topology=1Mx2Nx4K",
        "cta_threads=256 output_cohort_threads=64",
        "warp_k_extent=32 warp_k_cohorts=4",
        "artifact_tile_k=64 artifact_axis=WarpK32",
        "xplane::place_derived_warp_k<4, 16, 128, 128, 16, 64, 1,",
        "32, 64>(dst, q, row, col);",
        "xplane::recover_derived_warp_k<4, 16, 128, 128, 16, 64, 1,",
        "roundtrip_bad=%zu/%zu",
        "if (!options.marlin || options.persistent || options.streamk ||",
        "test_lowbit_dense_marlin_wk4_ab is Marlin-only: pass --marlin",
        "test_lowbit_dense_marlin_wk4_ab requires --streamk_exact_fixture",
        "DP, persistent and Stream-K arms are not valid for a 4K-cohort CTA",
    ):
        if token not in bench:
            bad.append(f"aligned host route is missing {token!r}")
    exact(bench, "#if defined(DENSE_MARLIN_WK4_AB)\n  // Unlike the historical", 1,
          bad, "aligned result propagation")
    final_arm = bench.split(
        "#if defined(DENSE_MARLIN_WK4_AB)\n  // Unlike the historical", 1
    )[-1].split("#elif defined(DENSE_STREAMK_AB)", 1)[0]
    if "return final_result.passed ? 0 : 1;" not in final_arm:
        bad.append("aligned result propagation ignores the measured result")

    exact(build, '[ "$TARGET" = "test_lowbit_dense_marlin_wk4_ab" ]', 1,
          bad, "build route")
    for token in (
        "TARGET=test_lowbit_dense_marlin_wk4_ab",
        "COMMON=(--marlin --streamk_exact_fixture",
        "placement=WK4 artifact_tile_k=64 roundtrip_bad=0",
        "block_threads=256 warps/cta=8",
        "Gemm::maximum_active_blocks()",
        'if [ "$bpc" -gt "$CAP" ]',
        "NOT RUN: B=%d exceeds Gemm::maximum_active_blocks()=%d",
        "ILLEGAL=$((CAP + 1))",
        "produced $repeats/8 stable lock fingerprints",
    ):
        if token not in box:
            bad.append(f"box recipe is missing {token!r}")
    return bad


def main() -> int:
    files = {
        "cmake": CMAKE.read_text(),
        "unit": UNIT.read_text(),
        "bench": BENCH.read_text(),
        "build": BUILD.read_text(),
        "box": BOX.read_text(),
    }
    bad = audit(files)

    plants = (
        ("cmake", "aligned-WN-back-to-32", "set(_DENSE_MARLIN_WK4_WN 64)",
         "set(_DENSE_MARLIN_WK4_WN 32)"),
        ("cmake", "aligned-drops-WarpK-define",
         "-DDENSE_AB_WARP_K=${_DENSE_MARLIN_WK4_WARP_K}",
         "-DDENSE_AB_WARP_K=${_DENSE_MARLIN_WK4_TK}"),
        ("cmake", "old-target-mutated", "set(_DENSE_MARLIN_WN 32)",
         "set(_DENSE_MARLIN_WN 64)"),
        ("unit", "aligned-wrapper-launches-DP",
         "DENSE_AB_WARP_K>::MarlinGemm;",
         "DENSE_AB_WARP_K>::Gemm;"),
        ("bench", "aligned-reuses-WK1-writer", "xplane::place_derived_warp_k<",
         "xplane::place_derived<"),
        ("bench", "CLI-does-not-require-marlin", "if (!options.marlin || options.persistent",
         "if (false || options.persistent"),
        ("bench", "result-is-ignored",
         "// golden, occupancy check or 8-launch lock fingerprint is the process rc.\n"
         "  return final_result.passed ? 0 : 1;",
         "// golden, occupancy check or 8-launch lock fingerprint is the process rc.\n"
         "  return 0;"),
        ("build", "build-route-missing",
         '   [ "$TARGET" = "test_lowbit_dense_marlin_wk4_ab" ]; then',
         '   [ "$TARGET" = "test_lowbit_dense_marlin_ab" ]; then'),
        ("box", "over-cap-point-launched", 'if [ "$bpc" -gt "$CAP" ]; then',
         'if [ "$bpc" -gt 999 ]; then'),
    )
    for owner, label, old, new in plants:
        if files[owner].count(old) != 1:
            bad.append(f"cannot plant {label}: anchor missing or duplicated")
            continue
        planted = dict(files)
        planted[owner] = planted[owner].replace(old, new)
        if not audit(planted):
            bad.append(f"plant {label} was not rejected")

    if bad:
        for item in bad:
            print(f"[dense-marlin-wk4] FAIL: {item}", file=sys.stderr)
        return 1
    print(
        "[dense-marlin-wk4] PASS: isolated 1Mx2Nx4K type/artifact/CLI; "
        "historical target unchanged; nine structural plants rejected"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
