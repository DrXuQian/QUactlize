#!/usr/bin/env python3
"""Contract for the production *standalone* Marlin tactic sweep.

``DENSE_MARLIN_SWEEP`` is the retired generic mixed-input collective and is
not evidence for ``MarlinCollectivePPU``.  This gate therefore binds the
independent Cartesian authority, its generated eight-field row ABI, the
per-row standalone wrappers, and a separately named production target.  A
historical generic-Marlin target becoming green cannot satisfy this contract.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
AUTHORITY = ROOT / "quactlize/include/marlin_tactic_space_ppu.hpp"
EMITTER = ROOT / "dev/fold_derivation/emit_marlin_tactic_space.cpp"
GENERATED_HEADER = ROOT / "benchmarks/marlin_standalone_configs.inc"
RUNNER = ROOT / "dev/fold_derivation/run_l172_standalone_marlin_tactic_space.sh"
STAGE_RUNNER = ROOT / "dev/fold_derivation/run_l183_marlin_stage_ring.sh"
BENCH = ROOT / "benchmarks/test_lowbit_dense_bench.cu"
UNIT = ROOT / "benchmarks/lowbit_dense_unit.inc"
TACTIC_PARSER = ROOT / "quactlize/csrc/TacticTableUnits.cmake"
CMAKE = ROOT / "quactlize/csrc/CMakeLists.txt.in"
BUILD = ROOT / "build.sh"
BOX_RUNNER = ROOT / "tools/run_dense_marlin_standalone_sweep_box.sh"
TARGET = "test_lowbit_dense_marlin_standalone_sweep"


def target_block(source: str) -> str:
    """Return this target's CMake block, stopping at the next executable."""
    section = "# Independent standalone-Marlin tactic sweep."
    start = source.find(section)
    if start >= 0:
        end = source.find("\n# Root-cause cross-check", start + len(section))
        return source[start:] if end < 0 else source[start:end]
    marker = f"quactlize_ppu_executable(\n  {TARGET}"
    start = source.find(marker)
    if start < 0:
        marker = f"quactlize_ppu_executable({TARGET}"
        start = source.find(marker)
    if start < 0:
        return ""
    end = source.find("\nquactlize_ppu_executable(", start + len(marker))
    return source[start:] if end < 0 else source[start:end]


def main() -> int:
    missing = [str(path.relative_to(ROOT))
               for path in (AUTHORITY, EMITTER, GENERATED_HEADER, RUNNER,
                            STAGE_RUNNER,
                            BENCH, UNIT, TACTIC_PARSER, CMAKE, BUILD,
                            BOX_RUNNER)
               if not path.is_file()]
    if missing:
        print("[dense-marlin-sweep-contract] FAIL: missing " + ", ".join(missing))
        return 1

    source = AUTHORITY.read_text()
    bad: list[str] = []
    for token in (
        "struct MarlinTacticPPU", "kMarlinTileM", "kMarlinTileN",
        "kMarlinTileK", "kMarlinWarpM", "kMarlinWarpN", "kMarlinWarpK",
        "kMarlinStages", "kMarlinLoadKinds", "for_each_declared",
        "is_classic_subspace", "MarlinTacticExclusionPPU",
        "CurrentImplementation", "cartesian_size() == 60000",
    ):
        if token not in source:
            bad.append(f"standalone authority lacks {token!r}")
    for forbidden in ('#include "ppu_tactic_space.hpp"', "struct Candidate",
                      "DENSE_MARLIN_SWEEP"):
        if forbidden in source:
            bad.append(f"standalone authority imports generic seam {forbidden!r}")

    bench = BENCH.read_text()
    for token in (
        "defined(DENSE_MARLIN_STANDALONE_SWEEP)",
        '#include "marlin_standalone_configs.inc"',
        "scheduler=standalone-marlin",
        "schema=TM,TN,TK,WM,WN,WarpK,ST,Load",
        "lowbit_dense_marlin_cfg_tm##TM##_tn##TN##_tk##TK",
        "_wk##WK##_st##ST",
        "TileCfg{LOWBIT_DENSE_TAG_SYMBOL",
    ):
        if token not in bench:
            bad.append(f"standalone benchmark registry lacks {token!r}")

    unit = UNIT.read_text()
    for token in (
        "#if defined(DENSE_MARLIN_STANDALONE_SWEEP)",
        "lowbit_dense_run_standalone_marlin",
        "StandaloneMarlinCfg<",
        "Kernel::IsStandaloneMarlin",
        "Kernel::CollectiveMainloop::WarpK == WarpK",
        "LOWBIT_DENSE_UNIT_CONFIGS(LOWBIT_DENSE_DEFINE_STANDALONE_WRAPPER)",
    ):
        if token not in unit:
            bad.append(f"standalone generated-unit wrapper lacks {token!r}")

    parser = TACTIC_PARSER.read_text()
    for token in (
        "function(qz_parse_marlin_tactic_xmacro OUT_ROWS)",
        "X(TM,TN,TK,WM,WN,WarpK,ST,CP_ASYNC,B)",
        "duplicate standalone Marlin row",
    ):
        if token not in parser:
            bad.append(f"standalone eight-field parser lacks {token!r}")

    cmake = CMAKE.read_text()
    block = target_block(cmake)
    if not block:
        bad.append(f"CMake does not define {TARGET}")
    else:
        for token in (
            "qz_parse_marlin_tactic_xmacro",
            "marlin_standalone_configs.inc",
            "DENSE_MARLIN_STANDALONE_SWEEP=1",
            "LOWBIT_DENSE_UNIT_CONFIGS",
            "WarpK",
        ):
            if token not in block:
                bad.append(f"standalone CMake target lacks {token!r}")
        if "DENSE_MARLIN_SWEEP=1" in block:
            bad.append(
                "standalone CMake target falls back to retired DENSE_MARLIN_SWEEP"
            )
    if TARGET not in BUILD.read_text():
        bad.append(f"build.sh does not route {TARGET}")

    box_runner = BOX_RUNNER.read_text()
    for token in (
        TARGET,
        "/workspace/",
        "--list_configs",
        "--search_configs",
        "--streamk_exact_fixture",
        '"--marlin-blocks-per-cu=$bpc"',
        "run_sweep 1",
        "run_sweep 2",
        "bpc2.samples.jsonl",
        "NOT RUN: BPC%d exceeds every admitted kernel occupancy cap",
    ):
        if token not in box_runner:
            bad.append(f"standalone box runner lacks {token!r}")
    for forbidden in ("mktemp", "probe_box_identity", "--csv"):
        if forbidden in box_runner:
            bad.append(f"standalone box runner uses forbidden {forbidden!r}")

    generated = GENERATED_HEADER.read_text()
    generated_rows = [line for line in generated.splitlines()
                      if line.startswith("  X(")]
    for token in (
        "#define MARLIN_STANDALONE_CFG_ROWS 70",
        "#define MARLIN_STANDALONE_CFG_LIST(X, B)",
        "Schema: X(TM,TN,TK,WM,WN,WarpK,ST,LoadToken,B).",
    ):
        if token not in generated:
            bad.append(f"generated standalone registry lacks {token!r}")
    if len(generated_rows) != 70:
        bad.append(
            "generated standalone registry does not contain exactly seventy admitted rows"
        )

    if bad:
        print("[dense-marlin-sweep-contract] FAIL: " + "; ".join(bad))
        return 1

    run = subprocess.run(
        ["bash", str(RUNNER)], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    required = (
        "declared=60000 unique=60000 admitted=70 classic_subspace=60",
        "active-cardinality=2/3/2/2/2/2/5/1 family=m8,m16 geometries=7 stages=s2..s6",
        "negative_controls=4/4_RED emitter=PASS header=BYTE_IDENTICAL ",
        "header_negative_controls=2/2_RED result=PASS",
    )
    if run.returncode != 0 or any(token not in run.stdout for token in required):
        print(
            "[dense-marlin-sweep-contract] FAIL: L172 authority did not close\n"
            + run.stdout[-2400:], file=sys.stderr,
        )
        return 1

    stage = subprocess.run(
        ["bash", str(STAGE_RUNNER)], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    stage_required = (
        "stages=2..6 segment_k_tiles=1..32 exhaustive_cases=160",
        "positive=160/160_PASS negative_controls=3/3_RED result=PASS",
    )
    if stage.returncode != 0 or any(
            token not in stage.stdout for token in stage_required):
        print(
            "[dense-marlin-sweep-contract] FAIL: L183 stage ring did not close\n"
            + stage.stdout[-2400:], file=sys.stderr,
        )
        return 1

    print(
        "[dense-marlin-sweep-contract] PASS: standalone authority "
        "declared=60000 admitted=70 classic-subspace=60; production generated "
        "wrappers consume m8/m16 x seven-TN/TK/WN/WK-geometries x s2..s6; production-target=" + TARGET + "; "
        "generated-unit-row-abi=TM,TN,TK,WM,WN,WarpK,ST,Load; "
        "generic-DENSE_MARLIN_SWEEP=NOT_EVIDENCE; generated-header=BYTE_IDENTICAL; "
        "missing/extra-row-controls=2/2_RED; stage-ring-controls=3/3_RED; "
        "each rejected row has one reason; "
        "box-runner=BPC1/BPC2+EXACT+SEARCH+OVERCAP-NOT-RUN"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
