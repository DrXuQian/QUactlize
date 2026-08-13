#!/usr/bin/env python3
"""Contract for the *standalone* Marlin tactic authority.

There is intentionally no production standalone sweep target yet.  The old
``DENSE_MARLIN_SWEEP`` target enumerates the retired generic mixed-input
kernel and must not be reported as evidence for ``MarlinCollectivePPU``.

This gate proves the honest intermediate state: the standalone Cartesian
authority is exhaustive and causally checked, one fixed row is admitted, all
other rows have one named first-failure reason, and no benchmark currently
claims that this authority is a production sweep.  When a standalone target is
added this gate must be extended; turning the historical target green is not a
substitute.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
AUTHORITY = ROOT / "quactlize/include/marlin_tactic_space_ppu.hpp"
RUNNER = ROOT / "dev/fold_derivation/run_l172_standalone_marlin_tactic_space.sh"
PRODUCTION_CANDIDATES = (
    ROOT / "benchmarks/test_lowbit_dense_bench.cu",
    ROOT / "benchmarks/lowbit_dense_unit.inc",
    ROOT / "quactlize/csrc/CMakeLists.txt.in",
    ROOT / "build.sh",
)


def main() -> int:
    missing = [str(path.relative_to(ROOT)) for path in (AUTHORITY, RUNNER)
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

    # The fixed standalone target must consume the sole admitted row.  No
    # source may yet claim a multi-row standalone sweep: DENSE_MARLIN_SWEEP is
    # explicitly the historical generic target.
    bench = PRODUCTION_CANDIDATES[0].read_text()
    fixed_tokens = (
        '#if defined(DENSE_MARLIN_WK4_AB)',
        '#include "marlin_tactic_space_ppu.hpp"',
        "marlin_tactics_ppu::MarlinTacticPPU Tactic",
        "marlin_tactics_ppu::admitted(Tactic)",
    )
    if any(token not in bench for token in fixed_tokens):
        bad.append(
            "the fixed DENSE_MARLIN_WK4_AB target does not consume the standalone authority"
        )
    other_consumers = []
    for path in PRODUCTION_CANDIDATES[1:]:
        text = path.read_text()
        if "marlin_tactic_space_ppu.hpp" in text or "MarlinTacticPPU" in text:
            other_consumers.append(str(path.relative_to(ROOT)))
    if other_consumers:
        bad.append("unregistered standalone sweep consumer: " +
                   ", ".join(other_consumers))

    if bad:
        print("[dense-marlin-sweep-contract] FAIL: " + "; ".join(bad))
        return 1

    run = subprocess.run(
        ["bash", str(RUNNER)], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    required = (
        "declared=60000 unique=60000 admitted=1 classic_subspace=60",
        "active-cardinality=1/1/1/1/1/1/1/1",
        "negative_controls=4/4_RED emitter=PASS result=PASS",
    )
    if run.returncode != 0 or any(token not in run.stdout for token in required):
        print(
            "[dense-marlin-sweep-contract] FAIL: L172 authority did not close\n"
            + run.stdout[-2400:], file=sys.stderr,
        )
        return 1

    print(
        "[dense-marlin-sweep-contract] PASS: standalone authority "
        "declared=60000 admitted=1 classic-subspace=60; fixed target consumes "
        "the sole admitted row; each rejected row has one reason; "
        "multirow-production-sweep=NOT_IMPLEMENTED; "
        "DENSE_MARLIN_SWEEP=PRE_STANDALONE_NOT_EVIDENCE"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
