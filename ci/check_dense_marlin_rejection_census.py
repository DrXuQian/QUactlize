#!/usr/bin/env python3
"""Close the standalone Marlin tactic rejection census.

The former checker counted which rows from the generic dense tactic tables
were admitted by a vendor-scheduler CTA-thread whitelist.  That 4,790-row A2
census is historical: neither those rows nor that scheduler instantiate the
standalone Marlin stack.

L172 now emits the one authoritative standalone Cartesian domain.  This gate
checks that every declared row lands in exactly one bucket and reports the
hardware/resource/current-implementation split.  It deliberately does not
rewrite the historical TSV/Markdown as though those files described the new
stack.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "dev/fold_derivation/run_l172_standalone_marlin_tactic_space.sh"


def main() -> int:
    if not RUNNER.is_file():
        print(f"[dense-marlin-rejection-census] FAIL: missing {RUNNER.relative_to(ROOT)}")
        return 1

    with tempfile.TemporaryDirectory(prefix="quactlize-l172-census-") as tmp:
        env = dict(os.environ)
        env["QUACTLIZE_L172_OUT"] = tmp
        run = subprocess.run(
            ["bash", str(RUNNER)], cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        census_path = Path(tmp) / "census.txt"
        if run.returncode != 0 or not census_path.is_file():
            print(
                "[dense-marlin-rejection-census] FAIL: L172 did not emit its census\n"
                + run.stdout[-2000:], file=sys.stderr,
            )
            return 1
        census = census_path.read_text()

    header = re.search(
        r"schema=marlin-tactic-space-ppu-v1 declared=(\d+) admitted=(\d+) "
        r"classic_subspace=(\d+)", census,
    )
    admitted_row = re.search(r"^admitted=(.+)$", census, re.M)
    kind_line = re.search(
        r"kind\.ADMITTED=(\d+) kind\.HARDWARE_OR_ISA=(\d+) "
        r"kind\.RESOURCE_LIMIT=(\d+) kind\.CURRENT_IMPLEMENTATION=(\d+)",
        census,
    )
    reasons = {
        name: int(count)
        for name, count in re.findall(r"^reason\.([A-Z0-9_]+)=(\d+)$", census, re.M)
    }
    bad: list[str] = []
    if not header or tuple(map(int, header.groups())) != (60000, 1, 60):
        bad.append("schema/cardinality is not declared=60000 admitted=1 classic=60")
    if not kind_line:
        bad.append("missing four-way exclusion-kind census")
        kinds = ()
    else:
        kinds = tuple(map(int, kind_line.groups()))
        if sum(kinds) != 60000 or kinds[0] != 1:
            bad.append(f"kind census does not close: {kinds}")
        if any(value == 0 for value in kinds[1:]):
            bad.append(f"one named rejection class is empty: {kinds}")
    if sum(reasons.values()) != 60000 or reasons.get("NONE") != 1:
        bad.append(
            f"first-failure reason census does not close: sum={sum(reasons.values())} "
            f"NONE={reasons.get('NONE')}"
        )
    if not admitted_row or admitted_row.group(1) != "16,128,128,16,64,32,4,cp_async":
        bad.append("the sole admitted row is not the proved classic reference")
    if "negative_controls=4/4_RED emitter=PASS result=PASS" not in run.stdout:
        bad.append("L172 causal controls did not all turn red")

    if bad:
        print("[dense-marlin-rejection-census] FAIL: " + "; ".join(bad))
        return 1

    assert kinds
    nonzero_reasons = ",".join(
        f"{name}={count}" for name, count in reasons.items() if count
    )
    print("[dense-marlin-rejection-census] reasons: " + nonzero_reasons)
    print(
        "[dense-marlin-rejection-census] PASS: standalone declared=60000 "
        f"admitted=1 rejected=59999 grouped={{hardware:{kinds[1]},"
        f"resource:{kinds[2]},current:{kinds[3]}}}; first-failure sum exact; "
        "generic-A2-4790=PRE_STANDALONE_NOT_EVIDENCE"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
