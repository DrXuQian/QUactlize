#!/usr/bin/env python3
"""Contract for dA outer bases and logical-N metadata residues."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HELPER = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_argument_contract.hpp"
COLLECTIVES = (
    ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp",
    ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp",
    ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_mixed_input_2plane.hpp",
)
ORACLE = ROOT / "dev/fold_derivation/l128_mixed_argument_contract.cu"
RUNNER = ROOT / "dev/fold_derivation/run_l128_mixed_argument_contract.sh"
AUDIT = ROOT / "dev/fold_derivation/MIXED_ARGUMENT_ASSUMPTIONS.md"


def flat(text: str) -> str:
    return re.sub(r"\s+", "", text)


def audit(texts: list[str]) -> list[str]:
    helper, ordinary, folded, two_plane, oracle, runner, audit_doc = texts
    bad: list[str] = []
    h = flat(helper)
    for token in (
        "mixed_a_expert_base(",
        "int64_t(group_row_offsets[l_coord])*int64_t(cute::get<0>(dA))",
        "int64_t(l_coord)*int64_t(cute::get<2>(dA))",
        "mixed_logical_n_residue(",
        "N-int64_t(logical_tile_n)*int64_t(n_coord)",
    ):
        if token not in h:
            bad.append(f"shared argument helper lost {token!r}")

    for label, source in zip(("ordinary", "fold", "two-plane"),
                             (ordinary, folded, two_plane)):
        s = flat(source)
        if s.count("detail::mixed_a_expert_base(") != 1:
            bad.append(f"{label} does not consume the shared dA outer-base seam once")
        if s.count("detail::mixed_logical_n_residue(") != 1:
            bad.append(f"{label} does not consume the logical-N residue seam once")
        if "a_row_off*K" in s:
            bad.append(f"{label} restored compact A outer-base arithmetic")
        if "scale_residue_n=N-size<0>(gB)*n_coord" in s:
            bad.append(f"{label} restored physical-B metadata residue arithmetic")

    o = flat(oracle)
    for token in (
        "kRowPitch=kK+16",
        "kExpertPitch=kM*kRowPitch+32",
        "explicit-int64-stride-anchor",
        "physical_formula_red+=legacy!=expected",
        "scope=dA-outer-base+logical-N-residue",
    ):
        if token not in o:
            bad.append(f"L128 lost load-bearing token {token!r}")
    if "nvcc-std=c++17-xcu-arch=sm_80" not in flat(runner):
        bad.append("L128 runner no longer compiles the host oracle")
    for token in (
        "| `dS` |",
        "| Zero-plane stride (`dZ`) |",
        "| Outer A base versus `dA` |",
        "| Runtime `group_size` versus static schedule group size |",
        "| Logical-N residue for metadata in fold / 2-plane |",
        "| Interleaved `dB` |",
        "| Plane-2 `dB2` / `dB2_valid` |",
        "| Packed `ptr_S`, `dS`, and `ptr_Z` semantics |",
        "| Divisibility of interleave/fold/packed extents |",
        "**L128 FIXED**",
        "FORMAT RESTRICTION / MISSING FAIL-CLOSE",
        "place_derived -> recover_derived == identity",
    ):
        if token not in audit_doc:
            bad.append(f"mixed-argument audit lost {token!r}")
    return bad


def main() -> int:
    paths = (HELPER, *COLLECTIVES, ORACLE, RUNNER, AUDIT)
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    if missing:
        print("[mixed-argument-contract] FAIL: missing " + ", ".join(missing))
        return 1
    texts = [p.read_text() for p in paths]
    bad = audit(texts)
    if bad:
        print("[mixed-argument-contract] FAIL: " + "; ".join(bad))
        return 1

    plants = (
        (0, "int64_t(cute::get<0>(dA))", "int64_t(256)", "ragged A row pitch"),
        (0, "int64_t(cute::get<2>(dA))", "int64_t(l_coord) * 0 + int64_t(1792)",
         "uniform A expert pitch"),
        (0, "int64_t(logical_tile_n)", "int64_t(logical_tile_n / 2)",
         "logical N residue"),
        (1, "detail::mixed_a_expert_base(", "detail::planted_compact_a_base(",
         "ordinary helper bypass"),
        (2, "detail::mixed_logical_n_residue(", "detail::planted_physical_n_residue(",
         "fold residue bypass"),
        (4, "physical_formula_red += legacy != expected;",
         "physical_formula_red += false;", "residue negative control"),
    )
    for index, old, new, label in plants:
        planted = list(texts)
        if old not in planted[index]:
            print(f"[mixed-argument-contract] FAIL: cannot plant {label}")
            return 1
        planted[index] = planted[index].replace(old, new, 1)
        if not audit(planted):
            print(f"[mixed-argument-contract] FAIL: checker accepted planted {label}")
            return 1

    run = subprocess.run(
        ["bash", str(RUNNER)], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    required = (
        "A uniform_bad=0 ragged_bad=0 explicit-int64-stride-anchor=PASS",
        "A old-row-times-K uniform_red=4/5 ragged_red=4/5 -> EXPECTED-RED",
        "N-residue cases=585 bad=0 physical-TileN-over-F-red=132 -> PASS/EXPECTED-RED",
        "result=PASS scope=dA-outer-base+logical-N-residue",
    )
    if run.returncode != 0:
        print(f"[mixed-argument-contract] FAIL: L128 rc={run.returncode}: {run.stdout[-1200:]}")
        return 1
    absent = [token for token in required if token not in run.stdout]
    if absent:
        print("[mixed-argument-contract] FAIL: output missing " + repr(absent) +
              "\n" + run.stdout[-1200:])
        return 1
    print("[mixed-argument-contract] PASS: three collectives share dA/logical-N seams; "
          "585 residue cases and six source plants rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
