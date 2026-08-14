#!/usr/bin/env python3
"""Fail-closed contract for the ordinary dense instruction-M sweep filter.

The runtime field is evidence only because each generated unit derives it
from the instantiated production TiledMma atom and statically binds that atom
to the row.  This gate also pins the filter to ordinary dense + standalone
searches, excluding every named scheduler/mechanism target.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks/test_lowbit_dense_bench.cu"
UNIT = ROOT / "benchmarks/lowbit_dense_unit.inc"
TABLES = (
    ROOT / "benchmarks/lowbit_dense_configs.inc",
    ROOT / "benchmarks/lowbit_dense_i2_configs.inc",
    ROOT / "benchmarks/lowbit_dense_i1_configs.inc",
)


def audit(bench: str, unit: str) -> list[str]:
    bad: list[str] = []
    bench_required = (
        "LOWBIT_DENSE_INST_M_SYMBOL",
        "int instruction_m = 0;",
        'cmd.get_cmd_line_argument("instruction-m", marlin_instruction_m)',
        "c.instruction_m == options.marlin_instruction_m",
        "c.instruction_m != options.marlin_instruction_m",
        "selected.instruction_m != options.marlin_instruction_m",
        "instruction_m=%d eligible_rows=%d compiled_rows=%d",
        "bits=%d TSK=%d gs=%d instruction_m=%d",
        "defined(DENSE_NAMED_SCHEDULER) && !defined(DENSE_MARLIN_STANDALONE_SWEEP)",
    )
    unit_required = (
        "using TiledMma = typename TagCfg::Main::TiledMma;",
        "typename TiledMma::AtomShape_MNK{}",
        "static_assert(atom_m == 8 || atom_m == 16",
        "static_assert(atom_m == ((TM == 8 && WM == 8) ? 8 : 16)",
        "return atom_m;",
    )
    for token in bench_required:
        if token not in bench:
            bad.append("bench missing " + token)
    for token in unit_required:
        if token not in unit:
            bad.append("unit missing " + token)
    return bad


def rows(path: Path) -> list[tuple[int, int]]:
    # Ordinary tables use X(TM,TN,TK,WM,WN,ST,BC,A).  Keep only TM/WM: the
    # production unit independently proves how these fields bind to Atom M.
    out: list[tuple[int, int]] = []
    for match in re.finditer(
        r"^\s*X\((\d+),\s*\d+,\s*\d+,\s*(\d+),\s*\d+,\s*\d+,\s*\d+,",
        path.read_text(), re.MULTILINE,
    ):
        out.append(tuple(map(int, match.groups())))
    return out


def main() -> int:
    missing = [str(p.relative_to(ROOT)) for p in (BENCH, UNIT, *TABLES)
               if not p.is_file()]
    if missing:
        print("[dense-instruction-m-filter] FAIL missing " + ", ".join(missing))
        return 1
    bench = BENCH.read_text()
    unit = UNIT.read_text()
    bad = audit(bench, unit)
    if bad:
        print("[dense-instruction-m-filter] FAIL: " + "; ".join(bad))
        return 1

    counts: list[str] = []
    for table in TABLES:
        parsed = rows(table)
        m8 = sum(tm == 8 and wm == 8 for tm, wm in parsed)
        m16 = len(parsed) - m8
        if not parsed or not m8 or not m16:
            print(f"[dense-instruction-m-filter] FAIL {table.name}: "
                  f"rows={len(parsed)} m8={m8} m16={m16}")
            return 1
        counts.append(f"{table.name}:{m8}/{m16}")

    # Each plant removes a different load-bearing seam.  The same audit must
    # turn red; otherwise this check merely recognizes nearby text.
    plants = (
        (bench.replace("c.instruction_m == options.marlin_instruction_m",
                       "((c.tm == 8 && c.wm == 8) ? 8 : 16) == options.marlin_instruction_m", 1), unit),
        (bench.replace("selected.instruction_m != options.marlin_instruction_m",
                       "selected.tm != options.marlin_instruction_m", 1), unit),
        (bench, unit.replace("typename TiledMma::AtomShape_MNK{}",
                             "cute::Shape<cute::Int<TM>>{}", 1)),
        (bench, unit.replace("static_assert(atom_m == ((TM == 8 && WM == 8) ? 8 : 16)",
                             "static_assert(atom_m == atom_m", 1)),
    )
    red = sum(bool(audit(b, u)) for b, u in plants)
    if red != len(plants):
        print(f"[dense-instruction-m-filter] FAIL plants red={red}/{len(plants)}")
        return 1

    print("[dense-instruction-m-filter] PASS ordinary tables m8/m16=" +
          ",".join(counts) +
          f"; metadata=compiled-TiledMma-AtomShape; plants={red}/{len(plants)}_RED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
