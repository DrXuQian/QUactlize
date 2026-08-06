#!/usr/bin/env python3
"""A format the registry lists must have the collective its own row implies present in the tree.

    python3 ci/check_format_table_buildable.py

THE HOLE THIS CLOSES, and it is specific. ppu_format_config.inc lists five k-quants and formats.py carries four
capability sets. schemes.py is explicit that the sets are LITERALS cross-checked against the table by
tests/test_schemes_consistency.py rather than derived -- and it records why that wording matters: an earlier
header CLAIMED derivation, there was none, and the two drifted exactly as the warning said.

But "the table and the sets agree" is a statement about two hand-written things agreeing with each other. Neither
is checked against what the TREE CAN BUILD. Remove the two-plane collective -- which is the point of making
features revertible commits -- and the table still lists Q3_K/Q5_K/Q6_K, formats.py still claims them, and
test_schemes_consistency still passes, because the two copies still agree. The first thing that notices is a run.

THE ROW ALREADY CARRIES THE ANSWER, so this invents no second source. A row's high_bits field IS the plane
count: high_bits == 0 is single-plane and needs only the base collective; high_bits != 0 is a bit-plane concat
and cannot be instantiated without ppu_mma_aiu_mixed_input_2plane.hpp. That is not a convention chosen here --
it is what the two-plane collective is for, and the builder selects it on exactly that condition.

WHAT THIS IS NOT. It does not prove the format works, or that the kernel is correct, or that a build with the
header actually compiled that format in -- a build can also drop a format through QUACTLIZE_DENSE_ONLY without
touching any header, and this gate cannot see that. It answers one question: does the tree still contain the
thing a listed row needs. That is the question a feature revert gets wrong, and it is answerable with no build,
no device, and no library.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TABLE = ROOT / "quactlize" / "include" / "ppu_format_config.inc"
EXT = ROOT / "quactlize" / "include" / "quactlize_extensions" / "cutlass" / "gemm" / "collective"

# What a row's own fields imply about the collective it needs. Keyed on a predicate over the parsed row, so a new
# format is covered the day it is added rather than the day someone remembers to extend a list.
REQUIREMENTS = [
    (lambda r: r["high_bits"] != 0,
     EXT / "ppu_mma_aiu_mixed_input_2plane.hpp",
     "high_bits != 0 is a bit-plane concat; the builder selects MainloopPPUAiuMixedInput2Plane for it"),
    (lambda r: True,
     EXT / "quactlize_mma_mixed_input.hpp",
     "every format goes through the single-plane collective, two-plane ones for their low plane"),
]

ROW = re.compile(
    r"^\s*X\(\s*(\w+)\s*,\s*\"([^\"]+)\"\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,"
    r"\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", re.M)


def rows():
    if not TABLE.is_file():
        print(f"[format-buildable] SKIP: {TABLE} does not exist")
        sys.exit(0)
    out = []
    for m in ROW.finditer(TABLE.read_text()):
        out.append({"id": m.group(1), "name": m.group(2), "qtype": int(m.group(3)),
                    "low_bits": int(m.group(4)), "high_bits": int(m.group(5)),
                    "group_size": int(m.group(6))})
    return out


def main() -> int:
    parsed = rows()
    # A PARSE THAT FINDS NOTHING MUST NOT REPORT SUCCESS. The regex is pinned to a nine-field row; if the table
    # gains a field it stops matching, and a gate that silently checks zero rows is the failure this repository
    # keeps paying for.
    if len(parsed) < 3:
        print(f"[format-buildable] ERROR: parsed {len(parsed)} row(s) from {TABLE.name}; the X-macro's shape "
              f"changed and this regex no longer describes it. Not reporting a verdict.")
        return 1

    missing = []
    for r in parsed:
        for applies, header, why in REQUIREMENTS:
            if applies(r) and not header.is_file():
                missing.append((r, header, why))

    if missing:
        print(f"[format-buildable] FAIL: {len(missing)} listed format(s) need a collective the tree does not have:")
        for r, header, why in missing:
            print(f"    {r['name']} (qtype {r['qtype']}, {r['low_bits']}+{r['high_bits']} bits)")
            print(f"        needs {header.relative_to(ROOT)}")
            print(f"        because {why}")
        print("    Either the collective was removed and the row must go with it, or the row is new and the")
        print("    collective has not landed. Do not 'fix' this by deleting the requirement.")
        return 1

    two = sum(1 for r in parsed if r["high_bits"])
    print(f"[format-buildable] PASS: {len(parsed)} format(s) listed, {two} of them two-plane, and every collective "
          f"their rows imply is present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
