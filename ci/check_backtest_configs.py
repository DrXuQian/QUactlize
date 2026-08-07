#!/usr/bin/env python3
"""Every config the record calls a WINNER must still be reachable by the sweep that would find it.

WHY THIS EXISTS. On 2026-08-07 a MoE sweep reported q3 at 14.4% MFU. The record (BACKTEST A5) has Q3 at 53.8%,
reached "after all four levers", one of which was the warp shape w64x32. The Q3_K table carried six w64x32 rows
out of 202. Nothing was broken -- the grid had simply been emitted narrower than the grid the record was measured
on, and there was no check anywhere that compares the two. A sweep cannot find a winner that is not in its table,
and it does not say so: it reports the best of what it has, which reads exactly like a measurement.

WHAT IT CHECKS. docs/BACKTEST.md is the record. Every row that names a concrete tile config is parsed, mapped to
the table a sweep would search for it, and looked up. A recorded winner absent from the current table is an ERROR.

WHAT IT REFUSES TO DO. Silently skip. A config string this parser does not understand is reported as UNPARSED and
counts against the run, because "we checked everything we could parse" is the failure mode where the parser
quietly narrows and the gate keeps passing. Same for a record whose format has no table at all -- that is the
loudest finding available and it must not degrade to a skip.

DERIVED, NOT DUPLICATED. The expected configs are parsed out of BACKTEST.md rather than listed here. A checker
carrying its own copy of the answer cannot fail when the record gains a row -- and a new record is exactly when
this check has something to say.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "docs" / "BACKTEST.md"
TABLES = ROOT / "benchmarks"

# A record row names a width in prose ("int4", "Q3 (int2+int1 B-concat)") and a sweep searches a per-format table.
WIDTH_TO_FORMAT = {
    "int4": "i4",
    "int2": "i2",
    "int1": "i1",
    "q3": "Q3_K",
    "q5": "Q5_K",
    "q6": "Q6_K",
}

# Which space a section's rows would be swept in.
SECTION_SPACE = {"A": "dense", "B": "dense", "C": "grouped", "D": "dense", "E": None}
# D4 is explicitly a grouped measurement inside the decode section.
ROW_SPACE_OVERRIDE = {"D4": "grouped"}


def load_table(path: Path) -> set[tuple[int, ...]]:
    rows = set()
    for m in re.finditer(r"^ *X\(([\d,]+),B\)", path.read_text(), re.M):
        rows.add(tuple(int(v) for v in m.group(1).split(",")))
    return rows


def parse_config(s: str):
    """-> (fields, kind) where fields is a 6-list with None for unknown, kind in {'full','partial'}; or None.

    Six notations appear in the record and they are not interchangeable. Each is anchored so a string that merely
    resembles one cannot be read as another -- `64x64:64 s4` and `64x64:64x32:s3` differ by one segment and mean
    different things (TileK vs WarpM).
    """
    s = s.strip().replace("×", "x")
    # MoE tag:            i4 64x128:64 w64x16 s6
    m = re.fullmatch(r"(?:\w+ )?(\d+)x(\d+):(\d+) w(\d+)x(\d+) s(\d+)", s)
    if m:
        return [int(v) for v in m.groups()], "full"
    # dense tag, current: 16x16x256:16x16:s2
    m = re.fullmatch(r"(\d+)x(\d+)x(\d+):(\d+)x(\d+):s(\d+)", s)
    if m:
        return [int(v) for v in m.groups()], "full"
    # dense tag, pre-TileK: 64x64:64x32:s3 -- TileK was not a row field yet, and the record states this string
    # became `64x64x64:...`, i.e. TileK equalled the artifact's 64. Anything else would be inventing a field.
    m = re.fullmatch(r"(\d+)x(\d+):(\d+)x(\d+):s(\d+)", s)
    if m:
        tm, tn, wm, wn, st = (int(v) for v in m.groups())
        return [tm, tn, 64, wm, wn, st], "full"
    # tuple + warp + stage:  (64,128,64) w64x64 s2
    m = re.fullmatch(r"\((\d+),(\d+),(\d+)\) w(\d+)x(\d+) s(\d+)", s)
    if m:
        return [int(v) for v in m.groups()], "full"
    # tile + TileK + stage, no warp shape:  64x64:64 s4
    m = re.fullmatch(r"(\d+)x(\d+):(\d+) s(\d+)", s)
    if m:
        tm, tn, tk, st = (int(v) for v in m.groups())
        return [tm, tn, tk, None, None, st], "partial"
    # bare tuple:  32,128,128
    m = re.fullmatch(r"(\d+),(\d+),(\d+)", s)
    if m:
        tm, tn, tk = (int(v) for v in m.groups())
        return [tm, tn, tk, None, None, None], "partial"
    # tile / warp / stage, no TileK:  64x64 / 32x32 / s4
    m = re.fullmatch(r"(\d+)x(\d+) */ *(\d+)x(\d+) */ *s(\d+)", s)
    if m:
        tm, tn, wm, wn, st = (int(v) for v in m.groups())
        return [tm, tn, None, wm, wn, st], "partial"
    # bare warp shape:  w64x32 -- "after adding `w64x32`" is a claim about the WINNER's warp shape and nothing
    # else. Treating it as unparseable would have hidden the one lever the record calls its largest.
    m = re.fullmatch(r"w(\d+)x(\d+)", s)
    if m:
        wm, wn = (int(v) for v in m.groups())
        return [None, None, None, wm, wn, None], "partial"
    # tile with TileK / stage, no warp:  64x64x128 / s3
    m = re.fullmatch(r"(\d+)x(\d+)x(\d+) */ *s(\d+)", s)
    if m:
        tm, tn, tk, st = (int(v) for v in m.groups())
        return [tm, tn, tk, None, None, st], "partial"
    return None


def matches(row: tuple[int, ...], want: list) -> bool:
    return len(row) == len(want) and all(w is None or w == r for r, w in zip(row, want))


def main() -> int:
    if not RECORD.is_file():
        print(f"[backtest-configs] ERROR: {RECORD} is missing")
        return 1

    tables = {}
    for p in sorted(TABLES.glob("lowbit_*_configs.inc")):
        if p.name == "lowbit_dense_configs.inc":
            key = ("dense", "i4")       # the one legacy generic name remains the sole i4 table
        elif p.name.startswith("lowbit_dense_"):
            key = ("dense", p.name.removeprefix("lowbit_dense_").removesuffix("_configs.inc"))
        elif p.name.startswith("lowbit_grouped_"):
            key = ("grouped", p.name.removeprefix("lowbit_grouped_").removesuffix("_configs.inc"))
        else:
            continue
        if key in tables:
            print(f"[backtest-configs] ERROR: {p.name} and {tables[key][0]} both claim {key}; "
                  "one format/space must have one table identity")
            return 1
        tables[key] = (p.name, load_table(p))
    if not any(space == "dense" for space, _ in tables):
        print("[backtest-configs] ERROR: no dense tables found")
        return 1

    text = RECORD.read_text()
    errors, unparsed, checked, na = [], [], [], []

    for line in text.splitlines():
        m = re.match(r"^\| ([A-E])(\d+) \|(.*)$", line.strip())
        if not m:
            continue
        section, rid = m.group(1), m.group(1) + m.group(2)
        cells = [c.strip() for c in m.group(3).split("|")]
        space = ROW_SPACE_OVERRIDE.get(rid, SECTION_SPACE.get(section))
        if space is None:
            continue

        # The width is whatever appears in the row's prose; the config is whatever is in backticks. Take the FIRST
        # backticked run that parses as a config -- rows carry other backticked things (memory slugs, `[A path]`).
        row_text = " | ".join(cells)
        # THE MoE TAG CARRIES ITS OWN WIDTH (`i4 64x128:64 ...`), the dense prose spells it out ("int4"), and some
        # rows state neither. Accept both spellings; a row that states neither is a defect IN THE RECORD and is
        # reported as such rather than defaulted -- guessing the width is how a check starts agreeing with itself.
        width = None
        for w, short in (("int4", "i4"), ("int2", "i2"), ("int1", "i1"),
                         ("q3", "Q3_K"), ("q5", "Q5_K"), ("q6", "Q6_K")):
            if re.search(rf"\b{w}\b", row_text, re.I) or re.search(rf"`{short} ", row_text):
                width = w
                break

        cfg = None
        for tick in re.findall(r"`([^`]+)`", row_text):
            got = parse_config(tick)
            if got:
                cfg = (tick, *got)
                break
        if cfg is None:
            # Rows with no config at all (a bare percentage, a prose note) are not this gate's business; rows that
            # HAVE a backticked config-looking token and did not parse are.
            for tick in re.findall(r"`([^`]+)`", row_text):
                if re.search(r"\d+x\d+|\d+,\d+,\d+", tick) and "path" not in tick:
                    unparsed.append((rid, tick))
            continue

        tick, want, kind = cfg
        # PPU_B_CHUNK became the seventh row field after these records were written. A2/A3 name the requested
        # mode explicitly, so ignoring it would let a bc0 row "reproduce" a bc1 winner. Absence means unknown,
        # as with an omitted warp or stage; a bare B_CHUNK spelling is the historical spelling for mode 1.
        bc_m = re.search(r"\b(?:PPU_)?B_CHUNK\b(?:\s*=\s*(\d+))?", row_text)
        want = [*want, int(bc_m.group(1)) if bc_m and bc_m.group(1) else (1 if bc_m else None)]
        if width is None:
            unparsed.append((rid, f"{tick} (no width named in the row)"))
            continue
        fmt = WIDTH_TO_FORMAT[width]
        tkey = (space, fmt)
        if tkey not in tables:
            na.append((rid, tick, width, f"no {space} {fmt} table exists"))
            continue

        name, rows = tables[tkey]
        hits = [r for r in rows if matches(r, want)]
        if hits:
            checked.append((rid, tick, name, kind, len(hits)))
        else:
            errors.append((rid, tick, width, name, kind, want))

    print(f"[backtest-configs] {len(checked)} recorded config(s) still reachable, "
          f"{len(errors)} missing, {len(unparsed)} unparsed, {len(na)} with no table")
    for rid, tick, name, kind, n in checked:
        print(f"  OK      {rid:<4} `{tick}` -> {name} ({kind}, {n} matching row(s))")
    for rid, tick, width, why in na:
        print(f"  NOTABLE {rid:<4} `{tick}` [{width}] {why}")
    for rid, tick in unparsed:
        print(f"  UNPARSED {rid:<4} `{tick}`")
    for rid, tick, width, name, kind, want in errors:
        print(f"  MISSING {rid:<4} `{tick}` [{width}] is NOT in {name} ({kind}: {want})")

    if errors or unparsed or na:
        print("[backtest-configs] a winner the sweep cannot reach is not a config choice -- it is a hole in the grid")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
