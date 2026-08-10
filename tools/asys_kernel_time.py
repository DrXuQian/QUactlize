#!/usr/bin/env python3
"""Kernel-only time for every swept config, from ONE asys capture.

WHY THIS REPLACES THE 21-LAUNCH PROTOCOL. The protocol in BOX.md -- run each config alone under MOE_ONLY, count
exactly 21 target launches, drop the warm-up, average the other 20 -- was designed for a world with no per-kernel
timeline. It reconstructs a kernel time from host wall-clock by repeating until the launch overhead averages out.
It does not average out. `time_it` wraps the launch and hggcDeviceSynchronize in the host clock, and the grouped
path calls `initialize` plus a blocking prefix H2D on every iteration, so the overhead is IN each of the 20 timed
samples, not amortised across them.

The size of that error is not a rounding detail. On S068 the whole decode table's winner reads 20.62 us of host
wall-clock while moving 5.28 MB; the same binary at N=K=2048 moves 21.0 MB in 20.74 us. Four times the bytes, the
same time. A number that barely moves when the work quadruples is not measuring the work.

WHAT THIS DOES INSTEAD. asys exports a per-kernel activity timeline. Each launch is already a row with its own
device duration, and launcher, H2D, synchronisation and idle gaps are simply not in it. So one capture of a full
sweep yields the kernel-only time of EVERY row in the table, and repetition becomes a question about variance
rather than a mechanism for removing bias.

THE POINT IS THE RANKING, NOT THE WINNER'S TIMESTAMP. With roughly 13 us of fixed cost on a 7 us kernel, two
configs whose kernel times differ by 2 us are 20.6 us against 22.6 us at the host -- inside this harness's recorded
13% cross-run spread. Every ranking taken through that timer is a ranking of noise plus a constant. Re-reading the
same runs from the timeline can, and probably will, change which config is called the winner.

USAGE

    python3 tools/asys_kernel_time.py --schema capture.sqlite
    python3 tools/asys_kernel_time.py capture.sqlite --log run.log
    python3 tools/asys_kernel_time.py capture.sqlite --table X --name-col Y --dur-col Z

Schema names differ between asys builds, so nothing is hard-coded: the tables are inspected and the kernel
activity table is identified by shape. When detection is wrong, the three overrides say so explicitly rather than
leaving the caller to guess which table was picked -- the chosen table and columns are always printed.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import statistics
import sys

# Activities that are never part of a kernel-only time. Reported by name and count rather than dropped in
# silence: "we excluded the right things" is only checkable if the exclusions are visible.
EXCLUDE_PAT = re.compile(r"bench_floor_nop|memcpy|memset|_nop$", re.I)

NAME_HINTS = ("demangledname", "shortname", "name", "kernelname", "demangled", "symbol")
DUR_HINTS = ("duration", "gpu_duration", "elapsed", "dur")
START_HINTS = ("start", "starttime", "begin", "ts")
END_HINTS = ("end", "endtime", "stop")


def columns(con: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    return [(r[1], (r[2] or "").upper()) for r in con.execute(f'PRAGMA table_info("{table}")')]


def tables(con: sqlite3.Connection) -> list[str]:
    q = "SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name"
    return [r[0] for r in con.execute(q)]


def pick(cols: list[str], hints: tuple[str, ...]) -> str | None:
    low = {c.lower(): c for c in cols}
    for h in hints:                       # exact first, so `end` never loses to `endtime`
        if h in low:
            return low[h]
    for h in hints:
        for lc, orig in low.items():
            if h in lc:
                return orig
    return None


def find_kernel_table(con: sqlite3.Connection) -> tuple[str, str, str | None, str | None]:
    """-> (table, name_col, dur_col, (start_col,end_col) encoded) chosen by shape, preferring 'kernel' in the name."""
    best = None
    for t in tables(con):
        cols = [c for c, _ in columns(con, t)]
        if not cols:
            continue
        name_c = pick(cols, NAME_HINTS)
        dur_c = pick(cols, DUR_HINTS)
        st_c, en_c = pick(cols, START_HINTS), pick(cols, END_HINTS)
        if name_c is None or (dur_c is None and not (st_c and en_c)):
            continue
        try:
            n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        except sqlite3.Error:
            continue
        if n == 0:
            continue
        # A table whose own name says kernel beats a bigger one that does not; among equals, more rows wins.
        score = (2 if "kernel" in t.lower() else 0, n)
        if best is None or score > best[0]:
            best = (score, t, name_c, dur_c, st_c, en_c)
    if best is None:
        raise SystemExit("[asys] no table with a name column and a duration (or start/end) pair; "
                         "run with --schema and pass --table/--name-col/--dur-col")
    _, t, name_c, dur_c, st_c, en_c = best
    return t, name_c, dur_c, f"{st_c}|{en_c}" if (st_c and en_c) else None


def string_table(con: sqlite3.Connection) -> dict[int, str] | None:
    """asys, like nsys, may store names as ids into a strings table. Detect a two-column (int,text) map."""
    for t in tables(con):
        cols = columns(con, t)
        if len(cols) != 2:
            continue
        (c0, t0), (c1, t1) = cols
        if "INT" in t0 and ("CHAR" in t1 or "TEXT" in t1 or t1 == ""):
            try:
                rows = con.execute(f'SELECT "{c0}","{c1}" FROM "{t}"').fetchall()
            except sqlite3.Error:
                continue
            if rows and any(isinstance(r[1], str) and len(r[1]) > 8 for r in rows):
                return {int(a): b for a, b in rows if b is not None}
    return None


def load(path: str, table=None, name_col=None, dur_col=None) -> list[tuple[str, float]]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    if table and name_col:
        cols = [c for c, _ in columns(con, table)]
        dur = dur_col or pick(cols, DUR_HINTS)
        pair = None if dur else f"{pick(cols, START_HINTS)}|{pick(cols, END_HINTS)}"
        t, nc, dc, sp = table, name_col, dur, pair
    else:
        t, nc, dc, sp = find_kernel_table(con)
    print(f"[asys] table={t}  name={nc}  time={dc or sp}")

    expr = f'"{dc}"' if dc else f'("{sp.split("|")[1]}" - "{sp.split("|")[0]}")'
    rows = con.execute(f'SELECT "{nc}", {expr} FROM "{t}" ORDER BY rowid').fetchall()
    strings = None
    if rows and not isinstance(rows[0][0], str):
        strings = string_table(con)
        if strings is None:
            raise SystemExit(f"[asys] {nc} is not text and no strings table was found; pass --name-col")
        print(f"[asys] resolved {len(strings)} names through a strings table")
    out = []
    for nm, d in rows:
        if d is None:
            continue
        s = strings.get(int(nm), f"<id {nm}>") if strings else nm
        out.append((s, float(d)))
    con.close()
    return out


def segments(acts: list[tuple[str, float]]) -> list[tuple[str, list[float]]]:
    """Contiguous runs of one kernel name. Each config's timing loop is one run, which is what lets a full
    sweep be split back into configs without parsing mangled template arguments."""
    segs: list[tuple[str, list[float]]] = []
    for name, dur in acts:
        if segs and segs[-1][0] == name:
            segs[-1][1].append(dur)
        else:
            segs.append((name, [dur]))
    return segs


def tags_from_log(path: str) -> list[str]:
    """Config tags in the order the bench ran them. `TMxTN:TK wWMxWN s<stages> bc<a>-><b>`."""
    pat = re.compile(r"([a-z0-9_]+ \d+x\d+:\d+ w\d+x\d+ s\d+ bc\d+->\d+)")
    seen, out = set(), []
    for line in open(path, errors="replace"):
        m = pat.search(line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            out.append(m.group(1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite")
    ap.add_argument("--schema", action="store_true", help="print tables and columns, then stop")
    ap.add_argument("--log", help="the bench stdout, to name each segment with the config it belongs to")
    ap.add_argument("--table"), ap.add_argument("--name-col"), ap.add_argument("--dur-col")
    ap.add_argument("--scale", type=float, default=1e-3,
                    help="multiply raw durations by this to get microseconds (default 1e-3: asys stores ns)")
    ap.add_argument("--drop-first", type=int, default=1, help="warm-up launches to drop per segment")
    a = ap.parse_args()

    if a.schema:
        con = sqlite3.connect(f"file:{a.sqlite}?mode=ro", uri=True)
        for t in tables(con):
            n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            print(f"\n== {t}  ({n} rows)")
            for c, ty in columns(con, t):
                print(f"     {c:<28} {ty}")
        return 0

    acts = load(a.sqlite, a.table, a.name_col, a.dur_col)
    if not acts:
        print("[asys] the chosen table has no usable rows", file=sys.stderr)
        return 1

    dropped: dict[str, int] = {}
    kept = []
    for nm, d in acts:
        if EXCLUDE_PAT.search(nm):
            dropped[nm] = dropped.get(nm, 0) + 1
        else:
            kept.append((nm, d))
    if dropped:
        print("[asys] excluded from every figure below:")
        for nm, n in sorted(dropped.items(), key=lambda x: -x[1]):
            print(f"         {n:>6} x {nm[:88]}")
    print(f"[asys] {len(kept)} kernel activities in {len(segments(kept))} contiguous segment(s)\n")

    segs = segments(kept)
    tags = tags_from_log(a.log) if a.log else []
    if tags and len(tags) != len(segs):
        # Loud, not fatal: the numbers are still right, only the labels are unavailable.
        print(f"[asys] WARNING: {len(tags)} tag(s) in the log but {len(segs)} segment(s) in the timeline. "
              f"Not labelling; the per-segment numbers below are unaffected.\n")
        tags = []

    rows = []
    for i, (name, durs) in enumerate(segs):
        timed = durs[a.drop_first:] if len(durs) > a.drop_first else durs
        us = [d * a.scale for d in timed]
        mean = statistics.fmean(us)
        spread = (max(us) - min(us)) / mean * 100 if mean and len(us) > 1 else 0.0
        rows.append((tags[i] if tags else name[:52], len(durs), mean,
                     statistics.median(us), min(us), max(us), spread))

    rows.sort(key=lambda r: r[2])
    w = max((len(r[0]) for r in rows), default=10)
    print(f"{'config':<{w}} {'launches':>8} {'mean us':>10} {'median':>9} {'min':>9} {'max':>9} {'spread':>8}")
    for tag, n, mean, med, lo, hi, sp in rows:
        print(f"{tag:<{w}} {n:>8} {mean:>10.3f} {med:>9.3f} {lo:>9.3f} {hi:>9.3f} {sp:>7.1f}%")

    if rows:
        print(f"\n[asys] fastest kernel-only: {rows[0][0]}  {rows[0][2]:.3f} us")
        print("[asys] this ranking is kernel duration only. Host wall-clock from the bench includes launcher,")
        print("       per-iteration initialize, a blocking prefix H2D and the final sync, and is NOT comparable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
