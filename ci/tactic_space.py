#!/usr/bin/env python3
"""DO THE DENSE AND GROUPED SEARCH SPACES COINCIDE? -- and does anything notice when they stop?

WHY THIS EXISTS. The user's constraint (2026-08-04) is that the final tactic search space must be the same shape
for dense and for grouped. Today it will be, because both are being written from one intent. The failure is
later and quiet: one side gains a width or a tile shape, the sets drift, nothing breaks, and the next sweep
covers a space that is no longer the union while reporting a winner as if it were. That is the same defect as
the incomplete Q6 tactic and as l105 classifying all-zero buffers -- a result whose coverage is narrower than
its reader believes.

So the sets are EMITTED by each launcher (from the constants it actually dispatches on, not a transcription --
see .coord/INBOX.md 031) and compared here.

INPUT FORMAT, one configuration per line, `#` comments and blank lines ignored:

    dense    bits=2 tk=128 f=1 tile=64x128x128 warp=32x32  reachable
    grouped  bits=2 tk=64  f=2 tile=64x128x64  warp=32x64  reachable
    dense    bits=1 tk=64  f=4 tile=64x128x64  warp=32x64  excluded: fpA pins the low plane at F=1

AN EXCLUDED ROW IS DATA, NOT AN OMISSION. A cell one side cannot reach is fine; the property is that the
difference is STATED. A missing row and a reachable row are indistinguishable to a reader, which is why the
comparator treats "present on one side, absent on the other" as the failure and "excluded with a reason" as a
pass.

    python3 ci/tactic_space.py <emitted.txt>       # compare
    python3 ci/tactic_space.py --self-test         # prove it can fail, with no input at all
"""
import argparse
import pathlib
import re
import sys

KEY = ("bits", "tk", "f", "tile", "warp")
ROW = re.compile(r"^\s*(dense|grouped)\s+(.*?)\s*$")


def parse(text: str):
    """-> ({side: {key: reason_or_None}}, [complaint, ...]). A malformed line is a complaint, never a skip:
    a parser that ignores what it does not understand reports agreement between two files it did not read."""
    sides, bad = {"dense": {}, "grouped": {}}, []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        m = ROW.match(line)
        if not m:
            bad.append(f"line {n}: does not start with dense/grouped: {raw.strip()!r}")
            continue
        side, rest = m.group(1), m.group(2)
        fields, reason = {}, None
        if "excluded:" in rest:
            rest, reason = rest.split("excluded:", 1)
            reason = reason.strip()
            if not reason:
                bad.append(f"line {n}: 'excluded:' with no reason -- an unexplained exclusion is an omission")
        elif "reachable" in rest:
            rest = rest.replace("reachable", "")
        else:
            bad.append(f"line {n}: neither 'reachable' nor 'excluded:' -- status must be explicit")
        for tok in rest.split():
            if "=" not in tok:
                bad.append(f"line {n}: token {tok!r} is not k=v")
                continue
            k, v = tok.split("=", 1)
            fields[k] = v
        missing = [k for k in KEY if k not in fields]
        if missing:
            bad.append(f"line {n}: missing {','.join(missing)}")
            continue
        key = tuple(fields[k] for k in KEY)
        if key in sides[side]:
            bad.append(f"line {n}: {side} repeats {key}")
        sides[side][key] = reason
    return sides, bad


def compare(sides):
    """-> [complaint, ...]. Empty means the spaces coincide, or differ only where both sides say so."""
    out = []
    d, g = sides["dense"], sides["grouped"]
    if not d or not g:
        out.append(f"one side is EMPTY (dense {len(d)}, grouped {len(g)}) -- an empty set trivially 'matches'")
        return out
    for key in sorted(set(d) | set(g)):
        fmt = " ".join(f"{k}={v}" for k, v in zip(KEY, key))
        in_d, in_g = key in d, key in g
        if in_d != in_g:
            out.append(f"{fmt}: present in {'dense' if in_d else 'grouped'} only, ABSENT from the other. "
                       f"State it as excluded with a reason, or make it reachable.")
        elif (d[key] is None) != (g[key] is None):
            side = "dense" if d[key] else "grouped"
            out.append(f"{fmt}: reachable on one side, excluded on {side} ({d[key] or g[key]}). That is allowed "
                       f"-- but the sweep must then not treat this cell as covered on both.")
    return out


SELF_TEST = """
# Proves the comparator can FAIL. Every row below is designed to trip exactly one check.
dense    bits=4 tk=256 f=1 tile=64x128x256 warp=32x32 reachable
grouped  bits=4 tk=256 f=1 tile=64x128x256 warp=32x32 reachable
grouped  bits=2 tk=64  f=2 tile=64x128x64  warp=32x64 reachable
dense    bits=1 tk=64  f=4 tile=64x128x64  warp=32x64 excluded: low plane pinned at F=1
grouped  bits=1 tk=64  f=4 tile=64x128x64  warp=32x64 reachable
dense    bits=9 tk=64
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("emitted", nargs="?", help="the file each launcher appended its legal set to")
    ap.add_argument("--self-test", action="store_true",
                    help="run the planted-divergence fixture; passes only if it detects all four defects")
    a = ap.parse_args()

    if a.self_test:
        sides, bad = parse(SELF_TEST)
        problems = bad + compare(sides)
        # THE FIXTURE IS PLANTED, so the assertion is on the COUNT and the KIND, not on "it complained". A
        # comparator that reports one generic failure for a file with four distinct defects is not doing the
        # job it will be trusted with.
        want = {"missing": 0, "only": 0, "one side": 0}
        for p in problems:
            if "missing" in p:
                want["missing"] += 1
            if "only, ABSENT" in p:
                want["only"] += 1
            if "excluded on" in p:
                want["one side"] += 1
        print("\n".join("  " + p for p in problems) or "  (nothing)")
        ok = want["missing"] >= 1 and want["only"] >= 1 and want["one side"] >= 1
        print(f"\nself-test: malformed={want['missing']} one-sided={want['only']} "
              f"asymmetric-status={want['one side']} -> {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    if not a.emitted:
        ap.error("give a file, or --self-test")
    p = pathlib.Path(a.emitted)
    if not p.is_file():
        # SKIP, NOT PASS, AND IT SAYS WHY. The emitters do not exist yet (INBOX 031). A gate that returns 0
        # because its input is absent is the shape this whole file argues against.
        print(f"SKIP: {p} does not exist. The dense/grouped emitters are not written yet (.coord/INBOX.md 031); "
              f"until they are, this check has nothing to read and MUST NOT report success.")
        return 77
    sides, bad = parse(p.read_text())
    problems = bad + compare(sides)
    for x in problems:
        print(f"  {x}")
    print(f"\ndense {len(sides['dense'])} configs, grouped {len(sides['grouped'])} -> "
          f"{'COINCIDE' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
