#!/usr/bin/env python3
"""A BUILD SWITCH NOBODY CAN TURN ON IS NOT A SWITCH -- it is dead code wearing an option's clothes.

    python3 ci/check_switch_macros.py            verdict
    python3 ci/check_switch_macros.py --list     the whole inventory, live ones included

WHAT THIS COST, and it is not hypothetical. Three macros in this tree mean "shrink A's 15/16 padding at small M":
PPU_A_PACK (a binary-wide #if that WINS over everything downstream), PPU_A_CPASYNC (a policy default that an
explicit per-row value silently overrides), and the ACR column of the tactic table. docs/CHECKPOINT.md lists the
first two on one line as if they were one thing. On 2026-08-06 a measurement was filed as "compact A at capacity 1
is 45% slower" and could not be attributed afterwards, because the run's A provider was never witnessed and two of
the three spellings were reachable only by remembering which was passed. One idea, three spellings, and the cost
landed on a number in BACKTEST rather than on a compile.

WHAT IT CHECKS, and it is deliberately one narrow thing: every preprocessor switch this repository OWNS must be
reachable -- something must be able to define it. Two ways count:

  DEFINER   a `#define X` in our own sources, including the `#ifndef X / #define X` self-default idiom.
  SETTER    a `-DX` or `X=` in a script, CMake file, or checked-in doc that drives a build.

A switch with neither is reported. That is NOT automatically "delete it": the five found when this was written
split into two kinds, and the distinction is the useful part --

  * an unreachable FEATURE (PPU_MAXREG caps registers to raise occupancy; LOWBIT_QMODE selects ScaleOnly;
    QUACTLIZE_DENSE_ONLY drops formats, and ci/check_format_table_buildable.py's docstring CITES it as a thing a
    build can do). These are coverage gaps. Deleting them hides the gap instead of closing it.
  * an UNRUN EXPERIMENT whose comment already states what each outcome would mean (PPU_PACKED_PAIR=0,
    PPU_F16X2_EARLYCLOBBER=0, PPU_F16X2_NOFTZ=1 -- all three bisecting the same open rowC failure). Deleting one
    destroys a pending answer, and each is a single build. They are in .coord/BOX.md as E1/E2/E3.
  * a TOOL (GEMV_GATE_FAST narrows an axis while iterating and says "build the FULL matrix before trusting a
    result"; PPU_B_CHUNK_BISECT exists BECAUSE PPU_B_CHUNK=2 once shipped a debug mode inside the flag that turns
    the feature on). Deleting the second invites back the mistake that separating it fixed.

So this prints the inventory and fails; a human decides which kind each is.

"NOBODY WIRED IT" IS A STATEMENT ABOUT THE BUILD, NOT ABOUT THE SWITCH -- and the first draft of this file got
that wrong. It sorted the eight by reading their names and comments, put five in a "debugging knob -> delete"
bucket, and three of those five turned out to be experiments carrying written decision rules for an unresolved
numerical failure. The classification has to come from reading what the switch DOES and whether a measurement for
it is on file, which is the same rule this repository applies to everything else and which a name cannot answer.

WHAT IT DOES NOT DO. It cannot follow an interpolated define (`-DFOO_${x}=1`), so it looks for that pattern and
reports it as a hole rather than pretending the scan was complete -- the same rule as
ci/check_generated_include_edges.py, and for the same reason: extracting the name statically IS the mechanism.
It also does not know whether a live switch is still WORTH having; PPU_PACKED_* is ten macros around one scale
path and several encode decisions that have long since shipped, which is a merge question, not a reachability one.
"""
import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIRS = ["quactlize/include", "quactlize/csrc", "benchmarks", "tests", "dev"]
SRC_EXT = {".cu", ".cuh", ".hpp", ".h", ".inl", ".inc", ".cpp"}
# Files that can plausibly turn a switch on. Docs count: BOX.md and the HANDOFFs are how a box command is handed
# over, so a macro that only a doc names is reachable by a human, which is the property under test.
SETTER_EXT = {".sh", ".py", ".md", ".txt", ".in", ".cmake", ".toml", ".yml"}
# Only switches this repository owns. actlize's own macros are the vendor's to justify.
OWNED = re.compile(r"^(PPU_|QUACTLIZE_|LOWBIT_|MOE_|GEMV_|BENCH_|SK_|DENSE_)")
COND = re.compile(
    r"^\s*#\s*(?:if|elif)\s+(?:!\s*)?defined\s*\(?\s*([A-Z_][A-Z0-9_]*)"
    r"|^\s*#\s*ifn?def\s+([A-Z_][A-Z0-9_]*)"
    r"|^\s*#\s*(?:if|elif)\s+\(?\s*([A-Z_][A-Z0-9_]*)\s*[!=<>]", re.M)
INTERPOLATED = re.compile(r"-D([A-Z_]+)\$\{")

# Intentional exceptions. Each needs a reason, and the reason has to be about REACHABILITY -- "we might want it
# later" is what produced the list this gate exists to shorten.
ALLOWED = {
    # DATED DEBT, NOT ACCEPTED EXCEPTIONS. These eight were found the day this gate was written (2026-08-07) and
    # each is an open decision, not a justified switch -- they are here so the gate's job becomes "no NEW
    # unreachable switch appears" while the existing set stays visible and owned. Task #40 tracks the merge.
    # An entry may only leave this dict by being deleted or wired, never by being re-justified.
    "GEMV_GATE_FAST":         "claude  -- TOOL: narrows the gs axis while iterating; its own comment says build the FULL matrix before trusting a result. Needs a documented invocation, not deletion",
    "LOWBIT_QMODE":           "claude  -- '=1 selects ScaleOnly'; a COVERAGE GAP, the MoE sweep cannot reach ScaleOnly",
    "QUACTLIZE_DENSE_ONLY":   "claude  -- ci/check_format_table_buildable.py's docstring cites it as a thing a build can do",
    "PPU_B_CHUNK_BISECT":     "codex   -- TOOL: exists BECAUSE PPU_B_CHUNK=2 shipped a debug mode inside the feature flag. Deleting it invites that back",
    "PPU_F16X2_EARLYCLOBBER": "codex   -- UNRUN EXPERIMENT E2 in .coord/BOX.md: was \"=&r\" the rowC fix, or did the failure merely go away across four commits?",
    "PPU_F16X2_NOFTZ":        "codex   -- UNRUN EXPERIMENT E3 in .coord/BOX.md: does ppu.fma.rtte.f16x2 flush its subnormal input? One build; a build failure kills the hypothesis for free",
    "PPU_MAXREG":             "codex   -- caps registers to raise occupancy; an unreachable OCCUPANCY LEVER, and "
                              "occupancy is exactly what the M=1 42.2% question turns on",
    "PPU_PACKED_PAIR":        "codex   -- UNRUN EXPERIMENT E1 in .coord/BOX.md: does PAIR=0 restore rowC? Indicts or exonerates the packed f16x2 arithmetic, which has zero local coverage",
}


def sources():
    for d in SRC_DIRS:
        p = ROOT / d
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix in SRC_EXT and "third_party/" not in str(f):
                    yield f


def setter_files():
    # SKIP THIS FILE. Its docstring necessarily contains the patterns it searches for -- `-DFOO_${x}=1` as the
    # example of what it cannot follow -- so the first run reported its own prose as a hole in the scan. A probe
    # that matches itself is a check that cannot report the truth about anything else.
    me = pathlib.Path(__file__).resolve()
    for f in sorted(ROOT.rglob("*")):
        if not f.is_file() or ".git/" in str(f) or "third_party/" in str(f) or f.resolve() == me:
            continue
        if f.suffix in SETTER_EXT or f.name == "build.sh":
            yield f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print every switch, not only the unreachable ones")
    args = ap.parse_args()

    consumers, definers = {}, {}
    for f in sources():
        text = f.read_text(errors="replace")
        for m in COND.finditer(text):
            name = m.group(1) or m.group(2) or m.group(3)
            if name and OWNED.match(name):
                consumers.setdefault(name, set()).add(str(f.relative_to(ROOT)))

    # A PARSE THAT FINDS NOTHING MUST NOT PASS. The regex is pinned to four preprocessor spellings; if the tree
    # stops matching it, that is the scan breaking, not the switches disappearing.
    if len(consumers) < 10:
        print(f"[switch-macros] ERROR: found only {len(consumers)} switch(es). This tree had 45 when the gate was "
              f"written; the conditional patterns no longer describe it. Not reporting a verdict.")
        return 1

    for f in sources():
        text = f.read_text(errors="replace")
        for name in consumers:
            if re.search(rf"^\s*#\s*define\s+{re.escape(name)}\b", text, re.M):
                definers.setdefault(name, set()).add(str(f.relative_to(ROOT)))

    setters, interpolated = {}, set()
    for f in setter_files():
        text = f.read_text(errors="replace")
        interpolated.update(INTERPOLATED.findall(text))
        for name in consumers:
            if re.search(rf"-D{re.escape(name)}\b", text) or re.search(rf"(?:^|[\s\"'=]){re.escape(name)}\s*=", text):
                setters.setdefault(name, set()).add(str(f.relative_to(ROOT)))

    unreachable = sorted(n for n in consumers
                         if n not in definers and n not in setters and n not in ALLOWED)

    if args.list:
        print(f"{'switch':<28} {'uses':>5}  reachable via")
        for n in sorted(consumers):
            via = sorted(definers.get(n, set()) | setters.get(n, set()))
            short = ", ".join(pathlib.Path(v).name for v in via)[:60] or "*** NOTHING ***"
            print(f"{n:<28} {len(consumers[n]):>5}  {short}")
        print()

    if interpolated:
        print("[switch-macros] ERROR: a define is built by interpolation, which this cannot follow:")
        for n in sorted(interpolated):
            print(f"    -D{n}${{...}}")
        print("    Extracting the name statically is the whole mechanism. Not reporting a verdict.")
        return 1

    if unreachable:
        print(f"[switch-macros] FAIL: {len(unreachable)} switch(es) that nothing can turn on:")
        for n in unreachable:
            print(f"    {n}")
            for c in sorted(consumers[n]):
                print(f"        used by {c}")
        print("    Each is one of two things and they need opposite fixes: a debugging knob nobody wired (delete "
              "it), or a FEATURE with no way in (wire it -- deleting hides the gap). Add it to ALLOWED with a "
              "reachability reason only if it is genuinely neither.")
        return 1

    print(f"[switch-macros] PASS: {len(consumers)} owned switch(es), every one reachable by a #define or a -D")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
