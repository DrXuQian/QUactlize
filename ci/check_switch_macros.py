#!/usr/bin/env python3
"""A BUILD SWITCH WITH NO RECORDED WAY IN IS ONE NOBODY WILL USE.

    python3 ci/check_switch_macros.py            verdict
    python3 ci/check_switch_macros.py --list     the whole inventory, live ones included

WHAT THIS COST, and it is not hypothetical. Three macros in this tree mean "shrink A's 15/16 padding at small M":
PPU_A_PACK (a binary-wide #if that WINS over everything downstream; the only survivor), PPU_A_CPASYNC (a policy default that an
explicit per-row value silently overrides), and the ACR column of the tactic table. docs/CHECKPOINT.md lists the
first two on one line as if they were one thing. On 2026-08-06 a measurement was filed as "compact A at capacity 1
is 45% slower" and could not be attributed afterwards, because the run's A provider was never witnessed and two of
the three spellings were reachable only by remembering which was passed. One idea, three spellings, and the cost
landed on a number in BACKTEST rather than on a compile.

WHAT IT CHECKS, AND WHAT IT DOES NOT. It reports every owned switch with no RECORDED way in. It does NOT report
unreachable switches, and the first version of this file said it did -- which was wrong, and wrong in the
direction this repository keeps paying for: I measured "no literal -DX or X= in any tracked file" and named the
result "nobody can turn it on".

build.sh forwards PPU_DEFS generically -- CMakeLists.txt.in:11 turns every NAME=VALUE into -DNAME=VALUE for the
device compile -- so ANY macro in a file that goes through build.sh can be set today, with no code change:

    PPU_DEFS=LOWBIT_QMODE=1 TARGET=test_lowbit_moe_bench ./build.sh

Every one of the eight found on 2026-08-07 is reachable that way. What none of them had was a written-down
invocation, and that is still worth failing on: an option nobody has recorded is one nobody will find, and
BACKTEST's D7 is what that costs. Writing the command into a doc is a real fix, not a workaround -- the pattern
below matches `PPU_DEFS=NAME=` because that IS the way in.

Two ways count as recorded:

  DEFINER   a `#define X` in our own sources, including the `#ifndef X / #define X` self-default idiom.
  SETTER    a `-DX` or `X=` in a script, CMake file, or checked-in doc that drives a build.

A switch with neither is reported. That is NOT automatically "delete it": the five found when this was written
split into two kinds, and the distinction is the useful part --

  * a FEATURE with no recorded invocation (PPU_MAXREG caps registers to raise occupancy; QUACTLIZE_DENSE_ONLY
    drops formats, and ci/check_format_table_buildable.py's docstring CITES it as a thing a build can do). Write
    the command down. LOWBIT_QMODE was in this bucket and left it that way on 2026-08-07 -- its header now
    carries `PPU_DEFS=LOWBIT_QMODE=1 ...`, which is the whole fix.
  * a RECORDED DIAGNOSTIC. PPU_PACKED_PAIR=0 is the surviving example: it has a build command and a historical
    rowC result, so it needs no ALLOWED exemption. The earlyclobber experiment was already run and retired; the
    guessed arithmetic `.noftz` switch was deleted after independent end-to-end evidence excluded its FTZ theory.
  * a TOOL. PPU_B_CHUNK_BISECT exists BECAUSE PPU_B_CHUNK=2 once shipped a debug mode inside the flag that turns
    the feature on, so deleting it invites back the mistake that separating it fixed. GEMV_GATE_FAST was the
    other example here and is now the counter-example: it narrowed an axis while telling you to "build the FULL
    matrix before trusting a result" -- a switch that shortens a run whose result you are then told not to trust
    -- and it was DELETED (tests/test_gemv_lowbit.cu:25). Narrowing that axis needs no macro; GEMV_GS_LIST on the
    command line is the documented way. So "a tool" is a reason to keep a switch only when the tool is one
    somebody would actually reach for.

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
    # DATED DEBT, NOT ACCEPTED EXCEPTIONS. Items leave by deletion or by acquiring a recorded build route.
    # E1/E2 were incorrectly labelled UNRUN after their results were already recorded; E1 is wired as a recurrence
    # diagnostic and E2 was retired. E3's guessed `.noftz` grammar was deleted because end-to-end rowC evidence,
    # not assembler acceptance, answered the relevant physical question. Only unresolved debt stays here.
    # An entry may only leave this dict by being deleted or wired, never by being re-justified.
    "PPU_B_CHUNK_BISECT":     "codex   -- TOOL: exists BECAUSE PPU_B_CHUNK=2 shipped a debug mode inside the feature flag. Deleting it invites that back",
    "PPU_MAXREG":             "codex   -- caps registers to raise occupancy; an unreachable OCCUPANCY LEVER, and "
                              "occupancy is exactly what the M=1 42.2% question turns on",
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

    # THE FILE'S OWN HEADER COUNTS, in exactly one form. This gate's FAIL message has always advised
    # "PPU_DEFS=NAME=VALUE in the file's own header or a doc" while the scan above looked only at setter_files()
    # -- .sh/.py/.md/... -- so a header comment in the .cu or .hpp that OWNS the switch satisfied the advice and
    # not the check. LOWBIT_QMODE is the precedent the docstring cites as the model fix, and it passes today only
    # because ci/local_gates.py happens to also carry -DLOWBIT_QMODE=1; its header line was never what counted.
    #
    # ONLY the literal `PPU_DEFS=NAME=` is accepted here, not the general setter pattern. Turning the general one
    # loose on sources would make every switch self-satisfying: the SETTER regex's `NAME\s*=` arm matches
    # `#if !defined(NAME) || NAME == 12`, i.e. the guard the switch is FOR. A build-command prefix cannot appear
    # by accident in C++.
    for f in sources():
        text = f.read_text(errors="replace")
        for name in consumers:
            if f"PPU_DEFS={name}=" in text:
                setters.setdefault(name, set()).add(str(f.relative_to(ROOT)) + " (header invocation)")

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

    # A STALE EXEMPTION IS INVISIBLE DEBT, and until 2026-08-11 nothing could see one. ALLOWED was consumed at
    # exactly one place -- as a negative filter in `unreachable` above -- so an entry whose switch had since been
    # DELETED stayed in the dict forever, and the dict's own rule ("an entry may only leave by being deleted or
    # wired") could be satisfied without anyone noticing the entry had not left. GEMV_GATE_FAST was in exactly
    # that state: tests/test_gemv_lowbit.cu:25 says "it is DELETED: nothing in the tree could define it" while
    # the dict still carried it as open debt owned by claude.
    #
    # The worse case is the one this prevents: a future switch reusing a name that is already exempted would be
    # waved through on the strength of a note about something else.
    stale = sorted(n for n in ALLOWED if n not in consumers)
    if stale:
        print(f"[switch-macros] FAIL: {len(stale)} ALLOWED entr(ies) name a switch that no longer exists:")
        for n in stale:
            print(f"    {n}    -- recorded as: {ALLOWED[n].strip()[:90]}")
        print("    The switch left by being DELETED, which the dict's own rule permits -- but the ENTRY has to")
        print("    leave with it. Delete these lines. Keeping them exempts a name, not a decision, so the next")
        print("    switch to reuse the name inherits an exemption written about something else.")
        return 1

    if unreachable:
        print(f"[switch-macros] FAIL: {len(unreachable)} owned switch(es) with no recorded way in "
              f"(PPU_DEFS can set any of them; nothing writes the command down):")
        for n in unreachable:
            print(f"    {n}")
            for c in sorted(consumers[n]):
                print(f"        used by {c}")
        print("    Fix by RECORDING the invocation (PPU_DEFS=NAME=VALUE in the file's own header or a doc),")
        print("    by deleting the switch, or by promoting it to a real option. Do not add to ALLOWED without a reason.")
        return 1

    print(f"[switch-macros] PASS: {len(consumers)} owned switch(es), every one reachable by a #define or a -D")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
