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

A switch with neither is reported. That is NOT automatically "delete it": the initially exempted switches split
into three kinds, and the distinction is the useful part --

  * a FEATURE with no recorded invocation. PPU_MAXREG and QUACTLIZE_DENSE_ONLY were in this bucket; their owning
    files now carry the exact `PPU_DEFS=... TARGET=... ./build.sh` command, which is the whole reachability fix.
  * a RECORDED DIAGNOSTIC. PPU_PACKED_PAIR=0 is the surviving example: it has a build command and a historical
    rowC result, so it needs no ALLOWED exemption. The earlyclobber experiment was already run and retired; the
    guessed arithmetic `.noftz` switch was deleted after independent end-to-end evidence excluded its FTZ theory.
  * a TOOL. PPU_B_CHUNK_BISECT exists BECAUSE PPU_B_CHUNK=2 once shipped a debug mode inside the flag that turns
    the feature on, so deleting it invites back the mistake that separating it fixed. Its owning collective now
    records BOTH required defines and an input that can see scale-register errors. GEMV_GATE_FAST was the
    counter-example: it shortened a run whose own comment required the full matrix, so it was deleted.

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
# A gate can also invent a define that no source consumes.  The original inventory starts from #if consumers, so
# DENSE_DRYRUN=1 lived only in ci/local_gates.py and was invisible: it was forwarded all the way to hgcc, where no
# token read it.  Restrict this second scan to quoted gate arguments and require complete absence outside the gate;
# ordinary CMake/env variables remain valid when their name is consumed by build.sh/CMake/Python rather than #if.
GATE_ASSIGNMENT = re.compile(
    r"[\"'](?:-D)?((?:PPU_|QUACTLIZE_|LOWBIT_|MOE_|GEMV_|BENCH_|SK_|DENSE_)[A-Z0-9_]*)=")

# Temporary reachability debt only. This is intentionally empty. The stale check below rejects both permitted exits
# if their entry is forgotten: a deleted switch and a switch that has acquired a real definer/setter.
ALLOWED = {}


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


def unresolved_switches(consumers, definers, setters, allowed):
    return sorted(n for n in consumers if n not in definers and n not in setters and n not in allowed)


def stale_allowed(consumers, definers, setters, allowed):
    deleted = sorted(n for n in allowed if n not in consumers)
    resolved = sorted(n for n in allowed if n in consumers and (n in definers or n in setters))
    return deleted, resolved


def gate_only_dead_assignments(gate_text: str, external_text: str):
    """Quoted NAME= values in local_gates that have no identifier use anywhere else."""
    assigned = set(GATE_ASSIGNMENT.findall(gate_text))
    return sorted(name for name in assigned
                  if not re.search(rf"\b{re.escape(name)}\b", external_text))


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

    # GATE-ONLY DEFINES ARE A SEPARATE FAILURE SHAPE from an unwired source switch.  Build one external corpus from
    # executable/build inputs (not docs, which can mention a dead name without consuming it), then ask whether a
    # local-gate assignment names anything outside the gate that sets it.
    gate_path = ROOT / "ci/local_gates.py"
    gate_text = gate_path.read_text(errors="replace")
    external_paths = list(sources()) + [ROOT / "build.sh", ROOT / "CMakeLists.txt"]
    external_paths += sorted((ROOT / "quactlize/csrc").glob("*.in"))
    external_paths += sorted((ROOT / "ci").glob("*.py"))
    external_text = "\n".join(
        p.read_text(errors="replace") for p in external_paths
        if p.is_file() and p.resolve() != gate_path.resolve() and p.resolve() != pathlib.Path(__file__).resolve())
    gate_dead = gate_only_dead_assignments(gate_text, external_text)

    # Both directions are load-bearing: a planted gate-only define must red, and adding an actual consumer must
    # clear it.  Otherwise this check would merely special-case the historical spelling DENSE_DRYRUN.
    gate_control = "DENSE_GATE_ONLY_CONTROL"
    planted_gate = gate_text + f'\n("boxdry", "plant", "{gate_control}=1")\n'
    if gate_only_dead_assignments(planted_gate, external_text) != [gate_control]:
        print("[switch-macros] ERROR: planted gate-only define was not reported")
        return 1
    if gate_only_dead_assignments(planted_gate, external_text + f"\n#if defined({gate_control})\n"):
        print("[switch-macros] ERROR: planted gate-only define stayed dead after adding a consumer")
        return 1

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

    unreachable = unresolved_switches(consumers, definers, setters, ALLOWED)

    # In-memory classifier controls. ALLOWED debt is valid only while its consumer exists AND remains unresolved;
    # exercise that legal state plus every transition out of it. Definer- and setter-resolved controls are distinct
    # so dropping either half of stale_allowed's OR cannot hide behind a tree-selected name that has both routes.
    control = "PPU_SWITCH_GATE_CONTROL"
    planted_consumers = dict(consumers)
    planted_consumers[control] = {"<planted-consumer>"}
    if control not in unresolved_switches(planted_consumers, definers, setters, ALLOWED):
        print("[switch-macros] ERROR: the planted unwired consumer was not reported")
        return 1
    planted_allowed = {control: "open unresolved control"}
    if control in unresolved_switches(planted_consumers, definers, setters, planted_allowed) or \
       stale_allowed(planted_consumers, definers, setters, planted_allowed) != ([], []):
        print("[switch-macros] ERROR: a live unresolved ALLOWED entry was not accepted as temporary debt")
        return 1
    planted_setters = dict(setters)
    planted_setters[control] = {"<planted-setter>"}
    if control in unresolved_switches(planted_consumers, definers, planted_setters, ALLOWED):
        print("[switch-macros] ERROR: the planted consumer stayed unreachable after wiring")
        return 1
    if stale_allowed(consumers, definers, setters, {control: "deleted control"}) != ([control], []):
        print("[switch-macros] ERROR: the deleted-switch ALLOWED control did not become stale")
        return 1
    defined_control = next((n for n in sorted(consumers) if n in definers and n not in setters), None)
    setter_control = next((n for n in sorted(consumers) if n in setters and n not in definers), None)
    if defined_control is None or \
       stale_allowed(consumers, definers, setters, {defined_control: "defined control"}) != ([], [defined_control]):
        print("[switch-macros] ERROR: the definer-resolved ALLOWED control did not become stale")
        return 1
    if setter_control is None or \
       stale_allowed(consumers, definers, setters, {setter_control: "setter control"}) != ([], [setter_control]):
        print("[switch-macros] ERROR: the setter-resolved ALLOWED control did not become stale")
        return 1

    if args.list:
        print(f"{'switch':<28} {'uses':>5}  recorded via")
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

    # An exemption can go stale in BOTH permitted ways. The original check covered deletion only; wiring a macro
    # still left its old debt entry silently alive. In either case the entry now exempts a NAME, not an open decision,
    # so a future switch reusing that name would inherit somebody else's exception.
    stale_deleted, stale_resolved = stale_allowed(consumers, definers, setters, ALLOWED)
    if stale_deleted or stale_resolved:
        print(f"[switch-macros] FAIL: {len(stale_deleted) + len(stale_resolved)} stale ALLOWED entr(ies):")
        for n in stale_deleted:
            print(f"    {n}    -- switch was deleted; recorded as: {ALLOWED[n].strip()[:90]}")
        for n in stale_resolved:
            print(f"    {n}    -- switch now has a definer/setter; recorded as: {ALLOWED[n].strip()[:90]}")
        print("    Delete each entry together with the switch or when its invocation is recorded.")
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


    if gate_dead:
        print(f"[switch-macros] FAIL: {len(gate_dead)} define assignment(s) exist only in ci/local_gates.py; "
              "the build forwards them but no implementation consumes them:")
        for n in gate_dead:
            print(f"    {n}")
        print("    Remove the dead gate argument or add the real source/build consumer. A forwarded -D is not evidence "
              "that anything read it.")
        return 1

    print(f"[switch-macros] PASS: {len(consumers)} owned switch(es), every one has a recorded #define/-D route; "
          "six classifier controls plus two gate-only assignment controls passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
