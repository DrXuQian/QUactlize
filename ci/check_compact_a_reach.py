#!/usr/bin/env python3
"""The tactic space's compact-A clause and the collectives' witnesses must say the same thing.

    python3 ci/check_compact_a_reach.py

WHAT THE ORDER IS, AND WHY IT CANNOT STAY A SENTENCE. Compact A shrinks the A smem tile from TileM rows to a
capacity. A is fp16 activations, so nothing about it depends on the weight format -- the fold is B-only
(ppu_mma_aiu_fold.hpp: "the fold lives only in the load layer ... no fold-specific compute code at all") and the
two-plane collective differs only by a second B plane. It is restricted today because the reader was written into
ONE of the three collectives; the other two hardwire

    static constexpr int compact_a_rows = 0;

and ppu_tactic_space.hpp's common_compact_a_supported() refuses capacities for exactly those configs.

Deleting that clause before porting the collectives does not fail. It emits rows labelled `acr=1` whose collective
IGNORES the field and allocates TileM*TK*2 anyway, so the sweep times a capacity-0 kernel under a capacity-1 name
-- which is precisely how D7 ("compact A at capacity 1 is 45% slower") became unattributable. A number that is
wrong in a way no output can reveal is the expensive kind.

So the sequence is: port the collectives, then delete the clause. This makes that sequence checkable instead of
remembered, in BOTH directions:

  clause dropped, collective still hardwires 0  -> the emitter would produce mislabelled rows. FAIL.
  collective honours a capacity, clause remains -> silent coverage loss: legal rows nothing can emit. FAIL.

WHAT IT READS. Each collective's `static constexpr int compact_a_rows = <expr>;` -- a literal 0 means "cannot
compact", anything else (kACompactRows) means "honours the policy" -- and the conjuncts of
common_compact_a_supported()'s return expression. Both are single, unambiguous lines today; if either stops being
one the check says so rather than guessing, because a scan that cannot find its subject must not report agreement.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPACE = ROOT / "quactlize" / "include" / "ppu_tactic_space.hpp"
COLL = ROOT / "quactlize" / "include" / "quactlize_extensions" / "cutlass" / "gemm" / "collective"

# The three collectives, and the clause in common_compact_a_supported() that exists for each. The mapping is the
# claim under test: "two-plane cannot compact" is spelled `high_bits == 0`, "folded cannot compact" is spelled
# `artifact_low_fold(c) == 1`.
COLLECTIVES = [
    ("quactlize_mma_mixed_input.hpp", None,
     "the ordinary unfolded one-plane collective, where the reader lives"),
    ("ppu_mma_aiu_mixed_input_2plane.hpp", "high_bits",
     "two-plane formats (Q3_K / Q5_K / Q6_K)"),
    ("ppu_mma_aiu_fold.hpp", "artifact_low_fold",
     "folded low planes (i1/i2 at the TileKs where they fold)"),
]
WITNESS = re.compile(r"^\s*static\s+constexpr\s+int\s+compact_a_rows\s*=\s*([^;]+);", re.M)
SUPPORTED = re.compile(r"constexpr\s+bool\s+common_compact_a_supported\s*\([^)]*\)\s*\{\s*return\s+([^;]+);", re.S)


def main() -> int:
    for p in (SPACE, COLL):
        if not p.exists():
            print(f"[compact-a-reach] SKIP: {p.relative_to(ROOT)} does not exist")
            return 0

    m = SUPPORTED.search(SPACE.read_text())
    if not m:
        print("[compact-a-reach] ERROR: could not find common_compact_a_supported's return expression in "
              f"{SPACE.name}. The predicate is the subject of this check; without it there is no verdict.")
        return 1
    clause_text = " ".join(m.group(1).split())

    problems = []
    seen = []
    for name, clause, what in COLLECTIVES:
        f = COLL / name
        if not f.is_file():
            print(f"[compact-a-reach] ERROR: {name} is missing; the mapping this check tests no longer describes "
                  f"the tree. Not reporting a verdict.")
            return 1
        w = WITNESS.search(f.read_text())
        if not w:
            print(f"[compact-a-reach] ERROR: {name} has no `static constexpr int compact_a_rows = ...;`. Every "
                  f"collective carries that witness; its absence means this check cannot see the truth.")
            return 1
        expr = w.group(1).strip()
        hardwired_zero = (expr == "0")
        seen.append((name, expr))
        if clause is None:
            if hardwired_zero:
                problems.append((name, "hardwires 0 but is THE collective the reader lives in; the feature has no "
                                       "implementation anywhere and every capacity row would be mislabelled"))
            continue
        guarded = clause in clause_text
        if hardwired_zero and not guarded:
            problems.append((name, f"hardwires compact_a_rows = 0, but common_compact_a_supported no longer "
                                   f"mentions `{clause}` -- so the emitter will produce acr>0 rows for {what} "
                                   f"whose collective ignores the field and allocates TileM*TK*2. The sweep would "
                                   f"time a capacity-0 kernel under a capacity-1 name. PORT FIRST."))
        if not hardwired_zero and guarded:
            problems.append((name, f"witnesses `{expr}`, so it honours a capacity -- but "
                                   f"common_compact_a_supported still excludes it via `{clause}`. Those rows are "
                                   f"legal and nothing can emit them: silent coverage loss. DROP THE CLAUSE."))

    if problems:
        print(f"[compact-a-reach] FAIL: {len(problems)} disagreement(s) between the clause and the witnesses:")
        for name, why in problems:
            print(f"    {name}\n        {why}")
        print(f"    common_compact_a_supported returns: {clause_text}")
        return 1

    impl = [n for n, e in seen if e != "0"]
    print(f"[compact-a-reach] PASS: {len(impl)} of {len(seen)} collective(s) honour a capacity "
          f"({', '.join(impl)}), and the tactic clause excludes exactly the rest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
