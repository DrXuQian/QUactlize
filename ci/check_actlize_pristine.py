#!/usr/bin/env python3
"""actlize must carry NO quactlize work: no owned symbol, and no file changed outside the fix allow-list.

    python3 ci/check_actlize_pristine.py            # both properties
    python3 ci/check_actlize_pristine.py --symbols  # just the symbol property (works mid-migration)

WHAT THIS IS FOR. quactlize's extensions are ADDITIVE: quactlize_actlize.hpp includes actlize's umbrella
unmodified and layers our headers on top, and that is only sound while every specialisation we declare is keyed
on a dispatch tag we own. A quactlize collective specialised on an ACTLIZE tag compiles perfectly and silently
takes over the vendor path for actlize's own callers -- the failure is a wrong kernel, not a diagnostic. Nothing
else in the tier can see that, because both halves are individually valid C++.

THE TWO PROPERTIES, and why neither alone is enough:

  SYMBOLS  No name in OWNED appears anywhere under third_party/actlize. This is the one that catches the
           dangerous direction -- our policy tags, our collectives, our scale format leaking back into the
           vendor tree, where they would be built into actlize's own examples and outlive any decision we made.

  FILES    The set of files differing from upstream v1.0.0 is exactly ALLOWED. This catches what SYMBOLS cannot:
           an in-place edit to an existing actlize specialisation that introduces no new name at all. The
           converter rewrite was precisely that shape -- 679 lines added and 13 replaced, and the 13 were the
           body of upstream's int8->fp16 path. A symbol scan sees nothing wrong with it.

THE ALLOW-LIST HAS TWO TIERS, and they answer to different standards. A FIX has to be right about actlize --
nvcc/EDG portability, true whether or not quactlize exists. An EXTENSION has to be harmless to actlize -- it
widens a vendor facility unreachable from outside the file, and stays backwards compatible for actlize's own
callers. Anything expressible as an addition belongs in quactlize_extensions and in NEITHER tier. Adding a file
to either is a claim of the corresponding kind and should be argued in the commit message, not typed in quietly.

IT COMPARES THE WORKING TREE, not HEAD. What gets compiled is what is on disk, and an uncommitted edit to a
vendor file is exactly as damaging as a committed one -- more so, because it survives no review. A dirty actlize
tree therefore fails this gate, which is the intended behaviour and not a bug to work around.

WHAT THIS CANNOT TELL YOU. It compares against the v1.0.0 tag in the submodule. It cannot detect that upstream
itself moved, and it says nothing about whether our headers are CORRECT -- only that they are elsewhere.
"""
import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ACTLIZE = ROOT / "third_party" / "actlize"
BASELINE = "v1.0.0"

# Names quactlize owns. A tag here appearing inside actlize means the separation has rotted.
#
# The three PPU-prefixed ones look like actlize names and are not: `git diff v1.0.0` shows them arriving in the
# +130 lines we added to dispatch_policy.hpp. They keep the historical spelling so the rename stays one commit;
# what makes them ours is that upstream has no such template, which is the thing this list records.
OWNED = [
    "MainloopQuactlizeMixedInput",       # our collective's tag; the rename that made the extension additive
    "MainloopPPUAiuFold",                # N-fold mainloop, ours, no upstream counterpart
    "MainloopPPUAiuMixedInput2Plane",    # bit-plane Q3/Q5/Q6 mainloop, ours
    "KernelAiuFold",                     # the fold schedule
    "fold_schedule_traits",              # and its traits hook
    "KernelAiuMultistageMixedInputFinegrainedGs32",  # gs=32 schedule (Q4_0/Q4_1/Q4_K-as-AWQ)
    "GgufPackedScale",                   # the packed k-quant scale format
    "quactlize_extensions",              # any include path into our tree
]

# Files actlize is allowed to differ from v1.0.0 in. TWO KINDS, kept apart because they answer to different
# standards: a FIX has to be right about actlize, an EXTENSION has to be harmless to actlize.
#
# FIXES are corrections to actlize, true whether or not quactlize exists. They came in as cd17c2b9 on the
# nvcc-portability branch off v1.0.0, and every one is nvcc/EDG rejecting what clang accepted.
FIXES = {
    "include/cute/arch/copy_ppu.hpp":
        "#114: ppu001's six assembler-rejected plain-LDSM atoms are deleted at the C++ call site; legacy helper "
        "templates carry a dependent static_assert, while the ppu0015 tc02 API and bodies stay unchanged",
    "include/cutlass/arch/memory_ppu.h":
        "#114: ppu001's six assembler-rejected explicit ldsm specializations are deleted at the C++ call site; "
        "the ppu0015 tc02 specializations stay unchanged",
    "include/cute/arch/util.hpp":
        "cd17c2b9: CUTE_DEVICE -> CUTE_HOST_DEVICE; nvcc rejects the host call",
    "include/cutlass/arch/mma_ppu.h":
        "cd17c2b9: CUTLASS_HOST_DEVICE -> CUTLASS_DEVICE where the body calls __hfma2",
    "include/cutlass/epilogue/collective/builders/ppu_builder.inl":
        "cd17c2b9: trailing comma in a template argument list",
    "include/cutlass/gemm/collective/builders/ppu_mma_builder.inl":
        "cd17c2b9: missing `template` disambiguator on a dependent member template",
    "include/cutlass/gemm/collective/ppu_mma_aiu_multistage_batch_array_overlap_prologue.hpp":
        "cd17c2b9: {0,0,0} -> dim3(0,0,0); nvcc will not brace-init dim3 here",
    "include/cutlass/gemm/config/gemm_configs.hpp":
        "cd17c2b9: trailing commas in template argument lists",
    "include/cutlass/gemm/kernel/ppu_tile_scheduler_stream_k.hpp":
        "cd17c2b9: trailing commas in template argument lists",
    "include/cutlass/fast_numeric_conversion_for_mix_gemm.h":
        "c48cb105: the int8 converter's ppu.prmt/ppu.sub behind __HGGC_ARCH__, with a plain-C++ arm. It is a FULL "
        "specialisation, so the body reaches ptxas whether or not it is called, and nvcc cannot assemble `ppu`",
}

# EXTENSIONS widen a vendor facility that quactlize needs and that cannot be reached from outside the file. Each
# one is BACKWARDS COMPATIBLE for actlize's own callers -- that is the bar, and it is why each entry states the
# mechanism rather than the motive. They are candidates to send upstream to T-Head; until then they are the
# honest cost of the extension model and they are enumerated so the cost stays visible.
#
# Anything expressible as an addition does NOT belong here. The four collectives, the dispatch policies, the
# converter widths and the NoZero marker were all in this position on 2026-08-06 and all moved out.
EXTENSIONS = {
    "include/cute/arch/mma_ppu0010.hpp":
        "30 added, 0 removed: one ppu001-only m8n16 raw atom; every existing atom is byte-for-byte unchanged",
    "include/cute/atom/mma_traits_ppu0010.hpp":
        "20 added, 0 removed: the additive register/layout traits for the new ppu001 m8n16 atom",
    "include/cute/arch/copy_aiu_base.hpp":
        "13 added, 0 removed: new AIU descriptor fields, purely additive",
    "include/cute/arch/copy_ppu0010_aiu.hpp":
        "PPU0010_TSM_LD_SWZL gains `int CubePitch = 0` -- DEFAULTED, so every existing spelling still matches; "
        "plus new PPU0010_AIU_LOAD arms for 2- and 1-bit elements, which are additions",
    "include/cute/atom/copy_traits_ppu0010_aiu.hpp":
        "Copy_Traits follows the CubePitch signature above; without it the trait no longer matches the atom",
    "include/cutlass/detail/collective.hpp":
        "deduce_mixed_width_dtype's guard widened from index 2 to 3 for the second B bit plane. Strictly "
        "permissive: the body already returned void out of range, so 0/1/2 are unchanged",
    "include/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input.hpp":
        "enable_if gains `&& !isGroupProblemShape_v<ProblemShape_>` so the grouped shape can select a different "
        "specialisation; narrows this arm only where another one now exists",
    "include/cutlass/gemm/config/gemm_operands.hpp":
        "adds a defaulted shape-aware selector alongside the existing type-only selector; only exact ppu001 "
        "TileM=WarpM=8 chooses m8n16, so all existing shapes and every other architecture retain their old atom",
}

# A FILE CAN BE BOTH, and until 2026-08-11 saying so lost information silently. `{**FIXES, **EXTENSIONS}` is a
# merge: a key present in both keeps only the EXTENSIONS note and the FIX's reason disappears, with no error and
# no change in the count. The two dicts answer to different standards -- a fix has to be right about actlize, an
# extension has to be harmless to it -- so a file carrying both owes BOTH justifications, not the later one.
#
# MIXED is where such a file goes, and its note has to name both halves. The first case is arriving now:
# ppu_tile_scheduler_stream_k.hpp already holds a trailing-comma FIX and is about to gain a defaulted template
# parameter (the Stream-K minimum k-stripe), which is an extension. Landing that by leaning on the file's
# existing FIXES entry would hide a new change behind an old allowance -- which is the same shape as a stale
# ALLOWED entry in ci/check_switch_macros.py exempting a name rather than a decision.
MIXED: dict = {}

_collisions = sorted((set(FIXES) & set(EXTENSIONS)) | (set(MIXED) & (set(FIXES) | set(EXTENSIONS))))
if _collisions:
    raise SystemExit(
        "[actlize-pristine] FAIL: these paths appear in more than one allow-list, and the merge below would "
        "silently keep one reason and drop the other:\n    " + "\n    ".join(_collisions) +
        "\nA file that is both a fix and an extension belongs in MIXED, with a note naming both halves.")

ALLOWED = {**FIXES, **EXTENSIONS, **MIXED}

SOURCE_SUFFIXES = (".h", ".hpp", ".inl", ".cu", ".cuh", ".cc", ".cpp")


def git(*args):
    return subprocess.run(["git", "-C", str(ACTLIZE), *args], capture_output=True, text=True)


def check_symbols() -> list:
    """-> list of (symbol, path, line, text) for every OWNED name found inside actlize."""
    findings = []
    for name in OWNED:
        # git grep so submodule-ignored build output cannot produce phantom hits, and so this is fast.
        r = git("grep", "-n", "--fixed-strings", name, "--", "*.h", "*.hpp", "*.inl", "*.cu", "*.cuh")
        if r.returncode not in (0, 1):
            print(f"[actlize-pristine] git grep failed for {name}: {r.stderr.strip()}")
            sys.exit(1)
        for line in r.stdout.splitlines():
            path, _, rest = line.partition(":")
            lineno, _, text = rest.partition(":")
            findings.append((name, path, lineno, text.strip()[:90]))
    return findings


def check_files() -> tuple:
    """-> (unexpected, missing) file paths relative to the actlize root."""
    r = git("rev-parse", "--verify", f"{BASELINE}^{{commit}}")
    if r.returncode:
        print(f"[actlize-pristine] SKIP: no {BASELINE} tag in the submodule; cannot establish the baseline")
        sys.exit(0)
    # BASELINE against the WORKING TREE. Using HEAD here would pass a tree with a vendor file edited and not
    # yet committed -- which is the state every such leak passes through, and the one a build actually sees.
    r = git("diff", "--name-only", BASELINE, "--")
    if r.returncode:
        print(f"[actlize-pristine] git diff failed: {r.stderr.strip()}")
        sys.exit(1)
    changed = {p for p in r.stdout.split() if p.endswith(SOURCE_SUFFIXES)}
    return sorted(changed - set(ALLOWED)), sorted(set(ALLOWED) - changed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", action="store_true",
                    help="check only the symbol property; use while the file move is in flight")
    a = ap.parse_args()

    if not (ACTLIZE / ".git").exists():
        print("[actlize-pristine] SKIP: third_party/actlize is not checked out")
        return 0

    failed = False
    leaks = check_symbols()
    if leaks:
        failed = True
        print(f"[actlize-pristine] FAIL: {len(leaks)} quactlize symbol reference(s) inside actlize:")
        for name, path, lineno, text in leaks[:25]:
            print(f"    {path}:{lineno}  {name}   {text}")
        if len(leaks) > 25:
            print(f"    ... and {len(leaks) - 25} more")
        print("    These belong in quactlize/include/quactlize_extensions/. Moving them is the fix; adding them")
        print("    to OWNED's exceptions is not -- there are none, deliberately.")
    else:
        print(f"[actlize-pristine] PASS symbols: none of the {len(OWNED)} owned names appear in actlize")

    if not a.symbols:
        unexpected, missing = check_files()
        if unexpected:
            failed = True
            print(f"[actlize-pristine] FAIL: {len(unexpected)} file(s) differ from {BASELINE} without being fixes:")
            for p in unexpected:
                print(f"    {p}")
            print("    Either the change is quactlize's and belongs in quactlize_extensions, or it is a genuine")
            print("    actlize correction and belongs in ALLOWED with the reason spelled out.")
        if missing:
            print(f"[actlize-pristine] NOTE: {len(missing)} allow-listed file(s) no longer differ from "
                  f"{BASELINE} -- the fix was upstreamed or reverted, so drop them from ALLOWED:")
            for p in missing:
                print(f"    {p}")
        if not unexpected:
            # THE TALLY HAS TO ADD UP TO len(ALLOWED), or a category can be added and not counted -- the same
            # silent under-report the merge itself used to produce. The assert is here rather than in a test
            # because this line is the only place a reader sees the number.
            assert len(FIXES) + len(EXTENSIONS) + len(MIXED) == len(ALLOWED), \
                "a category is missing from the tally, so the printed count understates the vendor delta"
            mixed = f" + {len(MIXED)} both" if MIXED else ""
            print(f"[actlize-pristine] PASS files: {len(FIXES)} fix(es) + {len(EXTENSIONS)} vendor extension(s)"
                  f"{mixed} = {len(ALLOWED)} file(s) differing from {BASELINE}, all allow-listed")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
