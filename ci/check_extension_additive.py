#!/usr/bin/env python3
"""Every type actlize_extensions defines must be a NEW name, or a specialisation on a name quactlize owns.

    python3 ci/check_extension_additive.py

WHAT PROPERTY THIS IS. quactlize_actlize.hpp includes actlize's umbrella unmodified and adds our headers after
it, so both trees are in one translation unit. That is only legal while our headers ADD. Two ways it can stop:

  REDEFINITION   we define `struct X` at a namespace scope where actlize already defines `struct X`. The compiler
                 catches this one -- loudly, at the first build.
  AMBIGUITY      we declare a partial specialisation of a VENDOR template whose constraint overlaps a vendor
                 specialisation's. The compiler catches it only for the argument lists actually instantiated, so a
                 table that never exercises the overlap builds green and the overlap ships.

The second is why this exists. On 2026-08-06 quactlize's builder claimed
{KernelTmaWarpSpecialized*MixedInput, PerCol, Gs128, Gs64} on CollectiveBuilder -- every one of them also claimed
by actlize's builder -- and nothing said so, because quactlize's copy REPLACED actlize's in the include list
instead of joining it. The moment the two coexisted it was six ambiguous specialisations.

WHAT IT CHECKS, and where each is decided:

  1. No name defined at namespace scope in actlize_extensions is also defined in actlize, UNLESS it is a
     specialisation (`struct X<...>`), which is an addition by construction.
  2. Every specialisation of a template actlize also specialises is declared in SPECIALISATION_OWNERS with the
     reason its constraint cannot overlap. That list is short on purpose: it is the set of places where "additive"
     is a claim about constraints rather than about names, and the compiler cannot check it for you.

WHAT IT CANNOT DO. It does not parse C++. It matches definitions with a regex over namespace-scope lines, so a
name introduced by a macro, or a specialisation whose head is split unusually, is invisible to it. It is a net
under the design, not a proof of it -- the compile is still the arbiter for anything it does catch.
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXT = ROOT / "quactlize" / "include" / "actlize_extensions"
ACTLIZE = ROOT / "third_party" / "actlize"

# Templates that BOTH trees specialise. Each needs an argument-level reason the two constraints are disjoint,
# because names alone cannot show it and the compiler only sees the instantiations a build happens to reach.
SPECIALISATION_OWNERS = {
    "CollectiveMma":
        "keyed on the mainloop policy: ours are MainloopQuactlizeMixedInput / MainloopPPUAiuFold / "
        "MainloopPPUAiuMixedInput2Plane, none of which actlize defines, so no argument list matches both",
    "CollectiveBuilder":
        "keyed on KernelScheduleType: ours claims {Gs32, KernelAiuFold<...>} and actlize's claims "
        "{KernelTmaWarpSpecialized*MixedInput, PerCol, Gs128, Gs64}. Disjoint by construction -- Gs32 and "
        "KernelAiuFold are quactlize's own tags. See the comment on that enable_if for why every quactlize "
        "caller reaches ours even at gs=128",
    "MixGemmNumericArrayConverter":
        "keyed on the source element type: ours are uint8_t, uint2b_t and uint1b_t; actlize's are int8_t and "
        "int4b_t. No pair collides",
    "fold_schedule_traits":
        "quactlize's own primary template, declared in quactlize_dispatch_policy.hpp; actlize has no such name",
    "MainloopQuactlizeMixedInput": "quactlize's own primary template",
    "MainloopPPUAiuFold": "quactlize's own primary template",
    "MainloopPPUAiuMixedInput2Plane": "quactlize's own primary template",
    "MixGemm_AIU_Operand":
        "quactlize's copy lives in namespace quactlize_detail, not detail, so it shares no scope with actlize's",
    "contig_elems": "quactlize's own, in quactlize_detail::aiu_detail",
    "ScaleSwizzleFor": "quactlize's own primary template",
    "MaybeScaleSwizzle": "quactlize's own primary template",
    "a_provider_schedule_traits":
        "quactlize owns both this primary template and its KernelAiuPackedA specialisation; actlize defines "
        "neither name, so no vendor argument list can overlap",
}

# A DELIBERATELY SMALL SCANNER, not a C++ parser. It tracks `namespace A::B {` and brace depth so a name is
# recorded as `A::B::Name` rather than bare `Name`, and it accepts a declaration head that begins on a
# continuation line -- both of which the first version got wrong and both of which mattered on the first run:
# `get_tiled_mma` exists in actlize's `detail` and quactlize's `quactlize_detail` and is NOT a clash, while
# `MixGemm_AIU_Operand` is written `> struct MixGemm_AIU_Operand;` after a multi-line template head and was
# invisible. A namespace-blind name check reports the safe one and misses the dangerous one.
NS_OPEN = re.compile(r"^\s*namespace\s+([A-Za-z_][\w:]*)\s*\{")
NS_ANON = re.compile(r"^\s*namespace\s*\{")
# The head may begin on a continuation line: `> struct MixGemm_AIU_Operand;` follows a multi-line template head,
# and matching only column 0 hid it entirely.
#
# THE TRAILING CHARACTER IS NOT COSMETIC. `class ElementAOptionalTuple,` inside a template parameter list looks
# exactly like a definition to a name-only pattern, and the version without this reported 24 clashes -- every one
# of them a template PARAMETER that both trees naturally spell the same way. A checker that cries wolf 24 times
# gets its output skimmed, which is the same end state as a checker that finds nothing. So the name must be
# followed by `{` (definition), `;` (forward declaration) or `<` (specialisation) -- never `,`, `=` or `>`.
# `$` needs a lookahead, not a free pass: `  class KernelScheduleType` as the LAST template parameter has no
# trailing comma either, and matching a bare end-of-line accepted it. A bare head is a definition only when the
# next line opens the body.
DEFN = re.compile(r"^\s*(?:>\s*)?(?:struct|class)\s+([A-Za-z_]\w*)\s*(?:(<)|(\{)|(;)|$)")


def scan(path):
    """-> (definitions, specialisations), qualified, for namespace-scope declarations only.

    NAMESPACE SCOPE IS `depth == ns_depth`, and getting that wrong is not a near miss. The first version compared
    depth against the depth BEFORE the namespace's own brace, so the test was false everywhere inside any
    namespace and the scan reported one definition for each tree -- and passed. A checker that finds nothing
    reports the same thing as a checker that finds nothing wrong.
    """
    ns, defs, specs = [], set(), set()
    depth = ns_depth = 0
    src = path.read_text(errors="replace").splitlines()
    for idx, raw in enumerate(src):
        line = raw.split("//")[0]
        m = NS_OPEN.match(line) or NS_ANON.match(line)
        if m:
            ns.append(m.group(1) if m.re is NS_OPEN else "<anon>")
            depth += 1
            ns_depth += 1
            continue
        if depth == ns_depth:
            d = DEFN.match(line)
            if d:
                is_spec, has_brace, has_semi = bool(d.group(2)), bool(d.group(3)), bool(d.group(4))
                bare = not (is_spec or has_brace or has_semi)
                nxt = src[idx + 1].lstrip() if idx + 1 < len(src) else ""
                if not bare or nxt.startswith("{"):
                    (specs if is_spec else defs).add("::".join(ns + [d.group(1)]))
        opened, closed = line.count("{"), line.count("}")
        depth += opened - closed
        while ns and depth < ns_depth:
            ns.pop()
            ns_depth -= 1
    return defs, specs


def collect(files):
    defs, specs, owner = set(), set(), {}
    for f in files:
        d, s = scan(f)
        for n in d:
            owner.setdefault(n, []).append(f)
        defs |= d
        specs |= s
    return defs, specs, owner


def self_test(ext_files, act_files) -> str:
    """Facts established by hand on 2026-08-06. A scan that cannot reproduce them is not evidence of anything.

    Every gate in this repo that silently stopped checking did so by finding nothing and reporting no problem, so
    this refuses to report at all until the scan recovers things known to be there.
    """
    ours, our_specs, _ = collect(ext_files)
    theirs, their_specs, _ = collect(act_files)
    musts = [
        ("cutlass::gemm::collective::quactlize_detail::MixGemm_AIU_Operand" in ours,
         "quactlize's MixGemm_AIU_Operand, whose head is on a continuation line"),
        ("cutlass::gemm::collective::detail::MixGemm_AIU_Operand" in theirs,
         "actlize's MixGemm_AIU_Operand, the one it must NOT be confused with"),
        ("cutlass::gemm::MainloopQuactlizeMixedInput" in ours,
         "our renamed mainloop policy"),
        (any(n.endswith("CollectiveMma") for n in our_specs),
         "our CollectiveMma specialisations"),
        (any(n.endswith("CollectiveBuilder") for n in our_specs),
         "our CollectiveBuilder specialisation"),
        (len(ours) >= 10, f"at least 10 extension definitions (found {len(ours)})"),
        (len(theirs) >= 100, f"at least 100 actlize definitions (found {len(theirs)})"),
        # NEGATIVE CONTROLS. These are template PARAMETER names, spelled identically in both trees because both
        # descend from the same cutlass signatures. Counting them as definitions produced 24 false clashes.
        (not any(n.endswith(("::ElementAOptionalTuple", "::TileShapePair_", "::StrideA_", "::kContinous",
                             "::KernelScheduleType"))
                 for n in ours),
         "template parameter names must NOT be counted as definitions (KernelScheduleType is the trailing one, "
         "with no comma after it)"),
    ]
    missing = [why for ok, why in musts if not ok]
    return "" if not missing else "; ".join(missing)


def main() -> int:
    if not EXT.is_dir():
        print(f"[additive] SKIP: {EXT} does not exist")
        return 0
    if not (ACTLIZE / "include").is_dir():
        print("[additive] SKIP: actlize is not checked out")
        return 0

    ext_files = sorted(p for p in EXT.rglob("*") if p.suffix in (".h", ".hpp", ".inl"))
    act_files = sorted(p for p in (ACTLIZE / "include").rglob("*") if p.suffix in (".h", ".hpp", ".inl"))
    if not ext_files:
        print(f"[additive] SKIP: no extension headers under {EXT}")
        return 0

    broken = self_test(ext_files, act_files)
    if broken:
        print(f"[additive] ERROR: the scanner cannot find things known to be present: {broken}")
        print("           Not reporting a verdict -- fix scan() first. A scan that sees nothing passes silently.")
        return 1

    ours, our_specs, our_where = collect(ext_files)
    theirs, _, their_where = collect(act_files)
    failed = False

    clashes = sorted(ours & theirs)
    if clashes:
        failed = True
        print(f"[additive] FAIL: {len(clashes)} qualified name(s) defined in BOTH trees at namespace scope:")
        for n in clashes:
            print(f"    {n}")
            print(f"      ours:    {', '.join(str(p.relative_to(EXT)) for p in our_where[n])}")
            print(f"      actlize: {', '.join(str(p.relative_to(ACTLIZE)) for p in their_where[n])}")
        print("    A shared name is a redefinition when both headers are included, which quactlize_actlize.hpp")
        print("    now does. Rename ours, or move it into a quactlize_* namespace.")
    else:
        print(f"[additive] PASS names: {len(ours)} extension definition(s), none shared with actlize's {len(theirs)}")

    undeclared = sorted({n.rsplit("::", 1)[-1] for n in our_specs} - set(SPECIALISATION_OWNERS))
    if undeclared:
        failed = True
        print(f"[additive] FAIL: {len(undeclared)} specialised template(s) with no ownership reason recorded:")
        for n in undeclared:
            print(f"    {n}")
        print("    If actlize also specialises it, state in SPECIALISATION_OWNERS why the constraints cannot")
        print("    overlap. Overlap is ambiguity, and it only surfaces for argument lists a build reaches.")
    else:
        print(f"[additive] PASS specialisations: all {len(our_specs)} carry an ownership reason")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
