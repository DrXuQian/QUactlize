#!/usr/bin/env python3
"""Every file naming a quactlize_extensions type must have the defining header in its include closure.

    python3 ci/check_owned_symbol_includes.py

WHY THIS EXISTS, and it is not hypothetical. On 2026-08-06 the extraction moved our collectives, policies and
converters out of the actlize fork. 17 files that included actlize's umbrella were repointed at quactlize's. That
was incomplete in a way nothing local noticed: 29 more files reach the same types through DIRECT includes of the
specific cutlass headers, never through any umbrella. They kept compiling right up until actlize stopped
defining the type, and then the box build died on

    ppu_group_schedule.hpp:28: no type named 'KernelAiuMultistageMixedInputFinegrainedGs32' in '::cutlass::gemm'
    xplane_offline.hpp:94:     no member named 'MixGemmMmaPermK' in namespace 'cutlass'

-- two of twenty-nine, because a compiler stops. Fixing what the log shows and rebuilding is a twenty-nine round
trip on a machine that takes minutes per build.

THE SYMBOLS ARE DERIVED, NOT LISTED. They come from scanning quactlize_extensions for namespace-scope
definitions, so a type added there is covered the day it is added. A hand-written list would have to be updated
by the same person who forgot the include, at the same moment.

THE CLOSURE IS REAL, not a grep for the include line. `gcc -M -MG` preprocesses the file with the project's -I
paths and reports every header actually reached, so a file that gets the definition transitively -- through
quactlize_actlize.hpp, or through another of our headers -- passes without naming it directly, which is the
normal and correct case for most consumers.

TWO FILTERS, both of which it needed on its first run and neither of which is optional. COMMENTS are stripped
before matching: fold_traits.hpp explains a failure mode "or get_tiled_mma degenerates", and xplane_offline.hpp
notes "MixGemm_AIU_Operand's own arithmetic" -- neither uses the type. And a name the file DEFINES ITSELF is
skipped: ppu_tactic_space.hpp has an `Exclusion::ScaleCopyCoverage` that has nothing to do with the metadata
policy's, and demanding an include there would be wrong as well as noisy. Four of twelve first-run hits were one
of these two, and a checker with a one-in-three false rate gets its output skimmed.

WHAT IT CANNOT DO. It has no PPU SDK, so missing SDK headers are stubbed and `-MG` treats anything still missing
as generated rather than failing. Comment stripping is textual, so a name inside a string literal still counts.
"""
import os
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXT = ROOT / "quactlize" / "include" / "quactlize_extensions"
ACTLIZE = ROOT / "third_party" / "actlize"
SEARCH_ROOTS = ("quactlize", "benchmarks", "tests", "dev")
SUFFIXES = (".cu", ".cuh", ".hpp", ".h")

DEFN = re.compile(r"^\s*(?:>\s*)?(?:struct|class)\s+([A-Za-z_]\w*)\s*(?:(<)|(\{)|(;)|$)")
NS_OPEN = re.compile(r"^\s*namespace\s+([A-Za-z_][\w:]*)\s*\{")
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def code_of(path: pathlib.Path) -> str:
    """The file with comments removed -- a name discussed in prose is not a use."""
    text = BLOCK_COMMENT.sub(" ", path.read_text(errors="replace"))
    return "\n".join(line.split("//")[0] for line in text.splitlines())


def declares(path: pathlib.Path, name: str) -> bool:
    """Does this file define the name itself? Then the owner header is irrelevant to it."""
    return re.search(rf"^\s*(?:>\s*)?(?:struct|class|enum(?:\s+class)?)\s+{re.escape(name)}\b",
                     code_of(path), re.M) is not None or \
           re.search(rf"^\s*{re.escape(name)}\s*,\s*$", code_of(path), re.M) is not None

# Names too generic to key on: a hit would be noise rather than a missing include.
IGNORE = {"CollectiveMma", "CollectiveBuilder", "MixGemmNumericArrayConverter", "contig_elems", "NoZero"}


def ambiguous_names(files) -> set:
    """Names ALSO defined outside quactlize_extensions -- this checker cannot speak about them.

    It matches bare names, so it cannot tell `gguf_scale::GroupScale` (defined in gguf_scale_decode.hpp) from
    `cutlass::gguf_packed::GroupScale` (ours). Two unrelated types with one spelling is a fact about this
    codebase, not a defect to route around, so the honest move is to drop the name and say how many were
    dropped -- claiming coverage of a name whose uses cannot be attributed would be worse than the gap.
    """
    out = set()
    for p in files:
        out |= defined_names(p)
    return out


def defined_names(path: pathlib.Path) -> set:
    """Namespace-scope type names this header defines (specialisations excluded -- they need no separate include)."""
    names, depth, ns_depth = set(), 0, 0
    src = path.read_text(errors="replace").splitlines()
    for idx, raw in enumerate(src):
        line = raw.split("//")[0]
        if NS_OPEN.match(line) or re.match(r"^\s*namespace\s*\{", line):
            depth += 1
            ns_depth += 1
            continue
        if depth == ns_depth:
            d = DEFN.match(line)
            if d and not d.group(2):
                bare = not (d.group(3) or d.group(4))
                nxt = src[idx + 1].lstrip() if idx + 1 < len(src) else ""
                if not bare or nxt.startswith("{"):
                    names.add(d.group(1))
        depth += line.count("{") - line.count("}")
        if ns_depth and depth < ns_depth:
            ns_depth = depth
    return names


def closure(path: pathlib.Path, stub: str) -> set:
    incs = ["-I" + stub,
            "-I" + str(ROOT / "quactlize" / "include"),
            "-I" + str(ROOT / "third_party" / "actlize" / "include"),
            "-I" + str(ROOT / "third_party" / "actlize" / "tools" / "util" / "include"),
            "-I" + str(ROOT / "benchmarks")]
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as tf:
        tf.write(f'#include "{path}"\n')
        tmp = tf.name
    try:
        # DEFINE THE DEVICE-COMPILER MACROS. Without them a `#if defined(__HGGCCC__)` include is invisible to the
        # preprocessor, so a header that correctly reaches its definition only on the device pass reads as missing
        # -- and "fixing" it by hoisting the include out of the guard is how gguf_vecdot.hpp stopped building
        # against stock cutlass on 2026-08-06. A closure that cannot see the guarded arm cannot judge it.
        r = subprocess.run(["gcc", "-std=c++17", "-x", "c++", "-M", "-MG",
                            "-D__HGGCCC__", "-D__CUDACC__", *incs, tmp],
                           capture_output=True, text=True)
    finally:
        os.unlink(tmp)
    return set(re.split(r"[\s\\]+", r.stdout))


def make_stub(tmpdir: str) -> str:
    """Empty stand-ins for the PPU SDK headers, so the preprocessor can walk the whole graph off-device."""
    stub = pathlib.Path(tmpdir) / "stub"
    for h in ("acComplex.h", "driver_types.h", "hggc.h", "hggc_bf16.h", "hggc_fp16.h",
              "hggc_runtime.h", "hggc_runtime_api.h", "hggc/std/utility", "hggc/std/type_traits"):
        p = stub / h
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
    return str(stub)


def main() -> int:
    if not EXT.is_dir():
        print(f"[owned-includes] SKIP: {EXT} does not exist")
        return 0
    if subprocess.run(["which", "gcc"], capture_output=True).returncode:
        print("[owned-includes] SKIP: no gcc to compute include closures")
        return 0

    owner = {}
    for h in sorted(p for p in EXT.rglob("*") if p.suffix in (".h", ".hpp", ".inl")):
        for n in defined_names(h) - IGNORE:
            owner.setdefault(n, h)
    if len(owner) < 5:
        print(f"[owned-includes] ERROR: only {len(owner)} owned type(s) found under quactlize_extensions; the "
              f"scan is broken and a green verdict would mean nothing")
        return 1

    files = [p for r in SEARCH_ROOTS for p in (ROOT / r).rglob("*")
             if p.suffix in SUFFIXES and EXT not in p.parents and str(EXT) not in str(p)]

    shared = ambiguous_names(files) & set(owner)
    for n in shared:
        del owner[n]

    missing = []
    with tempfile.TemporaryDirectory(prefix="quactlize-owned-") as td:
        stub = make_stub(td)
        for p in sorted(files):
            text = code_of(p)
            used = {n for n in owner
                    if re.search(rf"\b{re.escape(n)}\b", text) and not declares(p, n)}
            if not used:
                continue
            reached = closure(p, stub)
            for n in sorted(used):
                want = str(owner[n].relative_to(ROOT / "quactlize" / "include"))
                if not any(want in c for c in reached):
                    missing.append((p.relative_to(ROOT), n, want))

    if missing:
        seen, shown = set(), 0
        print(f"[owned-includes] FAIL: {len(missing)} name(s) used without the defining header in the closure:")
        for f, n, want in missing:
            if (f, want) in seen:
                continue
            seen.add((f, want))
            shown += 1
            if shown <= 20:
                print(f"    {f}\n        names {n}, needs #include \"{want}\"")
        if shown > 20:
            print(f"    ... and {shown - 20} more (file, header) pairs")
        return 1

    checked = sum(1 for p in files if any(re.search(rf"\b{re.escape(n)}\b", code_of(p)) for n in owner))
    note = f"; {len(shared)} name(s) skipped as ambiguous ({', '.join(sorted(shared))})" if shared else ""
    print(f"[owned-includes] PASS: {checked} file(s) name one of {len(owner)} quactlize types, and every one "
          f"reaches its defining header{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
