#!/usr/bin/env python3
"""Every #include the CMake generator WRITES into a generated source must resolve to a file in the tree.

    python3 ci/check_generated_include_edges.py

THE BLIND SPOT THIS COVERS. quactlize/csrc/CMakeLists.txt.in generates one .cu per unit at configure time:

    file(WRITE "${_src}.in"
      "#define UNIT_TM ${_tm}\\n"  ...
      "#include \\"moe_splitk_unit.inc\\"\\n")

No tracked file includes moe_splitk_unit.inc, so an include closure over the repository cannot see the edge --
the file that would have carried it does not exist until cmake runs. This is not a gap in the closure tool; it
is what the closure tool models. It reads what a compiler reads, and this build has a code-generation step in
between, so the real graph is generator -> generated source -> include.

WHY IT NEEDED A CHECK. Assembling main format by format on 2026-08-06, moe_bench_unit.inc and gemv_perf_unit.inc
were each missed by the closure and each added by hand afterwards -- twice, the same way. A third would have gone
the same way: nothing in the tooling could have said otherwise, and remembering is not a mechanism.

WHAT IT DOES, and it is deliberately narrow: it reads the generator, extracts the include NAMES that appear
inside file(WRITE) string literals, and checks each resolves. The names are derived, not listed, so a fourth
generated unit is covered the day it is written.

WHAT IT DOES NOT DO. It does not run cmake, so it cannot see an include produced by string interpolation
(`#include "${_thing}.inc"`), nor one added through configure_file, nor whether the generated source compiles.
Today the generator uses neither of those -- checked, not assumed -- and this fails loudly if the pattern stops
matching rather than reporting that zero edges are fine.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
GENERATORS = [ROOT / "quactlize" / "csrc" / "CMakeLists.txt.in",
              ROOT / "quactlize" / "csrc" / "CMakeLists.txt",
              ROOT / "CMakeLists.txt"]
# `"#include \"name\"` inside a CMake string. The backslash-escaped quotes are what distinguish a line the
# generator WRITES from one CMake merely mentions.
EDGE = re.compile(r'#include\s+\\"([^"\\]+)\\"')
# An interpolated include would defeat the extraction silently, so its presence is an error rather than a miss.
INTERPOLATED = re.compile(r'#include\s+\\"[^"\\]*\$\{')
SEARCH_DIRS = ["benchmarks", "quactlize/include", "quactlize/csrc", "tests", "dev/fold_derivation"]
# A GENERATED INCLUDE MAY NAME A FILE THE GENERATOR ITSELF WRITES, into the BUILD directory. That is not a
# missing file -- it is the normal case for per-band metadata:
#
#     file(WRITE "${_MOE_GEN_DIR}/moe_bench_band.inc.in" ...)
#     execute_process(... copy_if_different "${_MOE_GEN_DIR}/moe_bench_band.inc.in"
#                                           "${_MOE_GEN_DIR}/moe_bench_band.inc")
#
# The source tree will never contain moe_bench_band.inc, and it should not: it is per-configure output. This gate
# reported it as "resolve to nothing" and told the reader "the generator will emit a source that cannot compile",
# which is a claim about the code for a defect in the check -- the same shape as the units gate counting row
# fields and the provenance gate covering one table of six. It has been failing since at least 66a5994.
#
# So an edge is satisfied by EITHER a file in the tree OR a write of that same name by a generator. The second
# clause is derived from the CMake text, not a list of exemptions: adding a new generated .inc needs no edit here,
# and deleting the file(WRITE) that produces one turns its edge red again.
WRITTEN = re.compile(r'(?:file\s*\(\s*WRITE|copy_if_different)[^)]*?/([A-Za-z0-9_.\-]+\.inc)(?:\.in)?["\s)]')


def main() -> int:
    present = [g for g in GENERATORS if g.is_file()]
    if not present:
        print("[generated-edges] SKIP: no CMake generator found")
        return 0

    edges, interpolated = [], []
    for g in present:
        text = g.read_text(errors="replace")
        for m in EDGE.finditer(text):
            edges.append((g, m.group(1)))
        if INTERPOLATED.search(text):
            interpolated.append(g)

    if interpolated:
        print("[generated-edges] ERROR: a generated #include is built by interpolation, which this cannot follow:")
        for g in interpolated:
            print(f"    {g.relative_to(ROOT)}")
        print("    Extracting the name statically is the whole mechanism. Not reporting a verdict.")
        return 1

    if not edges:
        print("[generated-edges] ERROR: found no generated #include at all. The generator writes units that "
              "include a .inc; zero matches means the pattern stopped describing it, not that the edges are gone.")
        return 1

    generated_names = set()
    for g in present:
        for m in WRITTEN.finditer(g.read_text(errors="replace")):
            generated_names.add(m.group(1))

    missing = []
    for g, name in edges:
        in_tree = any((ROOT / d / name).is_file() for d in SEARCH_DIRS)
        if not in_tree and name not in generated_names:
            missing.append((g, name))

    if missing:
        print(f"[generated-edges] FAIL: {len(missing)} generated include(s) resolve to nothing:")
        for g, name in missing:
            print(f"    {name}   written by {g.relative_to(ROOT)}")
        print("    The generator will emit a source that cannot compile. Either the file is missing from this")
        print("    tree -- the usual cause when a subset was assembled from an include closure -- or the")
        print("    generator names it wrongly.")
        return 1

    print(f"[generated-edges] PASS: {len(edges)} generated include(s) from {len(present)} generator(s), all resolve"
          + (f" ({len(generated_names)} of them written by the generator into the build dir)" if generated_names else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
