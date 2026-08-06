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

    missing = []
    for g, name in edges:
        if not any((ROOT / d / name).is_file() for d in SEARCH_DIRS):
            missing.append((g, name))

    if missing:
        print(f"[generated-edges] FAIL: {len(missing)} generated include(s) resolve to nothing:")
        for g, name in missing:
            print(f"    {name}   written by {g.relative_to(ROOT)}")
        print("    The generator will emit a source that cannot compile. Either the file is missing from this")
        print("    tree -- the usual cause when a subset was assembled from an include closure -- or the")
        print("    generator names it wrongly.")
        return 1

    print(f"[generated-edges] PASS: {len(edges)} generated include(s) from {len(present)} generator(s), all resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
