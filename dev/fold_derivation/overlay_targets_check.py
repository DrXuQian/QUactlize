#!/usr/bin/env python3
"""RUN CMAKE OVER THE ACTUAL OVERLAY -- locally, with no PPU SDK and no compiler.

THE INVARIANT NOBODY ENFORCED. CMakeLists.txt names sources; build.sh decides which files are copied into actlize's
example tree. Those two lists live in different files, in different languages, and nothing compared them until cmake
ran on the accelerator. A source CMake names but the overlay lacks is a CONFIGURE-time error, so it fails for EVERY
target at once and the message names only whichever file it tripped over first -- which is how two box round-trips
went into rediscovering one problem, one file name at a time.

WHY THIS RUNS CMAKE INSTEAD OF PARSING. The first version of this check parsed the CMake registrations with a regex
and reconstructed build.sh's globs in python. Both models were wrong, and review found the errors rather than use:

  * It applied the shared extension list to dev/, which build.sh's dev glob does not (no .cpp), so a registration
    naming a dev/*.cpp read as present while the overlay would lack it.
  * It parsed `_src_dirs=(...)` with python .split(), so writing the entries quoted -- semantically identical bash --
    made the check fail.
  * It discarded ${...} arguments, so it could not see that the CMake guard mishandled generated sources: those
    expand to ABSOLUTE build-tree paths, the guard prefixed them with CMAKE_CURRENT_SOURCE_DIR, and
    test_moe_splitk_bench was silently skipped. The check reported all 33 targets present while cmake failed.
  * CMake command names are case-insensitive and tolerate formatting a regex does not, so a reformatted registration
    could vanish from the check while remaining valid CMake.

Every one of those is a second implementation of something that already exists. So now: build.sh prints its own
manifest (`--print-overlay`), this materialises it, and CMAKE parses the CMakeLists. The only模型 left is the stub
prelude, which is small and whose failure is loud.

WHAT THE STUBS DO. actlize supplies cutlass_example_add_executable; here it records the target name. The per-target
commands are overridden with no-ops, because a real target needs a real toolchain and the question is which targets
get CREATED, not what flags they carry. CMake makes the originals available as _target_compile_options and so on, so
nothing is lost -- they are simply not exercised.

    ./overlay_targets_check.py            check
    ./overlay_targets_check.py --list     print every target cmake created
    ./overlay_targets_check.py --keep     leave the scratch overlay for inspection
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("QUACTLIZE_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))

# Targets other things break without. run_batch.sh builds exactly these: seven splitk variants, four q4k gates, two
# verifier builds. do_check needs the latter two; do_perf needs splitk. Each is asserted to have been CREATED, not
# merely parsed -- the previous check consulted this list only while iterating registrations its regex happened to
# match, so a registration that failed to parse took its requirement with it and the summary still printed "3
# required target(s) present", a constant rather than a measurement.
REQUIRED = ("test_moe_splitk_bench", "test_q4k_packed_gemm", "test_moe_grouped_verify")

STUB = r"""
cmake_minimum_required(VERSION 3.16)
project(quactlize_overlay_check NONE)

# actlize's example helper. Records the name so the caller can assert which targets exist.
function(cutlass_example_add_executable NAME)
  add_custom_target(${NAME})
  file(APPEND "${CMAKE_BINARY_DIR}/created_targets.txt" "${NAME}\n")
endfunction()

# The per-target commands, overridden to no-ops. add_custom_target does not accept them, and giving every target a
# real compiled library would need a toolchain this check exists to avoid. CMake keeps the originals reachable as
# _target_compile_options etc., so this hides nothing it could otherwise have checked: the question here is WHICH
# TARGETS GET CREATED, and flags are the box's business.
function(target_compile_options)
endfunction()
function(target_compile_definitions)
endfunction()
function(target_include_directories)
endfunction()
function(target_link_libraries)
endfunction()
function(set_target_properties)
endfunction()
function(add_dependencies)
endfunction()

set(CUTLASS_EXAMPLES_COMMON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
"""


def manifest():
    """The files build.sh would copy, straight from build.sh. Lines are either a path, or DEST|path where DEST is a
    subdirectory (ending in /) or a rename."""
    r = subprocess.run([os.path.join(ROOT, "build.sh"), "--print-overlay"],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"  [FAIL] overlay_targets: build.sh --print-overlay failed: {r.stderr.strip()[:300]}")
    return [l for l in r.stdout.splitlines() if l.strip()]


def materialise(entries, dst):
    n = 0
    for line in entries:
        if "|" in line:
            where, src = line.split("|", 1)
            if where.endswith("/"):
                os.makedirs(os.path.join(dst, where), exist_ok=True)
                shutil.copy(src, os.path.join(dst, where, os.path.basename(src)))
            else:
                shutil.copy(src, os.path.join(dst, where))
        else:
            shutil.copy(line, os.path.join(dst, os.path.basename(line)))
        n += 1
    return n


def main():
    show_all, keep = "--list" in sys.argv, "--keep" in sys.argv
    if not shutil.which("cmake"):
        print("  [SKIP] overlay_targets: cmake not installed")
        return 0

    entries = manifest()
    if not entries:
        print("  [FAIL] overlay_targets: build.sh --print-overlay listed nothing")
        return 1
    if not any(l.startswith("CMakeLists.txt|") for l in entries):
        print("  [FAIL] overlay_targets: the manifest has no CMakeLists.txt -- nothing would configure")
        return 1

    work = tempfile.mkdtemp(prefix="quactlize-overlay-")
    src, bld = os.path.join(work, "src"), os.path.join(work, "b")
    os.makedirs(src)
    try:
        n = materialise(entries, src)
        cml = os.path.join(src, "CMakeLists.txt")
        with open(cml) as f:
            body = f.read()
        with open(cml, "w") as f:
            f.write(STUB + "\n" + body)

        r = subprocess.run(["cmake", "-S", src, "-B", bld], capture_output=True, text=True)
        created = []
        tf = os.path.join(bld, "created_targets.txt")
        if os.path.exists(tf):
            created = [l.strip() for l in open(tf) if l.strip()]

        skipped = [l for l in r.stdout.splitlines() if "ppu_w4a16: skipping" in l]
        if r.returncode != 0:
            print("  [FAIL] overlay_targets: cmake could not configure the overlay:")
            for l in [x for x in (r.stdout + r.stderr).splitlines() if "Error" in x or "error" in x][:6]:
                print(f"           {l.strip()}")
            for l in skipped[:4]:
                print(f"           {l.strip()}")
            return 1

        missing_req = [t for t in REQUIRED if t not in created]
        if missing_req:
            print(f"  [FAIL] overlay_targets: cmake configured, but did NOT create {', '.join(missing_req)}")
            print("           benchmarks/run_batch.sh builds these; without them the gates cannot run.")
            for l in skipped[:6]:
                print(f"           {l.strip()}")
            return 1

        if show_all:
            for t in created:
                print(f"    created  {t}")
            for l in skipped:
                print(f"    {l.strip()}")
        print(f"  [ok]   overlay_targets: cmake configured {n} overlaid file(s) and created {len(created)} target(s); "
              f"all {len(REQUIRED)} required present, {len(skipped)} optional skipped")
        return 0
    finally:
        if keep:
            print(f"           scratch overlay kept at {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
