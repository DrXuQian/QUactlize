#!/usr/bin/env python3
"""Does a given kernel ROUTE admit a given tile/warp geometry? Ask the compiler, with controls.

WHY THIS EXISTS AS A GATE AND NOT AS A ONE-OFF. "Is config X legal on route Y?" has now been asked four times
this week and answered three different ways, twice from reading type strings out of a device abort and once
from a host-side predicate that turned out to be quarantining its own measurement. The compiler knows. The only
reason not to ask it was that nobody had wired the question up.

THE CONTROLS ARE THE POINT. A probe that compiles a geometry and reports "no errors" is worthless unless you
know it CAN report errors -- the local tier has already shipped one gate whose accepted-noise baseline swallowed
a fatal error and reported green over a file the front end never finished parsing (see
dev/fold_derivation/syntax_baseline). So every run here compiles cases that MUST FAIL alongside the ones that
must pass, and a control that stops failing fails the gate louder than the subject does:

    a subject that unexpectedly fails  -> the route rejects a geometry we thought it took   (a finding)
    a control that unexpectedly passes -> THE PROBE IS BLIND, and every green above it is void

REQUIRES nvcc. The probe itself odr-uses the selected kernel types; product headers do not expose a gate-only
instantiation switch.

A CORRECTION THAT COST TWO ROUNDS AND IS THE REASON THE THIRD CONTROL EXISTS. The counts are identical with the
ordinary compile because probe_shapes() reads DenseKernel::SharedStorageSize and that already drags the
collective in. So the first two controls only prove
TYPE-LEVEL errors are visible -- and the failures we care about (the metadata assert(false) sites that became
static_asserts) live in DEVICE FUNCTION BODIES, which type completeness does not reach.

The plant control settles it by measurement rather than by argument: mirror actlize's include tree with symlinks,
replace one mainloop header with a copy carrying a static_assert INSIDE the device operator()'s body, and require
it to fire. The mirror is needed because ppu_include.hpp includes the mainloop with QUOTES from its own
directory, so a plain -I in front never gets consulted -- the first attempt at this control silently compiled the
real header and reported zero firings, which reads exactly like "bodies are not instantiated".
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "dev" / "dense_warp_probe.cu"
STUB = ROOT / "dev" / "fold_derivation" / "stub_inc"
ACT = ROOT / "third_party" / "actlize"

# Dropped as ENVIRONMENTAL, matching dev/fold_derivation/syntax_check.sh exactly rather than by fresh guesswork:
# CUTE_INLINE_CONSTANT resolves to `static constexpr` under the local stubs while the box takes the
# `static const __device__` branch. Neither can mask a real error -- a real one has a different message.
NOISE = re.compile(r'identifier "cute::(_|product)" is undefined in device code')
ERROR = re.compile(r": (error|fatal error|catastrophic error): ")

# (label, -D defines, expectation). The two rejects are not decoration; see the module docstring.
CASES = [
    # THE SUBJECT: docs/BACKTEST.md A1's geometry, (64,64,64) w64x32 s3 -- 2 warps, 211.33 us / 65.0% measured on
    # the GROUPED route. A four-warp minimum kept it out of the dense table until 2026-08-05; this case is what
    # established the route admits it, and it stays as the regression guard for that.
    ("dense w64x32 (2 warps, the recorded int4 winner)",
     ["-DPROBE_WM=64", "-DPROBE_WN=32"], "admits"),
    # THE CONTROL FOR THE SUBJECT: same everything, four warps. If this ever stops compiling the probe has drifted
    # away from what the dense bench builds and the subject's verdict means nothing.
    ("dense w32x32 (4 warps, the always-allowed control)",
     ["-DPROBE_WM=32", "-DPROBE_WN=32"], "admits"),
    # POSITIVE CONTROL 1 -- WarpM > TileM makes warpOnM zero and the builder divides by it. Fires in
    # gemm_operands.hpp and ppu_builder.inl, i.e. deep inside the instantiation rather than at parse time.
    ("WarpM(128) > TileM(64)",
     ["-DPROBE_WM=128", "-DPROBE_WN=32"], "rejects"),
    # POSITIVE CONTROL 2 -- a group size the schedule ladder has no rung for. Fires the collective builder's own
    # "Could not build a collective for given parameters", which is the failure shape a genuinely illegal tactic
    # would produce.
    ("group_size = 7 (no rung in the schedule ladder)",
     ["-DPROBE_GS=7"], "rejects"),
]


def compile_case(defs, extra_flags, include_root=None, quactlize_root=None):
    cmd = ["nvcc", "-std=c++17", "-arch=sm_80", "--expt-relaxed-constexpr",
           "-D__HGGCCC__", *defs, *extra_flags,
           "-I" + str(STUB), "-I" + str(include_root or (ACT / "include")),
           "-I" + str(ACT / "tools" / "util" / "include"),
           "-I" + str(ROOT / "tests"), "-I" + str(ROOT / "benchmarks"),
           "-I" + str(quactlize_root or (ROOT / "quactlize" / "include")), "-I" + str(ROOT / "dev"),
           "-cuda", "-o", "/dev/null", "-x", "cu", str(PROBE), "-Wno-deprecated-gpu-targets"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout + p.stderr
    real = [l for l in out.splitlines() if ERROR.search(l) and not NOISE.search(l)]
    return real, out


# THE MAINLOOP THE DENSE ROUTE ACTUALLY INSTANTIATES, which moved out of actlize on 2026-08-06. Planting into
# actlize's copy after that date fires zero times and reads as "device bodies are not instantiated" -- the exact
# false conclusion this control exists to prevent, now reachable by the control itself being stale. It is relative
# to quactlize/include, not actlize/include, and the mirror below follows.
MAINLOOP = "actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp"
PLANT_MSG = "PLANTED-IN-DEVICE-BODY"


def plant_control(tmp):
    """Require a static_assert planted INSIDE the mainloop's device operator() to fire.

    -> (ok, detail). This is the only control that speaks to function-body visibility; the other two are
    satisfied by type completeness alone, which is a weaker property than the one this gate's verdicts rest on.

    The symlink mirror is load-bearing, not tidiness: the umbrella includes the mainloop by a path that resolves
    inside quactlize/include, so a -I ahead of it is never consulted and a plain overlay silently compiles the
    ORIGINAL header. That produced a clean zero firings and read as "device bodies are not instantiated".
    """
    mirror = tmp / "mirror"
    if mirror.exists():
        subprocess.run(["rm", "-rf", str(mirror)], check=True)
    mirror.mkdir(parents=True)
    # cp -rs: symlink every file, materialise the directories, so one header can be swapped for pennies.
    QINC = ROOT / "quactlize" / "include"
    if subprocess.run(["cp", "-rs", str(QINC) + "/.", str(mirror)],
                      capture_output=True).returncode != 0:
        return None, "could not mirror quactlize/include (cp -rs unavailable?)"
    target = mirror / MAINLOOP
    target.unlink(missing_ok=True)
    lines = (QINC / MAINLOOP).read_text().splitlines(True)
    # The device entry point, found by its signature rather than by a line number that drifts with every edit.
    i = next((n for n, l in enumerate(lines) if "CUTLASS_DEVICE void" in l), None)
    if i is None:
        return None, f"no 'CUTLASS_DEVICE void' entry point found in {MAINLOOP}"
    while "{" not in lines[i]:
        i += 1
    lines.insert(i + 1, f'    static_assert(sizeof(FrgTensorC) == 0, "{PLANT_MSG}");\n')
    target.write_text("".join(lines))

    _, out = compile_case(["-DPROBE_WM=64", "-DPROBE_WN=32"], [], quactlize_root=mirror)
    n = out.count(PLANT_MSG)
    return (n > 0), f"planted device-body static_assert fired {n} time(s)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flags", default="", help="extra -D flags forwarded to every case")
    ap.add_argument("-v", "--verbose", action="store_true", help="print the first error line of every case")
    a = ap.parse_args()

    if not PROBE.is_file():
        print(f"[route-admits] ERROR: probe missing: {PROBE}")
        return 2
    if subprocess.run(["bash", "-c", "command -v nvcc"], capture_output=True).returncode != 0:
        # A COMPILE GATE WITHOUT A COMPILER MUST NOT REPORT GREEN. syntax_check.sh learned this the hard way:
        # with nvcc off PATH it printed "clean (0 known-noise lines, 0 new)" and exited 0 for ten of the tier's
        # twenty-two checks.
        print("[route-admits] SKIP: nvcc is not on PATH -- this gate compiles, so it cannot report without one")
        return 0

    extra = a.flags.split() if a.flags else []
    blind, wrong = [], []

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ok, detail = plant_control(Path(td))
        if ok is None:
            print(f"[route-admits] !! plant control could not run: {detail}")
            blind.append("plant control (device-body static_assert) could not run")
        else:
            print(f"[route-admits] {'ok ' if ok else '!! '}"
                  f"{'plant: static_assert inside the device body':<48} expected fires   {detail}")
            if not ok:
                blind.append("plant control: a static_assert in the mainloop's device operator() did NOT fire")

    for label, defs, expect in CASES:
        errs, _ = compile_case(defs, extra)
        got = "rejects" if errs else "admits"
        ok = got == expect
        mark = "ok " if ok else "!! "
        print(f"[route-admits] {mark}{label:<48} expected {expect:<8} got {got}"
              + (f"  ({len(errs)} error line(s))" if errs else ""))
        if a.verbose and errs:
            print("               " + errs[0][:160])
        if not ok:
            (blind if expect == "rejects" else wrong).append(label)

    if blind:
        print("\n[route-admits] FAIL -- A CONTROL STOPPED FAILING. The probe cannot see errors, so every other")
        print("               verdict above is void, including the ones that look green:")
        for b in blind:
            print(f"                 {b}")
        print("               The planted device-body error was not instantiated; inspect the probe's type odr-use.")
        return 1
    if wrong:
        print("\n[route-admits] FAIL -- a route rejected a geometry it was expected to admit:")
        for w in wrong:
            print(f"                 {w}")
        print("               That is a FINDING, not necessarily a bug in this gate. Read the error, then decide")
        print("               whether the route changed or the expectation was wrong.")
        return 1
    print(f"\n[route-admits] PASS -- {len(CASES)} case(s) + 3 controls (two must fail, one planted assert must"
          f" fire), so the green verdicts above carry information")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
