#!/usr/bin/env python3
"""The A0 fixture is EXACT BY CONSTRUCTION, and this is what makes that claim checkable.

WHY THIS EXISTS. For a day we asked whether A0's 233 SK-split mismatches were a kernel defect or FP32
reassociation, and answered it by building ever more machinery: bucketed diagnostics, a device-side
pre-fixup capture, a same-order CPU replay, a capture-perturbation control. Every one of those was
built to *interpret* an ambiguity. The ambiguity was avoidable.

If every partial sum is exactly representable, every accumulation order produces the SAME number. Then
"the device and the reference differ" means "one of them is wrong", with no third explanation -- no
replay, no ULP threshold, no interpretation. Reassociation stops being a hypothesis because it stops
being possible.

TWO BOUNDS, AND THE SECOND IS THE TIGHT ONE. It is not enough for the FP32 accumulator to be exact:
D is stored as fp16, so a defect smaller than fp16's ULP at the output magnitude would be rounded
away and never observed. The output has to be exact too, and fp16 carries 11 mantissa bits against
FP32's 24 -- so the storage bound is 8192x tighter and it is the one that shapes the fixture.

The construction that satisfies both while keeping scale AND zero load-bearing (a fixture that pinned
scale=1, zero=0 would be exact and would also stop testing the ScaleZero path):

    A       sparse: at most NZ nonzeros per row, values in {-1, +1}
    q       the full int4 range [0, 16)
    scale   integers, so (q-8)*scale is an integer
    zero    integers, so (q-8)*scale + zero is an integer

Every weight is then an integer, every partial sum over any subset of any order is an integer, and the
bound on all of them is NZ * max|w|.

THE CHECK IS DERIVED, NOT RESTATED. It reads the fixture's own constants out of the source and
recomputes the bound. Writing the expected numbers here instead would make it a copy that can never
disagree -- the failure mode this repo has hit before, where a check restates what it is checking.
Absent the fixture it reports NOT PRESENT, which is neither a pass nor a failure: an environment that
cannot run a check must not be able to green it.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks" / "test_lowbit_dense_bench.cu"

FP32_EXACT_INT = 1 << 24   # float has a 24-bit significand
FP16_EXACT_INT = 1 << 11   # half has an 11-bit significand -- 8192x tighter, and the binding one


def weight_bound(scales, zeros):
    """max |(q-8)*s + z| over the full int4 range. Integer by construction when s and z are."""
    return max(abs((q - 8) * s + z) for q in range(16) for s in scales for z in zeros)


def verdict(nz, scales, zeros):
    """(fp32_ok, fp16_ok, wmax, dmax) -- the whole property in four numbers."""
    if not all(float(v).is_integer() for v in list(scales) + list(zeros)):
        return False, False, None, None       # a non-integer scale/zero breaks the premise outright
    wmax = weight_bound(scales, zeros)
    dmax = nz * wmax                          # A is +-1, so |D| <= (nonzeros) * max|w|
    return dmax <= FP32_EXACT_INT, dmax <= FP16_EXACT_INT, wmax, dmax


# The fixture publishes its parameters as these three constants so this check can find them.  They are
# deliberately not duplicated in this file.
PATTERNS = {
    "nz":     r"kExactFixtureNonzerosPerRow\s*=\s*(\d+)",
    "scales": r"kExactFixtureScales\s*\[\s*\]\s*=\s*\{([^}]*)\}",
    "zeros":  r"kExactFixtureZeros\s*\[\s*\]\s*=\s*\{([^}]*)\}",
}


def read_fixture(text):
    out = {}
    for key, pat in PATTERNS.items():
        m = re.search(pat, text)
        if not m:
            return None
        out[key] = m.group(1)
    return (int(out["nz"]),
            [int(x) for x in re.findall(r"-?\d+", out["scales"])],
            [int(x) for x in re.findall(r"-?\d+", out["zeros"])])


def self_test():
    """A check that has only ever been seen to pass is not evidence. Both directions, explicitly."""
    ok32, ok16, wmax, dmax = verdict(32, [1, 2, 4], [-3, 0, 3])
    assert ok32 and ok16 and wmax == 35 and dmax == 1120, (ok32, ok16, wmax, dmax)
    # One nonzero too many is enough to lose fp16 exactness -- the bound is not slack.
    ok32, ok16, _, dmax = verdict(64, [1, 2, 4], [-3, 0, 3])
    assert ok32 and not ok16 and dmax == 2240, (ok32, ok16, dmax)
    # A non-integer scale destroys the premise even when the magnitudes look small.
    ok32, ok16, _, _ = verdict(4, [1], [0])
    assert ok32 and ok16
    assert verdict(4, [0.5], [0])[:2] == (False, False)
    return True


def main():
    self_test()
    if not BENCH.exists():
        print(f"[exact-fixture] {BENCH} not found -- cannot check", file=sys.stderr)
        return 2
    found = read_fixture(BENCH.read_text())
    if found is None:
        print("[exact-fixture] NOT PRESENT: the exact A0 fixture constants "
              "(kExactFixtureNonzerosPerRow / kExactFixtureScales / kExactFixtureZeros) are not in "
              f"{BENCH.name}. This is neither a pass nor a failure -- there is nothing to check yet.")
        return 3
    nz, scales, zeros = found
    ok32, ok16, wmax, dmax = verdict(nz, scales, zeros)
    print(f"[exact-fixture] nz={nz} scales={scales} zeros={zeros} -> max|w|={wmax} max|D|={dmax}")
    print(f"[exact-fixture] FP32 accumulation exact: {ok32}  (bound {dmax} vs {FP32_EXACT_INT})")
    print(f"[exact-fixture] fp16 output exact:       {ok16}  (bound {dmax} vs {FP16_EXACT_INT})")
    if ok32 and ok16:
        print("[exact-fixture] PASS -- every accumulation order yields the same FP32 value and the "
              "same fp16 store, so any device/reference difference is a defect and not reassociation")
        return 0
    print("[exact-fixture] FAIL -- the fixture no longer forces bit-identical results across "
          "accumulation orders, so a mismatch would once again be ambiguous", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
