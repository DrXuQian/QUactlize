#!/usr/bin/env python3
"""Reject known logical-code/packed-byte unit confusions.

This is deliberately a source contract rather than a second allocator.  The
vendor DeviceAllocation API itself is asymmetric, and changing it would alter
every upstream test.  Owned packed buffers must make their unit explicit.
"""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
DEVICE = ROOT / "third_party/actlize/tools/util/include/cutlass/util/device_memory.h"
AUDIT = ROOT / "dev/fold_derivation/SUBBYTE_UNIT_AUDIT.md"
FIXED = {
    ROOT / "benchmarks/moe_splitk_bench_common.hpp": 1,
    ROOT / "benchmarks/lowbit_moe_bench.hpp": 3,
    ROOT / "tests/test_lowbit_grouped.cu": 3,
}
LEGACY = ROOT / "tests/test_moe_gemm_ppu.cu"
REGISTRY = ROOT / "ci/registry.py"


def flat(s: str) -> str:
    return re.sub(r"\s+", "", s)


def audit(texts: dict[Path, str]) -> list[str]:
    bad: list[str] = []
    dev = flat(texts[DEVICE])
    if "count*sizeof(T)" not in dev or "count*sizeof_bits<T>::value/8" not in dev:
        bad.append("DeviceAllocation allocation/copy unit asymmetry changed without updating this contract")

    for path, expected_bytes in FIXED.items():
        src = texts[path]
        # These aliases are the exact failure shape: a physical byte count was
        # passed to a typed sub-byte allocation and then to its typed copy.
        if re.search(r"DeviceAllocation<(?:LOELEM|HIELEM|ELEM|int4_t)>\s+\w+\s*\([^;]*(?:_per|_lo|_hi|\bper\b)", src):
            bad.append(f"{path.relative_to(ROOT)} restored a typed sub-byte owner over a physical byte count")
        count = len(re.findall(r"DeviceAllocation<uint8_t>\s+\w+\s*\(", src))
        if count < expected_bytes:
            bad.append(f"{path.relative_to(ROOT)} exposes only {count}/{expected_bytes} required byte owners")

    legacy = texts[LEGACY]
    if "if (L != 1)" not in legacy or "retired for L>1" not in legacy:
        bad.append("legacy uniform-M wrapper no longer fails closed before unsafe L>1 sub-byte addressing")
    reg = texts[REGISTRY]
    if '"test_moe_gemm_ppu": [FP16_P]' in reg:
        bad.append("retired legacy target is again advertised as a validated FP16 path")
    if not re.search(r'"test_moe_gemm_ppu"\s*:\s*\(\[\]\s*,\s*"none"', reg):
        bad.append("retired legacy target is not classified as oracle=none")

    doc = texts[AUDIT]
    for token in (
        "logical code", "packed byte", "DeviceAllocation<uint8_t>",
        "exactly seven active under-copies", "interleaved `dB`/`dB2`",
        "Marlin scheduler units", "tactic-K tiles, not elements or bytes",
    ):
        if token not in doc:
            bad.append(f"sub-byte audit lost load-bearing statement {token!r}")
    return bad


def main() -> int:
    paths = (DEVICE, AUDIT, *FIXED, LEGACY, REGISTRY)
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    if missing:
        print("[subbyte-units] FAIL: missing " + ", ".join(missing))
        return 1
    texts = {p: p.read_text() for p in paths}
    bad = audit(texts)
    if bad:
        print("[subbyte-units] FAIL: " + "; ".join(bad))
        return 1

    plants = (
        (ROOT / "benchmarks/moe_splitk_bench_common.hpp",
         "DeviceAllocation<uint8_t> db", "DeviceAllocation<int4_t> db",
         "split-K physical-byte owner"),
        (ROOT / "benchmarks/lowbit_moe_bench.hpp",
         "DeviceAllocation<uint8_t> _b1", "DeviceAllocation<LOELEM> _b1",
         "two-plane low physical-byte owner"),
        (ROOT / "tests/test_lowbit_grouped.cu",
         "DeviceAllocation<uint8_t> db", "DeviceAllocation<ELEM> db",
         "grouped single-plane physical-byte owner"),
        (LEGACY, "if (L != 1)", "if (false)", "legacy L>1 fail-close"),
        (REGISTRY, '"test_moe_gemm_ppu":        ([],                  "none"',
         '"test_moe_gemm_ppu":        (["int4"],            "self"',
         "legacy oracle classification"),
    )
    for path, old, new, label in plants:
        planted = dict(texts)
        if old not in planted[path]:
            print(f"[subbyte-units] FAIL: cannot plant {label}")
            return 1
        planted[path] = planted[path].replace(old, new, 1)
        if not audit(planted):
            print(f"[subbyte-units] FAIL: checker accepted planted {label}")
            return 1
    print("[subbyte-units] PASS: 7 physical-byte under-copies fixed; legacy L>1 seam fails closed; 5 plants rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
