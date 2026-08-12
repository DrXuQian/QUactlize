#!/usr/bin/env python3
"""Inspect non-empty device code emitted from the real dense Marlin Cfg."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_dense_marlin_codegen.py l134.ptx", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.is_file() or path.stat().st_size == 0:
        print("[marlin-codegen] FAIL: no non-empty PTX artifact")
        return 1
    ptx = path.read_text(errors="replace")
    bad = []
    entry_start = re.search(
        r"(?:\.visible\s+)?\.entry\s+l134_marlin_runtime_probe\s*\(", ptx)
    entry = ""
    if not entry_start:
        bad.append("runtime production-scheduler probe symbol absent")
    else:
        brace = ptx.find("{", entry_start.start())
        end = ptx.find("\n}", brace)
        if brace < 0 or end < 0:
            bad.append("cannot isolate runtime production-scheduler probe body")
        else:
            entry = ptx[entry_start.start():end + 2]
    if "l134_marlin_constexpr_witness" not in ptx:
        bad.append("constexpr production-scheduler witness symbol absent")
    # Runtime q/k decomposition and M/N/L decode require integer quotient and
    # remainder arithmetic.  NVCC is free to spell remainder either as rem or
    # as q=div; r=x-q*y.  Pin the semantic ingredients, not one optimizer
    # spelling; a probe whose body was constant-folded away satisfies neither.
    if entry and "div.u64" not in entry:
        bad.append("runtime scheduler algebra lost u64 quotient")
    if entry and "rem.u64" not in entry and not (
            "mul.lo.s64" in entry and "sub.s64" in entry):
        bad.append("runtime scheduler algebra lost u64 remainder reconstruction")
    if entry and "ctaid.x" not in entry:
        bad.append("runtime probe no longer consumes CTA x index")
    if entry and ("ctaid.y" in entry or "ctaid.z" in entry):
        bad.append("scheduler probe consumed non-x CTA coordinate")
    if entry and not re.search(
            r"ld\.param\.u32\s+[^,]+,\s*\[l134_marlin_runtime_probe_param_5\]",
            entry):
        bad.append("runtime scheduler algebra lost blocks-per-CU input")
    if entry:
        labels = {m.group(1): m.start() for m in re.finditer(
            r"^(\$L[^:]+):", entry, re.M)}
        has_backedge = any(
            labels.get(m.group(1), len(entry)) < m.start()
            for m in re.finditer(r"\bbra(?:\.uni)?\s+(\$L[^;\s]+)", entry))
        if not has_backedge:
            bad.append("runtime scheduler fetch loop lost its device-code backedge")
    # NVCC emits an unsigned-long-long constant as `.b8 symbol[280]` on this
    # path, i.e. 35 little-endian u64 values represented by 280 byte literals.
    # Some frontends retain a typed `[35]` initializer, so accept both forms
    # and decode them to the same semantic vector before comparing it.
    m = re.search(
        r"l134_marlin_constexpr_witness\[(35|280)\]\s*=\s*\{([^}]*)\}",
        ptx, re.S)
    if not m:
        bad.append("cannot parse 35-field constexpr witness initializer")
    else:
        raw = [int(x, 0) for x in re.findall(
            r"0x[0-9a-fA-F]+|\b\d+\b", m.group(2))]
        if m.group(1) == "280":
            # PTX permits a short aggregate initializer; the omitted tail is
            # zero-filled.  The final witness value is 128, so NVCC emits its
            # low byte and elides the seven trailing zero bytes.
            if len(raw) > 280 or any(x > 255 for x in raw):
                got = []
            else:
                raw.extend([0] * (280 - len(raw)))
                got = [sum(raw[8*i + j] << (8*j) for j in range(8))
                       for i in range(35)]
        else:
            got = raw
        expected = [20, 13, 0, 13, 3, 2, 0, 0, 0, 0, 1, 0, 13,
                    1, 0, 10, 2, 1, 1, 2048, 1, 1, 1,
                    72, 15, 15, 3, 9, 3, 19, 4, 128, 144, 8, 128]
        if got != expected:
            bad.append(f"constexpr witness differs: got={got}")
    if bad:
        print("[marlin-codegen] FAIL: " + "; ".join(bad))
        return 1
    print(f"[marlin-codegen] PASS: {path.stat().st_size} byte PTX; real-Cfg raw-shape seam, constants, CTA-x, quotient/remainder and fetch loop retained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
