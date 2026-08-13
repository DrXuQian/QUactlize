#!/usr/bin/env python3
"""Report PPU codegen for the one shipping standalone Marlin symbol.

No opcode count in this report is guessed or copied from source.  It consumes
only hgobjdump output for the exact symbol selected by the runner.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instructions(text: str) -> list[str]:
    result: list[str] = []
    # hgobjdump variants put either an address or encoded bytes before the
    # opcode.  Require an opcode-shaped token and ignore source/ELF headings.
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("//", "#", ".", "File ", "Function ")):
            continue
        match = re.search(
            r"(?:^|\s)([sv]?\.?[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+)\s+",
            line,
        )
        if match:
            result.append(match.group(1).lower())
    return result


def count_like(counter: collections.Counter[str], needle: str) -> int:
    return sum(value for opcode, value in counter.items() if needle in opcode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--line", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--demangled", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    try:
        line = args.line.read_text(errors="replace")
        resource = args.resource.read_text(errors="replace")
        source = json.loads(args.source_manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[l176:ppu] FAIL: cannot read evidence: {exc}", file=sys.stderr)
        return 1
    if not line.strip():
        print("[l176:ppu] FAIL: exact-symbol disassembly is empty", file=sys.stderr)
        return 1
    if not resource.strip():
        print("[l176:ppu] FAIL: exact-symbol resource report is empty", file=sys.stderr)
        return 1
    for token in ("device_kernel", "MarlinKernelPPU", "MarlinCollectivePPU"):
        if token not in args.demangled:
            print(f"[l176:ppu] FAIL: selected symbol lacks {token}: {args.demangled}", file=sys.stderr)
            return 1

    required_sources = (
        "marlin_load_ppu.hpp",
        "marlin_dequant_ppu.hpp",
        "marlin_mma_ppu.hpp",
        "marlin_collective_ppu.hpp",
        "marlin_kernel_ppu.hpp",
    )
    missing = [name for name in required_sources if name not in line]
    if missing:
        print(
            "[l176:ppu] FAIL: exact symbol lost line bindings for " + ", ".join(missing),
            file=sys.stderr,
        )
        return 1
    forbidden_sources = (
        "quactlize_mma_mixed_input.hpp",
        "ppu_mma_aiu_fold.hpp",
        "ppu_mma_aiu_mixed_input_2plane.hpp",
        "fast_numeric_conversion_for_mix_gemm.h",
    )
    leaked = [name for name in forbidden_sources if name in line]
    if leaked:
        print(
            "[l176:ppu] FAIL: standalone symbol contains generic-collective source: "
            + ", ".join(leaked), file=sys.stderr,
        )
        return 1

    inst = instructions(line)
    if not inst:
        print("[l176:ppu] FAIL: could not parse any exact-symbol instruction", file=sys.stderr)
        return 1
    counts = collections.Counter(inst)
    mma = count_like(counts, "mma")
    if mma == 0:
        print("[l176:ppu] FAIL: exact standalone symbol contains no MMA opcode", file=sys.stderr)
        return 1

    focus_names = (
        "xnor", "lop3", "byte.prmt", "shrl", "shra", "mul.i", "madl",
        "s.mov", "s.cmp", "s.cbr", "v.mov.v2s", "s.wait", "cp.async",
        "tsm.ld", "smem.ld", "smem.st",
    )
    focus = {name: count_like(counts, name) for name in focus_names}
    spill_words = re.findall(r"(?i)\b(?:spill|stack|scratch|local(?:[- ]memory)?)\b[^\n]*", resource)
    payload = {
        "schema": "quactlize.l176.standalone-marlin-ppu-codegen.v1",
        "git_sha": source["git_sha"],
        "binary_sha256": sha(args.binary),
        "source_manifest_sha256": sha(args.source_manifest),
        "symbol": args.symbol,
        "demangled": args.demangled,
        "instruction_total_parsed": len(inst),
        "opcode_counts": dict(sorted(counts.items())),
        "focus_counts": focus,
        "line_sha256": sha(args.line),
        "resource_sha256": sha(args.resource),
        "resource_spill_text": spill_words,
        "claims": {
            "same_shipping_symbol": True,
            "helper_line_bindings": list(required_sources),
            "generic_collective_source": "absent",
            "source_flat_accumulator_escape": "absent-by-source-manifest",
            # Do not infer zero physical spilling from source.  The raw
            # resource report and exact opcode inventory are deliberately
            # retained for an explicit backend verdict.
            "backend_spill": "reported-not-inferred",
        },
    }
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"[l176:ppu] PASS: exact shipping symbol parsed={len(inst)} mma={mma} "
        f"xnor={focus['xnor']} lop3={focus['lop3']} s.mov={focus['s.mov']} "
        f"s.cmp={focus['s.cmp']} generic-source=ABSENT; backend spill is raw evidence, not inferred"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
