#!/usr/bin/env python3
"""Reject retired diagnostic and global-policy switches from product source.

Historical timing ablations may remain documented outside ``quactlize/``, but
the installable source must expose only real paths selected through ordinary
types and policies.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


BANNED = (
    "GGUF_VECDOT_CODE_NOP",
    "GGUF_VECDOT_FP32_ACTIVATION",
    "GGUF_VECDOT_SCALE_NOP",
    "MOEG_FORCE3D",
    "MOEG_PERSISTENT_GRID_CTAS",
    "MOEG_SMEM",
    "PPU_A_PACK",
    "PPU_B_CHUNK_BISECT",
    "PPU_FORCE_INSTANTIATE",
    "PPU_MAXREG",
    "PPU_MIXED_LEGACY_B_INDEXED_A_COPY",
    "PPU_MIXED_LEGACY_MODULO_METADATA_PUBLISHERS",
    "PPU_MMA_PROBE",
    "PPU_PACKED_PAIR",
    "PPU_PACKED_SCALE_" "FUSED",
    "PPU_Q4_A32_EXACT_TYPE_PROBE",
    "PPU_Q4_KPACK4_LEGACY_LOADER_OUTPUT_LAYOUT",
    "PPU_SCALE_PAD",
    "PPU_SCALE_PREFETCH",
    "PPU_SCALE_SWIZZLE",
    "PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE",
    "Q4K_BC_PLANT_WRONG_MAGIC",
    "QUACTLIZE_W4_SPLITK_SEVER_PREPARE_EDGE",
)
SOURCE_SUFFIXES = {
    ".c", ".cc", ".cmake", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".in",
    ".inc", ".inl", ".json", ".md", ".py", ".pyi", ".sh", ".toml",
    ".txt", ".yaml", ".yml",
}
PATTERN = re.compile(r"\b(?:" + "|".join(map(re.escape, BANNED)) + r")\b")


def violations(product_root: Path) -> list[str]:
    hits: list[str] = []
    for path in sorted(product_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        rel = path.relative_to(product_root)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(lines, 1):
            match = PATTERN.search(line)
            if match:
                hits.append(f"{rel}:{lineno}: forbidden diagnostic {match.group(0)}")
    return hits


def self_test() -> None:
    for name in BANNED:
        planted = f"#if defined({name})\n"
        if PATTERN.findall(planted) != [name]:
            raise RuntimeError(f"retired-macro deny pattern missed planted {name}")
        if PATTERN.search(f"{name}_RENAMED"):
            raise RuntimeError(f"retired-macro deny pattern is not bounded for {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "quactlize",
        help="product source root (defaults to this checkout's quactlize directory)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"product source root does not exist: {root}")

    self_test()
    hits = violations(root)
    if hits:
        print("[product-retired-macros] FAIL")
        for hit in hits:
            print(f"  {hit}")
        return 1
    print(f"[product-retired-macros] PASS root={root} banned={len(BANNED)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
