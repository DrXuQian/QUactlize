#!/usr/bin/env python3
"""Reject collaboration-tool provenance from the installable product source.

Product comments must explain technical facts that remain useful without access to a
particular development session. Development notes and local skills are outside this
check because only ``quactlize/`` is part of its source boundary.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SOURCE_SUFFIXES = {
    ".c", ".cc", ".cmake", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".in", ".inc",
    ".inl", ".json", ".md", ".py", ".pyi", ".sh", ".toml", ".txt", ".yaml", ".yml",
}
BANNED = re.compile(r"\b(?:codex|claude)\b", re.IGNORECASE)


def violations(product_root: Path) -> list[str]:
    hits: list[str] = []
    for path in sorted(product_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(product_root)
        if BANNED.search(rel.as_posix()):
            hits.append(f"{rel}: forbidden provenance in path")
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            hits.append(f"{rel}: product source is not UTF-8")
            continue
        for lineno, line in enumerate(lines, 1):
            if BANNED.search(line):
                hits.append(f"{rel}:{lineno}: {line.strip()}")
    return hits


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

    hits = violations(root)
    if hits:
        print("[product-source-provenance] FAIL")
        for hit in hits:
            print(f"  {hit}")
        return 1
    print(f"[product-source-provenance] PASS root={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
