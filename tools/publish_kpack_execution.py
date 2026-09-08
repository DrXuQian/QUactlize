#!/usr/bin/env python3
"""Copy verified, locally compiled execution payloads into a fresh LFS package."""

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import sha


def publish(execution, grouped, output):
    execution, grouped, output = (
        execution.resolve(),
        grouped.resolve(),
        output.resolve(),
    )
    if output.parent != ROOT / "prebuilt/ppu0010" or output.exists():
        raise ValueError(
            "publication needs a fresh immediate child of prebuilt/ppu0010"
        )
    one = json.loads((execution / "manifest.json").read_text())
    many = json.loads((grouped / "manifest.json").read_text())
    if (
        one["schema"] != "quactlize.kpack-execution-build.v1"
        or many["schema"] != "quactlize.kpack-grouped-device-build.v2"
    ):
        raise ValueError("build schema differs")
    pairs = [
        (execution / "manifest.json", output / "manifest.json"),
        (grouped / "manifest.json", output / "grouped/manifest.json"),
    ]
    if (
        one["library"] != "libquactlize_ppu_execution.so"
        or sha(execution / one["library"]) != one["sha256"]
    ):
        raise ValueError("execution library identity differs")
    pairs.append((execution / one["library"], output / one["library"]))
    for module in many["modules"]:
        relative = module["path"]
        if not re.fullmatch(r"modules/[0-9a-f]{64}/kernel\.so", relative):
            raise ValueError("module path is not content-addressed and relative")
        if sha(grouped / relative) != module["sha256"]:
            raise ValueError("grouped payload identity differs")
        pairs.append((grouped / relative, output / "grouped" / relative))
    if len(many["modules"]) != 10:
        raise ValueError("grouped module denominator")
    output.mkdir(parents=False)
    for src, dst in pairs:
        if src.is_symlink() or not src.is_file():
            raise ValueError("payload is not a regular file")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if sha(src) != sha(dst):
            raise ValueError("published payload copy differs")
    print(
        f"KPACK_EXECUTION_PUBLISHED files={len(pairs)} output={output} device_validated=0"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execution", type=Path, required=True)
    p.add_argument("--grouped", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    publish(args.execution, args.grouped, args.output)
