#!/usr/bin/env python3
"""Copy verified native runtime payloads into a new, focused LFS package."""

import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.verify_kpack_dispatch import verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    src = a.build.resolve(strict=True)
    dst = a.output.resolve()
    m = verify(src)
    if dst.exists():
        raise ValueError("publication output already exists")
    if not dst.is_relative_to(ROOT / "prebuilt/ppu0010"):
        raise ValueError("publication must be under prebuilt/ppu0010")
    dst.mkdir(parents=True)
    paths = [
        "manifest.json",
        "plan.json",
        "libquactlize_kpack_dispatch.so",
        "libquactlize_ppu_execution.so",
    ]
    paths += [r["path"] for r in m["modules"]]
    for name in paths:
        target = dst / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / name, target)
    verify(dst)
    print(
        f"KPACK_NATIVE_PUBLISHED modules={len(m['modules'])} output={dst} device_validation=PENDING"
    )


if __name__ == "__main__":
    main()
