#!/usr/bin/env python3
"""Copy verified native runtime payloads into a new, focused LFS package."""

import argparse
import json
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
        "libquactlize_kpack_dispatch.so",
        "libquactlize_ppu_execution.so",
    ]
    paths += [r["path"] for r in m["modules"]]
    for name in paths:
        target = dst / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / name, target)
    if m.get("jit_required"):
        # A JIT-only package has no compiled closure. Model prewarm uses the
        # caller's actual requests, not the development coverage census.
        (dst / "plan.json").write_text(json.dumps(dict(parents=[], requests=[],
            scope="ON_DEMAND_JIT_USE_MODEL_PREWARM_PLAN"), indent=2)+"\n")
    else:
        shutil.copy2(src / "plan.json", dst / "plan.json")
    verify(dst)
    print(
        f"KPACK_NATIVE_PUBLISHED modules={len(m['modules'])} output={dst} device_validation=PENDING"
    )


if __name__ == "__main__":
    main()
