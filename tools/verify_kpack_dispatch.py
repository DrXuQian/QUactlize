#!/usr/bin/env python3
"""Validate a native dispatch package before allowing model execution."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import Compiler, sha, source_contract
from quactlize.runtime.tuning import digest


def verify(root):
    root = Path(root).resolve(strict=True)
    m = json.loads((root / "manifest.json").read_text())
    if (m.get("schema") != "quactlize.kpack-native-dispatch.v1" or
        not isinstance(m.get("modules"), list) or
        (not m["modules"] and m.get("jit_required") is not True)):
        raise ValueError("native package schema or module set differs")
    for name, field in (
        ("libquactlize_kpack_dispatch.so", "dispatch_sha256"),
        ("libquactlize_ppu_execution.so", "execution_sha256"),
    ):
        if sha(root / name) != m[field]:
            raise ValueError("native payload differs: " + name)
    if m.get("jit_required") or "jit_source_contract" in m:
        if source_contract(m.get("jit_source_identity", {})) != m.get("jit_source_contract"):
            raise ValueError("JIT source contract differs")
    seen = set()
    for r in m["modules"]:
        if (
            r["key"] in seen
            or len(r["key"]) != 64
            or any(c not in "0123456789abcdef" for c in r["key"])
        ):
            raise ValueError("duplicate/invalid module key")
        seen.add(r["key"])
        expected = "modules/" + r["key"] + "/kernel.so"
        path = (root / expected).resolve(strict=True)
        if (
            r["path"] != expected
            or not path.is_relative_to(root)
            or sha(path) != r["sha256"]
        ):
            raise ValueError("module payload differs: " + r["key"])
        source = Compiler.source(None, r["parent"], "")
        if (
            digest(dict(identity=r["identity"], parent=r["parent"], source=source))
            != r["key"]
        ):
            raise ValueError("module compiler identity differs")
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    a = p.parse_args()
    manifest = verify(a.bundle)
    print(
        f"KPACK_NATIVE_PACKAGE VERIFIED modules={len(manifest['modules'])} "
        f"execution_sha256={manifest['execution_sha256']} device_admission=PENDING"
    )
