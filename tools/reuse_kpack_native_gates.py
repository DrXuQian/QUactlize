#!/usr/bin/env python3
"""Reuse complete native/GEMV measurements after an adapter-only failure."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.export_kpack_prefill_policy import export as prefill_export
from tools.export_kpack_gemv_policy import export as gemv_export
from quactlize.runtime.compiler import sha
from quactlize.runtime.native import SDK
from tools.run_kpack_pack_gate import device_identity

NATIVE_SOURCES = (
    "tools/run_kpack_native_gate.py",
    "tools/run_kpack_grouped_device_gate.py",
    "tools/run_kpack_pack_gate.py",
    "quactlize/dispatch/native.py",
    "quactlize/runtime/native.py",
)


def validate(native, gemv, device, native_hash, execution_hash):
    if native.get("device") != device or gemv.get("device") != device:
        raise ValueError("resume device receipt differs")
    if (
        native.get("manifest_sha256") != native_hash
        or gemv.get("native_manifest_sha256") != native_hash
    ):
        raise ValueError("resume native package differs")
    if gemv.get("execution_sha256") != execution_hash:
        raise ValueError("resume execution image differs")
    prefill_export(native)
    _, recipes = gemv_export(gemv)
    expected = {
        (q, n, k, 256, 2, t * 8, ch, 8, 1)
        for q, n, k in ((12, 512, 2048), (13, 2048, 512))
        for t in (1, 4, 16)
        for ch in (1, 8)
    } | {(14, 248320, 2048, 1, 0, m, 1, 1, 1) for m in (1, 4)}
    if {tuple(r["key"]) for r in recipes} != expected:
        raise ValueError("resume lacks the exact 14 model GEMV contexts")


def reuse(source, output, bundle, legacy, execution, device):
    native = json.loads((source / "native-gate/summary.json").read_text())
    gemv = json.loads((source / "gemv-gate/summary.json").read_text())
    validate(
        native,
        gemv,
        device,
        sha(bundle / "manifest.json"),
        sha(bundle / "libquactlize_ppu_execution.so"),
    )
    if gemv.get("manifest_sha256") != sha(execution / "manifest.json"):
        raise ValueError("resume GEMV candidate package differs")
    libraries = {
        f"libquactlize_ppu_fmt{i}.so": sha(legacy / f"libquactlize_ppu_fmt{i}.so")
        for i in range(5)
    }
    if gemv.get("baseline_libraries") != libraries:
        raise ValueError("resume incumbent libraries differ")
    if (source / "quactlize-dirty.patch").read_bytes():
        raise ValueError("cannot reuse an unversioned measurement source")
    commit = (source / "quactlize-source.txt").read_text().strip()
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("invalid measurement source commit")
    for name in NATIVE_SOURCES:
        before = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        if hashlib.sha256(before).hexdigest() != sha(ROOT / name):
            raise ValueError(f"native measurement source changed: {name}")
    from tools.run_kpack_gemv_gate import subprocess_source

    if gemv.get("source") != subprocess_source():
        raise ValueError("GEMV measurement source changed")
    files = []
    for directory in ("native-gate", "gemv-gate"):
        if (output / directory).exists():
            raise ValueError("resume destination already contains gates")
        if (source / directory).is_symlink():
            raise ValueError("gate directories must not be symlinks")
        for path in sorted((source / directory).iterdir()):
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise ValueError(f"unexpected gate artifact: {path}")
            files.append(path)
        files.append(source / f"{directory}.log")
    if any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError("gate logs must be regular files")
    if (output / "reused-gates.json").exists() or any(
        (output / p.relative_to(source)).exists() for p in files
    ):
        raise ValueError("resume destination already contains receipts")
    receipt = dict(
        source_results=str(source),
        source_commit=commit,
        device=device,
        native_contexts=28,
        gemv_contexts=14,
        files={str(p.relative_to(source)): sha(p) for p in files},
        scope="REUSED_MICROBENCHMARKS_MODEL_TIMING_NOT_REUSED",
    )
    for directory in ("native-gate", "gemv-gate"):
        (output / directory).mkdir()
    for path in files:
        shutil.copy2(path, output / path.relative_to(source))
    with (output / "reused-gates.json").open("x") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")
    print(
        f"KPACK_NATIVE_RESUME PASS native=28 gemv=14 source={source} model_rerun=1",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "bundle", "legacy", "execution", "sdk"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.execution / "manifest.json").read_text())
    for name, value in manifest["runtime"].items():
        if sha(a.sdk / "lib" / name) != value:
            raise ValueError(f"resume SDK runtime differs: {name}")
    reuse(
        a.source.resolve(strict=True),
        a.output.resolve(strict=True),
        a.bundle,
        a.legacy,
        a.execution,
        device_identity(SDK(a.sdk)),
    )


if __name__ == "__main__":
    main()
