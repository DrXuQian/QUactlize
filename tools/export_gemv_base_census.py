#!/usr/bin/env python3
"""Export the base GEMV tactic census from the real C++ authority."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent
EMITTER = ROOT / "dev/fold_derivation/emit_gemv_tactic_space.cpp"
INCLUDE = ROOT / "quactlize/include"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authority-log", type=Path, required=True)
    args = parser.parse_args()

    # This is a provenance authority, not a developer convenience build.  An
    # ambient CXX would let two otherwise identical bundles derive their base
    # census with different compilers while recording neither distinction.
    compiler = "g++"
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise SystemExit("base census authority requires the literal g++ command")
    identity = subprocess.run(
        [compiler, "--version"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if identity.returncode or not identity.stdout.splitlines():
        raise SystemExit("base census authority could not identify g++:\n" +
                         identity.stdout)
    with tempfile.TemporaryDirectory(prefix="qz-gemv-base-census-") as td:
        binary = Path(td) / "emit-gemv-tactic-space"
        build_argv = [compiler, "-std=c++17", "-I", str(INCLUDE),
                      str(EMITTER), "-o", str(binary)]
        build = subprocess.run(build_argv, cwd=ROOT, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if build.returncode:
            raise SystemExit("base census authority failed to compile:\n" + build.stdout)
        run = subprocess.run([str(binary)], cwd=ROOT, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if run.returncode:
            raise SystemExit("base census authority failed to run:\n" + run.stdout)

    authority_header = "\n".join((
        "AUTHORITY_COMPILER_ARGV," + json.dumps(build_argv, separators=(",", ":")),
        "AUTHORITY_COMPILER_PATH," + json.dumps(
            str(Path(compiler_path).resolve()), separators=(",", ":")),
        "AUTHORITY_COMPILER_IDENTITY," + json.dumps(
            identity.stdout.splitlines()[0], separators=(",", ":")),
    )) + "\n"
    args.authority_log.parent.mkdir(parents=True, exist_ok=True)
    args.authority_log.write_text(authority_header + run.stdout, encoding="utf-8")
    census: dict[str, int] = {}
    reasons: dict[str, int] = {}
    result = None
    for row in csv.reader(io.StringIO(run.stdout)):
        if row[:1] == ["CENSUS"] and len(row) == 3:
            census[row[1]] = int(row[2])
        elif row[:1] == ["EXCLUSION"] and len(row) == 3:
            reasons[row[1]] = int(row[2])
        elif row[:1] == ["RESULT"] and len(row) == 2:
            result = row[1]
    required = {"total", "legal", "rejected"}
    if set(census) != required or result != "PASS":
        raise SystemExit(
            f"base census authority is incomplete: census={census} result={result!r}")
    if census["legal"] + census["rejected"] != census["total"]:
        raise SystemExit("base census does not close")
    if sum(reasons.values()) != census["rejected"]:
        raise SystemExit("base census exclusion histogram does not close")

    value = {
        "schema": "quactlize-gemv-base-census-v1",
        "total": census["total"],
        "legal": census["legal"],
        "pruned": census["rejected"],
        "prune_reasons": dict(sorted(reasons.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
