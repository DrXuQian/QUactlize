#!/usr/bin/env python3
"""Compile and execute the host-readable Q8 policy/record authority."""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "dev" / "fold_derivation" / "l209_scalefirst_policy_contract.cpp"
INCLUDE = ROOT / "quactlize" / "include"
PREFIX = "Q8_POLICY_CELL "
GRID_RE = re.compile(r"^Q8_GRID_WITNESS capacity=(\d+) balanced=(\d+) Q=(\d+) CU=(\d+) b=(\d+)$", re.M)


def strict_json(text: str) -> dict:
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate JSON key {key}")
            out[key] = value
        return out
    return json.loads(text, object_pairs_hook=pairs)


def validate(text: str) -> None:
    lines = [line for line in text.splitlines() if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise ValueError(f"machine record count {len(lines)}, expected 1")
    record = strict_json(lines[0][len(PREFIX):])
    required = {"capacity_b_mask", "balanced_b_mask", "candidate_denominator",
                "sample_us", "MFU_pct", "distinct_MBU_model_pct"}
    if not required.issubset(record):
        raise ValueError(f"machine record missing {sorted(required - set(record))}")
    if record["grid"] != 512 or record["candidate_denominator"] != 2501:
        raise ValueError("machine record lost the G512/2501 witness")
    match = GRID_RE.search(text)
    if match is None or tuple(map(int, match.groups())) != (576, 512, 2048, 72, 8):
        raise ValueError("capacity/balanced grid witnesses drifted")


def main() -> int:
    build = pathlib.Path("/workspace") / f"quactlize-scalefirst-policy-check-{os.getpid()}"
    if build.exists():
        print(f"[scalefirst-policy] FAIL: refusing stale directory {build}", file=sys.stderr)
        return 2
    build.mkdir(parents=True)
    try:
        binary = build / "l209"
        command = [os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra",
                   "-Wformat=2", "-Werror=format", f"-I{INCLUDE}", str(SOURCE),
                   "-o", str(binary)]
        built = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if built.returncode:
            print(built.stderr, file=sys.stderr)
            print("[scalefirst-policy] FAIL: typed format/grid contract did not compile", file=sys.stderr)
            return 2
        ran = subprocess.run([str(binary)], cwd=ROOT, text=True, capture_output=True)
        if ran.returncode:
            print(ran.stdout + ran.stderr, file=sys.stderr)
            print(f"[scalefirst-policy] FAIL: contract executable rc={ran.returncode}", file=sys.stderr)
            return 2
        validate(ran.stdout)

        # Negative 1: a repeated machine key must be rejected, not last-write-wins.
        duplicate = ran.stdout.replace(
            '"capacity_b_mask":"0x0",',
            '"capacity_b_mask":"0x0","capacity_b_mask":"0x100",', 1)
        try:
            validate(duplicate)
        except ValueError as error:
            if "duplicate JSON key capacity_b_mask" not in str(error):
                raise
        else:
            raise ValueError("duplicate mask key negative did not turn red")

        # Negative 2: collapsing balanced into capacity must lose G512 and red.
        collapsed = ran.stdout.replace("balanced=512", "balanced=576", 1)
        try:
            validate(collapsed)
        except ValueError as error:
            if "grid witnesses drifted" not in str(error):
                raise
        else:
            raise ValueError("balanced-grid collapse negative did not turn red")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[scalefirst-policy] FAIL: {error}", file=sys.stderr)
        return 2
    finally:
        shutil.rmtree(build)
    print("[scalefirst-policy] PASS: typed -Werror=format record, unique JSON keys, "
          "G576 capacity, G512 balanced; duplicate-key and collapsed-grid negatives red")
    return 0


if __name__ == "__main__":
    sys.exit(main())
