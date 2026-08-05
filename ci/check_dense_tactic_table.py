#!/usr/bin/env python3
"""Reject a stale lowbit dense generated table before the device compiler sees it."""

import argparse
import hashlib
import os
import re
import subprocess
import tempfile
from itertools import zip_longest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "quactlize" / "include" / "ppu_tactic_space.hpp"
EMITTER = ROOT / "benchmarks" / "emit_tactic_configs.cpp"
DEFAULT_TABLE = ROOT / "benchmarks" / "lowbit_dense_configs.inc"
CANONICAL_ARGS = ("4", "64", "--space=dense", "2", "3", "4", "6", "8", "12")


def fnv1a64(data: bytes) -> str:
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def macro(text: str, name: str):
    match = re.search(rf'^#define\s+{re.escape(name)}\s+(?:"([0-9a-f]+)"|(\d+))\s*$', text, re.M)
    return next((value for value in match.groups() if value is not None), None) if match else None


def fail(message: str) -> int:
    print(f"[dense-table] ERROR: {message}")
    print("[dense-table] regenerate from the repository root:")
    print("  c++ -std=c++17 -Iquactlize/include benchmarks/emit_tactic_configs.cpp -o /tmp/emit_tactic && \\")
    print("  /tmp/emit_tactic 4 64 --space=dense 2 3 4 6 8 12 > benchmarks/lowbit_dense_configs.inc")
    return 1


def first_difference(expected: bytes, actual: bytes) -> str:
    for line, (want, got) in enumerate(
            zip_longest(expected.splitlines(), actual.splitlines(), fillvalue=b"<end>"), 1):
        if want != got:
            return (f"first difference at line {line}: expected {want[:100]!r}, "
                    f"found {got[:100]!r}")
    return "byte content differs"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE,
                        help="table to validate (default: the committed dense table)")
    args = parser.parse_args()
    table = args.table.resolve()

    for path in (SPACE, EMITTER, table):
        if not path.is_file():
            return fail(f"missing {path}")

    actual = table.read_bytes()
    text = actual.decode()
    stamped_rows = macro(text, "LOWBIT_DENSE_CFG_ROWS")
    stamped_space = macro(text, "LOWBIT_DENSE_CFG_SPACE_FNV1A64")
    stamped_emitter = macro(text, "LOWBIT_DENSE_CFG_EMITTER_FNV1A64")
    missing = [name for name, value in (("row count", stamped_rows), ("space hash", stamped_space),
                                        ("emitter hash", stamped_emitter)) if value is None]
    if missing:
        return fail(f"{table} has no {', '.join(missing)} provenance; it predates the durability guard")

    listed_rows = len(re.findall(rb'^\s*X\(', actual, re.M))
    if int(stamped_rows) != listed_rows:
        return fail(f"{table} stamps {stamped_rows} rows but its X-macro contains {listed_rows}")

    current_space = fnv1a64(SPACE.read_bytes())
    current_emitter = fnv1a64(EMITTER.read_bytes())
    if stamped_space != current_space:
        return fail(f"{table} space hash is {stamped_space}, current ppu_tactic_space.hpp is {current_space}")
    if stamped_emitter != current_emitter:
        return fail(f"{table} emitter hash is {stamped_emitter}, current emit_tactic_configs.cpp is {current_emitter}")

    compiler = os.environ.get("CXX", "c++")
    with tempfile.TemporaryDirectory(prefix="quactlize-dense-table-") as temp_dir:
        binary = Path(temp_dir) / "emit_tactic"
        built = subprocess.run(
            [compiler, "-std=c++17", f"-I{ROOT / 'quactlize' / 'include'}", str(EMITTER), "-o", str(binary)],
            cwd=ROOT, capture_output=True, text=True)
        if built.returncode:
            detail = built.stderr.strip().splitlines()
            return fail(f"current emitter does not compile with {compiler}: {detail[-1] if detail else 'no diagnostic'}")
        emitted = subprocess.run([str(binary), *CANONICAL_ARGS], cwd=ROOT, capture_output=True)
        if emitted.returncode:
            detail = emitted.stderr.decode(errors="replace").strip().splitlines()
            return fail(f"current emitter failed: {detail[-1] if detail else 'no diagnostic'}")
        expected = emitted.stdout

    if actual != expected:
        return fail(
            f"{table} does not exactly match a freshly built emitter; {first_difference(expected, actual)}; "
            f"expected md5={hashlib.md5(expected).hexdigest()}, found md5={hashlib.md5(actual).hexdigest()}")

    print(f"[dense-table] verified rows={listed_rows} space_fnv1a64={current_space} "
          f"emitter_fnv1a64={current_emitter}; exact regeneration matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
