#!/usr/bin/env python3
"""Exercise the dense CMake table parser/batcher and its fail-closed controls."""

import pathlib
import re
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE = ROOT / "quactlize" / "csrc" / "TacticTableUnits.cmake"
TABLE = ROOT / "benchmarks" / "lowbit_dense_configs.inc"


def configure(table_text: str, batch: int):
    work = pathlib.Path(tempfile.mkdtemp(prefix="dense-unit-generator-"))
    src, build = work / "src", work / "build"
    src.mkdir()
    table = src / TABLE.name
    table.write_text(table_text)
    (src / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.19)
project(dense_unit_generator_check NONE)
include("{MODULE}")
qz_parse_tactic_xmacro(rows FILE "{table}"
  LIST_MACRO LOWBIT_DENSE_CFG_LIST COUNT_MACRO LOWBIT_DENSE_CFG_ROWS)
qz_batch_tactic_rows(batches rows "{batch}")
list(LENGTH rows row_count)
list(LENGTH batches batch_count)
file(WRITE "${{CMAKE_BINARY_DIR}}/summary.txt" "${{row_count}},${{batch_count}}\n")
""")
    result = subprocess.run(["cmake", "-S", src, "-B", build], capture_output=True, text=True)
    return work, table, build, result


def expect_rejected(text: str, fragment: str, label: str) -> None:
    work, _, _, result = configure(text, 4)
    try:
        output = result.stdout + result.stderr
        normalized = " ".join(output.split())
        if result.returncode == 0 or " ".join(fragment.split()) not in normalized:
            raise RuntimeError(f"{label} was not rejected for the expected reason ({fragment!r})\n{output[-1200:]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    if not shutil.which("cmake"):
        print("[dense-units] ERROR: cmake is not installed; the generator cannot be checked")
        return 1
    if not MODULE.is_file() or not TABLE.is_file():
        print("[dense-units] ERROR: parser module or dense table is missing")
        return 1

    text = TABLE.read_text()
    stamped = re.search(r"^#define LOWBIT_DENSE_CFG_ROWS\s+(\d+)\s*$", text, re.M)
    rows = re.findall(r"^  X\((\d+,\d+,\d+,\d+,\d+,\d+),B\)\s*\\?\s*$", text, re.M)
    if not stamped or len(rows) < 2:
        print("[dense-units] ERROR: cannot form controls from the table's own count and rows")
        return 1
    row_count = int(stamped.group(1))

    for batch in (1, 2, 4, 8):
        work, table, build, result = configure(text, batch)
        try:
            if result.returncode:
                raise RuntimeError((result.stdout + result.stderr)[-1600:])
            got_rows, got_batches = map(int, (build / "summary.txt").read_text().strip().split(","))
            expected_batches = (row_count + batch - 1) // batch
            if (got_rows, got_batches) != (row_count, expected_batches):
                raise RuntimeError(
                    f"k={batch}: parsed/batched {(got_rows, got_batches)}, expected {(row_count, expected_batches)}")
            dependency_file = build / "CMakeFiles" / "Makefile.cmake"
            if not dependency_file.is_file() or str(table) not in dependency_file.read_text(errors="replace"):
                raise RuntimeError("the dense .inc is absent from CMAKE_CONFIGURE_DEPENDS")
        finally:
            shutil.rmtree(work, ignore_errors=True)

    first = f"  X({rows[0]},B)"
    second = f"  X({rows[1]},B)"
    expect_rejected(text.replace(first, second, 1), "duplicate tactic row", "duplicate row")
    malformed = re.sub(r"\d+(?=,B\)$)", "0 + 1", first)
    expect_rejected(text.replace(first, malformed, 1), "malformed tactic row", "non-integer row")
    expect_rejected(text.replace(stamped.group(0), f"#define LOWBIT_DENSE_CFG_ROWS  {row_count - 1}", 1),
                    f"parsed {row_count} tactic rows", "stamped count mismatch")
    expect_rejected(text.replace("#define LOWBIT_DENSE_CFG_LIST(X, B) \\",
                                 "#define LOWBIT_DENSE_CFG_LIST(X,B) \\", 1),
                    "malformed LOWBIT_DENSE_CFG_LIST declaration", "malformed list declaration")

    print(f"[dense-units] PASS: {row_count} rows; k=1/2/4/8 batch counts, configure dependency, and fail-closed controls verified")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[dense-units] FAIL: {exc}")
        raise SystemExit(1)
