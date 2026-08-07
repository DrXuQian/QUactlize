#!/usr/bin/env python3
"""Exercise the dense CMake table parser/batcher and its fail-closed controls for every format table."""

from collections import Counter
import pathlib
import re
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE = ROOT / "quactlize" / "csrc" / "TacticTableUnits.cmake"
TABLE_DIR = ROOT / "benchmarks"
PREFIX_RE = re.compile(r"^#define (LOWBIT_DENSE(?:_[A-Z0-9]+)?)_CFG_ROWS\s+(\d+)\s*$", re.M)
ROW_RE = re.compile(r"^  X\((\d+(?:,\d+){6}),B\)\s*\\?\s*$", re.M)


def configure(table_name: str, table_text: str, prefix: str, batch: int):
    work = pathlib.Path(tempfile.mkdtemp(prefix="dense-unit-generator-"))
    src, build = work / "src", work / "build"
    src.mkdir()
    table = src / table_name
    table.write_text(table_text)
    (src / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.19)
project(dense_unit_generator_check NONE)
include("{MODULE}")
qz_parse_tactic_xmacro(rows FILE "{table}"
  LIST_MACRO {prefix}_CFG_LIST COUNT_MACRO {prefix}_CFG_ROWS)
set(batch_count 0)
foreach(batch_bc IN ITEMS 0 1)
  set(rows_bc "")
  foreach(row IN LISTS rows)
    string(REPLACE "," ";" fields "${{row}}")
    list(LENGTH fields field_count)
    if(NOT field_count EQUAL 7)
      message(FATAL_ERROR "internal dense row '${{row}}' has ${{field_count}} fields, expected 7")
    endif()
    list(GET fields 6 bc)
    if(bc STREQUAL batch_bc)
      list(APPEND rows_bc "${{row}}")
    endif()
  endforeach()
  if(rows_bc)
    qz_batch_tactic_rows(batches_bc rows_bc "{batch}")
    list(LENGTH batches_bc batches_bc_count)
    math(EXPR batch_count "${{batch_count}} + ${{batches_bc_count}}")
  endif()
endforeach()
list(LENGTH rows row_count)
file(WRITE "${{CMAKE_BINARY_DIR}}/summary.txt" "${{row_count}},${{batch_count}}\n")
""")
    result = subprocess.run(["cmake", "-S", src, "-B", build], capture_output=True, text=True)
    return work, table, build, result


def expect_rejected(table_name: str, text: str, prefix: str, fragment: str, label: str) -> None:
    work, _, _, result = configure(table_name, text, prefix, 4)
    try:
        output = result.stdout + result.stderr
        normalized = " ".join(output.split())
        if result.returncode == 0 or " ".join(fragment.split()) not in normalized:
            raise RuntimeError(f"{table_name}: {label} was not rejected for the expected reason "
                               f"({fragment!r})\n{output[-1200:]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def check_table(table: pathlib.Path) -> tuple[int, int]:
    text = table.read_text()
    stamped = PREFIX_RE.search(text)
    rows = ROW_RE.findall(text)
    if not stamped or len(rows) < 2:
        raise RuntimeError(f"{table.name}: cannot form controls from the table's own count and seven-field rows")
    prefix, stamped_count = stamped.group(1), int(stamped.group(2))
    row_count = len(rows)
    if row_count != stamped_count:
        raise RuntimeError(f"{table.name}: stamps {stamped_count} rows but contains {row_count}")

    bc_counts = Counter(int(row.split(",")[-1]) for row in rows)
    if set(bc_counts) - {0, 1}:
        raise RuntimeError(f"{table.name}: unexpected PPU_B_CHUNK fields {sorted(bc_counts)}")

    units_at_four = 0
    for batch in (1, 2, 4, 8):
        work, copied_table, build, result = configure(table.name, text, prefix, batch)
        try:
            if result.returncode:
                raise RuntimeError(f"{table.name}: {(result.stdout + result.stderr)[-1600:]}")
            got_rows, got_batches = map(int, (build / "summary.txt").read_text().strip().split(","))
            expected_batches = sum((count + batch - 1) // batch for count in bc_counts.values())
            if (got_rows, got_batches) != (row_count, expected_batches):
                raise RuntimeError(f"{table.name}: k={batch}: parsed/batched {(got_rows, got_batches)}, "
                                   f"expected {(row_count, expected_batches)} after bc partitioning")
            dependency_file = build / "CMakeFiles" / "Makefile.cmake"
            if not dependency_file.is_file() or str(copied_table) not in dependency_file.read_text(errors="replace"):
                raise RuntimeError(f"{table.name}: the dense .inc is absent from CMAKE_CONFIGURE_DEPENDS")
            if batch == 4:
                units_at_four = expected_batches
        finally:
            shutil.rmtree(work, ignore_errors=True)

    first = f"  X({rows[0]},B)"
    second = f"  X({rows[1]},B)"
    expect_rejected(table.name, text.replace(first, second, 1), prefix,
                    "duplicate tactic row", "duplicate row")
    malformed = re.sub(r"\d+(?=,B\)$)", "0 + 1", first)
    expect_rejected(table.name, text.replace(first, malformed, 1), prefix,
                    "malformed tactic row", "non-integer row")
    expect_rejected(table.name,
                    text.replace(stamped.group(0), f"#define {prefix}_CFG_ROWS  {row_count - 1}", 1), prefix,
                    f"parsed {row_count} tactic rows", "stamped count mismatch")
    expect_rejected(table.name,
                    text.replace(f"#define {prefix}_CFG_LIST(X, B) \\",
                                 f"#define {prefix}_CFG_LIST(X,B) \\", 1), prefix,
                    f"malformed {prefix}_CFG_LIST declaration", "malformed list declaration")
    return row_count, units_at_four


def main() -> int:
    if not shutil.which("cmake"):
        print("[dense-units] ERROR: cmake is not installed; the generator cannot be checked")
        return 1
    tables = sorted(TABLE_DIR.glob("lowbit_dense*_configs.inc"))
    if not MODULE.is_file() or not tables:
        print("[dense-units] ERROR: parser module or dense format tables are missing")
        return 1

    total_rows = 0
    summaries = []
    for table in tables:
        row_count, units = check_table(table)
        total_rows += row_count
        summaries.append(f"{table.name}={row_count} rows/{units} units@k4")
    print(f"[dense-units] PASS: {len(tables)} format table(s), {total_rows} rows; "
          "bc-partitioned k=1/2/4/8 batches, configure dependencies, and fail-closed controls verified; "
          + "; ".join(summaries))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[dense-units] FAIL: {exc}")
        raise SystemExit(1)
