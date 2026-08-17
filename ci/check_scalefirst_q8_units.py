#!/usr/bin/env python3
"""Exercise the production CMake X-macro parser on the full Q8 authority."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
TABLE = ROOT / "benchmarks" / "lowbit_dense_i8_configs.inc"
MODULE = ROOT / "quactlize" / "csrc" / "TacticTableUnits.cmake"


def main() -> int:
    out = pathlib.Path("/workspace") / f"quactlize-scalefirst-q8-units-{os.getpid()}"
    if out.exists():
        print(f"[scalefirst-q8-units] FAIL: refusing stale directory {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    try:
        script = out / "check.cmake"
        script.write_text(f'''cmake_minimum_required(VERSION 3.19)
include("{MODULE}")
qz_parse_tactic_xmacro(ROWS FILE "{TABLE}"
  LIST_MACRO LOWBIT_DENSE_I8_CFG_LIST COUNT_MACRO LOWBIT_DENSE_I8_CFG_ROWS)
list(LENGTH ROWS COUNT)
if(NOT COUNT EQUAL 2501)
  message(FATAL_ERROR "Q8 parser returned ${{COUNT}} rows, expected 2501")
endif()
set(BC1 0)
foreach(ROW IN LISTS ROWS)
  string(REPLACE "," ";" FIELDS "${{ROW}}")
  list(LENGTH FIELDS FIELD_COUNT)
  if(NOT FIELD_COUNT EQUAL 7)
    message(FATAL_ERROR "row '${{ROW}}' has ${{FIELD_COUNT}} fields")
  endif()
  list(GET FIELDS 6 BC)
  if(NOT BC EQUAL 0)
    math(EXPR BC1 "${{BC1}} + 1")
  endif()
endforeach()
if(NOT BC1 EQUAL 0)
  message(FATAL_ERROR "Q8 authority contains ${{BC1}} B-chunk rows")
endif()
message(STATUS "Q8_CMAKE_ROWS=${{COUNT}} BC1=${{BC1}}")
''')
        ran = subprocess.run(["cmake", "-P", str(script)], cwd=ROOT,
                             text=True, capture_output=True)
        if ran.returncode or "Q8_CMAKE_ROWS=2501 BC1=0" not in ran.stdout + ran.stderr:
            print(ran.stdout + ran.stderr, file=sys.stderr)
            print("[scalefirst-q8-units] FAIL: production parser contract", file=sys.stderr)
            return 2
    finally:
        shutil.rmtree(out)
    print("[scalefirst-q8-units] PASS: production CMake parser sees 2501 exact bc0 rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
