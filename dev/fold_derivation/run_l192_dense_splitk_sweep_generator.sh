#!/usr/bin/env bash
# Execute the production CMake generator under cmake -P, then link all emitted
# wrapper symbols through the same named multi-TU seam as the PPU target.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cml="${repo}/quactlize/csrc/CMakeLists.txt.in"
module="${repo}/quactlize/csrc/TacticTableUnits.cmake"
base="${QUACTLIZE_L192_OUT:-/workspace/quactlize-l192-dense-splitk-generator}"
out="${base}/run-$$"
mkdir -p "${out}"

minimum="$(grep -m1 -oE '^cmake_minimum_required\(VERSION [0-9.]+' \
  "${repo}/CMakeLists.txt" | grep -oE '[0-9.]+$')"
awk '/^function\(qz_resolve_sources/{p=1} p{print} p&&/^endfunction\(\)/{exit}' \
  "${cml}" >"${out}/resolve.cmake"
awk '/^# Final dense fixed Split-K performance search\./{p=1} \
     p&&/^quactlize_ppu_executable\(/{exit} p{print}' \
  "${cml}" >"${out}/generator.cmake"
grep -Fq 'qz_parse_tactic_xmacro(_DENSE_SPLITK_SOURCE_ROWS' \
  "${out}/generator.cmake" || {
  echo '[l192] FAIL: production generator anchors moved' >&2
  exit 1
}

{
  printf 'cmake_minimum_required(VERSION %s)\n' "${minimum}"
  printf 'set(CMAKE_CURRENT_BINARY_DIR "%s")\n' "${out}/generated"
  printf 'set(CMAKE_CURRENT_SOURCE_DIR "%s")\n' "${repo}/quactlize/csrc"
  printf 'set(QZ_ROOT "%s")\n' "${repo}"
  printf 'set(QZ_SRC_DIRS "%s/tests" "%s/benchmarks" "%s/dev" "%s/quactlize/csrc/device" "%s/quactlize/csrc")\n' \
    "${repo}" "${repo}" "${repo}" "${repo}" "${repo}"
  printf 'include("%s")\n' "${out}/resolve.cmake"
  printf 'include("%s")\n' "${module}"
  printf 'file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")\n'
  printf 'include("%s")\n' "${out}/generator.cmake"
} >"${out}/run.cmake"

cmake -P "${out}/run.cmake" >"${out}/cmake.log" 2>&1 || {
  echo '[l192] FAIL: production CMake generator did not execute' >&2
  sed -n '1,140p' "${out}/cmake.log" >&2
  exit 1
}

python3 - "${repo}" "${out}" <<'PY'
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

repo, out = map(Path, sys.argv[1:])
gen = out / "generated"
header = gen / "dense_splitk_sweep_configs.inc"
units = sorted(
    (gen / "dense_splitk_sweep_units").glob("dense_splitk_sweep_unit_*.cu"),
    key=lambda path: int(path.stem.rsplit("_", 1)[1]),
)
if not header.is_file() or len(units) != 51:
    raise SystemExit(f"L192 generator: header={header.is_file()} units={len(units)}, expected true/51")

row_re = re.compile(r"X\((dense_splitk_cfg_[^,]+),([0-9,]+)\)")
header_symbols = row_re.findall(header.read_text())
if len(header_symbols) != 201 or len({name for name, _ in header_symbols}) != 201:
    raise SystemExit(f"L192 generator: header rows/unique={len(header_symbols)}/{len(set(header_symbols))}, expected 201/201")

unit_symbols: list[str] = []
sizes: list[int] = []
for unit in units:
    matches = row_re.findall(unit.read_text())
    sizes.append(len(matches))
    unit_symbols.extend(name for name, _ in matches)
if sizes[:-1] != [4] * 50 or sizes[-1:] != [1]:
    raise SystemExit(f"L192 generator: batch sizes={sizes}, expected 50x4+1")
if sorted(unit_symbols) != sorted(name for name, _ in header_symbols):
    raise SystemExit("L192 generator: unit symbol multiset differs from registry header")

main_source = repo / "benchmarks/test_lowbit_dense_splitk_sweep.cu"
if '#include "dense_splitk_sweep_configs.inc"' not in main_source.read_text():
    raise SystemExit("L192 generator: main no longer consumes generated registry header")

# Tiny host ABI link uses the exact generated symbol names and the production
# named namespace.  It catches the anonymous-namespace declaration regression
# without compiling any substitute kernel arithmetic.
decls = "\n".join(f"int {name}();" for name, _ in header_symbols)
calls = " +\n".join(f"dense_splitk_sweep_generated::{name}()" for name, _ in header_symbols)
(out / "link-main.cpp").write_text(
    "namespace dense_splitk_sweep_generated {\n" + decls + "\n}\n"
    "int main(){ return (" + calls + ") == 201 ? 0 : 1; }\n"
)
objects: list[str] = []
for index, unit in enumerate(units):
    names = [name for name, _ in row_re.findall(unit.read_text())]
    source = out / f"link-unit-{index}.cpp"
    source.write_text(
        "namespace dense_splitk_sweep_generated {\n" +
        "\n".join(f"int {name}(){{return 1;}}" for name in names) +
        "\n}\n"
    )
    obj = out / f"link-unit-{index}.o"
    subprocess.run(["g++", "-std=c++17", "-w", "-c", str(source), "-o", str(obj)], check=True)
    objects.append(str(obj))
main_obj = out / "link-main.o"
subprocess.run(["g++", "-std=c++17", "-w", "-c", str(out / "link-main.cpp"), "-o", str(main_obj)], check=True)
binary = out / "link-positive"
subprocess.run(["g++", str(main_obj), *objects, "-o", str(binary)], check=True)
subprocess.run([str(binary)], check=True)

# Same definitions, one changed property: declarations regain internal
# linkage.  It must reproduce the undefined-reference class that this seam
# exists to prevent.
(out / "link-negative.cpp").write_text(
    "namespace {\n" + decls + "\n}\n"
    "int main(){ return (" + calls.replace("dense_splitk_sweep_generated::", "") +
    ") == 201 ? 0 : 1; }\n"
)
negative_obj = out / "link-negative.o"
subprocess.run(["g++", "-std=c++17", "-w", "-c", str(out / "link-negative.cpp"), "-o", str(negative_obj)], check=True)
negative = subprocess.run(
    ["g++", str(negative_obj), *objects, "-o", str(out / "link-negative")],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
if negative.returncode == 0 or "undefined reference" not in negative.stdout:
    raise SystemExit("L192 generator: anonymous-namespace negative did not reproduce undefined references")

print("[l192] generated=201 unique=201 units=51 batches=50x4+1 "
      "positive-multitu-link=PASS anonymous-declarations=EXPECTED-UNDEFINED")
PY

echo "[l192] PASS: production CMake generator and multi-TU seam exact; artifacts=${out}"
