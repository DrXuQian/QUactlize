#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tmp="$(mktemp -d /tmp/quactlize-l180.XXXXXX)"
trap 'rm -rf "${tmp}"' EXIT

flags=(-std=c++17 -x cu -arch=sm_80 -w --expt-relaxed-constexpr
  -D__HGGCCC__
  -I "${repo}/dev/fold_derivation/stub_inc"
  -I "${repo}/third_party/actlize/include"
  -I "${repo}/third_party/actlize/tools/util/include"
  -I "${repo}/quactlize/include")
source="${repo}/dev/fold_derivation/l180_marlin_scheduler_hot_state.cpp"
rel=quactlize_extensions/cutlass/gemm/kernel/marlin_scheduler_ppu.hpp
production="${repo}/quactlize/include/${rel}"

nvcc "${flags[@]}" -o "${tmp}/positive" "${source}"
"${tmp}/positive"

# Each control mutates the real Args->device-state lowering seam in an include
# overlay, then rebuilds the same exhaustive oracle.  A parallel test-only
# model cannot make any of these controls red.
for plant in kt-from-output output-from-active total-from-active \
             iters-from-active invalid-survives; do
  overlay="${tmp}/overlay-${plant}"
  probe="${overlay}/${rel}"
  mkdir -p "$(dirname "${probe}")"
  cp "${production}" "${probe}"
  python3 - "${probe}" "${plant}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
plant = sys.argv[2]
text = path.read_text()
changes = {
    "kt-from-output": (
        "p.k_tiles_per_output_, p.output_tiles_, p.total_k_tiles_,",
        "p.output_tiles_, p.output_tiles_, p.total_k_tiles_,"),
    "output-from-active": (
        "p.k_tiles_per_output_, p.output_tiles_, p.total_k_tiles_,",
        "p.k_tiles_per_output_, p.active_blocks_, p.total_k_tiles_,"),
    "total-from-active": (
        "p.output_tiles_, p.total_k_tiles_,\n              p.iters_per_block_}",
        "p.output_tiles_, p.active_blocks_,\n              p.iters_per_block_}"),
    "iters-from-active": (
        "p.total_k_tiles_,\n              p.iters_per_block_}",
        "p.total_k_tiles_,\n              p.active_blocks_}"),
    "invalid-survives": (
        "return p.valid_\n        ? DeviceTraversalState{",
        "return true\n        ? DeviceTraversalState{"),
}
needle, replacement = changes[plant]
if text.count(needle) != 1:
    raise SystemExit(
        f"L180 production seam is not unique: {plant} count={text.count(needle)}")
path.write_text(text.replace(needle, replacement, 1))
PY
  set +e
  nvcc -I "${overlay}" "${flags[@]}" -o "${tmp}/${plant}" "${source}" \
    >"${tmp}/${plant}.compile.log" 2>&1
  compile_rc=$?
  set -e
  if [[ ${compile_rc} -ne 0 ]]; then
    if grep -Eq 'Marlin scheduler must (reproduce|preserve)' \
         "${tmp}/${plant}.compile.log"; then
      echo "[l180:red] plant=${plant} phase=compile production-reference=RED"
      continue
    fi
    sed -n '1,100p' "${tmp}/${plant}.compile.log" >&2
    echo "[l180] FAIL: ${plant} failed to compile for an unrelated reason" >&2
    exit 1
  fi
  set +e
  "${tmp}/${plant}" >"${tmp}/${plant}.log" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -ne 1 ]] || ! grep -Fq '[l180:red]' "${tmp}/${plant}.log"; then
    cat "${tmp}/${plant}.log" >&2
    echo "[l180] FAIL: production lowering plant ${plant} was not rejected" >&2
    exit 1
  fi
done

# The state itself must stay pointer-free and four words.  Adding the lock
# pointer back is rejected by the production ABI assertion at compile time.
overlay="${tmp}/overlay-lock-pointer"
probe="${overlay}/${rel}"
mkdir -p "$(dirname "${probe}")"
cp "${production}" "${probe}"
python3 - "${probe}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
needle = "uint32_t iters_per_block_ = 0;\n  };\n\n  enum class PeerReleaseAction"
replacement = (
    "uint32_t iters_per_block_ = 0;\n"
    "    BarrierType* locks_ = nullptr;  // L180 forbidden hot pointer\n"
    "  };\n\n  enum class PeerReleaseAction")
if text.count(needle) != 1:
    raise SystemExit("L180 lock-pointer plant seam is not unique")
path.write_text(text.replace(needle, replacement, 1))
PY
set +e
nvcc -I "${overlay}" "${flags[@]}" -o "${tmp}/lock-pointer" "${source}" \
  >"${tmp}/lock-pointer.log" 2>&1
rc=$?
set -e
if [[ ${rc} -eq 0 ]] ||
   ! grep -Fq 'standalone Marlin device traversal state ABI changed' \
     "${tmp}/lock-pointer.log"; then
  sed -n '1,80p' "${tmp}/lock-pointer.log" >&2
  echo '[l180] FAIL: device hot-state lock pointer was not rejected' >&2
  exit 1
fi

# Replacing the private hot object with the whole public Params is a source-
# compatible regression after also restoring its constructor.  Only the
# scheduler-object ABI assertion distinguishes it from the intended design.
overlay="${tmp}/overlay-hot-params"
probe="${overlay}/${rel}"
mkdir -p "$(dirname "${probe}")"
cp "${production}" "${probe}"
python3 - "${probe}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
changes = (
    ("DeviceTraversalState traversal_state_{};", "Params traversal_state_{};"),
    (": traversal_state_(make_device_traversal_state(p)) {}",
     ": traversal_state_(p) {}"),
)
for needle, replacement in changes:
    if text.count(needle) != 1:
        raise SystemExit(f"L180 hot-Params plant seam is not unique: {needle!r}")
    text = text.replace(needle, replacement, 1)
path.write_text(text)
PY
set +e
nvcc -I "${overlay}" "${flags[@]}" -o "${tmp}/hot-params" "${source}" \
  >"${tmp}/hot-params.log" 2>&1
rc=$?
set -e
if [[ ${rc} -eq 0 ]] ||
   ! grep -Fq 'L180 shipping scheduler object must contain only hot traversal state' \
     "${tmp}/hot-params.log"; then
  sed -n '1,80p' "${tmp}/hot-params.log" >&2
  echo '[l180] FAIL: whole public Params survived in the device scheduler object' >&2
  exit 1
fi

echo '[l180:runner] positive=262144-schedule-equivalence negative_controls=7/7_RED result=PASS'
