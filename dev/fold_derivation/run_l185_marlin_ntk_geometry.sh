#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L185_OUT:-/workspace/quactlize-l185}"
mkdir -p "$out"

flags=(-std=c++17 -x cu -arch=sm_80 -w --expt-relaxed-constexpr
  -I "$repo/dev/fold_derivation/stub_inc"
  -I "$repo/third_party/actlize/include"
  -I "$repo/third_party/actlize/tools/util/include"
  -I "$repo/quactlize/include")
source="$repo/dev/fold_derivation/l185_marlin_ntk_geometry.cu"
nvcc "${flags[@]}" -o "$out/l185" "$source"
"$out/l185"

plants=(old-m8-k-cadence drop-last-a-round zero-scale-phase local-output-tile)
for plant in "${plants[@]}"; do
  log="$out/${plant}.log"
  set +e
  "$out/l185" "--plant=$plant" >"$log" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 1 ]] || ! grep -Fq "L185 EXPECTED-RED plant=$plant" "$log"; then
    cat "$log" >&2
    echo "[l185] FAIL: plant $plant did not produce its named RED" >&2
    exit 1
  fi
done

# One changed variable: return the production permutation tile to the old
# fixed 32x64 spelling.  Non-default N/K geometry must fail the exact type
# assertion, proving that a parameterized thread layout with a stale
# permutation cannot pass as a real axis.
overlay="$out/overlay-fixed-permutation"
rel=actlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp
mkdir -p "$overlay/$(dirname "$rel")"
cp "$repo/quactlize/include/$rel" "$overlay/$rel"
python3 - "$overlay/$rel" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
needle = "cute::Int<M::InstructionM>"  # ensure this is not the production file
old = "cute::Int<InstructionM>, cute::Int<16 * WarpOnN>,\n          cute::Int<16 * WarpOnK>"
new = "cute::Int<InstructionM>, cute::_32, cute::_64"
if s.count(old) != 1:
    raise SystemExit("L185 fixed-permutation seam is not unique")
p.write_text(s.replace(old, new, 1))
PY
set +e
nvcc -I "$overlay" "${flags[@]}" -o "$out/fixed-permutation" "$source" \
  >"$out/fixed-permutation.log" 2>&1
rc=$?
set -e
if [[ $rc -eq 0 ]] || ! grep -Fq \
    "production TiledMma permutation does not scale with N/K cohorts" \
    "$out/fixed-permutation.log"; then
  sed -n '1,80p' "$out/fixed-permutation.log" >&2
  echo '[l185] FAIL: fixed permutation type control did not turn RED' >&2
  exit 1
fi

# Reach representative rows through the production generated-unit wrapper,
# not just through the host oracle.  The local NVIDIA front end cannot cleanly
# finish even the pre-existing TN128/TK128 baseline unit: it reports two
# int_tuple.hpp product-object diagnostics after full instantiation.  L169's
# established causal proof therefore remains the honest local boundary: plant
# one dependent assertion at run_segment(), require every real row to reach it,
# then sever only the generated wrapper edge and require a clean compile.
#
# The set includes the two-round m16 TN64/TK128 A producer, both m8/m16
# active-prefix TN256/TK64 producers, and the two-tactic-tiles-per-gs128
# TN128/TK64 cadence.
unit_source="$repo/dev/fold_derivation/l185_marlin_ntk_generated_unit.cu"
unit_overlay="$out/overlay-generated-unit"
unit_collective_rel=actlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp
unit_collective="$unit_overlay/$unit_collective_rel"
unit_wrapper="$unit_overlay/lowbit_dense_unit.inc"
mkdir -p "$(dirname "$unit_collective")"
cp "$repo/quactlize/include/$unit_collective_rel" "$unit_collective"
cp "$repo/benchmarks/lowbit_dense_unit.inc" "$unit_wrapper"
python3 - "$unit_collective" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
namespace = "namespace marlin_ppu_detail {"
if text.count(namespace) != 1:
    raise SystemExit("L185 marlin_ppu_detail namespace seam is not unique")
helper = '''namespace marlin_ppu_detail {

template <class L185Accumulator>
CUTLASS_DEVICE void l185_device_body_marker(L185Accumulator const&) {
  static_assert(sizeof(L185Accumulator) == 0,
                "L185_NTK_DEVICE_BODY_INSTANTIATED");
}'''
text = text.replace(namespace, helper, 1)
needle = '''  CUTLASS_DEVICE static void run_segment(
      CtaState const& state, SegmentState const& segment,
      SharedBases const& shared, Accumulator& accum) {'''
if text.count(needle) != 1:
    raise SystemExit("L185 collective run_segment seam is not unique")
text = text.replace(
    needle, needle + "\n    marlin_ppu_detail::l185_device_body_marker(accum);", 1)
path.write_text(text)
PY

unit_flags=(
  -std=c++17 -arch=sm_80 --expt-relaxed-constexpr -D__HGGCCC__ -w
  -Xcudafe --error_limit=100000
  -DDENSE_MARLIN_STANDALONE_SWEEP=1
  -DDENSE_MARLIN_AB=1 -DDENSE_STREAMK_AB=1
  -DBENCH_GS=128 -DBENCH_TSK=64
  -DDENSE_AB_BITS=4 -DDENSE_AB_ARTIFACT_TK=64
  -DDENSE_AB_TM=16 -DDENSE_AB_TN=128 -DDENSE_AB_TK=128
  -DDENSE_AB_WM=16 -DDENSE_AB_WN=64 -DDENSE_AB_WARP_K=32
  -DDENSE_AB_ST=4 -DDENSE_AB_BC=0
  -DTILE_M=16 -DTILE_N=128 -DWARP_M=16 -DWARP_N=64 -DSTAGES=4
  -I "$unit_overlay"
  -I "$repo/dev/fold_derivation/stub_inc"
  -I "$repo/third_party/actlize/include"
  -I "$repo/third_party/actlize/tools/util/include"
  -I "$repo/tests" -I "$repo/benchmarks" -I "$repo/quactlize/include"
  -I "$repo/dev" -cuda -x cu
)
unit_marker='error: static assertion failed with "L185_NTK_DEVICE_BODY_INSTANTIATED"'
for unit_case in 0 1 2 3 4 5 6 7; do
  unit_out="$out/l185-generated-unit-${unit_case}.cu.cpp"
  unit_log="$out/l185-generated-unit-${unit_case}.log"
  set +e
  nvcc "${unit_flags[@]}" -DL185_UNIT_CASE="$unit_case" \
      "$unit_source" -o "$unit_out" >"$unit_log" 2>&1
  unit_rc=$?
  set -e
  if [[ $unit_rc -eq 0 ]] || \
      [[ "$(grep -Fc "$unit_marker" "$unit_log" || true)" -ne 1 ]]; then
    echo "[l185] FAIL: representative production generated-unit case=$unit_case did not reach its exact device body once" >&2
    sed -n '1,80p' "$unit_log" >&2
    exit 1
  fi
  unexpected="$out/l185-generated-unit-${unit_case}-unexpected.log"
  grep -E ': (error|fatal error|catastrophic error):' "$unit_log" \
    | grep -Fv "$unit_marker" >"$unexpected" || true
  if [[ -s "$unexpected" ]]; then
    echo "[l185] FAIL: device-body case=$unit_case carried an unrelated compiler error" >&2
    sed -n '1,40p' "$unexpected" >&2
    exit 1
  fi
done

# Same eight types, one changed variable: leave the marker in place but sever
# the production standalone wrapper's call to run<G>.  Every compile must now
# finish and the marker must disappear.
python3 - "$unit_wrapper" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
start = text.index("#if defined(DENSE_MARLIN_STANDALONE_SWEEP)")
end = text.index("\n#else", start)
arm = text[start:end]
needle = '  return run<G>(options, dense_tactic(cfg), "marlin");'
if arm.count(needle) != 1:
    raise SystemExit("L185 standalone wrapper call seam is not unique")
arm = arm.replace(needle, "  return {};  // L185 route-severed control")
path.write_text(text[:start] + arm + text[end:])
PY
for unit_case in 0 1 2 3 4 5 6 7; do
  unit_out="$out/l185-route-severed-${unit_case}.cu.cpp"
  unit_log="$out/l185-route-severed-${unit_case}.log"
  if ! nvcc "${unit_flags[@]}" -DL185_UNIT_CASE="$unit_case" \
      "$unit_source" -o "$unit_out" >"$unit_log" 2>&1; then
    echo "[l185] FAIL: route-severed generated-unit case=$unit_case did not compile cleanly" >&2
    sed -n '1,80p' "$unit_log" >&2
    exit 1
  fi
  test -s "$unit_out" || {
    echo "[l185] FAIL: route-severed generated-unit case=$unit_case produced no artifact" >&2
    exit 1
  }
  if grep -Fq 'L185_NTK_DEVICE_BODY_INSTANTIATED' "$unit_log"; then
    echo "[l185] FAIL: device-body marker survived route severing for case=$unit_case" >&2
    exit 1
  fi
done

echo '[l185:runner] positive=70-types/full-domains production-device-bodies=8/8_REACHED route-severed=8/8_CLEAN negative=5/5_RED result=PASS'
