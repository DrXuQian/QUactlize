#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
variant="${QUACTLIZE_L169_VARIANT:-m16}"
pipe_roll="${QUACTLIZE_L169_PIPE_ROLL:-0}"
case "$pipe_roll" in
  0|1|2) ;;
  *) echo "[l169] FAIL: unknown pipe-roll mode '$pipe_roll'" >&2; exit 2 ;;
esac
case "$variant" in
  m16)
    source_file="$repo/dev/fold_derivation/l169_standalone_marlin_unit.cu"
    defs='-DDENSE_MARLIN_WK4_AB=1 -DDENSE_MARLIN_AB=1 -DDENSE_STREAMK_AB=1 -DBENCH_GS=128 -DBENCH_TSK=64 -DDENSE_AB_BITS=4 -DDENSE_AB_ARTIFACT_TK=64 -DDENSE_AB_TM=16 -DDENSE_AB_TN=128 -DDENSE_AB_TK=128 -DDENSE_AB_WM=16 -DDENSE_AB_WN=64 -DDENSE_AB_WARP_K=32 -DDENSE_AB_ST=4 -DDENSE_AB_BC=0 -DTILE_M=16 -DTILE_N=128 -DWARP_M=16 -DWARP_N=64 -DSTAGES=4'
    expected_accumulator='L169Accumulator=cutlass::gemm::collective::marlin_ppu_detail::MarlinAccumulatorPPU'
    expected_tile='TileShape_=cute::tuple<cute::_16, cute::_128, cute::_128>'
    expected_warp='WarpShape_=cute::tuple<cute::C<16>, cute::C<64>, cute::C<32>>'
    ;;
  m8)
    source_file="$repo/dev/fold_derivation/l181_standalone_marlin_m8_unit.cu"
    # DENSE_AB_* is the real standalone m8 row.  TILE_M/WARP_M only feed
    # unused legacy vendor helper aliases in the shared benchmark TU; keep
    # those m16 so the oracle matches the production target's two authorities.
    defs='-DDENSE_MARLIN_WK4_AB=1 -DDENSE_MARLIN_M8_AB=1 -DDENSE_MARLIN_AB=1 -DDENSE_STREAMK_AB=1 -DBENCH_GS=128 -DBENCH_TSK=64 -DDENSE_AB_BITS=4 -DDENSE_AB_ARTIFACT_TK=64 -DDENSE_AB_TM=8 -DDENSE_AB_TN=128 -DDENSE_AB_TK=128 -DDENSE_AB_WM=8 -DDENSE_AB_WN=64 -DDENSE_AB_WARP_K=32 -DDENSE_AB_ST=4 -DDENSE_AB_BC=0 -DTILE_M=16 -DTILE_N=128 -DWARP_M=16 -DWARP_N=64 -DSTAGES=4'
    expected_accumulator='L169Accumulator=cutlass::gemm::collective::marlin_ppu_detail::MarlinAccumulatorM8PPU'
    expected_tile='TileShape_=cute::tuple<cute::C<8>, cute::C<128>, cute::C<128>>'
    expected_warp='WarpShape_=cute::tuple<cute::C<8>, cute::C<64>, cute::C<32>>'
    ;;
  *) echo "[l169] FAIL: unknown variant '$variant'" >&2; exit 2 ;;
esac
if [[ "$pipe_roll" != 0 ]]; then
  defs+=" -DPPU_MARLIN_PIPE_ROLL=$pipe_roll"
fi

command -v nvcc >/dev/null 2>&1 || {
  echo '[l169] FAIL: nvcc is required for the generated-unit compile oracle' >&2
  exit 1
}

tmp="${QUACTLIZE_L169_OUT:-/workspace/quactlize-l169-${variant}}"
mkdir -p "$tmp"

overlay="$tmp/overlay"
collective_rel=actlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp
kernel_rel=actlize_extensions/cutlass/gemm/kernel/marlin_kernel_ppu.hpp
collective_src="$repo/quactlize/include/$collective_rel"
collective_probe="$overlay/$collective_rel"
kernel_src="$repo/quactlize/include/$kernel_rel"
kernel_probe="$overlay/$kernel_rel"
unit_probe="$overlay/lowbit_dense_unit.inc"
mkdir -p "$(dirname "$collective_probe")" "$(dirname "$kernel_probe")"
cp "$collective_src" "$collective_probe"
cp "$kernel_src" "$kernel_probe"
cp "$repo/benchmarks/lowbit_dense_unit.inc" "$unit_probe"

# Do not infer device-body instantiation from an unrelated warning or from an
# environmental error's template stack.  Instead, put one uniquely named
# call to a dependent failing helper at the entrance to the production
# collective's run_segment() in a TEMPORARY include overlay.  Keeping the
# assertion in a function template matters: a non-template member assertion
# fires when the enclosing class is formed and would make the route-severed
# control a false red.  The source tree remains untouched.  The exact generated
# TU must instantiate the helper through:
#   lowbit_dense_run_config -> run<G> -> maximum_active_blocks
#     -> device_kernel -> MarlinKernelPPU::operator() -> run_segment().
python3 - "$collective_probe" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
namespace = "namespace marlin_ppu_detail {"
if text.count(namespace) != 1:
    raise SystemExit("L169 marlin_ppu_detail namespace seam is not unique")
helper = '''namespace marlin_ppu_detail {

template <class L169Accumulator>
CUTLASS_DEVICE void l169_device_body_marker(L169Accumulator const&) {
  static_assert(sizeof(L169Accumulator) == 0,
                "L169_DEVICE_BODY_INSTANTIATED");
}'''
text = text.replace(namespace, helper, 1)
needle = """  CUTLASS_DEVICE static void run_segment(
      CtaState const& state, SegmentState const& segment,
      SharedBases const& shared, Accumulator& accum) {"""
if text.count(needle) != 1:
    raise SystemExit("L169 collective run_segment seam is not unique")
plant = (
    needle
    + '\n    marlin_ppu_detail::l169_device_body_marker(accum);'
)
path.write_text(text.replace(needle, plant), encoding="utf-8")
PY

# shellcheck disable=SC2206  # defs is the intentional compile-definition list.
def_args=($defs)
common=(
  -std=c++17 -arch=sm_80 --expt-relaxed-constexpr -D__HGGCCC__
  "${def_args[@]}" -Xcudafe --error_limit=100000
  -I"$overlay"
  -I"$repo/dev/fold_derivation/stub_inc"
  -I"$repo/third_party/actlize/include"
  -I"$repo/third_party/actlize/tools/util/include"
  -I"$repo/tests" -I"$repo/benchmarks" -I"$repo/quactlize/include"
  -I"$repo/dev" -cuda -x cu "$source_file" -Wno-deprecated-gpu-targets
)

compile_probe() {
  local name="$1"
  set +e
  nvcc "${common[@]}" -o "$tmp/$name.cu.cpp" >"$tmp/$name.log" 2>&1
  local rc=$?
  set -e
  printf '%s' "$rc"
}

positive_rc="$(compile_probe positive)"
[ "$positive_rc" -ne 0 ] || {
  echo '[l169] FAIL: device-body plant did not stop the generated unit' >&2
  exit 1
}
marker='error: static assertion failed with "L169_DEVICE_BODY_INSTANTIATED"'
[ "$(grep -Fc "$marker" "$tmp/positive.log" || true)" -eq 1 ] || {
  echo '[l169] FAIL: generated unit did not reach the exact device-body marker once' >&2
  sed -n '1,30p' "$tmp/positive.log" >&2
  exit 1
}
unexpected="$tmp/positive-unexpected.log"
grep -E ': (error|fatal error|catastrophic error):' "$tmp/positive.log" \
  | grep -Fv "$marker" >"$unexpected" || true
if [[ -s "$unexpected" ]]; then
  echo '[l169] FAIL: positive device-body witness carried an unrelated error' >&2
  sed -n '1,20p' "$unexpected" >&2
  exit 1
fi
for token in \
  "$expected_accumulator" \
  'MarlinCollectivePPU<TileShape_, WarpShape_, Stages_, GroupSize_' \
  "$expected_tile" \
  "$expected_warp" \
  'Stages_=4, GroupSize_=128' \
  'LoadPolicy_=cutlass::gemm::collective::MarlinCpAsyncLoadPolicyPPU' \
  'MarlinKernelPPU<ProblemShape_' \
  'instantiation of "void cutlass::device_kernel<Operator>' \
  'instantiation of "Result run<Gemm>' \
  'instantiation of "Result lowbit_dense_run_config'; do
  grep -Fq "$token" "$tmp/positive.log" || {
    echo "[l169] FAIL: device-body instantiation chain lost $token" >&2
    sed -n '1,80p' "$tmp/positive.log" >&2
    exit 1
  }
done

# Same plant, same generated TU, one changed variable: sever only the standalone
# wrapper's call to run<G>.  The compile must now finish and the marker must be
# absent.  This proves the positive is caused by the real wrapper/device-body
# edge rather than by parsing the kernel header or ambient compiler noise.
cp "$repo/benchmarks/lowbit_dense_unit.inc" "$unit_probe"
python3 - "$unit_probe" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
function = text.index("static Result lowbit_dense_run_config")
start = text.index("#if defined(DENSE_MARLIN_WK4_AB)", function)
end = text.index("\n#else", start)
arm = text[start:end]
needle = '  return run<G>(options, dense_tactic(cfg), "marlin");'
if arm.count(needle) != 1:
    raise SystemExit("L169 standalone wrapper call seam is not unique")
arm = arm.replace(needle, "  return {};  // L169 route-severed control")
path.write_text(text[:start] + arm + text[end:], encoding="utf-8")
PY
negative_rc="$(compile_probe route-severed)"
[ "$negative_rc" -eq 0 ] || {
  echo "[l169] FAIL: route-severed control did not compile cleanly (nvcc rc=$negative_rc)" >&2
  sed -n '1,30p' "$tmp/route-severed.log" >&2
  exit 1
}
[ -s "$tmp/route-severed.cu.cpp" ] || {
  echo '[l169] FAIL: route-severed control produced no completion artifact' >&2
  exit 1
}
if grep -Fq 'L169_DEVICE_BODY_INSTANTIATED' "$tmp/route-severed.log"; then
  echo '[l169] FAIL: device-body marker survived after its only generated route was severed' >&2
  exit 1
fi

# Restore the real generated wrapper, then sever only the kernel-to-collective
# call.  A second named assertion at that exact call site proves the kernel
# body still instantiated while the collective marker disappeared.  Thus a
# future unused collective type cannot masquerade as device-body coverage.
cp "$repo/benchmarks/lowbit_dense_unit.inc" "$unit_probe"
cp "$kernel_src" "$kernel_probe"
python3 - "$kernel_probe" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
needle = """        CollectiveMainloop::run_segment(
            cta_state, segment, shared_bases, accum);"""
if text.count(needle) != 1:
    raise SystemExit("L169 kernel-to-collective call seam is not unique")
plant = (
    '        static_assert(sizeof(CollectiveMainloop) == 0, '
    '"L169_KERNEL_BODY_COLLECTIVE_SEVERED");'
)
path.write_text(text.replace(needle, plant), encoding="utf-8")
PY
kernel_severed_rc="$(compile_probe collective-severed)"
[ "$kernel_severed_rc" -ne 0 ] || {
  echo '[l169] FAIL: kernel-body control did not stop after severing run_segment' >&2
  exit 1
}
kernel_marker='error: static assertion failed with "L169_KERNEL_BODY_COLLECTIVE_SEVERED"'
[ "$(grep -Fc "$kernel_marker" "$tmp/collective-severed.log" || true)" -eq 1 ] || {
  echo '[l169] FAIL: collective-severed control did not reach the exact kernel-body marker once' >&2
  sed -n '1,30p' "$tmp/collective-severed.log" >&2
  exit 1
}
if grep -Fq 'L169_DEVICE_BODY_INSTANTIATED' "$tmp/collective-severed.log"; then
  echo '[l169] FAIL: collective marker survived after the kernel call was severed' >&2
  exit 1
fi
unexpected="$tmp/collective-severed-unexpected.log"
grep -E ': (error|fatal error|catastrophic error):' "$tmp/collective-severed.log" \
  | grep -Fv "$kernel_marker" >"$unexpected" || true
if [[ -s "$unexpected" ]]; then
  echo '[l169] FAIL: collective-severed kernel witness carried an unrelated error' >&2
  sed -n '1,20p' "$unexpected" >&2
  exit 1
fi

if [[ "$variant" == m16 ]]; then
  echo '[l169] PASS: generated wrapper reaches standalone Marlin kernel + collective device bodies; route-severed and collective-severed same-source controls suppress the exact marker'
else
  echo "[l169] PASS: variant=m8 pipe_roll=$pipe_roll generated wrapper reaches standalone Marlin kernel + collective device bodies; route-severed and collective-severed same-source controls suppress the exact marker"
fi
