#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
base="${QUACTLIZE_L199_OUT:-/workspace/quactlize-l199-dense-splitk-multiformat}"
mkdir -p "${base}"
out="${base}/run-$$"
mkdir "${out}"

command -v nvcc >/dev/null 2>&1 || {
  echo '[l199] FAIL: nvcc is required for the multiformat type gate' >&2
  exit 1
}

source_file="${repo}/dev/fold_derivation/l199_dense_splitk_multiformat_type.cu"
jobs="${L199_JOBS:-2}"
case "${jobs}" in
  ''|*[!0-9]*|0) echo "[l199] FAIL: L199_JOBS must be a positive integer" >&2; exit 1 ;;
esac

base_flags=(
  -std=c++17 -arch=sm_80 --expt-relaxed-constexpr -D__HGGCCC__
  -I "${repo}/dev/fold_derivation/stub_inc"
  -I "${repo}/third_party/actlize/include"
  -I "${repo}/third_party/actlize/tools/util/include"
  -I "${repo}/quactlize/include"
  -Wno-deprecated-gpu-targets
)

packed_format_for_qtype() {
  case "$1" in
    10) printf 2 ;;
    11) printf 3 ;;
    12) printf 0 ;;
    13) printf 1 ;;
    14) printf 4 ;;
    *) return 1 ;;
  esac
}

run_arm() {
  local qtype="$1" metadata="$2" bchunk="$3"
  local label="q${qtype}-${metadata}-bc${bchunk}"
  local defs=("-DL199_QTYPE=${qtype}" "-DPPU_B_CHUNK=${bchunk}")
  if [[ "${metadata}" == packed ]]; then
    local packed_format
    packed_format="$(packed_format_for_qtype "${qtype}")"
    defs+=( -DL199_PACKED_METADATA=1 -DPPU_PACKED_SCALE=1
            "-DPPU_PACKED_FORMAT=${packed_format}" )
  else
    defs+=( -DL199_PACKED_METADATA=0 )
  fi
  nvcc "${base_flags[@]}" "${defs[@]}" "${source_file}" \
    -o "${out}/${label}" >"${out}/${label}.build.log" 2>&1 || {
      echo "[l199] FAIL: ${label} did not compile" >&2
      tail -n 80 "${out}/${label}.build.log" >&2
      return 1
    }
  "${out}/${label}" >"${out}/${label}.run.log" 2>&1 || {
    echo "[l199] FAIL: ${label} executable failed" >&2
    tail -n 80 "${out}/${label}.run.log" >&2
    return 1
  }
  grep -Fq "[l199] PASS q=${qtype} metadata=$([[ ${metadata} == packed ]] && echo packed || echo scale-zero) bchunk=${bchunk}" \
    "${out}/${label}.run.log" || {
      echo "[l199] FAIL: ${label} returned without its exact PASS summary" >&2
      tail -n 20 "${out}/${label}.run.log" >&2
      return 1
    }
  if grep -Fq 'FALSE_GREEN' "${out}/${label}.run.log"; then
    echo "[l199] FAIL: ${label} contains a false-green cell" >&2
    grep -F 'FALSE_GREEN' "${out}/${label}.run.log" >&2
    return 1
  fi
  printf '[l199:arm] %s %s\n' "${label}" \
    "$(grep '^\[l199\] PASS' "${out}/${label}.run.log")"
}

pids=()
labels=()
batch_failed=0
wait_batch() {
  local i
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[l199] FAIL: arm ${labels[$i]} failed" >&2
      batch_failed=1
    fi
  done
  pids=()
  labels=()
}

for bchunk in 0 1; do
  for metadata in scale packed; do
    for qtype in 10 11 12 13 14; do
      run_arm "${qtype}" "${metadata}" "${bchunk}" &
      pids+=("$!")
      labels+=("q${qtype}-${metadata}-bc${bchunk}")
      if [[ "${#pids[@]}" -ge "${jobs}" ]]; then
        wait_batch
      fi
    done
  done
done
if [[ "${#pids[@]}" -gt 0 ]]; then wait_batch; fi
if [[ "${batch_failed}" -ne 0 ]]; then exit 1; fi

mapfile -t summaries < <(grep -h '^\[l199\] PASS' "${out}"/*.run.log)
if [[ "${#summaries[@]}" -ne 20 ]]; then
  echo "[l199] FAIL: expected 20 format/metadata/BChunk summaries, got ${#summaries[@]}" >&2
  exit 1
fi
cell_rows="$(grep -hEc '^\[l199:cell\]' "${out}"/*.run.log | awk '{s += $1} END {print s + 0}')"
cells="$(printf '%s\n' "${summaries[@]}" | sed -nE 's/.* cells=([0-9]+).*/\1/p' | awk '{s += $1} END {print s + 0}')"
admitted="$(printf '%s\n' "${summaries[@]}" | sed -nE 's/.* admitted=([0-9]+).*/\1/p' | awk '{s += $1} END {print s + 0}')"
rejected="$(printf '%s\n' "${summaries[@]}" | sed -nE 's/.* rejected=([0-9]+).*/\1/p' | awk '{s += $1} END {print s + 0}')"
if [[ "${cells}" -ne 3520 ]] || [[ "${cell_rows}" -ne 3520 ]] || \
   [[ $((admitted + rejected)) -ne 3520 ]]; then
  echo "[l199] FAIL: denominator drift cells=${cells} rows=${cell_rows} admitted=${admitted} rejected=${rejected}" >&2
  exit 1
fi
if grep -hE ' REJECT[[:space:]]*$| REJECT UNKNOWN$' "${out}"/*.run.log >/dev/null; then
  echo '[l199] FAIL: at least one rejected denominator cell lacks a named reason' >&2
  exit 1
fi
if [[ "$(printf '%s\n' "${summaries[@]}" | grep -Fc 'A=fp16 Acc=fp32 D=fp16')" -ne 20 ]]; then
  echo '[l199] FAIL: audited activation/accumulator/destination types missing from a summary' >&2
  exit 1
fi

# Independent-fold evidence comes from the exact production MainloopPolicy in
# the admitted rows, not from a parallel fold table in this script.
for spec in \
  'q11-packed-bc0:folds=2/4' \
  'q13-packed-bc0:folds=1/4' \
  'q14-packed-bc0:folds=2/4'; do
  label="${spec%%:*}"
  fold="${spec##*:}"
  grep -Fq "${fold}" "${out}/${label}.run.log" || {
    echo "[l199] FAIL: ${label} did not admit independent ${fold}" >&2
    exit 1
  }
done

# BChunk is a requested/effective pair.  Q2's currently admitted one-plane
# collectives remain 1/0, Q4 is a named unsupported tactic, and every two-plane
# family must make the request effective for every admitted row.
for metadata in scale packed; do
  grep -Fq ' bchunk=1 effective=0 ' "${out}/q10-${metadata}-bc1.run.log" || {
    echo "[l199] FAIL: q10 ${metadata} did not report requested/effective BChunk=1/0" >&2
    exit 1
  }
  if grep ' ADMITTED$' "${out}/q10-${metadata}-bc1.run.log" | grep -Fv 'bchunk=1/0' >/dev/null; then
    echo "[l199] FAIL: q10 ${metadata} falsely advertised an effective BChunk cell" >&2
    exit 1
  fi
  grep -Fq ' bchunk=1 effective=REJECT ' "${out}/q12-${metadata}-bc1.run.log" || {
    echo "[l199] FAIL: q12 ${metadata} did not report requested/effective BChunk=1/REJECT" >&2
    exit 1
  }
  grep -Fq 'single-plane PPU_B_CHUNK requires a 1- or 2-bit format' \
    "${out}/q12-${metadata}-bc1.run.log" || {
    echo "[l199] FAIL: q12 ${metadata} lacks the named BChunk rejection" >&2
    exit 1
  }
  for qtype in 11 13 14; do
    grep -Fq ' bchunk=1 effective=1 ' "${out}/q${qtype}-${metadata}-bc1.run.log" || {
      echo "[l199] FAIL: q${qtype} ${metadata} did not report effective BChunk=1" >&2
      exit 1
    }
    if grep ' ADMITTED$' "${out}/q${qtype}-${metadata}-bc1.run.log" | \
        grep -Fv 'bchunk=1/1' >/dev/null; then
      echo "[l199] FAIL: q${qtype} ${metadata} admitted an ineffective BChunk cell" >&2
      exit 1
    fi
  done
done
if [[ "$(printf '%s\n' "${summaries[@]}" | grep -Fc ' bchunk=0 effective=0 ')" -ne 10 ]]; then
  echo '[l199] FAIL: a BC0 arm reported a nonzero or mixed effective state' >&2
  exit 1
fi

# Compile-time REDs: one variable changes at each owned type seam.
compile_red() {
  local label="$1" expected="$2"
  shift 2
  if nvcc "${base_flags[@]}" "$@" "${source_file}" -o "${out}/${label}" \
      >"${out}/${label}.log" 2>&1; then
    echo "[l199] FAIL: planted ${label} compiled" >&2
    exit 1
  fi
  grep -Fq "${expected}" "${out}/${label}.log" || {
    echo "[l199] FAIL: ${label} failed for an unrelated reason" >&2
    tail -n 80 "${out}/${label}.log" >&2
    exit 1
  }
  echo "[l199:red] ${label} EXPECTED_RED"
}

compile_red omit-b2-type 'PlaneB2 must exactly match the shipping high-plane type' \
  -DL199_QTYPE=11 -DL199_PACKED_METADATA=0 -DPPU_B_CHUNK=0 \
  -DL199_PLANT_OMIT_B2_TYPE=1
compile_red packed-mode-drift \
  'packed-unit and fp16-plane metadata call sites must name the selected shipping collective' \
  -DL199_QTYPE=11 -DL199_PACKED_METADATA=1 -DPPU_B_CHUNK=0 \
  -DPPU_PACKED_SCALE=1 -DPPU_PACKED_FORMAT=3 -DL199_PLANT_METADATA_MODE=1
compile_red decode-default-ordinary \
  'L199_DECODE_DEFAULT_PACKED_A_SHIPPING_SEAM' \
  -DL199_QTYPE=10 -DL199_PACKED_METADATA=1 -DPPU_B_CHUNK=0 \
  -DPPU_PACKED_SCALE=1 -DPPU_PACKED_FORMAT=2 \
  -DL199_PLANT_DECODE_DEFAULT_ORDINARY=1
for qtype in 10 12; do
  compile_red "device-q${qtype}-decode-default-ordinary" \
    'L199_DEVICE_DECODE_DEFAULT_PACKED_A_SEAM' \
    "-DL199_QTYPE=${qtype}" -DL199_PACKED_METADATA=1 -DPPU_B_CHUNK=0 \
    -DPPU_PACKED_SCALE=1 "-DPPU_PACKED_FORMAT=$(packed_format_for_qtype "${qtype}")" \
    -DL199_FORCE_DEVICE_BODY=1 -DL199_EXPECT_EFFECTIVE_BCHUNK=0 \
    -DL199_PLANT_DEVICE_DECODE_DEFAULT_ORDINARY=1
done

# Device-body witness.  The temporary overlay plants one dependent assertion
# at the entrance to the real fixed-SplitK operator.  All five packed formats
# and the three two-plane scale-zero formats must reach it exactly once;
# normal host arms above contain no marker and therefore form the route-severed controls.
overlay="${out}/overlay"
kernel_rel='actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_splitk_parallel.hpp'
mkdir -p "${overlay}/$(dirname "${kernel_rel}")"
cp "${repo}/quactlize/include/dense_splitk_multiformat_ppu.cuh" "${overlay}/"
cp "${repo}/quactlize/include/dense_splitk_parallel_ppu.cuh" "${overlay}/"
cp "${repo}/quactlize/include/${kernel_rel}" "${overlay}/${kernel_rel}"
python3 - "${overlay}/${kernel_rel}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
needle = """  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {"""
if text.count(needle) != 1:
    raise SystemExit("L199 production Split-K device seam is not unique")
text = text.replace(
    needle,
    needle + """
    static_assert(sizeof(CollectiveMainloop) == 0,
                  "L199_MULTIFORMAT_DEVICE_BODY_INSTANTIATED");""",
    1,
)
path.write_text(text, encoding="utf-8")
PY

device_flags=(
  -I "${overlay}"
  "${base_flags[@]}"
  -DL199_FORCE_DEVICE_BODY=1
  -Xcudafe --error_limit=100000
  -cuda -x cu
)
marker='error: static assertion failed with "L199_MULTIFORMAT_DEVICE_BODY_INSTANTIATED"'
run_device_probe() {
  local qtype="$1" metadata="$2" requested="$3" effective="$4"
  local label="device-q${qtype}-${metadata}-bc${requested}"
  local defs=("-DL199_QTYPE=${qtype}" "-DPPU_B_CHUNK=${requested}"
              "-DL199_EXPECT_EFFECTIVE_BCHUNK=${effective}")
  if [[ "${metadata}" == packed ]]; then
    defs+=( -DL199_PACKED_METADATA=1 -DPPU_PACKED_SCALE=1
            "-DPPU_PACKED_FORMAT=$(packed_format_for_qtype "${qtype}")" )
  else
    defs+=( -DL199_PACKED_METADATA=0 )
  fi
  set +e
  nvcc "${device_flags[@]}" "${defs[@]}" "${source_file}" \
    -o "${out}/${label}.cu.cpp" >"${out}/${label}.log" 2>&1
  local rc=$?
  set -e
  if [[ "${rc}" -eq 0 ]] || \
     [[ "$(grep -Fc "${marker}" "${out}/${label}.log" || true)" -ne 1 ]]; then
    echo "[l199] FAIL: ${label} did not reach the exact production device body once" >&2
    tail -n 100 "${out}/${label}.log" >&2
    exit 1
  fi
  local unexpected="${out}/${label}.unexpected"
  grep -E ': (error|fatal error|catastrophic error):' "${out}/${label}.log" \
    | grep -Fv "${marker}" >"${unexpected}" || true
  if [[ -s "${unexpected}" ]]; then
    echo "[l199] FAIL: ${label} carried an unrelated device diagnostic" >&2
    sed -n '1,60p' "${unexpected}" >&2
    exit 1
  fi
  for token in \
    'GemmUniversalMixedInputSplitKParallel<ProblemShape_, CollectiveMainloop_, CollectivePartialEpilogue_' \
    'CollectivePartialEpilogue_=dense_splitk_parallel_ppu::AdapterVisiblePartialEpilogue' \
    'AcConvert<float, 1, float'; do
    grep -Fq "${token}" "${out}/${label}.log" || {
      echo "[l199] FAIL: ${label} lost production instantiation token ${token}" >&2
      tail -n 120 "${out}/${label}.log" >&2
      exit 1
    }
  done
  echo "[l199:device] ${label} exact production body reached requested/effective=${requested}/${effective}"
}

for qtype in 11 13 14; do run_device_probe "${qtype}" scale 0 0; done
for qtype in 10 11 12 13 14; do run_device_probe "${qtype}" packed 0 0; done
for qtype in 11 13 14; do run_device_probe "${qtype}" packed 1 1; done

# Source-authority audit for the semantics that cannot be inferred from the
# host exact fixture.  This checks the shipping constructor itself: packed
# units occupy ptr_S, ptr_Z is null, A is uint16/fp16, and both ABIs instantiate
# the same DenseKernelTypes/epilogue authority.
python3 - "${repo}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
backend = (root / "quactlize/csrc/device/ppu_dense_backend.cu").read_text()
types = (root / "quactlize/include/fpA_intB_ppu.cuh").read_text()
required_backend = [
    "int dense_fully_quantized_device(uint16_t const* act",
    "config, act, low, high, units, nullptr, out",
    "QM::FinegrainedScaleZero",
    "Low, High, PackedScale",
]
required_types = [
    "using ElementC = cutlass::half_t;",
    "using ElementD = cutlass::half_t;",
    "using ElementAccumulator = float;",
    "using ElementA = typename MainloopPolicy::ElementA;",
    "using GemmKernel = cutlass::gemm::kernel::GemmUniversal<",
]
missing = [x for x in required_backend if x not in backend]
missing += [x for x in required_types if x not in types]
if missing:
    raise SystemExit("L199 source-authority drift: " + repr(missing))
print("[l199:source] PASS: fully-quantized=packed-S/Z, A=fp16, ptr_Z=null; one DenseKernelTypes authority")
PY

echo "[l199] PASS: denominator=3520 admitted=${admitted} rejected=${rejected}; all Q2/Q3/Q4/Q5/Q6 scale+packed real types/host admission; BChunk requested/effective Q2=1/0 Q4=1/REJECT Q3/Q5/Q6=1/1; device bodies=11 (packed BC0=5, two-plane scale BC0=3, two-plane packed BC1=3); REDs closed; artifacts=${out}"
