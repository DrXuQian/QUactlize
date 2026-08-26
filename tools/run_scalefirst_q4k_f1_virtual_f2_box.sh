#!/usr/bin/env bash
# Exact three-arm closure for F1-native/fold2-style compute.
#
# Device invocation for the opt-in arm is intentionally recorded literally for
# the switch audit: PPU_DEFS=PPU_Q4_F1_VIRTUAL_F2=1.
set -uo pipefail

fail() { printf '[q4-f1-virtual-f2] FAIL: %s\n' "$*" >&2; return 2; }

main() {
  [[ $# -eq 0 ]] || { fail 'no positional arguments are accepted'; return $?; }
  local root sha short stamp out jobs corr_shape perf_shape perf_iterations corr_repeats perf_repeats
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-q4-f1-virtual-f2-${short}-${stamp}-$$}")" || return 2
  case "$out" in /workspace/*) ;; *) fail "OUT must be a strict /workspace child: $out"; return $? ;; esac
  [[ ! -e "$out" ]] || { fail "refusing existing OUT=$out"; return $?; }
  [[ -z "${PPU_DEFS:-}" && -z "${PPU_EXTRA_DEFS:-}" ]] || {
    fail 'ambient PPU_DEFS/PPU_EXTRA_DEFS would invalidate the arm identity'; return $?; }
  jobs="${JOBS:-16}"
  corr_shape="${CORRECTNESS_SHAPE:-64x1024x5120}"
  perf_shape="${PERF_SHAPE:-4096x5120x8192}"
  perf_iterations="${PERF_ITERATIONS:-21}"
  corr_repeats="${CORRECTNESS_REPEATS:-32}"
  perf_repeats="${PERF_CORRECTNESS_REPEATS:-8}"
  case "$jobs:$perf_iterations:$corr_repeats:$perf_repeats" in
    *[!0-9:]*|0:*|*:0:*|*:*:0:*|*:*:*:0) fail 'numeric controls must be positive integers'; return $? ;;
  esac
  mkdir -p "$out"/{generated,build,raw,results,inputs} || return 2

  local -a authority=(
    benchmarks/scalefirst_internal_sweep_bench.hpp
    benchmarks/scalefirst_internal_sweep_unit.inc
    benchmarks/test_scalefirst_internal_sweep.cu
    dev/fold_derivation/l224_q4_f1_virtual_f2.cu
    dev/fold_derivation/l225_q4_f1_virtual_f2_type.cu
    dev/fold_derivation/l226_q4_f1_virtual_f2_body.cu
    dev/fold_derivation/nvidia_nvcc_or_skip.sh
    dev/fold_derivation/q4_f1_virtual_f2.expected.txt
    dev/fold_derivation/run_l224_q4_f1_virtual_f2.sh
    dev/fold_derivation/run_l225_q4_f1_virtual_f2_type.sh
    dev/fold_derivation/run_l226_q4_f1_virtual_f2_body.sh
    ci/check_q4_f1_virtual_f2_committed_evidence.py
    quactlize/include/actlize_extensions/cutlass/gemm/quactlize_dispatch_policy.hpp
    quactlize/include/actlize_extensions/cutlass/gemm/collective/builders/quactlize_mma_builder.inl
    quactlize/include/fpA_intB_ppu.cuh
    quactlize/include/ppu_mixed_policy.hpp
    tools/analyze_scalefirst_q4k_f1_virtual_f2.py
    tools/gen_scalefirst_internal_units.py
    tools/run_scalefirst_q4k_f1_virtual_f2_box.sh
  )
  local dirty
  dirty="$(git -C "$root" status --porcelain -- "${authority[@]}")" || return 2
  [[ -z "$dirty" ]] || { fail "source authority is dirty:\n$dirty"; return $?; }

  # L224-L226 are NVIDIA-nvcc/stub host oracles.  The PPU box executable named
  # nvcc delegates device preprocessing to ppu_clang++, which enables the PPU
  # fp8 bridge without carrying targets/<triple>/include and dies on
  # hggc_fp8.h.  Never paper over this with a fake stub: consume exact evidence
  # from the result SHA, then build all three shipping arms fresh through hgcc.
  local proof_evidence="$out/inputs/q4_f1_virtual_f2.expected.txt"
  git -C "$root" show "$sha:dev/fold_derivation/q4_f1_virtual_f2.expected.txt" \
    >"$proof_evidence" || { fail 'result SHA lacks virtual-fold host evidence'; return $?; }
  {
    printf '[q4-f1-virtual-f2] host-evidence=committed-local-oracle source-sha=%s fresh-box-execution=0\n' "$sha"
    python3 -B "$root/ci/check_q4_f1_virtual_f2_committed_evidence.py" \
      --committed-only --evidence "$proof_evidence"
  } 2>&1 | tee "$out/results/host-evidence.log"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || return 2
  python3 -B "$root/tools/analyze_scalefirst_q4k_f1_virtual_f2.py" --self-test \
    >"$out/results/analyzer-self-test.log" 2>&1 || {
      cat "$out/results/analyzer-self-test.log" >&2; fail 'analyzer self-test failed'; return $?; }

  local native_symbol f1_symbol
  native_symbol=sf_q12_a32_tm64_tn128_tk128_wm64_wn64_s3_bc0
  f1_symbol=sf_q12_a64_tm64_tn128_tk128_wm64_wn64_s3_bc0

  build_arm native-f2 32 "$native_symbol" '' || return $?
  build_arm f1 64 "$f1_symbol" '' || return $?
  build_arm virtual-f2 64 "$f1_symbol" 'PPU_Q4_F1_VIRTUAL_F2=1' || return $?

  local arm binary log rc
  for arm in native-f2 f1 virtual-f2; do
    binary="$(<"$out/results/$arm.binary")"
    log="$out/raw/$arm-correctness.log"
    "$binary" --shape="$corr_shape" --iterations=1 \
      --correctness-repeats="$corr_repeats" --algorithm=nonpersistent \
      --fixture=exact --fixture-binding >"$log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      tail -n 120 "$log" >&2; fail "$arm correctness rc=$rc"; return $?;
    fi
    log="$out/raw/$arm-perf.log"
    "$binary" --shape="$perf_shape" --iterations="$perf_iterations" \
      --correctness-repeats="$perf_repeats" --algorithm=full-output \
      --fixture=exact --fixture-binding >"$log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      tail -n 120 "$log" >&2; fail "$arm performance rc=$rc"; return $?;
    fi
  done

  python3 -B "$root/tools/analyze_scalefirst_q4k_f1_virtual_f2.py" \
    --arm native-f2 "$out/raw/native-f2-correctness.log" "$out/raw/native-f2-perf.log" \
    --arm f1 "$out/raw/f1-correctness.log" "$out/raw/f1-perf.log" \
    --arm virtual-f2 "$out/raw/virtual-f2-correctness.log" "$out/raw/virtual-f2-perf.log" \
    --output "$out/results/summary.json" | tee "$out/results/summary.txt" || return 2

  python3 -B - "$root" "$out" "$sha" "${authority[@]}" <<'PY' || return 2
import hashlib,json,os,pathlib,sys
root,out,commit=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3]
authority=sys.argv[4:]
files={rel:hashlib.sha256((root/rel).read_bytes()).hexdigest() for rel in authority}
for rel in ["results/summary.json","results/summary.txt",
            "raw/native-f2-correctness.log","raw/f1-correctness.log","raw/virtual-f2-correctness.log",
            "raw/native-f2-perf.log","raw/f1-perf.log","raw/virtual-f2-perf.log"]:
 files[rel]=hashlib.sha256((out/rel).read_bytes()).hexdigest()
doc={"schema":"quactlize.q4_f1_virtual_f2_bundle.v1","git_sha":commit,"files":files}
p=out/"bundle.json"; t=out/f".bundle.json.current.{os.getpid()}"
with t.open("w") as f:
 json.dump(doc,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(t,p)
PY
  printf '[q4-f1-virtual-f2] PASS sha=%s artifacts=%s\n' "$sha" "$out"
}

build_arm() {
  local arm="$1" artifact="$2" symbol="$3" defs="$4"
  local generated="$out/generated/$arm" build="$out/build/$arm" log="$out/build/$arm.log" binary rc
  python3 -B "$root/tools/gen_scalefirst_internal_units.py" \
    --qtype 12 --artifact-tk "$artifact" --bchunk 0 --per-unit 1 \
    --select-symbol "$symbol" --out-dir "$generated" || return 2
  (cd "$root" && PPU_BUILD_DIR="$build" PPU_ARCHS=ppu0010 JOBS="$jobs" \
    PPU_DEFS="$defs" TARGET=test_scalefirst_internal_sweep \
    SCALEFIRST_SWEEP_GENERATED_DIR="$generated" SCALEFIRST_SWEEP_QTYPE=12 \
    SCALEFIRST_SWEEP_ARTIFACT_TK="$artifact" SCALEFIRST_SWEEP_BCHUNK=0 ./build.sh) \
    >"$log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    tail -n 120 "$log" >&2; fail "$arm build rc=$rc"; return $?;
  fi
  binary="$build/ppu_targets/test_scalefirst_internal_sweep"
  [[ -x "$binary" ]] || { fail "$arm build returned no binary"; return $?; }
  if [[ "$arm" == virtual-f2 ]]; then
    grep -Fq "PPU_DEFS verified on test_scalefirst_internal_sweep's compile command: -DPPU_Q4_F1_VIRTUAL_F2=1" \
      "$log" || { fail 'virtual-f2 build did not prove its compile definition'; return $?; }
  elif grep -Fq -- '-DPPU_Q4_F1_VIRTUAL_F2' "$log"; then
    fail "$arm control unexpectedly contains the virtual-fold definition"; return $?
  fi
  printf '%s\n' "$binary" >"$out/results/$arm.binary"
  sha256sum "$binary" >"$out/results/$arm.binary.sha256"
  printf '[q4-f1-virtual-f2] build arm=%s A=%s symbol=%s sha256=%s\n' \
    "$arm" "$artifact" "$symbol" "$(cut -d' ' -f1 "$out/results/$arm.binary.sha256")"
}

main "$@"
