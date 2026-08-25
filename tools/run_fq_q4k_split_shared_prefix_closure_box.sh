#!/usr/bin/env bash
# Extend the 4288d8f prefix bundle with the true legacy-output negative.
# Existing prefix binaries are hash-verified and reused; only one binary builds.
set -uo pipefail

fail() {
  echo "[fq-shared-prefix-closure] FAIL: $*" >&2
  return 2
}

main() {
  local root workspace_root source_out out source_sha head_sha jobs repeats
  local sdk_root hgobjdump generated build_dir build_log legacy_binary
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace_root="$(realpath -e /workspace)" || return 2
  [ -n "${SOURCE_OUT:-}" ] || {
    fail 'SOURCE_OUT must name the completed 4288d8f prefix artifact'
    return $?
  }
  source_out="$(realpath -e -- "$SOURCE_OUT")" || return 2
  case "$source_out" in
    "$workspace_root"/*) ;;
    *) fail "SOURCE_OUT must resolve below /workspace: $source_out"; return $? ;;
  esac
  [ -d "$source_out/results" ] && [ -d "$source_out/generated/closure" ] || {
    fail "SOURCE_OUT is not a prefix bundle: $source_out"
    return $?
  }
  source_sha=4288d8f651c5c8556e399bcf43392e621d692f7b
  head_sha="$(git -C "$root" rev-parse HEAD)" || return 2
  git -C "$root" merge-base --is-ancestor "$source_sha" "$head_sha" || {
    fail "$source_sha is not an ancestor of $head_sha"
    return $?
  }
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    fail 'ambient defines change the exact legacy arm'
    return $?
  fi
  jobs="${JOBS:-16}"
  repeats="${PROBE_REPEATS:-2048}"
  case "$jobs" in *[!0-9]*|0) fail 'JOBS must be positive'; return $? ;; esac
  case "$repeats" in
    *[!0-9]*|0) fail 'PROBE_REPEATS must be positive'; return $? ;;
  esac

  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-shared-prefix-closure-${head_sha:0:8}-$(date -u +%Y%m%dT%H%M%SZ)-$$}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) fail "OUT must resolve below /workspace: $out"; return $? ;;
  esac
  [ ! -e "$out" ] || { fail "refusing to overwrite $out"; return $?; }
  mkdir -p "$out/results" || return 2

  cmp "$source_out/results/source.before.sha256" \
      "$source_out/results/source.after.sha256" >/dev/null || {
    fail 'source bundle changed while its original run executed'
    return $?
  }
  sha256sum -c "$source_out/results/authority.sha256" \
    >"$out/results/source-authority-check.log" 2>&1 || {
      fail 'SOURCE_OUT result authority no longer verifies'
      return $?
    }
  local expected actual path
  while read -r expected path; do
    [ -n "$expected" ] && [ -n "$path" ] || {
      fail 'malformed source.before.sha256'
      return $?
    }
    actual="$(git -C "$root" cat-file blob "$source_sha:$path" \
      | sha256sum | awk '{print $1}')" || return 2
    [ "$actual" = "$expected" ] || {
      fail "SOURCE_OUT is not bound to $source_sha: $path"
      return $?
    }
  done <"$source_out/results/source.before.sha256"

  # No compiled source changed after the parent device bundle.  The closure
  # commit may change only adjudication, runners, their source gate and record.
  local changed
  while IFS= read -r changed; do
    case "$changed" in
      ci/check_fq_split_shared_prefix_root.py|\
      ci/check_fq_split_shared_prefix_closure.py|\
      dev/fold_derivation/Q4K_FQ_SPLITK_PACKED_OWNER_DEBUG.md|\
      tools/check_fq_split_shared_prefix_root.py|\
      tools/report_fq_split_shared_prefix_codegen.py|\
      tools/run_fq_q4k_split_shared_prefix_root_box.sh|\
      tools/run_fq_q4k_split_shared_prefix_closure_box.sh) ;;
      *) fail "compiled-source identity cannot be inferred; changed=$changed"; return $? ;;
    esac
  done < <(git -C "$root" diff --name-only "$source_sha..$head_sha")
  local -a compiled_paths=(
    quactlize/include/dense_splitk_parallel_ppu.cuh
    quactlize/include/actlize_extensions/cutlass/gemm/kernel/detail/ppu_splitk_direct_accumulator_store.hpp
    quactlize/include/actlize_extensions/cutlass/gemm/kernel/detail/ppu_splitk_shared_epilogue_sync_probe.hpp
    quactlize/include/actlize_extensions/cutlass/gemm/kernel/detail/ppu_splitk_shared_prefix_probe.hpp
    quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_splitk_parallel.hpp
    third_party/actlize/include/cutlass/epilogue/collective/ppu_epilogue_vectorized_parallel.hpp
  )
  for path in "${compiled_paths[@]}"; do
    git -C "$root" diff --quiet "$source_sha" -- "$path" || {
      fail "compiled source differs from source bundle: $path"
      return $?
    }
  done

  local -a reused_arms=(
    production-direct accumulator-opaque clone-opaque cta-only
    flat-constant-disjoint flat-accumulator-disjoint
    r2s-vector-disjoint r2s-scalar-disjoint r2s-snapshot-disjoint
    r2s-s2r-vector-disjoint r2s-s2r-scalar-disjoint full-discard
  )
  local arm binary binary_sha expected_sha
  for arm in "${reused_arms[@]}"; do
    binary="$source_out/build-$arm/ppu_targets/test_fully_quantized_internal_sweep"
    [ -x "$binary" ] && [ ! -L "$binary" ] || {
      fail "missing reusable binary: $binary"
      return $?
    }
    expected_sha="$(sed -n \
      "s/^FQ_SHARED_PREFIX_BINARY arm=$arm .* sha256=\\([0-9a-f]\\{64\\}\\) .*/\\1/p" \
      "$source_out/results/summary.log" | head -n1)"
    [ -n "$expected_sha" ] || {
      fail "missing binary authority for $arm"
      return $?
    }
    binary_sha="$(sha256sum "$binary" | awk '{print $1}')" || return 2
    [ "$binary_sha" = "$expected_sha" ] || {
      fail "$arm binary hash differs"
      return $?
    }
    for path in "$source_out/results/$arm-direct.log" \
                "$source_out/results/$arm-probe.log"; do
      [ -f "$path" ] && [ ! -L "$path" ] || {
        fail "missing reusable result: $path"
        return $?
      }
    done
  done

  python3 -B "$root/tools/check_fq_split_shared_prefix_root.py" --self-test || return 2
  python3 -B "$root/tools/report_fq_split_shared_prefix_codegen.py" --self-test || return 2
  python3 -B "$root/ci/check_fq_split_shared_prefix_root.py" || return 2
  python3 -B "$root/ci/check_fq_split_shared_prefix_closure.py" || return 2

  generated="$source_out/generated/closure"
  [ -f "$generated/manifest.json" ] || {
    fail 'reused exact generated closure lacks manifest'
    return $?
  }
  build_dir="$out/build-legacy-shared-output"
  build_log="$out/results/legacy-shared-output-build.log"
  mkdir -p "$build_dir" || return 2
  (cd "$root" && PPU_BUILD_DIR="$build_dir" PPU_ARCHS=ppu0010 JOBS="$jobs" \
    PPU_DEFS='PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE=1' \
    TARGET=test_fully_quantized_internal_sweep \
    FQ_SWEEP_GENERATED_DIR="$generated" FQ_SWEEP_QTYPE=12 \
    FQ_SWEEP_ARTIFACT_TK=64 FQ_SWEEP_BCHUNK=0 \
    FQ_SWEEP_PACKED_FORMAT=0 ./build.sh) >"$build_log" 2>&1
  local build_rc=$?
  if [ "$build_rc" -ne 0 ]; then
    tail -180 "$build_log" >&2
    fail "legacy-shared-output build rc=$build_rc artifacts=$out"
    return "$build_rc"
  fi
  grep -Fq -- '-DPPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE=1' "$build_log" || {
    fail 'legacy define did not reach the compiler command'
    return $?
  }
  legacy_binary="$build_dir/ppu_targets/test_fully_quantized_internal_sweep"
  [ -x "$legacy_binary" ] && [ ! -L "$legacy_binary" ] || {
    fail "legacy binary missing: $legacy_binary"
    return $?
  }

  local direct_log="$out/results/legacy-shared-output-direct.log"
  local probe_log="$out/results/legacy-shared-output-probe.log"
  "$legacy_binary" --shape=1x1024x5120 --iterations=1 \
    --correctness-repeats="$repeats" --tm8-max-m=8 --bc-mode=skip \
    >"$direct_log" 2>&1
  local direct_rc=$?
  "$legacy_binary" --shape=1x1024x5120 --iterations=1 \
    --correctness-repeats="$repeats" --tm8-max-m=8 --bc-mode=skip \
    --split-workspace-probe >"$probe_log" 2>&1
  local probe_rc=$?
  if { [ "$direct_rc" -ne 0 ] && [ "$direct_rc" -ne 1 ]; } || \
      [ "$probe_rc" -ne 0 ]; then
    fail "legacy execution rc direct=$direct_rc probe=$probe_rc"
    return $?
  fi

  sdk_root="${PPU_SDK:-${PPU_HOME:-${PPU_SDK_SITE_DEFAULT:-}}}"
  hgobjdump="${HGOBJDUMP:-${sdk_root:+$sdk_root/bin/hgobjdump}}"
  if [ -z "$hgobjdump" ] || [ ! -x "$hgobjdump" ]; then
    hgobjdump="$(command -v hgobjdump 2>/dev/null || true)"
  fi
  [ -n "$hgobjdump" ] && [ -x "$hgobjdump" ] || {
    fail 'SDK hgobjdump is required'
    return $?
  }
  hgobjdump="$(readlink -f "$hgobjdump")" || return 2
  "$hgobjdump" --version >"$out/results/hgobjdump.version" 2>&1 || return 2
  sha256sum "$hgobjdump" >"$out/results/hgobjdump.sha256" || return 2

  local -a codegen_arms=(
    production-direct clone-opaque cta-only flat-constant-disjoint
    flat-accumulator-disjoint r2s-vector-disjoint r2s-scalar-disjoint
    r2s-snapshot-disjoint full-discard legacy-shared-output
  )
  local list_elf candidate pretty count resource line demangled
  : >"$out/results/codegen-summary.log"
  for arm in "${codegen_arms[@]}"; do
    if [ "$arm" = legacy-shared-output ]; then
      binary="$legacy_binary"
    else
      binary="$source_out/build-$arm/ppu_targets/test_fully_quantized_internal_sweep"
    fi
    list_elf="$out/results/$arm-list-elf.txt"
    "$hgobjdump" -lelf "$binary" >"$list_elf" \
      2>"$out/results/$arm-list-elf.err" || return 2
    count=0
    while IFS= read -r candidate; do
      pretty="$(c++filt "$candidate")"
      [[ "$pretty" == *GemmUniversalMixedInputSplitKParallel* ]] || continue
      count=$((count + 1))
      demangled="$out/results/$arm-kernel-$count.demangled"
      resource="$out/results/$arm-kernel-$count.resource"
      line="$out/results/$arm-kernel-$count.line"
      printf '%s\n' "$pretty" >"$demangled"
      "$hgobjdump" "-res-usage=$candidate" "$binary" >"$resource" \
        2>"$out/results/$arm-kernel-$count.resource.err" || return 2
      "$hgobjdump" -line "-func=$candidate" "$binary" >"$line" \
        2>"$out/results/$arm-kernel-$count.line.err" || return 2
      python3 -B "$root/tools/report_fq_split_shared_prefix_codegen.py" \
        --arm "$arm" --kernel "$count" --line "$line" \
        --resource "$resource" --demangled "$demangled" --binary "$binary" \
        | tee -a "$out/results/codegen-summary.log" || return 2
    done < <(sed -n \
      's/^.*Func [0-9][0-9]*:[[:space:]]*\([^[:space:]]*\).*$/\1/p' \
      "$list_elf")
    [ "$count" -eq 2 ] || {
      fail "$arm exact kernel denominator=$count, expected 2"
      return $?
    }
  done
  local provider_count provider
  for arm in "${codegen_arms[@]}"; do
    for provider in standard-aiu packed-row; do
      provider_count="$(grep -Ec \
        "^FQ_SHARED_PREFIX_CODEGEN arm=$arm kernel=[12] provider=$provider " \
        "$out/results/codegen-summary.log" || true)"
      [ "$provider_count" -eq 1 ] || {
        fail "$arm provider=$provider codegen denominator=$provider_count, expected 1"
        return $?
      }
    done
  done

  local -a verdict_args=()
  for arm in "${reused_arms[@]}"; do
    verdict_args+=(
      "--$arm-direct" "$source_out/results/$arm-direct.log"
      "--$arm-probe" "$source_out/results/$arm-probe.log")
  done
  verdict_args+=(
    --legacy-shared-output-direct "$direct_log"
    --legacy-shared-output-probe "$probe_log")
  python3 -B "$root/tools/check_fq_split_shared_prefix_root.py" \
    "${verdict_args[@]}" | tee "$out/results/verdict.log"
  local verdict_rc=${PIPESTATUS[0]}

  {
    echo 'schema=quactlize.fq-shared-prefix-closure.v1'
    echo "source_bundle=$source_out"
    echo "source_sha=$source_sha"
    echo "closure_sha=$head_sha"
    echo "legacy_binary_sha256=$(sha256sum "$legacy_binary" | awk '{print $1}')"
    echo "legacy_direct_rc=$direct_rc"
    echo "legacy_probe_rc=$probe_rc"
    echo "probe_repeats=$repeats"
  } >"$out/results/manifest.txt"
  find "$out/results" -maxdepth 1 -type f \
    ! -name authority.sha256 -print0 \
    | sort -z | xargs -0 sha256sum \
    >"$out/results/authority.sha256" || return 2
  if [ "$verdict_rc" -ne 0 ]; then
    echo "[fq-shared-prefix-closure] UNADJUDICATED artifacts=$out" >&2
    return "$verdict_rc"
  fi
  echo "[fq-shared-prefix-closure] ROOT_CAUSE_COMPLETE artifacts=$out"
}

main "$@"
