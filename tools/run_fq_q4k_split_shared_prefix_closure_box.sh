#!/usr/bin/env bash
# Extend the 4288d8f prefix bundle with the true legacy-output negative.
# Existing prefix binaries are hash-verified and reused; only one binary builds.
set -uo pipefail

fail() {
  echo "[fq-shared-prefix-closure] FAIL: $*" >&2
  return 2
}

finalize_closure() {
  local root="$1" source_out="$2" out="$3" source_sha="$4"
  local head_sha="$5" legacy_binary="$6" repeats="$7"
  local legacy_direct_rc="$8" legacy_probe_rc="$9" execution="${10}"
  local codegen_log="$out/results/codegen-summary.log"
  local direct_log="$out/results/legacy-shared-output-direct.log"
  local probe_log="$out/results/legacy-shared-output-probe.log"
  [ -x "$legacy_binary" ] && [ ! -L "$legacy_binary" ] || {
    fail "legacy binary missing during finalize: $legacy_binary"
    return $?
  }
  for path in "$codegen_log" "$direct_log" "$probe_log"; do
    [ -f "$path" ] && [ ! -L "$path" ] || {
      fail "closure evidence missing during finalize: $path"
      return $?
    }
  done
  grep -Fq "correctness_repeats=$repeats" "$direct_log" || {
    fail "legacy direct repeat authority differs from $repeats"
    return $?
  }
  grep -Fq "correctness_repeats=$repeats" "$probe_log" || {
    fail "legacy probe repeat authority differs from $repeats"
    return $?
  }
  local legacy_codegen_sha legacy_actual_sha
  legacy_codegen_sha="$(sed -n \
    's/^FQ_SHARED_PREFIX_CODEGEN arm=legacy-shared-output .* binary_sha256=\([0-9a-f]\{64\}\) .*/\1/p' \
    "$codegen_log" | sort -u)"
  [ "$(printf '%s\n' "$legacy_codegen_sha" | grep -Ec '^[0-9a-f]{64}$' || true)" -eq 1 ] || {
    fail 'legacy codegen binary authority is missing or non-unique'
    return $?
  }
  legacy_actual_sha="$(sha256sum "$legacy_binary" | awk '{print $1}')" || return 2
  [ "$legacy_actual_sha" = "$legacy_codegen_sha" ] || {
    fail 'legacy binary differs from captured exact-symbol codegen'
    return $?
  }

  # This SDK erases the source-level packed-A schedule wrapper from the
  # demangled device spelling.  Preserve the two exact ELF ordinals instead
  # of manufacturing an AP0/AP1 label from a missing token.
  local -a codegen_arms=(
    production-direct clone-opaque cta-only flat-constant-disjoint
    flat-accumulator-disjoint r2s-vector-disjoint r2s-scalar-disjoint
    r2s-snapshot-disjoint full-discard legacy-shared-output
  )
  local arm kernel count
  for arm in "${codegen_arms[@]}"; do
    count="$(grep -Ec \
      "^FQ_SHARED_PREFIX_CODEGEN arm=$arm kernel=[12] " \
      "$codegen_log" || true)"
    [ "$count" -eq 2 ] || {
      fail "$arm exact kernel denominator=$count, expected 2"
      return $?
    }
    for kernel in 1 2; do
      count="$(grep -Ec \
        "^FQ_SHARED_PREFIX_CODEGEN arm=$arm kernel=$kernel " \
        "$codegen_log" || true)"
      [ "$count" -eq 1 ] || {
        fail "$arm kernel=$kernel codegen denominator=$count, expected 1"
        return $?
      }
    done
  done

  local -a reused_arms=(
    production-direct accumulator-opaque clone-opaque cta-only
    flat-constant-disjoint flat-accumulator-disjoint
    r2s-vector-disjoint r2s-scalar-disjoint r2s-snapshot-disjoint
    r2s-s2r-vector-disjoint r2s-s2r-scalar-disjoint full-discard
  )
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
    echo "execution=$execution"
    echo 'codegen_provider_binding=EXACT-ELF-ORDINAL-ONLY'
    echo "legacy_binary_sha256=$(sha256sum "$legacy_binary" | awk '{print $1}')"
    echo "legacy_direct_rc=$legacy_direct_rc"
    echo "legacy_probe_rc=$legacy_probe_rc"
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

main() {
  local root workspace_root source_out out source_sha head_sha jobs repeats resume
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
  resume="${RESUME:-0}"
  case "$jobs" in *[!0-9]*|0) fail 'JOBS must be positive'; return $? ;; esac
  case "$repeats" in
    *[!0-9]*|0) fail 'PROBE_REPEATS must be positive'; return $? ;;
  esac
  case "$resume" in
    0|1) ;;
    *) fail 'RESUME must be 0 or 1'; return $? ;;
  esac

  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-shared-prefix-closure-${head_sha:0:8}-$(date -u +%Y%m%dT%H%M%SZ)-$$}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) fail "OUT must resolve below /workspace: $out"; return $? ;;
  esac
  if [ "$resume" -eq 1 ]; then
    [ -n "${OUT:-}" ] && [ -d "$out/results" ] || {
      fail 'RESUME=1 requires OUT to name the preserved closure artifact'
      return $?
    }
  else
    [ ! -e "$out" ] || { fail "refusing to overwrite $out"; return $?; }
    mkdir -p "$out/results" || return 2
  fi

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

  if [ "$resume" -eq 1 ]; then
    legacy_binary="$out/build-legacy-shared-output/ppu_targets/test_fully_quantized_internal_sweep"
    finalize_closure "$root" "$source_out" "$out" "$source_sha" \
      "$head_sha" "$legacy_binary" "$repeats" RECOVERED RECOVERED \
      ANALYSIS-ONLY-RESUME
    return $?
  fi

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
  finalize_closure "$root" "$source_out" "$out" "$source_sha" \
    "$head_sha" "$legacy_binary" "$repeats" "$direct_rc" "$probe_rc" \
    FRESH-ONE-BUILD
}

main "$@"
