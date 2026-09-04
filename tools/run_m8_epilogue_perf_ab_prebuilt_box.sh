#!/usr/bin/env bash
# Compile-free PPU execution gate for the exact TM8 epilogue pre/post A/B.
# The bundle is built from two frozen source/submodule pairs; this runner only
# verifies it, disassembles the exact S1 kernel, and alternates timing order.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
ANALYZER="$ROOT/tools/analyze_m8_epilogue_perf_ab.py"

fail() {
  printf '[m8-epilogue-perf-ab-box] FAIL phase=%s artifacts=%s\n' "$1" "${RUN_DIR:-UNCREATED}" >&2
  return 2
}

main() {
  local workspace bundle sdk_root hgobjdump runtime_path manifest_inspector_hash
  local arm binary list_elf symbol demangled line resource round slot rc
  local threshold=${PERF_REGRESSION_THRESHOLD:-0.03}
  local -a order runtime_dirs

  [ "$#" -eq 0 ] || { fail positional_arguments_forbidden; return 2; }
  workspace="$(realpath -e /workspace 2>/dev/null)" || {
    fail workspace_missing; return 2;
  }
  RUN_DIR="$(realpath -m -- "${OUT:-/workspace/quactlize-m8-epilogue-perf-ab-$(date -u +%Y%m%dT%H%M%SZ)-$$}")" || {
    fail invalid_output; return 2;
  }
  case "$RUN_DIR" in
    "$workspace"/*) ;;
    *) fail output_not_workspace_child; return 2 ;;
  esac
  [ ! -e "$RUN_DIR" ] || { fail output_already_exists; return 2; }
  case "${CUDA_VISIBLE_DEVICES:-}" in
    ''|*,*|*[!0-9]*) fail one_numeric_CUDA_VISIBLE_DEVICES_required; return 2 ;;
  esac
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    fail ambient_ppu_defs; return 2
  fi
  python3 -B - "$threshold" <<'PY' || {
import sys
value = float(sys.argv[1])
assert 0.0 < value < 0.2
PY
    fail invalid_regression_threshold
    return 2
  }

  bundle="$(realpath -e -- "${M8_EPILOGUE_PERF_AB_BUNDLE:-/nonexistent}" 2>/dev/null)" || {
    fail exact_prebuilt_bundle_required; return 2;
  }
  case "$bundle" in
    "$workspace"/*) ;;
    *) fail bundle_not_workspace_child; return 2 ;;
  esac
  [ -f "$bundle/manifest.json" ] && [ ! -L "$bundle/manifest.json" ] || {
    fail bundle_manifest_missing; return 2;
  }
  sdk_root="$(realpath -e -- "${PPU_SDK:-${PPU_HOME:-/nonexistent}}" 2>/dev/null)" || {
    fail exact_ppu_sdk_required; return 2;
  }
  hgobjdump="$sdk_root/bin/hgobjdump"
  [ -x "$hgobjdump" ] && [ ! -L "$hgobjdump" ] || {
    fail sdk_lacks_hgobjdump; return 2;
  }

  mkdir -p "$RUN_DIR"/{inputs,codegen,runs,results} || {
    fail create_output; return 2;
  }
  python3 -B "$ANALYZER" self-test || {
    fail analyzer_self_test; return 2;
  }
  python3 -B "$ANALYZER" verify-bundle --bundle "$bundle" || {
    fail bundle_verification; return 2;
  }
  manifest_inspector_hash="$(python3 -B - "$bundle/manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["build_sdk"]["inspector_sha256"])
PY
  )" || { fail read_inspector_authority; return 2; }
  [ "$(sha256sum "$hgobjdump" | awk '{print $1}')" = "$manifest_inspector_hash" ] || {
    fail inspector_identity_differs; return 2;
  }

  runtime_dirs=(
    "$sdk_root/lib"
    "$sdk_root/targets/x86_64-linux/lib"
    "$sdk_root/CUDA_SDK/lib64"
    "$sdk_root/CUDA_SDK/targets/x86_64-linux/lib")
  runtime_path=
  for runtime_dir in "${runtime_dirs[@]}"; do
    [ -d "$runtime_dir" ] || continue
    runtime_path="${runtime_path:+$runtime_path:}$runtime_dir"
  done
  [ -n "$runtime_path" ] || { fail sdk_runtime_paths_missing; return 2; }
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    runtime_path="$runtime_path:$LD_LIBRARY_PATH"
  fi

  cp -- "$bundle/manifest.json" "$RUN_DIR/inputs/bundle-manifest.json" || {
    fail copy_manifest; return 2;
  }
  {
    printf 'runner_source=%s\n' "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf UNKNOWN)"
    printf 'device_ordinal=%s\n' "$CUDA_VISIBLE_DEVICES"
    printf 'bundle=%s\n' "$bundle"
    printf 'sdk=%s\n' "$sdk_root"
    sha256sum "$bundle/manifest.json" "$bundle/bin/baseline" \
      "$bundle/bin/candidate" "$hgobjdump" "$ANALYZER" \
      "$ROOT/tools/run_m8_epilogue_perf_ab_prebuilt_box.sh"
  } >"$RUN_DIR/inputs/result-authority.txt" || {
    fail result_authority; return 2;
  }

  for arm in baseline candidate; do
    binary="$bundle/bin/$arm"
    [ -x "$binary" ] && [ ! -L "$binary" ] || {
      fail "${arm}_binary_missing"; return 2;
    }
    mkdir -p "$RUN_DIR/codegen/$arm" || return 2
    env LD_LIBRARY_PATH="$runtime_path" ldd "$binary" \
      >"$RUN_DIR/codegen/$arm/ldd.txt" 2>&1 || {
      cat "$RUN_DIR/codegen/$arm/ldd.txt" >&2
      fail "${arm}_runtime_linkage"; return 2;
    }
    if grep -F 'not found' "$RUN_DIR/codegen/$arm/ldd.txt" >/dev/null; then
      cat "$RUN_DIR/codegen/$arm/ldd.txt" >&2
      fail "${arm}_runtime_linkage"; return 2
    fi
    python3 -B "$ANALYZER" runtime-linkage --arm "$arm" \
      --ldd "$RUN_DIR/codegen/$arm/ldd.txt" \
      --output "$RUN_DIR/codegen/$arm-runtime.json" || {
      fail "${arm}_runtime_identity"; return 2;
    }
    list_elf="$RUN_DIR/codegen/$arm/list-elf.txt"
    "$hgobjdump" -lelf "$binary" >"$list_elf" \
      2>"$RUN_DIR/codegen/$arm/list-elf.err" || {
      fail "${arm}_list_elf"; return 2;
    }
    symbol="$RUN_DIR/codegen/$arm/kernel-symbol.txt"
    demangled="$RUN_DIR/codegen/$arm/kernel-symbol-demangled.txt"
    python3 -B "$ANALYZER" select-symbol --list-elf "$list_elf" \
      --symbol-output "$symbol" --demangled-output "$demangled" || {
      fail "${arm}_select_s1"; return 2;
    }
    line="$RUN_DIR/codegen/$arm/kernel-line.txt"
    resource="$RUN_DIR/codegen/$arm/resource-usage.txt"
    "$hgobjdump" -line "-func=$(cat "$symbol")" "$binary" >"$line" \
      2>"$RUN_DIR/codegen/$arm/kernel-line.err" || {
      fail "${arm}_line_disassembly"; return 2;
    }
    "$hgobjdump" "-res-usage=$(cat "$symbol")" "$binary" >"$resource" \
      2>"$RUN_DIR/codegen/$arm/resource-usage.err" || {
      fail "${arm}_resource_usage"; return 2;
    }
    python3 -B "$ANALYZER" codegen --arm "$arm" --line "$line" \
      --resource "$resource" --symbol "$symbol" --binary "$binary" \
      --output "$RUN_DIR/codegen/$arm.json" || {
      fail "${arm}_codegen_analysis"; return 2;
    }
  done

  printf 'round\tslot\tarm\n' >"$RUN_DIR/inputs/execution.tsv"
  for round in 1 2 3 4 5 6; do
    if ((round % 2)); then
      order=(baseline candidate)
    else
      order=(candidate baseline)
    fi
    slot=0
    for arm in "${order[@]}"; do
      slot=$((slot + 1))
      binary="$bundle/bin/$arm"
      local log="$RUN_DIR/runs/round-$(printf '%02d' "$round")-slot-$slot-$arm.log"
      printf '%d\t%d\t%s\n' "$round" "$slot" "$arm" \
        >>"$RUN_DIR/inputs/execution.tsv"
      printf '[m8-epilogue-perf-ab-box] run round=%d slot=%d arm=%s\n' \
        "$round" "$slot" "$arm"
      rc=0
      env LD_LIBRARY_PATH="$runtime_path" "$binary" \
        --shape=8x3072x512 --iterations=31 --correctness-repeats=7 \
        --schedule-seed=0x6a09e667f3bcc909 --only-split=1 \
        --tm8-max-m=8 --symbols-file="$bundle/inputs/symbol.txt" \
        --bc-mode=skip >"$log" 2>&1 || rc=$?
      if [ "$rc" -ne 0 ]; then
        tail -120 "$log" >&2
        fail "runtime_${arm}_round_${round}"; return "$rc"
      fi
      grep -E '^FQ_(SHARD|TC_CELL|SHAPE_DONE) ' "$log" || true
      sha256sum "$log" >"$log.sha256" || return 2
    done
  done

  if ! python3 -B "$ANALYZER" analyze --bundle "$bundle" \
      --runs "$RUN_DIR/runs" --codegen "$RUN_DIR/codegen" \
      --execution "$RUN_DIR/inputs/execution.tsv" --threshold "$threshold" \
      --output-json "$RUN_DIR/results/summary.json" \
      --output-tsv "$RUN_DIR/results/summary.tsv"; then
    fail comparison; return 2
  fi
  sha256sum "$RUN_DIR/results/summary.json" "$RUN_DIR/results/summary.tsv" \
    "$RUN_DIR/inputs/execution.tsv" >"$RUN_DIR/results/result.sha256" || {
    fail result_hash; return 2;
  }
  printf 'FQ_M8_EPILOGUE_PERF_AB_GATE verdict=PASS shape=8x3072x512 arms=2 rounds=6 samples_per_arm=186 repeats=7 artifacts=%s\n' \
    "$RUN_DIR"
}

main "$@"
