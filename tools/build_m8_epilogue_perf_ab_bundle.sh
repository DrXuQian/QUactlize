#!/usr/bin/env bash
# Build the two exact source arms used by the TM8 epilogue performance gate.
# The resulting bundle is compile-free on the PPU box.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
ANALYZER="$ROOT/tools/analyze_m8_epilogue_perf_ab.py"
CANDIDATE_SOURCE=6ec447ac25477a40a29be8c2809a933c84d0b7ad
CANDIDATE_ACTLIZE=423253c00df333ead6fb72ea623d526f24f56b5a
BASELINE_SOURCE=a0fa8d03013d3cd0bc340e876cb0d646f3cfb72d
BASELINE_TREE=1bdee8cc1386ca29e8116d893b63f705b118eecc
BASELINE_PARENT=$CANDIDATE_SOURCE
BASELINE_ACTLIZE=9d063e4c5fde5119d4d68bfbe124aacd8ed2ec88
CUTLASS_SOURCE=f94ec46f4f63f96003d6cfdf2014731e7672c281
TARGET=test_fully_quantized_internal_sweep
OUT_DIR="$(realpath -m -- "${OUT:-/root/autodl-tmp/m8-epilogue-perf-ab-build-9d063e4-423253c}")"

fail() {
  printf '[m8-epilogue-perf-ab-build] FAIL phase=%s artifacts=%s\n' "$1" "$OUT_DIR" >&2
  return 1
}

create_baseline_source() {
  local index=$1 tree commit changed
  [ ! -e "$index" ] || return 1
  GIT_INDEX_FILE="$index" git -C "$ROOT" read-tree "$BASELINE_PARENT" || return 1
  GIT_INDEX_FILE="$index" git -C "$ROOT" update-index --add --cacheinfo \
    "160000,$BASELINE_ACTLIZE,third_party/actlize" || return 1
  tree="$(GIT_INDEX_FILE="$index" git -C "$ROOT" write-tree)" || return 1
  [ "$tree" = "$BASELINE_TREE" ] || return 1
  commit="$(printf '%s\n' \
      'Build authority: parent 6ec447a with pre-fix actlize 9d063e4c' | \
    env GIT_AUTHOR_NAME='Quactlize Build Authority' \
      GIT_AUTHOR_EMAIL='build-authority@invalid' \
      GIT_AUTHOR_DATE='2000-01-01T00:00:00+0000' \
      GIT_COMMITTER_NAME='Quactlize Build Authority' \
      GIT_COMMITTER_EMAIL='build-authority@invalid' \
      GIT_COMMITTER_DATE='2000-01-01T00:00:00+0000' \
      git -C "$ROOT" commit-tree "$tree" -p "$BASELINE_PARENT")" || return 1
  [ "$commit" = "$BASELINE_SOURCE" ] || return 1
  changed="$(git -C "$ROOT" diff-tree --no-commit-id --name-only -r "$commit")" || return 1
  [ "$changed" = third_party/actlize ] || return 1
  [ "$(git -C "$ROOT" ls-tree "$commit" third_party/actlize | awk '{print $3}')" = \
    "$BASELINE_ACTLIZE" ] || return 1
  printf '[m8-epilogue-perf-ab-build] synthetic baseline=%s parent=%s tree=%s only=third_party/actlize\n' \
    "$commit" "$BASELINE_PARENT" "$tree"
}

prepare_source() {
  local name=$1 revision=$2 actlize=$3 source=$4 log=$5
  git -C "$ROOT" worktree add --detach "$source" "$revision" >"$log" 2>&1 || return 1
  git -C "$source" -c protocol.file.allow=always submodule update --init --recursive \
    >>"$log" 2>&1 || return 1
  [ "$(git -C "$source" rev-parse HEAD)" = "$revision" ] || return 1
  [ "$(git -C "$source/third_party/actlize" rev-parse HEAD)" = "$actlize" ] || return 1
  [ "$(git -C "$source/third_party/cutlass" rev-parse HEAD)" = "$CUTLASS_SOURCE" ] || return 1
  [ -z "$(git -C "$source" status --porcelain --untracked-files=no)" ] || return 1
  printf '[m8-epilogue-perf-ab-build] source arm=%s sha=%s actlize=%s\n' \
    "$name" "$revision" "$actlize"
}

make_preflight() {
  local source=$1 receipt=$2 log=$3
  python3 -B "$source/tools/kpack_global_build_preflight.py" create \
    --root "$source" --output "$receipt" >"$log" 2>&1
}

build_arm() {
  local name=$1 source=$2 build=$3 receipt=$4 log=$5
  printf '[m8-epilogue-perf-ab-build] compile arm=%s source=%s\n' "$name" "$source"
  (cd "$source" && env \
    PPU_SDK="$PPU_SDK_ROOT" PPU_BUILD_DIR="$build" PPU_BUILD_RESUME=0 \
    PPU_ARCHS=ppu0010 JOBS="$JOBS_PER_ARM" \
    QUACTLIZE_KPACK_GLOBAL_PREFLIGHT_RECEIPT="$receipt" \
    TARGET="$TARGET" FQ_SWEEP_GENERATED_DIR="$GENERATED" \
    FQ_SWEEP_QTYPE=12 FQ_SWEEP_ARTIFACT_TK=0 FQ_SWEEP_BCHUNK=0 \
    FQ_SWEEP_PACKED_FORMAT=0 FQ_SWEEP_WEIGHT_LAYOUT=1 \
    PPU_DEFS= PPU_EXTRA_DEFS= CFLAGS= CXXFLAGS= CPPFLAGS= LDFLAGS= \
    ./build.sh) >"$log" 2>&1
}

main() {
  local default_sdk=${PPU_SDK:-${PPU_HOME:-}}
  [ -n "$default_sdk" ] || { fail missing_ppu_sdk; return 1; }
  PPU_SDK_ROOT="$(realpath -e -- "$default_sdk" 2>/dev/null)" || {
    fail invalid_ppu_sdk; return 1;
  }
  [ -x "$PPU_SDK_ROOT/bin/hgcc" ] && [ -x "$PPU_SDK_ROOT/bin/hgobjdump" ] || {
    fail sdk_tools; return 1;
  }
  case "${JOBS_PER_ARM:-16}" in
    ''|*[!0-9]*|0) fail invalid_jobs_per_arm; return 1 ;;
  esac
  JOBS_PER_ARM=${JOBS_PER_ARM:-16}
  [ ! -e "$OUT_DIR" ] || { fail output_already_exists; return 1; }
  mkdir -p "$OUT_DIR"/{source,generated,build,preflight,logs,bundle/bin,bundle/inputs} || {
    fail create_output; return 1;
  }

  create_baseline_source "$OUT_DIR/preflight/baseline.index" || {
    fail synthetic_baseline; return 1;
  }
  python3 -B "$ANALYZER" self-test || { fail analyzer_self_test; return 1; }
  for revision in "$BASELINE_SOURCE" "$CANDIDATE_SOURCE"; do
    git -C "$ROOT" cat-file -e "$revision^{commit}" || {
      fail missing_source_commit; return 1;
    }
  done
  local relevant
  for relevant in \
      build.sh CMakeLists.txt quactlize/csrc/CMakeLists.txt.in \
      benchmarks/test_fully_quantized_internal_sweep.cu \
      benchmarks/fully_quantized_splitk_producer_bench.hpp \
      benchmarks/fully_quantized_splitk_producer_unit.inc \
      tools/gen_fully_quantized_kpack_discovery_units.py \
      tools/fully_quantized_kpack_discovery_matrix.py \
      tools/scalefirst_internal_matrix.py \
      quactlize/include/ppu_mixed_policy.hpp \
      quactlize/include/fpA_intB_ppu.cuh; do
    git -C "$ROOT" diff --quiet "$BASELINE_SOURCE..$CANDIDATE_SOURCE" -- "$relevant" || {
      printf 'changed relevant input: %s\n' "$relevant" >&2
      fail source_arm_not_isomorphic; return 1
    }
  done

  BASELINE_ROOT="$OUT_DIR/source/baseline"
  CANDIDATE_ROOT="$OUT_DIR/source/candidate"
  prepare_source baseline "$BASELINE_SOURCE" "$BASELINE_ACTLIZE" \
    "$BASELINE_ROOT" "$OUT_DIR/logs/source-baseline.log" || {
    tail -80 "$OUT_DIR/logs/source-baseline.log" >&2
    fail prepare_baseline; return 1
  }
  prepare_source candidate "$CANDIDATE_SOURCE" "$CANDIDATE_ACTLIZE" \
    "$CANDIDATE_ROOT" "$OUT_DIR/logs/source-candidate.log" || {
    tail -80 "$OUT_DIR/logs/source-candidate.log" >&2
    fail prepare_candidate; return 1
  }

  GENERATED="$OUT_DIR/generated/row"
  if ! python3 -B "$CANDIDATE_ROOT/tools/gen_fully_quantized_kpack_discovery_units.py" \
      --qtype 12 --out-dir "$GENERATED" \
      --parent-begin 4809 --parent-count 1 --per-unit 1 \
      >"$OUT_DIR/logs/generate.log" 2>&1; then
    cat "$OUT_DIR/logs/generate.log" >&2
    fail generate; return 1
  fi

  local baseline_preflight_rc=0 candidate_preflight_rc=0
  make_preflight "$BASELINE_ROOT" "$OUT_DIR/preflight/baseline.json" \
    "$OUT_DIR/logs/preflight-baseline.log" &
  local baseline_preflight_pid=$!
  make_preflight "$CANDIDATE_ROOT" "$OUT_DIR/preflight/candidate.json" \
    "$OUT_DIR/logs/preflight-candidate.log" &
  local candidate_preflight_pid=$!
  wait "$baseline_preflight_pid" || baseline_preflight_rc=$?
  wait "$candidate_preflight_pid" || candidate_preflight_rc=$?
  [ "$baseline_preflight_rc" -eq 0 ] || {
    tail -100 "$OUT_DIR/logs/preflight-baseline.log" >&2
    fail baseline_preflight; return 1
  }
  [ "$candidate_preflight_rc" -eq 0 ] || {
    tail -100 "$OUT_DIR/logs/preflight-candidate.log" >&2
    fail candidate_preflight; return 1
  }

  local baseline_build_rc=0 candidate_build_rc=0
  build_arm baseline "$BASELINE_ROOT" "$OUT_DIR/build/baseline" \
    "$OUT_DIR/preflight/baseline.json" "$OUT_DIR/logs/build-baseline.log" &
  local baseline_build_pid=$!
  build_arm candidate "$CANDIDATE_ROOT" "$OUT_DIR/build/candidate" \
    "$OUT_DIR/preflight/candidate.json" "$OUT_DIR/logs/build-candidate.log" &
  local candidate_build_pid=$!
  wait "$baseline_build_pid" || baseline_build_rc=$?
  wait "$candidate_build_pid" || candidate_build_rc=$?
  [ "$baseline_build_rc" -eq 0 ] || {
    tail -160 "$OUT_DIR/logs/build-baseline.log" >&2
    fail build_baseline; return 1
  }
  [ "$candidate_build_rc" -eq 0 ] || {
    tail -160 "$OUT_DIR/logs/build-candidate.log" >&2
    fail build_candidate; return 1
  }

  local baseline_bin="$OUT_DIR/build/baseline/ppu_targets/$TARGET"
  local candidate_bin="$OUT_DIR/build/candidate/ppu_targets/$TARGET"
  [ -x "$baseline_bin" ] && [ -x "$candidate_bin" ] || {
    fail binary_missing; return 1;
  }
  [ "$(cat "$OUT_DIR/build/baseline/.quactlize-source-head")" = "$BASELINE_SOURCE" ] || {
    fail baseline_build_authority; return 1;
  }
  [ "$(cat "$OUT_DIR/build/candidate/.quactlize-source-head")" = "$CANDIDATE_SOURCE" ] || {
    fail candidate_build_authority; return 1;
  }
  install -m 0755 "$baseline_bin" "$OUT_DIR/bundle/bin/baseline" || return 1
  install -m 0755 "$candidate_bin" "$OUT_DIR/bundle/bin/candidate" || return 1
  install -m 0644 "$GENERATED/manifest.json" \
    "$OUT_DIR/bundle/inputs/generated-manifest.json" || return 1
  install -m 0644 "$GENERATED/fq_tc_registry.inc" \
    "$OUT_DIR/bundle/inputs/fq_tc_registry.inc" || return 1
  install -m 0644 "$GENERATED/units/fq_kpack_dense_unit_00000.cu" \
    "$OUT_DIR/bundle/inputs/fq_kpack_dense_unit_00000.cu" || return 1
  printf '%s\n' fqk_tc_q12_l1_a0_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap0_dn16 \
    >"$OUT_DIR/bundle/inputs/symbol.txt"
  python3 -B "$ANALYZER" make-bundle-manifest \
    --bundle "$OUT_DIR/bundle" --sdk "$PPU_SDK_ROOT" \
    --baseline "$OUT_DIR/bundle/bin/baseline" \
    --candidate "$OUT_DIR/bundle/bin/candidate" || {
    fail manifest; return 1;
  }
  python3 -B "$ANALYZER" verify-bundle --bundle "$OUT_DIR/bundle" || {
    fail verify_bundle; return 1;
  }
  "$PPU_SDK_ROOT/bin/hgobjdump" -lelf "$OUT_DIR/bundle/bin/baseline" \
    >"$OUT_DIR/logs/baseline.list-elf.txt" 2>&1 || {
    fail inspect_baseline; return 1;
  }
  "$PPU_SDK_ROOT/bin/hgobjdump" -lelf "$OUT_DIR/bundle/bin/candidate" \
    >"$OUT_DIR/logs/candidate.list-elf.txt" 2>&1 || {
    fail inspect_candidate; return 1;
  }
  printf 'M8_EPILOGUE_PERF_AB_BUNDLE verdict=READY baseline=%s candidate=%s bundle=%s\n' \
    "$BASELINE_SOURCE" "$CANDIDATE_SOURCE" "$OUT_DIR/bundle"
}

main "$@"
