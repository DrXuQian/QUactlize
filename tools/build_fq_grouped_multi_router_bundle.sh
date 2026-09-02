#!/usr/bin/env bash
# Build a resumable five-format grouped multi-router bundle for box handoff.
set -uo pipefail

fail() {
  printf '[fq-grouped-multi-router-build] FAIL: %s\n' "$*" >&2
  return 2
}

terminate_worker_groups() {
  local pgid pid
  for pgid in "${worker_pgids[@]:-}"; do
    if [[ "$pgid" =~ ^[0-9]+$ ]] && [ "$pgid" -gt 1 ] && \
       kill -0 -- "-$pgid" 2>/dev/null; then
      kill -TERM -- "-$pgid" 2>/dev/null || true
    fi
  done
  for pid in "${worker_pids[@]:-}"; do wait "$pid" 2>/dev/null || true; done
}

wait_worker_batch() {
  local index pid pgid rc=0
  for index in "${!worker_pids[@]}"; do
    pid="${worker_pids[$index]}"
    wait "$pid"
    rc=$?
    if [ "$rc" -ne 0 ]; then
      printf '[fq-grouped-multi-router-build] worker q=%s failed\n' \
        "${worker_qtypes[$index]}" >&2
      terminate_worker_groups
      return "$rc"
    fi
  done
  worker_pids=(); worker_pgids=(); worker_qtypes=()
}

main() {
  [ "$#" -eq 0 ] || { fail 'no positional arguments'; return 2; }
  local root out sdk jobs resume release authority current_authority bundle_jobs
  local q fmt build log binary library target_make build_resume rc
  local -a source_paths manifest_args worker_pids worker_pgids worker_qtypes
  trap 'terminate_worker_groups' EXIT
  trap 'terminate_worker_groups; exit 130' INT
  trap 'terminate_worker_groups; exit 143' TERM
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  out="$(realpath -m -- "${OUT:-/root/autodl-tmp/fq-grouped-multi-router-bundle}")" || return 2
  case "$out" in /root/autodl-tmp/*) ;; *) fail 'OUT must be a strict /root/autodl-tmp child'; return 2;; esac
  resume="${RESUME:-0}"
  case "$resume" in 0|1) ;; *) fail 'RESUME must be 0 or 1'; return 2;; esac
  if [ -e "$out" ]; then
    [ "$resume" = 1 ] || { fail "OUT exists without RESUME=1: $out"; return 2; }
  else
    [ "$resume" = 0 ] || { fail 'RESUME=1 requires an existing OUT'; return 2; }
  fi
  jobs="${JOBS:-16}"
  bundle_jobs="${PPU_BUNDLE_JOBS:-2}"
  [[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { fail 'JOBS must be a positive integer'; return 2; }
  case "$bundle_jobs" in 1|2|3|4|5) ;; *) fail 'PPU_BUNDLE_JOBS must be in [1,5]'; return 2;; esac
  [ -z "$(git -C "$root" status --porcelain -- \
      build.sh CMakeLists.txt quactlize benchmarks \
      tools/fq_grouped_multi_router.py \
      tools/plan_fq_grouped_multi_router.py \
      tools/analyze_fq_grouped_multi_router.py \
      tools/fq_grouped_multi_router_manifest.py \
      tools/build_fq_grouped_multi_router_bundle.sh \
      tools/run_fq_grouped_multi_router_prebuilt_box.sh \
      tools/probe_box_identity.py \
      ci/check_fq_grouped_multi_router.py)" ] || {
    fail 'tracked/staged/untracked build input is dirty'
    return 2
  }
  sdk="${PPU_SDK:-${PPU_HOME:-}}"
  [ -x "$sdk/bin/hgcc" ] && [ -x "$sdk/bin/hgobjdump" ] || { fail 'real PPU SDK is required'; return 2; }
  [[ "$("$sdk/bin/hgcc" --version 2>&1 | head -n1 || true)" != *stub* ]] || { fail 'stub hgcc is forbidden'; return 2; }
  release="$sdk/release.yaml"
  [ -f "$release" ] || { fail 'SDK release.yaml is required'; return 2; }
  source_paths=(
    build.sh CMakeLists.txt quactlize/csrc/CMakeLists.txt.in
    quactlize/csrc/fq_kquant_layout_perf.cmake.in
    quactlize/csrc/fq_grouped_multi_router_perf.cmake.in
    benchmarks/test_fq_kquant_layout_perf.cu
    benchmarks/test_fq_grouped_multi_router_perf.cu
    quactlize/csrc/device/ppu_dense_backend.cu
    quactlize/include/ppu_grouped_configs.inc
    quactlize/include/ppu_q4_kpack4_shipping_policy.hpp
    tools/fq_grouped_multi_router.py
    tools/plan_fq_grouped_multi_router.py
    tools/analyze_fq_grouped_multi_router.py
    tools/fq_grouped_multi_router_manifest.py
    tools/build_fq_grouped_multi_router_bundle.sh
    tools/run_fq_grouped_multi_router_prebuilt_box.sh
    tools/probe_box_identity.py
    ci/check_fq_grouped_multi_router.py
  )
  git -C "$root" submodule foreach --quiet --recursive \
    'test -z "$(git status --porcelain)"' || {
      fail 'recursive submodule worktree is dirty'
      return 2
    }
  mkdir -p "$out/bin" "$out/build" "$out/logs" "$out/inputs" || return 2
  authority="$out/inputs/build-authority.json"
  current_authority="$out/inputs/build-authority.current.json"
  python3 -B - "$root" "$sdk" "$release" "$current_authority" "${source_paths[@]}" <<'PY' || return 2
import hashlib
import json
import pathlib
import subprocess
import sys

root, sdk, release, output = map(pathlib.Path, sys.argv[1:5])
source_paths = sys.argv[5:]
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
submodules = {}
for line in subprocess.check_output(
        ["git", "-C", str(root), "submodule", "status", "--recursive"],
        text=True).splitlines():
    fields = line.strip().split()
    submodules[fields[1]] = fields[0]
document = {
    "schema": "quactlize.fq-grouped-multi-router-build-authority.v1",
    "source_sha": subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "source_files": {path: digest(root / path) for path in source_paths},
    "submodules": submodules,
    "sdk": {
        "release_sha256": digest(release),
        "compiler_sha256": digest(sdk / "bin/hgcc"),
        "inspector_sha256": digest(sdk / "bin/hgobjdump"),
    },
    "build": {
        "target": "test_fq_grouped_multi_router_perf",
        "arch": "ppu0010",
        "qtypes": [10, 11, 12, 13, 14],
    },
}
output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
PY
  if [ "$resume" = 1 ]; then
    [ -f "$authority" ] && cmp -s "$authority" "$current_authority" || {
      fail 'RESUME build authority differs'
      return 2
    }
    rm -f -- "$current_authority"
  else
    mv -- "$current_authority" "$authority" || return 2
  fi

  for q in 10 11 12 13 14; do
    case "$q" in 10) fmt=2;; 11) fmt=3;; 12) fmt=0;; 13) fmt=1;; 14) fmt=4;; esac
    build="$out/build/q$q"
    log="$out/logs/build-q$q.log"
    binary="$out/bin/q$q/test_fq_grouped_multi_router_perf"
    library="$out/bin/q$q/libquactlize_ppu.so"
    if [ "$resume" = 1 ] && [ -x "$binary" ] && [ -f "$library" ] && \
       [ -f "$out/bin/q$q/payload.sha256" ]; then
      (cd "$out" && sha256sum -c "bin/q$q/payload.sha256") >/dev/null || {
        fail "q$q completed payload differs"
        return 2
      }
      printf '[fq-grouped-multi-router-build] reuse q=%s\n' "$q"
    else
      build_resume=0
      [ ! -e "$build" ] || build_resume=1
      env -u CC -u CXX -u CMAKE_GENERATOR -u CMAKE_TOOLCHAIN_FILE \
        PPU_BUILD_DIR="$build" PPU_BUILD_RESUME="$build_resume" \
        PPU_ARCHS=ppu0010 JOBS="$jobs" \
        TARGET=test_fq_grouped_multi_router_perf FQ_KQUANT_PERF_QTYPE="$q" \
        PPU_DEFS="PPU_PACKED_SCALE=1 PPU_PACKED_FORMAT=$fmt QUACTLIZE_DENSE_ONLY=$q" \
        setsid bash -c 'cd "$1" && exec ./build.sh' _ "$root" > "$log" 2>&1 &
      worker_pids+=("$!")
      worker_qtypes+=("$q")
      worker_pgids+=("$(ps -o pgid= -p "$!" | tr -d ' ')")
      [[ "${worker_pgids[-1]}" =~ ^[0-9]+$ ]] && \
        [ "${worker_pgids[-1]}" = "${worker_pids[-1]}" ] || {
        [ -z "${worker_pgids[-1]}" ] || kill -TERM -- "-${worker_pgids[-1]}" 2>/dev/null || true
        wait "${worker_pids[-1]}" 2>/dev/null || true
        fail "q$q worker did not acquire an exact private PGID"
        return 2
      }
      if [ "${#worker_pids[@]}" -eq "$bundle_jobs" ]; then
        wait_worker_batch || { fail 'parallel build batch failed; rerun with RESUME=1'; return 2; }
      fi
    fi
  done
  [ "${#worker_pids[@]}" -eq 0 ] || \
    wait_worker_batch || { fail 'parallel build batch failed; rerun with RESUME=1'; return 2; }

  for q in 10 11 12 13 14; do
    case "$q" in 10) fmt=2;; 11) fmt=3;; 12) fmt=0;; 13) fmt=1;; 14) fmt=4;; esac
    build="$out/build/q$q"
    log="$out/logs/build-q$q.log"
    binary="$(find "$build" -type f -name test_fq_grouped_multi_router_perf -perm -u+x -print -quit)"
    library="$(find "$build" -type f -name libquactlize_ppu.so -print -quit)"
    [ -x "$binary" ] && [ -f "$library" ] && [ ! -L "$binary" ] && [ ! -L "$library" ] || { fail "q$q build output missing"; return 2; }
    target_make="$(find "$build" -type f -path '*test_fq_grouped_multi_router_perf.dir/build.make' -print -quit)"
    grep -Fqx "[build.sh] FQ_KQUANT_PERF_QTYPE=$q" "$log" && \
      grep -F "PPU_PACKED_FORMAT=$fmt" "$log" >/dev/null && \
      grep -F "QUACTLIZE_DENSE_ONLY=$q" "$log" >/dev/null && \
      grep -F "FullyQuantized grouped multi-router perf: qtype=$q layout=K-pack-only" "$build/cmake.log" >/dev/null && \
      [ -n "$target_make" ] && grep -F -- "-DFQ_KQUANT_PERF_QTYPE=$q" "$target_make" >/dev/null || { fail "q$q build identity differs"; return 2; }
    if [ ! -x "$out/bin/q$q/test_fq_grouped_multi_router_perf" ]; then
      mkdir -p "$out/bin/q$q" || return 2
      cp -- "$binary" "$out/bin/q$q/test_fq_grouped_multi_router_perf" || return 2
      cp -- "$library" "$out/bin/q$q/libquactlize_ppu.so" || return 2
      (cd "$out" && sha256sum "bin/q$q/test_fq_grouped_multi_router_perf" \
        "bin/q$q/libquactlize_ppu.so" > "bin/q$q/payload.sha256") || return 2
    fi
    manifest_args+=("$q" "bin/q$q/test_fq_grouped_multi_router_perf" \
      "$(sha256sum "$out/bin/q$q/test_fq_grouped_multi_router_perf" | awk '{print $1}')" \
      "bin/q$q/libquactlize_ppu.so" \
      "$(sha256sum "$out/bin/q$q/libquactlize_ppu.so" | awk '{print $1}')")
  done
  python3 -B - "$authority" "$out" "${manifest_args[@]}" <<'PY' || return 2
import json
import pathlib
import sys

authority = json.load(open(sys.argv[1]))
output = pathlib.Path(sys.argv[2])
args = sys.argv[3:]
binaries = {
    args[index]: {
        "path": args[index + 1],
        "sha256": args[index + 2],
        "library_path": args[index + 3],
        "library_sha256": args[index + 4],
    }
    for index in range(0, len(args), 5)
}
document = {
    "schema": "quactlize.fq-grouped-multi-router-prebuilt.v1",
    "source_sha": authority["source_sha"],
    "source_files": authority["source_files"],
    "submodules": authority["submodules"],
    "sdk": authority["sdk"],
    "build": authority["build"],
    "binaries": binaries,
}
(output / "manifest.json").write_text(
    json.dumps(document, indent=2, sort_keys=True) + "\n")
PY
  printf '[fq-grouped-multi-router-build] PASS bundle=%s\n' "$out"
}

main "$@"
