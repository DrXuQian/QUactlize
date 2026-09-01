#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[ppu-runtime-bundle-build] FAIL: %s\n' "$*" >&2
  exit 1
}

assert_source_state() {
  local root="$1" source_sha="$2" recorded_submodules="$3" phase="$4"
  local head current_submodules recorded line
  head="$(git -C "$root" rev-parse HEAD 2>/dev/null)" ||
    fail "$phase: cannot resolve source HEAD"
  [[ "$head" == "$source_sha" ]] ||
    fail "$phase: source HEAD changed from $source_sha to $head"
  git -C "$root" diff --quiet --ignore-submodules=none "$source_sha" -- ||
    fail "$phase: tracked source or submodule state changed"
  current_submodules="$(git -C "$root" submodule status --recursive)" ||
    fail "$phase: cannot read submodule state"
  recorded="$(cat "$recorded_submodules")" ||
    fail "$phase: cannot read recorded submodule state"
  [[ "$current_submodules" == "$recorded" ]] ||
    fail "$phase: submodule commits changed"
  if printf '%s\n' "$current_submodules" | grep -Eq '^[+U-]'; then
    fail "$phase: submodules are not at the exact recorded commits"
  fi
  while IFS= read -r line; do
    [[ "$line" == '?? '* ]] || continue
    case "${line#?? }" in
      quactlize/*|third_party/*|cmake/*|tests/*|benchmarks/*|dev/*|ci/*|CMakeLists.txt|build.sh|setup.py|pyproject.toml)
        fail "$phase: untracked build input is not allowed: ${line#?? }" ;;
    esac
  done < <(git -C "$root" status --porcelain=v1 --untracked-files=all)
}

assert_role_source() {
  local build_dir="$1" role="$2" source_sha="$3" marker have
  marker="$build_dir/.quactlize-source-head"
  [[ -f "$marker" && ! -L "$marker" ]] ||
    fail "$role has no regular source-authority marker"
  IFS= read -r have <"$marker" || fail "$role source-authority marker is unreadable"
  [[ "$have" == "$source_sha" ]] ||
    fail "$role was built from $have, expected $source_sha"
  [[ ! -e "$build_dir/.quactlize-source-dirty" ]] ||
    fail "$role was built from tracked working-tree changes"
}

wait_build_batch() {
  local logs="$1" pair pid role
  local -a failed=()
  shift
  for pair in "$@"; do
    pid="${pair%%:*}"
    role="${pair#*:}"
    if wait "$pid"; then
      printf '[ppu-runtime-bundle-build] role=%s PASS\n' "$role"
    else
      failed+=("$role")
    fi
  done
  if ((${#failed[@]})); then
    for role in "${failed[@]}"; do
      printf '[ppu-runtime-bundle-build] role=%s failed; log tail follows\n' "$role" >&2
      tail -n 120 "$logs/$role.log" >&2 || true
    done
    return 1
  fi
}

terminate_active_builds() {
  local pair pid
  for pair in "${batch[@]}"; do
    pid="${pair%%:*}"
    # Each worker is a setsid process-group leader. Kill only that exact
    # negative PGID so make/hgcc/llc descendants cannot survive the builder.
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  for pair in "${batch[@]}"; do
    pid="${pair%%:*}"
    wait "$pid" 2>/dev/null || true
  done
  batch=()
}

on_build_signal() {
  local signal="$1" code="$2"
  trap - INT TERM
  terminate_active_builds
  printf '[ppu-runtime-bundle-build] FAIL: interrupted by %s; work preserved at %s and stage at %s\n' \
    "$signal" "$work" "$stage" >&2
  exit "$code"
}

on_build_exit() {
  local status=$?
  if ((status != 0 && ${#batch[@]})); then
    terminate_active_builds
  fi
}

main() {
  local root out parent sdk sdk_archive sdk_archive_sha sdk_release release_file jobs bundle_jobs
  local work stage records roles logs submodules_file
  local role packed_scale packed_format qtype filename defs build_dir log bin count
  local source_sha compiler
  local -a bins batch
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  [[ $# -eq 1 ]] || fail "usage: $0 OUT_DIR"
  out="$1"
  [[ "$out" = /* ]] || fail 'OUT_DIR must be absolute'
  parent="$(dirname "$out")"
  mkdir -p "$parent"
  parent="$(cd "$parent" && pwd -P)"
  out="$parent/$(basename "$out")"
  [[ ! -e "$out" && ! -L "$out" ]] || fail "refusing to overwrite $out"

  sdk="${PPU_SDK:-${PPU_HOME:-}}"
  [[ -n "$sdk" && -x "$sdk/bin/hgcc" && -x "$sdk/bin/hgobjdump" ]] ||
    fail 'set PPU_SDK to an installed PPU SDK with hgcc and hgobjdump'
  sdk_archive="${PPU_SDK_ARCHIVE:-}"
  [[ -n "$sdk_archive" && "$sdk_archive" = /* ]] ||
    fail 'set PPU_SDK_ARCHIVE to the absolute pinned PPU SDK archive path'
  [[ -f "$sdk_archive" && ! -L "$sdk_archive" ]] ||
    fail "PPU_SDK_ARCHIVE is not a regular non-symlink file: $sdk_archive"
  sdk_archive_sha="$(sha256sum "$sdk_archive" | awk '{print $1}')"
  [[ "$sdk_archive_sha" == '63ca196b152f2fec667fce8b18c04f1d6d0fa9e7bc7f72e18f017c96d11731dd' ]] ||
    fail "PPU_SDK_ARCHIVE digest is not the admitted 2.1.1-a5c56e archive: $sdk_archive_sha"
  release_file="$sdk/release.yaml"
  [[ -f "$release_file" && ! -L "$release_file" ]] ||
    fail "installed SDK has no regular release.yaml: $release_file"
  sdk_release="$(sed -n 's/^version:[[:space:]]*//p' "$release_file")"
  [[ "$sdk_release" == '2.1.1-a5c56e' ]] ||
    fail "installed SDK release is not admitted: ${sdk_release:-missing}"
  jobs="${JOBS:-2}"
  [[ "$jobs" =~ ^[1-9][0-9]*$ ]] || fail 'JOBS must be a positive integer'
  bundle_jobs="${PPU_BUNDLE_JOBS:-1}"
  [[ "$bundle_jobs" =~ ^[1-9][0-9]*$ && "$bundle_jobs" -le 6 ]] ||
    fail 'PPU_BUNDLE_JOBS must be an integer from 1 through 6'
  command -v setsid >/dev/null 2>&1 || fail 'setsid is required for isolated parallel build workers'

  source_sha="$(git -C "$root" rev-parse HEAD)"
  compiler="$($sdk/bin/hgcc --version 2>&1)"
  [[ -n "$compiler" && "$compiler" != *stub* ]] || fail 'hgcc identity is empty or a stub'
  [[ "$compiler" == *"Release version $sdk_release"* ]] ||
    fail 'hgcc release identity disagrees with the installed SDK receipt'
  compiler="$(printf '%s\n' "$compiler" | tr '\n' ';' | sed 's/;$//')"

  work="$(mktemp -d "$parent/.quactlize-ppu-build.XXXXXX")"
  stage="$(mktemp -d "$parent/.quactlize-ppu-bundle.XXXXXX")"
  records="$work/libraries.tsv"
  roles="$work/roles.tsv"
  logs="$work/build-logs"
  submodules_file="$work/submodules.status"
  mkdir -p "$logs"
  git -C "$root" submodule status --recursive >"$submodules_file" ||
    fail "cannot record submodule state; work preserved at $work and stage at $stage"
  assert_source_state "$root" "$source_sha" "$submodules_file" 'before build'

  cat >"$roles" <<'EOF'
default 0 -1 12 libquactlize_ppu.so
fmt0 1 0 12 libquactlize_ppu_fmt0.so
fmt1 1 1 13 libquactlize_ppu_fmt1.so
fmt2 1 2 10 libquactlize_ppu_fmt2.so
fmt3 1 3 11 libquactlize_ppu_fmt3.so
fmt4 1 4 14 libquactlize_ppu_fmt4.so
EOF
  printf 'role\tfilename\tpacked_scale\tpacked_format\tqtype\tdense_only\tsize\tsha256\tdefs\n' >"$records"
  printf '[ppu-runtime-bundle-build] source=%s role_workers=%s compile_jobs_per_role=%s\n' \
    "$source_sha" "$bundle_jobs" "$jobs"

  # role packed-scale packed-format-or--1 qtype installed-filename
  batch=()
  trap 'on_build_signal INT 130' INT
  trap 'on_build_signal TERM 143' TERM
  trap on_build_exit EXIT
  while read -r role packed_scale packed_format qtype filename; do
    build_dir="$work/$role"
    log="$logs/$role.log"
    defs="PPU_PACKED_SCALE=$packed_scale QUACTLIZE_DENSE_ONLY=$qtype"
    if [[ "$packed_format" -ge 0 ]]; then
      defs="$defs PPU_PACKED_FORMAT=$packed_format"
    fi
    printf '[ppu-runtime-bundle-build] role=%s defs=%s\n' "$role" "$defs"
    setsid env PPU_SDK="$sdk" PPU_ARCHS=ppu0010 PPU_BUILD_DIR="$build_dir" \
      PPU_BUILD_RESUME=0 PPU_DEFS="$defs" TARGET=quactlize_ppu JOBS="$jobs" \
      bash "$root/build.sh" >"$log" 2>&1 &
    batch+=("$!:$role")
    if ((${#batch[@]} == bundle_jobs)); then
      wait_build_batch "$logs" "${batch[@]}" ||
        fail "one or more builds failed; work preserved at $work and stage at $stage"
      batch=()
    fi
  done <"$roles"
  if ((${#batch[@]})); then
    wait_build_batch "$logs" "${batch[@]}" ||
      fail "one or more builds failed; work preserved at $work and stage at $stage"
  fi

  # Workers own only isolated build trees and logs. The parent installs and
  # records results in canonical order, so parallel completion cannot affect
  # the stage inventory or manifest bytes and no shared append needs a lock.
  while read -r role packed_scale packed_format qtype filename; do
    build_dir="$work/$role"
    assert_role_source "$build_dir" "$role" "$source_sha"
    mapfile -t bins < <(find "$build_dir" -type f -name libquactlize_ppu.so -print)
    count=${#bins[@]}
    [[ "$count" -eq 1 ]] ||
      fail "$role produced $count libquactlize_ppu.so files; work preserved at $work and stage at $stage"
    bin="${bins[0]}"
    install -m 0755 "$bin" "$stage/$filename"
    defs="PPU_PACKED_SCALE=$packed_scale QUACTLIZE_DENSE_ONLY=$qtype"
    if [[ "$packed_format" -ge 0 ]]; then
      defs="$defs PPU_PACKED_FORMAT=$packed_format"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$role" "$filename" "$packed_scale" \
      "$([[ "$packed_format" -ge 0 ]] && printf '%s' "$packed_format" || printf null)" \
      "$qtype" "$qtype" "$(stat -c %s "$stage/$filename")" \
      "$(sha256sum "$stage/$filename" | awk '{print $1}')" "$defs" >>"$records"
  done <"$roles"
  assert_source_state "$root" "$source_sha" "$submodules_file" 'after builds'

  python3 - "$root" "$stage" "$records" "$source_sha" "$compiler" \
    "$sdk_release" "$sdk_archive_sha" "$submodules_file" <<'PY'
import csv, json, pathlib, sys

root = pathlib.Path(sys.argv[1])
stage = pathlib.Path(sys.argv[2])
records = pathlib.Path(sys.argv[3])
source_sha = sys.argv[4]
compiler = sys.argv[5]
sdk_release = sys.argv[6]
sdk_archive_sha = sys.argv[7]
submodules_file = pathlib.Path(sys.argv[8])
libraries = []
with records.open(newline="", encoding="utf-8") as src:
    for row in csv.DictReader(src, delimiter="\t"):
        libraries.append({
            "role": row["role"],
            "filename": row["filename"],
            "packed_scale": int(row["packed_scale"]),
            "packed_format": None if row["packed_format"] == "null" else int(row["packed_format"]),
            "qtype": int(row["qtype"]),
            "dense_only": int(row["dense_only"]),
            "size": int(row["size"]),
            "sha256": row["sha256"],
            "definitions": row["defs"].split(),
        })
submodules = []
for line in submodules_file.read_text(encoding="utf-8").splitlines():
    fields = line[1:].split()
    if len(fields) >= 2:
        submodules.append({"commit": fields[0], "path": fields[1]})
manifest = {
    "schema": "quactlize.ppu-runtime-bundle",
    "schema_version": 1,
    "source": {
        "commit": source_sha,
        "tree_state": "clean",
        "submodules": submodules,
    },
    "toolchain": {
        "arch": "ppu0010",
        "sdk_release": sdk_release,
        "sdk_archive_sha256": sdk_archive_sha,
        "hgcc": compiler,
    },
    "libraries": libraries,
}
(stage / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

  PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m quactlize.ppu_bundle "$stage" --ppu-sdk "$sdk" ||
    fail "bundle verification failed; work preserved at $work and stage at $stage"
  PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}" \
    python3 "$root/tools/verify_kquant_selected_config.py" "$stage" ||
    fail "selected-config ABI verification failed; work preserved at $work and stage at $stage"
  while read -r role packed_scale packed_format qtype filename; do
    assert_role_source "$work/$role" "$role" "$source_sha"
  done <"$roles"
  assert_source_state "$root" "$source_sha" "$submodules_file" 'before publish'
  mv -- "$stage" "$out"

  # Only the exact mktemp directory created above is disposable.  Validate its
  # parent, basename and file type before removing the six intermediate build
  # trees; the installed bundle has already been atomically published.
  case "$work" in
    "$parent"/.quactlize-ppu-build.*) ;;
    *) fail "refusing to remove unexpected build path $work" ;;
  esac
  [[ -d "$work" && ! -L "$work" ]] || fail "build work path changed type: $work"
  rm -rf -- "$work"
  trap - EXIT INT TERM
  printf '[ppu-runtime-bundle-build] PASS source=%s libraries=6 output=%s\n' \
    "$source_sha" "$out"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
