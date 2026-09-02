#!/usr/bin/env bash
# Build and seal the exact FQ Split-K reducer lookup executable locally.
set -euo pipefail

fail() {
  printf '[fq-splitk-reducer-build] FAIL: %s\n' "$*" >&2
  exit 2
}

main() {
  [[ $# -le 1 ]] || fail "usage: $0 [BUNDLE_DIR]"
  local root artifact_root out work sdk jobs resume authority preflight build
  local build_resume log binary plan include build_make stage identity_probe identity_log
  local -a build_makes binaries

  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
  artifact_root="$(realpath -e -- "${FQ_SPLITK_REDUCER_ROOT:-/root/autodl-tmp}")" ||
    fail 'artifact root is absent'
  out="$(realpath -m -- "${1:-$artifact_root/fq-splitk-reducer-prebuilt-$(git -C "$root" rev-parse --short=8 HEAD)}")"
  case "$out" in "$artifact_root"/*) ;; *) fail 'bundle must be a strict artifact-root child' ;; esac
  work="$out.work"
  case "$work" in "$artifact_root"/*.work) ;; *) fail 'work path escaped artifact root' ;; esac
  sdk="$(realpath -e -- "${PPU_SDK:-${PPU_HOME:-/nonexistent}}")" ||
    fail 'exact PPU_SDK is required'
  jobs="${JOBS:-16}"
  resume="${RESUME:-0}"
  [[ "$jobs" =~ ^[1-9][0-9]*$ ]] || fail 'JOBS must be positive'
  case "$resume" in 0|1) ;; *) fail 'RESUME must be 0 or 1' ;; esac
  [[ -z "${PPU_DEFS:-}${PPU_EXTRA_DEFS:-}" ]] ||
    fail 'ambient PPU_DEFS/PPU_EXTRA_DEFS are forbidden'

  python3 -B "$root/tools/plan_fq_splitk_reducer_lookup.py" self-test
  python3 -B "$root/tools/analyze_fq_splitk_reducer_lookup.py" self-test
  python3 -B "$root/tools/fq_splitk_reducer_prebuilt.py" self-test
  if [[ -e "$out" || -L "$out" ]]; then
    [[ "$resume" = 1 && -d "$out" && ! -L "$out" ]] ||
      fail 'completed bundle exists; use RESUME=1 for strict verification'
    python3 -B "$root/tools/fq_splitk_reducer_prebuilt.py" verify \
      --bundle "$out" --source-root "$root" --sdk "$sdk"
    printf '[fq-splitk-reducer-build] PASS reused portable bundle=%s\n' "$out"
    return
  fi

  if [[ -e "$work" || -L "$work" ]]; then
    [[ "$resume" = 1 && -d "$work" && ! -L "$work" ]] ||
      fail 'incomplete work exists; use RESUME=1 or inspect it'
  else
    [[ "$resume" = 0 ]] || fail 'RESUME=1 requires existing work or bundle'
    mkdir "$work"
  fi
  authority="$work/build-authority.json"
  preflight="$work/global-preflight.json"
  if [[ -f "$authority" && ! -L "$authority" ]]; then
    [[ "$resume" = 1 ]] || fail 'unexpected existing build authority'
    python3 -B "$root/tools/fq_splitk_reducer_prebuilt.py" verify-build-authority \
      --file "$authority" --source-root "$root" --sdk "$sdk"
  else
    [[ ! -e "$authority" && ! -L "$authority" ]] || fail 'build authority has an invalid type'
    python3 -B "$root/tools/fq_splitk_reducer_prebuilt.py" write-build-authority \
      --output "$authority" --source-root "$root" --sdk "$sdk"
  fi
  if [[ -f "$preflight" && ! -L "$preflight" ]]; then
    [[ "$resume" = 1 ]] || fail 'unexpected existing global preflight receipt'
    python3 -B "$root/tools/kpack_global_build_preflight.py" verify \
      --root "$root" --receipt "$preflight"
  else
    [[ ! -e "$preflight" && ! -L "$preflight" ]] || fail 'preflight receipt has an invalid type'
    python3 -B "$root/tools/kpack_global_build_preflight.py" create \
      --root "$root" --output "$preflight"
  fi

  identity_probe="$work/box_identity_probe"
  identity_log="$work/identity-probe-build.log"
  if [[ -e "$identity_probe" || -L "$identity_probe" ]]; then
    [[ "$resume" = 1 && -f "$identity_probe" && ! -L "$identity_probe" &&
       -x "$identity_probe" && -s "$identity_log" && ! -L "$identity_log" ]] ||
      fail 'resumed identity probe evidence is incomplete'
  else
    [[ ! -e "$identity_log" && ! -L "$identity_log" ]] ||
      fail 'identity probe log exists without its binary'
    python3 -B - "$root" "$sdk" "$identity_probe" "$identity_log" <<'PY'
import os,pathlib,subprocess,sys
root,sdk,output,log=map(pathlib.Path,sys.argv[1:])
sys.path.insert(0,str(root/'tools'))
import probe_box_identity
command=probe_box_identity._probe_compile_command(sdk,output,dict(os.environ))
completed=subprocess.run(command,text=True,stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,check=False)
log.write_text('command='+repr(command)+'\n'+completed.stdout,encoding='utf-8')
if completed.returncode or not output.is_file():
 raise SystemExit(f'identity probe build failed rc={completed.returncode}')
output.chmod(output.stat().st_mode | 0o111)
PY
  fi

  build="$work/build"
  build_resume=0
  [[ ! -f "$build/CMakeCache.txt" ]] || build_resume=1
  log="$work/build.log"
  if [[ "$build_resume" = 1 ]]; then
    printf '[fq-splitk-reducer-build] RESUME existing CMake tree\n' >>"$log"
  else
    [[ ! -e "$log" && ! -L "$log" ]] || fail 'fresh build log already exists'
  fi
  env -u CC -u CXX -u CMAKE_GENERATOR -u CMAKE_TOOLCHAIN_FILE \
    -u PPU_DEFS -u PPU_EXTRA_DEFS \
    PPU_SDK="$sdk" PPU_BUILD_DIR="$build" PPU_BUILD_RESUME="$build_resume" \
    PPU_PRESERVE_STALE_BUILD_TREES=1 PPU_ARCHS=ppu0010 JOBS="$jobs" \
    QUACTLIZE_KPACK_GLOBAL_PREFLIGHT_RECEIPT="$preflight" \
    TARGET=test_fq_splitk_reducer_lookup bash "$root/build.sh" >>"$log" 2>&1 || {
      tail -n 160 "$log" >&2
      fail "build failed; preserved work=$work"
    }

  mapfile -t binaries < <(find -P "$build" -type f \
    -name test_fq_splitk_reducer_lookup -perm -u+x)
  mapfile -t build_makes < <(find -P "$build" -type f \
    -path '*test_fq_splitk_reducer_lookup.dir/build.make')
  [[ ${#binaries[@]} = 1 && ${#build_makes[@]} = 1 ]] ||
    fail 'build did not produce one exact binary/build.make'
  binary="${binaries[0]}"
  build_make="${build_makes[0]}"
  plan="$build/ppu_targets/fq_splitk_reducer_lookup/reducer-plan.json"
  include="$build/ppu_targets/fq_splitk_reducer_lookup/fq_splitk_reducer_lookup_cases.inc"
  [[ -f "$plan" && ! -L "$plan" && -f "$include" && ! -L "$include" &&
     -f "$build/cmake.log" && ! -L "$build/cmake.log" ]] ||
    fail 'generated plan/include/CMake evidence is missing'

  stage="$artifact_root/.$(basename "$out").publish.$$"
  [[ ! -e "$stage" && ! -L "$stage" ]] || fail 'publish stage exists'
  python3 -B "$root/tools/fq_splitk_reducer_prebuilt.py" create \
    --bundle "$stage" --source-root "$root" --sdk "$sdk" \
    --build-authority "$authority" --global-preflight "$preflight" \
    --binary "$binary" --generated-plan "$plan" --generated-include "$include" \
    --build-log "$log" --cmake-log "$build/cmake.log" --build-make "$build_make" \
    --identity-probe "$identity_probe" --identity-probe-build-log "$identity_log"
  [[ ! -e "$out" && ! -L "$out" ]] || fail 'bundle appeared during publish'
  mv -- "$stage" "$out"
  python3 -B "$root/tools/fq_splitk_reducer_prebuilt.py" verify \
    --bundle "$out" --source-root "$root" --sdk "$sdk"
  printf '[fq-splitk-reducer-build] PASS portable bundle=%s work=%s\n' "$out" "$work"
}

main "$@"
