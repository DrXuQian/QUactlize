#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[l254:q4-n16k64-fragment] FAIL: %s\n' "$*" >&2
  exit 1
}

main() {
  local root sdk sdk_archive sdk_archive_sha sdk_release compiler_identity
  local out build target source binary list_elf isa symbol defs
  local source_sha actlize_sha cutlass_sha
  local aiu_any aiu_plain reader_any reader_x4
  local converter_lop converter_add converter_fma
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  sdk="${PPU_SDK:-${PPU_HOME:-${PPU_SDK_SITE_DEFAULT:-}}}"
  [[ -n "$sdk" && -x "$sdk/bin/hgcc" && -x "$sdk/bin/hgobjdump" ]] ||
    fail 'set PPU_SDK to a real PPU SDK root'
  sdk="$(realpath -e -- "$sdk")"
  [[ -f "$sdk/release.yaml" && ! -L "$sdk/release.yaml" ]] ||
    fail "missing regular SDK receipt: $sdk/release.yaml"
  sdk_release="$(sed -n 's/^version:[[:space:]]*//p' "$sdk/release.yaml")"
  [[ "$sdk_release" == '2.1.1-a5c56e' ]] ||
    fail "this gate is pinned to SDK 2.1.1-a5c56e, got ${sdk_release:-missing}"
  sdk_archive="${PPU_SDK_ARCHIVE:-}"
  [[ -n "$sdk_archive" && "$sdk_archive" = /* && -f "$sdk_archive" &&
     ! -L "$sdk_archive" ]] ||
    fail 'set PPU_SDK_ARCHIVE to the absolute regular pinned SDK archive'
  sdk_archive_sha="$(sha256sum "$sdk_archive" | awk '{print $1}')"
  [[ "$sdk_archive_sha" == '63ca196b152f2fec667fce8b18c04f1d6d0fa9e7bc7f72e18f017c96d11731dd' ]] ||
    fail "PPU_SDK_ARCHIVE digest is not admitted: $sdk_archive_sha"
  compiler_identity="$("$sdk/bin/hgcc" --version 2>&1 | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
  [[ -n "$compiler_identity" && "$compiler_identity" != *stub* &&
     "$compiler_identity" == *"Release version $sdk_release"* ]] ||
    fail 'hgcc identity is empty, a stub, or disagrees with the SDK receipt'

  git -C "$root" diff --quiet --ignore-submodules=none HEAD -- ||
    fail 'tracked source or submodule state is dirty; commit the exact candidate first'
  if git -C "$root" submodule status --recursive | grep -Eq '^[+U-]'; then
    fail 'submodules are not at the exact recorded commits'
  fi
  while IFS= read -r line; do
    [[ "$line" == '?? '* ]] || continue
    case "${line#?? }" in
      quactlize/*|third_party/*|cmake/*|CMakeLists.txt|build.sh|dev/test_q4_n16k64_fragment_numeric.cu)
        fail "untracked build input is not allowed: ${line#?? }" ;;
    esac
  done < <(git -C "$root" status --porcelain=v1 --untracked-files=all)

  out="${QUACTLIZE_L254_OUT:-/root/autodl-tmp/quactlize-l254-q4-n16k64-fragment}"
  build="$out/build"
  target=test_q4_n16k64_fragment_numeric
  source="$root/dev/test_q4_n16k64_fragment_numeric.cu"
  defs='PPU_PACKED_SCALE=1 PPU_PACKED_FORMAT=0 QUACTLIZE_DENSE_ONLY=12'
  mkdir -p "$out"

  for token in \
    'q4_n16k64_direct::prepare' \
    'AiuPlainProvider<kN, kK>' \
    'Q4N16K64UniversalReader<kN, kWarpN, kK>' \
    'MixGemmNumericArrayConverter<' \
    'partition_fragment_B(logical_smem)' \
    '--plant-layout1' \
    '--plant-deepgemm' \
    'hggcDeviceSynchronize()'; do
    grep -F -- "$token" "$source" >/dev/null ||
      fail "numeric source lost required seam: $token"
  done
  printf '[l254:q4-n16k64-fragment:source] PASS layout3 prepare, exact physical chain, converter, MMA owner, poison and two wrong maps\n'

  PPU_SDK="$sdk" \
  PPU_ARCHS=ppu0010 \
  PPU_BUILD_DIR="$build" \
  PPU_BUILD_RESUME=0 \
  PPU_DEFS="$defs" \
  TARGET="$target" \
  JOBS="${JOBS:-16}" \
    "$root/build.sh" >"$out/build.log" 2>&1 || {
      tail -n 180 "$out/build.log" >&2
      fail 'runnable PPU numeric target did not build'
    }

  mapfile -t binaries < <(
    find "$build" -type f -name "$target" -perm -u+x -print | sort
  )
  [[ ${#binaries[@]} -eq 1 ]] ||
    fail "expected one $target executable, found ${#binaries[@]}"
  binary="${binaries[0]}"
  list_elf="$out/list-elf.txt"
  isa="$out/isa.txt"
  "$sdk/bin/hgobjdump" --list-elf "$binary" >"$list_elf" \
    2>"$out/list-elf.err" || fail 'hgobjdump could not parse the binary'
  "$sdk/bin/hgobjdump" --dump-isa "$binary" >"$isa" \
    2>"$out/isa.err" || fail 'hgobjdump could not disassemble the binary'
  [[ "$(grep -c '^Func [0-9][0-9]*:' "$list_elf" || true)" -eq 1 ]] ||
    fail 'numeric executable must contain exactly one device kernel'
  symbol="$(awk '/^Func [0-9][0-9]*:/ {print $3}' "$list_elf")"
  [[ "$symbol" == *q4_n16k64_fragment_numeric_kernel* ]] ||
    fail "unexpected device kernel: $symbol"
  grep -F "Disassembly of section .text.kernel.$symbol" "$isa" >/dev/null ||
    fail 'ISA is not bound to the listed numeric kernel'

  aiu_any="$(grep -c 'vmem\.aiu\.ld\.tsm' "$isa" || true)"
  aiu_plain="$(grep -c \
    'vmem\.aiu\.ld\.tsm\.l1\.t0\.p0\.s0\.m0\.2d\.b32\.kp1' \
    "$isa" || true)"
  reader_any="$(grep -c 'tsm\.ld\.' "$isa" || true)"
  reader_x4="$(grep -c 'tsm\.ld\.b32x4' "$isa" || true)"
  converter_lop="$(grep -c 'v\.lop3\.b32' "$isa" || true)"
  converter_add="$(grep -c 'v\.add\.f16x2' "$isa" || true)"
  converter_fma="$(grep -c 'v\.fma\.f16x2' "$isa" || true)"
  [[ "$aiu_any" -eq 4 && "$aiu_plain" -eq 4 ]] ||
    fail "AIU plain-b32 lowering changed: exact=$aiu_plain all-aiu=$aiu_any want=4"
  [[ "$reader_any" -eq 1 && "$reader_x4" -eq 1 ]] ||
    fail "Universal lowering changed: b32x4=$reader_x4 all-tsm=$reader_any want=1"
  [[ "$converter_lop" -eq 16 && "$converter_add" -eq 8 &&
      "$converter_fma" -eq 8 ]] ||
    fail "int4 fast converter lowering changed: lop=$converter_lop add=$converter_add fma=$converter_fma"
  if grep -E 'tsm\.ld\.swzl|vmem\.aiu\.ld\.tsm\.[^[:space:]]*\.s1\.' \
       "$isa" >/dev/null; then
    fail 'swizzled delivery appeared in the direct numeric kernel'
  fi
  grep -F 'vmem.acp.commit.grp' "$isa" >/dev/null ||
    fail 'AIU publication commit is absent'
  grep -F 's.wait' "$isa" | grep -F 'commit_group(0)' >/dev/null ||
    fail 'AIU publication wait is absent'
  grep -F 's.blksyn' "$isa" >/dev/null ||
    fail 'CTA publication edge is absent'

  source_sha="$(git -C "$root" rev-parse HEAD)"
  actlize_sha="$(git -C "$root" ls-tree HEAD third_party/actlize | awk '{print $3}')"
  cutlass_sha="$(git -C "$root" ls-tree HEAD third_party/cutlass | awk '{print $3}')"
  {
    printf 'source_sha=%s\n' "$source_sha"
    printf 'submodule.third_party/actlize=%s\n' "$actlize_sha"
    printf 'submodule.third_party/cutlass=%s\n' "$cutlass_sha"
    printf 'sdk=%s\n' "$sdk"
    printf 'sdk_release=%s\n' "$sdk_release"
    printf 'sdk_archive_sha256=%s\n' "$sdk_archive_sha"
    printf 'compiler=%s\n' "$compiler_identity"
    printf 'arch=ppu0010\n'
    printf 'target=%s\n' "$target"
    printf 'ppu_defs=%s\n' "$defs"
    printf 'binary=%s\n' "$binary"
    printf 'binary_size=%s\n' "$(stat -c '%s' "$binary")"
    printf 'binary_sha256=%s\n' "$(sha256sum "$binary" | awk '{print $1}')"
  } >"$out/handoff.env"
  sha256sum "$binary" "$source" "$out/build.log" "$list_elf" "$isa" \
    >"$out/authority.sha256"
  printf 'L254_Q4_N16K64_FRAGMENT_BUILD PASS binary=%s sha256=%s kernels=1 aiu_plain_b32=4 universal_tsm_b32x4=1 converter=16/8/8\n' \
    "$binary" "$(sha256sum "$binary" | awk '{print $1}')" |
    tee "$out/build-verdict.log"
  if [[ "${QUACTLIZE_L254_BUILD_ONLY:-0}" == 1 ]]; then
    printf '[l254:q4-n16k64-fragment] BUILD_ONLY artifacts=%s\n' "$out"
    return 0
  fi

  "$binary" | tee "$out/run.log"
  grep -E '^FQ_Q4_N16K64_FRAGMENT_NUMERIC verdict=PASS .*codes=4096 offline_bad=0 raw_bad=0 sentinel=0 .*launch=\[before:0,immediate:0,sync:0,copy:0\] plant=none$' \
    "$out/run.log" >/dev/null || fail 'positive numeric verdict is absent'

  local plant offline_bad
  while read -r plant offline_bad; do
    if "$binary" "--plant-$plant" >"$out/red-$plant.log" 2>&1; then
      fail "$plant wrong-map unexpectedly passed"
    fi
    grep -E "^FQ_Q4_N16K64_FRAGMENT_NUMERIC verdict=FAIL .*offline_bad=$offline_bad raw_bad=$offline_bad sentinel=0 .*plant=$plant$" \
      "$out/red-$plant.log" >/dev/null ||
      fail "$plant wrong-map did not produce the exact RED verdict"
    printf '[l254:q4-n16k64-fragment:red] PASS plant=%s result=RED\n' "$plant"
  done <<'EOF'
layout1 3840
deepgemm 3840
EOF
  printf 'L254_Q4_N16K64_FRAGMENT_NUMERIC PASS mapping_id=0x51344e3136440001 codes=4096 raw_bad=0 sentinel=0 reds=2\n' |
    tee "$out/verdict.log"
  printf '[l254:q4-n16k64-fragment] artifacts=%s\n' "$out"
}

main "$@"
