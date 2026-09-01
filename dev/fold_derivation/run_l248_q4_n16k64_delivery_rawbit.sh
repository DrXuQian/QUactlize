#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[l248:q4-n16k64-rawbit] FAIL: %s\n' "$*" >&2
  exit 1
}

main() {
  local root sdk_root sdk_archive sdk_archive_sha sdk_release hgcc hgobjdump
  local out build target source binary defs source_sha actlize_sha cutlass_sha
  local compiler_identity sdk_identity
  local list_elf isa symbol kernel_count aiu_any aiu_plain reader_any reader_x4

  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  sdk_root="${PPU_SDK:-${PPU_HOME:-${PPU_SDK_SITE_DEFAULT:-}}}"
  [[ -n "$sdk_root" ]] || fail 'set PPU_SDK to a real PPU SDK'
  hgcc="$sdk_root/bin/hgcc"
  hgobjdump="$sdk_root/bin/hgobjdump"
  [[ -x "$hgcc" && -x "$hgobjdump" ]] || fail \
    "PPU_SDK must own executable bin/hgcc and bin/hgobjdump: $sdk_root"
  sdk_archive="${PPU_SDK_ARCHIVE:-}"
  [[ -n "$sdk_archive" && "$sdk_archive" = /* && -f "$sdk_archive" && ! -L "$sdk_archive" ]] ||
    fail 'set PPU_SDK_ARCHIVE to the absolute regular pinned SDK archive'
  sdk_archive_sha="$(sha256sum "$sdk_archive" | awk '{print $1}')"
  [[ "$sdk_archive_sha" == '63ca196b152f2fec667fce8b18c04f1d6d0fa9e7bc7f72e18f017c96d11731dd' ]] ||
    fail "PPU_SDK_ARCHIVE digest is not admitted: $sdk_archive_sha"
  [[ -f "$sdk_root/release.yaml" && ! -L "$sdk_root/release.yaml" ]] ||
    fail 'installed SDK has no regular release.yaml receipt'
  sdk_release="$(sed -n 's/^version:[[:space:]]*//p' "$sdk_root/release.yaml")"
  [[ "$sdk_release" == '2.1.1-a5c56e' ]] ||
    fail "installed SDK release is not admitted: ${sdk_release:-missing}"

  compiler_identity="$($hgcc --version 2>&1 | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
  [[ -n "$compiler_identity" && "$compiler_identity" != *stub* ]] || fail \
    "hgcc identity is empty or a stub: ${compiler_identity:-<empty>}"
  [[ "$compiler_identity" == *"Release version $sdk_release"* ]] ||
    fail 'hgcc identity disagrees with installed SDK release receipt'
  sdk_identity="$(realpath "$sdk_root")"

  git -C "$root" diff --quiet --ignore-submodules=none HEAD -- ||
    fail 'tracked source or submodule state is dirty; commit the exact candidate first'
  if git -C "$root" submodule status --recursive | grep -Eq '^[+U-]'; then
    fail 'submodules are not at the exact recorded commits'
  fi
  while IFS= read -r line; do
    [[ "$line" == '?? '* ]] || continue
    case "${line#?? }" in
      quactlize/*|third_party/*|cmake/*|CMakeLists.txt|build.sh|dev/test_q4_n16k64_delivery_rawbit.cu)
        fail "untracked build input is not allowed: ${line#?? }" ;;
    esac
  done < <(git -C "$root" status --porcelain=v1 --untracked-files=all)

  out="${QUACTLIZE_L248_OUT:-/tmp/quactlize-l248-q4-n16k64-delivery-rawbit}"
  build="$out/build"
  target=test_q4_n16k64_delivery_rawbit
  source="$root/dev/test_q4_n16k64_delivery_rawbit.cu"
  defs='PPU_PACKED_SCALE=1 PPU_PACKED_FORMAT=0 QUACTLIZE_DENSE_ONLY=12'
  mkdir -p "$out"

  for token in \
    'q4_n16k64_delivery_rawbit_kernel' \
    'AiuPlainProvider<kTileN, kTileK>' \
    'Q4N16K64UniversalReader<kTileN, kWarpN, kTileK>' \
    'UINT64_C(0x51344e3136440001)' \
    'hggcGetLastError()' \
    'hggcDeviceSynchronize()' \
    'source_word(lane, value, n_cohort, k_block)'; do
    grep -F "$token" "$source" >/dev/null || fail \
      "raw-bit source lost required seam: $token"
  done
  printf '[l248:q4-n16k64-rawbit:source] PASS exact writer/reader, independent oracle, poison and launch audit\n'

  PPU_SDK="$sdk_root" \
  PPU_ARCHS=ppu0010 \
  PPU_BUILD_DIR="$build" \
  PPU_BUILD_RESUME=0 \
  PPU_DEFS="$defs" \
  TARGET="$target" \
  JOBS="${JOBS:-1}" \
    "$root/build.sh" >"$out/build.log" 2>&1 || {
      tail -n 180 "$out/build.log" >&2
      fail 'runnable PPU target did not build'
    }

  mapfile -t binaries < <(
    find "$build" -type f -name "$target" -perm -u+x -print | sort
  )
  [[ ${#binaries[@]} -eq 1 ]] || fail \
    "expected exactly one executable $target, found ${#binaries[@]}"
  binary="${binaries[0]}"

  list_elf="$out/list-elf.txt"
  isa="$out/isa.txt"
  "$hgobjdump" --list-elf "$binary" >"$list_elf" 2>"$out/list-elf.err" ||
    fail 'hgobjdump could not parse the runnable binary'
  "$hgobjdump" --dump-isa "$binary" >"$isa" 2>"$out/isa.err" ||
    fail 'hgobjdump could not disassemble the runnable binary'
  kernel_count="$(grep -c '^Func [0-9][0-9]*:' "$list_elf" || true)"
  [[ "$kernel_count" -eq 1 ]] || fail \
    "expected one device kernel in the binary, found $kernel_count"
  symbol="$(awk '/^Func [0-9][0-9]*:/ {print $3}' "$list_elf")"
  [[ "$symbol" == *q4_n16k64_delivery_rawbit_kernel* ]] || fail \
    "the sole device kernel is not the raw-bit probe: $symbol"
  grep -F "Disassembly of section .text.kernel.$symbol" "$isa" >/dev/null ||
    fail 'ISA is not bound to the exact listed kernel symbol'

  aiu_any="$(grep -c 'vmem\.aiu\.ld\.tsm' "$isa" || true)"
  aiu_plain="$(grep -c \
    'vmem\.aiu\.ld\.tsm\.l1\.t0\.p0\.s0\.m0\.2d\.b32\.kp1' \
    "$isa" || true)"
  reader_any="$(grep -c 'tsm\.ld\.' "$isa" || true)"
  reader_x4="$(grep -c 'tsm\.ld\.b32x4' "$isa" || true)"
  [[ "$aiu_any" -eq 4 && "$aiu_plain" -eq 4 ]] || fail \
    "AIU plain-b32 lowering changed: exact=$aiu_plain all-aiu=$aiu_any want=4"
  [[ "$reader_any" -eq 16 && "$reader_x4" -eq 16 ]] || fail \
    "Universal uint128 lowering changed: b32x4=$reader_x4 all-tsm=$reader_any want=16"
  if grep -E 'tsm\.ld\.swzl|vmem\.aiu\.ld\.tsm\.[^[:space:]]*\.s1\.' \
       "$isa" >/dev/null; then
    fail 'swizzled load/store form appeared in the plain-delivery probe'
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
    printf 'sdk=%s\n' "$sdk_identity"
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
  sha256sum "$binary" "$source" "$out/build.log" "$out/list-elf.txt" \
    "$out/isa.txt" >"$out/authority.sha256"

  printf 'L248_Q4_N16K64_DELIVERY_BUILD PASS binary=%s sha256=%s kernels=1 aiu_plain_b32=4 universal_tsm_b32x4=16\n' \
    "$binary" "$(sha256sum "$binary" | awk '{print $1}')" |
    tee "$out/build-verdict.log"
  if [[ "${QUACTLIZE_L248_BUILD_ONLY:-0}" == 1 ]]; then
    printf '[l248:q4-n16k64-rawbit] BUILD_ONLY artifacts=%s\n' "$out"
    return 0
  fi

  "$binary" | tee "$out/run.log"
  grep -E '^FQ_Q4_N16K64_DELIVERY_RAWBIT verdict=PASS .*layout=3 .*mapping_id=0x51344e3136440001 .*words=2048 raw_bad=0 sentinel=0 .*launch=\[before:0,immediate:0,sync:0,copy:0\] plant=none$' \
    "$out/run.log" >/dev/null || fail 'positive raw-bit verdict is absent'

  if "$binary" --plant-wrong-oracle >"$out/red.log" 2>&1; then
    fail 'wrong-oracle negative unexpectedly returned success'
  fi
  grep -E '^FQ_Q4_N16K64_DELIVERY_RAWBIT verdict=FAIL .*layout=3 .*mapping_id=0x51344e3136440001 .*raw_bad=[1-9][0-9]* .*plant=wrong-oracle$' \
    "$out/red.log" >/dev/null || fail 'wrong-oracle negative did not turn red'
  printf '[l248:q4-n16k64-rawbit:red] PASS plant=wrong-oracle result=RED\n'
  printf 'L248_Q4_N16K64_DELIVERY_RAWBIT PASS mapping_id=0x51344e3136440001 words=2048 raw_bad=0 sentinel=0 launch=0/0/0/0 reds=1\n' |
    tee "$out/verdict.log"
  printf '[l248:q4-n16k64-rawbit] artifacts=%s\n' "$out"
}

main "$@"
