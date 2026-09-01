#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[l247:q4-n16k64-codegen] FAIL: %s\n' "$*" >&2
  exit 1
}

main() {
  local root sdk_root hgcc hgobjdump out build target object
  local compiler_identity objdump_identity list_elf isa
  local kernel_count symbol aiu_any aiu_plain reader_any reader_x4

  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  sdk_root="${PPU_SDK:-${PPU_HOME:-${PPU_SDK_SITE_DEFAULT:-}}}"
  [[ -n "$sdk_root" ]] || fail \
    'set PPU_SDK to a real PPU SDK; this is a genuine HGCC codegen gate'
  hgcc="$sdk_root/bin/hgcc"
  hgobjdump="$sdk_root/bin/hgobjdump"
  [[ -x "$hgcc" && -x "$hgobjdump" ]] || fail \
    "PPU_SDK must own executable bin/hgcc and bin/hgobjdump: $sdk_root"

  compiler_identity="$($hgcc --version 2>&1 | head -n 1 || true)"
  objdump_identity="$($hgobjdump --version 2>&1 | head -n 1 || true)"
  [[ -n "$compiler_identity" && "$compiler_identity" != *stub* ]] || fail \
    "hgcc identity is empty or a stub: ${compiler_identity:-<empty>}"
  [[ -n "$objdump_identity" && "$objdump_identity" != *stub* ]] || fail \
    "hgobjdump identity is empty or a stub: ${objdump_identity:-<empty>}"

  out="${QUACTLIZE_L247_OUT:-/tmp/quactlize-l247-q4-n16k64-delivery-codegen}"
  build="$out/build"
  target=test_q4_n16k64_delivery_codegen
  mkdir -p "$out"

  # build.sh configures the production PPU toolchain, while this custom target
  # stops at hgcc -c.  No host executable link and no device launch are part of
  # this closure.
  PPU_SDK="$sdk_root" \
  PPU_ARCHS=ppu0010 \
  PPU_BUILD_DIR="$build" \
  PPU_BUILD_RESUME=0 \
  TARGET="$target" \
  JOBS=1 \
    "$root/build.sh" >"$out/build.log" 2>&1 || {
      tail -n 160 "$out/build.log" >&2
      fail 'compile-only CMake target did not build'
    }

  mapfile -t objects < <(
    find "$build" -type f \
      -name 'test_q4_n16k64_delivery_codegen_*.o' -print | sort
  )
  [[ ${#objects[@]} -eq 1 ]] || fail \
    "expected exactly one HGCC object, found ${#objects[@]}"
  object="${objects[0]}"
  if find "$build" -type f -name "$target" -perm -u+x -print -quit |
       grep -q .; then
    fail 'compile-only target unexpectedly produced a host executable'
  fi

  list_elf="$out/list-elf.txt"
  isa="$out/isa.txt"
  "$hgobjdump" --list-elf "$object" >"$list_elf" 2>"$out/list-elf.err" ||
    fail 'hgobjdump could not parse the HGCC object'
  "$hgobjdump" --dump-isa "$object" >"$isa" 2>"$out/isa.err" ||
    fail 'hgobjdump could not disassemble the HGCC object'

  kernel_count="$(grep -c '^Func [0-9][0-9]*:' "$list_elf" || true)"
  [[ "$kernel_count" -eq 1 ]] || fail \
    "expected one device kernel in the object, found $kernel_count"
  symbol="$(awk '/^Func [0-9][0-9]*:/ {print $3}' "$list_elf")"
  [[ "$symbol" == *q4_n16k64_delivery_codegen_kernel* ]] || fail \
    "the sole device kernel is not the delivery gate: $symbol"
  grep -F "Disassembly of section .text.kernel.$symbol" "$isa" >/dev/null ||
    fail 'ISA is not bound to the exact listed kernel symbol'

  # TN64 contains four N16 writer cubes.  The Universal reader moves the
  # 8192-byte stage as 32 lanes x 16 uint128 loads.  Require both exact backend
  # forms and reject any alternative AIU/TSM load form, so a fallback or a
  # return to swizzled delivery cannot look green.
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
    fail 'swizzled load/store form appeared in the plain-delivery gate'
  fi
  grep -F 'vmem.acp.commit.grp' "$isa" >/dev/null ||
    fail 'AIU publication commit is absent'
  grep -F 's.wait' "$isa" | grep -F 'commit_group(0)' >/dev/null ||
    fail 'AIU publication wait is absent'
  grep -F 's.blksyn' "$isa" >/dev/null ||
    fail 'CTA publication edge is absent'
  grep -F 'vmem.st.b32' "$isa" >/dev/null ||
    fail 'reader result is not observable in global memory'

  sha256sum "$hgcc" "$hgobjdump" "$object" \
    "$root/dev/test_q4_n16k64_delivery_codegen.cu" \
    >"$out/authority.sha256"
  printf 'L247_Q4_N16K64_DELIVERY_CODEGEN PASS kernels=1 aiu_plain_b32=%s universal_tsm_b32x4=%s linked=0 launched=0\n' \
    "$aiu_plain" "$reader_x4" | tee "$out/verdict.log"
  printf '[l247:q4-n16k64-codegen] artifacts=%s\n' "$out"
}

main "$@"
