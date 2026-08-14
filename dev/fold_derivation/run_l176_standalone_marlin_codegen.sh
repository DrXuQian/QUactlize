#!/usr/bin/env bash
# Source/codegen boundary for the exact shipping standalone Marlin unit.
#
# Local mode proves the generated route and source contracts and then reports
# PPU codegen as SKIP if the real SDK is absent.  ppu10 mode builds the real
# CMake-generated unit, selects exactly one MarlinKernelPPU device symbol and
# records its hgobjdump line/resource evidence.  It never opens a device.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MODE=local
OUTPUT=""

usage() {
  cat <<'EOF'
usage: run_l176_standalone_marlin_codegen.sh [--target local|ppu10] [--output DIR]

local: prove the shipping generated route/source contracts.  If a real PPU
       compiler/disassembler is unavailable, print an explicit PPU SKIP.
ppu10: compile and disassemble the exact test_lowbit_dense_marlin_wk4_ab
       generated unit.  This is compile-only; no PPU device is opened.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; MODE="$2"; shift 2 ;;
    --output) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; OUTPUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[l176] FAIL: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done
case "$MODE" in local|ppu10) ;; *) echo "[l176] FAIL: target must be local or ppu10" >&2; exit 2 ;; esac

SOURCE="$HERE/l176_standalone_marlin_codegen.py"
REPORT="$HERE/l176_standalone_marlin_codegen_report.py"
RUN169="$HERE/run_l169_standalone_marlin_unit.sh"
RUN174="$HERE/run_l174_marlin_compute_contract.sh"
RUN175="$HERE/run_l175_native_fragment_contract.sh"
COMMITTED_CHECK="$ROOT/ci/check_l143_wk4_committed_evidence.py"
COMMITTED_EVIDENCE_REL=dev/fold_derivation/l143_standalone_marlin.expected.txt
for path in "$SOURCE" "$REPORT" "$RUN169" "$RUN174" "$RUN175" "$COMMITTED_CHECK"; do
  [[ -f "$path" ]] || { echo "[l176] FAIL: missing $path" >&2; exit 1; }
done

tmp="${QUACTLIZE_L176_WORK:-/workspace/quactlize-l176-local}"
mkdir -p "$tmp"

python3 "$SOURCE" --output "$tmp/source.json"
for plant in \
  generated-row generic-wrapper runtime-nblock flat-accumulator missing-lineinfo \
  m8-x4-fallback m8-discarded-destinations m8-padded-a m8-broadens-m; do
  set +e
  python3 "$SOURCE" --plant "$plant" >"$tmp/$plant.log" 2>&1
  rc=$?
  set -e
  [[ $rc -ne 0 ]] || { echo "[l176] FAIL: plant $plant escaped" >&2; exit 1; }
  grep -Fq "[l176:red] plant=$plant" "$tmp/$plant.log" || {
    echo "[l176] FAIL: plant $plant failed for an unrelated reason" >&2
    sed -n '1,20p' "$tmp/$plant.log" >&2
    exit 1
  }
done

if [[ "$MODE" == local ]]; then
  bash "$RUN169"
  bash "$RUN174"
  bash "$RUN175"
  echo '[l176:local] PASS: exact generated route/source hashes/cadence/native m16+m8 fragments/packed-M1 ledger/lineinfo; 9/9 causal controls RED'
else
  # L169/L174/L175 are local compile-time facts.  The PPU box's nvcc delegates
  # device preprocessing to ppu_clang++ and cannot compile their NVIDIA/stub
  # fixture (hggc_fp8.h/GCC13 are known incompatible seams).  Consume the
  # exact evidence committed at the result SHA; do not turn a non-executable
  # host oracle into a fresh box PASS.  The real generated unit is still built,
  # linked and disassembled below with hgcc/hgobjdump.
  committed="$tmp/l143-committed-evidence.txt"
  git -C "$ROOT" show "HEAD:$COMMITTED_EVIDENCE_REL" >"$committed" || {
    echo '[l176:ppu] FAIL: result SHA lacks committed standalone admission evidence' >&2
    exit 1
  }
  python3 "$COMMITTED_CHECK" --committed-only --evidence "$committed"
  echo "[l176:ppu-admission] PASS: local compile oracles consumed from result SHA $(git -C "$ROOT" rev-parse HEAD); fresh-box-execution=0"
fi

resolve_executable() {
  local candidate="$1"
  [[ -n "$candidate" ]] || return 1
  if [[ "$candidate" == */* ]]; then
    [[ -x "$candidate" ]] && { readlink -f "$candidate"; return 0; }
  else
    command -v "$candidate" 2>/dev/null && return 0
  fi
  return 1
}

sdk_root="${PPU_SDK:-${PPU_HOME:-${PPU_SDK_SITE_DEFAULT:-}}}"
hgcc=""
if [[ -n "${HGCC:-}" ]]; then
  hgcc="$(resolve_executable "$HGCC" || true)"
elif [[ -n "$sdk_root" ]]; then
  hgcc="$(resolve_executable "$sdk_root/bin/hgcc" || true)"
else
  hgcc="$(resolve_executable "$(command -v hgcc 2>/dev/null || true)" || true)"
fi
if [[ -n "$hgcc" && -z "$sdk_root" ]]; then
  sdk_root="$(cd "$(dirname "$hgcc")/.." && pwd)"
fi
hgobjdump=""
if [[ -n "${HGOBJDUMP:-}" ]]; then
  hgobjdump="$(resolve_executable "$HGOBJDUMP" || true)"
elif [[ -n "$sdk_root" ]]; then
  hgobjdump="$(resolve_executable "$sdk_root/bin/hgobjdump" || true)"
fi
[[ -n "$hgobjdump" ]] || hgobjdump="$(resolve_executable "$(command -v hgobjdump 2>/dev/null || true)" || true)"

compiler_identity=""
if [[ -n "$hgcc" ]]; then compiler_identity="$($hgcc --version 2>&1 | head -n 1 || true)"; fi
if [[ -z "$hgcc" || -z "$hgobjdump" || -z "$compiler_identity" || "$compiler_identity" == *stub* ]]; then
  if [[ "$MODE" == ppu10 ]]; then
    echo "[l176:ppu] SKIP: real hgcc + hgobjdump unavailable (hgcc=${hgcc:-<none>} identity=${compiler_identity:-<none>} hgobjdump=${hgobjdump:-<none>})"
    echo 'L176 box command: PPU_SDK=/path/to/sdk bash dev/fold_derivation/run_l176_standalone_marlin_codegen.sh --target ppu10 --output /workspace/quactlize-l176-ppu'
    exit 3
  fi
  echo "[l176:ppu] SKIP: real hgcc + hgobjdump unavailable; no PPU opcode or backend-spill claim was made"
  echo 'L176 box command: PPU_SDK=/path/to/sdk bash dev/fold_derivation/run_l176_standalone_marlin_codegen.sh --target ppu10 --output /workspace/quactlize-l176-ppu'
  exit 0
fi

if [[ "$MODE" == local ]]; then
  echo "[l176:ppu] AVAILABLE: $compiler_identity; rerun with --target ppu10 for the compile-only postcondition"
  exit 0
fi

sdk_root="$(cd "$sdk_root" && pwd)"
sdk_hgcc="$(resolve_executable "$sdk_root/bin/hgcc" || true)"
[[ -n "$sdk_hgcc" && "$sdk_hgcc" == "$hgcc" ]] || {
  echo "[l176:ppu] FAIL: hgcc '$hgcc' is not owned by SDK '$sdk_root'" >&2
  exit 1
}
sdk_hgobjdump="$(resolve_executable "$sdk_root/bin/hgobjdump" || true)"
[[ -n "$sdk_hgobjdump" && "$sdk_hgobjdump" == "$hgobjdump" ]] || {
  echo "[l176:ppu] FAIL: hgobjdump '$hgobjdump' is not owned by SDK '$sdk_root'" >&2
  exit 1
}
hgobjdump_identity="$($hgobjdump --version 2>&1 | head -n 1 || true)"
[[ -n "$hgobjdump_identity" && "$hgobjdump_identity" != *stub* ]] || {
  echo "[l176:ppu] FAIL: hgobjdump identity is empty or a stub: ${hgobjdump_identity:-<none>}" >&2
  exit 1
}
hgcc_sha256="$(sha256sum "$hgcc" | awk '{print $1}')"
hgobjdump_sha256="$(sha256sum "$hgobjdump" | awk '{print $1}')"
[[ "$hgcc_sha256" =~ ^[0-9a-f]{64}$ && "$hgobjdump_sha256" =~ ^[0-9a-f]{64}$ ]] || {
  echo '[l176:ppu] FAIL: SDK tool hashes are not canonical SHA-256 values' >&2
  exit 1
}
help_text="$($hgobjdump -h 2>&1 || true)"
grep -Eq -- '(^|[[:space:],])-isa([,[:space:]]|$)|--dump-isa' <<<"$help_text" || {
  echo "[l176:ppu] SKIP: $hgobjdump has no ISA disassembly mode"; exit 3;
}

if [[ -z "$OUTPUT" ]]; then
  OUTPUT="/workspace/quactlize-l176-ppu"
  mkdir -p "$OUTPUT"
else
  OUTPUT="$(readlink -m "$OUTPUT")"
  if [[ -e "$OUTPUT" && ! -d "$OUTPUT" ]]; then
    echo "[l176:ppu] FAIL: output is not a directory: $OUTPUT" >&2; exit 1
  fi
  if [[ -e "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[l176:ppu] FAIL: output directory is not empty: $OUTPUT" >&2; exit 1
  fi
  mkdir -p "$OUTPUT"
fi
build="$OUTPUT/build"
mkdir -p "$build"

# Every file that can change this one generated symbol must be committed.  An
# unrelated dirty benchmark/document does not invalidate the file-hash
# authority, but a dirty target source cannot be attributed to HEAD.
mapfile -t source_paths < <(python3 - "$tmp/source.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
print(*data["source_files"].keys(), sep="\n")
PY
)
if ! dirty="$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all -- "${source_paths[@]}")"; then
  echo '[l176:ppu] FAIL: git could not establish source-authority cleanliness' >&2
  exit 1
fi
if [[ -n "$dirty" ]]; then
  echo '[l176:ppu] FAIL: a source that owns the exact symbol differs from HEAD:' >&2
  printf '%s\n' "$dirty" >&2
  exit 1
fi

# This is an exact codegen probe, not a tuning entry point.  Neutralize every
# public build override that can add flags or select a second configuration;
# otherwise PPU_DEFS/CFLAGS from an operator shell can produce a binary that
# still contains all required tokens but is not the result-SHA target.
env -u CMAKE_GENERATOR -u CMAKE_TOOLCHAIN_FILE -u CC -u CXX \
  PPU_SDK="$sdk_root" PPU_HOME= PPU_SDK_SITE_DEFAULT= \
  PPU_BUILD_DIR="$build" PPU_ARCHS=ppu0010 \
  TARGET=test_lowbit_dense_marlin_wk4_ab JOBS=1 \
  PPU_DEFS= PPU_EXTRA_DEFS= CFLAGS= CXXFLAGS= CPPFLAGS= LDFLAGS= \
  TILE_M=16 TILE_N=128 WARP_M=16 WARP_N=64 STAGES=4 \
  QUANT=int4 TSK=64 BENCH_GS=128 LOWBIT_DENSE_CONFIGS_PER_UNIT=1 \
  MOE_FORMATS= MOE_TM_LIST= MOE_TN_LIST= MOE_WM_LIST= MOE_STAGES= \
  MOE_CORES= GEMV_GROUPS= "$ROOT/build.sh" \
  >"$OUTPUT/build.log" 2>&1 || {
    first="$(grep -m1 -E 'fatal error:|error:|CMake Error|\[FAIL\]' "$OUTPUT/build.log" || tail -n 1 "$OUTPUT/build.log")"
    echo "[l176:ppu] FAIL: exact shipping target did not build: $first" >&2
    echo "[l176:ppu] artifacts: $OUTPUT" >&2
    exit 1
  }
bin="$(find "$build" -type f -name test_lowbit_dense_marlin_wk4_ab -perm -u+x -print -quit)"
[[ -n "$bin" ]] || { echo '[l176:ppu] FAIL: linked target absent' >&2; exit 1; }
generated="$(find "$build" -type f -name lowbit_dense_marlin_wk4_ab_unit.cu -print -quit)"
[[ -n "$generated" ]] || { echo '[l176:ppu] FAIL: generated unit absent' >&2; exit 1; }
python3 "$SOURCE" --generated-unit "$generated" --output "$OUTPUT/source.before.json"

build_make="$(find "$build" -path '*test_lowbit_dense_marlin_wk4_ab.dir/build.make' -print -quit)"
[[ -n "$build_make" ]] || { echo '[l176:ppu] FAIL: target build.make absent' >&2; exit 1; }
python3 - "$build_make" "$generated" "$hgcc" >"$OUTPUT/hgcc-command.txt" <<'PY'
import shlex, sys
from pathlib import Path
build_make, generated, hgcc = map(Path, sys.argv[1:])
lines = [x.strip() for x in build_make.read_text(errors="replace").splitlines()
         if str(generated) in x and str(hgcc) in x]
if len(lines) != 1:
    raise SystemExit(f"L176 command audit: expected one hgcc command, got {len(lines)}")
tokens = shlex.split(lines[0])
arches = [x for x in tokens if x.startswith("-arch=")]
if arches != ["-arch=ppu_10"]:
    raise SystemExit(f"L176 command audit: arch flags are {arches}")
required = {
    "-DDENSE_MARLIN_WK4_AB=1", "-DDENSE_MARLIN_AB=1",
    "-DDENSE_STREAMK_AB=1", "-DBENCH_GS=128", "-DBENCH_TSK=64",
    "-DDENSE_AB_BITS=4", "-DDENSE_AB_ARTIFACT_TK=64",
    "-DDENSE_AB_TM=16", "-DDENSE_AB_TN=128", "-DDENSE_AB_TK=128",
    "-DDENSE_AB_WM=16", "-DDENSE_AB_WN=64", "-DDENSE_AB_WARP_K=32",
    "-DDENSE_AB_ST=4", "-DDENSE_AB_BC=0",
    "-DTILE_M=16", "-DTILE_N=128", "-DWARP_M=16", "-DWARP_N=64",
    "-DSTAGES=4",
}
missing = sorted(required - set(tokens))
if missing:
    raise SystemExit("L176 command audit: missing " + ",".join(missing))
wrong_count = sorted(token for token in required if tokens.count(token) != 1)
if wrong_count:
    raise SystemExit("L176 command audit: exact defines are not unique: " + ",".join(wrong_count))
keys = {token.split("=", 1)[0] for token in required}
conflicts = sorted({token for token in tokens
                    if token.startswith("-D") and token.split("=", 1)[0] in keys
                    and token not in required})
if conflicts:
    raise SystemExit("L176 command audit: conflicting exact defines: " + ",".join(conflicts))
if tokens.count("-lineinfo") != 1:
    raise SystemExit(
        f"L176 command audit: -lineinfo occurs {tokens.count('-lineinfo')} times, expected one"
    )
print(lines[0])
PY

$hgobjdump -lelf "$bin" >"$OUTPUT/list-elf.txt" 2>"$OUTPUT/list-elf.err" || {
  echo '[l176:ppu] FAIL: hgobjdump -lelf failed' >&2; exit 1;
}
mapfile -t all_symbols < <(sed -n 's/^.*Func [0-9][0-9]*:[[:space:]]*\([^[:space:]]*\).*$/\1/p' "$OUTPUT/list-elf.txt")
symbols=()
demangled=()
for symbol in "${all_symbols[@]}"; do
  pretty="$(c++filt "$symbol")"
  if [[ "$pretty" == *device_kernel* && "$pretty" == *MarlinKernelPPU* && "$pretty" == *MarlinCollectivePPU* ]]; then
    symbols+=("$symbol")
    demangled+=("$pretty")
  fi
done
[[ "${#symbols[@]}" -eq 1 ]] || {
  echo "[l176:ppu] FAIL: exact linked binary has ${#symbols[@]} standalone Marlin device symbols, expected 1" >&2
  printf '%s\n' "${demangled[@]}" >&2
  exit 1
}
symbol="${symbols[0]}"
pretty="${demangled[0]}"
printf '%s\n' "$symbol" >"$OUTPUT/kernel-symbol.txt"
printf '%s\n' "$pretty" >"$OUTPUT/kernel-symbol-demangled.txt"

$hgobjdump -line "-func=$symbol" "$bin" >"$OUTPUT/kernel-line.txt" 2>"$OUTPUT/kernel-line.err" || {
  echo '[l176:ppu] FAIL: exact-symbol line disassembly failed' >&2; exit 1;
}
$hgobjdump "-res-usage=$symbol" "$bin" >"$OUTPUT/resource-usage.txt" 2>"$OUTPUT/resource-usage.err" || {
  echo '[l176:ppu] FAIL: exact-symbol resource report failed' >&2; exit 1;
}
$hgobjdump -isa "$bin" >"$OUTPUT/isa.txt" 2>"$OUTPUT/isa.err" || {
  echo '[l176:ppu] FAIL: whole-binary ISA inventory failed' >&2; exit 1;
}

python3 "$REPORT" --line "$OUTPUT/kernel-line.txt" \
  --resource "$OUTPUT/resource-usage.txt" --source-manifest "$OUTPUT/source.before.json" \
  --symbol "$symbol" --demangled "$pretty" --binary "$bin" \
  --json "$OUTPUT/codegen.json" | tee "$OUTPUT/codegen.txt"
python3 "$SOURCE" --generated-unit "$generated" --output "$OUTPUT/source.after.json"
cmp "$OUTPUT/source.before.json" "$OUTPUT/source.after.json" || {
  echo '[l176:ppu] FAIL: source authority changed while compiling/disassembling' >&2; exit 1;
}

{
  echo 'schema=quactlize.l176.bundle.v1'
  echo "git_sha=$(git -C "$ROOT" rev-parse HEAD)"
  echo "binary=$bin"
  echo "binary_sha256=$(sha256sum "$bin" | awk '{print $1}')"
  echo "hgcc=$hgcc"
  echo "hgcc_identity=$compiler_identity"
  echo "hgcc_sha256=$hgcc_sha256"
  echo "hgobjdump=$hgobjdump"
  echo "hgobjdump_identity=$hgobjdump_identity"
  echo "hgobjdump_sha256=$hgobjdump_sha256"
  echo 'arch=-arch=ppu_10'
  echo "symbol=$symbol"
  echo 'executed_device_code=0'
} >"$OUTPUT/manifest.txt"
echo '[l176] PASS: exact shipping generated unit compiled; unique standalone symbol is source/resource/opcode-bound; no device code executed'
echo "[l176] artifacts: $OUTPUT"
