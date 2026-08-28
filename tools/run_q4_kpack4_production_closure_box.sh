#!/usr/bin/env bash
# Production ABI closure for the one canonical Q4_K K-pack4 artifact.
# Builds the two deliberately separate device libraries, audits their exact
# descriptors/inventories, then runs the independent Python numeric oracle.
set -uo pipefail

main() {
  local root workspace_root sha short stamp out jobs timeout_s
  local packed_log sf_log packed_so sf_so host_log audit_log pytest_log rc
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace_root="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-q4-kpack4-production-${short}-${stamp}-$$}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) printf '[q4-kpack4-production] FAIL: OUT must be a strict /workspace child: %s\n' "$out" >&2; return 2 ;;
  esac
  if [ -e "$out" ]; then
    printf '[q4-kpack4-production] FAIL: refusing to overwrite %s\n' "$out" >&2
    return 2
  fi
  jobs="${JOBS:-16}"
  timeout_s="${BUILD_TIMEOUT:-1200}"
  case "$jobs:$timeout_s" in
    *[!0-9:]*|0:*|*:0) printf '[q4-kpack4-production] FAIL: JOBS/BUILD_TIMEOUT must be positive integers\n' >&2; return 2 ;;
  esac
  mkdir -p "$out/results" "$out/build-packed" "$out/build-scalefirst" || return 2

  for module in torch pytest gguf; do
    python3 -c "import $module" >/dev/null 2>&1 || {
      printf '[q4-kpack4-production] FAIL: python module %s is required\n' "$module" >&2
      return 2
    }
  done

  packed_log="$out/results/build-packed.log"
  timeout "$timeout_s" env \
    PPU_BUILD_DIR="$out/build-packed" PPU_ARCHS=ppu0010 JOBS="$jobs" \
    PPU_DEFS='PPU_PACKED_SCALE=1 PPU_PACKED_FORMAT=0 QUACTLIZE_DENSE_ONLY=12' \
    TARGET=quactlize_ppu "$root/build.sh" >"$packed_log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '[q4-kpack4-production] FAIL: packed-format0 build rc=%d\n' "$rc" >&2
    tail -n 120 "$packed_log" >&2
    printf 'artifacts: %s\n' "$out" >&2
    return "$rc"
  fi
  packed_so="$(grep -m1 '^built: ' "$packed_log" | cut -d' ' -f2-)"
  if [ -z "$packed_so" ] || [ ! -f "$packed_so" ]; then
    printf '[q4-kpack4-production] FAIL: packed build reported no library\n' >&2
    return 2
  fi
  for define in PPU_PACKED_SCALE=1 PPU_PACKED_FORMAT=0 QUACTLIZE_DENSE_ONLY=12; do
    grep -qF "PPU_DEFS verified on quactlize_ppu's compile command: -D$define" "$packed_log" || {
      printf '[q4-kpack4-production] FAIL: packed compile command lacks -D%s\n' "$define" >&2
      return 2
    }
  done

  sf_log="$out/results/build-scalefirst.log"
  timeout "$timeout_s" env \
    PPU_BUILD_DIR="$out/build-scalefirst" PPU_ARCHS=ppu0010 JOBS="$jobs" \
    PPU_DEFS='QUACTLIZE_DENSE_ONLY=12' TARGET=quactlize_ppu \
    "$root/build.sh" >"$sf_log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '[q4-kpack4-production] FAIL: ScaleFirst build rc=%d\n' "$rc" >&2
    tail -n 120 "$sf_log" >&2
    printf 'artifacts: %s\n' "$out" >&2
    return "$rc"
  fi
  sf_so="$(grep -m1 '^built: ' "$sf_log" | cut -d' ' -f2-)"
  if [ -z "$sf_so" ] || [ ! -f "$sf_so" ]; then
    printf '[q4-kpack4-production] FAIL: ScaleFirst build reported no library\n' >&2
    return 2
  fi
  grep -qF "PPU_DEFS verified on quactlize_ppu's compile command: -DQUACTLIZE_DENSE_ONLY=12" "$sf_log" || {
    printf '[q4-kpack4-production] FAIL: ScaleFirst compile command lacks narrowed Q4 build\n' >&2
    return 2
  }
  if grep -Eq -- '(^|[[:space:]])-DPPU_PACKED_SCALE=1([[:space:]]|$)' "$sf_log"; then
    printf '[q4-kpack4-production] FAIL: ScaleFirst library accidentally enabled packed metadata\n' >&2
    return 2
  fi

  host_log="$out/results/build-host-extension.log"
  (cd "$root" && python3 setup.py build_ext --inplace) >"$host_log" 2>&1 || {
    printf '[q4-kpack4-production] FAIL: host extension build failed\n' >&2
    tail -n 80 "$host_log" >&2
    return 2
  }

  audit_log="$out/results/inventory-audit.log"
  PACKED_SO="$packed_so" SF_SO="$sf_so" PYTHONPATH="$root" \
    python3 - <<'PY' | tee "$audit_log"
import ctypes, os

class Arrangement(ctypes.Structure):
    _fields_ = [("version", ctypes.c_int32), ("layout", ctypes.c_int32),
                ("bits", ctypes.c_int32), ("high_bits", ctypes.c_int32),
                ("artifact_tile_k", ctypes.c_int32), ("transport_tile_k", ctypes.c_int32),
                ("group_size", ctypes.c_int32), ("reserved", ctypes.c_int32),
                ("mapping_id", ctypes.c_uint64)]

class ConfigV3(ctypes.Structure):
    _fields_ = [("enable_cuda_kernel", ctypes.c_bool), ("name", ctypes.c_char_p),
                ("tile_m", ctypes.c_int32), ("tile_n", ctypes.c_int32),
                ("tactic_tile_k", ctypes.c_int32), ("artifact_tile_k", ctypes.c_int32),
                ("warp_m", ctypes.c_int32), ("warp_n", ctypes.c_int32),
                ("stages", ctypes.c_int32)]

class ConfigV4(ctypes.Structure):
    _fields_ = ConfigV3._fields_ + [("split_k_slices", ctypes.c_int32)]

packed = ctypes.CDLL(os.environ["PACKED_SO"])
sf = ctypes.CDLL(os.environ["SF_SO"])
arr = Arrangement(2, 1, 4, 0, 0, 64, 32, 0, 0x51344B5034540001)
bad = Arrangement(2, 1, 4, 0, 0, 64, 32, 0, arr.mapping_id ^ 1)

dense = packed.quactlize_ppu_list_valid_dense_fully_quantized_configs_for_arrangement_v2_v4
dense.argtypes = [ctypes.POINTER(ConfigV4), ctypes.c_int32] + [ctypes.c_int] * 5 + [ctypes.POINTER(Arrangement)]
dense.restype = ctypes.c_int32
count = dense(None, 0, 1, 1024, 5120, 32, 12, ctypes.byref(arr))
assert count > 0
rows = (ConfigV4 * count)()
assert dense(rows, count, 1, 1024, 5120, 32, 12, ctypes.byref(arr)) == count
assert any(r.split_k_slices == 4 and r.artifact_tile_k == 0 for r in rows)
assert dense(None, 0, 1, 1024, 5120, 32, 12, ctypes.byref(bad)) == 0

grouped = packed.quactlize_ppu_list_valid_grouped_fully_quantized_configs_for_arrangement_v2
grouped.argtypes = [ctypes.POINTER(ConfigV3), ctypes.c_int32] + [ctypes.c_int] * 7 + [ctypes.POINTER(Arrangement)]
grouped.restype = ctypes.c_int32
gcount = grouped(None, 0, 6, 256, 5120, 32, 4, 3, 12, ctypes.byref(arr))
assert gcount > 0
grows = (ConfigV3 * gcount)()
assert grouped(grows, gcount, 6, 256, 5120, 32, 4, 3, 12, ctypes.byref(arr)) == gcount
assert all(r.tactic_tile_k == 256 and r.artifact_tile_k == 0 for r in grows)
assert grouped(None, 0, 6, 256, 5120, 32, 4, 3, 12, ctypes.byref(bad)) == 0

sfvalid = sf.quactlize_ppu_dense_lowbit_config_valid_for_arrangement_v2
sfvalid.argtypes = [ctypes.c_int] * 5 + [ctypes.POINTER(Arrangement), ctypes.c_char_p]
sfvalid.restype = ctypes.c_int32
assert sfvalid(2048, 1024, 5120, 32, 12, ctypes.byref(arr), None) == 1
assert sfvalid(2048, 1024, 5120, 32, 12, ctypes.byref(bad), None) == 0

for symbol in ("quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v2",
               "quactlize_ppu_grouped_fully_quantized_dev_for_arrangement_v2",
               "quactlize_ppu_dense_lowbit_dev_for_arrangement_v2"):
    assert getattr(packed if "lowbit" not in symbol else sf, symbol, None), symbol
print(f"Q4_KPACK4_PRODUCTION_INVENTORY dense={count} grouped={gcount} "
      "decode_split4=PASS prefill_persistent=PASS bad_mapping=EXPECTED_RED")
PY
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || {
    printf '[q4-kpack4-production] FAIL: inventory audit rc=%d\n' "$rc" >&2
    return "$rc"
  }

  pytest_log="$out/results/pytest-production.log"
  env QUACTLIZE_PPU_LIB="$sf_so" QUACTLIZE_PPU_LIB_FMT0="$packed_so" \
    QUACTLIZE_PACKED_FORMAT=12 PYTHONPATH="$root" \
    python3 -m pytest -q -rs -s -m kpack4_production \
      "$root/tests/test_gguf_routes.py" | tee "$pytest_log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ] || grep -qi 'skipped' "$pytest_log" || \
     ! grep -Eq '(^| )1 passed' "$pytest_log"; then
    printf '[q4-kpack4-production] FAIL: numeric closure did not run to one unskipped pass rc=%d\n' "$rc" >&2
    printf 'artifacts: %s\n' "$out" >&2
    return 1
  fi

  sha256sum "$packed_so" "$sf_so" "$audit_log" "$pytest_log" \
    >"$out/results/authority.sha256" || return 2
  printf '[q4-kpack4-production] PASS sha=%s descriptor=v2/layout1/mapping-0x51344b5034540001 dense=M1,M4,M64,M2048 prefill=PERSISTENT grouped=RAGGED+EMPTY-EXPERT artifacts=%s\n' \
    "$sha" "$out"
}

main "$@"
