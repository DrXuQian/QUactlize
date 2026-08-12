#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
NVCC=${NVCC:-nvcc}
CXX=${CXX:-c++}
BUILD=$(mktemp -d /tmp/quactlize-q36-zero.XXXXXX)
trap 'rm -rf -- "$BUILD"' EXIT

# THIS ORACLE CALLS THE REAL TORCH PREPROCESSING OPS.  Rebuilding the small
# device-library stand-in below is not enough if quactlize/_C itself predates
# gguf_prepass_ops.cpp or one of the headers compiled into it.  A focused
# `local_gates -k` invocation does not pass through pytest's session-level
# freshness hook, so enforce the same precondition here rather than reporting
# on code nobody is running.
python3 - "$ROOT" <<'PY'
from pathlib import Path
import os
import sys

root = Path(sys.argv[1])
shared = sorted((root / "quactlize").glob("_C*.so"))
if not shared:
    raise SystemExit(
        "L139 FAIL: quactlize/_C is absent; build the host extension with "
        "`python3 setup.py build_ext --inplace`")
built = shared[0].stat().st_mtime
if os.environ.get("L139_PLANT_STALE_EXTENSION"):
    built = float("-inf")  # contract negative: every real input must be reported newer
inputs = [
    root / "quactlize/csrc/preprocess/cutlass_kernels/cutlass_preprocessors.cpp",
    root / "quactlize/csrc/preprocess/thop/weight_preprocess_ops.cpp",
    root / "quactlize/csrc/preprocess/thop/gguf_prepass_ops.cpp",
    root / "quactlize/csrc/preprocess/thop/ppu_backend.cpp",
]
for directory in (root / "quactlize/csrc/preprocess", root / "quactlize/include"):
    inputs.extend(p for p in directory.rglob("*") if p.suffix in (".h", ".hpp", ".cuh"))
newer = sorted(p.relative_to(root).as_posix() for p in inputs if p.is_file() and p.stat().st_mtime > built)
if newer:
    raise SystemExit(
        "L139 FAIL: quactlize/_C is older than its source inputs: "
        + ", ".join(newer[:8])
        + (" ..." if len(newer) > 8 else "")
        + "; rebuild with `python3 setup.py build_ext --inplace`")
PY

# These translation units expose host layout/packing entry points and launch no
# CUDA kernel.  Compile their dormant device bodies for a conservative baseline
# architecture instead of querying a GPU: the local proof needs nvcc, not a
# runtime device.  An override exists only for compiler bring-up diagnostics.
ARCH=${L139_CUDA_ARCH:-sm_80}

COMMON=(
  "$NVCC" -std=c++17 -O2 "-arch=$ARCH" --expt-relaxed-constexpr -Xcompiler=-fPIC
  "-I$ROOT/quactlize/include"
  "-I$ROOT/dev/fold_derivation/stub_inc"
  "-I$ROOT/third_party/actlize/include"
  "-I$ROOT/third_party/cutlass/include"
)

"${COMMON[@]}" -c -o "$BUILD/layout.o" "$ROOT/quactlize/csrc/device/ppu_dense_layout.cu"
"${COMMON[@]}" -x cu -c -o "$BUILD/units.o" "$ROOT/quactlize/csrc/device/ppu_unit_pack.cpp"
"$CXX" -std=c++17 -O2 -fPIC -c -o "$BUILD/stub.o" \
  "$ROOT/dev/fold_derivation/q36_ppu_loader_stub.cpp"
"$NVCC" -shared "-arch=$ARCH" -o "$BUILD/libquactlize_ppu.so" \
  "$BUILD/layout.o" "$BUILD/units.o" "$BUILD/stub.o"

cd "$ROOT"
QUACTLIZE_PPU_LIB="$BUILD/libquactlize_ppu.so" \
  python3 dev/fold_derivation/q36_zero_redundancy.py "$@"
