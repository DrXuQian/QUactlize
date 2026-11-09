#!/usr/bin/env bash
# Build the prototype MMA-reader library from the worktree headers (parallel TUs).
#   bash tools/build_mma_proto.sh OUT_SO            (register-direct reader, tools/mma_proto_tu.cu)
#   STAGED=1 bash tools/build_mma_proto.sh OUT_SO   (staged reader, tools/mma_staged_tu.cu)
set -euo pipefail
source /workspace/quactlize-decode-gemv-20261108/env.sh
OUT=${1:?output .so}
S=$QGEMV_SRC
W=$(mktemp -d /dev/shm/qgemv/protobuild.XXXXXX)
TU=mma_proto_tu.cu
[ "${STAGED:-0}" = 1 ] && TU=mma_staged_tu.cu
# Production device flags (quactlize/runtime/compiler.py FLAGS), not a hand-picked subset.
mapfile -t PFLAGS < <(cd "$S" && python3 -c 'from quactlize.runtime.compiler import FLAGS; print("\n".join(FLAGS))')
FLAGS=("${PFLAGS[@]}"
       -I$S/quactlize/execution -I$S/quactlize/include -I$S/third_party/actlize/include
       -I$S/third_party/actlize/tools/util/include)
# name qtype mode compute ; every entry is built for T=2,4,8.
families=("q12_d_f16 12 0 0" "q12_i_bf16 12 2 1" "q13_d_f16 13 0 0" "q13_i_bf16 13 2 1"
          "q14_d_f16 14 0 0" "q8_d_f16 8 0 0")
pids=()
for spec in "${families[@]}"; do
  read -r fam q mode compute <<<"$spec"
  for t in 2 4 8; do
    name=${fam}_t$t
    ( "$PPU_SDK/bin/hgcc" "${FLAGS[@]}" -DQQ=$q -DMODE=$mode -DCOMPUTE=$compute -DTT=$t -DQGEMV_TU=$name \
        -c "$QGEMV_ROOT/tools/$TU" -o "$W/$name.o" > "$W/$name.log" 2>&1 ||
        { echo "FAIL $name"; head -60 "$W/$name.log"; exit 1; } ) &
    pids+=($!)
  done
done
g++ -std=c++17 -O2 -fPIC -I$S/quactlize/execution -I$S/quactlize/include -c "$QGEMV_ROOT/tools/mma_proto_router.cpp" -o "$W/router.o"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail = 0 ] || { echo "build failed ($W)"; exit 1; }
g++ -shared -o "$OUT" "$W"/*.o -L"$PPU_SDK/lib" -lhggc_wrapper -lhggcrt1 -lhggc -lhg_wrapper
"$PPU_SDK/bin/hgobjdump" --dump-resource-usage=all "$OUT" > "${OUT%.so}.resources.log" 2>&1 || true
echo "built $OUT ($W)"
