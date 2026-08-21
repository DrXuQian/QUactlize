#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L181_OUT:-/workspace/quactlize-l181}"
mkdir -p "${out}"

cxx="${CXX:-g++}"
source_file="${repo}/dev/fold_derivation/l181_standalone_marlin_m8.cpp"
binary="${out}/l181_standalone_marlin_m8"

"${cxx}" -std=c++17 -O2 -Wall -Wextra -Werror \
  -I "${repo}/quactlize/include" \
  -o "${binary}" "${source_file}"

"${binary}" | tee "${out}/positive.log"
grep -Fq 'L181 PASS: cooperative packed-row A + plain-x2 provider + m8 output + 4->2->1 scratch + M-invariant artifacts are closed' \
  "${out}/positive.log"

plants=(
  missing-a-chunk
  duplicate-a-chunk
  drop-b-thread
  nvidia-provider
  shifted-word
  padded-a-rows
  m16-output-values
  skip-second-reduction-step
  m16-scratch
  m-dependent-artifact
)
for plant in "${plants[@]}"; do
  set +e
  "${binary}" --plant="${plant}" >"${out}/${plant}.log" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -ne 2 ]] ||
      ! grep -Fq "L181 EXPECTED-RED plant=${plant}" "${out}/${plant}.log"; then
    cat "${out}/${plant}.log" >&2
    echo "[l181] FAIL: plant ${plant} was not rejected" >&2
    exit 1
  fi
done

# Bind the arithmetic oracle to production instead of accepting a second,
# self-consistent host implementation.  These tokens name the causal seams:
# packed one-row storage/copy, PPU x2 provider pointers, all-thread B copy,
# two-register FragmentA8, M==1 admission, output extent and reduction stride.
python3 - "${repo}" <<'PY'
from pathlib import Path
import sys

repo = Path(sys.argv[1])
root = repo / "quactlize/include/actlize_extensions/cutlass/gemm"
load = (root / "collective/marlin_load_ppu.hpp").read_text()
collective = (root / "collective/marlin_collective_ppu.hpp").read_text()
kernel = (root / "kernel/marlin_kernel_ppu.hpp").read_text()

def body(text: str, signature: str) -> str:
    if text.count(signature) != 1:
        raise SystemExit(f"[l181:source] FAIL: expected one {signature!r}")
    start = text.find(signature)
    brace = text.find("{", start)
    depth = 0
    for i in range(brace, len(text)):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            return text[brace:i + 1]
    raise SystemExit(f"[l181:source] FAIL: unterminated {signature}")

m8_load = body(load, "void ldmatrix_a_m8")
for token in (
    "ppu.ldmatrix.sync.aligned.m8n8.x2.shared.b16",
    '"=r"(a[0])',
    '"=r"(a[1])',
):
    if token not in m8_load:
        raise SystemExit(f"[l181:source] FAIL: plain-x2 seam missing: {token}")
for forbidden in ("m8n8.x4", "discarded_v2", "discarded_v3", "a[2]", "a[3]"):
    if forbidden in m8_load:
        raise SystemExit(f"[l181:source] FAIL: m8 retained x4 baggage: {forbidden}")

for token in (
    "struct FragmentA8",
    "__half2 value[2];",
    "sizeof(FragmentA8) == 2 * sizeof(uint32_t)",
    "using FragmentAFor = std::conditional_t<InstructionM == 8, FragmentA8, FragmentA>",
):
    if token not in load:
        raise SystemExit(f"[l181:source] FAIL: FragmentA8 seam missing: {token}")

for token in (
    "static constexpr int AStoredRows = InstructionM == 8 ? 1 : TileM;",
    "static constexpr int ASharedStage = ASharedStride * AStoredRows;",
    "InstructionM == 8 ? 34816 : 50176",
    "bool const m_supported = InstructionM == 8 ? m == 1",
    "marlin_ppu_detail::FragmentAFor<InstructionM>",
):
    if token not in collective:
        raise SystemExit(f"[l181:source] FAIL: packed-m8 seam missing: {token}")

init = body(collective, "static CtaState init_cta_state")
for token in (
    "int const linear = a_producer_linear(i, tid);",
    "bool const active = a_producer_active(linear, problem_m);",
    "state.a_smem_write[i] = transform_a_index(active ? linear : 0);",
    "state.a_copy_pred[i] = active;",
    "int const k_inner = i / BLoadsPerKInner;",
    "int const k_block = k_inner * WarpOnK + warp_k;",
    "k_block * 16 + 4 * (lane % 2) +",
    "8 * (lane / 16)",
    "state.a_thread_base[i] = a + a_global_stride * (active ? row : 0) + col;",
    "state.scale_thread_base = scale + tid % ScaleSharedStride;",
):
    if token not in init:
        raise SystemExit(f"[l181:source] FAIL: packed-copy/provider seam missing: {token}")

run = body(collective, "static void run_segment")
for token in (
    "for (int i = 0; i < BInnerIters; ++i)",
    "&b_stage[Threads * i + tid]",
    "b_pointer[i]",
    "auto const* a_half = reinterpret_cast<ElementA const*>(a_stage);",
    "constexpr int slot = inner % BInnerIters;",
    "&a_half[state.a_smem_read[slot]]",
):
    if token not in run:
        raise SystemExit(f"[l181:source] FAIL: all-thread-B/x2-read seam missing: {token}")

# A uses the predicated helper; B must use the unconditional helper in the
# same copy_stage body.  This prevents a future rewrite from silently making
# only the 16 A providers responsible for B as well.
copy_stage = body(run, "auto copy_stage =")
if "cp_async_16_if" not in copy_stage or "cp_async_16(" not in copy_stage:
    raise SystemExit("[l181:source] FAIL: A/B cp.async participation split is absent")
if copy_stage.count("&b_stage[Threads * i + tid]") != 1:
    raise SystemExit("[l181:source] FAIL: B copy is not one all-thread indexed loop")

reduce = body(kernel, "static void thread_block_reduce")
for token in (
    "int(OutputThreads) * NBlocksPerWarp * AccumulatorHalves",
    "for (int step = red_off; step > 0; step /= 2)",
    "for (int half = 0; half < AccumulatorHalves; ++half)",
):
    if token not in reduce:
        raise SystemExit(f"[l181:source] FAIL: reduction seam missing: {token}")

print("[l181:source] PASS: packed-M1 copy, all-thread B, plain-x2 provider, m8 output/reduction seams are bound")
PY

echo "[l181:runner] positive=6-contracts negative=${#plants[@]}/${#plants[@]}_RED source=PASS result=PASS"
