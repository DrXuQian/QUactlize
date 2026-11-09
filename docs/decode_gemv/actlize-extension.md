# Exact-code MMA actlize extension

The implementation is an **optional standalone collective**, like the standalone
Marlin extension. It does not specialize an actlize-owned dispatch tag or change
the existing SF/FQ mixed-input collective. The actlize gitlink is unchanged.

## Interface and ownership

Headers under `quactlize/include/actlize_extensions/cutlass/gemm/collective/`:

- `ppu_mma_exact_code.hpp`: `ExactCodeCollectivePPU`, register-direct delivery.
- `ppu_mma_exact_code_staged.hpp`: the cp.async weight-staging specialization.
- `detail/ppu_exact_code_reader.hpp`: canonical code/metadata readers and the
  existing actlize PPU0010 F16/BF16 m8n16k16 atom.
- `detail/ppu_exact_code_geometry.hpp`: host-testable ownership and shared sizes.

```cpp
#include "actlize_extensions/cutlass/gemm/collective/ppu_mma_exact_code_staged.hpp"
namespace ec = cutlass::gemm::collective;
// Q4_K, F16 compute, T=4 columns/lane, WN=4, WK=1, capacity=1 token.
using Direct = ec::ExactCodeCollectivePPU<
    12, cutlass::half_t, 4, 4, 1, 1, ec::ExactCodeRegisterDirectPPU<2>>;
using Staged = ec::ExactCodeCollectivePPU<
    12, cutlass::half_t, 4, 4, 1, 1, ec::ExactCodeCpAsyncPPU<2>>;
// Inside a CTA launched with Direct::Threads and Direct::shared_bytes(k_partition):
// Direct::run(args, shared, store);  store(token, absolute_n, fp32_value)
```

`Arguments` holds one expert's canonical low/high/metadata planes, original N
pitch, a F32 activation pointer, element stride, active token count, N tile start,
and a half-open interval of 256-K superblocks. All CTA threads participate.
Pointers, N/K bounds, alignment and scratch lifetime are caller responsibilities.
Shared memory must be 128-byte aligned. No allocation, launch, JIT, host wait or
routing-ID readback occurs in the collective.

The execution headers `mma_reader.cuh` and `mma_staged.cuh` are thin adapters:
they bind the existing qkg call and device IDs, launch the collective, and invoke
the unchanged ordered FP32 Split-K reducer. Kernel names/config identities remain
compatible with the existing investigation tools. The adapters reject invalid
geometry, Split-K sizes and insufficient scratch before launch.

## Preserved contract and limits

F32 activations are rounded once to F16/BF16; exact integer weight codes enter
MMA; MMA accumulation, group affine and reduction use FP32. This is not the
per-weight rounded reconstruction of the old mixed-input mainloop.

Current scope: Q4_K/Q5_K/Q6_K/Q8_0; Q8 F16 only. Register-direct capacities are
1/2/4/8 tokens; active rows may be smaller. Staged capacities are 1/2 and stages
2/3/4. Indexed adapters process one token/expert slot per CTA row; this does not
limit the entire MoE call to two slots. Q6 staged Split-K boundaries retain
512-K alignment. No Q2/Q3, new layout, new public ABI or serving inactive-ID
semantics are added. Invalid IDs retain the prototype's NaN diagnostic.

The native atom's A/B/C lane maps are checked at compile time. The shared
producer/consumer layout, pipeline waits, affine order, CTA K fold and output
partial layout are preserved from b23397ed. These are structural contracts,
not a new GPU accuracy or speed claim.

## Verification

```bash
QUACTLIZE_CI_SUITE=host python3 -m pytest -q \
  tests/test_exact_code_extension.py tests/test_product_tree.py
python3 tools/decode_gemv/check_extension.py --sdk "$PPU_SDK" --jobs 2
```

The compile check instantiates 11 direct and 9 staged cases both without the
execution ABI and through its adapter. Optional `--baseline PATH` compiles the
same adapter fixture against the unmodified prototype. Output includes command
lines, source hashes, objects, native ISA and resources; no GPU is launched.
The inventory includes CTA K-warp reduction, 2/4/8-column fragments and 2/3/4
shared stages, not only the single-K-warp branches.

Local verification (HGCC 2.2.1-184ee6, CUDA13 variant, ppu_10):
17 host tests passed; all 20 standalone/adapter configurations compiled.
Compared with the same-compiler b23397ed control, all 20 adapter kernels retain
their native MMA instruction inventory and use zero stack. VREG is unchanged
for 19 and drops 96 to 94 for one; no VREG increase. Static body counts differ
by -13 to +5 instructions, not a measured latency result. Existing reducer
instruction sequences match after ignoring function-index prefixes on BB labels.

Pass the tile arguments and store functor by value. With reference-bearing
epilogue capture, this HGCC generated extra parameter reloads and scalar output
stores; value capture removes that abstraction cost. The underlying arithmetic,
store addresses and reduction order do not change.

The historical 46 direct / 29 valid staged correctness cases belong to the old
prototype. The extracted extension still needs PPU numerical and full-call
timing rechecks before any production routing promotion. Do not reuse the
aborted staged sweep as timing evidence.

## Box retest (no performance sweep)

With NumPy, PyTorch and gguf-py installed and the PPU SDK selected:

```bash
DEVICE=1 JOBS=4 python3 tools/decode_gemv/retest_extension.py
```

This compiles only the configurations in the 46 direct / 29 valid staged
historical cases, plus the unchanged GPU packer. It needs neither an installed
product package nor a model file, and never fetches a dependency or installs
over a production library. The two historical rejected staged configurations
are listed as excluded, not counted as passing checks. M=1..8 dense and indexed
BF16 cases retain their original shapes, channels, precision and .005 absolute-dot
normalization. Weights are checked against the independent gguf-py decoder.

The run includes eager warmup, graph capture, upload and two replays, but emits
no timing claim. Success is `MMA_RETEST_DONE status=PASS`, with `direct=46/46`
and `staged=29/29` in `summary.json`. Upload the printed `review.tgz`; only JSON
and text logs are included, not binaries, traces or ACU reports. Without
`--output`, a new `qz-mma-retest-*` directory is created under the system temp
directory and retained for debugging. `--plan-only` and `--build-only` are
explicit non-device modes and never report a device PASS.
