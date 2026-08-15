# Dense fixed Split-K M=1 reducer

## Production state

The producer equivalence is committed at `2ae4708`: for the TN64 row, the
external `(1,32768,512)` GEMM and internal `(1,4096,4096), S=8` producer differ
by `-0.0016 us` under one exact warm-resident protocol, with raw-bit-correct
outputs after reduction.

The production reduction handle now dispatches an M=1 fast path when all of
the following are true:

- `S` is exactly 2, 4, or 8;
- N is a complete 64-column stripe;
- D is contiguous;
- the FP32 workspace is 128-byte aligned and D is 16-byte aligned;
- the one-warp CTA count fits `dim3.x`.

It launches 64 one-warp CTAs at N=4096, instead of launching 16 four-warp CTAs
and retiring three warps in every CTA. `S` is a compile-time parameter and the
load/add order is fixed at `0,1,...,S-1`. Wider M, N tails, custom strides,
weaker alignment, HostAdapter builds and oversized grids retain the checked
generic reducer as a complete fallback.

The vector fixed-order primitive is independent of the launch wrapper. A
future last-arriver epilogue can call the same primitive after its release/
arrival protocol and thereby remove the second launch without changing the
arithmetic order.

## Local evidence

`l189` proves the production dispatcher and behavior:

- S=2/4/8 fast cases are raw-bit exact;
- output is poisoned before every case and all 4096 columns have distinct
  signatures;
- M1/N66 tail, workspace+16 B and D+2 B each execute the fallback and remain
  raw-bit exact;
- the oversized-grid case cannot enter the fast dispatcher;
- `compute-sanitizer` reports zero errors.

`l194` compares the legacy topology, CUTLASS vector candidates, and the exact
production type. On the local RTX 5090, S=8 moved from about `2.22 us` to
`1.61 us`; the matching empty launch is about `1.60 us`. A named S4-to-S2
dispatcher plant produces numeric RED, so the gate cannot pass by silently
using the old reducer.

The sm_120 disassembly of the production EPA=2 bodies contains exactly 2/4/8
`LDG.E.64` instructions for S=2/4/8 and one vector store in each body. Register
counts are 13/20/28 with zero local-memory spill. This is a local NVIDIA
code-generation fact only; the PPU run must still confirm hgcc emits the
corresponding vector loads without spill.

`l195` isolates the memory body with a 142,606,336-byte S=8 stream. EPA=2 is
the production choice because it is the best local bandwidth point and maps
N=4096 to 64 active warps on a 72-CU PPU:

| EPA | local logical GB/s | local nameplate utilization |
|---:|---:|---:|
| 1 | 1693.9 | 94.5% |
| **2** | **1707.1** | **95.3%** |
| 4 | 1698.6 | 94.8% |
| 8 | 1703.8 | 95.1% |

This is a body-bandwidth result, not a claim that the shipping 136-KiB S=8
reduction reaches 95% HBM. At N=4096 the standalone kernel is launch-limited:
its logical traffic is only `N * (4*S + 2) = 139,264` bytes. On PPU the old
measured empty-launch floor was about `1.845 us`, so 80--90% of nameplate is
not a physically meaningful small-shape acceptance criterion. The product
acceptance criteria are instead: raw-bit correctness, exact fast dispatch,
minimal/coalesced transactions, and total reducer latency close to the
matching launch floor.

## PPU measurement

After checkout of the committed SHA, run:

```bash
cd /sim/eec/shared/junfu.qx/quactlize
OUT=/workspace/quactlize-dense-splitk-fast-reducer \
EXACT_WARM_AB=1 ITERATIONS=200 \
bash tools/run_dense_splitk_sweep_box.sh
```

Each TN64/TN128 row prints, under one timing protocol:

- `packedA_internal_S8_producer`;
- `packedA_internal_S8_reducer_only`;
- `packedA_internal_S8_full_e2e`;
- `fast_path_selected`;
- raw-bit post-timing correctness.

The earlier PPU S=8 reducer was `6.04 us`. The first target is to move the
standalone reducer close to the PPU launch floor while keeping the producer at
the already reproduced ~7.9-us point. If the body is fast but total latency is
still launch-bound, the next step is the last-arriver fusion described above,
not another standalone memory-kernel rewrite.
