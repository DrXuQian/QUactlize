# quactlize

Low-bit weight-only GEMM kernels for the T-Head PPU (ZW810), built on a CUTLASS-3 style collective.
W4A16, W2A16 and W1A16, plus the GGUF k-quant formats that are not a single power-of-two width —
Q3_K, Q5_K and Q6_K — reached through **bit-plane decomposition** rather than a separate kernel per format.

## What is here

| | |
|---|---|
| **Mixed-input grouped GEMM** | MoE, one weight matrix per expert, per-group scale and zero. `quactlize/include/moe_grouped_ppu.cuh` |
| **Dense mixed-input GEMM** | `fpA_intB_ppu.cuh` |
| **Split-K grouped GEMM** | the slice on `gridDim.z`, a light reduce kernel after. `moe_splitk_ppu.cuh` |
| **CUDA-core GEMV** | the decode band, no shared memory in the main loop. `quactlize/include/gemv_lowbit/` |
| **Format definitions and the offline** | `gguf_scale_layout.hpp`, `xplane_offline.hpp`, `tools/` |

### Formats

| format | weight | scale | how |
|---|---|---|---|
| W4A16 | int4 | per-group fp16, optional zero | one plane |
| W2A16, W1A16 | int2, int1 | per-group | one plane, N-fold to clear the AIU's 32 B floor |
| GGUF Q4_K | int4, gs=32 | affine (`d`, `dmin`) | in-tile scale, the min folded in for free |
| GGUF Q3_K | int2 + int1 | symmetric, gs=16 | **two planes concatenated in the converter** |
| GGUF Q5_K | int4 + int1 | | same |
| GGUF Q6_K | int4 + int2 | symmetric | same |

Q3_K and Q6_K carry **no zero plane at all**: the format's centre rides on the converter's own bias
(`1 << (W-1)` over the concatenated code), so the whole zero channel disappears instead of being materialised.

## Layout

```
quactlize/include/     the kernels: launchers, format definitions, the fold/delivery traits
quactlize/include/gemv_lowbit/ the CUDA-core GEMV family (shares nothing with the cutlass path)
tests/                 correctness, each against an INDEPENDENT oracle -- a native golden or a host reference
tests/data/            real-weight fixtures the tests read
benchmarks/            timing harnesses, and run_batch.sh which pins one row and builds the traps in
tools/                 offline: weight extraction from gguf/GPTQ, the derived reorder, fixture generation
docs/                  formats, the reorder derivation, and the project checkpoint
third_party/           cutlass (NVIDIA) and actlize (the PPU CUTLASS-3 fork this builds on)
```

The `dev` branch adds `dev/` — the probes, the ablation switches, the sweeps, and `fold_derivation/`, the
derivation record. That is a working area, not a product, and it is deliberately absent from `main`.

## Building

Requires the PPU SDK (`hgcc`, the `hggc` runtime) and a `ppu001` device; nothing here builds for the device
without them. `build.sh` overlays a target into actlize's example tree and builds just that one:

```bash
git submodule update --init --recursive
TARGET=test_moe_grouped_verify ./build.sh
$BIN/test_moe_grouped_verify 8 1
```

`PPU_DEFS=<space-separated defines>` reaches both the host and the device compile, and the build prints
`PPU_DEFS verified on <target>'s compile command` — if that line is absent the macro did not reach the device
and any A/B taken from that binary is comparing a build with itself.

## Benchmarking

`benchmarks/run_batch.sh` builds a set of variants, keeps each binary before the next build deletes it, checks
they are byte-distinct, runs the correctness gates, and only then times one pinned configuration. Same-config
run-to-run spread on this part is around 13%, so numbers are only comparable within one invocation.

## Status

See `docs/CHECKPOINT.md`. In short: the formats above are validated against real weights on `ppu001`; the
largest single measured cost on the decode band is the int4→fp16 dequant pipeline at 11.1% of kernel time.
