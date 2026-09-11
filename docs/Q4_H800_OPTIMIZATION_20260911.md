# Q4 SIMT GEMV on H800: implementation gains, not a layout claim

Scope: M=1, dense, one expert, FP16 activations, FP32 dot/reduction/output.
The canonical K-pack low plane and 16-byte packed units are unchanged.
This is development-only CUDA work; PPU numerical/performance admission and
production routing are unchanged.

## Two separate experiments

1. **Per-weight FP16 reconstruction.** Wider adjacent-N loads, less repeated
   activation loading, exact nibble conversion, cooperative metadata, and a
   warp reduce-scatter improve the original K-pack reader. Accumulation stays
   FP32. These experiments retain the per-weight half-precision boundary;
   they are separate from the algorithm below.
2. **FP32 group affine.** For a 32-weight group, compute
   `scale * dot(q, A) - min * sum(A)` directly in FP32. Scale/min are decoded
   from the raw packed-unit header in FP32. There is no per-weight FP16
   reconstruction and no expanded workspace. It eliminates repeated affine
   operations but changes the floating-point evaluation order. The old
   Xplane/ref controls also use FP32 accumulation, but still reconstruct each
   weight in FP16. Thus a win here is an implementation/algorithm win, **not
   proof that K-pack's layout is intrinsically faster**. Applying the same
   group-affine optimization to the controls could change the comparison.

The supplied `gemv_ref.cu` remains unchanged on disk. Its generated FP32
control changes dot accumulators, fold and output to FP32, retaining its raw
GGUF reader and per-weight FP16 reconstruction. Its launch inventory was
expanded to 60 valid `(columns, N-warps, intra-CTA K-warps)` configurations.
Those K-warps require no extra kernel. The historical Xplane control has its
original 12 advertised configurations and Wk=1 restriction.

## Larger shapes: frozen configuration, independent six-round confirmation

The candidate is `affine4-early-fast-bare`, `(Columns,Warps,S)=(4,8,1)` for
**every row below**, with TileN=16 and 256 threads. There is one kernel, no
inter-CTA Split-K, no separate reducer, and no cache-mode-dependent tactic.
Units are microseconds; negative percentages mean K-pack is faster.

| N x K | Cache | K-pack | Xplane FP32 | Raw ref FP32 | vs Xplane | vs ref |
|---|---|---:|---:|---:|---:|---:|
| 4096 x 2048 | warm | 5.333 | 5.301 | 5.943 | +0.61% | -10.26% |
| 4096 x 2048 | rotating | 6.332 | 6.351 | 7.084 | -0.30% | -10.61% |
| 4096 x 4096 | warm | 8.511 | 8.354 | 9.635 | +1.89% | -11.67% |
| 4096 x 4096 | rotating | 9.938 | 9.800 | 11.528 | +1.41% | -13.79% |
| 5120 x 8192 | warm | 14.667 | 16.091 | 18.700 | -8.86% | -21.57% |
| 5120 x 8192 | rotating | 17.756 | 18.700 | 22.166 | -5.05% | -19.89% |
| 8192 x 5120 | warm | 15.233 | 15.888 | 19.294 | -4.12% | -21.05% |
| 8192 x 5120 | rotating | 17.521 | 19.239 | 22.489 | -8.93% | -22.09% |

All eight cells pass `K-pack <= 1.05 * Xplane` and `K-pack <= ref` for these
stated controls. This does not close the six-shape task: the two smaller
shapes still have regressions against the newly tuned ref.

The full-output normalized error against independent official-GGUF FP64 dots
is at most `1.20e-8` for the confirmed group-affine candidate. The denominator
is `sum(abs(A * W))`, not `abs(output)`. All outputs and guards are checked;
zeroed-code negative controls must fail. This is a kernel fixture check, not
model-level accuracy admission, and more activation fixtures remain useful.

## Measurement boundary

- H800 PCIe, 114 SMs, reported L2 52,428,800 bytes, CUDA 12.8.
- Six alternating-order rounds; 15 event samples per round; median of round
  medians. Candidate/config is frozen before confirmation.
- Warm uses one resident weight. Rotating completes a weight ring at least
  2.25 times L2 on every replay; it is not called a guaranteed DRAM flush.
- A graph contains at least 32 calls and complete ring traversals. Initial
  graph launches/warmups and offline conversion are excluded for all arms.
- All in-kernel work, barriers, stores and reductions are timed.
- No concurrent GPU benchmarks. NCU counters are unavailable on this H800
  (`ERR_NVGPUCTRPERM`); do not claim H800 hardware-stall counter evidence.

Raw records: [large confirmation](measurements/q4_h800_opt_20260911/large-confirmation.json),
[full screening](measurements/q4_h800_opt_20260911/full-screen.json), and
[small follow-up](measurements/q4_h800_opt_20260911/small-screen-r17.json).
They retain all per-recipe samples, fixture hashes, payload paths and log
hashes. `MEASURED` indicates complete measurements, not a global parity verdict.
The [evidence archive](measurements/q4_h800_opt_20260911/large-evidence.tgz)
also preserves generated sources, manifests and raw logs (not compiled objects).

An additional [S2 experiment](measurements/q4_h800_opt_20260911/s2-screen.json)
includes the producer **and** FP32 reducer in every timed call. For
`1024 x 5120`, the best observed complete-call times were 6.401 us warm and
7.686 us rotating, slower than S1, so it is not selected. Its W5/W10 variants
use a private development dispatcher with that exact S2 predicate enabled;
this does not broaden shipping admission. All 18 recipes per S2 arm passed
the full-output oracle after the benchmark query predicate was aligned with
the templates.

## Code and remaining work

- `dev/gemv_cuda/q4_group_affine.cuh`: vectorized FP32 group-affine candidate.
- `q4_cooperative_affine.cuh`: group dot distributed across residue lanes.
- `q4_nwide.cuh`, `q4_cooperative_metadata.cuh`: per-weight FP16 reader arms.
- `q4_warp_reduce_scatter.cuh`: FP32 reduction/ownership transformation.
- `build_h800_candidates.py`: immutable, bounded experiment builds.
- `build_h800_reference.py`: raw-reader launch-only control expansion.
- `run_h800_candidates.py`: screening and frozen-shortlist confirmation.
- `tests/test_cuda_gemv_h800.py`: CPU ownership, bit permutation, independent
  GGUF group-affine, and negative controls. These are not GPU admission.

Remaining: make the smaller shapes no slower than the tuned raw reference;
freeze and confirm all six shapes; validate additional dense activation
fixtures; keep per-weight-half and group-affine conclusions distinct; then
build/test the relevant PPU candidates. No experimental flag is a production
default and no NVIDIA-specific code is admitted to product main by this work.
