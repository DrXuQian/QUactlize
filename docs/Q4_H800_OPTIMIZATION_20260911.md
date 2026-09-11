# Q4 SIMT GEMV on H800: implementation gains, not a layout claim

Scope: M=1, dense, one expert, FP16 activations, FP32 dot/reduction/output.
The canonical K-pack low plane and 16-byte packed units are unchanged.
This is development-only CUDA work; PPU numerical/performance admission and
production routing are unchanged.

## Final H800 verdict: within five percent of both controls

The accepted criterion is now **less than 5% regression against each of
Xplane and the supplied raw reference**, rather than requiring zero regression
against ref. On six frozen shapes and two cache regimes, all **12/12** cells
pass six-round confirmation. Worst regression is **1.88545% versus Xplane**
and **4.40764% versus ref**, calculated before rounding.

The offline format is unchanged. All selected candidates are S1, one kernel,
with no inter-CTA Split-K, separate reducer, or expanded metadata workspace.
This closes this H800 experiment only, not PPU or model-level admission.

| N x K | Frozen candidate | (Columns, Warps, S) |
|---|---|---|
| 512 x 2048 | `meta-static-global-rs-fast-bare-av` | (1, 16, 1) |
| 1024 x 5120 | `affine8-early-fast-bare-a4` | (2, 10, 1) |
| Remaining four shapes below | `affine4-early-fast-bare` | (4, 8, 1) |

Each shape uses the same candidate/config for warm and rotating weights.
The first candidate retains per-weight FP16 reconstruction; the `affine*`
candidates use the distinct FP32 group-affine arithmetic explained below.

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

## Smaller shapes: final six-round confirmation

| N x K | Cache | K-pack | Xplane FP32 | Raw ref FP32 | vs Xplane | vs ref |
|---|---|---:|---:|---:|---:|---:|
| 512 x 2048 | warm | 2.762 | 2.750 | 2.746 | +0.44% | +0.58% |
| 512 x 2048 | rotating | 3.056 | 3.435 | 3.036 | -11.04% | +0.66% |
| 1024 x 5120 | warm | 5.036 | 5.062 | 5.104 | -0.51% | -1.32% |
| 1024 x 5120 | rotating | 6.178 | 6.630 | 5.917 | -6.82% | +4.41% |

The small candidate reduces repeated activation loads with a bit-exact
register transpose. The medium candidate's `a4` arm replaces two 32-bit
activation loads with one aligned 64-bit load; SASS confirms that change.
Neither changes the offline weight bytes. These are implementation choices,
not evidence of an inherent format advantage.

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

All eight larger cells pass both the accepted five-percent bound and the
earlier stricter `K-pack <= ref` criterion for these stated controls.

The full-output normalized error against independent official-GGUF FP64 dots
is at most `1.20e-8` for the confirmed group-affine candidate. The denominator
is `sum(abs(A * W))`, not `abs(output)`. All outputs and guards are checked;
zeroed-code negative controls must fail. This is a kernel fixture check, not
model-level accuracy admission.

## Additional random-activation validation

Two new dense Gaussian activation seeds (`93711`, `93719`), rounded to FP16,
reuse the exact original raw GGUF and K-pack weight bytes. Gold and
conditioning are independently recomputed with the official GGUF decoder
and FP64 dots. The already frozen candidate/config is checked on every shape
in both cache regimes: **24/24 cells pass**, including both controls and
zeroed-code negative checks. These runs do not retune the candidate or replace
the six-round timing evidence above.

Worst normalized errors on these random fixtures are `5.36e-5` for the small
per-weight-FP16 candidate, `9.97e-9` for the medium group-affine candidate,
and `1.72e-8` for the larger group-affine candidate. Controls are at most
`5.87e-5`. These use `sum(abs(A * W))` as the denominator, not `abs(output)`.

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
The [final evidence archive](measurements/q4_h800_opt_20260911/final-evidence.tgz)
adds the frozen small/medium generated kernels, build logs, selections,
confirmation logs, and all random-validation logs. No new candidate was
selected from the random-validation timings.

The [final verdict](measurements/q4_h800_opt_20260911/final-verdict.json)
can be regenerated locally without a GPU; use a new output path:

```bash
python dev/gemv_cuda/summarize_h800_confirmation.py \
  --confirmation docs/measurements/q4_h800_opt_20260911/{small,medium,large}-confirmation.json \
  --validation docs/measurements/q4_h800_opt_20260911/random-{small,medium,large}-{93711,93719}.json \
  --fixtures docs/measurements/q4_h800_opt_20260911/random-fixtures.json \
  --output /tmp/q4-h800-final-verdict.json
```

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
- `prepare_h800_validation.py`: new FP16 activations and independent FP64
  oracles, preserving all original weight bytes.
- `summarize_h800_confirmation.py`: checks complete rounds, samples, frozen
  recipes, numeric errors and random-fixture coverage before adjudication.
- `tests/test_cuda_gemv_h800.py`: CPU ownership, bit permutation, independent
  GGUF group-affine, and negative controls. These are not GPU admission.

Remaining: port/build/test the selected ideas on PPU, whose earlier N4 result
did not meet parity. Keep per-weight-half and group-affine conclusions
distinct when evaluating PPU numeric and performance results. No experimental
flag is a production default and no NVIDIA-specific code is admitted to
product main by this work.
