# Dense prefill helpers

Current dense selection is documented in
[DENSE_PREFILL_ROUTING.md](DENSE_PREFILL_ROUTING.md); the latest measured
admission and final Box command are in
[PPU_DENSE_CONTIGUOUS_IO.md](PPU_DENSE_CONTIGUOUS_IO.md).
A helper-only benchmark is not a new route table or a model latency result.

## Retained implementations

Typed dense dequant has its own stable ID namespace in
`quactlize/prefill/dense_api.h`, separate from the SF/full weight config.

| IDs | Implementation | Formats | Output |
|---|---|---|---|
| 0 | Direct K-quant / Q8 reader | Q2_K..Q6_K, Q8_0 | F16 or BF16 |
| 8 | N32/K128 pair | Q4_K/Q5_K/Q6_K | F16 or BF16 |
| 20/21 | N64 cooperative K128/K256 | Q4_K/Q5_K/Q6_K | F16 or BF16 |
| 22/23 | N64 plane-pair K128/K256 | Q2_K/Q3_K/Q8_0 | F16 or BF16 |
| 38 | N64/K128 fast half2, 256 threads | Q8_0 | F16 only |
| 41 | N64/K128 paired-scale fast half2, 128 threads | Q8_0 | F16 only |
| 43 | Same as41, byte-permute reconstruction; opt-in candidate | Q8_0 | F16 only |

IDs 24..37, 39, 40 and 42 were unselected experiments and are retired.
They fail validation; no ID is renumbered or reinterpreted. Their source,
commands and earlier investigation narrative remain at
`618298d6f8761b8eb4489ad2f1a217dd57f0bd47`. Raw measurement records remain
under `docs/measurements/`. Old source-bound runs must use their frozen
scripts and binaries, not resume against the current library.

Q8 packed-half arithmetic preserves exact signed codes and original F16
scale bits. It does not change GEMM accumulation. BF16 must not pass through
rounded F16: IDs38/41/43 reject BF16 at both prepare and launch. Canonical
planes, metadata, output layout and arithmetic of all retained kernels are
unchanged by cleanup.

IO0..4 and compact reducer widths2/4/8 remain available. They are selected
only where the measured complete-call policy admits them; a faster isolated
helper does not automatically overwrite a route. Full's private
dequant-before-input order is exact-only. The public v1 default enqueue order,
workspace contract and dynamic pointers/streams remain unchanged.

## Selection and evidence

- `policies/dense_prefill_v1.json`: complete-call recipes, 225 exact points
  and 147 bounded bucket predictions. Selection binds main, expansion,
  conversion and reduction together; predictions are not new-shape proofs.
- `policies/dense_full_helpers_20261023.json`: the separate Full-only
  typed-helper measurements.
- `quactlize/prefill/dense_helpers.hpp`: the original TC/SF helper domain;
  a Full-only record must not replace a TC/SF choice.

`tools/fit_dense_prefill.py --check` and
`tools/generate_dense_full_helpers.py --check` verify generated tables.
This cleanup changes neither catalog nor generated selection tables.

DQ43 passed the bounded six-shape Box comparison on20261027. It reduced
integer instructions and the two focus cases' ACU cycles, but the normal
rotating-event gains overDQ41 were only0.25%/0.57%. Small shapes still prefer
DQ38. The [measurement record](measurements/q8_paired_dequant_20261023.txt)
keeps both timing contexts and the source/archive identities. DQ43 remains
opt-in: no complete-call table or bucket has been changed to select it.

The first helper rounds R1–R5 are recoverable at
`8a004d42a91a23664e69762509d34c3b76bbf8e8`.
The high-plane R5 candidate was not admitted; removing it does not invalidate
its recorded results. Production/test code contains only the retained
implementations and their independent oracles.

## Isolated dequant comparison

The maintained Q8 cohort tests IDs0/22/23/38/41. Add `Q8_NATIVE=1` to call
the existing llama Q8 F16 converter directly; it does not rebuild llama or
time its GEMM. Optional explicit N/K pairs are bounded to eight shapes and
must obey the execution API's N256/K256 alignment, not just the helper tile.

```bash
(
  set -euo pipefail
  cd /sim/eec/shared/junfu.qx/quactlize
  git pull --ff-only origin develop
  unset RUN RESUME_RUN POINTS PROFILE_POINTS Q8_SHAPES
  unset Q8_EXCHANGE Q8_FAST_TAIL Q8_HALF2 Q8_PAIRED Q8_BYTEPERM
  PPU_SDK=/usr/local/PPU_SDK GPU=1 L2_BYTES=67108864 IDLE_CONFIRMED=1 \
  LLAMA_CI_BUILD_DIR=/sim/eec/shared/junfu.qx/llama.cpp.ncp/build-ci-32F-cublas \
  Q8=1 Q8_NATIVE=1 PROFILE_ACU=1 \
  bash tools/run_dense_dequant_box.sh
)
```

Use an idle assigned GPU. `Q8_SHAPES=8192x3072,4096x2048` replaces the
default six Q8 families, without changing their historical point numbering.
Omit `Q8_NATIVE=1` to avoid a llama dependency. Without `Q8=1`, the runner
compares the generic Q3/Q8 helpers0/22/23, checking both F16/BF16 and timing
F16. Retired experiment switches fail explicitly.

For the retained byte-permute candidate, add `Q8_BYTEPERM=1` with `Q8=1`.
This compares38/41/43 and, with `Q8_NATIVE=1`, the existing native converter.
It uses the same runner and correctness gates, not a separate experiment
entrypoint. Omit or unset `Q8_BYTEPERM` for the default product cohort.

All256codes x63488finiteF16scales, distinct paired scales, subnormals,
signed zeros, overflow bits and planted faults are checked before Q8
timing. Whole-output GGUF/guard checks and changed addresses remain required.

Timing uses alternating single-kernel rotating graphs, input and output
rings each exceeding2.25L2, with warmup excluded. The1700MHz lock request
is required; observed deviations are advisory, not failures or a reason
to normalize latency. ACU captures one checked direct launch, no graphs
or cold loops. Those counters are separate from rotating-event times.

Upload the printed `review-*.tgz`: proofs, samples, source/build/device
identities and raw counters only. ACU reports, binaries and weights stay
on Box. No production promotion follows a runner PASS.

## Tests and final validation

`run_tests.sh --suite host` covers real host readers, layouts, rounding,
typed runtime binding, invalid IDs and selected helper support.
Q8 paired-scale tests are part of product host CI, not only experiment tests.
`tests/test_prefill_dense.py` and `tests/test_dense_q8_paired.py` retain
independent GGUF and exhaustive arithmetic checks; no archived PASS replaces
them. PPU tests use the built public package.

The cleanup compile audit removed28 unselected GPU specializations and
retained44. Every retained native instruction hash and resource record was
identical before/after with SDK2.1.1. This is compile evidence, not a fresh
GPU accuracy or performance run. The outstanding public-library/whole-model
validation remains the single final round linked above; cleanup adds no
new parameter sweep.

The20261027 follow-up factors the identical fast half2 reconstruction and
scale multiply of38/41/43 into `DenseQ8Half2::weight_encoded`. All45 dense
kernel instruction hashes, opcode inventories and resource records match the
pre-cleanup bc06873b build using the same local SDK/compiler. The56 related
host checks pass. This is a code cleanup, not a new device performance claim.
