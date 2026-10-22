# Dense high-plane unique ownership (R5)

Experimental DQ22/23, compared with DQ20/21 at the same N64 K128/K256.
No production selector, canonical K-pack bytes, metadata delivery, shared
consumer, output store or FP32 affine rounding changes. F16 and BF16 final
weight conversion are both retained. Not a decode/MoE/GEMM config sweep.

## What changes

Q6 physical high words feed two logical K32 tasks; Q5 additionally shares
two N8 halves. The new producer owns each physical16B vector once per CTA,
then publishes its fields to the established shared locations. It is not
just a lane permutation of the old redundant requests.

For N64/K256, Q6 high vector requests512->256; Q5 512->128. These are
source-level request counts, NOT measured HBM traffic or a speedup. Q5
publication interleaves K64 groups and alternates the K32 destination phase
to avoid creating same-warp halfword writers in the static32x4B bank model.
Actual shared conflicts still require per-PC device evidence.

Native PPU0010/SDK2.1.1 compilation retains all136 original dequant machine
instruction bodies. Eight new specializations have no stack and unchanged
shared allocation. Same-precision K256 static vector-load sites: Q6 9->7,
Q5 9->6 (includes low and metadata). Native affine arithmetic, shared reader,
vector output-store and barrier opcode counts remain equal. Q5/K128 adds
static instructions671->724 despite fewer loads; do not assume it wins.

Host proofs cover unique aligned in-range reads, single-writer fanout,
32/64/128B footprints, the actual reader versus independent raw GGUF,
F16/BF16 raw equality against controls, nonzero K phases and missing-load /
wrong-high-half negatives. Compile/host proof is not GPU correctness or speed
admission. Production DQ20/21 remains unchanged until Box confirmation.

## Run on an idle ZW810

```bash
git pull --ff-only origin develop
unset RUN
DENSE_ROUND=5 GPU=1 IDLE_CONFIRMED=1 \
  PPU_SDK=/usr/local/PPU_SDK JOBS=4 L2_BYTES=67108864 \
  bash tools/run_prefill_dense_opt_box.sh
```

Four families: Q6 and Q5, each N5120/K25600 and N1024/K5120, M2048.
Eight arms/family: both precisions x DQ20/22/21/23. Numerical gates run
first, dequant is timed alone, then selected whole calls are confirmed
separately. cuBLAS uses warmed ordinary calls, never CUDA Graph capture.
Four matched-tile ACU pairs use ABBA:16 reports total, one dequant launch
per report; no cold ring in the profiler. Request1700MHz; actual clocks
remain recorded/advisory rather than a clock-deviation failure.

The runner prints the unique run directory. Reports are in
`results/profile-<id>.acurep`; `results/acu-<id>.receipt.json` identifies
each exact arm/dtype/tile and its control. The build enables line-info for
this experiment and exports `results/build/high-shared-sites.json` with
shared load/store PCs, operands and source lines. This is a location index,
not conflict attribution: correlate it with instruction-level ACU samples.
Do not derive per-instruction conflicts from a whole-kernel aggregate.

Upload `results/review-*.tgz`: JSON/log/CSV only, no binaries or acurep.
Keep acurep locally for instruction-level inspection; selected reports can
be supplied separately if raw counters do not localize the remaining stall.

If admitted, update only affected helper choices and Full-dequant costs.
Re-evaluate nearby FQ/SF decisions with existing compiled GEMM candidates;
no global resweep is needed. Metadata sub-block loading versus a larger K
tile is the next independent experiment, not mixed into this result.
