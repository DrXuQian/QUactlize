# Paired gate/up full-call performance gate

This is a component experiment, not a model/TPOT admission. The preceding
numerical gate covered all six qtypes; this next cohort deliberately focuses
on the two actual gate/up roles, not another Cartesian product.

Returned result: [all16 points passed and show a confirmed fusion gain](GATE_UP_PAIRED_N4_PERF_REVIEW_20260917.md).
Q8 shared selects SIMT throughout; Q4 routed benefits from per-token SIMT/TC
choices. Production and whole-model admission remain unchanged and pending.

| Role | One projection N/K | Experts | Tokens | Input/output | A compute | Projection rounding |
|---|---|---:|---|---|---|---|
| Q8 shared expert | 512/2048 | 1 | 1..8 | F32 | F16 | none (current shared path) |
| Q4 routed experts | 512/2048 | 256, top8 | 1..8 | F32 | BF16 | BF16 before SwiGLU |

There are **16 points**. Q4's existing concatenated gate/up has N1024; the
paired candidate's public N remains512 and its physical N is1024. Input IDs
are distinct per token, with overlap permitted across tokens. This is one
deterministic spread profile, not all router distributions. All arms use the
same original GGUF weights, A, IDs and final logical channel order. No A INT8
quantization, clipping, down GEMM or routing/top-k is included.

## Baseline authority and measured boundary

The immutable execution image is copied from `tools/kpack_q4_model_artifact.json`
(artifact3935fe9, execution SHA2563247b57f6c457b88571f18242dac0ea173144c75682aa7d14c45e0b844baa544).
Each point retains its exact `smallm-matched-policy.json` choice, applying the
Q8 vector override only if its old recipe matches. In particular:

- Q8 M1 uses the production hoist entry, not the old narrow-load body.
- Q8 M7/M8 retain the selected SF TC parent and S8 reducer, separately for G/U.
- Q4 M8 retains the selected grouped BF16 TC parent.
- Other points retain their measured per-token SIMT recipe, including Split-K.

Only the two necessary TC parents are rebuilt as explicit typed endpoints;
the existing fusion and39MB production execution libraries are unchanged.
The component boundary starts with resident F32 A/IDs and packed weights and
ends with F32 SwiGLU output. Shared baseline: two selected projections plus
post-op. Routed baseline: one concatenated projection plus post-op. All
reducers are included. The standalone Q4 TC control includes its indexed
prepare/GEMM/finish path, since this component's input is F32 plus device IDs.
It is **not** the already-prepared inner GEMM of a shared whole-MoE chain.

The baseline post-op is a minimal standalone SwiGLU with matching precision,
not a copy of application routing/bookkeeping overhead. The experiment thus
does not assign all model helper costs to the unfused arm. Differences are
whole component implementation differences, not proof that the epilogue alone
caused a gain. Model caller integration and native-llama comparison remain
separate gates; do not infer TPOT by multiplying these times by layer count.

## Bounded candidates and timing

Per point: eight fused SIMT configurations (W4/W8 × S1/2/4/8) and eight fused
TC configurations (TM8/TM16 × S1/2/4/8), plus the exact incumbent. This is272
screening cells, not a global-optimality claim. No production selector changes.
Each cell first checks two changed inputs at the first/last physical weight
copy, output/workspace guards, a zero-A negative and graph replay correctness.
Incorrect output has no timing admission.

Three alternating screening samples select one SIMT and one TC finalist.
Those two plus the incumbent receive fresh confirmation samples: default four
rounds ×11, reversing arm order every round. Screen samples are not recycled
into confirmation medians. Both backend finalists remain visible even if one
loses. Setup, CPU fixture work, packing, H2D/D2D, handle preparation, JIT and
first graph upload/replay are outside the timed region; there is no JIT on box.

Every timed graph makes one complete traversal of a weight ring containing
at least2.25× verified L2 capacity in **active weight slices**. The ring uses
different addresses, preserves real E256 strides and includes packed metadata.
Only inactive expert slices may be zero-filled. A and intermediates can be
warm; this is rotating-weight timing, not proof that all data is cold. Plane
payloads are256-byte aligned despite their guards; returned receipts record
the actual alignments, active IDs, raw hashes and copied-byte denominator.
GPU idleness is not programmatically proven: use an idle device.

Modeled MBU uses distinct active gate+up packed bytes / complete-call latency,
including metadata, against the user's2700GB/s peak assumption. It is not an
ACU measured DRAM counter. ACU and whole-model Asys follow a measured finalist.

## Address and decoder invariants

The candidate SIMT C4/P8 mapping is unchanged from the admitted numerical
image: lane `l` owns physical N `(l % 4)*8` within a32-column tile. Four
neighboring lanes read four adjacent16-byte vectors per packed K address,
forming one64-byte span; a warp contains eight such K-worker groups, not one
512-byte request. With256-byte plane bases and N1024, these64-byte spans are
64-byte aligned (two32-byte sectors; one64-byte footprint; one128-byte
footprint). Multiple K groups are not conflated in the footprint count.
W4/W8 and S change worker/pass counts, not that per-warp address relation.

The Q8 scale vector also spans64 adjacent bytes per four lanes when aligned.
K-quant metadata uses the existing cooperative packed-unit loads. The same
register-reuse fast code extraction/FP32 affine accumulation is retained; the
paired shuffle changes which G/U values occupy contiguous physical rows, not
their packing width. TC uses the existing AIU collective and changes the
final paired register epilogue. These source relations are not measured
transaction counts or proof that every candidate matches the optimized
incumbent's instruction mix; the incumbent hoist/H32 paths stay intact.

## Run / resume

Use `tools/run_gate_up_perf_box.sh` with the pinned prebuilt bundle. Select one
idle GPU. Set `L2_BYTES` only to a verified capacity; a positive conflicting
device attribute is rejected. The known test box previously used67108864.

```bash
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
CUDA_VISIBLE_DEVICES=0 L2_BYTES=67108864 \
bash tools/run_gate_up_perf_box.sh
```

Each point runs in a fresh process. A failed point preserves its log and does
not stop the remaining points. To retry, set `RESUME_RUN` to the run directory;
only complete, identity-matching points are reused. Old failed files are kept.
The wrapper packages the result JSONs, sample distributions and logs, without
binaries/weights. It preserves the caller's Docker shell. The returned summary
must contain16 complete points; elapsed/remaining estimates use observed point
wall time, not kernel microseconds. No fixed overnight-duration claim is made.
