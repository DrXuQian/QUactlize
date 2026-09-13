# Incremental Q4 decode sweep

This adds the optimized SIMT readers to the retained Tensor Core candidates.
It does **not** change the production heuristic, offline format or llama.cpp
entry. Box data is required before admitting a new selection.

## Coverage

| Operator | Weight families | Requests | Cases |
|---|---|---|---:|
| Dense | Earlier 11 real N/K families plus the missing 4096×4096 family | M=1 through 8 | 96 |
| MoE | Earlier four families plus two doubled-N gate/up widths | tokens=1 through 8, top8, E256, shared/slot-specific A; old weighted router plus spread/partial/full collisions | 276 |
| Total | Q4 only | Decode, not a prefill Cartesian product | 372 |

The dense N/K pairs are 256×3072, 512×2048, 1024×5120, 2048×4096,
3072×4096, 4096×2048, 4096×3072, 4096×4096, 5120×8192, 5120×25600,
8192×5120 and 25600×5120. The MoE pairs are 512×2048, 512×3072,
2048×512, 3072×512, 1024×2048 and 1024×3072.

Each MoE family has 32 cases (eight token counts × two A conventions ×
spread/weighted routing), plus 14 token8 cases (two A conventions ×
cluster/repeat2/repeat3/repeat4/repeat5/repeat6/repeat7). The weighted router
reproduces the earlier workload inventory's exact expert-ID bytes. Partial
collisions exercise expert-local rows 2–7, rather than inferring a token-only
heuristic from just the spread and fully clustered endpoints. The CPU
generates the fixture but **does not compact or route a timed call**.

## Candidate and timing contract

- The 17 distinct N/K payloads reuse `q4_s1_readers.cuh`. A thin test wrapper
  exposes the previously admitted C8/P4 alternative and resolves row/expert
  addresses; dequantization, dot and CTA reduction bodies remain unchanged.
  C8/P8 is excluded because its cooperative metadata ownership needs 64 lanes.
- There are 6,208 SIMT and 8,020 TC screen cells, including explicit structural
  exclusions. Each family keeps the earlier published SIMT winners, TC
  anchors/winners and today's selected TC recipe. The retained TC pool has
  25 prebuilt parents; runtime S1/2/4/8 and bounded persistent-grid alternatives
  require no further compilation. Packed-row A (`ap=1`) is **FP16 A delivery**,
  not activation quantization; its M1 restriction is explicit.
- This cohort compares **F32 caller inputs rounded to F16** and F32 outputs.
  Dense TC includes F32→F16 input cast, selected TC plus real Split-K reducer,
  and F16→F32 output cast. Indexed TC includes GPU ranks/gather, metadata,
  directory, TC/reducer and scatter. SIMT reads indexed inputs directly, uses
  S1 with no inter-CTA reducer, and writes F32 output. META uses per-weight
  FP16 reconstruction; affine readers retain FP32 group-affine arithmetic.
  Those rounding/order differences are checked against independent GGUF, not
  required to be bit-equal to TC.
- **Do not compare these dense endpoint times directly with the old F16-input
  microbenchmark.** This is also not whole-model timing or the fused MoE
  gate/up/activation/down chain. Dispatch integration must preserve fusion.
- Shipping fused indexed TC supports at most 32 total routed rows. For
  tokens5–8 (40–64 rows), this sweep reuses the separately validated development
  GPU adapter around the unchanged native TC call. A fast result there does
  not expand the shipping ABI by itself.
- Every measurable candidate must pass independent original-GGUF numerics,
  code/A negatives, output guards and mutable-input graph replay. Transferred
  historical readers additionally compare exact bits with their immutable old
  images. New shapes have no old image and are labelled accordingly.
- Both arms traverse the same rotating weight ring, at least 2.25× the
  **verified** L2 size counted from active expert weights, with whole-ring
  graph replays. Setup, graph upload and first launch are excluded. This
  controls weight residency, not the cache state of every input.
- Screen five samples per candidate. Confirm the best two per arm and,
  if not already included, the current TC policy in six alternating rounds
  of fifteen samples. All per-round distributions remain available. A bounded
  search finds the best **measured** config, not proof of global optimality.
- Per-SIMT-recipe results include source warp byte-address/32/64/128-byte
  footprints, actual pointer alignment and native fast-dequant/FP32 ISA
  receipts. These are not measured DRAM traffic. By default ACU captures both
  winners for 18 representative workloads (six dense, twelve MoE); forced-cold
  profiler counters remain separate from rotating event timings.

Q2/Q3/Q5/Q6 optimized-reader porting and the DeepGEMM prefill comparison are
separate tasks. This sweep does not retire their existing canonical TC paths.

## Run on the box

The new package is about 23.3 MiB plus its manifest. The box fetches it and
the old immutable SIMT controls via Git LFS; **no compilation or JIT runs**.
Use one idle PPU and keep other inference/profiling processes off that card.

```bash
git pull --ff-only origin develop && \
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
CUDA_VISIBLE_DEVICES=0 \
bash tools/run_q4_decode_sweep_ppu_box.sh
```

The script runs inside a subshell, so an error does not exit the caller's
Docker shell. It prints per-candidate progress and an observed-case-wall ETA.
The ETA is advisory; ACU time is reported as not estimated. The previous
108-case MoE comparison took 31.35 minutes. This sweep has 3.44× as many
cases and about 3.6× as many potentially measurable screen cells; roughly
two hours is a planning estimate, **not a measured completion bound**, and
larger dense fixtures/profiling can extend it. Nothing is killed for a slow ETA.

Optional `OPERATORS=dense` or `OPERATORS=grouped` runs a declared subset in a
new result directory. Do not change this subset inside an existing resume.
`ACU=0` skips profiling and can be changed back to `ACU=1` on resume without
discarding valid timing records.

```bash
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
CUDA_VISIBLE_DEVICES=0 RESUME_RUN=/workspace/q4-decode-sweep-ppu.REPLACE_ME \
bash tools/run_q4_decode_sweep_ppu_box.sh
```

Each failed candidate is checkpointed before a fresh process continues the
remaining candidates. A case with a numeric failure stays `INCOMPLETE` and
cannot contribute a production winner. An explicit resume retries failed
cells only; successful screens remain valid. If the new screen changes the
finalists, affected confirmation is repeated and earlier samples are archived.
ACU failure does not erase valid timing and can be retried independently.
Source, runtime, package, physical device and Python-package identities must
match on resume; this is not a stale-results override.

Upload the printed `q4-decode-sweep-ppu.*.results.tgz`. It contains the
manifest-bound inventory, `summary.tsv`, proposed `selection-input.json`,
per-case proofs and raw samples, logs, failures and ACU reports. The selection
input is **pending review**, not an automatically installed heuristic.

## Rebuild locally

Only needed after kernel/source changes, not to execute this sweep:

```bash
python dev/gemv_ppu/build_decode_sweep.py \
  --sdk /root/ppu-sdk/2.1.1 \
  --cache /root/autodl-tmp/kpack-jit-gridfix-20260910.PWASqI \
  --output /root/autodl-tmp/q4-decode-sweep-NEW --jobs 24
```

Build receipts and static ISA verification are not PPU numerical admission.
