# Verified reader/prepare integration and model profiling

The completed replay is `/root/kpack-local-closure.SPoAer.results.tgz`,
SHA256 `b9f38b2effbb84238d4b6f64c09291b3900e5ef7782b4d66e3cd2bdc6acdd1ae`.
It closes the pending cases in the [previous review](KPACK_LOCAL_CLOSURE_REPLAY_20260916.md).
It does not measure the newly assembled production bundle or whole-model speedup.

| Returned gate | Verified result |
| --- | --- |
| BF16 capability | 746/746: grouped TC226, SIMT396, complete MoE116, range/FP16 negatives8 |
| Prepare numerical | 3,840 contexts, four replays; ordered map-swap negative observed |
| Prepare timing | 48 complete comparisons; candidate lower in38; all-SIMT16/16 lower |
| Q8 reused evidence | Previous `wqNPz5`: 12,480 numeric and50 rotating complete-call comparisons;48 lower medians |

The Q4 BF16 midpoint case retains the original output hash and the0.005 stage
bound; comparison is to the independent unrounded dot. The complete-chain
bound remains0.02. No clipping or production numerical workaround is added.

## Production changes

- Q8 vector reader is now in `quactlize/execution/simt_q8_vector.cuh`.
  Public variants4/5 map to experimental readers0/1. Input storage isF32;
  computation is explicitlyF16 orBF16; accumulation/output areF32. Offline
  K-pack2 bytes, FP32 group-affine arithmetic and K order are unchanged.
  Split-K calls include the existing real reducer.
- `policies/kpack_q8_vector_v1.json` overlays11 **dense** exact SIMT incumbents.
  The baseline geometry must match the contemporaneous control and the new
  median must be lower. TC choices, differing recipes and regressions remain.
  All16 indexed timings usedE16, not the model'sE256, so they are not promoted
  as exact model results. Policy15 identifies the new measured reader.
- `quactlize/execution/moe_prepare.cuh` promotes only merged all-SIMT mask5,
  tokens1/2/4/8, gateN1024/K512 or2048, downN2048/K512, softmax+normalization
  without bias. One warp/token computes routing; unused TC arrays and sorting
  are omitted. TC/mixed and other shapes retain their incumbent implementation.
- The private caller accepts policy15, preserves its recipe and logs actual
  experts/channels/topk for subsequent profiling. Existing BF16 compute wiring
  remains. Decode is **not** forced entirely to SIMT.

For BF16 K2048 all-SIMT prepare, M1 decreases7.291875->6.445us and M8
34.8678135->6.2575us in the helper experiment. These are neither a model TPOT
measurement nor justification to substitute BF16 Q4 geometry on F16 timings.

## One box workflow

`tools/run_kpack_q4_model_box.sh` uses the pinned LFS runtime and the private
caller's `.aoneci/scripts/build.sh`. No llama executable is published here.
The runtime refresh reuses all unchanged TC images; it does not compile a
large sweep. The old isolated experimental bundle remains in artifact history.

Delivery: source `eb01ba8`, private caller `0b22fe43e`, artifact `9ee6749`.
Execution SHA256 is `0605f190f8e48fb16a9cb0c7e09845db8344127b172f3b4e76c72d3e1307069c`.
There are54 LFS payload paths; repeated execution paths share one object.
Local validation:158 host/tool tests and33 caller tests pass; the real PPU SDK
compiles both the execution library and updated caller translation unit.

With `MODEL_COMPUTE=bf16 MODEL_ACU=1`, order is:

1. Production Q8 v2 ABI gate12,480; same-image BF16 capability746 and selected
   Q4 BF16 coverage258; selected mixed-chain gate.
2. Whole-model ABBA, PP2048/TG128, one request sequence. Each process excludes
   its first complete PP/TG pass/JIT. Timings come from `llama-batched-bench`,
   not profiler callbacks. `MODEL_PHASES=all` additionally runs paired logits;
   `perf` explicitly does not re-admit model numerics.
3. Asys reference/native full PP2048+TG16 requests, after same-process warmup.
   Original `.asysrep` and SQLite remain on box; all kernel summaries are packed.
4. ACU: up to three dense SIMT recipes, two routed recipes and one prepare
   recipe actually observed in that Asys. The production DSO/C entrypoints are
   reused, but input data and IDs are synthetic, not captured model tensors.
   Five warmups/correctness occur outside profiling; each captured call includes
   its reducer. Reports explicitly use forced-cache replay, not model latency.

Outputs are under `results/benchmark`, `results/trace`, `results/acu`.
Every ACU report is retained with raw counter CSV, execution hash, selected
recipe, shape/compute type, independent oracle and source-trace hashes.
One failed report does not discard other captures. The terminal summary
remainsFAIL if a requested capture is incomplete. No TPOT improvement is
claimed until the returned warmed model timing and full trace are reviewed.
