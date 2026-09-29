# Prepare-only integration

The supplied patch is integrated on `dev/gemv-model-tuning`. Its two production
headers are unchanged from the upload (SHA256 of the patch:
`6a147d96e69bb42a77c8b16f5f30446d78c37f34a5e0ad59a690da8de7d7631c`).
This is a development delivery; the reported20% gain is not a new model result.

## Scope

Stable register-local top8 sorting replaces repeated per-lane scans. The
single-warp path retains selected IDs in registers, coalesces their global
publication and checks duplicates with warp match. The multi-warp all-SIMT
path publishes router-independent identity maps/status early and omits the
redundant duplicate scan and CTA barriers. No weight data is copied here.

Production admission is unchanged: merged all-SIMT, E256/top8, tokens1/2/4/8,
gate N1024 K512/2048 and down N2048 K512, ordinary normalized softmax without
bias. Delayed-softmax, sigmoid, mixed/TC and other geometry keep the incumbent
production entry. The new sorted helper is not a general admission of those
router modes. M1's permitted logits/weights alias still snapshots every logit
before stores; multi-token overlap remains rejected by the existing binding.

## Local validation and device boundary

-268 related host and runner tests pass. The actual router headers execute under a host
 warp-collective model:40 ID/weight-bit and alias comparisons, including ties,
 signed zero, NaNs, infinities and a lane supplying all eight winners. An
 unstable-tie mutation is rejected. This is not a PPU numerical verdict.
-PPU execution build completed in179.327s. The existing seven GEMV readers
 retain their admitted native opcode counts/resources and zero stack.
-The package retains all seven TC module images and the pack/prefill payloads.
 The paired library is rebuilt against the matching source receipt; its two
 measured paired readers retain the same inspected instructions/resources.
 Public ABI, canonical/paired formats, selectors and caller are unchanged.

Runtime artifact: `artifacts/kpack-model-prepare-v1`,
`87b996559ff64d0fd17123ab8402617075cf613a`. Manifest SHA256:
`0bcf7ebf01ed534ea47fd73ca2dc384ac998ca4ef07f970cb8186267190aa624`.
59 binary payloads remain byte-identical; four unique new binaries total
44,992,056 bytes (the execution image appears twice but LFS deduplicates it).
No llama binaries are delivered. Source integration commit:
`3c805f1969a7af4ce19da51b28bbf386af559755`.

The router gate additionally has56 special-value/alias contexts, three actual
device implementations each: unchanged shipping router, new ordinary ID store
and new coalesced ID store. The existing360-context alias gate remains enabled.
The prepare-only before/after comparison contains16 cases: tokens1/2/4/8,
K512/2048 and F16/BF16. Both arms run four changed-input graph replays before
timing; each has60 measurements of64-launch graph replays after warmup.
The control uses exact pre-patch headers at
`39e9c6fdb82611ecd7c05a0b0511e0ae4f41f269`, with only include paths and
symbol namespaces renamed. It is not the older multi-CTA preparation path.

The model runner then checks the paired chain and measures the original
PP2048/TG128 single-request ABBA protocol, excluding the first complete pass
in each process. Asys captures the second request separately. New device
numerics and model performance remain pending until those results return.
Use `run_kpack_q4_model_box.sh`, not a performance-only resume of old gates:
this runtime changed and requires its new prepare checks.

## Unpatched returned baseline

`kpack-q4-resume.7l1wtue2.results.tgz`, SHA256
`528e996965ffc8215f4342ec0ff5ceb104b1499823b0c17bf9a061037156901b`:

| Metric | Native reference | K-pack before this patch |
|---|---:|---:|
| Prefill us/token |143.634277|109.347168|
| Decode ms/token |7.713883|6.609484|

Four unprofiled measured samples per arm; first passes excluded. Component
gate hashes match the reused original results. Model numerical metrics were
not retested. The exported Asys trace contains600 M1 prepares averaging
7.884423us, and both optimized paired kernels actually execute. The full
`.asysrep` is on the box, not in the uploaded summary archive.

At40 prepares/token, retaining a20% prepare reduction would save approximately
0.063ms/token. This is a conditional component estimate, not a measured TPOT
for the new patch. Do not credit this patch for the baseline's14.32% decode
latency advantage over the native reference.

## Returned prepare integration

`kpack-q4-model.TUQKwW.results.tgz`, SHA256
`6df2122962b29df239869d5dffd14ae3a39c1625de19b8a97f56fcec8a50bfbb`,
completed at source `ea438cca7ef58a0afb7b218d90a1b6d6ed66591e` with the
manifest above and unchanged caller `c3d9cdaa4bb1b3f111397edb3ab05d3935ac81b0`.
Prepare16/16, router edges56, alias360, BF16746/746, paired and reader
integration gates passed. This perf-only run did not rerun model perplexity.

| Metric | Native reference | K-pack with prepare patch |
|---|---:|---:|
| Prefill us/token |143.790039|109.197998|
| Total PP2048 ms |294.482000|223.637500|
| Decode ms/token |7.708527|6.537199|

The unprofiled ABBA medians exclude each process's first complete pass.
The separate second-request Asys trace contains600 M1 prepares averaging
5.868545us, versus7.884423us in the previous trace. Its modeled saving is
0.080635ms/token at40 layers; the historical model comparison saves
0.072285ms/token. This cross-run comparison is not a same-session patch A/B.
The isolated prepare A/B records26.7--29.8% reductions across16 cases;
its receipt still requires an external competing-load audit for strict
performance admission. Do not rewrite the immutable build receipt.

## Other-model coverage and the 122B topology correction

The user requires two GPUs for Qwen3.5-122B-A10B. The earlier single-device
122B plan and combined continuation command are withdrawn.
`tools/kpack_batched_other_int4_2048.json` now preserves the two-device
tensor split from the original catalog, at PP2048/TG128/NPL1. K-pack tensor
split is not admitted: the continuation rejects this plan instead of silently
measuring a single-device, offloaded or native-only replacement. Layer split
is not a substitute: the user explicitly confirmed TP2 (tensor split).
122B coverage remains blocked on K-pack TP2 integration and validation.
Qwen3-32B remains a separate single-device dense workload.

The extension verifies the unchanged runtime/caller hashes and completed
component gates, then writes a new result directory. No library or caller
build, bundle fetch or component sweep is launched. First-use JIT for new
selected shapes is still possible and is excluded by a full warmup pass.
All GGUF shard headers are inventoried; payloads are not read for this step.
Pure dense models are not required to emit nonexistent MoE fusion kernels;
every actually selected paired operation still needs device evidence.
Benchmark and Asys failures preserve independent model/phase results.

Upload the printed `.results.tgz`. Full Asys reports remain under
`<new-run>/results/trace/<model>/{reference,native}/proof.asysrep`.
This adds performance/selection coverage only, not a new whole-model
numerical admission or a claim that the35B-specific exact readers apply
to every shape in32B/122B. Dense remains F16 compute with F32 endpoints;
MoE retains BF16 compute.

## New-machine NCP host-header build repair

The new box stopped in DeepGEMM's host `compiler.hpp` while compiling NCP:
`hggc_runtime_api.h` was not found. SDK2.1.1 installs its native headers at
`targets/x86_64-linux/include`; `envsetup.sh` does not add that directory to
the host C++ search path. The CI hook had added the native wrapper library
but not its matching headers.

Caller `5ebe4b878b3d39adbad0357e2d676f80d61a593a` adds the selected compiler
SDK's native include directory to `ncp_moe` only. Both SDK layouts are
supported, required headers fail closed, and other targets keep their flags.
All37 caller host tests pass, including real C++ compile/link fixtures,
missing-header negatives and preservation of unrelated object files.
Separately, the official2.1.1 headers pass host C++ syntax checking with the
explicit include path and fail without it. This is not a full NCP/device run.
The Quactlize runtime artifact, model policy and kernel payloads are unchanged.

Reuse the partial NCP checkout through `NCP_CI_DIR` and let the model runner
fetch the new pinned caller. Leave `LLAMA_CI_DIR` and `LLAMA_CI_BUILD_DIR`
unset for a fresh caller build. Do not erase the failed CI directory or reuse
objects from the old SDK2.0 environment.
