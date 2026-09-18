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
