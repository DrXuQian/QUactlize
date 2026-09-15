# New-format SIMT timing sweep

Run from `develop`; the two PPU libraries are prebuilt Git LFS payloads.
The box does not compile or JIT. Only idle GPUs should be selected.

```bash
(
    set -e
    cd /sim/eec/shared/junfu.qx/quactlize
    GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only origin develop
    PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
    DEVICES="0 1 2 3 4 5 6 7" \
    L2_BYTES=67108864 \
    bash tools/run_simt_formats_box.sh
)
```

The 64 MiB override is for the previously identified PPU-ZW810; it is not
an inferred capacity for other devices. Runtime identity is probed with a
kernel from the candidate image and the SDK PCI query. Duplicate physical
GPUs are rejected. Different SDK/compiler file paths do not force a rebuild;
the actual runtime wrapper hash is recorded and bound to resume identity.

## Scope

Five formats: Q2_K, Q3_K, Q5_K, Q6_K and Q8_0. For each format and token
count 1 through 8, the following weight families are measured:

| Family | N | K | Experts | Input indexing |
| --- | ---: | ---: | ---: | --- |
| Dense small | 512 | 2048 | 1 | One row per token |
| Dense medium | 1024 | 5120 | 1 | One row per token |
| Dense large | 5120 | 8192 | 1 | One row per token |
| MoE gate/up | 1024 | 2048 | 256 | Top-8, shared A per token |
| MoE down | 2048 | 512 | 256 | Top-8, independent A per slot |

This is **200 contexts**, not the full historical shape registry. Set
`TOKENS="1"` for a 25-context first pass. `QTYPES="13 14"` selects Q5/Q6.
`DEVICES="0"` runs sequentially on one GPU. The worker ETA uses completed
case wall times, including fixture setup, and is advisory.

Each context screens all 240 new configurations (120 for Q8) and the old
scalar/pair control inventory. Split factors are 1/2/4/8. The top two per
implementation receive four alternating rounds of eleven samples. The
old control is **not** the optimized Q4 reader, Xplane, raw reference, or a
TC winner. Q4 is intentionally excluded. No production heuristic changes.

All calls use F32 input/output storage, F16 register-rounded activations
and F32 accumulation. New dequantization is F32 group-affine; old readers
reconstruct individual weights in F16. Both are checked against independent
official-GGUF dots. These bounded, F16-exact inputs do **not** validate
wide-range model activations or fix the separate BF16 requirement.

Each recipe must pass numerical/guard checks before timing; each context
also checks changing graph inputs and a zero-A negative. Graph upload and
first replay are excluded. Split-K timings include the reducer. Indexed
SIMT consumes IDs directly, without standalone gather/scatter. Routing IDs
are supplied, so this is not a full MoE-chain or model timing.

The weight ring covers at least 2.25 times L2 using the union of **accessed**
expert slices. Unselected expert allocation is not counted as cold traffic.
Only complete ring traversals are measured. Summary bandwidth is a weight
byte model at 2700 GB/s, not a measured ACU DRAM counter. No profiler is run
in this timing cohort.

## Results and retry

The script prints `results=/workspace/simt-formats-ppu.XXXXXX.results.tgz`.
Return that archive: it includes the summary, raw samples, recipe choices,
source/image/runtime identity, launch commands, logs and per-case status.

Each case runs in a fresh process. A failed case does not stop other cases.
To retry failures, keep the same source, bundle, GPU list and options, and
add `RESUME_RUN=/workspace/simt-formats-ppu.XXXXXX` to the command. Complete
valid cases are reused; failed attempts remain available for diagnosis.
