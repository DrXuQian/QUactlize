# Model-driven tuning (experimental)

Entry: `python3 tools/tune_model.py {plan,prepare,sweep,export}`.
This lives on the experiment branch. It does not install a production policy,
change a kernel, or claim native/model parity.

## Inventory first

```bash
python3 tools/tune_model.py plan --model /exact/model-00001-of-00003.gguf \
  --tp 2 --llama-source /path/to/llama.cpp --output /workspace/model-plan
```

`--models models.json` instead accepts a list of `{ "name": "any-name",
"path": "/exact/first-shard.gguf", "tp": 1 }`. There is no model-name registry.
Alternatively repeat `--model A.gguf --tp 1 --model B.gguf --tp 2`;
one `--tp` applies to all inputs, or supply one for each input in the same order.
Only GGUF headers are read. TP extents come from the caller's C++ split function
on CPU, not dividing every N/K by TP. The existing TP adapter currently covers
Qwen3.5 MoE and DeepSeek4, equal TP1/2/4; a new architecture's split semantics
must be supplied before it can be admitted. Unsupported types/roles stay in
`coverage.json`. This is a shape inventory, not a proof of the actual graph's
fusion, placement or logit-selection behavior.

Default M: 1–8,16,32,64,128,256,512,1024,2048,4096,5120.
The quant types come from actual weights, not a model × every-format product.
Layers, ranks and models share a timing only when the complete workload matches.
Dense is F32 storage/F16 operands/F32 accumulation; routed MoE is BF16 operands.

Three contracts are deliberately separate:

- `projection`: the original dense or indexed projection.
- `concat`: canonical gate/up 2N projection; **does not include SwiGLU**.
- `swiglu`: paired G4/U4 layout, logical N output, fused activation. Its Asys
  carrier is still pending. It is listed, never timed as a plain 2N GEMM.

K-split partial dots cannot apply SwiGLU before the collective. Merged N-split
weights require evidence that both gate/up halves have corresponding segments.

## Current executable boundary

```bash
# New directory; host plan and explicit incumbent/rule inventory first.
python3 tools/tune_model.py prepare --models models.json \
  --forms projection,concat --llama-source /path/to/llama.cpp \
  --output /workspace/model-sweep

# On idle PPU cards. This compiles on Box, then screens and confirms via Asys.
python3 tools/tune_model.py sweep --resume --output /workspace/model-sweep \
  --sdk /usr/local/PPU_SDK --devices 1,2,3,4 --jobs 8 \
  --l2-bytes 67108864 --confirm-idle

# Reparse complete results without launching a kernel.
python3 tools/tune_model.py export --output /workspace/model-sweep
```

The L2 value is an example: verify the actual device's capacity. Default output
is a retained temporary directory; successful or failed runs are not deleted.
Resume keeps the same source, SDK, device assignment and workload. No dirty-tree
ban: HEAD and actual source hashes are both retained. Do not edit sources during
a campaign. The indexed-prefill `--budget-hours` is not a global timeout for the
dense/small-M backends. Candidate counts/resource counts precede execution; no
duration guarantee is inferred from kernel microseconds.

`tools/tune_model_box.sh` wraps the same entry in a committed source archive,
including the pinned **local** actlize commit (no network/submodule update).
For the gate/up concatenation cohort use `--forms concat`; for example:

```bash
GPUS=0,1,2,3 IDLE_CONFIRMED=1 LLAMA_CI_DIR=/path/to/llama.cpp \
  bash tools/tune_model_box.sh --model /models/A.gguf --tp 1 \
  --model /models/B-00001-of-00003.gguf --tp 2 --forms concat
```

Use `plan` or `prepare` as the first argument for a host-only run. Continue with
`RESUME_RUN=/workspace/kpack-model-tune.XXXXXXXX IDLE_CONFIRMED=1 bash tools/tune_model_box.sh`.
Upload the printed `review-*.tgz`: it includes policy/config/component evidence
and diagnostic log tails, not `.so`, caches, model weights or `.asysrep` files.
The original traces remain on Box. `ACTLIZE_ROOT` may point to another local
checkout containing the required pin; its HEAD/worktree is not changed.

The four existing engines are reused, with public Quactlize helpers for MoE.
TC/SIMT, persistent/Split-K, SF and legal Full routes retain their separate
resource and arithmetic contracts. Shared helper samples are counted once per
actual invocation, with each Full arm's reset included. cuBLAS is not timed by
extracting CUDA Graph nodes. No old latency is imported into the new campaign.
Small-M and MoE generators audit historical recipe recall independently of the
rule pool. Dense currently uses its existing M-span geometric basis and separately
measured incumbent/nearby controls; it is not yet a new blind-recall closure of
all dense historical winners. Missing/failing points cannot silently disappear.

## Table and bucket artifact

`policy.json` retains raw confirmed samples, per-component costs, recipes,
incumbent comparisons, commit/source/SDK/device/binary identities and proof
references. The statistic is the arithmetic mean of complete component sums,
not the minimum observed timestamp. Existing raw reports remain on Box.

Exact choices share one host configuration across tested routing profiles; no
runtime routing-concentration readback. A bucket is separated by format,
precision, layout, operator, expert topology, M/N/K bands and aspect. Validation
holds out entire N/K families; a single-family M interpolation cannot claim
cross-N/K coverage. Unvalidated buckets are explicit gaps. Large dense FQ cannot
become a fallback. Lookup returns exact, validated bucket proposal (requiring
resource validation), or miss—never invented latency.

Still pending: paired-epilogue Asys execution, dense blind-recall closure,
portable incremental policy merging and a production C++ table emitter/installer.
Box validation has not been run for this unified coordinator. Current output is
an offline policy artifact, not proof that the production selector consumes it.
