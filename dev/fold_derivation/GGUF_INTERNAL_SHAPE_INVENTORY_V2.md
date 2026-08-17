# GGUF internal shape inventory v2

`tools/gguf_internal_shape_inventory.py` is the immutable bridge from the
resolved real-model set to the ScaleFirst and FullyQuantized sweep
denominators.  It reads GGUF headers only; it neither guesses filenames nor
runs a device kernel.

## Invocation

First bind the checked-in model catalog to exact files (or complete standard
split directories):

```bash
python3 -B tools/resolve_internal_sweep_models.py resolve \
  --bind qwen3.5-35b-a3b-q4_k_m=/workspace/models/qwen35-35b \
  --bind qwen3-32b-q4_k_m=/workspace/models/qwen3-32b \
  --bind qwen3.5-122b-a10b-q4_k_m-tp2=/workspace/models/qwen35-122b \
  --output /workspace/internal-sweep/resolved-models.json

mkdir -p /workspace/internal-sweep/inventory
python3 -B tools/gguf_internal_shape_inventory.py \
  --resolved /workspace/internal-sweep/resolved-models.json \
  --output-dir /workspace/internal-sweep/inventory
```

The output directory must be new or empty.  Existing files and stale
`.partial` files are refused rather than overwritten.

## Authorities and fail-closed boundaries

The resolved catalog owns stable model IDs, the ordered file set, TP policy,
workload axes, and shape-directory templates.  GGUF headers own tensor names,
dimensions, qtypes, architecture, split metadata, expert count, and per-token
top-k.  `*.expert_used_count` means top-k; it is never reused as the number of
active experts.

For a split GGUF the inventory requires all three split keys on every shard,
ordered `split.no = 0..count-1`, exact total tensor count, globally unique
tensor names, and byte-identical metadata values across shards except the
value of `split.no`.  The exact resolved size/hash and ordered fileset hash are
rechecked while loading.

Recognised rank-2 `MUL_MAT` and rank-3 `MUL_MAT_ID` weights become sweep
cells.  A known role with the wrong rank and an ambiguous role rule are hard
errors.  A future unknown rank-2/rank-3 tensor remains visible as
`UNSUPPORTED`/`UNCLASSIFIED_TENSOR_ROLE` (or
`TP_PARTITION_UNKNOWN` under TP), and is also materialised in the model's
`unsupported_tensors.json`; it is never guessed into the denominator.

## Grouped/MoE identity

`E` comes from `{architecture}.expert_count`; top-k comes from
`{architecture}.expert_used_count`.  Active experts, total grouped rows,
`Mmax`, and `row_offsets` come from the versioned
`benchmarks/moe_router_fixture.hpp` fixture
`token-topk-hot16x4-wor-sm64-s44-v1`.  Its seed, source hash, histogram,
offset hash, and fixture ID are emitted.  The catalog's old
`balanced-active`/`one-heavy` placeholders have been removed rather than
assigned invented formulas or published as two measurements.  The catalog
names this pinned authority directly, and `expert_used_count` is exposed only
as `top_k_source`, never as an active-expert count.

The pinned E=256/top-k=8 `(active,Mmax)` ladder is:

```text
tokens 1/2/4/64/2048/4096
active 8/15/30/212/256/256
Mmax   1/2/3/12/239/447
```

## TP, quant blocks, and tied output

GGUF dimensions are fast-first: dense `[K,N]`, grouped `[K,N,E]`.  The
catalog's role policy computes local N/K.  The catalog declares symmetric local
kernels, so rank 0 is the measured representative while TP world and partition
remain part of its stable identity.  A K-sharded quantized tensor is admitted only
when local K is aligned to its GGUF storage block (256 for K-quants); an
unknown storage block or a misaligned local K is explicit `UNSUPPORTED`.
FoldN is always based on local physical N.

When `output.weight` is absent and `token_embd.weight` exists, the physical
embedding remains a `GET_ROWS` tensor and one logical `output.weight` alias is
added as the LM-head `MUL_MAT` source.  A real `output.weight` suppresses the
alias.  This follows llama.cpp's optional/tied-output construction without
double-counting the physical tensor.

## Consumer bridge

Top-level provenance contains the exact `{model_id: fileset_sha256}` map, its
SHA-256 over canonical JSON, and the catalog-owned shape-directory templates.
Each singleton-M `sweep_shapes` row carries the stable model/dedup identity,
source tensor list (plus full source triples), TP world/rank/partition,
dense/grouped route, qtype/group size, grouped routing identity, local
`M/N/K/L`, terminal support state, and shape directory.  Tensor names sharing
the same complete performance identity merge only into `source_tensors`; TP,
qtype, route, grouped identity, or shape changes never alias.
`source_tensors` always names physical GGUF storage.  For a tied LM head it is
`token_embd.weight`; the additive `logical_consumer_tensors` field records
`output.weight`, so offline packing cannot be directed at a nonexistent file
entry while the logical operation remains auditable.

## Local evidence

```bash
python3 -B tools/gguf_internal_shape_inventory.py --self-test
python3 -B ci/check_gguf_internal_shape_inventory.py
```

The synthetic two-shard fixture covers dense and fused grouped roles, TP=2
rank publication, an aligned and a planted-misaligned K split, tied and real
LM heads, versioned routing, unknown rank-2/rank-3 visibility, unknown qtype
`gUNKNOWN`, metadata/count/order/name/hash failures, and dedup identity.  The
bridge check serialises through JSON before feeding the real internal-sweep
consumer, so Python-only agreement is not evidence.
