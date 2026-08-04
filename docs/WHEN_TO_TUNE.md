# When to tune the tactic, and where the answer lives

The sweep decides WHICH config wins. This is the separate question of WHEN that decision is made and what
carries it. TRT-LLM is the reference because it is the only one of these systems that treats tactic selection
as a build artifact rather than a runtime search, and because we already borrowed its neighbour (machete /
fpA_intB's "compile every config in, search at run time, cache by shape").

## What TRT-LLM actually does

`GemmPluginProfiler`, at **engine build time**:

  * enumerates the plugin's compiled tactics;
  * profiles each over a set of **M buckets** -- not every M, and not one M;
  * stores `map<GemmIdCore{n, k, dtype}, vector<tactic>>`, one entry per bucket;
  * serialises that map into the engine plan.

At run time `getBestConfig(m, gemmId)` is a lookup: find the bucket containing `m`, return its tactic. There is
no search, no timing, no first-call spike. The cost was paid once, by `trtllm-build`.

Two properties are worth naming because they are the reason it works:

  * **the profile is an artifact, not a cache.** A cache can miss; an artifact cannot. Every shape the engine
    can see was profiled, because the engine's shape range is declared up front.
  * **M is bucketed, and the buckets are part of the contract.** They are powers of two by convention, chosen
    before any measurement.

## Where we are

`test_lowbit_dense_bench` already has the machete-shaped half:

```
--search_configs      time every compiled config, print a table, pick a winner
--save_tactic FILE    append  "<m>,<n>,<k>,<g>|config=<name>,<tf>"
--tactic FILE         look up that exact key
```

`tactic_key` is `m,n,k,g` -- **the exact M**. That is the difference that matters. In serving, M changes every
step; an exact-M key misses almost always, and a miss has nowhere to go but a search or a default. TRT-LLM does
not have this problem because it never keys on an exact M.

The offline packer already writes a per-tensor manifest carrying `PlacedArrangement`. It carries no tactic.

## The gap, stated as three moments

| moment | what belongs there | today |
|---|---|---|
| offline pack (`tools/pack_gguf.py`) | the ARRANGEMENT: which bytes are resident | done |
| build / pack time | the TACTIC TABLE: `(n, k, gs, M-bucket) -> config` | **missing** |
| first inference | nothing | -- |
| every call | a lookup | a cache that usually misses |

The middle row is the whole of this document. It is TRT-LLM's row, and we do not have it.

## The one place we should NOT copy TRT-LLM

**Their bucket boundaries are assumed; ours can be measured.**

TRT-LLM picks powers of two before profiling because it has no reason to prefer anything else. We are about to
have the reason: INBOX 025's question is exactly "does the winner move with M", and `benchmarks/fixtures.py`
already emits the six token counts the user fixed -- 1, 2, 4, 64, 2048, 4096 -- against real Qwen shapes.

So the honest order is:

1. run the sweep across those M values on one shape;
2. read WHERE the winner changes. If it never changes, there are no buckets and the table is keyed on
   `(n, k, gs)` alone -- a strictly better artifact than TRT-LLM's, and smaller;
3. if it does change, the bucket boundaries are the M values where it changed, not the powers of two that
   happen to bracket them.

Doing step 3 before steps 1-2 would bake in an assumption we are one measurement away from replacing. That is
the same error as an enumerated guard set: a number nobody checked, in a place nobody revisits.

## What the artifact should carry

The manifest already records the arrangement per tensor. The tactic belongs beside it, keyed the same way:

```json
{"name": "blk.0.ffn_gate.weight", "n": 512, "k": 2048,
 "arrangement": {"bits": 4, "tile_k": 64, "high_bits": 0},
 "tactic": [{"m_max": 4, "config": "..."}, {"m_max": null, "config": "..."}]}
```

`m_max: null` is the open-ended last bucket. One entry means M does not move the winner, and the reader needs
no bucket logic at all.

TWO PROPERTIES TO HOLD, both learned the hard way this week:

  * **the tactic must name the config set it was chosen from.** A tactic picked from 17 hand-written rows and a
    tactic picked from 227 generated ones are not comparable, and a manifest that records only the winner
    cannot tell them apart. Store the generator's `(bits, tile_k, stages)` header line with it.
  * **a tactic the binary cannot select is worse than no tactic.** `LOWBIT_DENSE_DISPATCH` exits(1) on an
    unknown config name. If the manifest outlives a table regeneration, the lookup must degrade to the
    compiled default and say so, not abort.

## Is llama.cpp's warmup the place?

No, and the reason is structural rather than about the seconds.

WHAT WARMUP IS. `--warmup` is documented in common/arg.cpp:1727 as "perform warmup with an empty run" -- one
eval, to fault pages in. It is on by default and users turn it off when they want load-to-ready to be fast.

WHAT A TUNE COSTS. 227 configs x 5 repeats x 20 iterations, on a prefill-sized dense shape, is 2.6-5.2 s per
(shape, M) at 300 and 150 TF/s respectively. A model has roughly 7 distinct (n, k):

    M does not move the winner   ->  7 sweeps      ->  18-36 s
    M does move it               ->  7 x 6 sweeps  ->  110-220 s

THE STRUCTURAL OBJECTION, which holds at either number. Put tuning in warmup and it lives behind a flag whose
whole purpose is being turned off. `--no-warmup` exists because people want fast startup; the moment startup
grows by half a minute they will pass it, and then tuning is off by default in practice while the code still
says it happens. A step people disable to go faster is not where a correctness-adjacent decision belongs.

THE SECOND OBJECTION. Warmup runs per PROCESS. The tune's result is a property of (weights, hardware) and does
not change between runs, so paying it every process start is paying repeatedly for an answer that was already
true.

## So where

TRT-LLM's answer is `trtllm-build`, and llama.cpp has no equivalent -- GGUF is loaded directly, which is most of
its appeal. That leaves three shapes, and which is right depends on a measurement we do not yet have:

  A. AN EXPLICIT STEP that writes a tactic file, run once per (model, machine). The honest analogue of
     `trtllm-build`, and the only option where nothing at inference time can be slow or absent. Costs the user
     a command they must know to run, and an artifact they must keep next to the model.
  B. LAZY PER (shape, M-bucket), cached to a file beside the model. Load-and-go is preserved; the first
     occurrence of each bucket pays. With one bucket per shape that is ~7 pauses of 3-5 s across the first
     conversation, which is noticeable but bounded and never repeats.
  C. SHIP A TABLE keyed on (n, k, gs, bucket) with the library, tuned by us on reference hardware. Zero cost to
     the user and wrong the moment their hardware differs from ours.

THE MEASUREMENT THAT PICKS ONE is the sweep already queued: if the winner does not move with M, the whole table
is 7 entries and B costs 7 short pauses -- comfortably the best trade. If it does move, B becomes 42 pauses
spread over a session, which is worse than asking for A once.

Deciding between them before that sweep would be choosing the ergonomics before knowing the size, which is the
same error as picking bucket boundaries before measuring where the winner changes.

## What this does not decide

Whether the tactic search itself should ever run at inference time. It should not, and TRT-LLM agrees, but that
is a consequence of having the table rather than an argument for building it.
