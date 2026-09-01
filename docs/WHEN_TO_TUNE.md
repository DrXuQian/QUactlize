# When to tune the tactic, and where the answer lives

The sweep decides WHICH config wins. This is the separate question of WHEN that decision is made and what
carries it. TRT-LLM is the reference because it is the only one of these systems that treats tactic selection
as a build artifact rather than a runtime search, and because we already borrowed its neighbour (machete /
fpA_intB's "compile every config in, search at run time, cache by shape").

## CORRECTION, 2026-08-05: the section below describes a TRT-LLM that no longer exists

Everything from here to "Where we are" was written from a **standalone extract**. The real repository was then
cloned and read (NVIDIA/TensorRT-LLM main), and the TensorRT-plugin path that owned `GemmPluginProfiler` **is
gone** — there is no `gemmPluginProfiler.{h,cpp}` in the tree at all. It is kept below because the shape it
describes is still the right one and it is what our design was built against, but it is history, not the
reference.

**What main does now** (`tensorrt_llm/_torch/autotuner.py`) is *closer to what we already built*:

  * tuning is an **explicit mode**, `with autotune(cache_path=...)`, which profiles and then `save_cache` /
    `load_cache` by path. Nothing tunes implicitly.
  * at inference (`is_tuning_mode == False`) it is a **pure lookup**: `search_cache` hit → that tactic; miss →
    `fallback_entry()` = `(runner 0, tactic -1)` plus a `warning_once`. **It never profiles and never traverses
    candidates at inference.**
  * `tactic == -1` is documented as "the fallback kernel **which should be able to implement any shapes**",
    needed both for a cache miss and so that "the autotuning process [is] an optional process, such that user
    can opt out".
  * the source comment at the miss branch reads **"Expect no cache miss in inference."**

Two consequences for us, both simplifying:

  * **Option A below is what the reference now does.** An explicit step producing a portable artifact, with
    load-and-go preserved because the artifact is optional. That was the option this document could not choose
    between; the reference has since chosen it.
  * **There is no heuristic on that path.** `cutlass_heuristic.cpp`'s `estimate_best_config_from_occupancies`
    (wave quantisation) is still in the tree but belongs to the C++ kernel-selection side. Our floor should be
    what theirs is — the compiled default plus a deduplicated warning — not a model. INBOX 054 withdraws the
    heuristic that 049 asked for on this basis.

## What TRT-LLM used to do (TensorRT plugin path, removed)

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
| offline pack (`quactlize-pack-gguf`) | the ARRANGEMENT: which bytes are resident | done |
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

**The section below overstates its case, and the real TRT-LLM shows where.** It argues warmup is
*structurally* the wrong place. It is not: TRT-LLM's own engine warmup is exactly where its tactics are filled.
`ModelEngine.warmup()` (`_torch/pyexecutor/model_engine.py:1151`) runs, in this order:

```
AutoTuner.get()                      # in EAGER context: "Otherwise the first get() can happen inside
                                     # torch.compile tracing and trigger non-traceable code
                                     # (time.time(), torch.cuda.*) in the cache."
_run_attention_warmup(...)
with self.no_cuda_graph():           # "Currently graph has not been captured, disable cuda graph"
    _general_warmup(...)             # specialise torch.compile across the key input shapes first
    MoERunner.clear_all_workspaces() # "so the autotuner can reclaim the memory"
    gc.collect(); torch.cuda.empty_cache()   # "autotuner may use additional memory"
_run_autotuner_warmup(...)           # <- the tactics are filled HERE, with context-only requests
                                     #    CUDA graph capture happens after
```

Three orderings are load-bearing and each is justified in the source: the tuning happens BEFORE capture (so a
search never runs inside one — the concern below is right, their structure is the answer to it), AFTER the
general warmup (so compiled graphs are already specialised), and with memory explicitly released for it.

**What makes warmup viable for them and not for us is not the seconds — it is `cache_path`.** Their tuning runs
at warmup and `save_cache`s the result; the next process `load_cache`s it. The cost is paid once per (model,
machine), not once per process. The objection below about paying repeatedly is answered by persistence, not by
moving the work.

So the honest conclusion is narrower than what the section originally claimed:

  * "warmup is the wrong place" — WRONG in general. It is where the reference does it.
  * "warmup is the wrong place FOR US" — still right, but for one reason only: we have no persistence layer
    at that point in llama.cpp, so a warmup tune would be re-paid every process, and `--no-warmup` exists and
    gets used. Give llama.cpp a cache file and the objection dissolves.

Our shape therefore stays: tune offline beside the pack, load the table at startup, fall back to the compiled
default with a warning when it is absent. That is TRT-LLM's flow with the *tuning* half already done.

The original argument follows, with its overreach now marked.

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

**2026-08-05: A is now also what the reference does**, and for a reason the sweep cannot overturn. TRT-LLM main's
`autotune(cache_path=...)` is exactly shape A -- an explicit mode, an artifact, a path -- and the property that
makes it work is the one B gives up: *the tuning process is optional and the fallback implements any shape*, so
a user who never tunes still runs. B makes the first occurrence of every bucket pay, which is a cost that
appears in the middle of somebody's first conversation and cannot be opted out of.

The sweep still decides the table's SIZE, and the size still decides whether A is a small ask or a large one.
It no longer decides the SHAPE.

## What this does not decide

Whether the tactic search itself should ever run at inference time. It should not, and TRT-LLM agrees, but that
is a consequence of having the table rather than an argument for building it.
