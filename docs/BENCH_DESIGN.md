# Benchmark infrastructure: what is crude, and the shape to move to

Written 2026-08-04, after a day in which every one of the defects below was hit for real rather than imagined.
The list comes first because the design is only justified by it.

## What is actually wrong

1. **Selection logic lives in C++, inside the bench, untested.** I moved the MoE bench from
   `if (u < b.us)` to median-over-interleaved-repeats with a `[min,max]` band and tie reporting. The fix is
   right and its *placement* is wrong: it sits in `lowbit_moe_bench.hpp`, has no unit test, and the dense bench
   will need a second copy. That is the same two-copy defect I removed from `test_lowbit_dense_bench.cu` this
   morning, recreated one level up, by me, hours later.

2. **Two benches, two unrelated ways to say "which configs to compile".** MoE: CMake generates one `.cu` per
   shape and takes Cartesian axes through `MOE_*_LIST` env vars. Dense: one `.cu` with a table, now emitted by
   `emit_tactic_configs.cpp`. Neither can express the other's pruned set, and the MoE side cannot express an
   arbitrary set at all -- only a product.

   **HALF-FIXED 2026-08-04.** The generator now takes `--space=dense|grouped` and emits the same table for
   either operator, so one pruning policy serves both and an arbitrary set is expressible for the MoE side.
   What remains is the *consumer*: the MoE bench still builds from `MOE_*_LIST`, so nothing yet reads
   `lowbit_grouped_configs.inc`. Until it does, the two operators still search sets defined two ways -- they
   are merely now provably the same set.

3. **Neither bench had a compile check.** Both syntax gates were added today. The gap is why a selection
   rewrite was committed without ever being compiled, and it was invisible because `-k lowbit` matched nothing
   and reported `0/0 passed`.

4. **Output is prose to be grepped.** `grep '^  WINNER m='`. Every downstream question -- did the winner move
   with M, do dense and grouped agree, is this guard inside the band -- requires re-parsing a human table, and
   a changed format silently breaks whatever was parsing it.

5. **Fixtures are integers in `argv` and modes inside `main`.** `"$BIN" 256 128 512 2048 32 2` is
   `L Rows N K gs mode`. Nothing names them, nothing records which distribution ran, and the row generator for
   mode 2 is inline in `main`. The one property that decides whether the MoE result means anything -- per-expert
   Mmax -- is not printed.

6. **A log does not describe its own run.** Not the config set, not the pruning policy that produced it, not
   the fixture seed, not the library's build defines. Two logs that disagree cannot be diagnosed.

## The shape to move to

### A. Samples out, selection out of C++

The bench's job becomes: run a candidate, emit a sample. One line per `(fixture, config, pass)`:

```json
{"fixture":"qwen35-a3b-expert-gate","n":512,"k":2048,"rows":128,"mmax":420,"dist":"skew-h8-v1",
 "schema":"i4","tm":64,"tn":128,"tk":64,"wm":64,"wn":64,"st":3,"pass":2,"us":233.76}
```

Everything above `us` is *what was run*; `us` is the only measurement. Median, band, ties, cross-fixture
comparison and guard expansion all move to an analyser in `benchmarks/analyse.py`, which can then be tested with
planted samples -- a leader with an overlapping guard, a bimodal candidate, a single-pass file that must refuse
to rank. None of that is testable while it lives in a `.cu`.

This also makes the 13% cross-run spread visible rather than absorbed: the samples are in the file, so a later
question ("was that separation real?") is answerable without re-running.

### B. One config-table generator for both operators

`emit_tactic_configs.cpp` reads `ppu_tactic_space.hpp` and emits an X-macro list. **The generator half is done**
(2026-08-04): `--space=dense|grouped` emits for either, with a per-space macro prefix so a grouped table
included where a dense one is expected is an undefined macro rather than a silent substitution. What remains is
letting the MoE side consume the list -- CMake's job shrinking from "loop over four axes and write 180 files" to
"compile the generated list, N configs per translation unit", with `MOE_*_LIST` becoming a filter over the
generated set rather than a second, weaker way to define it.

**AND `--space=compare`, which is the part worth keeping even after that lands.** The header deliberately keeps
`DenseSpace` and `GroupedSpace` as separate wrappers over one implementation so future divergence stays visible,
and explicitly says the emitter should ask each and have a comparator check them. That comparator did not exist,
so "the two are identical" was a property of the source that nobody re-established. It now walks the grid asking
both spaces every predicate the emitter uses -- kernel, sweep, static-sweep, per-stage topology, compact-A
support and the 1/2/4 compact capacities -- and exits non-zero on any disagreement.

Verified to FIRE, not merely to pass: with the four-warp minimum raised to eight in `GroupedSpace` only, it
reported 147 disagreements and rc=1; unmodified it reports 0 and rc=0. A comparator that has only ever agreed is
indistinguishable from one that compares nothing.

The `static_assert` tying a generated table to its binary's `(bits, TileK)` is the pattern to keep: a stale
table becomes a compile error rather than a sweep over tactics the binary cannot select.

### C. Fixtures as named objects

```cpp
struct Fixture {
  char const* name;          // "qwen35-a3b-expert-gate"
  int n, k, gs;
  char const* dist;          // "skew-h8-v1", "uniform-multinomial-v1", "route-topk-v1"
  int experts, topk, tokens; // rows/expert follows; Mmax follows from the generator
  uint32_t seed;
};
```

Two properties matter more than the tidiness. **The distribution is named and versioned**, so "ragged" stops
being a word and becomes something reproducible -- and changing the generator forces a new name rather than
silently reinterpreting old logs. **Mmax is computed and printed**, because for MoE it decides which compact-A
capacities are even reachable, and a fixture whose Mmax is 1 measures capacity 1 however the build was labelled.

Shapes come from `benchmarks/workloads.py`, which already holds the three target models and derives the
per-card projections; the fixture list should be generated from it rather than transcribed into argv.

### D. Every run states what it was

One header line, machine-readable, carrying: the config-set hash, the pruning policy version, the fixture
record, the library's build defines (`PPU_PACKED_SCALE`, `PPU_PACKED_FORMAT`, `QUANT`, `BENCH_TSK`), and the
repeat count. The existing bench already does some of this in prose -- it prints `MOE_ONLY`/`MOE_ACU` warnings
and refuses to say "fastest" for single-shot runs, which is the right instinct. This makes it structured so an
analyser can refuse to merge two incompatible runs instead of averaging them.

## Order

1. **Emit samples alongside the current output** (`BENCH_JSONL=path`). Purely additive, nothing to break.
2. **Write `analyse.py` with planted-sample tests**, and re-derive today's MoE verdict through it. Two paths
   agreeing is the check that the move is faithful.
3. **Delete the C++ selection** once the analyser reproduces it. Not before -- deleting first would leave a
   window with no verdict at all.
4. **Generalise the emitter to GroupedSpace**, and reduce the CMake generator to consuming it.
5. **Fixtures from `workloads.py`**, replacing positional argv.

Steps 1-3 are the ones that matter; 4 and 5 are tidying that becomes easy once results are structured.

## What this deliberately does not do

No new runner, no config file format, no abstraction over the two benches' kernels. The benches stay two
programs that know their own kernels; what is unified is *what they emit* and *who decides the winner*. An
abstraction over the kernels themselves would be the third mechanism in a document whose whole complaint is
that there are two.
