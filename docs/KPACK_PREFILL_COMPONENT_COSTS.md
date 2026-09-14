# Measured prefill component costs, 2026-09-14

The first composition is complete: **44 cost points**, comprising 20 dense
cuBLAS cells and 24 grouped DeepGEMM cells, Q4/Q5 and tokens 2048/4096.
No GPU work, JIT or new library build was required to compose the results.
No production route is selected by this table.

- [Full table](measurements/kpack_prefill_components_20260914/summary.tsv)
- [Component values, configurations and source receipts](measurements/kpack_prefill_components_20260914/result.json)
- [Remaining matched GEMM denominator](measurements/kpack_prefill_components_20260914/missing-gemm-plan.json)

## What is added, and what is not

For each call, with no cross-call weight or scale-workspace reuse assumed:

| Candidate | Cost | Present evidence |
|---|---|---|
| FQ | FQ GEMM, including any Split-K completion | Matched GEMM time missing |
| SF | Scale/zero-only expansion + SF GEMM | Expansion measured; matched GEMM missing |
| Full BF16 | Full weight expansion + BF16 GEMM provider call | Both components measured for all 44 points |

The last column must not be interpreted as a win for full BF16 merely
because it is the only complete candidate. FQ/SF costs are null, not zero,
and `selection=null`. SF never expands weight codes. Full expansion writes
BF16 `[E,N,K]`; the provider consumes those weights in its own separate
measurement. FP32 affine and BF16 RNE remain unchanged.

The grouped token count is not GEMM M: top8 makes tokens 2048/4096 become
16384/32768 total sorted rows. These particular routers use all 256 experts.
Sparse routes cannot borrow an E256 expansion time multiplied by active/E;
they need actual active-expert-only expansion measurements and an entry
that implements that operation.

## Representative sums

All times below are microseconds at tokens 2048. New dequant choices come
from v4; provider GEMM measurements are reused unchanged. The old sum uses
the original v2 dequant receipt, not the newer run's c4/c5 control timing.

| Format / N x K / E | New full dequant | BF16 GEMM | Previous sum | New sum | Sum reduction |
|---|---:|---:|---:|---:|---:|
| Q4 / 1024 x 5120 / 1 | 11.741 | 63.129 | 79.666 | 74.871 | 6.02% |
| Q4 / 5120 x 8192 / 1 | 83.926 | 550.800 | 639.371 | 634.726 | 0.73% |
| Q4 / 512 x 2048 / 256 | 376.810 | 379.560 | 859.710 | 756.370 | 12.02% |
| Q4 / 1024 x 3072 / 256 | 1098.440 | 1116.700 | 2545.930 | 2215.140 | 12.99% |
| Q5 / 2048 x 512 / 256 | 466.820 | 334.230 | 877.460 | 801.050 | 8.71% |
| Q5 / 1024 x 3072 / 256 | 1354.260 | 1111.590 | 2723.310 | 2465.850 | 9.45% |

Across both token counts, Q4 grouped sums decline 11.25--13.04%, and Q5
grouped sums decline 7.65--9.45%. Dense has a smaller expansion share:
Q4 sum reductions are 0.50--6.02%, Q5 0.05--6.40%. These are arithmetic
changes in an isolated-component model, not measured model speedups;
especially tiny differences should not be claimed as meaningful wins.

The TSV also supplies **derived break-even thresholds**, not new timings.
For Q4 N512/K2048/E256 at tokens2048:

- Full-BF16 sum: 756.370 us.
- FQ must be below 756.370 us to win within this component scope.
- SF metadata: 28.211 us, so SF GEMM must be below 728.159 us to win.

No claim is made about the BF16 consumer's cache state immediately after
dequantization. Input/output adapters, activation precision conversion and
external routing/gather are not included. The installed DeepGEMM provider
call includes its own internal block directory. JIT/first launch, graph
upload and warmup are excluded; they are not charged again in composition.

## Evidence binding and reproduction

The composition checks every input file against its measurement receipt,
then checks provider-to-original-dequant binding, same physical PPU/runtime,
same canonical input hashes and identical independent BF16 output hashes.
The original SF measurement is retained from the same packed units.
Different dequant binary hashes are expected: equal validated output bytes,
not equal implementation hashes, permit reusing a GEMM-only measurement.
Valid cuBLAS/DeepGEMM cells survive unrelated failures in their parent run;
missing cells or duplicate provider records never silently become timings.

Source archives:

| Input | Archive SHA256 |
|---|---|
| v2 dequant + cuBLAS: `kpack-prefill-cost.9jSpJD.results.tgz` | `bac13ed261e20d1c8cfbec292642aeb015fd939d73622ac7fd77d08189a03c87` |
| DeepGEMM: `kpack-deepgemm-python.XOCxcL.results.tgz` | `c31bc734a57459c06a03777d1da409cb5128a2481a937a85349606a8fc1da646` |
| v4 full dequant: `kpack-full-packed-ppu.NU9nBv.results.tgz` | `9904a22f22a194d12fef65021fb9cdcabf6f0d87fb9bd1981d7afdc27cd81648` |

With those original result folders present, run on a CPU host (NumPy/GGUF
and the existing repository environment; no PPU SDK or device required):

```bash
python3 tools/compose_kpack_prefill_costs.py \
  --reference-dequant /workspace/kpack-prefill-cost.9jSpJD/results/dequant \
  --full-dequant /workspace/kpack-full-packed-ppu.NU9nBv/results \
  --bf16-results /workspace/kpack-prefill-cost.9jSpJD/results/bf16 \
  --bf16-results /workspace/kpack-deepgemm-python.XOCxcL/results/bf16 \
  --output /workspace/kpack-prefill-components
```

The output directory must be new. Inputs are never modified. The tool
does not require the box's installed cuBLAS/DeepGEMM paths to exist locally;
their recorded implementation/image identities remain in the source chain.

## Remaining comparison, not another BF16 sweep

An executable **selected-heuristic** measurement is now provided in
[Selected prefill calls and standalone Split-K reduction](KPACK_PREFILL_SELECTED_MEASUREMENT.md).
It uses the actual production dispatcher (88 requests, currently all S1,
12 parent modules). It is not an online tuner or a new exhaustive search.
An independent 48-point reducer-only command supplies S2/S4/S8 diagnostic
costs without changing those heuristic choices. Both result archives have now
been reviewed: 88/88 GEMM and 48/48 reducer points passed. The original board
JSON below predates these uploads and has not yet been replaced by a new
production route table; its missing GEMM fields do not mean those box runs
are still outstanding. See the [evidence audit](KPACK_HEURISTIC_COST_AUDIT.md).

The matched plan covers 44 workload/token points and FQ/SF GEMM-only costs:
**88 completed route measurements**, before any small tactic challenge set. This is
not an exhaustive config Cartesian product. Use production-selected tactics
and explicitly retain applicable historical winning configurations as
challenges; record which wins instead of treating the compiled default as
optimal. The generated missing plan is a specification, not an executable
or a claim these measurements have run.

Existing tuner functions reuse fixed input addresses across launches
(`scalefirst_grouped_kpack_discovery.hpp`,
`scalefirst_internal_sweep_bench.hpp`, and the corresponding FQ benchmarks).
The old native FQ/SF gate prices per-call metadata together with GEMM and
uses M1/128, not this 2048/4096 denominator. Those timings remain valid in
their own scope, but are not inserted into the rotating-ring, independently
measured component board. Repeat only the missing matched GEMM components;
keep the dequant, cuBLAS and Python-JIT DeepGEMM evidence already collected.
