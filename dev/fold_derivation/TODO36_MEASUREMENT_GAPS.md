# TODO #36: three different measurement gaps

This audit separates three items that were being carried as one generic
"missing measurement".  They have different evidence, different consequences,
and different ways to close them.  No device result is invented here.

| item | classification at this revision | what is already established | what is not established |
|---|---|---|---|
| fully-quantized prefill | **Not measured at the shipping entry.** One direct Q4_K/K-pack4 internal-collective M=2048 pilot is raw-bit clean and timed, but it is not the shipping operator. | The packed artifact ABI, production readers, independent numerical golden, planted metadata fault, exact TM64=210/918 pilot denominator, and its S1 result (`101.76 us`) are covered.  A complete 774-row AP0 × 15-shape follow-up runner is registered. | No timing for either shipping dense or grouped-L1 fully-quantized entry, and no completed multi-M K-pack4 result yet.  Neither the ScaleFirst prefill result nor decode K-pack4 comparison may be extrapolated to those entries. |
| int4 gs=128 COARSE | **Not measured after the COARSE runtime-assert fix; measurable by the existing dense bench.** | The exact ScaleOnly row instantiates; local layout/metadata contracts cover the COARSE/FINE boundary. | Device numerical correctness and timing on the repaired ScaleOnly implementation.  BACKTEST B1/B2/B3 remain old-implementation targets, not current results. |
| int4 gs=16 post-`w64x32` | **Measured historically; current native-dense HEAD has not been refreshed.** | `HANDOFF_TASK12.md` records 234.16 µs, followed by 228.13 → 227.35 µs, at M=2048/N=K=4096 and gs=16.  Thus the old claim that 58.7% was only arithmetic is false. | A fresh regression result through today's dense operator and generated tactic row. |

## Evidence and impact

### Fully-quantized prefill

`tests/test_q4k_packed_gemm.cu` launches correctness cases but has no timing.
`tests/test_gguf_routes.py` calls the shipping dense route with an independent
golden and a planted metadata fault, but uses tail sizes (M=7 and M=65) and does
not time the launch.  Consequently:

- the scale-first A0/A1 result does not support a fully-quantized prefill claim;
- the decode-band +13.1% tax cannot be promoted to prefill; and
- the predicted 16-fold per-m-tile packed decode at M=2048 remains a mechanism,
  not a measured slowdown.

The K-pack4 pilot establishes raw-bit and timing evidence for the direct
production collective at one inventory-owned M=2048 cell; the complete
multi-M collective sweep remains pending and neither result can close this
shipping-entry gap.  Closing the gap requires a benchmark that calls
the shipping `*_fully_quantized` entry, prepare the artifact/H2D/workspace
outside the timed interval, run the existing independent golden and planted
metadata fault first, and use one device-event pair per launch.  Dense and
grouped-L1 are distinct shipping entries and must be reported separately.

### gs=128 COARSE

The old unconditional runtime assert is gone and the row now compiles, but a
compile is not a numerical result.  The missing run weakens every current-use
claim based on BACKTEST B1/B2/B3 (61%/25%/56.6%).  It does **not** weaken GGUF
shipping qtypes, whose registered group sizes are 16 or 32.  The queued mode-1
row can establish the symmetric GPTQ/ScaleOnly gs=128 path for one B1 topology;
it cannot establish asymmetric GPTQ/AWQ ScaleZero or refresh B2/B3.

The existing dense bench can close it.  The box queue deliberately schedules
the explicit historical B1 row before any search, so numerical failure cannot
be hidden by choosing another tactic.

### int4 gs=16

The historical source is `HANDOFF_TASK12.md`: its post-chunking/post-`w64x32`
table records 234.16 µs, and the later equivalence refactor records
228.13 → 227.35 µs.  `TODO.md` already marked the gs=16 remeasurement done.
The remaining gap is freshness and operator identity: those values came from
the grouped-L=1 route, while current HEAD should be checked through the native
dense route.  Until then the ~60% mechanism is measured historically, but it
is not a current regression result.  The new run is a cross-operator bridge;
a changed time alone is not proof that either kernel regressed.

## Queue discipline

The runnable gs=128 and gs=16 commands live in `.coord/BOX.md` under a section
explicitly ordered **after** the two frozen morning runs.  Fully-quantized
prefill remains blocked on its shipping-entry timing harness.  The K-pack4
internal-collective pilot is a narrower admission command and must not be
reported as closure of TODO #36.
