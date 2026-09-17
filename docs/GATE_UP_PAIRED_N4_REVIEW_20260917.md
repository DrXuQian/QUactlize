# Paired-N4 first PPU return

Evidence: `gate-up-paired-n4.mJSZYK.results.tgz`, SHA256
`12b3d85598297e3c40ba923c290111d987f5e059eadf31468a54ffa70111f4e4`.
Source `bcf1c3d7f7d3d9b410d9b1a9aba24c36f3a479f9`; candidate DSO SHA256
`88f3035397da23b8dc869824874d39fc868d6db8db7682d75b00aa4f822ef7c7`.
Returned manifest and all six harness source hashes match the local receipt.
One visible device, ordinal0, PCI `0000:08:00.0`.

## Numeric admission, not performance

| Format | SIMT | TC |
|---|---|---|
| Q2_K | PASS 1440/1440 | PASS 1728/1728 |
| Q3_K | PASS 1440/1440 | Incomplete: 376 configurations passed, large-BF16 replay failed |
| Q4_K | PASS 1440/1440 | PASS 1728/1728 |
| Q5_K | PASS 1440/1440 | PASS 1728/1728 |
| Q6_K | PASS 1440/1440 | Incomplete: 40 configurations passed, large-BF16 replay failed |
| Q8_0 | PASS 1440/1440 | PASS 1728/1728 |

Each of the ten complete parts has exact/unique configuration coverage,
80 changing-input replays and 32 detected negative controls. Thus 15,552
configuration cells belong to complete parts. Another 416 normal cells in
the two incomplete parts passed; these do not close those parts. The returned
run explicitly has `timing=NOT_MEASURED` and `production_selection=UNCHANGED`.
Neither whole-model accuracy nor a TPOT win follows from this return.

## Exact observed failure scope

Both exceptions are at `replay_checks`' **large BF16** check, not the normal
replays, launch, pack-byte check or a nonfinite-output check:

- Q3 TC: dense M8/N256/K2048/E1, BF16 compute, F32 input/output,
  projection rounding enabled, replay TM8.
- Q6 TC: dense M1/N256/K2048/E1, same type/rounding path, replay TM8.
- Both report normalized output error `0.00558659217877095` against 0.005.
- The original runner did not save which of S1/S8 failed, the actual output,
  pre-round projections, or the first differing coordinate. Do not invent them.

The fixture inserts F32 `243383.484`, rounded to BF16 `243712`, beside many
small activations. Exactly representable weights/inputs **do not** make that
mixed-magnitude FP32 sum order-independent.

## Constructive host counterexample (not a PPU root-cause verdict)

For the exact Q3 M8 fixture and channel88 of row5:

| Quantity | Wide-dot oracle then F32 | Legal sequential FP32 dot |
|---|---:|---:|
| Gate before BF16 rounding | 1427.999755859375 | 1428.0008544921875 |
| Gate BF16 bits | `0x44b2` | `0x44b3` |
| Fused output | -5422592 | -5453056 |

1428 is the BF16 rounding midpoint. This legal FP32 accumulation difference
produces exactly the returned Q3 normalized error, **0.558659217877095%**.
The no-projection-rounding comparison remains below 1e-5. Therefore the
output threshold alone cannot distinguish a bad TC reader/epilogue from this
rounding amplification. It is not evidence that PPU uses the sequential order.
There are no saved device outputs to prove this explanation for either failure,
and the Q6 device failure still needs its own direct evidence.

`tests/test_gate_up_rounding.py` preserves this counterexample and checks a
one-nonzero-A large-BF16 control that is exact across these accumulation orders.
No production code, old fixture, threshold or failed verdict has been changed.

## Next bounded diagnostic

Run with `GATE_UP_DIAGNOSTIC=1` using `tools/run_kpack_gate_up_box.sh`.
It reuses the identical DSO and runs only Q3 M8 and Q6 M1, TM8/TM16, S1/S8,
with/without projection rounding, and mixed/sparse large-BF16 inputs (32
observations total). It saves outputs, gold projections and Split-K device
partials in small NPZ files. The original mixed fixture remains present.

The diagnostic prints `DIAGNOSTIC_COMPLETE`, never numerical PASS merely
because it finished. Review the actual device projection errors and the
activation of those projections before changing either kernel or oracle.
All ten independently passed parts remain valid. Q3/Q6 TC performance and
production admission remain blocked on numeric interpretation.

## Second return: saved projections and rounding controls

Evidence: `gate-up-paired-n4.2NfLbX.results.tgz`, SHA256
`af9aaed0dd968ca8a00b3f934504ef38cbe1310e52977ea58696d115406932e3`.
Source `fe77897fcbff39165c84065ce423d11960544f0b`, unchanged candidate DSO and
runtime. Same PCI `0000:08:00.0`. All 32 saved-array hashes, six harness hashes
and diagnostic source hash match that revision. Do not compare the historical
harness hashes to a newer, edited checkout.

| Control | Q3_K | Q6_K |
|---|---:|---:|
| Sparse large A, all TM/S/rounding arms | exact | exact |
| Mixed large A, S1, projection rounding on | 0.558659% | 0.558659% |
| Mixed large A, S1, projection rounding off | 0.001039% | 0.001098% |
| Mixed large A, S8, projection rounding on | 0.417821% | 0.072338% |
| Mixed large A, S8, projection rounding off | 0.000147% | 0.000147% |

TM8 and TM16 give identical arrays in every paired control. All outputs are
finite and guards pass. S8 gate/up errors before rounding are at most
`1.026e-6` normalized; applying the activation to the saved device projections
agrees with final output within `2.80e-12` normalized. This last comparison is
not a claim of raw-bit agreement for small outputs across CPU/GPU exp.

The four original failures are precisely Q3/Q6 × TM8/TM16, S1, mixed-large,
rounded projections. At the actual device's worst coordinates:

| Quantity | Q3 row1/channel88 | Q6 row0/channel184 |
|---|---:|---:|
| Wide gate dot | 1428.0048065185547 | 14279.868438720703 |
| Wide up dot | -3808.0243530273438 | -22848.091369628906 |
| Permitted gate BF16 bins | 1424, 1432 | 14272 |
| Permitted up BF16 bins | -3808 | -22912, -22784 |
| Original gold output | -5453056 | -327000064 |
| Device output | -5422592 | -325173248 |

Here "permitted" means the discrete BF16 rounding images of an independent
F32 forward-error interval, not any real number in a wider output tolerance.
For exact BF16 operands/products, the bound is
`gamma_(K+S) * sum(abs(A * W))`, with `gamma_j = j*u/(1-j*u)`, `u=2^-24`.
It conservatively includes any F32 sum tree and Split-K reduction, plus the
F64 host summation bound and outward-rounded interval endpoints. Q3 bounds
are approximately 0.17513/0.46682 for G/U; Q6 bounds are 1.75107/2.80116.
Both returned values are **exact F32 products of permitted BF16 projections**.
All permitted gates are >=32, so the F32 sigmoid denominator is exactly one;
there is no exp approximation allowance in this check.

Thus the original fixed output threshold rejects numerically permissible
rounding outcomes. The sparse controls exclude F16 overflow on the transported
large value. The unrounded and S8 controls localize the amplification to the
intermediate rounding. S1 raw projections were not saved; this does not prove
its exact accumulation order or bitwise equivalence to a CPU sum.

## Narrow oracle correction and remaining coverage

`tools/gate_up_rounding_oracle.py` adds an opt-in check only for large-BF16 TC
replays with rounded projections and F32 storage/output. Normal configuration
cells, original diagnostic, output/stride/workspace guards, and nonfinite
checks are unchanged. The 0.005 threshold and its failures remain recorded as
`original_verdict=FAIL` with a separate `CERTIFIED_DISCRETE_ROUNDING` report.
Only threshold-exceeding coordinates get this exception; each must exactly
match one permitted BF16 product. Non-saturated gates, non-exact operands,
exceptional range, >8 possible bins, zero/sign/column mistakes, Inf/NaN and
even a one-F32-ULP deviation from a permitted product are rejected. The same
checker is exercised by the on-device zero-A negative.

The source-bound helper is included in resume identity. This is not a kernel
patch, threshold increase, clipping operation or production arithmetic change.
The existing DSO, load/conversion/MMA/barrier/branch counts and selectors are
unchanged. Historical receipts are not rewritten as PASS.

Q3/Q6 TC still lack 3,040 normal configuration cells from the interrupted first
run. Re-run those two complete parts (3,456 cells, plus replays/negatives) with:

```bash
GATE_UP_FORMATS=11,14 GATE_UP_BACKENDS=tc \
  bash tools/run_kpack_gate_up_box.sh
```

The collector records an explicit subset with `full_inventory=false`; combine
it with the ten previously complete parts only after reviewing its return.
No device compilation is required. Real-shape full-call latency, integration
and whole-model accuracy remain separate, pending admissions.
