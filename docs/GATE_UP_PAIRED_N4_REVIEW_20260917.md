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
