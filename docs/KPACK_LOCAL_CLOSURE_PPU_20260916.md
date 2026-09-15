# Returned PPU local closure

Input: `/root/kpack-local-closure.pah3UP.results.tgz`, SHA256
`988697a0321090cbf70d302ccc74b94ab3ddee2451e6054d054db495d4b5d021`.
The execution image is `1122afabaa9c8b749bb88a12281887c2635155468d3048956986b3befeb9be43`,
from artifact `14066dae14108f54a999c957f17ee68cb6117d49`.
[Compact evidence](measurements/local_closure_ppu_20260916.json) retains every
part count, the original failure records, and all 48 prepare comparisons.

## What ran

| Gate | Result | Scope |
| --- | --- | --- |
| BF16 grouped TC | 226/226 PASS | Six formats, FQ/SF, ordinary/persistent and legal Split-K cases |
| Generic SIMT | 396/396 PASS | Six formats, M1--8, dense/indexed, FP32/BF16 inputs and FP16 controls |
| Q6 range/outlier | 8/8 PASS | The independently tested large finite activation is preserved |
| Complete BF16 MoE chains | 85/116 PASS, 2 failed, 29 not run | Q4 down comparison and Q6 host scalar conversion below |
| Selected Q4 BF16 fast paths | 258/258 PASS | All 40 compiled recipes; 129 F16 controls and 129 overflow negatives |
| Candidate MoE prepare | 3,840/3,840 PASS; 48/48 paired helper cases PASS | Prepare and actual SwiGLU, not GEMM/model admission |
| Candidate Q8 vector reader | Did not load | No Q8 candidate numerical/performance verdict from this run |

The 746-case gate has 715 passes, two recorded failures, and 29 unexecuted
cases. `passed=13/26` for Q4 means 13 passed, one failed, 12 unexecuted;
it does not mean 13 kernels failed. The outer runner preserved other parts.

## Failure boundaries and local repairs

### Q8: host ELF linkage, not a device result

`q8-numeric.log` stops in `ctypes.CDLL` with
`undefined symbol: __hggcPopCallConfiguration`. The old link command put
SDK libraries before the referencing objects; `--as-needed` discarded them.
`readelf -d` confirms the old library has no SDK `DT_NEEDED` dependency.

The builder now places objects before libraries and uses `-z defs` to reject
unresolved symbols during linking. The rebuilt PPU library retains
`libhggc_wrapper.so`. The runner also initializes the explicit SDK before
loading the module. A real host shared-library regression reproduces the old
load failure, passes with the corrected order, and rejects a missing runtime.
Generated device source and production Q8 selection are unchanged.

### Q6: scalar NumPy conversion

`0607-moe-q14-tokens1-mergedFalse-kindtc-computebf16` completed its normal
chain checks, then failed while constructing the host golden for the injected
wide-range test. NumPy scalar integer promotion widened the shifted word to
64 bits; viewing that zero-dimensional scalar as F32 raised `ValueError`.
The BF16 bit oracle now uses explicit unsigned 32-bit shift operands for
both scalar and vector inputs. No compute kernel or range limit changed.
This case still needs to finish on PPU before it is counted as a pass.

### Q4: down-projection oracle consumed a different input

`0370-moe-q12-tokens8-mergedFalse-kindsimt-computebf16` uses E256, top8,
N=K=512, separate gate/up and generic SIMT V3/C4/W4/P4/S1. At the down
comparison, 8/32,768 values exceed the 0.005 normalized-error bound; all are
finite. First index 16,754 is row32/column370:
`0.07080078125 -> 0.0712890625`, one BF16 ULP. Maximum normalized error
is 0.0066948020. This is not a failure of the separately tested selected
Q4 fast kernel.

The harness allows one BF16 ULP between GPU and NumPy SwiGLU, but previously
fed NumPy's value into the down-projection oracle. That compares dots with
different inputs. A deterministic host negative demonstrates why an already
admitted one-ULP boundary difference can exceed the down threshold.

The stage-local oracle now dequantizes the original GGUF weights and uses
the actual downloaded down input. The SwiGLU boundary test and the independent
whole-chain golden remain in place. Neither 0.005 projection tolerance nor
0.02 whole-chain tolerance was enlarged. Wrong output-coordinate negatives
remain rejected. The next failure report records repeat, actual/host input
hashes and SwiGLU ULP count.

This proves a harness inconsistency, **not yet that it explains all eight
PPU differences**. The returned run did not save its intermediate tensor.
Replay the original cases; if the same-input dot still differs, investigate
the producer rather than weakening the oracle.

## Prepare performance

PPU, BF16, merged gate/up, K2048, fixed top8 helper fixtures. Each median
uses 60 samples across alternating arms; the benchmark excludes the first
graph launch. These are warm helper timings, not TPOT or a cold GEMV metric.
Printed medians were checked against the samples allowing only F32 arithmetic
and decimal-print rounding. External GPU-idle admission was not recorded.

| Tokens | All-SIMT old/new, us | Mixed SIMT/TC old/new, us | All-TC old/new, us |
| --- | --- | --- | --- |
| 1 | 7.296 / 6.448 | 8.078 / 8.898 | 9.165 / 10.296 |
| 2 | 15.379 / 6.683 | 15.366 / 12.077 | 15.844 / 13.841 |
| 4 | 13.184 / 6.421 | 13.211 / 12.286 | 14.281 / 15.599 |
| 8 | 35.196 / 6.442 | 35.181 / 17.963 | 36.239 / 23.633 |

All-SIMT improves in all 16 tested helper contexts (FP16/BF16, two K sizes,
four token counts), by 11.4--82.1%. M1 mixed/all-TC regress, and all-TC M4
depends on K. Do not enable the candidate globally or infer a whole-model
speedup. Its row-map change needs actual selected-producer composition before
production promotion. The NVIDIA-only old M1 gather fault did not reproduce:
both PPU arms completed all 48 helper comparisons.

## Next device evidence

Only Q8's small experimental DSO needs rebuilding. Dispatcher, execution,
40 TC modules, packer, prefill and caller are unchanged. Updated gate receipts
must explicitly record the host-harness change; do not disable source checks.
Run the repaired BF16 gate and Q8 numerical/performance tests. Retain the
selected-Q4 and prepare evidence above; no full tactic sweep is needed.
BF16 whole-model precision/performance admission remains pending.
