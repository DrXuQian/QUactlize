# Q4 TP local scale-field loss

Evidence: `q4-tp2-simt.7ADt8a.results.tgz`, SHA256
`817abf70c8864ba393f8a8093e5b9fc01adc2b9c1dd386fd8cb314c6c39385ad`,
source `380eba55c39a6b158f6765b2d40589d53c455c99`.
The independently re-parsed log inventory is complete: 12 processes, 72
output cells and 12 wrong-expert negatives. All negatives are red.

| Boundary | Observed result |
| --- | --- |
| Host/GPU packed low and units | Byte-exact in every process |
| Scalar canonical reader | 24/24 output cells pass |
| Shipped generic SIMT | 24/24 output cells fail |
| Fresh generic SIMT | 24/24 output cells fail |
| Storage/compute controls | F32 storage; F16/BF16 outputs are identical |
| Shipped/fresh images and HOST/GPU pack controls | All eight SIMT output copies per input are byte-identical |

Each SIMT output has exactly 256 bad entries out of 1024. They occur only at
`N % 4 == 0`. No llama graph, TP communication, cache writer or model is present.
Runtime library hashes match the shipped receipt. The compiler executable hash
differs, as recorded, but both images produce identical arithmetic failures.

## Constructive explanation

For each raw Q4 GGUF superblock and every fourth N column, clear only the low
nibble of byte14: `block[14] &= 0xf0`. This is scale group6's low four bits;
the high two scale bits, every min field and all quantized weight codes remain
unchanged. Decode these raw bytes with llama's host `dequantize_row_q4_K`, then
compute the original activation dot in F64 and round once to F32.

This counterfactual reconstructs **all 6,144 returned SIMT outputs** within
8.195639e-8 absolute error, including all 1,536 bad entries. Per-case maximum
errors, rank-major/replay-major, are 7.4505806e-8, 7.4505806e-8,
8.195639e-8, 7.4505806e-8, 5.9604645e-8, 8.195639e-8. The reference uses raw
GGUF decoding, not the K-pack consumer. It is retained locally in the analysis
of `/tmp/q4-tp2-analysis.jCmwlD` with the exact archived fixtures.

In the canonical packed unit, scale group6 occupies bits92..97 across two
32-bit words. Bits92..95 are the low four; bits96..97 are the high two. For
K512 the affected logical groups are 6 and14. This precisely identifies the
lost semantic contribution. It is not yet a proof of the compiler pass or
machine instruction responsible. In particular, the logical C++ extraction
looks valid; do not blame PCCL, BF16 range, or the entire reduction routine.

## Same-row repair and PPU closure

Run the existing single-device diagnostic with `Q4_TP2_FIELD_AB=1`. It adds
the existing `q4_affine_header32` extraction to the exact generic
V0/C4/W4/P4/S1 row through its `Changes=1` template axis. It changes only
scale/min extraction, not group ownership, code unpacking, dot accumulation,
warp reduction, shared memory topology, inputs or output layout. This A/B was
run at source `539ba5fa6c075b9aa2accc0d80691db21a2cd16c`, before promotion.
Use that revision with the original bundle to reproduce the retained legacy
negative; do not expect a repaired production image to reproduce the defect.

The candidate constructs each 24-bit scale/min stream with fixed register
operands and 32-bit shifts. For group6 its scale is
`((u.z >> 28) | (u.w << 4)) & 63`, preserving both pieces explicitly.
Local PPU object inspection keeps the legacy body identical, ignoring the
object filename. Both bodies have 132 static F32 FMA instructions, five
shuffle instructions and one CTA barrier. Candidate indirect register reads
decrease from27 to0. Static instruction counts are not device timing proof.

The gate requires all of the following before reporting
`SCALE_FIELD_LOSS_CLOSED_IN_ISOLATE`:

1. Shipped and fresh legacy SIMT remain red on all frozen inputs.
2. Candidate and scalar outputs are green for HOST/GPU packing and F16/BF16.
3. Clearing just packed bits92..95 at every fourth N column makes the candidate
   reproduce the legacy error, while rejecting the original golden in exactly
   256 places and no other columns.

The uploaded `q4-tp2-simt.62KWWD.results.tgz`, SHA256
`1c1cd4d43b058da5558374583249f8d87610100b6a74e9ffcf6f998f68130fd6`,
passes this gate. Its independently reparsed 12-process inventory has all 96
output cells, 12 wrong-expert negatives and six exact field-loss controls:

| Arm/control | Result |
| --- | --- |
| Shipped legacy SIMT | 24/24 expected failures |
| Fresh legacy SIMT | 24/24 expected failures |
| Scalar canonical reader | 24/24 pass |
| Fixed-register H32 candidate | 24/24 pass; maximum absolute error 8.94069672e-8 |
| Packed byte checks | All exact |
| Wrong-expert negatives | 12/12 red |
| Clear only bits92..95 at N%4=0 | 6/6 reproduce the legacy full output bit-for-bit; 256 bad entries each |

`field-loss.bin` equals `simt-0-1.bin` for all six frozen inputs, not merely
within tolerance. The verdict is `SCALE_FIELD_LOSS_CLOSED_IN_ISOLATE`.

## Static ISA follow-up: low-word read before reconvergence

Field-loss closure is not, by itself, identification of the underlying
compiler mechanism. A subsequent dataflow inspection found a concrete mask
ordering defect in the locally compiled legacy F16 symbol
`register_reuse<12,1,0,4,4,4,0,0>`. The inspected `legacy.isa` text has SHA256
`8334c1387620a37082e208a7ae994e0f8879e6f54dded2222431b36865a41451`.
The corresponding shipped image is retained locally. Its SHA256
`8ef56b91df13ece8816ffdcd29c057b4d0393dddcda2bd157d815c6b96591cea`
matches the box upload's shipped manifest. F16 and BF16 function instruction
bytes match the local isolated legacy exactly (only disassembler block labels
differ). Thus the mask defect is present in the actual failed library, not
merely in a new compiler experiment.

The first column's header is loaded into vreg28..31. Scale shift is in vreg27;
word index is in vreg32. The generated branch tests `shift < 27`, excluding
the cross-word scale group6. The relevant instruction order is:

| PC | Operation | Relevant execution scope |
| --- | --- | --- |
| 0x3120 | `s.lop.emsk sreg29, vcc # 0x8` | Enter the non-cross-word lanes |
| 0x3830 | `v.mov.b32 vreg39, ivreg` | Read first column's low word while those lanes alone are active |
| 0x3850 | `s.lop.emsk sreg29, sreg29 # 0xe` | Restore the other lanes only after the read |
| 0x3858 | `v.shrl.b32 vreg39, vreg39, vreg27` | Shift the value, including the lanes whose read was skipped |
| 0x40a0, 0x0a98, 0x0aa0 | Read high word, shift, OR with vreg39 | High fragment is available, low fragment was never assigned for group6 |

At local K512 there is one group-loop iteration per active worker. The
excluded lanes retain vreg39's entry zero. For V0/C4/W4/P4, the excluded mask
is `0x0f000000` (lanes24..27) in each active warp: groups6 and14. Their first
output columns are N residues0/4/8/12 within each16-column tile. Thus the
mask ordering explains both observed axes: scale low-nibble loss and only
N%4==0 outputs. The corresponding symbolic model maps every scale s in0..63
to `s & 0x30`; restoring the mask before the read recovers every original s.
This model is not a substitute for a device replay.

The source extraction for this exact group is within bounds: word indices2/3,
unsigned shifts28/4, width6. No out-of-range shift is required to explain the
failure. The H32 candidate avoids dynamic register indexing and this divergent
read path altogether; it does not change the stored scale or clip arithmetic.
The generated-code defect is now tied to the failed image. The compiler's
internal responsible pass is not identified. No claim about a particular pass
is needed to repair the consumer; retain that distinction in compiler reports.

`dev/gemv_simt/tp2_scale_isa.py` rechecks the exact shipped library hash, both
instruction streams, mask/read/reconvergence anchors and the field-loss model.
It also verifies that the newly built production functions are byte-identical
to the local H32 candidates used for the numerical A/B:

| Compute | Legacy code SHA256 | Production H32 code SHA256 |
| --- | --- | --- |
| F16 | `4a2ca349fccd0e714e74c25294439b0fc33bf745d2dd23cde615b42a71f71685` | `45b1b4636b047ca2af05f9cc5f749de4522338a17727d729c6773792b6e279cb` |
| BF16 | `8f6dd5293116bfa5043fc519fcb292eb06cef17a1cd33695d94489205dda0d12` | `8ae9a1787af4f33d9f7532233b35aa6bf610fbc2774ac23f56f340a786b15db4` |

The static check does not run a GPU and supplies no latency verdict. The next
box task is full TP2 integration, not another scale-loss bisection.

## Production scope

The shared `affine_selected` seam now uses H32 for **every Q4/Q5 recipe**.
Q5 uses exactly the same four-word, eight-group scale/min representation;
its extra code plane is unchanged. Ordinary, indexed and paired gate/up SIMT
bodies share this seam. Q2/Q3/Q6/Q8 take their previous branches. The offline
format, public ABI, config inventory, Split-K, dot/reduction order and output
types do not change. No runtime switch or local-shape workaround is added.

`tests/test_simt_scale_fields.py` compiles the actual helper and selector as
host arithmetic. It exhaustively checks 32,768 scale/min combinations, Q4/Q5,
all eight groups and four existing recipe modes, plus the precise field-loss
negative. This host check supplements, not replaces, the PPU A/B.

The standalone and paired execution libraries must be rebuilt; an old binary
does not acquire the fix from a checkout update. Unchanged TC parent modules,
prefill, GPU packer, JIT source contract and caller are reusable. The refreshed
package must bind each execution-dependent gate to the new image without
carrying a previous device-admission verdict. Full TP2 cold/hot arithmetic,
122B model accuracy and warmed performance still require the next box run.
