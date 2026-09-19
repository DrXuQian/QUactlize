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

## Same-row repair candidate

Run the existing single-device diagnostic with `Q4_TP2_FIELD_AB=1`. It adds
the existing `q4_affine_header32` extraction to the exact generic
V0/C4/W4/P4/S1 row through its `Changes=1` template axis. It changes only
scale/min extraction, not group ownership, code unpacking, dot accumulation,
warp reduction, shared memory topology, inputs or output layout. Production
headers, routing and delivered libraries are still unchanged.

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

There are now 96 output cells plus 12 wrong-expert and six field-loss controls.
The per-case coordinate/tolerance checks remain unchanged. Upload the same
small `q4-tp2-simt.*.results.tgz`. If this gate closes, promote the narrow
extraction repair, inspect Q5's shared metadata representation, rebuild the
affected SIMT image and validate the complete TP2 cold/hot and model paths.
This diagnostic alone is not full TP2 production admission.
