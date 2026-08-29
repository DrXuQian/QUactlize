# Q4_K K-pack4 production integration

Canonical artifact identity:

- descriptor version: `2`
- layout: `QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1`
- mapping: `0x51344b5034540001`
- low/high bits: `4/0`
- artifact TileK: `0` (there is no Xplane TileK axis)
- transport TileK / group size: `64 / 32`

This descriptor, rather than tensor shape or byte count, is the authority at every producer/consumer boundary.

## Integrated surfaces

| Surface | Production entry | State |
|---|---|---|
| Offline dense placement/recovery | `quactlize_ppu_{prepare,recover}_dense_for_arrangement_v2` | integrated; exact round-trip |
| Offline grouped placement/recovery | the same primitive, applied with explicit per-expert byte strides by the Torch ops | integrated; expert-major inverse |
| Packed metadata units | `quactlize_ppu_prepare_units{,_grouped}` / `quactlize_ppu_prepass_unit` | integrated; unchanged byte-neutral units |
| Dense FQ host launch | `quactlize_ppu_dense_fully_quantized_for_arrangement_v2` | integrated; M-aware S1/S4 shipping policy |
| Dense FQ device launch | `quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v2` plus arrangement workspace query | integrated; async caller-owned workspace |
| Dense tactic inventory | arrangement-v2 v3/v4 list and shared validity predicate | integrated; v4 records Split-K S |
| ScaleFirst prefill | `quactlize_ppu_dense_lowbit{,_dev}_for_arrangement_v2` | integrated; persistent S1 over the same code bytes |
| Grouped/MoE FQ host launch | `quactlize_ppu_grouped_fully_quantized_for_arrangement_v2` | integrated; existing ragged scheduler/epilogue, K-pack4 mainloop override |
| Grouped/MoE FQ device launch | grouped arrangement-v2 workspace query and device entry | integrated; async caller-owned workspace |
| Grouped tactic inventory | grouped arrangement-v2 list and shared validity predicate | integrated; tactic TK256, artifact TK0 |
| Torch producer/reader/inverse | `gguf_*_for_arrangement_v2` dense and grouped ops | integrated; optional dlsym keeps old libraries loadable, use is fail-closed |
| Python artifact and routes | `PlacedArrangementV2`, `PlacedArtifact`, dense/grouped FQ and ScaleFirst routes | integrated; Q4 `layout=auto` defaults to K-pack4, Xplane is explicit diagnostic compatibility |
| Dense runtime policy | `matmul_q4_kpack4_dense` + hoisted scale workspace | integrated; M<=8 FQ decode, M>8 persistent ScaleFirst over the same bytes |
| Whole-model pack/restore | `tools/pack_gguf.py` | integrated; Q4 has no layout switch and always emits K-pack4; rank-2 dense and rank-3 `[K,N,E]` grouped tensors retain route/E/N/K plus descriptor |
| Format-selected loading | `gguf_backend_for_qtype(12)` / `QUACTLIZE_PPU_LIB_FMT0` | integrated; v2 placement and packed units come from one format-owned handle |

## Deliberately unsupported

- BC/SIMT dense and MoE readers do not implement the K-pack4 physical map. Python refuses a v2 artifact before
  dispatch, and no v2 BC C entry is exported. Do not strip the descriptor and call the Xplane-v1 ABI.
- K-pack4 is Q4_K-only. Q2/Q3/Q5/Q6 continue to use their versioned Xplane arrangements.
- Grouped K-pack4 currently reuses the five compiled grouped tactic geometries with auto delivery-N. Correctness is
  type-closed; a grouped performance sweep is still required before claiming a tuned grouped default.
- Fused metadata-store remains an experiment, not the default. Decode measurements showed gains on some families
  and regressions on another, so the plain packed-unit provider remains the deployment contract.
- The legacy per-model tactic-cache writer does not benchmark K-pack4 bytes and therefore must not be used to name
  a K-pack4 winner. K-pack4 currently uses its measured compiled shipping policy; a future cache schema must carry
  descriptor version/layout/mapping and ingest a K-pack4-native sweep before it can override that policy.

## Evidence and remaining device closure

Local gates prove descriptor forwarding, mapping-drift rejection, dense/grouped inverse routing, exact policy
instantiation, and both packed/non-packed backend variants. Device closure still needs one fresh box run covering:

1. dense decode M=1/2/4/8 with the v4 Split-K inventory;
2. ScaleFirst persistent at M=64 and a real M=2048 prefill row (existing performance evidence also covers M=4096);
3. ragged grouped Q4_K with an empty expert and an independent dequantized golden;
4. a descriptor-mapping negative which must fail before launch.

Existing measured ScaleFirst evidence for `(N,K)=(1024,5120)` is within 0.6% of Xplane at M=2048 and M=4096,
with K-pack4 slightly faster in both rows. That result establishes the shared prefill bytes; it does not substitute
for the grouped device closure above.

## Non-Q4 format boundary

Q4's K-pack4 map is not a generic k-quant map. It relies on exactly four adjacent Q4 nibbles forming the b16
transport consumed by the transpose reader. Q2/Q3/Q5/Q6 have different low/high plane widths and must keep the
shared Xplane producer/reader until each format has its own byte-map derivation and device adjudication.

Two separate obligations follow:

1. A Q4 release must rerun the existing raw-bit/numeric device oracle for Q2/Q3/Q5/Q6, using one
   `PPU_PACKED_FORMAT` build per format, to prove the shared Xplane path was not regressed.
2. Replacing Xplane for any non-Q4 format is a separate design decision. It requires that format's own exact
   mapping proof plus real-shape decode and prefill sweeps; Q4's K-pack4 performance evidence cannot be reused.

Therefore archiving Q4 Xplane never authorizes deleting the shared Xplane implementation or its generic sweep.
