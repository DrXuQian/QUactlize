# The packed (FULLY_QUANTIZED) format: how to produce it, and how to consume it

For the llama.cpp side. `quactlize_ppu_vecdot_dense` / `_moe` read **raw GGUF blocks**; the packed GEMM is a
different entry point with a different input, and this file is both halves.

---

## 0. Read this first: the consumer ABI is HOST-pointer, the producer ABI is correctly host-pointer

`ppu_dense_backend.cu:41-68` — every packed GEMM entry does
`DevBuf d(...); d.from_host(ptr); launch on the default stream; rt_sync(); dout.to_host(out);`

For the **producer** that is right: packing is offline, the caller has host buffers, and a copy per tensor once
is free. For the **consumer** it is the same defect already recorded against the vecdot path
(`.coord/INBOX.md` 030): ggml hands you a device pointer that is already resident, and this ABI would treat it
as host memory and re-copy the whole weight on every token, on the default stream, with a full device sync.

So: **the packer below is usable today. The GEMM symbols below are the right names and shapes to build the call
site against, but a working integration needs device-pointer variants** — same signature plus a stream, no
allocation, no sync, no copies. That work is queued with the kernel author, deferred at the user's direction.
Build against these names; expect a `_dev` suffix or a version bump when they land.

---

## 1. The artifact: three buffers per tensor

`prepare` turns one GGUF tensor into **(low, high, units)**:

| buffer | what | absent when |
|---|---|---|
| `low` | the low code plane, placed for the AIU | never |
| `high` | the second code plane | single-plane formats (Q2_K, Q4_K) — pass `nullptr` |
| `units` | the format's own scale metadata, reordered into bulk-copyable units | never |

`units` is **byte-neutral**: the scale is not expanded to fp16 planes, it stays in the format's own bytes. That
is the whole difference from the SCALE_FIRST cell and why storage does not grow.

Per format:

| qtype | format | planes | code | TileK | scale unit (copyable) |
|---|---|---|---|---|---|
| 10 | Q2_K | 1 | i2 | 256 | 20 B |
| 11 | Q3_K | 2 | i2 + i1 | 256 | 28 B (two superblocks paired) |
| 12 | Q4_K | 1 | i4 | 256 | 16 B |
| 13 | Q5_K | 2 | i4 + i1 | 256 | 16 B |
| 14 | Q6_K | 2 | i4 + i2 | **128** | 36 B (paired) |

Q6_K's TileK is 128 and not a preference: at TK=256 the two-plane high map covers only half the logical K slots
and produces conditioned error 8.76e-1. The inverse caught it. Do not "fix" it to 256.

The pairing rule for the unit size is `supers = 2 if per_superblock_meta_bytes % 4 else 1`, because `ppu.cp.async`
moves only 4, 8 or 16 bytes and Q3's 14 and Q6's 18 are movable at none of them.

---

## 2. Producing it — C ABI (`quactlize/csrc/device/ppu_dense_layout.cu`)

```c
// Fixed per-format arrangement. qtype is the ggml enum (10..14).
int quactlize_ppu_prepare_dense(const uint8_t *low_native, const uint8_t *high_native,
                                uint8_t *low_layout, uint8_t *high_layout,
                                int n, int k, int qtype);

// Inverse. Exists so a packing error and a compute error can be told apart -- without it the only test is
// end-to-end and the two failure modes are indistinguishable.
int quactlize_ppu_recover_dense(const uint8_t *low_layout, const uint8_t *high_layout,
                                uint8_t *low_native, uint8_t *high_native,
                                int n, int k, int qtype);

// Tile-aware. SEPARATE SYMBOLS ON PURPOSE: an extension built against the old library cannot pass the extra
// integer to it and get silence. Fold is deliberately NOT a parameter -- producer and consumer both derive it
// from (bits, tile_k) by one expression, and a caller-supplied fold is a second source that can disagree
// without crashing (the weight is placed for one fold and read at another: finite, wrong numbers).
int quactlize_ppu_prepare_dense_for_tile(const uint8_t *low_native, const uint8_t *high_native,
                                         uint8_t *low_layout, uint8_t *high_layout,
                                         int n, int k, int qtype, int tile_k);
int quactlize_ppu_recover_dense_for_tile(const uint8_t *low_layout, const uint8_t *high_layout,
                                         uint8_t *low_native, uint8_t *high_native,
                                         int n, int k, int qtype, int tile_k);
```

Constraints: `n % 256 == 0 && k % 256 == 0`, else rc=20. Unknown qtype → rc=22.

**The fold, which you never pass:**

```
run_bytes = tile_k * bits / 8
F         = run_bytes >= 32 ? 1 : 32 / run_bytes
```

32 is the AIU's contiguous-delivery floor. Both consumers compute this identically
(`moe_grouped_ppu.cuh:363`, `fpA_intB_ppu.cuh:151`) and `quactlize/formats.py:fold_for` is the third copy for
the Python side; if you need it in llama.cpp, transcribe it from one of those, not from here.

## 3. Producing it — Python (`quactlize/routes.py`)

```python
from quactlize import routes, formats

low, high, units = routes.prepare_fully_quantized_dense(blocks, n, k, qtype)                 # fixed arrangement
low, high, units = routes.prepare_fully_quantized_dense(blocks, n, k, qtype, tile_k=128)     # tile-aware
low, high, units = routes.prepare_fully_quantized_grouped(blocks, n, k, qtype, num_experts)  # MoE

# inverses, for telling a packing bug from a compute bug
native            = routes.dequantize_fully_quantized(low, high, units, n, k, qtype)
scale, zero       = routes.dequantize_scale_from_units(units, qtype)
```

`blocks` is the raw GGUF tensor viewed as `[n * k/256, block_bytes]`, CPU, contiguous.
The tuple's shape does not change with format: `high` is EMPTY for a single-plane one, so `units` is always the
LAST element. Anything that indexes it positionally should use `[-1]`.

Whole-model driver: `tools/pack_gguf.py` (`--dry-run` reports the type mix without touching the device).

**The arrangement is recorded per tensor, not per format** — `formats.PlacedArrangement(bits, tile_k, high_bits)`,
with `fold`/`high_fold` as derived properties. It stores `tile_k`, never `fold`: storing a derived value is how a
manifest comes to disagree with the kernel that reads it.

---

## 4. Consuming it — the GEMM symbols

```c
// DENSE. act is fp16 in a uint16_t*. out is fp16.
int quactlize_ppu_dense_fully_quantized(const uint16_t *act,
                                        const uint8_t *low, const uint8_t *high, const uint8_t *units,
                                        uint16_t *out,
                                        int m, int n, int k, int qtype);

// GROUPED (MoE). Experts are back to back in `low`/`high`/`units`; `rows_per_expert` is int[experts] and
// `total_rows` is their sum. act rows are grouped by expert in the same order.
int quactlize_ppu_grouped_fully_quantized(const uint16_t *act,
                                          const uint8_t *low, const uint8_t *high, const uint8_t *units,
                                          const int *rows_per_expert,
                                          uint16_t *out,
                                          int total_rows, int n, int k, int experts, int qtype);
```

Return codes worth handling by name:

| rc | meaning |
|---|---|
| 0 | ok |
| 30 | a null pointer, a non-positive extent, or `n % 256` / `k % 256` |
| 33 | **`qtype` is not the format this library was built for** |
| 34 | the library was built without `PPU_PACKED_SCALE=1` |
| 36 | scale-first was requested from a packed-format build |

**rc=33 is the one that will bite.** `PPU_PACKED_FORMAT` is a **compile-time** macro — `0=Q4_K, 1=Q5_K, 2=Q2_K,
3=Q3_K, 4=Q6_K` — so **one shared library serves exactly one k-quant**, and a `Q4_K_M` checkpoint is mixed by
construction. A deployment needs one library per format present in the file, loaded side by side. The host-side
loader already models this: `ppu_backend.cpp` keys its state on the format and resolves
`QUACTLIZE_PPU_LIB_FMT<k>` before falling back to a `_fmt<k>` splice of `QUACTLIZE_PPU_LIB`.

Any build must define `PPU_PACKED_SCALE=1` or every call returns 34.

---

## 5. What to build first

1. Pack one tensor with `prepare_dense_for_tile`, recover it with `recover_dense_for_tile`, compare to the input.
   That closes the producer with no device GEMM involved and no golden needed.
2. Wire the GEMM call site against the section-4 signatures, and make `rc != 0` an abort rather than a fallback:
   on a non-zero return `out` has not been written, so falling through feeds the model uninitialised memory.
3. Do not benchmark until the device-pointer variants land — the current ABI's per-call weight copy would
   dominate anything you measure, and the number would be an artifact of the seam, not of the kernel.
