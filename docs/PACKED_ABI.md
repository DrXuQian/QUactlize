# `libquactlize_ppu.so` — the packed (FULLY_QUANTIZED) path, producer and consumer

For the llama.cpp side. `quactlize_ppu_vecdot_dense` / `_moe` read **raw GGUF blocks**; the packed GEMM takes a
different input and is a different entry point. Everything below is a symbol **in the shared library** — nothing
here asks you to compile a source file.

The shared library produces all three resident buffers: code-plane placement and the byte-neutral metadata reorder
are both host-pointer APIs, while the GEMM entries consume their results.

---

## 1. The whole symbol table

Everything `extern "C"` in the library today. Grouped by what it is for, so a missing pair is visible.

| symbol | direction | note |
|---|---|---|
| `quactlize_ppu_prepare_dense` | native → placed code planes | fixed per-format arrangement |
| `quactlize_ppu_prepare_dense_for_tile` | ″ | takes `tile_k` |
| `quactlize_ppu_recover_dense` | placed → native | the inverse |
| `quactlize_ppu_recover_dense_for_tile` | ″ | takes `tile_k` |
| `quactlize_ppu_units_bytes` | query | dense `units` allocation size |
| `quactlize_ppu_prepare_units` | blocks → `units` | dense forward metadata producer |
| `quactlize_ppu_prepare_units_grouped` | blocks → `units` | expert-major forward producer |
| `quactlize_ppu_prepass_unit` | `units` → (scale, zero) | metadata inverse |
| `quactlize_ppu_dense_fully_quantized` | consume | packed dense GEMM |
| `quactlize_ppu_grouped_fully_quantized` | consume | packed MoE GEMM |
| `quactlize_ppu_vecdot`, `_dense`, `_moe` | consume RAW blocks | the other path, not this one |
| `quactlize_ppu_bc_gemv`, `_gemv_lowbit`, `_dense_lowbit` | consume | GEMV / low-bit variants |
| `quactlize_ppu_dequantize`, `_prepass` | consume RAW | dequant and online scale prepass |

## 2. The asymmetry that decides how many libraries you build

**The packer is format-INDEPENDENT; the GEMM is format-SPECIFIC.**

`ppu_dense_layout.cu` and `ppu_unit_pack.cpp` contain **zero** `PPU_PACKED_FORMAT` references and switch on `qtype`
10..14 at run time. `ppu_dense_backend.cu` contains **sixteen** — `PPU_PACKED_FORMAT` is a compile-time macro
(`0=Q4_K, 1=Q5_K, 2=Q2_K, 3=Q3_K, 4=Q6_K`).

So **one build packs all five formats, and that same build computes exactly one.** Offline packing needs a
single library, whichever format it was built for. Serving a `Q4_K_M` checkpoint — mixed by construction — needs
one library per format present in the file, loaded side by side. `ppu_backend.cpp` already models that: it keys
its loader state on the format and resolves `QUACTLIZE_PPU_LIB_FMT<k>` before falling back to a `_fmt<k>` splice
of `QUACTLIZE_PPU_LIB`.

Every build must define `PPU_PACKED_SCALE=1` or every packed call returns **34**.

---

## 3. Producing the `units` channel

Both GEMM entries take **(low, high, units)**. These exported host functions provide the third buffer directly
from official GGUF blocks:

```c
int     quactlize_ppu_prepare_units(const uint8_t *blocks, uint8_t *units, int n, int k, int qtype);
int     quactlize_ppu_prepare_units_grouped(const uint8_t *blocks, uint8_t *units,
                                            int n, int k, int experts, int qtype);
int64_t quactlize_ppu_units_bytes(int n, int k, int qtype);        // so a caller can allocate
```

`quactlize_ppu_units_bytes` returns the dense byte count, or `-1` for an invalid shape/qtype. A grouped allocation
is `experts * quactlize_ppu_units_bytes(...)`. Q3_K and Q6_K pair two adjacent K-superblocks from the same column;
the query owns that rule, so the caller must not reproduce it.

The dense source is `[N,K/256,raw_block_bytes]` and destination is `[K-unit,N,unit_bytes]`. Grouped adds an
expert-major outer axis to both. `n` and `k` must be positive multiples of 256; Q3_K/Q6_K additionally require
`k % 512 == 0`. Producer returns are 0 on success, 20 for invalid pointers/extents, 22 for unknown qtype, and 24
when the format's superblocks cannot form a complete unit.

The torch producer calls these same symbols. `gguf_unit_pack.hpp` owns the single expert-aware packing loop used by
both dense and grouped, so the exported implementation is not a second reorder which can drift from Python.

---

## 4. Producing the code planes

```c
// qtype is the ggml enum: 10=Q2_K 11=Q3_K 12=Q4_K 13=Q5_K 14=Q6_K.
int quactlize_ppu_prepare_dense(const uint8_t *low_native, const uint8_t *high_native,
                                uint8_t *low_layout, uint8_t *high_layout,
                                int n, int k, int qtype);

int quactlize_ppu_recover_dense(const uint8_t *low_layout, const uint8_t *high_layout,
                                uint8_t *low_native, uint8_t *high_native,
                                int n, int k, int qtype);

// Tile-aware. SEPARATE SYMBOLS ON PURPOSE: an extension built against the older library cannot pass the extra
// integer to it and get silence.
int quactlize_ppu_prepare_dense_for_tile(const uint8_t *low_native, const uint8_t *high_native,
                                         uint8_t *low_layout, uint8_t *high_layout,
                                         int n, int k, int qtype, int tile_k);
int quactlize_ppu_recover_dense_for_tile(const uint8_t *low_layout, const uint8_t *high_layout,
                                         uint8_t *low_native, uint8_t *high_native,
                                         int n, int k, int qtype, int tile_k);
```

`n % 256 == 0 && k % 256 == 0`, else **rc=20**. Unknown qtype → **rc=22**.

Per format:

| qtype | format | planes | code | TileK | `units` unit, copyable |
|---|---|---|---|---|---|
| 10 | Q2_K | 1 | i2 | 256 | 20 B |
| 11 | Q3_K | 2 | i2 + i1 | 256 | 28 B (two superblocks paired) |
| 12 | Q4_K | 1 | i4 | 256 | 16 B |
| 13 | Q5_K | 2 | i4 + i1 | 256 | 16 B |
| 14 | Q6_K | 2 | i4 + i2 | **128** | 36 B (paired) |

`high_native` / `high_layout` are `nullptr` for the single-plane formats.

**Q6_K's 128 is not a preference.** At TK=256 the two-plane high map covers only half the logical K slots and
produces conditioned error 8.76e-1; the inverse is what caught it. Do not "fix" it to 256.

The pairing rule behind the unit size is `supers = 2 if per_superblock_meta % 4 else 1`, because `ppu.cp.async`
moves only 4, 8 or 16 bytes and Q3's 14 and Q6's 18 are movable at none of them.

**The fold is never a parameter.** Both consumers derive it, identically, from `(bits, tile_k)`:

```
run_bytes = tile_k * bits / 8
F         = run_bytes >= 32 ? 1 : 32 / run_bytes        // 32 = the AIU contiguous-delivery floor
```

A caller-supplied fold would be a second source for a value the kernel already computes, and disagreement is not
a crash — the weight is placed for one fold and read at another, giving finite wrong numbers.
(`moe_grouped_ppu.cuh:363`, `fpA_intB_ppu.cuh:151`, `quactlize/formats.py:fold_for`.)

## 5. Consuming it

```c
// DENSE. act is fp16 carried in uint16_t*; out is fp16.
int quactlize_ppu_dense_fully_quantized(const uint16_t *act,
                                        const uint8_t *low, const uint8_t *high, const uint8_t *units,
                                        uint16_t *out,
                                        int m, int n, int k, int qtype);

// GROUPED (MoE). Experts lie back to back in low/high/units; rows_per_expert is int[experts] and total_rows is
// their sum; act rows are grouped by expert in the same order.
int quactlize_ppu_grouped_fully_quantized(const uint16_t *act,
                                          const uint8_t *low, const uint8_t *high, const uint8_t *units,
                                          const int *rows_per_expert,
                                          uint16_t *out,
                                          int total_rows, int n, int k, int experts, int qtype);
```

| rc | meaning |
|---|---|
| 0 | ok |
| 30 | null pointer, non-positive extent, or `n % 256` / `k % 256` |
| **33** | **`qtype` is not the format this library was built for** — see §2 |
| 34 | built without `PPU_PACKED_SCALE=1` |
| 36 | scale-first requested from a packed-format build |
| **41** | **device allocation, H2D/D2H copy, synchronize, or host output-staging failure** |

On any non-zero return **`out` has not been written**. Abort; do not fall through to another kernel, or the model
consumes uninitialised memory.

Device-runtime failures return 41; the library never calls `exit` for them. The output D2H is staged through a
private host buffer and committed only after the copy succeeds, so this rule includes a failed output copy rather
than merely the failures which occur before launch.

### The pointer domain, which is not what ggml wants

`ppu_dense_backend.cu:41-68`: every packed GEMM does
`DevBuf d(...); d.from_host(ptr); launch on the DEFAULT stream; rt_sync(); dout.to_host(out);`

For the **producer** that is correct — packing is offline and the caller has host buffers. For the **consumer**
it is the defect already recorded against the vecdot path (`.coord/INBOX.md` 030): ggml hands you a device
pointer that is already resident, and this ABI treats it as host memory, re-copies the entire weight per call,
runs on the default stream and synchronises the device.

**Build the call site against these signatures; do not benchmark through them.** Any number you get is a
measurement of the seam, not of the kernel. Device-pointer variants — same arguments plus a stream, no
allocation, no sync, no copies — are queued and deferred.

---

## 6. The Python path

This is the higher-level wrapper over the §3 producer, and how `tools/pack_gguf.py` works.

```python
from quactlize import routes, formats

low, high, units = routes.prepare_fully_quantized_dense(blocks, n, k, qtype)                 # fixed arrangement
low, high, units = routes.prepare_fully_quantized_dense(blocks, n, k, qtype, tile_k=128)     # tile-aware
low, high, units = routes.prepare_fully_quantized_grouped(blocks, n, k, qtype, num_experts)  # MoE

native      = routes.dequantize_fully_quantized(low, high, units, n, k, qtype)   # inverses, for telling a
scale, zero = routes.dequantize_scale_from_units(units, qtype)                   # packing bug from a compute bug
```

`blocks` is the raw GGUF tensor viewed as `[n * k/256, block_bytes]`, CPU, contiguous.

The returned tuple's **shape does not vary with format**: `high` is EMPTY for a single-plane one, so `units` is
always the LAST element. Index it as `[-1]`, never as `[1]` or `[2]`.

Whole-model driver: `python3 tools/pack_gguf.py MODEL.gguf OUT_DIR [--dry-run]`. `--dry-run` reports the type mix
— i.e. how many format-specific libraries a deployment of that file needs — without touching the device.

The arrangement is recorded **per tensor**: `formats.PlacedArrangement(bits, tile_k, high_bits)`, with `fold` and
`high_fold` as derived properties. It stores `tile_k` and never `fold`, because storing a derived value is how a
manifest comes to disagree with the kernel that reads it.

---

## 7. Order to build in

1. **Round-trip one tensor**: `prepare_dense_for_tile` → `recover_dense_for_tile` → compare with the input. That
   closes the producer with no GEMM and no golden involved, and it is the check that separates a packing bug
   from a compute bug later.
2. **Wire the §5 call site**, with `rc != 0` as an abort.
3. **Produce `units` with §3** (or use the §6 wrapper), and run `prepass_unit` as its independent inverse.
4. **Do not benchmark** until the device-pointer variants exist.
