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
| `quactlize_ppu_prepare_dense_for_arrangement_v2` | ″ | explicit physical byte map; canonical Q4 K-pack4 producer |
| `quactlize_ppu_recover_dense` | placed → native | the inverse |
| `quactlize_ppu_recover_dense_for_tile` | ″ | takes `tile_k` |
| `quactlize_ppu_recover_dense_for_arrangement_v2` | ″ | physical-layout-aware inverse |
| `quactlize_ppu_units_bytes` | query | dense `units` allocation size |
| `quactlize_ppu_prepare_units` | blocks → `units` | dense forward metadata producer |
| `quactlize_ppu_prepare_units_grouped` | blocks → `units` | expert-major forward producer |
| `quactlize_ppu_prepass_unit` | `units` → (scale, zero) | metadata inverse |
| `quactlize_ppu_dense_fully_quantized` | consume | packed dense GEMM |
| `quactlize_ppu_dense_fully_quantized_for_arrangement_v1` | consume | versioned artifact descriptor; never guesses fold |
| `quactlize_ppu_dense_fully_quantized_for_arrangement_v2` | consume | Xplane-v2 or canonical Q4 K-pack4 bytes |
| `quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v1` | consume device pointers | asynchronous descriptor-aware dense reader |
| `quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v2` | consume device pointers | async K-pack4 S1/S4 reader |
| `quactlize_ppu_list_valid_dense_fully_quantized_configs_for_arrangement_v1` | query | descriptor-aware tactic inventory |
| `quactlize_ppu_dense_fully_quantized_config_valid_for_arrangement_v1` | query | shared inventory/launch predicate |
| `quactlize_ppu_grouped_fully_quantized` | consume | packed MoE GEMM |
| `quactlize_ppu_grouped_fully_quantized_for_arrangement_v2` | consume | K-pack4 ragged MoE, descriptor required |
| `quactlize_ppu_grouped_fully_quantized_dev_for_arrangement_v2` | consume device pointers | asynchronous K-pack4 ragged MoE |
| `quactlize_ppu_list_valid_grouped_fully_quantized_configs_for_arrangement_v2` | query | arrangement-aware grouped tactics |
| `quactlize_ppu_vecdot`, `_dense`, `_moe` | consume RAW blocks | the other path, not this one |
| `quactlize_ppu_bc_gemv`, `_gemv_lowbit`, `_dense_lowbit` | consume | legacy/default-arrangement GEMV / low-bit variants |
| `quactlize_ppu_bc_gemv_for_arrangement_v1`, `_dev_v1` | consume | descriptor-aware BC host/device readers |
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

int quactlize_ppu_prepare_dense_for_arrangement_v2(
    const uint8_t *low_native, const uint8_t *high_native,
    uint8_t *low_layout, uint8_t *high_layout, int n, int k, int qtype,
    const quactlize_ppu_placed_arrangement_v2 *arrangement);
int quactlize_ppu_recover_dense_for_arrangement_v2(
    const uint8_t *low_layout, const uint8_t *high_layout,
    uint8_t *low_native, uint8_t *high_native, int n, int k, int qtype,
    const quactlize_ppu_placed_arrangement_v2 *arrangement);
```

`n % 256 == 0 && k % 256 == 0`, else **rc=20**. Unknown qtype → **rc=22**.

Per format:

| qtype | format | planes | code | default artifact TileK | packed tactic TileK | `units` unit, copyable |
|---|---|---|---|---:|---:|---|
| 10 | Q2_K | 1 | i2 | **128** | 256 | 20 B |
| 11 | Q3_K | 2 | i2 + i1 | 256 | 256 | 28 B (two superblocks paired) |
| 12 | Q4_K | 1 | i4 | **64** | 256 | 16 B |
| 13 | Q5_K | 2 | i4 + i1 | 256 | 256 | 16 B |
| 14 | Q6_K | 2 | i4 + i2 | **128** | **128** | 36 B (paired) |

`high_native` / `high_layout` are `nullptr` for the single-plane formats.

The first TileK identifies resident bytes; the second identifies a compute tactic.  They are deliberately separate.
An explicit `*_for_tile` producer records its own value instead of using the default. **Q6_K's 128 is not a
preference.** At artifact TK=256 the two-plane high map covers only half the logical K slots and
produces conditioned error 8.76e-1; the inverse is what caught it. Do not "fix" it to 256.

The table's **default artifact TileK** is the no-`tile_k` Python producer's scale-first placement.  It is not a
retroactive change to the unversioned readers: legacy dense fully-quantized Q2/Q4 and legacy BC still consume their
historical fully-quantized placement at artifact TileK 256.  Therefore bytes from the default Q2/Q4 producer must
go through the arrangement-aware successor; passing them to the old C entry is an ABI mismatch, not compatibility.

Q4 deployment now uses the v2 `q4-kpack4-transpose-v1` descriptor instead of either Q4 Xplane TileK. Its exact
identity is `(version=2, layout=1, bits=4, high_bits=0, artifact_tile_k=0, transport_tile_k=64,
group_size=32, reserved=0, mapping_id=0x51344b5034540001)`. Dense and grouped readers compare every field. The
same byte count with a missing or mutated descriptor is not a compatible artifact.

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

// Arrangement-aware successor. The descriptor is part of the artifact identity and is checked by the same
// predicate used by the v3 tactic inventory. Unknown/mismatched arrangements fail before launch.
int quactlize_ppu_dense_fully_quantized_for_arrangement_v1(
    const uint16_t *act, const uint8_t *low, const uint8_t *high, const uint8_t *units,
    uint16_t *out, int m, int n, int k, int qtype,
    const quactlize_ppu_placed_arrangement_v1 *arrangement, const char *config_name);

int quactlize_ppu_dense_fully_quantized_for_arrangement_v2(
    const uint16_t *act, const uint8_t *low, const uint8_t *high, const uint8_t *units,
    uint16_t *out, int m, int n, int k, int qtype,
    const quactlize_ppu_placed_arrangement_v2 *arrangement, const char *config_name);

// GROUPED (MoE). Experts lie back to back in low/high/units; rows_per_expert is int[experts] and total_rows is
// their sum; act rows are grouped by expert in the same order.
int quactlize_ppu_grouped_fully_quantized(const uint16_t *act,
                                          const uint8_t *low, const uint8_t *high, const uint8_t *units,
                                          const int *rows_per_expert,
                                          uint16_t *out,
                                          int total_rows, int n, int k, int experts, int qtype);

int quactlize_ppu_grouped_fully_quantized_for_arrangement_v2(
    const uint16_t *act, const uint8_t *low, const uint8_t *high, const uint8_t *units,
    const int *rows_per_expert, uint16_t *out,
    int total_rows, int n, int k, int experts, int qtype,
    const quactlize_ppu_placed_arrangement_v2 *arrangement, const char *config_name);
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

The current packed tensor collective accepts folded Q3/Q5/Q6 two-plane artifacts through this descriptor. A
single-plane Q2/Q4 artifact with `F>1` is deliberately fail-closed: that collective does not yet stage packed
metadata for the folded delivery. This is an implementation boundary, not permission to reinterpret it as F=1.

Device-runtime failures return 41; the library never calls `exit` for them. The output D2H is staged through a
private host buffer and committed only after the copy succeeds, so this rule includes a failed output copy rather
than merely the failures which occur before launch.

### Host-pointer and device-pointer entries

`ppu_dense_backend.cu:41-68`: every packed GEMM does
`DevBuf d(...); d.from_host(ptr); launch on the DEFAULT stream; rt_sync(); dout.to_host(out);`

The un-suffixed/config host entries allocate, copy, run, synchronize and copy back. They are correctness and
integration conveniences, not latency APIs. Production uses the `_dev_*` entries declared in
`quactlize_ppu_device.h`: all data and workspace pointers are device-resident, the caller supplies the stream, and
a zero return means enqueued rather than completed. Arrangement-v2 workspace queries must succeed before launch.

---

## 6. The Python path

This is the higher-level wrapper over the §3 producer, and how the installed
`quactlize-pack-gguf` command works.

```python
from quactlize import routes, formats

artifact = routes.prepare_fully_quantized_dense(blocks, n, k, formats.QuantType.Q4_K) # Q4 auto -> K-pack4
artifact = routes.prepare_fully_quantized_dense(blocks, n, k, qtype, layout="xplane") # compatibility Xplane
artifact = routes.prepare_fully_quantized_dense(blocks, n, k, qtype, tile_k=128)     # tile-aware
artifact = routes.prepare_fully_quantized_dense(
    blocks, n, k, formats.QuantType.Q4_K, layout="q4-kpack4")                        # production Q4
grouped = routes.prepare_fully_quantized_grouped(
    blocks, n, k, formats.QuantType.Q4_K, num_experts, layout="q4-kpack4")            # MoE

native      = routes.dequantize_fully_quantized(artifact, qtype)                 # descriptor selects inverse
scale, zero = routes.dequantize_scale_from_units(artifact[-1], qtype)            # metadata-only diagnostic
workspace   = routes.prepare_q4_kpack4_scale_workspace(artifact)                 # hoist once for prefill
out         = routes.matmul_q4_kpack4_dense(a, artifact, scale_workspace=workspace)
```

`blocks` is the raw GGUF tensor viewed as `[n * k/256, block_bytes]`, CPU, contiguous.

`PlacedArtifact` remains tuple-compatible: `high` is EMPTY for a single-plane format and `units` is always last.
Its versioned arrangement is nevertheless part of the object identity and survives
copy/pickle/manifest round trips. Stripping it to `tuple` is rejected by all three readers; they never ask the caller
to remember a parallel `tile_k` argument.

`matmul_q4_kpack4_dense` is the production shape dispatcher: M=1..8 selects the measured fully-quantized decode
policy, while larger M selects persistent ScaleFirst over the same code bytes. It refuses a missing prefill
workspace rather than hiding scale expansion inside the hot call.

Whole-model driver: `quactlize-pack-gguf MODEL.gguf OUT_DIR [--dry-run]`. `--dry-run` reports the type mix
— i.e. how many format-specific libraries a deployment of that file needs — without touching the device. Recognised
rank-2 GGML `MUL_MAT` tensors use the dense producer. Recognised rank-3 fast-first `[K,N,E]` K-quant tensors use the
grouped producer and retain `route_class=grouped`, `experts=E`, and the v2 descriptor in the manifest; they are not
silently flattened into a dense matrix. Unknown and non-matrix roles fail closed instead of being guessed from rank.
The output directory is a persistent sidecar, not a rewritten GGUF. Bundle schema v3 stores a single headerless
`weights.bin`; each tensor owns one 128-byte-aligned, byte-neutral region with explicit low/high/units spans. It
also records the source GGUF byte size and SHA-256 plus each tensor's source ordinal, absolute data range and raw
SHA-256. A native loader cross-checks those fields against its parsed GGUF inventory; the manifest binding catches a
record-name/range reassociation before upload. A cache consumer calls
`load_kpack_bundle(OUT_DIR, source=MODEL.gguf)` before reuse. The `model`
path is diagnostic, so an identical model may move, while a different model at the same path is rejected. Omitting
`source=` validates only the sidecar's internal bytes and cannot authorize a cache hit.

The default Q4 pack command names the format-selected library explicitly:

```bash
QUACTLIZE_PPU_LIB_FMT0=/path/to/libquactlize_ppu.so \
  quactlize-pack-gguf MODEL.gguf OUT_DIR
```

Online loaders use the same producer rather than reimplementing GGUF field extraction:

```c
quactlize_ppu_prepare_fully_quantized_for_arrangement_v2(
    raw_gguf_blocks, low, high, units, n, k, experts, qtype, &arrangement);
```

The inverse `quactlize_ppu_recover_fully_quantized_for_arrangement_v2` restores official GGUF blocks for a
byte-exact admission check. Both are host-only and fail closed on a null or mismatched complete arrangement-v2;
single-plane formats require `high == nullptr`, while multi-plane formats require a real high buffer. All ranges
must be distinct. The older `_v1` complete seam remains a TileK/Xplane compatibility entry and must not produce the
K-pack cache or feed an arrangement-v2 device consumer.

`quactlize.gguf_backend_for_qtype(12)` reports that exact handle. `gguf_backend()` continues to report the
legacy/default handle used by Xplane and by ScaleFirst execution; the two queries are intentionally not aliases.

The arrangement is recorded **per tensor** as a complete `PlacedArrangementV2`. `quactlize.pack_gguf` has no layout
switch: whole-model production packing emits K-pack4 for Q4_K and the canonical per-plane K-pack mapping for
Q2_K/Q3_K/Q5_K/Q6_K. Explicit Xplane producers remain development compatibility paths only. Different layers may
use different arrangements; every batch/M for one stored layer must use that layer's one descriptor.

---

## 7. Order to build in

1. **Round-trip one tensor**: `prepare_dense_for_tile` → `recover_dense_for_tile` → compare with the input. That
   closes the producer with no GEMM and no golden involved, and it is the check that separates a packing bug
   from a compute bug later.
2. **Wire the §5 call site**, with `rc != 0` as an abort.
3. **Produce `units` with §3** (or use the §6 wrapper), and run `prepass_unit` as its independent inverse.
4. **Do not benchmark** until the device-pointer variants exist.
