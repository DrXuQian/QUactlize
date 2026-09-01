# llama.cpp K-pack integration handoff

This file is the single integration handoff for consuming Quactlize K-pack
artifacts from llama.cpp. Update it whenever the sidecar schema, public C ABI,
binary bundle, or loader contract changes.

Last updated: 2026-09-01 (develop; schema-v3 is current, and the complete host
producer is present in the source tree but not in the published runtime bundle).

## Runtime libraries

The currently published PPU0010 six-library bundle was built from
`5931ea04e5bbc94cd769f83314f713316fed2cf8` and is available locally at:

```text
/tmp/quactlize-runtime-artifact.HwHlQz/worktree/prebuilt/ppu0010/5931ea0/runtime6-59a6fabec958/bundle
```

Its durable Git authority is:

```text
origin/artifacts/ppu0010/5931ea0-runtime6-59a6fabec958
prebuilt/ppu0010/5931ea0/runtime6-59a6fabec958/bundle
```

Format selection is mandatory:

| GGML qtype | format | packed format | library |
|---:|---|---:|---|
| 10 | Q2_K, low2 Pack8 | FMT2 | `libquactlize_ppu_fmt2.so` |
| 11 | Q3_K, low2 Pack8 + high1 Pack16 | FMT3 | `libquactlize_ppu_fmt3.so` |
| 12 | Q4_K, low4 K-pack4 | FMT0 | `libquactlize_ppu_fmt0.so` |
| 13 | Q5_K, low4 Pack4 + high1 Pack16 | FMT1 | `libquactlize_ppu_fmt1.so` |
| 14 | Q6_K, low4 Pack4 + high2 Pack8 | FMT4 | `libquactlize_ppu_fmt4.so` |

`libquactlize_ppu.so` is the default Q4 ScaleFirst library. It is not the
format-unified fully-quantized decode library. After `dlopen`, call
`quactlize_ppu_build_packed_format_v1()` and require the exact FMT value before
using any arrangement-aware entry.

Before converting or uploading a tensor, obtain the descriptor from the
selected library rather than reconstructing it in llama.cpp:

```c
quactlize_ppu_placed_arrangement_v2 arrangement;
int rc = quactlize_ppu_canonical_arrangement_v2(ggml_qtype, &arrangement);
```

This host-only query succeeds only when `ggml_qtype` is the qtype owned by the
loaded FMT library. The default/ScaleFirst library and every other qtype fail;
on every non-null failure the output structure is all zero bytes. Treat every
nonzero result as a route decline, never as permission to retain a descriptor
returned by another library.

The published bundle already exports the arrangement-v2 dense and grouped
device APIs below. It predates the source-tree implementation of the complete
host producer/inverse and the loader-facing units query/producer contract, so
it must not be used for first-load conversion. A replacement six-library bundle
that passes the current `quactlize.ppu_bundle` export verifier is pending.

PPU runtime setup:

```bash
source /root/ppu-sdk/2.1.1/envsetup.sh
```

## Public headers

Use these declarations; do not duplicate the structs or function signatures in
llama.cpp:

```text
quactlize/include/quactlize_ppu_config.h
quactlize/include/quactlize_ppu_device.h
quactlize/include/quactlize_ppu_packed.h
```

They are included in the Python package data on current develop.

`quactlize_ppu_placed_arrangement_v2` is 40 bytes on the target ABI:

```c
typedef struct {
    int32_t  version;             // 2
    int32_t  layout;
    int32_t  bits;
    int32_t  high_bits;
    int32_t  artifact_tile_k;     // 0 for canonical K-pack
    int32_t  transport_tile_k;
    int32_t  group_size;
    int32_t  reserved;            // 0
    uint64_t mapping_id;
} quactlize_ppu_placed_arrangement_v2;
```

Canonical mapping IDs:

```text
Q4 K-pack4:       layout=1 mapping_id=0x51344b5034540001
Q2/Q3/Q5/Q6:     layout=2 mapping_id=0x514b504b54000001
```

## Fully-quantized device execution

Dense decode/prefill fallback:

```c
quactlize_ppu_dense_fully_quantized_workspace_bytes_for_arrangement_v2
quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v2
```

Grouped/MoE:

```c
quactlize_ppu_grouped_fully_quantized_workspace_bytes_for_arrangement_v2
quactlize_ppu_grouped_fully_quantized_dev_for_arrangement_v2
```

Contracts:

- `act`, `low`, `high`, `units`, `offsets`, `out`, and workspace are device
  pointers.
- Activation and output are FP16. llama.cpp must convert F32 to FP16 before the
  call and FP16 back to F32 afterward on the same stream.
- The workspace query returns `-1` when the qtype, shape, or arrangement is not
  admitted.
- A successful launch only enqueues work on the supplied stream.
- Q2/Q4 require `high == nullptr`; Q3/Q5/Q6 require a real high-plane pointer.
- Grouped `offsets` is cumulative device `int[experts+1]`; llama.cpp's existing
  expert bounds are directly usable.
- A K-pack buffer may never fall back to a raw GGUF reader after route decline.

The old raw vecdot device seam is
`quactlize_ppu_vecdot_dense_dev_v1(..., void * stream)`. Do not pass device
pointers to the host-pointer `quactlize_ppu_vecdot_dense` symbol.

## Persistent sidecar schema v3

The packer writes a sidecar directory, not a rewritten GGUF:

```text
SIDE_CAR/
  manifest.json
  weights.bin
```

`weights.bin` is headerless. Every tensor owns one 128-byte-aligned resident
region. The region contains canonical `low`, `high`, and `units` spans, each at
a 128-byte-aligned relative offset. For all admitted formats and geometries the
region is byte-neutral:

```text
region.size_bytes == original GGUF tensor nbytes
```

Top-level storage authority:

```json
"storage": {
  "file": "weights.bin",
  "size_bytes": 123,
  "alignment_bytes": 128,
  "sha256": "..."
}
```

Each tensor record contains:

```json
{
  "name": "blk.0.attn_q.weight",
  "ggml_type": 12,
  "type_name": "Q4_K",
  "route_class": "dense",
  "rank": 2,
  "n": 4096,
  "k": 4096,
  "experts": null,
  "arrangement_version": 2,
  "arrangement": {
    "layout": 1,
    "bits": 4,
    "high_bits": 0,
    "artifact_tile_k": 0,
    "transport_tile_k": 64,
    "group_size": 32,
    "reserved": 0,
    "mapping_id": 5851384623708504065
  },
  "source_tensor": {
    "index": 123,
    "data_offset": 456789,
    "size_bytes": 9437184,
    "sha256": "...",
    "binding_sha256": "..."
  },
  "region": { "offset_bytes": 0, "size_bytes": 9437184 },
  "spans": {
    "low":   { "offset_bytes": 0, "size_bytes": 8388608, "shape": [1,4096,2048], "sha256": "..." },
    "high":  { "offset_bytes": 8388608, "size_bytes": 0, "shape": [0], "sha256": "..." },
    "units": { "offset_bytes": 8388608, "size_bytes": 1048576, "shape": [16,4096,16], "sha256": "..." }
  }
}
```

This is a concrete Q4 dense shape example. Other qtypes and shapes have different
canonical descriptor fields and span extents; derive them from the manifest and
validate them against the public format contract.

`source_tensor.binding_sha256` is SHA-256 of compact UTF-8 JSON for this array:

```text
[name, ggml_type, rank, n, k, experts,
 source_index, source_data_offset, source_size_bytes, source_sha256]
```

The byte encoding is exactly:

```python
json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
```

There is no Unicode normalization and no trailing newline. Consequently a
non-ASCII tensor name is represented by JSON `\uXXXX` escapes (including a
surrogate pair when required), not by its literal UTF-8 bytes. Dense `experts`
is JSON `null`.

`source_tensor.data_offset` is an absolute byte offset from the start of the
source GGUF file. In GGUF terms its value is:

```text
tensor_data_start = align_up(end_of_metadata_and_tensor_info,
                             general.alignment or 32)
source_tensor.data_offset = tensor_data_start + tensor_info.offset
```

Do not store or compare the tensor-info-relative `tensor_info.offset` by itself.
In addition to checking the binding digest, llama.cpp must compare name,
ordinal, this absolute data offset, raw size, qtype and shape with the GGUF
inventory it already parsed.

Cache reuse requires all of the following:

1. Source GGUF size and whole-file SHA-256 match `manifest.source`.
2. Tensor inventory and each source byte-range digest match `source_tensor`.
3. Manifest has no duplicate keys or unlisted fields/files.
4. `weights.bin` whole-file and per-span hashes match.
5. Regions/spans are ordered, non-overlapping, canonically aligned and
   byte-neutral.
6. The complete arrangement-v2 is canonical for the qtype.

The loader should open manifest and blob with `O_NOFOLLOW`, validate and consume
each from one file descriptor, and reject row-split loading. Layer split is
allowed because a complete tensor remains on one device.

Schema v3 is deliberately a single-file contract: `manifest.source` binds one
regular GGUF file, every `source_tensor.data_offset` is absolute within that
same file, and `storage.file` names one `weights.bin`. It cannot represent a
split/sharded GGUF source because there is no source-file index per tensor. A
future multi-file schema must add per-file authorities and bind every tensor to
one of them; concatenating shard-relative offsets or treating them as offsets
in the first file is invalid. Runtime layer split does not relax this source
format limitation.

## Offline production and reuse

Create the persistent sidecar once:

```bash
QUACTLIZE_PPU_BUNDLE=/path/to/six-library/bundle \
  quactlize-pack-gguf MODEL.gguf MODEL.gguf.kpack
```

Subsequent loads validate and upload the saved regions; they do not repeat
placement. Do not overwrite or relabel the source GGUF.

The source tree now declares and implements the loader-facing complete host
producer/inverse:

```c
quactlize_ppu_prepare_fully_quantized_for_arrangement_v2
quactlize_ppu_recover_fully_quantized_for_arrangement_v2
```

These host-only entries take official GGUF blocks and separate low/high/units
pointers. Allocation of the units channel is governed by
`quactlize_ppu_units_bytes`; the standalone producers
`quactlize_ppu_prepare_units` and `quactlize_ppu_prepare_units_grouped` remain
part of the verified public bundle contract. The complete producer requires the
matching FMT library and complete canonical arrangement, and its inverse
supplies a byte-exact admission check. The legacy
`quactlize_ppu_prepare_fully_quantized_v1` produces Xplane and must not populate
the K-pack cache.

This source status is not a binary availability claim: the published
`5931ea0` bundle predates these loader-facing exports. Until a replacement
bundle is built and passes the verifier, `quactlize-pack-gguf` is the supported
producer for persistent sidecars and llama.cpp must decline in-process
first-load conversion when those symbols are absent.

## Integration order

1. Parse and validate schema-v3 sidecar against the source GGUF inventory.
2. Select the qtype-specific FMT library and verify its build identity.
3. Upload one complete tensor region and retain span offsets plus full
   arrangement-v2 in tensor metadata.
4. Route dense and grouped operations exclusively through the arrangement-v2
   device APIs.
5. Add optional first-load conversion only after the new complete host producer
   is present in the published six-library bundle.
6. Persist the result as the sidecar above; later loads never repack.
