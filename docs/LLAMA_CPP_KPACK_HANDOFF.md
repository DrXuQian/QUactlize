# llama.cpp K-pack integration handoff

This file is the single integration handoff for consuming Quactlize K-pack
artifacts from llama.cpp. Update it whenever the sidecar schema, public C ABI,
binary bundle, or loader contract changes.

Last updated: 2026-09-02. Persistent sidecar schema v3 is current. The
published `8f9fa07` measured-policy runtime bundle has passed strict binary
inspection, its selected-config oracle, and all 26 host ABI cases in a fresh
LFS checkout. Its PPU device gate is still **PENDING**. Host/ELF admission is
not device admission and does not authorize deployment by itself.

## Runtime libraries

The published PPU0010 six-library bundle was built from clean source commit
`8f9fa07de9694901a5db91d546d6c994720f86b1`. Its immutable artifact commit is
`f7f55d61ee1a58657f99df24876aa3bbb13d1a45`, whose only parent is that source
commit. The durable Git authority is:

```text
origin/artifacts/ppu0010/8f9fa07-runtime6-b3eb070bc65f
prebuilt/ppu0010/8f9fa07/runtime6-b3eb070bc65f/bundle
```

The bundle manifest SHA-256 is
`b3eb070bc65f42d5443626aa82baac468863657d5f479021caccba2d36f75097`.
Its manifest uses the separate `quactlize.ppu-runtime-bundle` schema v1; never
parse it as the persistent sidecar schema v3. No workstation cache path is an
authority.

Use two independent worktrees: a detached artifact worktree owns only bundle
hydration, while a clean develop worktree owns the verifier/oracle/gate
runners. First create the artifact worktree and hydrate its six LFS objects:

```bash
SOURCE_COMMIT=8f9fa07de9694901a5db91d546d6c994720f86b1
ARTIFACT_COMMIT=f7f55d61ee1a58657f99df24876aa3bbb13d1a45
ARTIFACT_BRANCH=artifacts/ppu0010/8f9fa07-runtime6-b3eb070bc65f
ARTIFACT_REL=prebuilt/ppu0010/8f9fa07/runtime6-b3eb070bc65f
ARTIFACT_WORKTREE=/workspace/quactlize-runtime-artifact-8f9fa07-b3eb070bc65f
RUNNER_WORKTREE=/workspace/quactlize-develop-runner-8f9fa07

git fetch origin \
  "refs/heads/${ARTIFACT_BRANCH}:refs/remotes/origin/${ARTIFACT_BRANCH}"
git fetch origin \
  refs/heads/develop:refs/remotes/origin/develop
test "$(git rev-parse "origin/${ARTIFACT_BRANCH}")" = "${ARTIFACT_COMMIT}"
test "$(git rev-list --parents -n 1 "${ARTIFACT_COMMIT}")" = \
  "${ARTIFACT_COMMIT} ${SOURCE_COMMIT}"
git worktree add --detach "${ARTIFACT_WORKTREE}" "${ARTIFACT_COMMIT}"
git -C "${ARTIFACT_WORKTREE}" lfs pull \
  --include="${ARTIFACT_REL}/bundle/*.so" \
  --exclude=""
git worktree add --detach "${RUNNER_WORKTREE}" origin/develop
git -C "${RUNNER_WORKTREE}" submodule update --init --recursive

BUNDLE="${ARTIFACT_WORKTREE}/${ARTIFACT_REL}/bundle"
test "$(sha256sum "${BUNDLE}/manifest.json" | awk '{print $1}')" = \
  b3eb070bc65f42d5443626aa82baac468863657d5f479021caccba2d36f75097
```

The artifact carries the strict verifier used for admission. It is pinned by
both content and Git identity:

```text
SHA-256  43266a59b0676f19a34740d46fecbbdb2fd1ab80d88ee3911765f4f2ca5a21e7
Git blob be9006407cd37ac21a861cdb9fc658f597a5188d
```

Run it from the artifact worktree, then run the source-pinned selected-config
oracle from the clean develop checkout:

```bash
ARTIFACT_REL=prebuilt/ppu0010/8f9fa07/runtime6-b3eb070bc65f
ARTIFACT_WORKTREE=/workspace/quactlize-runtime-artifact-8f9fa07-b3eb070bc65f
RUNNER_WORKTREE=/workspace/quactlize-develop-runner-8f9fa07
BUNDLE="${ARTIFACT_WORKTREE}/${ARTIFACT_REL}/bundle"
cd "${RUNNER_WORKTREE}"
source "${PPU_SDK:?set PPU_SDK to the admitted SDK root}/envsetup.sh"

test "$(sha256sum "${ARTIFACT_WORKTREE}/${ARTIFACT_REL}/verify_bundle.py" | awk '{print $1}')" = \
  43266a59b0676f19a34740d46fecbbdb2fd1ab80d88ee3911765f4f2ca5a21e7
test "$(git -C "${ARTIFACT_WORKTREE}" hash-object "${ARTIFACT_REL}/verify_bundle.py")" = \
  be9006407cd37ac21a861cdb9fc658f597a5188d
python3 "${ARTIFACT_WORKTREE}/${ARTIFACT_REL}/verify_bundle.py" "${BUNDLE}" \
  --ppu-sdk "${PPU_SDK:?set PPU_SDK to the admitted SDK root}"

test "$(sha256sum tools/verify_kquant_selected_config.py | awk '{print $1}')" = \
  43d5c0fb1ce07020489ff9e85e0528116e5a5cc920aef54ce388330bed209eea
test "$(git hash-object tools/verify_kquant_selected_config.py)" = \
  a0d355d351a579a46959ad03cdce52fb22df15e7
python3 tools/verify_kquant_selected_config.py "${BUNDLE}"
```

Do not use `--manifest-only`: it omits exported-symbol and embedded-image
inspection. Setting `QUACTLIZE_PPU_BUNDLE` only selects a directory and does
not perform this verification automatically.

The published libraries and the pinned headers include these host-only
selected-config exports:

```text
quactlize_ppu_dense_fully_quantized_selected_config_for_arrangement_v2
quactlize_ppu_grouped_fully_quantized_selected_config_for_arrangement_v2
```

Format selection is mandatory:

| GGML qtype | format | packed format | library |
|---:|---|---:|---|
| 10 | Q2_K, low2 Pack8 | FMT2 | `libquactlize_ppu_fmt2.so` |
| 11 | Q3_K, low2 Pack8 + high1 Pack16 | FMT3 | `libquactlize_ppu_fmt3.so` |
| 12 | Q4_K, low4 K-pack4 | FMT0 | `libquactlize_ppu_fmt0.so` |
| 13 | Q5_K, low4 Pack4 + high1 Pack16 | FMT1 | `libquactlize_ppu_fmt1.so` |
| 14 | Q6_K, low4 Pack4 + high2 Pack8 | FMT4 | `libquactlize_ppu_fmt4.so` |

`libquactlize_ppu.so` is the default Q4 ScaleFirst library. It is not a
fully-quantized FMT library. After `dlopen`, call
`quactlize_ppu_build_packed_format_v1()` and require the exact FMT value before
using any arrangement-aware entry.

The minimal llama.cpp integration documented here uses the qtype-selected FMT
fully-quantized path for every M, including Q4 prefill. The default ScaleFirst
library is therefore not required by this first integration. Quactlize's tuned
Q4 dispatcher uses ScaleFirst at M>=64, but its packed-units-to-FP16 metadata
expansion is not yet a public loader ABI; do not resolve an undocumented
prepass symbol to reproduce that route.

The product host floor is Ubuntu 24.04 with the bundle's admitted PPU SDK
2.1.1-a5c56e runtime. Load the SDK wrapper first with
`RTLD_NOW | RTLD_GLOBAL`, then load every Quactlize DSO with
`RTLD_NOW | RTLD_LOCAL`:

```text
${PPU_SDK}/lib/libhggc_wrapper.so -> RTLD_NOW | RTLD_GLOBAL
absolute path to selected FMT DSO -> RTLD_NOW | RTLD_LOCAL
```

Resolve every Quactlize function with `dlsym` on that format library's returned
handle. The six libraries intentionally export the same C symbol names, so a
process-global lookup can let the first loaded format answer for all later
formats. The default library reports build identity `-1`; the five
format-selected libraries report FMT0 through FMT4. A private host-loader shim
or a developer-only loader environment variable is not part of the deployment
contract.

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

The published bundle exports the arrangement-v2 dense/grouped device APIs, the
canonical descriptor query, complete host producer/inverse, packed-units ABI,
selected-config ABI, and ScaleFirst validity seam required by the strict
verifier. In a fresh LFS clone, the host-only ABI suite passed 26 cases over all
five formats, including dense/grouped exact round trips and fail-closed
negatives. The frozen strict verifier and selected-config oracle also passed.
PPU numerical execution remains `PENDING`; no performance claim is attached to
this bundle until its device gate passes.

PPU runtime setup uses the SDK's public environment script. The exact install
root is deployment-owned:

```bash
source "${PPU_SDK:?set PPU_SDK to the admitted SDK root}/envsetup.sh"
```

## Public headers

The runtime-bundle directory intentionally contains only its manifest and six
shared libraries. Pin the public headers from the exact source commit recorded
by that runtime manifest; do not duplicate the structs or function signatures
in llama.cpp:

```text
quactlize/include/quactlize_ppu_config.h
quactlize/include/quactlize_ppu_device.h
quactlize/include/quactlize_ppu_packed.h
```

They are included in both Python package manifests at source commit
`8f9fa07de9694901a5db91d546d6c994720f86b1`. Pin those exact three headers to
the six published libraries. Do not substitute headers from another source
revision, even when a structure name or export name appears unchanged.
`quactlize_ppu_config.h` at this commit contains config v3/v4 and both
selected-config declarations.

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

### Selected-config observability

The published bundle exposes the exact tactic chosen by the same host policy
used by workspace queries and launches. These calls are host-only: they create
no PPU context and enqueue no work.

```c
quactlize_ppu_dense_fully_quantized_selected_config_for_arrangement_v2
quactlize_ppu_grouped_fully_quantized_selected_config_for_arrangement_v2
```

For dense K-pack Q2/Q3/Q5/Q6, a null or empty requested name selects an exact
measured `(qtype,M,N,K)` point when present, then falls back to the established
compiled M-dependent default. Q4 layout 1 has its own shape policy over
`(M,N,K)` and returns its fixed Split-K value explicitly in the v4 record. A
known nonempty name is an explicit override. An unknown nonempty name fails and
clears the output record. Grouped currently exposes only
explicit/compiled-default selection because its measured sweep depends on the
full per-expert row distribution, which is not represented by the public
aggregate arguments.

The strict verifier and selected-config oracle for this bundle are the two
host-only gates shown in the runtime acquisition section. The dense result is
`quactlize_ppu_config_v4`, including `split_k_slices`; the grouped result is
`quactlize_ppu_config_v3`. Success is exactly `1`. Failure is `0` and clears
the complete output record. Do not bind the dense export to the older v3
record or infer split-K from the config name.
On success, `name` points to storage owned by the loaded DSO. Keep that DSO
loaded while using the pointer, or copy the string into loader-owned storage.

Device admission uses the already-hydrated artifact worktree and a separate,
clean develop checkout. The runner must be tracked and have SHA-256
`01ee28df26d8cc6c5cfe66ec94b99a9198794f390fb8e032cdfff5014c53ac0f`.
Its source-authority checks require the runtime implementation and submodule
commits to remain identical to bundle source `8f9fa07`; documentation-only
develop commits are allowed. On an Ubuntu 24.04 PPU box:

```bash
RUNNER_WORKTREE=/workspace/quactlize-develop-runner-8f9fa07
ARTIFACT_REL=prebuilt/ppu0010/8f9fa07/runtime6-b3eb070bc65f
ARTIFACT_WORKTREE=/workspace/quactlize-runtime-artifact-8f9fa07-b3eb070bc65f
BUNDLE="${ARTIFACT_WORKTREE}/${ARTIFACT_REL}/bundle"
cd "${RUNNER_WORKTREE}"

source "${PPU_SDK:?set PPU_SDK to the admitted SDK root}/envsetup.sh"
python3 -c 'import importlib.metadata as m; assert m.version("gguf") == "0.19.0"'
test "$(sha256sum tools/run_prebuilt_ppu_box_gate.py | awk '{print $1}')" = \
  01ee28df26d8cc6c5cfe66ec94b99a9198794f390fb8e032cdfff5014c53ac0f
CUDA_VISIBLE_DEVICES=0 \
  python3 tools/run_prebuilt_ppu_box_gate.py "${BUNDLE}" \
    --ppu-sdk "${PPU_SDK:?set PPU_SDK to the admitted SDK root}" \
    --output /workspace/quactlize-prebuilt-gate-8f9fa07
```

`CUDA_VISIBLE_DEVICES` must be one numeric ordinal and the runtime must expose
exactly one device. This gate compiles and links nothing: it loads the SDK
wrapper globally, the six prebuilt Quactlize DSOs locally, and executes the
public ctypes ABI. It covers all five formats at dense measured shape
`M=1,N=1024,K=5120` and grouped empty-expert shape
`rows=[2,0,3,1],N=256,K=512`, with official `gguf==0.19.0` plus independent
NumPy FP64 oracles and planted faults. It writes a new evidence directory and
refuses to overwrite one. Device status remains `PENDING` until this exact gate
reports `PASS` for the published files.

Dense fully-quantized decode/prefill:

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
- Pass `config_name == nullptr` unless the name came from the matching
  arrangement-aware inventory and passed that route's validity predicate. An
  unknown nonempty name is a hard decline, not a request for fallback.
- Keep activations, weights, outputs, offsets, and workspace alive until the
  caller's stream has completed the launch.
- Q2/Q4 require `high == nullptr`; Q3/Q5/Q6 require a real high-plane pointer.
  Although a Q2/Q4 schema-v3 manifest records the empty high span with its
  canonical offset and shape `[0]`, do not turn that offset into a non-null
  API pointer.
- Grouped `offsets` is a nondecreasing device `int32_t[experts+1]` with
  `offsets[0] == 0`, `offsets[experts] == total_rows`, and every expert extent
  at most `max_rows`. Require positive `experts`, `total_rows`, and `max_rows`,
  with `max_rows <= total_rows`.
- A K-pack buffer may never fall back to a raw GGUF reader after route decline.

All resident artifacts, workspace queries, and launches require positive N and
K multiples of 256; Q3_K/Q6_K additionally require `K % 512 == 0`. Dense also
requires positive M. Do not mirror the remaining tactic-specific shape policy
in llama.cpp: the workspace query and config-valid predicate are the admission
authority.

The old raw vecdot device seam is
`quactlize_ppu_vecdot_dense_dev_v1(..., void * stream)`. Do not pass device
pointers to the host-pointer `quactlize_ppu_vecdot_dense` symbol.

## Persistent sidecar schema v3

This sidecar schema is independent of the runtime-library bundle schema. Its
identity fields are:

```text
schema = "quactlize.kquant-kpack.bundle"
schema_version = 3
arrangement_version = 2
```

The top-level object contains exactly these keys:

```text
schema, schema_version, arrangement_version, model, selection,
source, storage, tensors, skipped
```

`model` records the nonempty source name supplied to the packer. Cache
authority comes from `source`, not from that path string:

```json
"source": {
  "format": "gguf",
  "size_bytes": 10485760,
  "sha256": "..."
}
```

Production selection is also explicit:

```json
"selection": {
  "layout_policy": "production-kpack-only",
  "packable_total": 42,
  "packed": 42,
  "skipped": 7
}
```

`tensors` contains every packable Q2_K/Q3_K/Q4_K/Q5_K/Q6_K dense or grouped
weight. `skipped` is the exact inventory of non-packable tensors, with records
containing only nonempty string `name`, `type_name`, and `reason` values. A
valid product sidecar has a nonempty `tensors` list, every selection count is a
nonnegative integer, `packable_total == packed == len(tensors)`, and
`selection.skipped == len(skipped)`. It never silently omits a packable weight.
Packed and skipped names are each unique and the two sets are disjoint.

The packer writes a sidecar directory, not a rewritten GGUF:

```text
SIDE_CAR/
  manifest.json
  weights.bin
```

`weights.bin` is headerless. Every packed tensor owns one 128-byte-aligned
resident region. The region contains canonical `low`, `high`, and `units`
spans, each at a 128-byte-aligned relative offset. For all admitted formats and
geometries the region is byte-neutral:

```text
region.size_bytes == original GGUF tensor nbytes
```

Top-level storage authority:

```json
"storage": {
  "file": "weights.bin",
  "size_bytes": 9437184,
  "alignment_bytes": 128,
  "sha256": "..."
}
```

Each packed-tensor record contains:

```json
{
  "name": "blk.0.attn_q.weight",
  "ggml_type": 12,
  "type_name": "Q4_K",
  "route_class": "dense",
  "layout_name": "q4-kpack4",
  "plane_packs": { "low": 4, "high": 0 },
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

This is a concrete Q4 dense shape example. Other qtypes and shapes have
different canonical descriptor fields and span extents; compute their expected
values independently as specified below.

The JSON `arrangement` deliberately omits `version`. Set the C structure's
`version` from the tensor record's `arrangement_version`; both the top-level and
tensor-level arrangement versions must equal 2. Copy the remaining eight fields
exactly. Parse `mapping_id`, offsets, extents, and byte counts as lossless
integers, never through an IEEE-754 double.

Read candidate values from the manifest, then independently compute the
expected layout name, plane packs, and span shapes from qtype, route, N, K, and
experts. Query the selected FMT library for the expected complete arrangement
and require exact equality before upload. Production sidecars admit only Q4
layout 1 and Q2/Q3/Q5/Q6 layout 2; reject Xplane layout 0 and experimental
direct layout 3 even though the public enum still reserves those values.

Let E=1 for dense and E=`experts` for grouped. The canonical span shapes are:

```text
low:   [E, N, K*low_bits/8]
high:  [0] when high_bits==0, otherwise [E, N, K*high_bits/8]
units: [K/(256*S), N, U] for dense
       [E, K/(256*S), N, U] for grouped
```

The `(S,U)` packed-unit pairs are Q2 `(1,20)`, Q3 `(2,28)`, Q4 `(1,16)`,
Q5 `(1,16)`, and Q6 `(2,36)`. For every span, require `size_bytes` to equal the
exact product of its canonical shape (zero for `[0]`).

Dense records require `(route_class, rank, experts) == ("dense", 2, null)` and
source GGUF shape `[K,N]` with a `MUL_MAT` consumer. Grouped records require
`("grouped", 3, E>0)`, source shape `[K,N,E]`, and a `MUL_MAT_ID` consumer.
Embeddings, SSM weights, unknown roles, and every `skipped` record remain on
their original route.

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

Regions start at offset zero in manifest order and continuously cover the
entire storage file. Within each region, low/high/units appear in that order at
`align_up(previous_end, 128)`; all alignment padding is zero and there is no
unlisted gap or tail. Source tensor indices are strictly increasing, and their
source byte ranges are ordered and disjoint.

The sidecar root must be a real, nonsymlink directory containing exactly
`manifest.json` and `weights.bin`. Open both children with `O_NOFOLLOW`,
validate and consume each from one file descriptor, and reject row-split
loading. Layer split is allowed because a complete tensor remains on one
device.

Schema v3 is deliberately a regular, single-file, little-endian GGUF contract:
`manifest.source` binds one source file, every `source_tensor.data_offset` is
absolute within that same file, and `storage.file` names one `weights.bin`. It
cannot represent a split/sharded GGUF source because there is no source-file
index per tensor. A future multi-file schema must add per-file authorities and
bind every tensor to one of them; concatenating shard-relative offsets or
treating them as offsets in the first file is invalid. Runtime layer split does
not relax this source-format limitation.

## Offline production and reuse

Create the persistent sidecar once:

```bash
QUACTLIZE_PPU_BUNDLE=/path/to/six-library/bundle \
  quactlize-pack-gguf MODEL.gguf MODEL.gguf.kpack
```

Subsequent loads validate and upload the saved regions; they do not repeat
placement. Do not overwrite or relabel the source GGUF.

The manifest-pinned `8f9fa07` source declares and implements the loader-facing
complete host producer/inverse:

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

For in-process first-load conversion, recovery and exact comparison with the
source GGUF tensor are mandatory before publication. This is a one-time cache
admission check; later loads use the manifest and span hashes instead of
repacking or repeating the inverse.

The complete producer shares the resident geometry above and requires positive
`experts` (use 1 for dense). Every nonempty input and output byte range must be
pairwise disjoint; in-place and partial aliasing are rejected before any write.
A format-selected library accepts only its own qtype. Any decline must leave the
tensor on its non-K-pack route; never label partial, Xplane, or unchanged bytes
as K-pack.

Use checked arithmetic for every allocation. Let E=1 for dense and
E=`experts` for grouped: low owns `E*N*K*bits/8` bytes; high is null exactly
when `high_bits==0`, otherwise it owns `E*N*K*high_bits/8` bytes; and units owns
`E*quactlize_ppu_units_bytes(N,K,qtype)` bytes after requiring that query to
return a nonnegative value.

## Integration order

1. Pin all six DSOs to artifact commit
   `f7f55d61ee1a58657f99df24876aa3bbb13d1a45` and all three public headers to
   source commit `8f9fa07de9694901a5db91d546d6c994720f86b1`. Require the strict verifier,
   selected-config oracle, and prebuilt single-device numeric gate to pass on
   those exact files before deployment.
2. On Ubuntu 24.04, load the admitted SDK wrapper globally first. Load each
   qtype-selected FMT DSO locally, resolve symbols from its own handle, and
   require its exact build identity.
3. Parse and structurally/source-validate the schema-v3 sidecar against the
   source GGUF inventory. Query the selected FMT DSO for the canonical
   arrangement and finish semantic tensor validation by exact comparison.
4. Upload one complete tensor region and retain its span offsets plus complete
   arrangement-v2 in tensor metadata.
5. Query the selected-config ABI with a null requested name and retain the
   complete returned record. Dense uses v4 and its explicit
   `split_k_slices`; grouped uses v3. Pass either null for automatic selection
   or an exact previously validated returned name to workspace/launch APIs.
6. Route dense and grouped operations exclusively through the arrangement-v2
   device APIs. A declined K-pack route must not fall back to a raw GGUF or
   Xplane reader for the same resident bytes.
7. For optional first-load conversion, query the canonical descriptor and
   allocate the three spans with the checked formulas above. Q2_K/Q4_K must
   pass a null high pointer to both producer and inverse. Call the complete v2
   producer, require the inverse to reproduce the exact source tensor bytes,
   and only then admit the result.
8. Persist the admitted result atomically as the sidecar above; later loads
   validate and reuse it without repacking.
