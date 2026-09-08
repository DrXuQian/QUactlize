# K-pack loader integration audit

Reviewed 2026-09-08 against `/root/llama.cpp` develop `26955be8a` and
Quactlize develop `05de781`, including the local changes listed below.
No llama.cpp commit, push, production route switch or main admission is part
of this audit.

## What can be reused

| Component in llama.cpp | Finding | Disposition |
|---|---|---|
| `ggml/src/ggml-cuda/quactlize-lib.cu` | Per-format `RTLD_LOCAL` loading, canonical-arrangement query, registry checks and any-M capability queries already exist. Host loader tests pass. | Reuse; these currently describe the old format DSOs, not the new selected modules. |
| `quactlize-buft.cu` | Owns one byte-neutral `[low][high][units]` device allocation; empty high planes use NULL. Non-Kpack readers/fusions are excluded. | Reuse allocation, artifact registry and validated sidecar upload seams. |
| `mul-mat-quactlize.cu` | Dense F32/FP16 boundary casts and row-major flattening are implemented. Execution is FQ with old `config_name=NULL`. | Reuse boundary handling; new heuristic/SF prefill is not wired. |
| `mmid-quactlize.cu` | Reuses `mm_ids_helper`, device bounds, gather and scatter without routing D2H. | Preserve this device-only property. |
| `src/llama-kpack-sidecar.{h,cpp}` | Schema-v3 parser, byte hashes, writer, no-replace publication and negative tests exist. Cross-check against `quactlize.pack_gguf.load_kpack_bundle` passes. | Reuse format handling after completing loader integration. |

## Gaps that a symbol-level replacement would miss

1. **Conversion is still host-side.** `qz_buffer_set_tensor` allocates three
   host planes plus a full recovered-GGUF buffer, calls
   `ggml_quactlize_convert_verified`, then synchronously uploads the planes.
   Conversion parallelism is across experts, so dense `experts=1` has no such
   parallelism. The metadata/code packer also constructs intermediate native
   code planes. This is not a GPU conversion path.
2. **Sidecar loading is not connected to model loading.** At this snapshot,
   the reader/writer and backend proc-address seams have implementations and
   tests, but `llama-model-loader.cpp`, `llama-model.cpp`, the public model
   parameters and common CLI have no callers. Existing sidecars therefore do
   not automatically bypass `set_tensor` conversion. The header's statement
   that the model loader wires them describes intended integration, not a
   current execution path.
3. **The new grouped ABI needs host rows.** `runtime/module.cuh` validates
   `rows_host`, derives exact group shapes, prefix sums, total tiles and the
   uniform-tile property from them. llama.cpp has only device `bounds` on its
   current hot path. An upper bound such as `n_tokens` cannot be passed as the
   measured per-expert row vector. A direct replacement would either require
   a synchronizing router readback or be incorrect. Close a device-directory
   grouped binding before claiming graph-capturable heuristic integration.
4. **Any-M admission cannot come from the exact table alone.** The old
   `*_any_m_valid` queries are capability promises for the old DSO execution
   path. They do not admit a new module selector that can decline unknown
   M/router/families after the original GGUF representation has been dropped.
   A numerical K-pack miss path must be explicitly admitted before enabling
   the new loader; do not silently use an arbitrary config string.
5. **Operand and device checks need a focused audit.** The early Kpack return
   in `ggml_backend_cuda_device_supports_op` bypasses the ordinary source
   device checks and does not check the F32/contiguous assumptions asserted
   by both execution wrappers. The capability query currently receives the
   weight only. Cover strided activations, mixed devices and unsupported
   sources before expanding its admission.
6. **The current sink is synchronous and borrowed.** `g_sink` and its context
   are global; the callback consumes CPU plane/source pointers that expire
   after `set_tensor`. It is neither a per-model asynchronous job owner nor
   safe to enqueue by merely retaining those pointers. The writer is ordered
   and not thread-safe. Background integration needs explicit ownership and
   cancellation, not just `cudaMemcpyAsync` substituted into the old call.

## Small fixes made during review

- The sidecar test target and its stub dependencies now remain inside
  `if(GGML_NCP_QUACTLIZE)`. Configuration with the option disabled succeeds
  and contains neither Kpack test target.
- `llama_kpack_sidecar_writer::begin` takes ownership of its staging path
  only after `mkdir` succeeds and rejects reentry on an active writer. Before
  the fix, refusal of an existing `.partial.<pid>` directory still left that
  path in the object, so destruction deleted another writer's files. A second
  `begin` could also abandon the first staging file. Both regressions were
  observed locally before the fix and pass afterward.
- The host loader and sidecar tests pass, including the optional Python
  bundle-reader interoperability check. These are not device inference tests.

## GPU producer implemented, not yet device-admitted

`quactlize/packing/` provides a separate PPU conversion leaf:

- `api.h`: size query and asynchronous device-pointer producer.
- `word_pack.hpp`: one writer per final b16 code word; source code fields and
  metadata use the existing CuTe-owned traits. Q5 high-plane mapping and
  paired Q3/Q6 units reuse their canonical maps. There is no floating-point
  dequantization/requantization or atomic nibble scatter.
- `sizes.cpp`, `ppu_pack.cu`: exact canonical-descriptor/shape/range guards,
  low/high/metadata launches, immediate error propagation, no allocation,
  transfer or synchronization inside the producer.

All five formats share this converter; dense is `experts=1`. Input and outputs
are deliberately out of place. The caller uploads raw GGUF to temporary device
storage and writes the planes directly into the final weight allocation.
It must not overwrite raw bytes in place while other threads still read them.
Grouped weights can be batched along the independent expert axis. This first
entry does not support arbitrary N/K subranges of a single expert; a dense
tensor needs raw scratch for that tensor. Add an explicit globally-strided
subrange contract if that exceeds the agreed temporary-memory budget.

Host proof executes the same word/metadata ownership against independent
Python canonical bytes for Q2-Q6, dense and three experts, two K sizes, guard
bytes and invalid-descriptor/overflow cases. The PPU library compiles locally
in 9.355 seconds and is 126,504 bytes. Local artifact:

```text
/root/autodl-tmp/kpack-pack-20260908-v1/libquactlize_ppu_pack.so
```

The same payload and its build receipt are provided under
`prebuilt/ppu0010/kpack-pack-v1/`; the shared library is stored with Git LFS.
Its SHA-256 is `611ec98c4315748e4504082aa3071ae9713ee78958cd09b6baaee770b28a3184`.
This is not a PPU numerical/performance pass. It has not replaced the old six DSOs. The GEMM
kernel identity remains exactly
`2a9791f23a252fbb1fc5d6692f7bc7e3238d8cd7ff318ac427513e3d68e2eeb4`;
the measured 55-module cache does not need rebuilding for this converter.

## Copy and persistence ordering

```text
raw GGUF -> bounded H2D scratch -> GPU pack -> packed-ready event
                                              |               |
                                     compute stream       copy stream
                                     wait(ready)          wait(ready)
                                     read-only GEMM       D2H pinned slot
                                                          copy-done event
                                                                  |
                                                     CPU hash/write worker
                                                                  |
                                                    ordered manifest commit
```

The inference dependency ends at packed-ready, not at D2H or disk completion.
Concurrent GEMM and backcopy may read the same immutable packed weights.
Actual overlap/speedup depends on the PPU engines and memory contention and
must be measured. Do not claim it from stream names alone.

Use a bounded pinned-memory pool and a single ordered writer. A queued job
owns its pinned slot, tensor source identity, device allocation lifetime and
completion event. Preserve source hashes or hold the source mapping alive;
do not borrow a temporary loader buffer past return. Recycle a slot only
after D2H and writing finish. Model cancellation/teardown must join or drain
jobs before freeing their device buffers. Publish only after every recorded
plane is complete and verified; no incomplete sidecar is a cache hit.

## Next checks and scope

The first box gate is `tools/run_kpack_pack_gate.py`: exact output bytes,
repeated calls, output guards, overlap/descriptor negatives and an event-
ordered D2H into pinned host memory. It times H2D, pack and D2H separately.
It does not benchmark overlap with a GEMM or implement the background writer.
Use the prebuilt converter; this command does not invoke a compiler:

```bash
(
  set -eo pipefail
  SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK
  source "$SDK/envsetup.sh"
  export LD_LIBRARY_PATH="$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export CUDA_VISIBLE_DEVICES=0
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  git lfs pull --include='prebuilt/ppu0010/kpack-pack-v1/libquactlize_ppu_pack.so' --exclude=''
  RUN=$(mktemp -d /workspace/kpack-pack-gate.XXXXXX)
  set +e
  python3 tools/run_kpack_pack_gate.py --sdk "$SDK" \
    --bundle prebuilt/ppu0010/kpack-pack-v1 \
    --output "$RUN/results" --real-anchor 2>&1 | tee "$RUN/console.log"
  rc=${PIPESTATUS[0]}
  set -e
  tar -czf "$RUN.results.tgz" -C "$RUN" .
  printf '\nrunner_rc=%s\nresults=%s\n' "$rc" "$RUN.results.tgz"
)
```

Update the develop checkout and its actlize submodule before running. The
gate checks the payload, its source receipt, SDK runtime libraries and the
one visible device's PCI identity. It does not require matching compiler or
inspector executables for execution. The optional real anchor adds all five formats at
N=1024,K=5120; CPU reference construction is reported separately from device
timing. Return the small `results` directory and build manifest/logs, not an
entire model. There are 15 cases, not a config sweep. Reference construction
can dominate wall time; this gate has not yet been timed on PPU, so there is
no measured total-time promise. If rebuilding the converter is needed, use
`tools/build_kpack_pack.py --sdk SDK --output FRESH_DIR` locally; the earlier
9.355-second build is for this small producer, not the GEMM modules.

## Final JIT deployment packaging

The current direction is a small selector/module-loader binding plus the
independent packing DSO and a cache of separate compute modules. Compile
small control/conversion libraries locally. Keep the existing admitted
kernel cache; a missing parent is compiled once with the target SDK, not
chosen by online profiling. There is no requirement to rebuild all six
legacy DSOs or to emit every config into a new monolithic library. Packaging
and C++ binding still need implementation; the old six-library bundle remains
unchanged as a transitional artifact. Future changes to a compute module's
ABI or body require rebuilding the affected modules, not merely relinking
the control library.

After this gate: connect GPU intake and sidecar hit/miss in the model loader,
add owned background persistence, close C++ complete-identity/any-M and the
device-only grouped binding, then run checkpoint-level dense/MoE prefill and
decode with writeback disabled/enabled. Do not restart the Cartesian sweep.
