# PPU1.7: PPU CUTLASS 4.3.0 migration

The active1.7 backend now uses the supplied PPU CUTLASS4.3.0 snapshot.
The3.6 dependency is retired from all new build entries, not removed from
historical artifacts. PPU1.0/1.5 actlize, canonical weight bytes, arithmetic,
public ABI and production selection remain unchanged.

## Dependency and source

- Branch: `dev/ppu17-cutlass430`, starting at `7c3ca75`.
- Source worktree: `/workspace/quactlize-ppu17-20260926/source`.
- Dependency: `/workspace/quactlize-ppu17-cutlass430-20260928/dependency`.
- Supplied archive: `/root/cutlass3-v4.3.0_ppu_dev-0eb6a3b218977892e2b6cd30d59880c440e54fd6.tar.gz`.
- Declared upstream revision: `0eb6a3b218977892e2b6cd30d59880c440e54fd6`.
- Archive SHA256: `d0fba143142cf0329474f6db21e9ee6dafd0c3987238423d1b2ac1fa863ebf22`.
- Source-tree SHA256: `0cbc390af07a84190daa5b9df02d80b28545cad5b0497febeebdcd79e5a75771`.

The archive contains no Git metadata. `dependency.py` verifies the actual
version header and complete source tree instead of inventing a local commit.
The basic backend, GPTQ comparison and GGUF collective builders share this
single authority. Modified files, escaping links and old versions are
rejected. Outputs stay outside the immutable dependency directory.

## API changes

| Area | 4.3 integration |
|---|---|
| Fixed Marlin scheduler | Provide `PipelineEmpty`, throttle/storage/response types and the new three-argument work-fetch overload. Integer decomposition, group-aligned K slices and FP32 fixup are unchanged. |
| Scheduler selector | Match the new **uint32_t** pipeline-stage parameter; reject nonzero dynamic-scheduler stages for the fixed Marlin implementation. |
| Conventional mixed input | Select `KernelTmaWarpSpecializedCooperative`; the element tuple selects mixed input. Remove the obsolete suffixed-tag policy specialization. |
| GGUF absolute K coordinate | Regenerate the optional hook at the unique MMA call in the4.3 outer kernel. Keep per-consumer register cursors and the existing Stream-K scheduler. |
| Workspace | Use4.3's native epilogue/scheduler workspace ordering and initialization through its GEMM adapter; never copy3.6 offsets. |
| Cluster1 TMA pipeline | Use4.3's matched per-consumer arrival count and signalling strategy. Do not combine its constructor with the old signalling predicate. |
| Runtime bundle | New identity `ppu17-cutlass430`; the Python backend rejects3.6 bundles. C/POD API stays unchanged. |

No dependency source is patched. The GGUF coordinate hook remains a generated,
hash-bound overlay for the single outer-kernel call site, as in the earlier
experimental path. The unchanged vLLM references use their own pinned4.4.2
CUTLASS tree; it is never substituted for the PPU4.3 candidate include graph.

## Validation tiers

Current artifacts live under `/workspace/quactlize-ppu17-cutlass430-20260928`.
Local compilation uses CUDA12.8 with explicit `compute_90a -> sm_90a`, not
caller-defined architecture macros.

| Gate | Current evidence |
|---|---|
| Basic six-format backend | Full build and host/C/Python ABI gate pass;216 live SM90 bodies,18 codegen negatives and2 target/provenance negatives. |
| GGUF Q4 SF/FQ | Full build including direct smoke/Split-K targets;24 live device bodies and6 codegen negatives pass. Packed-unit host ABI49152 checks and actual CuTe metadata/byte proofs pass. |
| Marlin integer scheduling |256 host shapes/SM-count combinations with exact coverage/reverse peers;7 negative checks pass. |
| GPTQ conventional/prepacked/full Marlin | Full builds pass. Conventional has12 live WGMMA/TMA bodies; prepacked has16 bodies for19 host configurations (three share device bodies). The six standalone Marlin bodies match the pinned reference instruction-for-instruction. |
| Packing / scheduler host proof |8192 packing coordinates,256 vectors,2738 blocks and a stale-layout negative pass. All54 Stream-K planner records are valid under the explicitly supplied114-SM H800 model. These are not device timings. |
| Host identity/runner tests |17 tests pass, including modified/old dependency rejection, rejection of an old runtime bundle before loading, and foreign-job isolation. |
| Physical H800 | NOT RUN for4.3: authorized port19819 currently refuses SSH connections. Old3.6 numerics/timing are not reused as4.3 proof. |
| Native PPU1.7 | NOT ADMITTED: local SDK2.1.1 still fails the genuine SM90A/10700 target probe. No bypass or fallback to1.0/1.5 is enabled. |

FQ being slower than **resident** SF is not by itself a failed performance
gate: FQ decodes metadata in the mainloop. Compare each implementation with
its same-semantics reference, and include separately measured SF expansion
when selecting between end-to-end routes. This migration makes no new speed
or performance-parity claim.

## Rebuild locally

From the source worktree:

```bash
python3 quactlize/backends/ppu17/build.py \
  --cutlass /workspace/quactlize-ppu17-cutlass430-20260928/dependency \
  --compiler /usr/local/cuda-12.8/bin/nvcc --target cuda-source \
  --output /workspace/quactlize-ppu17-cutlass430-20260928/backend --jobs 4

python3 tests/ppu17/gguf/build.py \
  --cutlass /workspace/quactlize-ppu17-cutlass430-20260928/dependency \
  --vllm /workspace/quactlize-ppu17-20260926/reference-vllm \
  --compiler /usr/local/cuda-12.8/bin/nvcc --output /workspace/q17-430-gguf-fresh

python3 tests/ppu17/reference/build.py \
  --ppu-cutlass /workspace/quactlize-ppu17-cutlass430-20260928/dependency \
  --cutlass /workspace/quactlize-ppu17-20260926/reference-cutlass \
  --vllm /workspace/quactlize-ppu17-20260926/reference-vllm \
  --compiler /usr/local/cuda-12.8/bin/nvcc --reader aligned \
  --output /workspace/q17-430-gptq-fresh
```

Use fresh output directories for the two diagnostic builders. Add the local
`check.py` / GGUF `check_codegen.py` / reference `check_marlin_codegen.py`
checks described in their READMEs. Before timing on a physical H800, rerun
numerical/memory gates with the new binaries on an idle device. Native PPU
simulation remains one target launch per process and a CPU-only oracle.

## Artifact handoff

All paths below are relative to the external artifact root above; no binaries
are committed to source or installed over existing libraries.

- Basic library: `backend/6d83c837a4469a450efc/libquactlize_ppu17.so`;
  host/codegen/application receipt: `backend-validation-v2/validation.json`.
- GGUF: `gguf-build-v1/manifest.json`, `gguf-codegen-v1.json`.
- GPTQ conventional: `gptq-conventional-v2/manifest.json`.
- GPTQ prepacked/standalone Marlin: `gptq-build-v3/manifest.json`;
  exact Marlin instruction comparison: `marlin-codegen.json`.
- Host proofs: `marlin-scheduler-host.log`, `prepacked-layout-host.log`,
  `streamk-plan-host.jsonl`.
- Rejected native attempt: `native-probe/9b15963c515479f0b8fe/target.log`.

The early `gptq-build-v1` failed because a scheduler non-type template
parameter was declared `int` rather than4.3's `uint32_t`.
`gptq-conventional-v1` failed because the retired suffixed mixed-input
schedule no longer matches the builder. Both failures are preserved; neither
was hidden with a fallback, changed math or an empty specialization.
