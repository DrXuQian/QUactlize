# GGUF cooperative pipeline checkpoint

2026-09-26; isolated branch `dev/ppu17-cutlass360`, parent `ee73b4e`.
This adapts Machete's register-dequant TMA/WGMMA pipeline to Q4_K's real
group32 scale/min semantics. No format merging or requantization. Existing
PPU1.0/1.5 code, canonical bytes, default dispatch and dependencies are intact.

## Implemented and checked

- Private lossless128N x64K code packing; CuTe/scalar inverse proof over8192
  coordinates and256 aligned16-byte thread vectors.
- Two physical scale/zero rows per64K stage, with group32 logical broadcasts.
  Host proof covers FP16/BF16, N128/256 and stages4/8. Wrong group/stage
  substitutions are rejected. Pipeline transactions count physical bytes.
- Three dense cooperative configs, explicit four-stage pipeline, native SM90
  Stream-K workspace/fixup included in the prepared call.
- CUDA assembly contains128-bit shared loads, TMA, direct BF16 pair multiply/
  add, and FP32-accumulating WGMMA. Six device bodies, five codegen negatives.
- H800:54/54 numerical cells for FP16 and54/54 for BF16; each has output
  guards, poison checks and three changed-input graph replays. Exact one-hot
  weights, distinct group32 metadata, M1/2/7/8/9/63/64/65/128, N128/256,
  K256/512. Stale-group data negatives are rejected for both dtypes.
- Largest relative L2 versus the typed raw-GGUF CPU-dot oracle:
  FP16 `2.494e-5`, BF16 `6.637e-5` in that numerical cohort.

The new descriptor remains diagnostic-only and is not published under a
canonical K-pack mapping ID. This milestone is **SF resident GEMM**, not FQ,
SF expansion performance, full-model accuracy or native PPU1.7 admission.

## Initial rotating-weight screen

H800 PCIe,114SM, queried L2=52,428,800 bytes; ring>2.25x L2; clocks1755/1593MHz
at both endpoints. Idle guard and sampled process monitoring admitted every
listed run. Each arm/config checks correctness before11 timed graph samples.
All scheduler clearing/fixup is included; first graph launch, allocation,
packing and SF expansion are excluded. Both arms use the same original GGUF
bytes and typed affine oracle.

| M/N/K | FP16 new SF (us) | BF16 new SF (us) |
|---|---:|---:|
| 1/512/2048 | 11.980 | 11.939 |
| 8/512/2048 | 11.938 | 12.002 |
| 128/512/2048 | 13.786 | 13.462 |
| 1/4096/4096 | 19.433 | 19.777 |
| 8/4096/4096 | 19.875 | 19.967 |
| 128/4096/4096 | 26.433 | 26.348 |

Selected among three configurations in each screen, **not confirmed winners**.
For example, M128/N4096/K4096's original bring-up control is443.411/449.882us
for FP16/BF16. This removes major bring-up overhead, but that old control is
not a mature library baseline. Packing, RS operand delivery and pipelining
changed together, so this is not evidence that shuffle alone caused the win.
GPTQ group128 Machete/Marlin timings are not equivalent GGUF baselines.

## Evidence and next gates

Artifacts: `/workspace/quactlize-ppu17-20260926/gguf-mainloop/`.

- `build-v1/manifest.json`, DSO SHA256
  `538068ed7286724cd5aa7cd6b00f6036e59e45099f05d18707b8129150794a2f`.
- `codegen-v1.json`, `numeric-f16-v2`, `numeric-bf16-v3`, `screen-v1`.
- The two earlier BF16 attempts were rejected before launch for foreign GPU
  processes; they were not numerical failures and were not reused as results.

Next: bounded stage-depth comparison with existing configs retained, memory/
race checks, and repeated finalist timing. Then add FQ packed metadata and a
canonical-reader control on this same pipeline before any shipping-format
decision. Other qtypes, grouped scheduling and native PPU1.7 remain open.
See [build/replay instructions](../tests/ppu17/gguf/README.md).
