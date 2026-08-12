# Local-tier red classification (2026-08-13)

This audit covers the nine `boxdry` failures and the one PPU-portability failure that were present at
`2ba676f`.  “Known red” is not a verdict: each row below records the mechanism, ownership, and what a persistent
failure could hide.

## Common cause

All ten failures had one repository-owned cause.  `build.sh` invokes
`dev/fold_derivation/ppu_portability_check.py` before CMake.  The checker conservatively scanned the top level of
`benchmarks/`, but treated that search namespace as if every file were a PPU target source.  The scan therefore
classified the standalone RTX5090 experiment
`q4k_pdf_5090_ab.cu` / `q4k_pdf_ab_fixture.hpp` / `q4k_pdf_reconstruction.cuh` as PPU input.  Those files are built
only by `q4k_pdf_5090_ab.py` with `nvcc -arch=sm_120` and NVML; no PPU CMake target names them.

The nine boxdry runs stopped on that preflight in 0.3--0.9 seconds, before CMake.  Their old message (“our
CMakeLists was not reached”) was a secondary diagnosis produced after the real preflight error.  This was not the
old googletest/network failure: tests/gtest are forced off and `FETCHCONTENT_FULLY_DISCONNECTED=ON` is set.  A
network requirement now fails offline instead of cloning.

## Per-row classification

Every row is **our checker/source-authority defect**, not an environment lacking a capability.  Consequently none
of these ten rows is legitimately a SKIP on this machine.

| Red row | Mechanism behind this red | Real regression it could mask |
|---|---|---|
| boxdry `test_moe_splitk_bench` | Common portability preflight stopped before CMake | target registration, `PPU_DEFS`, device compile, or host link |
| boxdry `test_q4k_packed_gemm` | Same preflight; the similarly named RTX5090 reconstruction is not this PPU target | packed-Q4_K target creation, `PPU_PACKED_SCALE`, compile, or link |
| boxdry restricted MoE axes/stages | Same preflight | env -> `build.sh` -> CMake -> device-`-D` forwarding for every restricted axis |
| boxdry dense persistent A/B | Same preflight | persistent one-row object graph and final host link |
| boxdry dense Stream-K A/B | Same preflight | generated Stream-K units, object graph, and link |
| boxdry dense Marlin A/B | Same preflight | DP/Stream-K/Marlin object graph and link |
| boxdry planted generated-unit undefined reference | Same preflight prevented the negative arm from reaching the linker | the exact cross-TU visibility defect that caused `c96fe8d`; this was the most dangerous hidden red |
| boxdry dense Marlin full sweep | Same preflight | omission of any private generated sweep unit from the link |
| boxdry grouped Stream-K | Same preflight | grouped Stream-K object graph and link |
| PPU portability lint | Directory scope was mistaken for PPU reachability, producing 31 NVIDIA-API hits in the RTX5090 island | a new NVIDIA-only identifier in a real PPU source: the row was already red, and the summary exposed only an old first hit |

## Q4_K applicability

The RTX5090 PDF-reconstruction experiment is **not applicable to PPU**, not “unimplemented on PPU.”  It explicitly
requires compute capability 12.0, `sm_120`, NVIDIA runtime events, and NVML.  PPU Q4_K coverage exists separately in
the registered `test_q4k_packed_gemm` and `test_q4k_native_scale` targets.

N/A is now a fail-closed boundary rather than an allow-list:

- the exact four-file local NVIDIA island must retain its `sm_120`/NVML build contract;
- none of its members may be named by the PPU CMake authority;
- no PPU candidate source may include an island header.

The portability row is therefore PASS when that boundary and all applicable sources verify.  If the island becomes
PPU-reachable, the row is FAIL; it does not inherit an exemption.

## PASS / SKIP / FAIL semantics

- **PASS**: the check ran and established its property.
- **SKIP(reason)**: this environment lacks an actual prerequisite such as `gcc`, `g++`, `cmake`, `make`, or (where
  required) `nm`.  Repository/source/configure/compile/link errors are never SKIP.  Offline dependency demand is a
  repository FAIL, not a network-capability SKIP.
- **FAIL**: the check ran and found a defect, or a repository-owned checker/source authority is missing.

The tier reports the three counts separately.  `--strict` makes any SKIP produce a nonzero process status without
renaming it FAIL or adding it to PASS.

Two negative arms prevent this cleanup from merely hiding red:

1. The portability checker plants an unconditional CUDA include in a PPU candidate, a PPU CMake edge into the
   NVIDIA island, and a PPU include edge into the island.  All three must be rejected.
2. Boxdry plants a missing cross-TU generated-unit symbol without enabling the “expected failure” policy.  The real
   host linker must return raw status 1, the diagnostic must name `qz_boxdry_generated_unit_anchor`, and the local
   tier must classify it FAIL rather than SKIP.  A separate expected-negative row proves the linker witness itself.
