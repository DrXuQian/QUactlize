# Standalone PPU K-pack library

`libquactlize_ppu.so` owns packing and the complete **dense** consumer. It is
independent of ncp_flash_lib and llama.cpp. ncp may remain a version/build
aggregator, but must not implement a second selector or C++ provider wrapper.
The existing low-level APIs and PoC integration remain available unchanged.
This change does not replace the separate grouped/MoE consumer API.

## Build and install

```sh
bash build.sh --library --output /absolute/fresh/package --jobs 16
```

SDK defaults to `PPU_SDK`, then `/usr/local/PPU_SDK`. An explicit `--sdk` is
also accepted. This is an off-device build, not a sweep or accuracy test.
The compiler **and runtime link libraries** must be installed. Intermediates
are kept in the sibling `package.build`, outside the installable directory.

Install these as a unit:

```text
libquactlize_ppu.so                  # the single public library
quactlize-runtime/                  # private execution/dispatcher/provider DSOs
  jit/                             # pinned header + compile-only Python closure
include/
  quactlize/dense/api.h
  quactlize/packing/api.h
  quactlize_ppu_config.h
manifest.json                      # source/SDK/payload receipt, not an admission
```

There is no public pack/dense forwarding DSO, no legacy six-library bundle,
and no dependency on the original checkout after installation. Relocate the
public library and its sibling runtime directory together. The private JIT
snapshot is required even though it is not part of the public include API.

```sh
python3 tools/build_ppu_library.py --verify --output /absolute/package
```

ncp only needs to invoke the build and install these outputs; its current
Quactlize/actlize gitlink checkout management can stay. llama can also consume
this install directly, without ncp. The other ncp FA/MoE dependencies are not
changed by this library. See [the ncp integration handoff](NCP_QUACTLIZE_INTEGRATION.md).

## Public dense lifecycle

1. `quactlize_ppu_dense_supported_v1`: pure host layout/size/shape admission.
   M=0 asks for all representable positive M, not a sampled ladder. Q8_0 and
   Q2_K..Q6_K have explicit coverage; N is a multiple of 256, K follows the
   format alignment. Byte extents and the M*N integer bound are checked.
2. `quactlize_ppu_dense_create_v1`: context on the current device. The default
   is the PoC's **F16 dense compute with F32 caller storage**. F32 values are
   not necessarily representable by F16; no implicit clipping is performed.
   Call `create_with_options_v1` to explicitly request BF16 for wider range.
3. `quactlize_ppu_dense_prepare_v1`: outside capture, choose and materialize
   one immutable plan; report actual route/config/split/precision/workspace.
   The context owns plans and modules, the caller owns weights and scratch.
4. `quactlize_ppu_dense_run_v1`: current A/C/workspace/stream per call. No
   tuning, JIT, allocation, D2H or device wait in Quactlize's run. Concurrent
   streams need disjoint output and scratch. A captured graph retains its
   captured addresses; it is not rebound by a later ordinary run.
5. After all outstanding calls/graphs complete, destroy the context and free
   its retained weights. Destruction does not synchronize the device.

JIT is selected-parent compilation during prepare, never an online tuner.
Paths are resolved relative to the installed library. Optional overrides:
`PPU_SDK`, `QUACTLIZE_PPU_CACHE_DIR`, `QUACTLIZE_PPU_PYTHON`. SIMT-only plans do
not require Python/JIT initialization. cuBLAS needs an eager warmup before
capture/timing, as it did in the PoC.

## Performance-preserving migration

The default route order remains matched small-M -> previous decode policy ->
component-ranked prefill -> same-family TC -> legacy canonical FQ fallback.
The exact parent, Split-K,
persistent recipe and measured SF dequant config are retained. Selected
module/resource errors do **not** silently change the winner. A genuine
hole outside the old fallback's supported geometry uses an explicitly
labeled AP0/S1 coverage parent, not a claimed measured optimum. Legacy
fallback is generated from the original shipping policies with DN0, not
replaced by a new small tile or relabeled as a measured winner. No old
six-library binary dependency is retained.

Matched compact M1..8 continues to use fused F32 TC endpoints or SIMT, with
no added gather/scatter. The legacy FQ fallback retains its flat FP16 casts,
including the old M<=8 exceptions. F16 prefill retains the native F16 endpoint, original row
adapter geometry, SF-before-input ordering and output rounding. Replacing
these with full-M F32 TC is **not** smuggled into the performance migration.
The additive general typed endpoint is used for explicit BF16, whose route
transfer outside the BF16 measured table remains a proposal.

Route values are the public enum (FQ=0, SF=1, SIMT=2, full-BF16=3), not the
private cost-table enum. TC tickets are consumed directly. Dense SF metadata
remains F16 even in the current BF16 dense core; it is not confused with the
different grouped BF16 metadata contract. Full dequant always emits BF16.

## Static migration check and device-admission boundary

```sh
python3 -m pytest -q tests/test_dense_provider.py tests/test_ppu_library_package.py \
  tests/test_kpack_dynamic.py
```

Host tests check actual selector/dispatcher/provider composition, all formats,
the measured dense tables, M-domain coverage, route enums, metadata types,
workspace and prepare failure semantics. Device launches in the host test
are explicitly fake; none are linked into product binaries.

`tests/dense_poc_selection.hpp` freezes the Auto dense control flow from
llama.cpp `abaccee355cc69169535bc142e29c339a6ae682e`. Across 76 measured weight
families and 19 boundary M values, all 1,444 requests select the same route,
parent/recipe, precision, metadata/prepass and endpoint/adapter contract.
This includes 18 legacy fallback requests, not merely the native-policy
hits. Separate checks cover 1,053 matched rows and 560 prefill rows across
F16/BF16. New coverage is tested separately and is not counted as PoC parity.

This migration's selection check is static; it does not require another
performance sweep. The proof is of selection/composition, not an assertion
that elapsed time or generated binaries have been measured equal.

Before declaring performance reproduced, on an **idle PPU** compare the same
weights/inputs/shapes against the working PoC. Require matching precision,
recipe, split, SF dequant config and launch chain first. Check correctness,
changing addresses, two streams and graph replay on the actual binaries.
Exclude initial JIT and provider warmup from both timing arms, use paired
repeats and collect asys when latency differs. Report full invocation time
(adapters/prepass/reducer included), not just GEMM time or estimated MBU.

The build manifest deliberately starts with `device_admission=PENDING` and
`performance_admission=NOT_MEASURED`. Host table equality and successful
compilation are not proof of numerical or timing parity.

Local checks (2026-10-13): 168 host regressions passed; 29 dynamic-parent
and 7 legacy/copy PPU compile-only cases passed. The legacy compile probes
assert identical Shipping/GemmKernel types to the old device backend.
The optional `test_llama_api_mirrors_and_dense_direct_pointers` was excluded:
its hard-coded private PoC checkout fails the copied-header text comparison
even against the unmodified Quactlize base. No private PoC files were changed
to make that integration-only check pass. Public C/C++ ABI/header checks are
included in the passing tests.

The local SDK lacks real `libhggc_wrapper.so` and `libhg_wrapper.so`; the
public library has not been fully linked here. Build the installable package
with a complete SDK. Compile-only objects are not runnable artifacts.
