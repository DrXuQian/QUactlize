# PPU1.7 programmatic dependent launch

This is an opt-in dependency protocol on the CUTLASS 4.3 backend, not a new
weight format, tactic, or production default. PPU1.0/1.5 are unchanged.
The first implementation does **not** move weight loads ahead of the wait.
Physical dependent-chain correctness must pass before that optimization.

## Interface and implementation

Build with `quactlize/backends/ppu17/build.py --enable-pdl`. The bundle then
contains both ordinary and PDL specializations. Existing v1 calls still use
ordinary stream ordering; `q17_call_v1` and `q17_plan_v1` remain compatible.
The optional ABI is explicit and does not read a process environment flag:

```cpp
q17_launch_options_v1 options{1, sizeof(q17_launch_options_v1), 1, 0};
q17_plan_v1 *plan = nullptr;
int status = q17_prepare_with_options_v1(&call, &options, &plan);
// Check status. Prepare outside capture, then enqueue on call.stream:
if (status == Q17_SUCCESS) {
    status = q17_execute_v1(plan);
    q17_destroy_v1(plan);
}
```

Python: `backend.prepare(call, pdl=True)` and
`backend.expand_scales(call, pdl=True)`. Unsupported builds reject the request
instead of silently using ordinary launches. `q17_pdl_compiled_v1()` is a
host-only build capability, not a hardware admission certificate.

| Path | Wait | Launch-completion hint |
|---|---|---|
| SIMT and cached Q4 | Before grouped offsets, activation or metadata reads | Before final reduction/output |
| Synchronous WGMMA and TMA/WGMMA | Before offsets and operand loads | After the existing MMA completion/stage-lifetime waits, before output stores |
| SF expansion | Before reading packed units | Before expansion; its consumer must wait for the actual writes |
| CUTLASS cooperative GPTQ / prepacked GPTQ / GGUF | Existing CUTLASS producer/epilogue/scheduler GDC sites | Existing last-work-segment GDC site |

`launch.cuh` owns extended launch and device helpers. Ordinary kernels use
compile-time `Pdl=false`, with no runtime policy branch or GDC instruction.
An empty CTA exits normally and supplies its implicit completion trigger.
Prepared calls retain caller-owned storage and stream lifetimes. Reusing a
workspace is also a dependency: PDL does not waive write-after-read hazards.

The cooperative comparison builders accept `--enable-pdl` too. They enable
the existing CUTLASS `launch_with_pdl` argument and `CUTLASS_ENABLE_GDC_FOR_SM90`
only for our cooperative provider. Standalone Marlin and external reference
libraries are **not** relabeled PDL-compatible. The fixed Marlin scheduler
adapter now supplies `is_last_tile`: last segment of the current CTA, not last
peer of a reduction or last tile of the whole grid. Its integer schedule and
FP32 fixup are unchanged. No vendor source is edited.

The launch hint is not a data-ready signal. Consumers wait for predecessor
completion and visibility before reading dependent storage. Launches must
remain on the declared stream or use explicit programmatic graph edges.
Concurrency is optional; no kernel spins waiting for a later kernel to start.
See the [CUDA PDL contract](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).

## Local checks

Use the pinned dependency described in [the CUTLASS 4.3 integration](ppu17-cutlass430.md).
All build outputs must be outside the source checkout.

```bash
python3 quactlize/backends/ppu17/build.py \
  --compiler /usr/local/cuda-12.8/bin/nvcc \
  --cutlass /workspace/quactlize-ppu17-cutlass430-20260928/dependency \
  --target cuda-source --enable-pdl --jobs 6 \
  --output /workspace/q17-pdl-build
```

Pass the printed bundle directory to `check.py`:

```bash
python3 quactlize/backends/ppu17/check.py \
  --bundle "$BUNDLE" --output /workspace/q17-pdl-validation --jobs 6
python3 -m unittest discover -s tests/ppu17 -p 'test_*.py' -v
```

Set `Q17_TEST_OUTPUT` to an existing external task directory to run the
temporary-fixture identity/isolation tests too. The codegen gate checks every
ordinary/PDL specialization, dependent-load ordering, live math/output and
assembled wait/release instructions. CUDA12.8 SM90a lowers these to ACQBULK /
PREEXIT; this spelling is not asserted for native PPU assembly.

`tests/ppu17/compare_ordinary_codegen.py` compares all216 ordinary encoded
device bodies with a pre-change bundle from the same compiler, normalizing
only the added false template tag and demangler whitespace.

## H800 / physical-device chain gate

This application launches dependent kernels and replays graphs. **Do not use
it in the existing single-target-invocation PPU simulator workflow.** All
numeric references are CPU-only. No timing is admitted by this diagnostic.

On an idle H800, after the local gate:

```bash
python3 quactlize/backends/ppu17/check_pdl.py \
  --bundle "$BUNDLE" --validation /workspace/q17-pdl-validation \
  --device 0 --expected-uuid "$GPU_UUID" --suite smoke \
  --output /workspace/q17-pdl-smoke
```

Use `nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader` to obtain the
expected UUID. The runner refuses a busy device and stops only its own child
if a foreign GPU job appears. Each result binds the binary and retained log.

- `smoke`: 52 cases across Q4/Q6, FP16/BF16, SIMT/WGMMA/TMA, FQ/SF, dense/grouped
  and cached Q4. Every case checks both ordinary and PDL, stream and graph,
  3 replays × 2 different activation/routing epochs (24 comparisons).
- `coverage`: 476 cases, all six supported quantization types, all 11 current
  config IDs where legal, M1–8/M65, K1024 stage wrap, tails, empty/ragged
  experts, and typed I/O.
- The chain is a bounded delayed input producer → optional SF expansion →
  actual prepared GEMM → output consumer. It checks CPU math, guards and
  ordinary-versus-PDL raw equality.
- A missing-wait negative has a waited positive control and checks that the
  producer really completed. If the device does not expose overlap, report
  `UNRESOLVED_NO_OVERLAP`, not a false PASS or an infinite handshake.
- Failed cases do not erase earlier results. Each log starts with its exact
  standalone reproduction command. `summary.json` retains all results.

## Status and remaining steps

Local artifacts: `/workspace/quactlize-ppu17-pdl-20261001/`.
Source parent: `63a6cd11d2dedb945e8ed9cadae920c86b9eeefd`.

The implementation covers the protocol, optional ABI, compile checks and
physical-chain test entrypoint. The physical H800 chain gate passed on
2026-10-01 as recorded below. Native PPU1.7 codegen/numerics and performance
remain pending; no route is promoted.

Local checks on 2026-10-01 (CUDA 12.8, no GPU launches):

| Check | Result |
|---|---|
| Ordinary build + host/ABI/codegen | PASS, 216 device specializations |
| PDL-enabled build + host/ABI/codegen | PASS, 432 specializations; wait/release checked per entry |
| Unchanged ordinary device code | PASS, all 216 encoded bodies equal the pre-change parent |
| CPU test suite, including optional/old binary ABI | PASS, 23 tests |
| Fixed Marlin scheduler | PASS, 256 cases; one release point per active CTA |
| Cooperative GPTQ / prepacked provider | PDL build PASS; physical chain not run |
| Cooperative GGUF Q4 SF/FQ | PDL build and 24-body math/codegen gate PASS; physical chain not run |

The first cooperative prepacked build exposed the missing `is_last_tile`
adapter method. Its failed logs are retained alongside the passing rebuild.
These checks do not establish a PDL speedup or native PPU1.7 admission.

### Physical H800 chain result

Implementation: `8e8bc0065dc538366641522c34ed9df5d2ef1012`. Reused the exact
CUDA 12.8 local-gate library and application on an idle H800 PCIe 80GB,
driver 595.71.05; no rebuild or arithmetic change. The controller monitored
GPU process isolation throughout both suites.

| Suite | Cases | Ordinary/PDL × stream/graph comparisons | Missing-wait negative | Result |
|---|---:|---:|---|---|
| Smoke | 52 | 1,248 | EXPECTED_RED; waited control PASS | PASS |
| Coverage | 476 | 11,424 | EXPECTED_RED; waited control PASS | PASS |

All comparisons passed the independent CPU oracle, output guards and raw
equality against ordinary execution. Coverage contains 76 cases each for
Q8/Q2/Q3/Q5/Q6 and 96 for Q4. The negative read the old marker `-1`, while
the waited control and completed producer returned 42. This verifies that
the device exposed the omitted dependency; overlap is not guaranteed on
every other launch.

Evidence retained locally under
`/workspace/quactlize-ppu17-pdl-20261001/h800-19641/{smoke,coverage}/` and
remotely under `/workspace/q17-pdl-h800-20261001/{smoke,coverage}/`.
Every downloaded log matched its recorded SHA256. Coverage summary SHA256:
`bf5b9ba86dc6b1e7d889b78f0ea744683104fb1763e3869c57199c1a8fd8ead4`.

This gate covers the q17 prepared SIMT/WGMMA/TMA paths, not the experimental
cooperative GPTQ/prepacked/GGUF provider chains. **No kernel performance was
measured**; the 60s/559s suite wall times are diagnostic execution time, not
latency or a speedup. Native PPU1.7 remains a separate pending gate.

After physical correctness:

1. Move a bounded first weight tile ahead of the dependency wait. Keep
   activation, fresh SF metadata and unready routing reads behind it.
2. Split TMA A/B issue while preserving exact transaction bytes and stage
   lifetime; keep the existing WGMMA completion protocol.
3. Extend to admitted PPU1.7 MoE auxiliary kernels as those paths are ported.
   The current compact-grouped API does not silently supply a full MoE chain.
4. Compare ordinary, protocol-only and early-weight-load variants with the
   same configs/cache conditions. Exclude compilation and first use; report
   the complete chain as well as individual kernels. Enable only on measured
   net benefit, retaining the ordinary path.
