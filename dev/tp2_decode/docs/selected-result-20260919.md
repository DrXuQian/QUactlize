# Selected decode: returned TP2 model results

Reviewed archive: `/root/kpack-tp2.8VYueY.results.tgz`, SHA256
`6031eb062043058e412fc236afbd0d01e999148a1fa2a7765fe408c565728b5a`.
Source: `e624e44f7297d5cd9ea8bd188fcd64f4c74d3e95`.
Caller: `1f7da3bd94b1fc6b9520e156456c89221fddd07f`.
Runtime manifest: `5bda5fabf565c54494a358479a3ce9923bcde8d204338aede0742517491a4aab`.
Execution: `2cb65b500a1bc872347a9cbd7eb82ed849ff350616900acaaa6c8529fef3f0a0`.
These identities match the published selected-decode handoff. The immutable
build manifest still records pending device admission; this later result
review does not rewrite that build receipt.

## Model performance

Qwen3.5-122B-A10B Q4_K_M, TP2 devices 0,1, request batch 1, PP2048/TG128.
ABBA uses two measured passes per process after excluding its first complete
pass. Four measured samples per arm are retained. Timings are unprofiled;
startup, first-use JIT and the warmup pass are not inference samples.

| Metric | Native reference | Selected K-pack | Latency change |
|---|---:|---:|---:|
| Prefill us/token | 285.995361 | 228.169678 | -20.2191% |
| Prefill total ms, 2048 tokens | 585.718500 | 467.291500 | -20.2191% |
| Decode ms/token | 12.700852 | 11.239113 | -11.5090% |

Recomputed decode samples (ms/token): reference 12.703969,12.697734,
12.674680,12.712508; selected 11.211422,11.216164,11.262930,11.262063.
All four processes exit successfully. No JIT cache miss appears in these
benchmark logs. This is a warmed synthetic-token benchmark, not a throughput
measurement over arbitrary traffic or a proof of global configuration optimality.
No independent idle-device audit is included.

Previous returned run `kpack-tp2.om9JQj` had K-pack TPOT 12.482082 ms and
native TPOT 12.724309 ms. The selected version is 1.242969 ms (-9.9580%) below
that previous K-pack measurement. This cross-run comparison is supporting
evidence; the matched current ABBA above is the primary performance comparison.

## Correctness and selection evidence

- Five caller host regression tests pass.
- All 17 selected SIMT/paired-entry numerical points pass. Their M1-selected
  recipes are exercised at tokens 1..8; this is not new M2..8 performance ranking.
- Both cold and hot TP2 processes pass 72 matrix cases, six chains and six
  segmented-upload cases, with three graph replays. Existing BF16 TC chain
  arithmetic and its separate high-precision error remain visible.
- Both numerical token-batch runs finish with finite likelihood metrics.
- Both measured K-pack processes report complete selected coverage and zero
  legacy fallbacks. Q8 QKV/attention gate use C8/W4/P4/S8; Q4/Q5 indexed
  decode uses BF16 SIMT. The Q6 large output head retains the confirmed TC S1
  winner. Dense activation compute remains F16, not indiscriminately BF16.
- Forty-eight shared-expert gate/up weight pairs and 96 per-device paired
  plans are recorded. The TP2 shared Q8 pair is SIMT S1/W8. Q4 TP2 paired
  projection is not forced; its unlike-scope component comparison remains open.

These are caller plan and successful execution results. The absence of Asys
means they are not a per-symbol kernel-duration or MBU measurement. In
particular, do not attribute the complete model gain to Q5's changed compiler
address lowering, or claim its isolated new-image latency is now proven.

## Model numerical limits

The corpus hashes match the previous run. These are short likelihood checks,
not a renewed task-accuracy evaluation or a bit-exact equivalence claim.

| Token batch | Scored tokens | PPL ratio | Mean KLD | Maximum KLD | Same top probability |
|---|---:|---:|---:|---:|---:|
| 1 | 254 | 0.996174 | 0.002444 | 0.296673 | 98.819% |
| 2048 | 2046 | 1.001439 | 0.001621 | 0.135317 | 99.316% |

Previous token-batch-1 mean KLD was 0.001708, maximum 0.176387 and same-top
99.213%. The new decode metrics are finite but have drifted; do not relabel
the short gate as unchanged precision or full accuracy admission. The current
token-batch-2048 metrics match the previous run at logged precision. Preserve
`accuracy_admission=PENDING_REVIEW` for the broader model acceptance decision.

## Remaining gap: Asys only

The runner exits at `model-trace`, before loading the reference model. Both
session preflight attempts exit 15 with `session ... create timeout` and
`Session ... not exist`. No current reference or native report was captured.

The CLI is from the selected 2.1.1 SDK, but the recorded shared `traced` and
`traced_perf` executables are under `/usr/local/PPU_SDK`. The device-service
PID file points to a nonexistent process, and the probe-service PID file is
absent. These establish an unhealthy/mixed-path profiler service state, not
a proven unique cause. The static environment checker additionally reports
driver 2.1.0 versus SDK 2.1.1. Do not kill shared services, remove their locks,
reset devices, or blame successful kernel execution without a discriminating
test.

Next: use existing `tools/run_kpack_model_trace.py --previous <this run>
--model qwen35-122b-q4km --profiler-scope private`, with the same caller,
SDK, compatibility bundle, model/JIT caches and devices. Private PID/mount
namespaces keep host profiler services untouched; container permission is
required and has not been proven by this upload. This is trace-only: no
library rebuild, sweep, numerical rerun or replacement of valid ABBA evidence.
If namespaces are denied, request a permitted isolated profiler environment
instead of escalating to a global service reset.

### Trace-only handoff

The source-only follow-up makes caller/SDK/result paths default to this run's
receipts and discovers the compatibility bundle by exact manifest hash within
the prior experiment root. Explicit mismatching overrides are still rejected.
Caller payload hashes, the runtime manifest, ABBA completion and selected
coverage are checked again; none of those completed jobs is executed again.
Any partial private capture keeps its actual report path in the failure output.

Run from `dev/dispatch-results` after fetching its trace-only follow-up:

```bash
(
    set -e
    cd /sim/eec/shared/junfu.qx/quactlize
    TRACE_PYTHON=$(command -v python3)
    test -x "$TRACE_PYTHON"
    source /sim/eec/shared/junfu.qx/ppu-sdk-2.1.1-a5c56e/PPU_SDK/envsetup.sh
    unset QUACTLIZE_PPU_BUNDLE
    CUDA_VISIBLE_DEVICES=0,1 "$TRACE_PYTHON" -u tools/run_kpack_model_trace.py \
        --previous /sim/eec/shared/junfu.qx/kpack-newbox.QYl8F9/kpack-tp2.8VYueY \
        --model qwen35-122b-q4km --profiler-scope private
)
```

The new result directory is beside the previous run. Upload the printed
`.results.tgz`; the separate `asys_reference` and `asys_kpack` paths identify
the large reports to open on the box. First complete requests are excluded.
No caller update/build or LFS download is required for this retry. The original
Docker shell survives a failure because the shell block is a subshell.

Local follow-up validation: 285 host orchestration/regression tests pass,
including receipt-derived paths, exact compatibility discovery, mismatch
rejection and reporting partial private captures. The unchanged seven-module
runtime verifies against the live SDK/source contract. SDK binary inspection
finds the Perfetto endpoints under `/tmp/asight`, consistent with isolating
that directory; successful PPU capture and namespace permission remain pending.
