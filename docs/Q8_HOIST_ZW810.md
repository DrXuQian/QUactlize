# Q8 hoist: bounded ZW810 candidate

Status: locally compiled and host-tested; ZW810 numerics/performance pending.
This is not a new selector or a port of the ZW610 config winners. Only the
Q8 vector header changes in the execution source receipt. Q4/Q5 readers,
metadata, MoE helpers, arithmetic, offline bytes and all policy files remain
unchanged. The existing caller only needs its trace-symbol parser updated.

| Dense F32 I/O, F16 A, F32 accumulate/output | Existing config | Grid x block | Candidate |
|---|---|---|---|
| M1 N512 K2048 | V5/C4/W8/P4/S1 | 32 x 256 | Hoist |
| M1 N2048 K512 | V5/C8/W4/P4/S1 | 64 x 128 | Hoist |
| M1 N2048 K4096 | V5/C8/W4/P4/S8 | 512 x 128, then reducer | Unchanged control |

All other shapes, M2..8, configs, indexed calls and BF16 compute retain the
narrow reader. A default-false template parameter also preserves direct
development callers. Hoist has two compiled true specializations, not a
runtime wider register window shared with Split-K.

## Address and instruction contract

`col = tile * Columns * P + (thread % Columns) * P`,
`g = partition * Workers + thread / Columns + iteration * split * Workers`,
`Workers = Warps * 32 / Columns`.

For each half/segment/residue the packed address remains
`low + ((g*16 + segment*8 + half*4 + residue)*N + col)*2` bytes.
Each thread requests eight adjacent bytes (P4); C4 supplies eight disjoint
32-byte contiguous runs per warp, while C8 supplies four 64-byte runs. Their
source 32/64/128-byte footprint utilization is respectively 100/50/25% and
100/100/50% for this load, assuming the observed aligned bases. Hoist changes
neither that footprint nor the number of source bytes. The runner records
actual base alignment and separate A/metadata footprints.

Hoist exposes 16 independent packed-row deliveries before the dot loop,
instead of retiring each four-row window. For P4, source-level live packed
words increase from 8 to 32 per thread. The native compiler reports 78 vector
registers for the two hoisted specializations versus 60 for their narrow
counterparts. This is a latency-hiding hypothesis, not an occupancy win.

I8 extraction still uses `lop3`/half2 exact integer construction followed by
FP32 FMA. The half/segment/slot/residue accumulation sequence is unchanged.
There is no FP16 accumulation or new activation clipping.

## Box protocol

Set `Q8_HOIST_AB=1`, `MODEL_COMPUTE=bf16`, `MODEL_PHASES=perf`,
`MODEL_NAMES=qwen35-35b-q4km`, `MODEL_ACU=1` when running
`tools/run_kpack_q4_model_box.sh`. Supply verified `L2_BYTES` if the SDK
reports no L2 size. The requested device is ZW810/72 CUs. The baseline artifact
is pinned separately in `tools/kpack_q8_hoist_baseline.json`.

1. Three fresh-process isolated comparisons through old/new production C ABI.
   Both images get independent GGUF correctness, same-order output-bit
   comparison, M1..8/F16+BF16 controls, weak scale alignment, changing graph
   replay and planted zero-input checks. Validate every ring copy before timing.
2. A weight ring >=2.25 times verified L2; six alternating rounds of fifteen
   samples and complete traversals. First graph upload/replay/sample excluded.
   S8 includes its existing ordered reducer. Weight-byte MBU uses the declared
   2700 GB/s reference roof, not an ACU-measured DRAM percentage.
3. Existing component gates, warmed model ABBA, then a separate Asys capture
   and exact Asys-observed M1 ACU profiles. Dense stays FP16; only MoE uses BF16.

Outputs below the printed run directory:

- `results/q8-hoist/{contract,summary,point-*}.json`: paired isolated evidence.
- `results/benchmark/`: native llama vs candidate K-pack PP2048/TG128, request
  batch1. First complete pass per process is excluded. This is not a paired
  old-K-pack/new-K-pack model comparison; historical TPOT is context only.
- `results/trace/*/{reference,native}/proof.asysrep`: full visual traces.
- `results/acu/`: exact observed producer/reducer reports and raw counters.

The result tar excludes large Asys files; full reports remain on the box.
Use normal event/model timings for admission, not profiler replay durations.
No new GPU numerical or performance PASS is claimed by the local build receipt.
