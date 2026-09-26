# PPU1.7 backend: Hopper validation, 2026-09-26

This report covers the isolated `dev/ppu17-cutlass360` backend, using PPU
CUTLASS3.6.0 revision `023e82d03e80b4d5982f664925e15499033df314` and CUDA12.8
SM90a code on a physical H800. It does not certify native PPU1.7 or change
PPU1.0/1.5 actlize, shipping libraries, offline bytes or production routing.

## Completed gates

| Gate | Baseline | Typed-affine-cache candidate |
|---|---|---|
| Typed CPU contracts | 5,875,034 PASS | 5,875,034 PASS |
| Live assembled specializations | 156 PASS | 156 PASS |
| Target/codegen negative controls | 20 rejected | 20 rejected |
| H800 extended numerical suite | 1,020/1,020 PASS | 1,020/1,020 PASS |
| Memcheck / racecheck / synccheck | 17 / 6 / 2 PASS | 17 / 6 / 2 PASS |
| Full output-byte A/B replay | 156/156 identical pairs | Includes output guards and scale/zero bytes |
| Q4 rotating timing screen | 72/72 PASS | 72/72 PASS |
| Hardware counters | Unavailable: `ERR_NVGPUCTRPERM` | Not collected |

Numerics cover Q2/Q3/Q4/Q5/Q6/Q8, FP16/BF16 compute with FP32 accumulation,
FQ/SF, six SIMT/WGMMA/TMA configurations, typed I/O, dense M1–8 and tile
boundaries, N tails, K1024 TMA phase wrap, ragged and empty experts. The
single-launch executable does not benchmark or claim simulator admission.

## Performance scope

- H800 PCIe80GB, 114 SMs, queried 50 MiB L2; driver595.71.05.
- Q4, N512/K2048; dense M1 and M128; grouped rows `[0,1,8,65]` (M74/E4).
- A storage matches FP16/BF16 compute, C is FP32. Same configuration and
  unchanged arithmetic, maps, reduction order and typed rounding in each pair.
- Weight rings exceed twice the queried L2, counting only active experts.
  Every copy passes complete typed CPU dot and output-guard checks.
- Prepared descriptors, allocations, transfers, initial graph upload and
  three warmup graphs are excluded. Median of 11 graph-averaged event samples.
- SF metadata expansion is excluded. These numbers cannot choose FQ versus
  SF end to end. They are an initial implementation baseline, not tuned
  production latency, global-optimal configurations or native PPU throughput.
- The runner checks idle state, admits its own live GPU PID and samples all
  GPU processes every100ms. This is not a driver-enforced reservation. No
  foreign process was observed during the four completed screen campaigns.

Representative **same-config** screen results (microseconds):

| Compute | Workload | Route/config | Baseline | Candidate | Change |
|---|---|---|---:|---:|---:|
| FP16 | Dense M1 | FQ/SIMT0 | 165.517 | 127.115 | -23.20% |
| BF16 | Dense M1 | FQ/SIMT0 | 185.476 | 133.142 | -28.22% |
| FP16 | Dense M128 | FQ/TMA5 | 233.982 | 186.978 | -20.09% |
| BF16 | Dense M128 | FQ/TMA5 | 251.554 | 210.964 | -16.14% |
| FP16 | Grouped M74 | FQ/TMA5 | 236.263 | 194.252 | -17.78% |
| BF16 | Grouped M74 | FQ/TMA5 | 256.561 | 217.530 | -15.21% |
| FP16 | Dense M1 | SF/SIMT0 | 157.795 | 145.341 | -7.89% |
| BF16 | Dense M1 | SF/SIMT0 | 141.949 | 149.073 | +5.02% |
| FP16 | Dense M128 | SF/TMA5 | 176.279 | 184.015 | +4.39% |
| BF16 | Dense M128 | SF/TMA5 | 183.200 | 190.591 | +4.03% |
| FP16 | Grouped M74 | SF/TMA5 | 184.484 | 196.865 | +6.71% |
| BF16 | Grouped M74 | SF/TMA5 | 195.041 | 199.874 | +2.48% |

All36 FQ points improved7.24–46.10%; 29/36 SF points regressed, with the
largest regression11.46%. Reuse avoids repeated packed-metadata decoding,
but adds cache predicates/registers to the already expanded SF path. This is
a source-level explanation consistent with timing, **not a measured hardware
stall diagnosis**. Static inspection found zero stack/local bytes in both
images; register counts changed and were recorded separately.

## Decision and interruption

Do not enable the cache globally. Restore the original reader; retain the
stronger tests, reproducible runners, immutable candidate binary and patch.
The next bounded candidate should cache FQ metadata without changing SF.
Other-format performance has not been measured in this experiment.

Reverse confirmation selected configs0/1/5, including every baseline winning
config and the largest observed SF regression. Ten BF16 points completed
cleanly; the eleventh overlapped a newly started
`/workspace/gdn-sm90-library-compare-20260926/venv/bin/python` process and was
INVALID. Our benchmark stopped; no foreign job was touched. Eight BF16 points
still need valid confirmation (one invalid plus seven not run), as do all18
FP16 points. The screen is complete,
but confirmation is **incomplete**. No performance promotion follows from it.

## Reproducibility

Task source: `/workspace/quactlize-ppu17-20260926/source`.
Remote root: `/workspace/ppu17-h800-20260926`.
Local evidence: `/workspace/quactlize-ppu17-20260926/h800-evidence/results`.

| Artifact | SHA256 |
|---|---|
| Baseline library | `a3285663bff9d2527acf0e32585f5677bda8b41b8d568394857cb97129f12030` |
| Candidate library | `39847acf9a8dc62f653196d822dcb6f44c41dee297398e10d89ff0c29cf79dc8` |
| Immutable benchmark | `bf3987657eb95ab38ba1ccf17a12bbde95bf992063de53f342fc9e7bc91d0f2c` |
| Full-byte replay application | `84cdea5282834245d45ac76a1a385f9c39a8cc68bacf9ae28fd863afd985c224` |

The external `tuning-hopper` directory holds `docs/plan.md`,
`affine-reuse.patch`, `resources.json`, `benchmark.csv`, `comparison.json`,
`candidates.jsonl` and `RESUME.md`. Summaries bind logs, binaries and telemetry;
transferred numerical, memory and replay logs were hash-verified locally.
The retained reader was rechecked using
`baseline-strengthened-validation/validation.json`.

The native SDK target guard remains intentional. SDK2.1.1 cannot substitute
for a capable PPU1.7 compiler, and Hopper success cannot close native gates.
