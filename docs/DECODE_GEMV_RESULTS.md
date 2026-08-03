# Decode GEMV support and measurement

Current result: all five GGUF k-quants are independently validated through four production CUDA-core launches:
`FULLY_QUANTIZED` dense/MoE and `SCALE_FIRST` dense/MoE. The MoE launches use grid.z experts and gathered ragged
rows; the scale-first launches consume a resident `(low, high, scale, zero)` artifact.

The gate is `test_device_decode_routes_match_official_oracle_and_reject_planted_faults` in
`tests/test_gguf_routes.py`. Its fixture is asymmetric (`n=24`, `k=2048`, four different experts, rows
`[2,0,3,1]`). Each launch must first make the official `gguf` oracle reject a launch-specific planted fault. The
20 positive comparisons then require conditioned error below `2^-11`; observed errors are `4.64e-5..1.21e-4`.

## Reproduce the measurement

```bash
python benchmarks/decode_routes_bench.py --reps 9 --experts 8
```

Measured 2026-08-03 on an RTX 5090, `K=2048`, nominal peak 1.792 TB/s. `N` is the output dimension per expert;
MoE processes one row for each of eight experts. Cold is one launch after an L2 flush. Warm timings batch up to 128
launches under one event pair, which resolves the 2.048-us event quantum at the shipping shape. Cold `% peak` is
resident operand traffic divided by time and peak HBM. Warm HBM-equivalent percentages can exceed 100% at N=2048
because the operand is L2-resident; they are cache rates, not claims about DRAM efficiency.

### Saturated shape: N=131072

| format | route | cold µs | cold Gelem/s | cold % peak | warm µs | warm Gelem/s |
|---|---|---:|---:|---:|---:|---:|
| Q2_K | native dense | 79.872 | 3360.8 | 61.9% | 60.192 | 4459.7 |
| Q2_K | native MoE | 595.968 | 3603.4 | 66.4% | 584.064 | 3676.8 |
| Q2_K | scale dense | 83.968 | 3196.9 | 89.4% | 85.248 | 3148.9 |
| Q2_K | scale MoE | 645.120 | 3328.8 | 93.1% | 643.552 | 3336.9 |
| Q3_K | native dense | 120.832 | 2221.6 | 53.5% | 110.912 | 2420.3 |
| Q3_K | native MoE | 1003.520 | 2140.0 | 51.5% | 1006.048 | 2134.6 |
| Q3_K | scale dense | 104.448 | 2570.0 | 89.8% | 104.992 | 2556.7 |
| Q3_K | scale MoE | 808.960 | 2654.6 | 92.7% | 812.800 | 2642.1 |
| Q4_K | native dense | 126.976 | 2114.1 | 66.6% | 127.584 | 2104.0 |
| Q4_K | native MoE | 942.080 | 2279.5 | 71.8% | 937.952 | 2289.5 |
| Q4_K | scale dense | 102.400 | 2621.4 | 91.6% | 103.904 | 2583.5 |
| Q4_K | scale MoE | 794.624 | 2702.5 | 94.4% | 796.928 | 2694.7 |
| Q5_K | native dense | 143.360 | 1872.5 | 72.0% | 143.936 | 1865.0 |
| Q5_K | native MoE | 1103.872 | 1945.4 | 74.8% | 1096.096 | 1959.2 |
| Q5_K | scale dense | 122.880 | 2184.5 | 91.5% | 124.288 | 2159.8 |
| Q5_K | scale MoE | 952.320 | 2255.0 | 94.5% | 954.784 | 2249.2 |
| Q6_K | native dense | 161.792 | 1659.1 | 76.1% | 160.256 | 1675.0 |
| Q6_K | native MoE | 1179.648 | 1820.4 | 83.5% | 1182.784 | 1815.6 |
| Q6_K | scale dense | 163.840 | 1638.4 | 91.5% | 174.688 | 1536.7 |
| Q6_K | scale MoE | 1269.760 | 1691.3 | 94.5% | 1270.976 | 1689.6 |

### Shipping shape: N=2048

| format | route | cold Gelem/s | cold % peak | warm µs | warm Gelem/s | warm HBM-equivalent |
|---|---|---:|---:|---:|---:|---:|
| Q2_K | native dense | 409.6 | 7.6% | 4.604 | 911.0 | 16.8% |
| Q2_K | native MoE | 1638.4 | 30.3% | 10.405 | 3225.0 | 59.6% |
| Q2_K | scale dense | 686.2 | 19.2% | 2.812 | 1491.8 | 41.8% |
| Q2_K | scale MoE | 2048.0 | 57.4% | 8.315 | 4035.7 | 113.0% |
| Q3_K | native dense | 292.6 | 7.1% | 7.117 | 589.4 | 14.2% |
| Q3_K | native MoE | 1260.3 | 30.4% | 17.252 | 1945.0 | 47.0% |
| Q3_K | scale dense | 512.0 | 17.9% | 3.614 | 1160.7 | 40.6% |
| Q3_K | scale MoE | 1638.4 | 57.3% | 11.516 | 2913.7 | 101.9% |
| Q4_K | native dense | 292.6 | 9.2% | 7.816 | 536.7 | 16.9% |
| Q4_K | native MoE | 963.8 | 30.4% | 29.227 | 1148.1 | 36.2% |
| Q4_K | scale dense | 686.2 | 24.0% | 2.716 | 1544.0 | 54.0% |
| Q4_K | scale MoE | 1820.4 | 63.7% | 7.857 | 4270.4 | 149.4% |
| Q5_K | native dense | 227.6 | 8.8% | 10.585 | 396.2 | 15.3% |
| Q5_K | native MoE | 819.2 | 31.6% | 33.364 | 1005.7 | 38.7% |
| Q5_K | scale dense | 512.0 | 21.5% | 3.212 | 1305.6 | 54.8% |
| Q5_K | scale MoE | 1638.4 | 68.8% | 10.317 | 3252.5 | 136.5% |
| Q6_K | native dense | 292.6 | 13.4% | 7.660 | 547.5 | 25.2% |
| Q6_K | native MoE | 963.8 | 44.3% | 17.206 | 1950.2 | 89.6% |
| Q6_K | scale dense | 514.0 | 28.7% | 3.421 | 1225.9 | 68.5% |
| Q6_K | scale MoE | 1367.1 | 76.4% | 12.040 | 2786.8 | 155.8% |

## Deliberately not filled

`SCALE_FIRST × DENSE` remains PARTIAL. The named seam, `fpA_intB_ppu.cuh`, hardcodes `cutlass::int4b_t`; it has no
int2 dense instantiation for Q2_K and no second-plane input/converter for Q3_K/Q5_K/Q6_K. That is a missing compute
mechanism for four of five formats, not host wiring. Q4_K could fit the int4 representation, but its existing dense
harness is a self-comparison and therefore could justify only IMPLEMENTED, not VALIDATED.
