# Q4 SIMT PPU: first uploaded comparison

Source: `q4-simt-ppu.0uCEmO.results.tgz`, uploaded on 2026-09-11.
The three SIMT arms completed all twelve shape/cache rows. All twelve FQ
screen failures were the same stale-dispatcher JIT source-contract rejection;
no FQ numeric or performance result exists in this receipt.

## N=5120, K=8192, M=1

Both warm and rotating confirmations chose Xplane `C2/W8/S1` and new K-pack
`C4/W8/S1`. Neither selected call uses a reducer. Rotating uses seven distinct
23,592,960-byte low+metadata allocations, and 35 calls per timed graph.

| Observation | Xplane FP32 | New K-pack FP32 |
|---|---:|---:|
| Warm complete-call time, unprofiled | 20.6081 us | 23.1150 us |
| Rotating complete-call time, unprofiled | 22.9280 us | 28.8857 us |
| Forced-cold ACU kernel duration | 24.1065 us | 28.3871 us |
| CTA count × threads | 320 × 256 | 320 × 256 |
| Registers/thread | 80 | 60 |
| Shared bytes/block | 16,384 | 512 |
| Theoretical occupancy, % | 75 | 100 |
| Active warp occupancy, % | 53.2775 | 54.4839 |
| DRAM bytes read | 23,635,328 | 23,663,872 |
| DRAM read throughput, GB/s | 1,002.7412 | 822.6368 |
| ACU DRAM throughput, % sustained peak | 35.7832 | 30.1238 |
| KVD/LSU global-read transactions | 737,280 | 2,293,760 |
| KVD/LSU global-read transaction bytes | 47,185,920 | 146,800,640 |
| KVD bytes loaded from L2 | 24,778,496 | 90,074,240 |
| L2 request hit rate, % | 4.6601 | 73.1301 |
| PU executed instructions | 9,689,600 | 10,214,720 |

The rotating regression is 25.9845%. These observations make insufficient
theoretical occupancy or a separate reducer poor explanations for this cell.
DRAM reads are nearly equal, but the new reader generates 3.11 times the
LSU read transactions and 3.64 times the KVD-from-L2 bytes. These counters
describe traffic at different cache levels, not distinct weight bytes. The
working hypothesis is inefficient/coalesced-at-too-small-a-granularity or
repeated requests inside the hierarchy, not additional useful weight data.
This is a lead to validate against load addresses and generated PPU ISA;
the counters alone do not identify the offending source expression.

The new reader executes about 5.4% more PU instructions in this capture, not
three times as many. Its higher L2 hit rate does not imply higher efficiency:
it can coexist with extra requests and lower delivered DRAM throughput.

## Authority and limitations

- Reports: `results/n5120-k8192-xplane.acu.acurep` and
  `results/n5120-k8192-new.acu.acurep` inside the upload.
- Report SHA256 respectively:
  `df697da1b867f7d25c25637500dd64cb93f8eedbb03ccd7c86db6f75e589bb09` and
  `86042e2caa465653b6ccf9eeaaa4f012730f05ae55fae7d99120a1883826327a`.
- Uploaded archive SHA256:
  `94eab9aa135647ce40b00f6d733139ac1fc379bac9db7657858ead7f883ac8db`.
- Actual kernel names are `gguf_scale::bc_q4_gemv::kernel<2,8,1>` and
  `kpack_ppu_new_q12::kpack_q4_large_static<4,8,5120,8192>`.
- Metrics were imported locally using SDK 2.1.1 `acu --import ... --csv --page raw`.
  `--page details` emitted only a header in this local import; no empty detail
  table was interpreted as a zero metric.
- `ppu__time_duration.sum` is converted from ns to us. Relevant metric names:
  `dram__bytes_read.sum`, `dram__bytes_read.sum.per_second`,
  `ppu__dram_throughput.avg.pct_of_peak_sustained_elapsed`,
  `kvd__transactions_pipe_lsu_mem_global_op_ld.sum`,
  `kvd__bytes_pipe_lsu_mem_global_op_ld.sum`, `kvd__bytes_load_pipe_l2.sum`.
- Profiling used full metrics, kernel replay, explicit cache-control `all`,
  27 passes. It is not the unprofiled rotating graph's timing authority.
- Both profiler logs warn that PID 3481764 (`python3`) has another device
  context. This alone does not establish simultaneous compute: the campaign
  parent also owns a context. Do not assert either proven interference or
  proven exclusivity from that warning alone.
- ACU device attributes report 72 CUs and 67,108,864 LLC bytes. Thus the 64 MiB
  override is now corroborated; the runner's properties-ABI `sm=1` is not the
  physical SM count. These SIMT grid formulas do not use that reported value.

The PPU-specific memory-transaction behavior is a reason to retain native
ACU/ISA checks even when NVIDIA NCU optimization succeeds. BF16 Tensor Core
peak alone cannot predict FP32-accumulating SIMT GEMV rankings.
