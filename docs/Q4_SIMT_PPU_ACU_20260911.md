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
| Global-load instructions (`ws__inst_executed_op_vmem_ld.sum`) | 92,160 | 286,720 |
| Transactions/global-load instruction | 8 | 8 |
| KVD/LSU global-read transaction bytes | 47,185,920 | 146,800,640 |
| KVD bytes loaded from L2 | 24,778,496 | 90,074,240 |
| L2 request hit rate, % | 4.6601 | 73.1301 |
| PU executed instructions | 9,689,600 | 10,214,720 |
| Vector-memory-pipe-busy stall / issue-active ratio | 0.000839 | 0.615353 |
| LSU active cycles, % elapsed | 35.3826 | 60.8853 |

The rotating regression is 25.9845%. These observations make insufficient
theoretical occupancy or a separate reducer poor explanations for this cell.
DRAM reads are nearly equal, but the new reader generates 3.11 times the
LSU read transactions and 3.64 times the KVD-from-L2 bytes. These counters
describe traffic at different cache levels, not distinct weight bytes.
Global-load instruction count also rises by exactly 3.11 times, while
transactions per instruction remain eight in both kernels. Thus this is not
proof that each individual load coalesces worse: extra load instructions and
repeated activation/metadata requests are primary suspects too. Validate
addresses, instruction widths and generated PPU ISA separately for B, A and
metadata; these aggregate counters do not identify the offending expression.

The vector-memory-pipe-busy stall metric is an issue-active-normalized ratio,
not a percentage of total kernel time. It confirms much greater vector-load
pipeline pressure, but by itself does not prove which load path causes it.

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

## RTX PRO 6000 Blackwell Server Edition controlled replay

A fresh CUDA 12.8 build was executed on a 96 GB RTX PRO 6000 Blackwell Server
Edition (188 SMs, 128 MiB L2, driver 580.119.02). The input NPZ SHA256 matches
the PPU receipt exactly; the exported standalone fixture preserves the same
official-GGUF FP64 dot oracle. Both layout recipes remain the PPU-selected
ones above; **no CUDA retuning** occurred.

| Regime | CUDA Xplane, us | CUDA new K-pack, us | CUDA delta | PPU delta |
|---|---:|---:|---:|---:|
| Warm | 10.2265 | 11.0675 | +8.2237% | +12.1645% |
| Rotating | 18.2441 | 18.2597 | +0.0854% | +25.9845% |

Six alternating-arm rounds, fifteen event samples per round, five graph
warmups discarded. Warm graphs contain 32 calls; rotating graphs contain
39 calls over thirteen copies (at least 2.25 times this device's L2).
Every graph ends on a whole-ring boundary. Maximum conditioned errors are
2.430e-5 for Xplane and 2.971e-5 for K-pack, matching the PPU observations
to the reported precision, below the unchanged 0.005 bound.

The rotating 26% regression does **not** reproduce on this CUDA device with
the same recipes. Warm still regresses by 8.22%; this is not all-regime parity
or a claim that either configuration is the CUDA optimum. Memory-request
behavior on PPU remains the immediate investigation target.

NCU 2025.1.1 is installed, but its real kernel probe returns
`ERR_NVGPUCTRPERM` even as container root. There are **no PRO 6000 hardware
counters** in this receipt. The host/container administrator must grant
profiling access; no driver/security policy was changed by this experiment.

Remote experiment directory: `/root/autodl-tmp/q4-ppu-repro.ZAlupc`.
The build took about 72.5 seconds for the candidate. `dev/gemv_cuda/build_profile_runner.py`
generates a whole-ring variant of the historical standalone harness without
changing its frozen source or any benchmark kernel. `reproduce_ppu.py` replays
the selected pair and preserves every raw sample and input hash.

Retained evidence:

- [Complete paired timing receipt](measurements/q4_ppu_fixed_pro6000_20260911.json).
- [Raw timing logs, build receipts and NCU permission failure](measurements/q4_ppu_fixed_pro6000_20260911.logs.tgz),
  SHA256 `3ee9ec26a0673ce5f1874d508980efe2c85fb33e35121538c320b6708d9157d8`.
