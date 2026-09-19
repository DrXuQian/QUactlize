# TP2 decode fast-path generalization

## Authority and scope

Baseline source: `c05900e2d0fe691981ca6f768e72547536d4a588`.
Runtime artifact: `006aa757f5b995ae4d8eac772cc2106782e14e90`.
Execution SHA256: `b1816dec751969548326ace1793fe04fb7c15f0e581aecb0d59f952526fb2bca`.
Caller binary source: `e15d7b411d5900be5be348105f6107fe5a77edc9`.
Returned evidence: `/root/kpack-tp2.om9JQj.results.tgz`.

The completed Qwen3.5-122B TP2 benchmark has warmed ABBA decode medians
12724.30859375 us/token reference and 12482.08203125 us/token K-pack.
The profiler failed during empty-process session creation. There is no valid
per-kernel trace from that run and no measured component MBU to infer from it.

Priority is M1 on the actual local TP2 shards below. M2..8, non-unit strides,
invalid IDs, tail geometry, large BF16 activations and previous admitted shapes
are correctness/regression controls, not permission for a Cartesian sweep.

| Operator | Q | Local N | Local K | Current selection |
|---|---:|---:|---:|---|
| Dense QKV | 8 | 6144 | 3072 | SF TC, TM16/TN64/TK64/WN16/DN64, S8 |
| Dense attention gate | 8 | 4096 | 3072 | Same SF TC S8 |
| Dense attention Q | 8 | 8192 | 3072 | SF TC, TM8/TN64/TK64/WN16/DN32, S8 |
| Dense attention/SSM output | 8 | 3072 | 4096 | SIMT V1/C4/W8/P4/S1 |
| Shared expert gate and up, each | 8 | 1024 | 3072 | SIMT V1/C4/W8/P2/S1 |
| Shared expert down | 8 | 3072 | 1024 | SIMT V1/C4/W2/P2/S1 |
| Dense attention K/V | 8 | 256 | 3072 | SIMT V1/C4/W8/P2/S1 |
| Output head | 14 | 124160 | 3072 | FQ TC, TM16/TN64/TK128/WN16/DN64, S1 |
| Routed merged gate/up | 12 | 1024 | 3072 | SIMT V3/C4/W4/P4/S1 |
| Routed down | 13 | 3072 | 512 | SIMT V3/C4/W2/P8/S1 |

Dense uses F32 endpoints/F16 activation compute; routed MoE uses F32
endpoints/BF16 compute. Accumulation is F32. No F16 intermediate in the BF16
path, clipping, quantized activations, offline byte-layout change or ABI change.

## Shape restrictions found in active production code

An exact measured selector is not itself a bug. The issue is making a reusable
implementation unavailable outside that selector. Separate structural legality,
the implementation/config inventory and measured selection authority.

| ID | Location | Current restriction | Generalization experiment / retained constraints |
|---|---|---|---|
| G01 | `quactlize/execution/simt_q8_vector.cuh::launch_v2` | Q8 hoist only for a few old N/K + geometry combinations, F16 only | Expose hoist as a reader option across valid N/K, S1 and bounded splits. Compare the non-hoist incumbent; check register pressure, aligned/scalar metadata loads and both compute types separately. |
| G02 | `simt_q8_vector.cuh::kernel_model`, `simt_kernel.cuh::register_reuse_model`, `fusion/simt.cu::simt_gate_up_model` | Fixed dimensions are embedded in the kernel body, including Q5 N2048/K512 | Parameterize N/K in generated specializations, with a runtime-dimensional implementation. Never remove the call guard while leaving assignments to the old N/K. |
| G03 | `quactlize/execution/simt_kernel.cuh::launch_v2` | Q5 unsigned-index/shared-fold optimization only BF16 indexed N2048/K512, top8/E256, one config | Compare `Changes=3` versus current `Changes=0` with actual dimensions, starting at N3072/K512. Prove index range/overflow and output ownership; separate fold and index changes if needed. |
| G04 | `simt_q8_vector.cuh::launch_v2` reducer branch | Ordered float2 reducer only dense M1 Q8 N2048/K4096, V5/C8/W4/P4/S8 | Reuse for other even N with identical `[row,split,N]` layout, 8-byte base alignment and correct row stride. S2/S4/S8 and other qtypes are candidates; keep scalar fallback. The current flat reducer cannot be used as-is for arbitrary M/indexed output stride. |
| G05 | `quactlize/execution/moe_prepare.cuh::admitted` | All-SIMT prepare gated on old gate/down N/K and tokens 1/2/4/8 | Candidate eligibility follows all-SIMT endpoints, router semantics, top8/E256 and capacity, not weight N/K. Its all-SIMT body does not read weight matrices. Test tokens 1..8, aliasing, ties, invalid IDs and changing graph replay. Retain TC/mixed fallback until separately validated. |
| G06 | `fusion/llama_graph.hpp`, caller copy `quactlize/gate_up_graph.hpp`, `fusion/validation.hpp::select` | Shared Q8 gate/up+SwiGLU only logical N512/K2048 | Structural graph matcher plus generic paired layout/query; select config independently. Add the actual TP2 logical N1024/K3072. Preserve equal inputs/shapes, contiguity, single output ownership and graph liveness checks. |
| G07 | `dispatch/binding.cpp::quactlize_kpack_dispatch_moe_bind_gate_up_v1`, caller `quactlize-execution.cu` | Routed paired fusion only physical N1024/K2048. Caller `paired_call()` and `paired_log()` hard-code logical N512/K2048; library sets `v.n=512` | Parameterize all layers together from local shard dimensions; require `gate.n == 2*down.k`, matching K/experts/metadata and PairedN4 layout. Test K3072 and K-split/local dimensions. Removing only the admission check would use wrong addresses. |
| G08 | `quactlize/dispatch/policy.hpp` and small-M tables | Old grouped compact overrides only two exact shape families; other TP2 dense paths use predicted buckets | Sweep a bounded inventory retaining actual incumbents and old measured winners. Exact entries may stay exact, but fallback must be able to choose the newly measured implementation/config rather than always old TC/SIMT. |

Q4/Q5 H32 scale/min extraction is already generic in `affine_selected` after
the TP2 correctness fix. It must remain generic; `Changes=1` is no longer a new
Q4 decoder improvement. The Q6 direct-metadata reader exists in the development
experiment but is not a shipping shape-limited optimization; keep it as a
candidate for the large head, not an alleged regression in production.

Also inspected dequant and TC/reducer implementation boundaries. Transport
TileN/TileK, E256/top8 warp-router topology, M1-only flat reduction layout,
vector alignment and caller graph liveness are structural restrictions, not
model shape constants to delete. Extending E/topk needs a new algorithm and
its own tests; this task first varies N/K and M within the existing contract.

## Order and admission

1. Restore trace capture using unchanged runtime and caller binaries; exclude
   the first complete request in each process. Do not rerun the completed
   numerical or ABBA phases. Record CLI and backend service identities.
2. Compare G05/G06/G07 (prepare and lost fusion) and G01/G03/G04 (readers/reducer)
   as separate candidates, not one inseparable rewrite. G02 is parameterization,
   not a license to instantiate every possible shape or inflate the library.
3. Derive lane N/K addresses, contiguous widths, useful/duplicate A/B/metadata
   bytes and 32/64/128-byte footprints before each kernel change. Inspect PPU
   load widths, memory traffic, bank conflicts, register pressure and stalls.
4. Use independent GGUF/typed reference, poison/guards, changing replay and
   negative controls. Stop timing a failing specialization, preserve independent
   passing cells and use fresh child processes after runtime errors.
5. Rotating active-weight ring >=2.25 times verified 64 MiB L2; same-device
   alternating incumbent/candidate samples. Time the full call including real
   reducers, then the complete fusion chain. Exclude JIT/first graph replay.
6. Promote only measured non-regressing scopes, retain a safe fallback and
   repeat warmed model ABBA. Asys is diagnostic, not the TPOT timing source.

The 40% small / 60% large MBU goals use the user's 2700 GB/s roof. Useful weight
bytes/event time is modeled bandwidth, not measured DRAM utilization. Report
missing raw-reference comparisons and failures; do not claim global optimality.

## Local versus box

Local: audit, parameterization, host eligibility/negative tests, narrow SDK
compile and ISA inspection. Box: PPU numerics, cold event timing, Asys/ACU and
model replay. Do not edit the currently pinned kernel checkout before the
unchanged-runtime trace; candidate builds belong in a separate worktree.
