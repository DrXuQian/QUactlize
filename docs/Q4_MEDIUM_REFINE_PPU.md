# Final medium Q4 cold refinement

Status: **locally compiled and host/ISA checked; PPU correctness/performance
pending**. Only dense M=1, Q4_K N1024/K5120 remains above the user's ref+5%
line. The [previous result](Q4_SMALL_LATENCY_RESULTS_20260912.md) is
5.582308 us versus 5.302115 us reference: +5.2845%, leaving 0.015086 us
to remove. The other five shapes are closed and are not remeasured here.

Canonical bytes, dequantization, FP32 dot/reduction order, shipping DSOs,
production selection and llama.cpp are unchanged. This is not a new format,
precision downgrade, Split-K or a different reduction tree.

## Concrete native issue

The old CTA-fold loop starts from a runtime signed expression:

```cpp
for (int w = tid / TileN; w < Warps; w += 32 / TileN)
    sum += partial[w * TileN + tid % TileN];
```

The legal first-warp domain requires at most three additions at P2/W10,
but the existing native image retains sixteen shared-load sites and a long
chain of conditional branches/merge masks, including comparisons with
negative bounds. These are **static code sites, not sixteen executed reads
per thread**. The previous numerical result was correct.

R1 expresses the fixed number of stripes with compile-time recursion and
an explicitly bounded lane. It reads the identical shared slots in the
identical order, including the partial final stripe. The unchanged shuffle
reduction and output store follow it. U1 separately makes only the existing
thread/K-work indices unsigned; all valid values are nonnegative. Either
can make the legal range easier for the compiler to see. No new barrier is
introduced, and a static improvement is not a measured speedup.

| P/W | Variant | CTA shared-load sites | CTA conditional branches |
|---|---|---:|---:|
| P2/W10 | old R0/U0 | 16 | 18 |
| P2/W10 | fixed fold R1/U0 | 3 | 3 |
| P2/W10 | unsigned R0/U1 | 3 | 3 |
| P4/W20 | old R0/U0 | 16 | 18 |
| P4/W20 | fixed fold R1/U0 | 10 | 2 |
| P4/W20 | unsigned R0/U1 | 10 | 2 |

All 22 contexts retain one CTA barrier, fast lop3/half2 code conversion and
FP32 FMA. The package's `isa-stats.json` records the whole CTA-fold native
sequence and per-context counts. Neither fewer static instructions nor
higher occupancy is itself the device-admission criterion.

## Bounded search and preserved anchors

Two immutable, independently timed anchors are retained from the previous
package. Their tiny previous timing difference is not a robust unique winner:

- `kpack-p2`: `affine-h1-l1-a1-w10-c4-p2`, 5.582308 us;
- `kpack-p4`: `affine-h1-l0-a0-w20-c4-p4`, 5.585577 us.

Those numbers select anchors only; every comparison time is measured again.

| Family | Warp counts | New factors | Contexts |
|---|---|---|---:|
| P2, C4, H1/L1/A1 | 8,10,12,16,20 | R0/R1, signed indices | 10 |
| P4, C4, H1/L0/A0 | 10,16,20,24 | R0/R1, signed indices | 8 |
| Two anchor geometries | P2/W10, P4/W20 | R0/R1 with unsigned indices | 4 |

Total22, explicitly enumerated by `medium_refine.py::plan()`. No repeated
A0/A1 source-vector search, shared-A staging, new header factor, AIU path or
other shape/qtype is added. Every K tail contains whole eight-group warps;
inactive warps contribute zero through the same CTA reduction. Grid is128
for P2 and64 for P4; threads are32*Warps. The plan prints passes, active
last-pass workers and logical shared-array bytes. ACU's allocated shared
size can include compiler/hardware padding and is reported separately.

For each candidate, the inherited A/B/unit source address patterns and
actual base alignment are recorded with32/64/128-byte footprints. R and U
change neither the physical addresses nor the logical global byte counts.
P2 lanes request4B of B; four adjacent lanes form16B, 25% of an aligned64B
footprint. P4 lanes request8B, forming32B, 50% of that footprint. Warp-count
changes alter concurrency and passes, not this per-request packing. These
are source models, not measured DRAM utilization.

## Numerical and local evidence

The actual `medium_reduce.hpp` helper is host-callable. Tests compile it
with address/undefined-behavior sanitizers and compare every first-warp
lane, all nine geometries and128 finite/cancellation fixtures against the
old loop's raw FP32 bits. Wrong-column and zero-fold plants are rejected.
Separate coordinate tests check exact K-pack B coverage and unchanged
per-lane read/add order. The dot/dequant and final shuffle/store source
sections remain identical to the parent.

Each box candidate must then pass independent original-GGUF dot, zero-code,
zero-A, guards, deterministic graph replay and exact FP32 comparison with an
untouched parent at the same geometry. At the two anchor geometries the
rebuilt parent additionally matches the actual immutable old DSO. Other
warp counts are new instantiations of unchanged code, not historical device
evidence. All paths accumulate/output FP32; cross-family rounding remains
different from the raw reference's per-weight FP16 reconstruction.

Local build took13.8seconds. The one experiment DSO and receipts total about
744KiB. Host tests and compilation do not establish PPU execution or speed.

## Box run

- Screen22 contexts plus four anchors, five samples each:26 timing cells.
- Confirm top three contexts plus both K-pack anchors/ref/Xplane with six
  alternating rounds x15samples:42 timing cells.
- Profile up to two confirmed **different geometries**, plus all four
  anchors:5–6 ACU reports. Avoid profiling two fold/index aliases of one
  geometry at the expense of the other anchor. ACU times do not replace
  event medians.

Total **68 timing cells, up to6 profiles**, at most41 child processes if no
cell fails. The previous66-child run took244.8seconds; allow roughly
**2–5minutes plus fixture generation** on an idle device. This estimate is
based on whole-run overhead, not just kernel latency, and is not a deadline.

```bash
CUDA_VISIBLE_DEVICES=0 \
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
L2_BYTES=67108864 \
bash tools/run_q4_medium_refine_ppu_box.sh
```

Pull the new LFS payload and its existing control packages before running.
No compilation/JIT happens on box. First-use/graph-upload/setup are excluded,
the weight ring remains at least2.25x verified L2, and complete rings are timed.
Do not run inference on the same device concurrently.

`FIXTURES` can reuse the preceding fixture directory; `RESUME_RUN` reuses the
exact printed result directory after verifying source/device/runtime/fixture
identity. Valid independent cells survive a failure; only missing/failed
cells rerun in fresh processes. `ACU=0` explicitly omits profiles. Run with
`bash`, never source it: failures preserve the caller's Docker shell.

Return the printed `results=...results.tgz`. It contains summary TSV/JSON,
all raw logs, plan, fingerprints, native ISA receipts and ACU reports.
Measurement completeness and ref+5% performance remain separate verdicts.
Package: `prebuilt/ppu0010/q4-medium-refine-v1`; all preceding packages stay
immutable. This is a development experiment, not a shipping-library update.
