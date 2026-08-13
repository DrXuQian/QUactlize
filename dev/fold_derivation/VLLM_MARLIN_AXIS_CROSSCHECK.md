# vLLM Marlin axis cross-check

This is the read-only INBOX 143 audit.  Its purpose is to identify performance
search axes, not to turn every vLLM feature flag into a quactlize tactic.  The
source snapshot is the vLLM export pinned as commit
`11ba93f3646d4c5476c3b3fd56835589701f0fb1` by
`/root/ref5090/marlin/fullrun/README.md`.  The two primary translation units
audited here have SHA-256 hashes:

- MoE `ops.cu`: `1c19218f6906f5a48d7692dafd36ca2c7afc56c10779bcdc2efb295bbcb9c8d4`;
- dense `marlin.cu`: `522ca0a4077ded473c2415e2bc20c14569e2c523e3456f632bbd71202befa00f`.

The result is deliberately three-state: common coverage, vLLM-only, and
quactlize-only.  “Template parameter” below does not automatically mean
“search axis”; several parameters encode an input representation or are fixed
by the architecture/shape.

## Corrections to the starting observations

1. **MoE does not take the first legal tuple.**  It visits all three priority
   tuples, reads the real instantiated kernel's `numRegs`, combines that with a
   host shared-memory formula, and retains the tuple with the largest
   `allow_count`; list order only breaks ties because the update is strict
   `>` (`moe/.../ops.cu:290-339`).  A work estimate can lower `allow_count`
   further (`:330-333`).
2. **Dense really does take the first legal tuple.**  Its selector returns at
   the first legal/generated kernel and always returns `blocks_per_sm=1`
   (`quantization/.../marlin.cu:276-324`).  The real-register occupancy logic
   exists only in the MoE path, not dense.
3. **`has_act_order` and `has_zp` are not independent kernel template axes.**
   The generated selectors do not branch on those two `get_marlin_kernel`
   arguments.  The kernel derives act order from `group_blocks==0` and zero
   point presence from the weight type (`moe/.../marlin_template.h:354-372`;
   the dense template has the same derivation).  Their presence in an ops
   function signature is not evidence of an independently generated family.

## What vLLM actually selects

### MoE

The generated kernel type varies over activation/weight/output/scale types,
CTA threads, M/N/K block extents, the special logical-M=8 bit, stages,
`group_blocks`, and `is_zp_float` (`moe/.../kernel.h:27-45` and
`generate_kernels.py:44-58,248-260`).  In the exported generator,
`is_zp_float` is instantiated only as false, so true is an API-visible mode,
not an available search candidate in this snapshot.

The automatic geometry is only three bound triples:

| band | `(thread_k, thread_n, threads)` in priority order |
|---|---|
| small M block | `(128,128,256)`, `(64,128,128)`, `(128,64,128)` |
| larger M block | `(64,256,256)`, `(64,128,128)`, `(128,64,128)` |

They are declared at `moe/.../ops.cu:126-146`.  Legality checks K/N
divisibility, minimum K/N extents, at least 128 threads, host-estimated shared
memory, and existence of a generated selector (`:226-257,290-318`).  The auto
`blocks_per_sm` calculation is

```
min(255 KiB / real-register-bytes,
    max-shared-memory / (host-cache-size + 1536))
```

then hard-clamped to 1..4 for one M block and 1..2 otherwise
(`:320-328`).  An explicit thread-K/thread-N route also accepts an explicit
blocks-per-SM value, but bypasses the real-register calculation and only
rechecks the partitioned host shared-memory estimate (`:467-507`).  Therefore
this is a runtime launch parameter and static-residency heuristic, not a
measured performance sweep.

Stages are architecture-derived (SM75=2, otherwise 4), while
`thread_m_blocks` and `m_block_size_8` are determined by the requested MoE
block (`:357-359,441-453`).  These are specializations, not choices for one
fixed problem.  The kernel also fixes a DP plus two-output-tile Stream-K
hybrid, including group-boundary alignment of stripes
(`moe/.../marlin_template.h:385-415`); scheduler algorithm is not swept.

### Dense GPTQ Marlin

The generated dense kernel varies the same type family plus CTA threads,
M/N/K blocks, logical-M=8, stages, `group_blocks`, and `is_zp_float`
(`quantization/.../generate_kernels.py:44-64,197-229`).  Its geometry is the
small/large three-tuple lists at `marlin.cu:133-153`, followed by a low-work
override to `(128,64,128)` (`:458-470`).  The generator's union is four
triples and M blocks `{0.5,1,2,3,4}`, hence logical
`TM={8,16,32,48,64}`.  Runtime stages remain SM75=2 or otherwise 4
(`:407-411`), and M is split by a separate launcher heuristic
(`:423-438`).

The four generated N/K/thread triples imply warp grids
`2N x 4K`, `4N x 2K`, `2N x 2K`, and `1N x 4K`; they all have per-warp
`WN=64, WK=32`.  vLLM does not expose independent WM/WN/WK axes: the triple
binds them together and a warp spans the CTA's M extent.

## Three-state difference

The table separates equivalent-performance axes from functionality.  A
feature in the middle column is not automatically a missing tactic.

| state | geometry / launch search | representation or functionality |
|---|---|---|
| **Both cover** | Logical M=8 and M16/32/64 families; N64/128/256; K64/128; 128/256-thread N/K warp organizations; shallow/deep architecture-appropriate stages; launch residency as a concept | FP16 activation, ordinary W4 grouped scaling, optional scale/zero semantics, dense and grouped operation |
| **vLLM has; quactlize lacks** | Dense `TM=48` is the only clear geometry point in vLLM's generated set absent from our Cartesian domain.  It is an unmeasured candidate, not a reason to add an axis by inspection. | Act order; BF16/INT8/FP8 activation families; broader GPTQ/AWQ INT8, FP8/NVFP4/MXFP4 representations; fused bias, top-k weight and global scale; packed/float zero variants; atomic/fp16/fp32 reduction policies.  These alter representation, precision, or mathematical epilogue semantics and are functionality families, not equivalent tactics. |
| **quactlize has; vLLM lacks** | Independent `TM={8,16,32,64,128,256}`, `TN={16,32,64,128,256}`, `TK={32,64,128,256}`, `WM={8,16,32,64}`, `WN={16,32,64,128}`; stage rows beyond 2/4; B-chunk; independent ArtifactTileK/TacticTileK; explicit WarpK capability for the proven ordinary-int4 classic seam; DP/persistent/StreamK/Marlin scheduler families; Marlin BPC sweep up to the real active-block limit. | int1/int2 and Q3/Q5/Q6, folded and two-plane artifacts, versioned offline placement. |

Two qualifications keep this comparison honest:

- Explicit WarpK is not yet a full-format Cartesian row axis.  Current source
  exposes the proven ordinary single-plane int4 F1/WK32 classic seam and fails
  other formats closed (`ppu_tactic_space.hpp:191-201`).
- ArtifactTileK is a resident-format identity shared by tactics, not merely
  another performance integer.  Its separation from TacticTileK still makes
  our available consumer space strictly broader than vLLM's bound tuples.

## `has_act_order`: functionality gap, not search axis

Act order is determined by the presence of `g_idx` and `perm`
(`moe/.../ops.cu:731-752`).  Enabling it launches a separate A-column
permutation, changes A/top-k handling, and—when K is not full—switches the
kernel to per-K group-index/scale behavior (`:399-433`).  Full-K can drop back
to the non-act-order kernel only after the representation has been permuted.

Consequently true and false do not compute the same fixed input using two
interchangeable tactics.  Quactlize's lack of this contract is a **functionality
gap**.  If implemented, act-order and ordinary inputs should form separate
correctness/representation families, each tuned internally; adding a bool to
the existing tactic enumeration would be a category error.

## Decision

No vLLM axis is added by this audit.  For a fixed format and correctness
contract, only geometry/residency choices can presently qualify as performance
axes.  vLLM supplies one unmeasured dense geometry candidate (`TM=48`) and a
useful external residency heuristic, but its auto path is not a performance
sweep and is not an authority for our search policy.

In particular, quactlize must not regress from real instantiated
`SharedStorageSize`/`maximum_active_blocks()` evidence to vLLM's mixed evidence
level (real register count plus a host shared-memory formula).  The latter is
exactly the class for which our independent type-size audit already found a
16-byte host/type discrepancy.
