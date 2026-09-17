# Paired-N4 gate/up: TC and SIMT

Status: implemented, host-tested and compiled for PPU0010. Device correctness,
performance and model admission are pending. No production selector or
llama.cpp caller is changed by this candidate.

## Offline contract

For each expert, start with two raw GGUF tensors of the **same qtype**, each
`[N,K]`. The physical rows of the merged tensor are:

```
G0 G1 G2 G3 U0 U1 U2 U3 G4 G5 G6 G7 U4 U5 U6 U7 ...
```

`G[n]` is physical row `8*(n/4)+n%4`; `U[n]` is four rows later. The GPU
packer uses these virtual raw rows as input to the existing canonical word
and metadata packer. Low bits, high bits and packed metadata all move together;
there is no dequantization/requantization or temporary concatenated tensor.
In particular Q5's high-plane mapping must not be permuted as a naive byte row.

The new descriptor is `qkg_gate_up_layout_v1`, with layout ID
`0x47554e3400000001`. Its nested arrangement retains the canonical bit-packing
contract; the outer ID defines the additional gate/up semantics. This is
**not** the existing gate-then-up artifact from `--fuse-gate-up-exps` and must
not reuse its disk-cache identity. Old APIs and their byte interpretation stay
unchanged. Callers must explicitly choose the new packer and consumer together.

Fused output is `[rows,N]` in original logical channel order. Down weights need
no further permutation. Expert slices remain same-index contiguous slices in
each plane; `high=NULL` is required for Q4/Q8/Q2. Raw sources and output planes
must be disjoint, and buffers must outlive queued stream work.

## Execution

| Backend | Implementation | Bounded configurations |
|---|---|---|
| SIMT | Existing fast Q8 or all-format register-reuse reader; paired final reduction/store | C4/P8, W4/W8, M1..8 |
| TC | Existing packed-metadata collective; paired accumulator epilogue | TM8/TM16, TN64, WN16, stages2 |

Qtypes: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0. TC TK is 128/256/64/256/128/64
respectively. Logical N is a multiple of 256; K obeys the canonical packing
constraints and TC additionally requires `K % (TK*split) == 0`.
This is a functional inventory, not a claim that these recipes are optimal.

Both backends support dense, compact grouped offsets, and direct indexed
expert IDs, including distinct per-slot activation channels. TC supports
multi-tile M and ragged tails. Its initial grouped grid uses a conservative
total-row bound per expert, **not** the optimized compact directory scheduler.
SIMT's grouped total-row bound is 64; dense/indexed token count is at most 8.

For the C4/P8 SIMT reader, lane `l` starts at physical column
`tile*32 + (l%4)*8`: four adjacent lanes load four consecutive 16-byte vectors
per K address. One warp has eight such K-worker groups, not one contiguous
512-byte span. The paired permutation changes the values of those adjacent
columns, not this access pattern or the number of packed bytes read. Q8 scale
loads use the same 16-byte-per-lane pattern when aligned (scalar fallback for
the public two-byte metadata alignment); K-quant metadata uses the inherited
cooperative unit loads. A ownership and fast dequant readers are unchanged.
This address calculation is not evidence of achieved DRAM bandwidth.

S1 activates only after the complete dot products. It never writes a full
gate/up intermediate to global memory. TC's PPU accumulator pairs N and N+4
in the same thread; the actual CuTe tiled ownership is tested on the host.
SIMT uses its existing warp/CTA reduction then applies the paired activation.
S2/S4/S8 write ordered `[row,split,2N]` FP32 partials; a single final reducer
adds gate/up separately and applies SwiGLU. Never apply SwiGLU to a partial sum.

Arithmetic is explicit:

- F32 storage may use F16 or BF16 A compute, converted in the reader.
- Native F16/BF16 A storage must match the selected compute type.
- Both accumulate FP32. SIMT retains FP32 group-affine weight arithmetic;
  TC reconstructs B in its selected 16-bit type. Arbitrary real weights do not
  promise bit-identical TC/SIMT answers.
- `round_projection=0` keeps complete G/U projections in FP32;
  `=1` rounds each separately to the selected 16-bit type before activation.
- Final output may be F32/F16/BF16. No clipping or INT8 activation path.

The public header is `quactlize/fusion/gate_up.h`; add `quactlize/include` to
the include search path. Call layout -> sizes/query -> pack -> run. `call.n`
is **one** projection's N; query returns the two-projection plane/workspace
sizes. `sf_plane_bytes` is inherited informational capacity, not an allocation
required by this FQ consumer. Strides count elements of their stated type.
Run performs no allocation, host routing, standalone gather/scatter or host
synchronization. It enqueues on the caller's stream, including any reducer.

## Build and validation

```bash
python3 tools/build_kpack_gate_up.py --sdk "$PPU_SDK" \
  --output /workspace/gate-up-build-r1 --jobs 192
python3 tools/run_kpack_gate_up.py --sdk "$PPU_SDK" \
  --bundle /workspace/gate-up-build-r1 --output /workspace/gate-up-results-r1
```

The build has 15 translation units, so 192 is a ceiling, not 192 busy cores.
The prebuilt box wrapper avoids compilation and JIT entirely:
`bash tools/run_kpack_gate_up_box.sh`. It fetches only the pinned small candidate,
checks hashes, executes all six format/backend pairs in separate processes,
prints progress and packages results. `BUNDLE` overrides the pinned candidate.
Set `RESUME_RUN` to the previous wrapper run directory to reuse complete
matching parts; failed/incomplete parts rerun, with earlier logs retained.

The numerical cohort uses N256/K2048, dense M1..8, indexed tokens1..8 with
channels1/8, and grouped M9/17/33 with spread or single-expert routing. TC also
tests M65/129. It checks both compute types, F32/native storage/output,
projection rounding, S1/2/4/8, strides/guards, changing-A/IDs graph replay,
large finite BF16 inputs and planted zero-A negatives. Dyadic GGUF fixtures
are exactly representable in F16/BF16, with independent official GGUF dots.
This is a numerical gate, not a performance benchmark or model-accuracy claim.
There are 1,440 SIMT and 1,728 TC configuration cells per qtype (19,008 total),
plus changing-input replays and negative controls. A part is complete only
with unique inventory coverage and its replay/negative-control evidence.

Local tests:

```bash
python3 -m pytest -q tests/test_gate_up_paired.py tests/test_kpack_device_pack.py \
  tests/test_simt_register_reuse.py tests/test_q8_topology.py
```

After the device gate: measure full-call latency against same-precision
unfused kernels and native reference on the actual shared/routed shapes;
include reducers, exclude setup/JIT, and keep cache regime fixed. Then choose
measured configs and integrate caller/cache identities. The small SIMT body
refactor keeps old entry signatures but changes JIT source fingerprints:
rebuild the small dispatcher when composing a new production package, never
mix this checkout with an old source-bound JIT dispatcher.
