# Sub-byte pointer and size unit audit

This audit was triggered by the grouped expert-pitch incident.  A CUTLASS
sub-byte scalar (`int4b_t`, `uint2b_t`, or `uint1b_t`) is a one-byte C++
wrapper, but its logical width is 4, 2, or 1 bit.  Therefore the following are
different units and may never share an unlabelled integer:

| unit | meaning | conversion to bytes |
|---|---|---|
| logical code | one quantized weight | `codes * bits / 8` |
| packed byte | physical allocation/copy offset | identity |
| AIU descriptor element | logical extent/stride consumed by the descriptor | descriptor performs its explicit bit-to-byte lowering |
| tile index | scheduler `(q,k)` unit | no byte interpretation |

## The allocator seam

`cutlass::DeviceAllocation<T>` has an asymmetric historical API:

- allocation uses `count * sizeof(T)`;
- typed host/device copies use `count * sizeof_bits<T> / 8`.

For a sub-byte wrapper `sizeof(T)==1`.  Consequently a logical-code count
allocates 2x/4x/8x too much but copies the right int4/int2/int1 payload, while a
physical-byte count allocates the right number of bytes but a typed copy copies
only 1/2, 1/4, or 1/8 of it.  Packed owners must therefore use
`DeviceAllocation<uint8_t>(physical_bytes)` plus a byte copy and introduce the
sub-byte type only at the launch ABI.

The reverse scan found exactly seven active under-copies and no eighth:

| source | packed owners | before | disposition |
|---|---:|---|---|
| `benchmarks/moe_splitk_bench_common.hpp` | 1 | int4 allocation and typed copy both received physical bytes | fixed: byte owner/copy, cast at launch |
| `benchmarks/lowbit_moe_bench.hpp` | 3 | two-plane low/high and single-plane buffers received physical bytes | fixed: byte owner/copy, cast at launch |
| `tests/test_lowbit_grouped.cu` | 3 | two-plane low/high and single-plane oracle buffers received physical bytes | fixed: byte owner/copy; per-expert offsets remain bytes |

All other typed sub-byte `DeviceAllocation` sites use logical-code capacity.
Their payload copies are complete, but their storage is intentionally
overallocated by the wrapper factor.  They fall into these finite groups:

- benchmarks: `test_grouped_int1_perf.cu`, `test_lowbit_dense_bench.cu`,
  `test_scalefirst_bench.cu`;
- derivation/device probes: `test_width_acu.cu`,
  `test_q3_bconcat_{probe,ablate}.cu`, `test_moe_grouped_dataslice.cu`,
  `test_int1_sweep.cu`, `test_fold_int2.cu`;
- correctness/diagnostic harnesses: `test_fpA_kquant_dense.cu`,
  `test_q4k_packed_gemm.cu`, `test_moe_grouped_{verify,real}.cu`,
  `test_q3_bconcat_real.cu`, `test_q65_bconcat_real.cu`,
  `test_w{1,2,4}a16_diag.cu`, `test_w{1,2}a16_grouped.cu`, and
  `test_w2a16_real.cu`.

The counts in those sites are divisible by the relevant codes-per-byte, so no
tail byte is truncated.  Perf-only allocations in
`test_moe_grouped_probe.cu`, `test_fpA_intB_ppu.cu`,
`test_moe_grouped_ppu.cu`, and `test_moe_gemm_ppu.cu` are uninitialized and do
not perform a typed copy.  They still overallocate when given logical codes,
but cannot under-copy a host payload.

## Production producer-to-consumer table

| quantity | producer unit | consumer unit | result |
|---|---|---|---|
| dense/grouped low/high buffers | backend allocation and H2D copy use packed bytes | tactic boundary casts to the declared sub-byte element | consistent |
| ordinary/fold/two-plane noninterleaved `dB` | logical sub-byte codes, including expert pitch | `mixed_subbyte_l_slice` advances a `subbyte_iterator`, checks byte alignment, then exposes a raw AIU base | consistent; fixed by `1c5f4e7` |
| two-plane `dB2` | logical high-plane codes with its own fold | same explicit sub-byte slice | consistent |
| interleaved `dB`/`dB2` | canonical inner strides are logical codes; expert pitch is explicitly packed bytes | expert byte base is selected before the logical inner tensor is formed | consistent; these two units must never be put in one raw-byte stride tuple |
| AIU descriptor extent/stride | logical elements/codes | descriptor initialization performs the explicit bit-width lowering | consistent |
| GEMV packed B | `uint8_t*` plus byte expert pitch | byte loads/decode | consistent |
| offline xplane placement/recovery | `uint8_t*`, all offsets packed bytes | `place_derived`/`recover_derived` plus roundtrip anchors | consistent |

The old `moe_gemm_ppu.cuh` uniform-M wrapper bypasses the owned seam and still
reaches a vendor collective where the noninterleaved path advances logical
codes as byte-sized C++ elements and the interleaved path gives L zero stride.
It is test-only and superseded; `test_moe_gemm_ppu` now fails closed for L>1
and is registered as `none`, not correctness evidence.

## Marlin scheduler units

The Marlin scheduler does not perform a sub-byte address calculation.  Its
complete producer-to-consumer unit map is:

| field | unit | consumer |
|---|---|---|
| `tiles_m_`, `tiles_n_`, `tiles_l_`, `output_tiles_` | output-tile counts | flatten/decode global `q` |
| `k_tiles_per_output_`, `K_idx`, `k_tile_count` | tactic-K tiles, not elements or bytes | mainloop absolute K-tile iterator |
| `total_k_tiles_`, `iters_per_block_`, `linear_*` | flattened `(q,k-tile)` cells | equal-stripe dispatcher |
| `grid_blocks_`, `active_blocks_`, `block_idx` | CTA counts/indices | scheduler-owned `G=max(Q,CU)` launch |
| `output_tile_idx`, `lock_idx` | global output-tile `q` | FP32 workspace tile and one lock per q |
| `M_idx`, `N_idx`, `L_idx` | output-tile coordinates | kernel problem coordinate only |
| reduction workspace offset | FP32 accumulator elements | `q * TM * TN`, converted to bytes only by the typed pointer |

No field crosses a logical-code/byte boundary.  The correctness gate must
still bind these host/device integer fields to the instantiated production
code; an algebraically correct parallel model alone would not detect a unit
change in generated code.
