# Q4 BC whole-word reader: local validation contract

The reader consumes the resident xplane artifact shared by prefill and decode. It does not introduce a raw-GGUF
decode copy and does not change `place_derived`, its permutation π, or any artifact byte.

Run:

```bash
python3 ci/check_bc_q4_fast_reader.py
```

Artifacts are written under `/workspace/quactlize-l187-bc-q4-fast-reader`.

## Authorities and exhaustive domain

`l187_bc_q4_fast_reader.cu` evaluates all `512 × 512` logical coordinates for each production-supported Q4
`ArtifactTileK`:

| ArtifactTileK | fold | coordinates | writer/recover | scalar address vs word plan | scalar value vs word plan |
|---:|---:|---:|---:|---:|---:|
| 32 | 2 | 262,144 | 0 bad | 0 bad | 0 bad |
| 64 | 1 | 262,144 | 0 bad | 0 bad | 0 bad |
| 128 | 1 | 262,144 | 0 bad | 0 bad | 0 bad |
| 256 | 1 | 262,144 | 0 bad | 0 bad | 0 bad |

The three authorities are deliberately different:

1. `xplane::place_derived` writes the bytes and `recover_derived` independently recovers the logical codes.
2. `gguf_bc_vecdot.hpp::xplane_physical_code` is the scalar `code_at` address oracle.
3. The shipping `Q4WordPlan` plus the closed-form `q4_group_byte_offset` names the four-word address and P4x32
   register permutation used by the fast consumer. The closed form is independently compared with the scalar
   fourteen-position inverse at every coordinate; the fast path does not call that inverse.

The same exhaustive loop also requires every group base to satisfy `group_byte % alignof(uint32_t) == 0`.
This is a separate shipping precondition: nibble closure alone would not justify the consumer's aligned word load.

The public device-pointer ABI has a stronger boundary condition for Q4: `x`, `low`, and `units` must each be
16-byte aligned because the measured topology issues `float4`/`uint4` global loads.  L187 exercises the shared
production predicate with one aligned tuple and three independently misaligned inputs.  Both public BC device
entries consume that predicate and return 25 before enqueue; the source gate removes one entry's check as a
negative control and requires the resulting half-wired ABI to fail.

The comparison count is **1,048,576**, not a representative sample. `Q4WordPlan::code_from_pair_lane` also binds
the production pair/lane ownership to the scalar nibble value. The sm120 device probe instantiates actual
`code_at` and `dequantize_word`; its SASS must contain balanced low/high `LOP3.LUT` and `HFMA2` arithmetic levels
and no `ppu.*` mnemonic.

The versionless C++ reader retains its historical default `ArtifactTileK=256`. That is distinct from the current
Python descriptor-producing Q4 path, whose shipping default is A64. Both are covered; neither default is inferred
from the other.

## Preregistered negative controls

* `wrong-permutation` XORs exactly one within-word position bit. It remains a bijection and stays in bounds, but
  must disagree with the scalar address oracle at every checked coordinate.
* `missing-denominator` omits A256 after deriving the expected count from `arrangement_supported_v`. It must report
  `3/4` and fail; coverage cannot be improved by shrinking a handwritten denominator.
* The ABI seam plant removes the Q4 alignment check from exactly one of the two public device entries.  It must fail
  even though the other entry and the shared predicate remain intact.

Both controls must return nonzero and print `PLANTED_RED ... DETECTED`. A control that exits successfully is itself
a gate failure.

## Scope

This establishes address, byte-map, pair ownership, target-dialect selection, and compile-time arithmetic shape.
It does not claim a latency improvement or a PPU instruction count. Those require the shipping benchmark and ACU.
It covers Q4 only; no conclusion is projected onto Q2/Q3/Q5/Q6.

The separate same-binary RTX 5090 performance and production-routing evidence is recorded in
`BC_Q4_GEMV_5090_AB.md`.  Keeping it separate is deliberate: the exhaustive local oracle proves the byte and
permutation contract, while the benchmark proves latency, resource use, public dispatch, and positive/signed
accuracy.  Neither is allowed to stand in for the other.
