# Q4_K/A32 prepare/consume case

Use this case as a pattern for mixed-input register lifetime failures.  The
authoritative repository record is
`dev/fold_derivation/Q4_A32_FOLDED_READER_DEBUG.md`.

## Signature

```text
shape=64x1024x5120
symbol=sf_q12_a32_tm64_tn64_tk128_wm16_wn32_s8_bc0
raw_bad=61184/65536
first want=0xc200 (-3), got=0xcc80 (-18)
```

## Decisive layout result

```text
A fragment MMA_K=8
A copy CPY_K=1, stride=0, whole-fragment load=64 values
B copy CPY_K=4, two MMA-K atoms per delivery
d3 -> next d0 overlap: A=16 values, B=0 values
```

The historical code reused B's `k_block` to index A.  B coordinates 1, 2 and
3 were outside A's one-element logical mode and physically aliased A0.  The
next-tile A0 prepare overwrote current-tile A atoms 6 and 7 before delivery 3
consumed them.

## Correct repair

Map A and B copy blocks through the common eight MMA-K atoms.  Load each A
block before its first consumer.  For the 1-A-block/4-B-delivery case, retain
B prepare-ahead but delay only the wrap A reload until after delivery 3.

The rejected broad workaround moved all prepare work after consume.  It made
the row pass but unnecessarily moved B conversion and look-ahead.

## Proof assets

- `dev/fold_derivation/l220_q4_a32_prepare_consume_layout.cu`
- `ci/check_mixed_a_register_schedule.py`
- `detail/ppu_mixed_a_schedule.hpp`
- `tools/run_scalefirst_q4_a32_exact_box.sh`

The exact negative is `PPU_MIXED_LEGACY_B_INDEXED_A_COPY=1`.
