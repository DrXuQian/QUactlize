# Q3_K / Q6_K zero-plane redundancy

## Result and scope

The external fp16 zero planes emitted by the Q3_K/Q6_K scale-first and placed
dense producers contain no independent information. This is a structural
consequence of the production formulas, not an extrapolation from random
samples:

| external-plane producer | Q3_K formula | Q6_K formula |
|---|---|---|
| scale-first GEMV | `half(-4 * S)` | `half(-32 * S)` |
| placed dense | `half(-4 * S)` | `half(float(half(-32*S)) + 8*float(S))` |

The formulas follow directly from the producer composition: Q3/Q6 first shift
signed codes to offset binary and subtract `4*S` / `32*S`; the int4 dense
converter then requires the existing `+8*S` adjustment for Q6. There is no
other information-bearing input to either external zero plane (the native
scale-only affine term has fixed zero magnitude). A deterministic 8,192-element
run in **each of the three arms, per format**, is retained as a bitwise witness
of that structural statement. Each arm's count is reported and checked
separately, so an empty dense or short packed output cannot turn into a vacuous
zero-mismatch result. The run is explicitly **not** an exhaustive enumeration
of all fp16 scales.

The fully-quantized path is different: its producer returns only
`(low, high, packed_unit)`. It already stores **no external fp16 zero plane**.
`gguf_unit_decode(units, qtype, correction)` is only a diagnostic: after the
caller supplies `-4` (Q3) or `-24` (Q6), its logical zero is reconstructed as
`half(+0 + correction*S)`. This gate does **not** prove that the production
two-plane collective selected the matching `kPackedZMul`; that constant lives
in the collective and is outside this producer/decoder proof.

The Q6 dense expression intentionally has two fp16 rounding stages. Replacing
it by algebraically equivalent `half(-24*S)` can change bits for legal producer
outputs. The overflow-boundary witness used to force that distinction is held
in a separate targeted fixture; it is not mixed into the main bitwise witness
or the official-GGUF anchor.

This proves that the two external zero planes *can in principle* be omitted.
It does **not** remove either plane, alter an ABI, change a kernel, or claim any
measured traffic saving. The packed route needs no such removal because it
already has no external fp16 zero plane.

## Constructive witness

`q36_zero_redundancy.py` synthesizes valid raw blocks, then calls the actual
torch producers:

1. `gguf_prepare_gemv` and reads its returned `(scale, zero)` planes;
2. `gguf_prepare_dense` and reads its returned placed-dense affine planes;
3. `gguf_prepare_fully_quantized_dense`, verifies its three-tensor return has
   no zero plane, then asks `gguf_unit_decode` for one explicitly selected
   logical correction.

For the two external-plane arms, the oracle uses only the scale returned by
that arm. It rebuilds zero with the formula in the table and compares fp16
storage as `uint16`, not with a floating-point tolerance. The packed diagnostic
similarly checks its caller-selected logical correction, but is not evidence
about the collective's independently compiled correction constant.

The gate includes controls which must be red:

- the other format's bias (`-32` for Q3, `-4` for Q6);
- a one-ULP perturbation of the packed diagnostic's reconstructed zero;
- the one-stage Q6 dense `half(-24*S)` expression versus the required
  two-stage expression, using the separate targeted fixture.

The targeted fixture reports both sides of that last control: the actual dense
producer must have zero mismatches against the staged formula, while the staged
formula must differ from the tempting one-stage formula in exactly one place.
The runner also refuses Python `-O` / `PYTHONOPTIMIZE`; disabling assertions
cannot turn these structural checks into an empty green.

The one-ULP control applies only to the packed diagnostic. It must not be read
as coverage of either external-plane arm. Together the controls prevent “all
zeros happen to compare equal” and “algebraically equivalent means bitwise
equivalent” from making their respective checks vacuous.

The local witness used 8,192 affine elements per format and reported:

| format | scale-first / dense / packed-diagnostic mismatches | wrong-bias witnesses | packed one-ULP witnesses | targeted dense one-stage witnesses |
|---|---:|---:|---:|---:|
| Q3_K | `0 / 0 / 0` | 8,080 | 1 | n/a |
| Q6_K | `0 / 0 / 0` | 8,153 | 1 | 1 |

These deterministic counts are witnesses, not a claim that 8,192 sampled
elements exhaust the input domain. The Q6 rounding control deliberately uses
one legal, finite GGUF scale at the fp16 overflow boundary so the two-stage and
one-stage expressions cannot agree merely because ordinary model scales are
small. It is a separate gate fixture, not a claim about the distribution of
checkpoint scales. Re-run output is authoritative if the witness counts change
while preserving the required nonzero controls.

## Independent GGUF anchor

The installed official `gguf` package gives these complete block sizes:

- Q3_K: `32 B hmask + 64 B qs + 12 B scales + 2 B d = 110 B`;
- Q6_K: `128 B ql + 64 B qh + 16 B scales + 2 B d = 210 B`.

There is no `dmin`, min, or zero field. The oracle also compares quactlize's
signed-code `code * scale` reconstruction with
`gguf.quants.dequantize`; this anchors the assertion to an implementation that
does not share quactlize's affine producer constants. Relative error is
normalised **per GGUF block** before taking the maximum, so a high-amplitude
block cannot hide an unrelated block's error behind one global denominator.

## Run

The pytest gate builds the repository's local CUDA stand-in because the placed
dense and fully-quantized producers cross the device-library ABI.  Its entry
points are host-only; nvcc compiles dormant device bodies for `sm_80`, and no
runtime GPU or PPU is required:

```bash
python3 -m pytest -q tests/test_q36_zero_redundancy.py
```

For an already built compatible library, the underlying oracle is directly
re-runnable:

```bash
QUACTLIZE_PPU_LIB=/path/to/libquactlize_ppu.so \
  python3 dev/fold_derivation/q36_zero_redundancy.py
```
