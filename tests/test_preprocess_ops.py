"""THE HOST HALF, TESTED AGAINST ORACLES THAT DO NOT CALL IT.

This is the first tier of the extension that can run anywhere torch is installed -- no PPU, no hgcc. It covers the
preprocessing ops, which own the PHYSICAL LAYOUT of a quantised weight. That ownership is why they get tested first
and hardest: a kernel that reads a correct layout wrongly fails loudly on the box, while a layout produced wrongly
here fails as wrong numbers much later, with nothing in the tensor's dtype or shape to say which physical format it
is actually in.

WHAT COUNTS AS AN ORACLE HERE, AND WHAT DOES NOT. Three strengths appear below, and each test says which it has,
because an earlier version of this docstring claimed all of them were the strongest and that was not true:

  (a) INDEPENDENT -- a different implementation. torch's own transpose; a torch reimplementation of the quantiser;
      hand-computed byte literals. These can catch a shared misconception.
  (b) INVARIANT -- a structural property that holds whatever the layout is: byte count preserved, value multiset
      preserved, the transform injective. Blind to which arrangement was chosen, sharp about corruption.
  (c) MUTUAL -- an inverse pair, or two ops sharing a convention. pack/unpack round trips, and any test that
      unpacks the output with the production unpacker, are this. A convention wrong in BOTH -- reversed nibble
      order, say -- survives every one of them.

The (c) checks are still worth having, and they are the majority: they catch corruption, shape errors and
regressions. But they cannot establish the packed-byte convention, so test_packed_byte_convention_is_pinned_by_hand
does that with literals worked out by hand, and the rest are honest about resting on it.

NOTHING HERE ESTABLISHES THE PHYSICAL LAYOUT THE DEVICE WANTS. The layout transforms are checked for
self-consistency and memory safety only; the oracle for "is this what the AIU reads" is a kernel on the box.

    pytest tests/test_preprocess_ops.py -v
"""
import glob
import os

import pytest

# importorskip, not a plain import: this file is collected on machines with no torch (the local CI tier runs the whole
# tests/ directory, and tests/test_formats.py deliberately needs neither torch nor a device). A plain import would
# turn "torch is not installed here" into a collection ERROR for the whole directory.
torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module", autouse=True)
def _load():
    """Load the op library. It has no PyInit_ -- it is an operator-only .so, which is the shape torch documents for
    custom ops with no python-visible module. Importing it as a python module fails with a confusing
    'does not define module export function'; this is the supported path."""
    so = glob.glob(os.path.join(ROOT, "quactlize", "_C*.so"))
    if not so:
        pytest.skip("quactlize/_C*.so not built -- run `python setup.py build_ext --inplace`")
    torch.ops.load_library(so[0])


Q = torch.ops.quactlize
INT4, INT8 = torch.quint4x2, torch.int8


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """int4 nibbles out of a packed int8 tensor, low nibble first, as unsigned 0..15.

    Written here rather than calling unpack_int4_packed_tensor_to_int8 on purpose: several checks below are about
    whether that op is right, and an oracle that calls it could not tell."""
    b = packed.flatten().to(torch.int32) & 0xFF
    return torch.stack([b & 0xF, (b >> 4) & 0xF], dim=1).flatten()


# ---------------------------------------------------------------------------------------------------------------
# pack / unpack
# ---------------------------------------------------------------------------------------------------------------

def test_packed_byte_convention_is_pinned_by_hand():
    """INDEPENDENT. The one check here that does not rest on the convention it is testing.

    Values -8..7 in order, so the nibbles are 0x8,0x9,0xA,...,0xF,0x0,0x1,...,0x7. Two int4 go per byte, LOW NIBBLE
    FIRST: byte i is (nibble[2i+1] << 4) | nibble[2i]. Byte 0 is therefore (0x9 << 4) | 0x8 = 0x98, and byte 4 --
    where the values cross zero -- is (0x1 << 4) | 0x0 = 0x10. Worked out by hand from that sentence rather than
    produced by running anything.

    Without this, every other test in this section is mutual: pack and unpack could agree on the REVERSED order and
    round-trip perfectly, the transpose oracle packs both sides with the same packer, and the quantiser comparison
    decodes with the same unpacker. A convention wrong in all of them is invisible to all of them."""
    w = torch.arange(-8, 8, dtype=torch.int8).reshape(1, 16)
    got = Q.pack_int8_tensor_to_packed_int4(w).flatten().tolist()
    want = [0x98, 0xBA, 0xDC, 0xFE, 0x10, 0x32, 0x54, 0x76]
    assert got == [b if b < 128 else b - 256 for b in want], f"packed bytes {[x & 0xFF for x in got]!r}"


def test_unpack_sign_extends_independently_of_the_packer():
    """INDEPENDENT of the packer: the input bytes are literals, and the expected values are read off the nibbles by
    hand. A high nibble of 0x8 is -8, not +8; an unpacker that forgot the sign extension would pass every round-trip
    test in this file and fail this one."""
    packed = torch.tensor([[0x80 - 256, 0x0F, 0x71]], dtype=torch.int8)
    got = Q.unpack_int4_packed_tensor_to_int8(packed).flatten().tolist()
    assert got == [0, -8, -1, 0, 1, 7]



@pytest.mark.parametrize("shape", [(64, 128), (2, 64, 128), (256, 256), (1, 32, 64)])
def test_int4_pack_unpack_is_exact_inverse(shape):
    """MUTUAL: an inverse pair. Catches corruption and shape errors; cannot see a convention wrong in both halves --
    that is what test_packed_byte_convention_is_pinned_by_hand is for."""
    w = torch.randint(-8, 8, shape, dtype=torch.int8)
    p = Q.pack_int8_tensor_to_packed_int4(w)
    assert p.shape[:-1] == w.shape[:-1] and p.shape[-1] == w.shape[-1] // 2
    assert torch.equal(Q.unpack_int4_packed_tensor_to_int8(p), w)


@pytest.mark.parametrize("shape", [(64, 128), (2, 64, 128)])
def test_uint4_pack_unpack_is_exact_inverse(shape):
    w = torch.randint(0, 16, shape, dtype=torch.int8)
    p = Q.pack_uint8_tensor_to_packed_uint4(w)
    assert torch.equal(Q.unpack_uint4_packed_tensor_to_uint8(p), w)


def test_int4_pack_covers_the_whole_code_range():
    """Every one of the 16 codes survives a round trip. A pack that dropped the sign bit, or saturated, would pass a
    random test at some rate and fail this one deterministically."""
    w = torch.arange(-8, 8, dtype=torch.int8).repeat(4, 4)
    assert torch.equal(Q.unpack_int4_packed_tensor_to_int8(Q.pack_int8_tensor_to_packed_int4(w)), w)


# ---------------------------------------------------------------------------------------------------------------
# subbyte_transpose: torch's own transpose is a genuinely independent oracle
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(64, 128), (128, 64), (2, 64, 128), (256, 256)])
def test_subbyte_transpose_matches_torch_transpose(shape):
    """INDEPENDENT for the TRANSPOSE, mutual for the packing. torch's transpose owes nothing to subbyte_transpose,
    so a wrong transpose is caught. But both sides are packed by the production packer, so a wrong packing convention
    cancels and is invisible here; test_packed_byte_convention_is_pinned_by_hand carries that.

    Compared FLATTENED, because the op's returned shape does not describe its contents -- see the next test."""
    w = torch.randint(-8, 8, shape, dtype=torch.int8)
    packed = Q.pack_int8_tensor_to_packed_int4(w)
    got = Q._subbyte_transpose(packed, INT4)
    want = Q.pack_int8_tensor_to_packed_int4(w.transpose(-2, -1).contiguous())
    assert torch.equal(got.flatten(), want.flatten()), \
        "packed int4 transpose disagrees with torch's transpose of the unpacked tensor"


def test_subbyte_transpose_returns_the_input_shape_not_the_transposed_one():
    """RECORDED, NOT ENDORSED. The op allocates with empty_like(input), so a (64,64) packed tensor holding a 64x128
    logical matrix comes back labelled (64,64) while its bytes are the 128x64 transpose, whose packed shape is
    (128,32). The shape is not merely stale -- it is inconsistent with the contents, and only square inputs hide it.

    This is pinned as a test because it is exactly the failure mode this file exists to catch: nothing in the tensor
    says which physical format it is in. Any caller must carry the logical shape itself. If the op is ever fixed to
    return the true shape, this test fails and points at every caller that has to change with it."""
    w = torch.randint(-8, 8, (64, 128), dtype=torch.int8)
    packed = Q.pack_int8_tensor_to_packed_int4(w)                       # (64, 64)
    got = Q._subbyte_transpose(packed, INT4)
    truly = Q.pack_int8_tensor_to_packed_int4(w.t().contiguous())       # (128, 32)
    assert tuple(got.shape) == tuple(packed.shape) != tuple(truly.shape)
    assert torch.equal(got.flatten(), truly.flatten())                  # the BYTES are right


def test_subbyte_transpose_is_an_involution():
    w = torch.randint(-8, 8, (128, 128), dtype=torch.int8)
    p = Q.pack_int8_tensor_to_packed_int4(w)
    assert torch.equal(Q._subbyte_transpose(Q._subbyte_transpose(p, INT4), INT4), p)


# ---------------------------------------------------------------------------------------------------------------
# preprocess_weights_for_mixed_gemm: the storage constraint, and the bias it composes with
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(64, 128), (2, 64, 128), (512, 256)])
def test_preprocess_does_not_grow_the_weight(shape):
    """THE CONSTRAINT THE WHOLE FORMAT PLAN RESTS ON: an offline reorder may permute bytes but must not add any. Every
    format's readiness was judged against this, so it is asserted rather than assumed."""
    p = Q.pack_int8_tensor_to_packed_int4(torch.randint(-8, 8, shape, dtype=torch.int8))
    out = Q._preprocess_weights_for_mixed_gemm(p, INT4, False, False)
    assert out.shape == p.shape and out.dtype == p.dtype
    assert out.numel() * out.element_size() == p.numel() * p.element_size()


@pytest.mark.parametrize("shape", [(64, 128), (2, 64, 128), (512, 256)])
def test_preprocess_is_a_permutation_composed_with_the_plus_8_bias(shape):
    """The layout transform moves nibbles; add_bias_and_interleave then maps signed -8..7 to unsigned 0..15. So the
    multiset of OUTPUT nibbles must equal the multiset of (input + 8). A transform that dropped, duplicated or
    corrupted a nibble breaks this no matter which positions it chose -- which is the point: this holds without
    knowing the layout, so it stays valid when the layout changes."""
    w = torch.randint(-8, 8, shape, dtype=torch.int8)
    p = Q.pack_int8_tensor_to_packed_int4(w)
    out = Q._preprocess_weights_for_mixed_gemm(p, INT4, False, False)
    got = torch.sort(unpack_nibbles(out).to(torch.int64)).values
    want = torch.sort(((w.flatten().to(torch.int64) + 8) & 0xF)).values
    assert torch.equal(got, want)


def test_preprocess_is_injective():
    """INVARIANT. A permutation is injective for free; a transform that overwrote part of its output -- an index
    formula that collides -- would not be, and would still pass the multiset check whenever the collision happened
    to duplicate an equal value.

    Every position is perturbed in turn along two diagonals rather than one hand-picked coordinate: a collision
    affects specific indices, and testing one index tests one index. Diagonals because they cross every row and
    every column of the tile structure the transform is built around."""
    a = torch.randint(-7, 7, (256, 256), dtype=torch.int8)
    pa = Q.pack_int8_tensor_to_packed_int4(a)
    ta = Q._preprocess_weights_for_mixed_gemm(pa, INT4, False, False)
    for i in range(0, 256, 8):
        for r, c in ((i, i), (i, 255 - i)):
            b = a.clone()
            b[r, c] = a[r, c] + 1                       # in range: a was drawn from [-7, 7)
            tb = Q._preprocess_weights_for_mixed_gemm(Q.pack_int8_tensor_to_packed_int4(b), INT4, False, False)
            assert not torch.equal(ta, tb), f"changing element ({r},{c}) vanished in preprocessing"


def test_int8_preprocess_shifts_by_128():
    w = torch.randint(-128, 127, (2, 64, 128), dtype=torch.int8)
    out = Q._preprocess_weights_for_mixed_gemm(w, INT8, False, False)
    got = torch.sort(out.flatten().to(torch.int64) & 0xFF).values
    want = torch.sort((w.flatten().to(torch.int64) + 128) & 0xFF).values
    assert torch.equal(got, want)


def test_is_int8_mma_skips_the_bias():
    """is_int8_mma=true is documented to leave the values alone and only move them. So the output multiset must equal
    the INPUT's, with no +8 -- the same invariant with the bias removed, which is how we know the flag is read."""
    w = torch.randint(-8, 8, (256, 256), dtype=torch.int8)
    p = Q.pack_int8_tensor_to_packed_int4(w)
    out = Q._preprocess_weights_for_mixed_gemm(p, INT4, True, False)
    got = torch.sort(unpack_nibbles(out).to(torch.int64)).values
    want = torch.sort((w.flatten().to(torch.int64) & 0xF)).values
    assert torch.equal(got, want)


# ---------------------------------------------------------------------------------------------------------------
# the AIU column interleave -- the PPU-specific step
# ---------------------------------------------------------------------------------------------------------------

def aiu_permutation(num_rows, num_cols, elts_in_int32=8, rows_per_tile=256):
    """Re-derivation of the AIU write map in python: which input uint32 lands at each output uint32.

    THIS IS A RE-DERIVATION, NOT AN INDEPENDENT IMPLEMENTATION -- the index arithmetic was read off
    interleave_column_major_tensor_aiu. It catches a transcription error, a wrong compile-time constant or a build
    that silently skipped the step; it cannot catch a shared misconception about what the AIU wants. The oracle that
    can is the kernel on the box, and that is where this layout is ultimately gated."""
    nvr, vrpt = num_rows // elts_in_int32, rows_per_tile // elts_in_int32
    src = torch.empty(nvr * num_cols, dtype=torch.int64)
    for col in range(num_cols):
        for vr in range(nvr):
            src[(vr // vrpt) * vrpt * num_cols + col * vrpt + (vr % vrpt)] = col * nvr + vr
    return src


@pytest.mark.parametrize("k,n", [(512, 256), (512, 512), (1024, 256)])
def test_aiu_interleave_matches_the_derived_index_map(k, n):
    w = torch.randint(-8, 8, (k, n), dtype=torch.int8)
    p = Q.pack_int8_tensor_to_packed_int4(w)
    plain = Q._preprocess_weights_for_mixed_gemm(p, INT4, True, False)   # is_int8_mma=True: no bias, layout only
    aiu = Q._preprocess_weights_for_mixed_gemm(p, INT4, True, True)
    src = aiu_permutation(k, n)
    a32 = aiu.flatten().view(torch.int32)
    p32 = plain.flatten().view(torch.int32)
    assert torch.equal(a32, p32[src]), "AIU interleave disagrees with the derived write map"


def test_aiu_interleave_is_the_identity_at_exactly_one_row_tile():
    """k=256 is ONE row tile, so num_tile is always 0 and the write offset collapses to the read offset. Recorded as a
    test because a k=256 shape was the first thing tried and its 'AIU changes nothing' result was nearly read as the
    flag being dead. A degenerate shape is a measurement of nothing; this pins which shape is degenerate and why."""
    p = Q.pack_int8_tensor_to_packed_int4(torch.randint(-8, 8, (256, 256), dtype=torch.int8))
    assert torch.equal(Q._preprocess_weights_for_mixed_gemm(p, INT4, True, True),
                       Q._preprocess_weights_for_mixed_gemm(p, INT4, True, False))


@pytest.mark.parametrize("k,n,why", [
    (64, 128, "k and n below the 256 column tile"),
    (512, 128, "n not a multiple of 256"),
    (128, 256, "k not a multiple of 256"),
])
def test_aiu_refuses_shapes_it_would_silently_skip(k, n, why):
    """The downstream branch skips the interleave for these shapes and returns the ORDINARY layout. A caller that
    asked for the AIU layout would get bytes in the wrong physical order with nothing to signal it. The op refuses
    instead. This is a regression test for a real observation: the first AIU call returned bytes identical to the
    non-AIU call, and both the missing USE_AIU define and this shape guard were responsible at once."""
    p = Q.pack_int8_tensor_to_packed_int4(torch.randint(-8, 8, (k, n), dtype=torch.int8))
    with pytest.raises(RuntimeError, match="divisible by 256"):
        Q._preprocess_weights_for_mixed_gemm(p, INT4, True, True)


# ---------------------------------------------------------------------------------------------------------------
# symmetric_quantize: a pure-torch quantiser is the oracle
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("quant,bits", [(INT4, 4), (INT8, 8)])
def test_symmetric_quantize_scale_matches_a_pure_torch_quantiser(quant, bits):
    """Scale is per output column: max|w| over the K axis divided by 2**(bits-1) -- 8 for int4, 128 for int8, NOT 7
    and 127. The full power-of-two range is the convention here; the top code is then unreachable except by clipping.
    Computed below in torch with no call into the extension. (An oracle written with 7 and 127 fails by 12.5% and
    0.78% respectively, which is how the constant was pinned rather than assumed.)"""
    w = torch.randn(128, 64, dtype=torch.float16) * 3.0
    _, _, scale = Q._symmetric_quantize_last_axis_of_batched_matrix(w, quant, 80)
    want = (w.abs().amax(dim=0).to(torch.float32) / float(1 << (bits - 1))).to(torch.float16)
    torch.testing.assert_close(scale.flatten(), want.flatten(), rtol=1e-3, atol=1e-8)


def test_symmetric_quantize_codes_match_an_independent_reimplementation():
    """FULL REIMPLEMENTATION AS THE ORACLE: round-to-nearest of w/scale, clamped to the int4 range, in torch.

    One subtlety this pins down, and the reason it is worth an exact comparison rather than an error bound: the codes
    are produced with the FP32 column max, while the scale that gets STORED is that value cast to fp16. The two are
    not the same number, so dequantising with the stored scale is not exactly the inverse of quantising. Anything
    downstream that reconstructs weights must use the stored scale and accept that, or it will chase a discrepancy
    that lives here."""
    torch.manual_seed(0)
    w = torch.randn(128, 64, dtype=torch.float16) * 3.0
    packed_unproc, _, stored_scale = Q._symmetric_quantize_last_axis_of_batched_matrix(w, INT4, 80)
    # Decoded with the pure-python nibble reader, not the production unpacker: the quantiser and the unpacker share
    # the low-nibble-first convention, so using the unpacker here would let a wrong convention cancel itself out.
    nib = unpack_nibbles(packed_unproc).to(torch.int64)
    got = torch.where(nib >= 8, nib - 16, nib).reshape(w.shape)          # sign-extend 4 bits, written out here

    scale_f32 = w.abs().amax(dim=0).to(torch.float32) / 8.0
    # C's round() is HALF AWAY FROM ZERO; torch.round is half to even. With torch.round, 3 of 8192 codes differ --
    # exactly the ties. Small, but it is a systematic disagreement at every .5, so it is reproduced rather than
    # tolerated: an importer that reconstructs codes with torch.round produces a weight that differs from the one
    # this quantiser wrote, and the difference would look like a kernel bug.
    x = w.to(torch.float32) / scale_f32
    want = torch.clamp(torch.sign(x) * torch.floor(x.abs() + 0.5), -8, 7).to(torch.int64)
    assert torch.equal(got, want), f"{int((got != want).sum())} of {got.numel()} codes differ from the reimplementation"
    torch.testing.assert_close(stored_scale.to(torch.float32), scale_f32.to(torch.float16).to(torch.float32),
                               rtol=0, atol=0)


def test_symmetric_quantize_round_trip_error_is_bounded():
    """Dequantising with the stored scale must land within one and a half steps: half a step from rounding, plus up to
    one more for the single element per column that hits the clamp -- w/scale reaches 8 there and 7 is the largest
    code. A bound of half a step alone would fail on exactly that element, which is a property of the 2**(bits-1)
    convention and not a bug."""
    w = torch.randn(128, 64, dtype=torch.float16) * 3.0
    packed_unproc, _, scale = Q._symmetric_quantize_last_axis_of_batched_matrix(w, INT4, 80)
    codes = Q.unpack_int4_packed_tensor_to_int8(packed_unproc).to(torch.float32)
    s = scale.to(torch.float32).unsqueeze(0)
    err = (w.to(torch.float32) - codes * s).abs()
    assert torch.all(err <= 1.5 * s + 1e-3), f"max error {float((err / s).max()):.3f} steps"
    assert int(codes.min()) >= -8 and int(codes.max()) <= 7
    # Away from the clamp, half a step is the real bound. "Away from the clamp" is NOT "not the column max": scale is
    # max/8, so everything with |w| >= 7.5/8 of the max also rounds to 8 and clips. The mask has to be computed from
    # the rounded value, not from the magnitude.
    unclipped = (w.to(torch.float32) / s).abs() <= 7.5
    assert torch.all(err[unclipped] <= (0.5 * s.expand_as(err))[unclipped] + 1e-3)


# ---------------------------------------------------------------------------------------------------------------
# argument guards
# ---------------------------------------------------------------------------------------------------------------

def test_non_contiguous_input_is_refused():
    """These ops walk the buffer directly, so a non-contiguous tensor would read the wrong bytes rather than fail."""
    w = torch.randint(-8, 8, (128, 256), dtype=torch.int8).t()
    assert not w.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        Q.pack_int8_tensor_to_packed_int4(w)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a device to make a device tensor")
def test_device_tensor_is_refused():
    w = torch.randint(-8, 8, (128, 256), dtype=torch.int8, device="cuda")
    with pytest.raises(RuntimeError, match="CPU"):
        Q.pack_int8_tensor_to_packed_int4(w)
