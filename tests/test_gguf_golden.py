"""THE SCALE CHANNEL AGAINST llama.cpp's OWN REFERENCE, not against ourselves.

WHY THIS FILE EXISTS AND WHY THE EXISTING CHECKS DO NOT REPLACE IT. Every k-quant constant in
quactlize/include/gguf_scale_layout.hpp was read off dev/.../dump_real_weights.py, which is OUR hand-written numpy
parser -- its own header says "no gguf libs needed". So a test that compares the C++ Traits against that script
compares two transcriptions of one belief, and would agree just as happily if the belief were wrong. Q3_K's
kScaleBias = 32 is exactly the kind of constant that has no other witness.

The official `gguf` package IS the llama.cpp reference: gguf.quants.dequantize covers Q2_K..Q6_K and is an
independent implementation. Everything here is anchored on it.

HOW THE SCALE AND ZERO ARE EXTRACTED WITHOUT REIMPLEMENTING ANY WEIGHT UNPACKING. Dequantisation is AFFINE in the
quantised code:

    W(q) = q * scale + intercept

so two byte-fills of a block's code region recover both, per group, using only the official implementation:

    fill 0x00 -> W(0)      = intercept
    fill 0xFF -> W(qmax)   = qmax * scale + intercept

That matters because Q3_K, Q5_K and Q6_K split their codes across bit planes, and hand-unpacking them here would put
a third transcription of the format into the tree -- another thing to be wrong, and wrong in a way that could cancel.

WHAT OUR SIDE MUST PRODUCE. The consumer's converter emits (q - ZMul), so the planes we hand it satisfy

    W(q) = (q - ZMul) * scale + zero        =>   zero = intercept + ZMul * scale

and that is the whole reason ZMul is not optional: a prepass producing only the format's -dmin*mn term would be off
by ZMul*scale everywhere, which looks like plausible weights and is silently wrong.

A DEGENERATE FIXTURE IS THE FAILURE MODE THIS FILE GUARDS AGAINST FIRST. The first version of this check filled all
144 bytes at random, which makes d and dmin NaN for most blocks; np.abs(nan) > tol is False, so every block "passed"
and the reported mismatch count was zero. Hence assert_finite below, run on the GOLDEN, before any comparison.
"""
import numpy as np
import pytest

# NOT importorskip. This module IS the independent oracle for every k-quant constant in the tree, and a skip is
# indistinguishable from a pass in a CI summary line -- the whole file would vanish and the run would still say
# "all passed". If the golden is missing the honest outcome is a failure that says to install it.
try:
    import gguf
    import gguf.quants  # noqa: F401
except ImportError as _e:                                    # pragma: no cover
    raise RuntimeError(
        "the official llama.cpp `gguf` package is the golden for these tests and is not importable "
        f"({_e}). Install it (pip install gguf) -- skipping would report green while checking nothing.") from _e
from gguf.constants import GGMLQuantizationType as GT     # noqa: E402
from gguf.constants import GGML_QUANT_SIZES               # noqa: E402

# THE BLOCK LAYOUT, READ OFF gguf.quants.<T>.dequantize_blocks RATHER THAN WRITTEN DOWN. The first version of this
# table assumed "header first, codes in the tail" and was wrong for three of the five formats -- Q2_K, Q3_K and Q6_K
# put d (and dmin) at the END -- so filling the "code" tail overwrote the scale factors and the affinity premise
# failed for exactly those three. The failure was in the assumption, not the formats, which is the same mistake this
# repository keeps paying for: a relation written down instead of read off the object.
#
#   Q2_K (84)   scales[16] | qs[64]                     | d[2] | dmin[2]
#   Q3_K (110)  hmask[32]  | qs[64]                     | scales[12] | d[2]
#   Q4_K (144)  d[2] | dmin[2] | scales[12]             | qs[128]
#   Q5_K (176)  d[2] | dmin[2] | scales[12] | qh[32]    | qs[128]
#   Q6_K (210)  ql[128] | qh[64]                        | scales[16] | d[2]
#
# code_ranges lists EVERY byte range that carries quantised codes, including the separate high-bit planes of Q3_K,
# Q5_K and Q6_K -- a fill that missed one would leave part of the code fixed and understate the slope.
FORMATS = [
    #  name    ggml type   header (d, dmin)      scales        code ranges                 qmax
    ("Q2_K", GT.Q2_K, [(80, 82), (82, 84)], (0, 16),   [(16, 80)],                 3),
    ("Q3_K", GT.Q3_K, [(108, 110)],         (96, 108), [(0, 32), (32, 96)],        7),
    ("Q4_K", GT.Q4_K, [(0, 2), (2, 4)],     (4, 16),   [(16, 144)],                15),
    ("Q5_K", GT.Q5_K, [(0, 2), (2, 4)],     (4, 16),   [(16, 48), (48, 176)],      31),
    ("Q6_K", GT.Q6_K, [(208, 210)],         (192, 208), [(0, 128), (128, 192)],    63),
]


def _assert_finite(a, what):
    assert np.isfinite(a).all(), f"{what} contains non-finite values -- the fixture is degenerate and any " \
                                 f"comparison against it passes vacuously"


def _make_blocks(qtype, hdr_ranges, scales_range, code_ranges, n, rng, code_fill=None):
    """Blocks with normal fp16 headers, random scale metadata, and codes either random or a constant fill."""
    _, type_size = GGML_QUANT_SIZES[qtype]
    raw = np.zeros((n, type_size), np.uint8)
    for lo, hi in hdr_ranges:                      # small, positive, normal -> the golden is finite by construction
        v = (rng.random(n) * 0.1 + 0.001).astype(np.float16)
        raw[:, lo:hi] = v.view(np.uint8).reshape(n, 2)
    lo, hi = scales_range
    raw[:, lo:hi] = rng.integers(0, 256, size=(n, hi - lo), dtype=np.uint8)
    for lo, hi in code_ranges:
        raw[:, lo:hi] = (rng.integers(0, 256, size=(n, hi - lo), dtype=np.uint8)
                         if code_fill is None else code_fill)
    return raw


@pytest.mark.parametrize("name,qtype,hdr,scales,codes,qmax", FORMATS)
def test_code_ranges_cover_every_plane(name, qtype, hdr, scales, codes, qmax):
    """Do the declared code ranges really cover ALL the code bits? This is the thing I can get wrong.

    WHY NOT TEST AFFINITY. The first two attempts did, with fills of 0x55 and 0xAA, and both failed -- on Q2_K/Q3_K/
    Q6_K because the header ranges were wrong, then on Q3_K/Q5_K for a deeper reason: their high-bit planes hold ONE
    BIT PER ELEMENT, so a fill of 0x55 gives bit 1 to even positions and 0 to odd, and the elements of a group end up
    with DIFFERENT codes. The ratio then varies per element and the check fails while nothing is wrong. 0x00 and 0xFF
    are the only byte values uniform in every bit, so a byte fill cannot produce a third code level at all, and
    affinity is not testable this way. It also does not need testing: every format's dequantize_blocks in
    gguf.quants is literally d*scale*q + (a per-group constant), i.e. affine by inspection of the reference itself.

    What IS worth testing is this: fill every declared code range with 0x00, and every element of a group must come
    out EQUAL, because they all carry code 0. If a code plane were missing from the table it would keep its random
    bytes, those elements would differ, and this fails -- which is exactly the mistake the two earlier versions of
    this file made about byte ranges, caught here instead of silently weakening the extraction below.
    """
    quactlize = pytest.importorskip("quactlize", reason="needs the built operator library")
    rng = np.random.default_rng(abs(hash(name)) & 0xFFFF)
    block_size, _ = GGML_QUANT_SIZES[qtype]
    # THE REAL GROUP SHAPE, from the C++ Traits. An earlier version hardcoded block_size // 16, which is right for
    # Q2_K/Q3_K/Q6_K and WRONG for Q4_K and Q5_K -- they have eight groups of 32, not sixteen of 16. It still passed,
    # because uniformity over a 32-element group implies it over each 16-element half, so the check was weaker than
    # it claimed rather than broken. Asking the op removes the guess.
    _blk, groups, gsz, _hm, _b, _s = quactlize.gguf_scale_block_shape(int(qtype))
    assert groups * gsz == block_size
    n = 32
    raw0 = _make_blocks(qtype, hdr, scales, codes, n, rng, code_fill=None)   # random everywhere first
    # BOTH FILLS. The scale extraction divides by (w_hi - w_lo), so a plane missing from the table breaks 0xFF just
    # as badly as 0x00 and only one of them was being checked.
    for fill in (0x00, 0xFF):
        raw = raw0.copy()
        for lo, hi in codes:
            raw[:, lo:hi] = fill
        w = gguf.quants.dequantize(raw.reshape(-1), qtype).reshape(n, block_size).astype(np.float64)
        _assert_finite(w, f"{name} golden at code fill 0x{fill:02X}")
        g = w.reshape(n, groups, gsz)
        spread = np.abs(g - g[:, :, :1]).max()
        assert spread < 1e-9 * max(1.0, np.abs(w).max()), \
            f"{name}: with every declared code range set to 0x{fill:02X}, a group is not constant " \
            f"(spread {spread:.3e}) -- the code_ranges table is missing a plane, so any scale extracted " \
            f"from a fill would be wrong"


def _our_q4k_sc_mn(s):
    """quactlize/include/gguf_scale_layout.hpp Traits<Q4_K>, transcribed. ScLo: byte t+8h shift 0 width 4;
    ScHi: byte t shift 4+2h width 2; MnLo: byte 4+t+4h shift 4h width 4; MnHi: byte 4+t shift 4+2h width 2."""
    out = []
    for g in range(8):
        t, h = g & 3, g >> 2
        sc = ((s[t + 8 * h] >> 0) & 0xF) | (((s[t] >> (4 + 2 * h)) & 0x3) << 4)
        mn = ((s[4 + t + 4 * h] >> (4 * h)) & 0xF) | (((s[4 + t] >> (4 + 2 * h)) & 0x3) << 4)
        out.append((sc, mn))
    return out


def test_q4k_field_layout_matches_llama_cpp():
    """The one format whose weight unpacking is simple enough to write out, checked end to end.

    This is the strongest single check available without the C++ binding: it validates the 6-bit scale and min field
    positions, the d/dmin header, and the affine formula together, against the reference implementation. If any field
    position were wrong the error would be large, not small.
    """
    rng = np.random.default_rng(0)
    n = 256
    blocks = np.zeros((n, 144), np.uint8)
    d = (rng.random(n) * 0.1 + 0.001).astype(np.float16)
    dmin = (rng.random(n) * 0.1 + 0.001).astype(np.float16)
    blocks[:, 0:2] = d.view(np.uint8).reshape(n, 2)
    blocks[:, 2:4] = dmin.view(np.uint8).reshape(n, 2)
    blocks[:, 4:144] = rng.integers(0, 256, size=(n, 140), dtype=np.uint8)   # full code space, both fields

    ref = gguf.quants.dequantize(blocks.reshape(-1), GT.Q4_K).reshape(n, 256).astype(np.float64)
    _assert_finite(ref, "Q4_K golden")

    worst = 0.0
    for b in range(n):
        s, qs = blocks[b, 4:16], blocks[b, 16:144]
        dd, dm = np.float64(d[b]), np.float64(dmin[b])
        w = np.empty(256)
        for g, (sc, mn) in enumerate(_our_q4k_sc_mn(s)):
            j = g // 2 * 32
            nib = (qs[j:j + 32] & 0xF) if g % 2 == 0 else (qs[j:j + 32] >> 4)
            w[g * 32:(g + 1) * 32] = dd * sc * nib.astype(np.float64) - dm * mn
        worst = max(worst, np.abs(w - ref[b]).max() / max(1e-9, np.abs(ref[b]).max()))
    assert worst < 1e-6, f"Q4_K field layout disagrees with llama.cpp: worst relative error {worst:.3e}"


# ===================================================================================================================
# THE WHOLE PRE-PASS, FOR ALL FIVE FORMATS, AGAINST THE OFFICIAL IMPLEMENTATION AND THROUGH THE PYTHON ENTRY.
#
# HOW scale AND zero ARE OBTAINED FROM THE REFERENCE WITHOUT WRITING DOWN A SINGLE FORMAT CONSTANT. Each format's
# dequantisation is w = code * scale + affine, with `code` running over an integer range whose OFFSET differs per
# format (Q3_K's codes are centred at -4, Q6_K's at -32, the rest start at 0). Writing those offsets down here is
# exactly the mistake this file has already made twice about byte ranges, so they are derived instead:
#
#   1. set dmin = 0, so the affine term vanishes and w = code * scale exactly
#   2. fill the codes 0x00 and 0xFF -- the only two byte values uniform in every bit, so every element of a group
#      gets the same code -- giving w(c_lo) and w(c_hi)
#   3. scale = (w_hi - w_lo) / (c_hi - c_lo), where the DIFFERENCE is just the bit width and nothing else
#   4. c_lo = w_lo / scale then falls out, and the test ASSERTS IT IS AN INTEGER rather than assuming its value.
#      If our scale were wrong by any factor, this ratio would not be a whole number.
#
# Then dmin is restored and the shift in w isolates the format's affine term, which is what `zero` must carry.
#
# WHY THIS IS THE ORACLE THAT MATTERS. Every constant in gguf_scale_layout.hpp came from our own numpy parser, so a
# test against that parser proves nothing about either. gguf.quants is a separate implementation of the same spec.
def _official(raw, qtype, block_size):
    w = gguf.quants.dequantize(raw.reshape(-1), qtype).reshape(len(raw), block_size).astype(np.float64)
    _assert_finite(w, "golden")
    return w


@pytest.mark.parametrize("name,qtype,hdr,scales,codes,qmax", FORMATS)
def test_prepass_scale_matches_llama_cpp(name, qtype, hdr, scales, codes, qmax):
    quactlize = pytest.importorskip("quactlize", reason="needs the built operator library")
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1234 + qmax)
    block_size, _ = GGML_QUANT_SIZES[qtype]
    blk_bytes, groups, gsize, has_min, _bias, _signed = quactlize.gguf_scale_block_shape(int(qtype))
    assert (scales[1] - scales[0]) == blk_bytes, \
        f"{name}: the layout table's scale range is {scales[1]-scales[0]} bytes but Traits says {blk_bytes}"
    assert groups * gsize == block_size, f"{name}: groups*group_size != block size"

    n = 64
    base = _make_blocks(qtype, hdr, scales, codes, n, rng, code_fill=0x00)
    if has_min:                                   # dmin = 0 isolates the scale; hdr[1] is the dmin range
        base[:, hdr[1][0]:hdr[1][1]] = 0

    ws = []
    for fill in (0x00, 0xFF):
        raw = base.copy()
        for lo, hi in codes:
            raw[:, lo:hi] = fill
        ws.append(_official(raw, qtype, block_size))
    w_lo = ws[0].reshape(n, groups, gsize)[:, :, 0]      # constant within a group; checked by the coverage test
    w_hi = ws[1].reshape(n, groups, gsize)[:, :, 0]
    scale_ref = (w_hi - w_lo) / qmax

    d = base[:, hdr[0][0]:hdr[0][1]].copy().view(np.float16).reshape(n)
    dmin = (base[:, hdr[1][0]:hdr[1][1]].copy().view(np.float16).reshape(n) if has_min
            else np.zeros(n, np.float16))
    sb = torch.from_numpy(np.ascontiguousarray(base[:, scales[0]:scales[1]]))
    scale_ours, zero_ours = quactlize.gguf_scale_prepass(
        sb, torch.from_numpy(d.copy()), torch.from_numpy(dmin.copy()), int(qtype), 0)
    scale_ours = scale_ours.numpy().astype(np.float64)

    m = np.abs(scale_ref) > 1e-7
    assert m.sum() > n, f"{name}: almost every reference scale is zero; the fixture says nothing"
    rel = np.abs(scale_ours[m] - scale_ref[m]) / np.abs(scale_ref[m])
    assert rel.max() < 2e-3, \
        f"{name}: our scale disagrees with llama.cpp, worst relative error {rel.max():.3e}"

    # THE CODE OFFSET, DERIVED. If it is not a whole number our scale is wrong by a factor, which the relative check
    # above could miss only if the reference were wrong the same way -- it cannot be, being a separate implementation.
    c_lo = w_lo[m] / scale_ours[m]
    # THE TOLERANCE MUST SCALE WITH |c_lo|, and that is arithmetic rather than a loosened number. The numerator is
    # the reference's float32 output; the denominator is OUR fp16 scale, whose relative error is up to 2^-11 ~ 4.9e-4.
    # The ratio therefore carries an ABSOLUTE error of |c_lo| * 4.9e-4, which at Q6_K's offset of -32 is 1.6e-2 --
    # a fixed 1e-2 bound fails there while nothing is wrong. It still discriminates: an offset that is genuinely
    # non-integral is off by O(0.1), two orders above this.
    tol = 1e-3 * np.maximum(1.0, np.abs(c_lo))
    assert (np.abs(c_lo - np.round(c_lo)) < tol).all(), \
        f"{name}: w(0x00)/scale is not an integer code (spread {c_lo.min():.3f}..{c_lo.max():.3f})"
    assert np.allclose(np.round(c_lo), np.round(c_lo)[0]), f"{name}: the derived code offset is not uniform"


@pytest.mark.parametrize("name,qtype,hdr,scales,codes,qmax",
                         [f for f in FORMATS if f[0] in ("Q2_K", "Q4_K", "Q5_K")])
def test_prepass_zero_carries_the_affine_term(name, qtype, hdr, scales, codes, qmax):
    """The min channel, isolated: with the codes held fixed, restoring dmin shifts w by exactly the affine term.

    This is the half that a prepass gets silently wrong. -dmin*mn looks like a small correction and dropping it
    produces weights that are the right magnitude and the wrong values.
    """
    quactlize = pytest.importorskip("quactlize", reason="needs the built operator library")
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(99 + qmax)
    block_size, _ = GGML_QUANT_SIZES[qtype]
    _blk, groups, gsize, has_min, _b, _s = quactlize.gguf_scale_block_shape(int(qtype))
    assert has_min, f"{name} should have a min channel"

    n = 64
    base = _make_blocks(qtype, hdr, scales, codes, n, rng, code_fill=0x00)
    for lo, hi in codes:
        base[:, lo:hi] = 0x00                       # codes fixed, so only the affine term can move
    with_dmin = base.copy()
    zero_dmin = base.copy()
    zero_dmin[:, hdr[1][0]:hdr[1][1]] = 0

    w_with = _official(with_dmin, qtype, block_size).reshape(n, groups, gsize)[:, :, 0]
    w_zero = _official(zero_dmin, qtype, block_size).reshape(n, groups, gsize)[:, :, 0]
    affine_ref = w_with - w_zero                    # = -dmin*mn, by construction of the two fixtures

    d = with_dmin[:, hdr[0][0]:hdr[0][1]].copy().view(np.float16).reshape(n)
    dmin = with_dmin[:, hdr[1][0]:hdr[1][1]].copy().view(np.float16).reshape(n)
    sb = torch.from_numpy(np.ascontiguousarray(with_dmin[:, scales[0]:scales[1]]))
    scale0, zero0 = quactlize.gguf_scale_prepass(
        sb, torch.from_numpy(d.copy()), torch.from_numpy(dmin.copy()), int(qtype), 0)
    zero0 = zero0.numpy().astype(np.float64)

    denom = np.maximum(np.abs(affine_ref).max(), 1e-6)
    err = np.abs(zero0 - affine_ref).max() / denom
    assert err < 5e-3, f"{name}: our zero does not carry llama.cpp's affine term (worst rel {err:.3e})"

    # ZMul IS ADDED ON TOP OF THAT, not instead of it. Getting this backwards is the recorded trap.
    scale8, zero8 = quactlize.gguf_scale_prepass(
        sb, torch.from_numpy(d.copy()), torch.from_numpy(dmin.copy()), int(qtype), 8)
    lhs = zero8.numpy().astype(np.float64)
    rhs = zero0 + 8.0 * scale0.numpy().astype(np.float64)
    assert np.abs(lhs - rhs).max() < 5e-3 * denom, f"{name}: zero(zmul=8) != zero(0) + 8*scale"


# ===================================================================================================================
# THE PURE CUDA-CORE DECODE, ALL FIVE FORMATS, THROUGH THE PYTHON ENTRY.
#
# WHY A DOT PRODUCT AND NOT A PER-GROUP COMPARISON. The pre-pass tests above compare scale and zero, which are
# per-group scalars -- so a decoder with the right scales and the WRONG ELEMENT ORDER passes every one of them. A dot
# product against the reference's own weights cannot be fooled that way: permute anything and the sum moves. This is
# therefore the first test in the tree that checks the k-quant weight unpacking at all, and for Q4_K and Q5_K it also
# cross-validates gguf_scale_layout.hpp's 6-bit field maps end to end, since vecdot reads them through scale_of/min_of.
@pytest.mark.parametrize("name,qtype,hdr,scales,codes,qmax", FORMATS)
def test_vecdot_matches_llama_cpp(name, qtype, hdr, scales, codes, qmax):
    quactlize = pytest.importorskip("quactlize", reason="needs the built operator library")
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4242 + qmax)
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    n = 64
    raw = rng.integers(0, 256, size=(n, type_size), dtype=np.uint8)
    for lo, hi in hdr:                       # normal fp16 headers, or the golden is NaN and every check passes
        v = (rng.random(n) * 0.1 + 0.001).astype(np.float16)
        raw[:, lo:hi] = v.view(np.uint8).reshape(n, 2)
    w = gguf.quants.dequantize(raw.reshape(-1), qtype).reshape(n, block_size).astype(np.float64)
    _assert_finite(w, f"{name} golden")

    x = (rng.random((n, block_size)) * 2 - 1).astype(np.float32)
    ref = (w * x.astype(np.float64)).sum(1)
    got = quactlize.gguf_vecdot(torch.from_numpy(raw), torch.from_numpy(x), int(qtype)).numpy().astype(np.float64)

    rel = np.abs(got - ref).max() / max(1e-9, np.abs(ref).max())
    # ~1e-7 is float32 summation noise over 256 terms. It is this tight ONLY because both sides accumulate in the
    # same element order; the shipping GEMV consumes the offline-reordered weight and will need a looser,
    # summation-order tolerance. See gguf_vecdot.hpp's header -- that path is not what this tests.
    assert rel < 2e-5, f"{name}: vecdot disagrees with llama.cpp, worst relative error {rel:.3e}"


@pytest.mark.parametrize("name,qtype,hdr,scales,codes,qmax", FORMATS)
def test_dequantize_matches_llama_cpp(name, qtype, hdr, scales, codes, qmax):
    """THE FALLBACK PATH: raw block -> fp16 weights, compared elementwise to the reference.

    Tolerance is fp16 rounding, not float32: the output dtype is half, so 2^-11 relative is the floor and asking for
    the ~1e-7 the vecdot reaches would fail on the one thing that is not a bug. This is the same traversal vecdot
    uses, so a disagreement here would mean the fp16 conversion, not the layout.
    """
    quactlize = pytest.importorskip("quactlize", reason="needs the built operator library")
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(555 + qmax)
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    n = 64
    raw = rng.integers(0, 256, size=(n, type_size), dtype=np.uint8)
    for lo, hi in hdr:
        v = (rng.random(n) * 0.1 + 0.001).astype(np.float16)
        raw[:, lo:hi] = v.view(np.uint8).reshape(n, 2)
    ref = gguf.quants.dequantize(raw.reshape(-1), qtype).reshape(n, block_size).astype(np.float64)
    _assert_finite(ref, f"{name} golden")
    got = quactlize.gguf_dequantize(torch.from_numpy(raw), int(qtype)).numpy().astype(np.float64)
    rel = np.abs(got - ref).max() / max(1e-9, np.abs(ref).max())
    assert rel < 1e-3, f"{name}: fp16 dequantise disagrees with llama.cpp, worst relative error {rel:.3e}"


@pytest.mark.parametrize("name,qtype,hdr,scales,codes,qmax", FORMATS)
def test_unpack_split_reconstructs(name, qtype, hdr, scales, codes, qmax):
    """codes/scale/zero must satisfy W = code*scale + zero against the reference, exactly.

    This is the split every offline packer in the tree consumes, so it is also the point where a traversal that hands
    out the wrong element for a code shows up: the per-group tests cannot see it, and the vecdot test sees it only as
    a changed sum. Here each element is compared on its own.
    """
    quactlize = pytest.importorskip("quactlize", reason="needs the built operator library")
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(777 + qmax)
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    n = 64
    raw = rng.integers(0, 256, size=(n, type_size), dtype=np.uint8)
    for lo, hi in hdr:
        v = (rng.random(n) * 0.1 + 0.001).astype(np.float16)
        raw[:, lo:hi] = v.view(np.uint8).reshape(n, 2)
    ref = gguf.quants.dequantize(raw.reshape(-1), qtype).reshape(n, block_size).astype(np.float64)
    _assert_finite(ref, f"{name} golden")

    c, s, z = quactlize.gguf_unpack(torch.from_numpy(raw), int(qtype))
    c = c.numpy().astype(np.float64); s = s.numpy().astype(np.float64); z = z.numpy().astype(np.float64)
    groups = s.shape[1]
    gsz = block_size // groups
    w = (c.reshape(n, groups, gsz) * s[:, :, None] + z[:, :, None]).reshape(n, block_size)
    rel = np.abs(w - ref).max() / max(1e-9, np.abs(ref).max())
    assert rel < 1e-3, f"{name}: code*scale+zero does not reconstruct llama.cpp's weights (rel {rel:.3e})"

    # The code range is a property of the format and is asserted, not printed: a decoder that silently clamped or
    # sign-extended wrongly would still reconstruct if the scale absorbed it, but the range would move.
    lo_hi = {"Q2_K": (0, 3), "Q3_K": (-4, 3), "Q4_K": (0, 15), "Q5_K": (0, 31), "Q6_K": (-32, 31)}[name]
    assert (c.min(), c.max()) == lo_hi, f"{name}: codes span {c.min()}..{c.max()}, expected {lo_hi}"


def test_q4k_reaches_the_existing_layout_packers():
    """The chain a GGUF checkpoint has to walk to reach the kernels that already work.

    raw block -> gguf_unpack -> shift to the signed code range -> pack_int4 -> preprocess_weights_to_layout.
    Every step after the first is an op this repo already validates, which is the point: the alternative was a new
    packer for GGUF, i.e. a second thing to be wrong about a layout.
    """
    quactlize = pytest.importorskip("quactlize", reason="needs the built operator library")
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(31)
    rows = 64                                        # 64 superblocks -> a (1, 64*256) row of codes
    raw = rng.integers(0, 256, size=(rows, 144), dtype=np.uint8)
    for lo, hi in [(0, 2), (2, 4)]:
        v = (rng.random(rows) * 0.1 + 0.001).astype(np.float16)
        raw[:, lo:hi] = v.view(np.uint8).reshape(rows, 2)
    c, _s, _z = quactlize.gguf_unpack(torch.from_numpy(raw), 12)

    # Q4_K codes are UNSIGNED 0..15; the packers take the signed -8..7 range, and the +8 that maps between them is
    # the same constant the in-kernel converter carries as kPackedZMul. Skipping it here would pack the right bits
    # under the wrong interpretation and only show up as wrong numbers much later.
    # SHAPE MATTERS TO THE PACKER, and the first version flattened to (1, 16384) and was refused with "number of
    # rows of quantized matrix must be a multiple of ...". Keeping the natural (rows, 256) -- one superblock per row
    # of the weight -- is both what a checkpoint looks like and what the packer's row constraint expects.
    signed = (c.to(torch.int8) - 8)
    packed = quactlize.pack_int4(signed)
    assert packed.numel() * 2 == signed.numel(), "pack_int4 must halve the element count"
    back = quactlize.unpack_int4(packed)
    assert torch.equal(back, signed), "pack/unpack_int4 must round-trip the GGUF codes"

    laid = quactlize.preprocess_weights_to_layout(packed, torch.quint4x2, "mixed_gemm")
    assert laid.numel() == packed.numel(), "the layout transform must be byte-neutral"
    assert not torch.equal(laid, packed), "the layout transform must actually rearrange something"
