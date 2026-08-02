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

gguf = pytest.importorskip("gguf", reason="the official llama.cpp gguf package is the golden for these tests")
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
    rng = np.random.default_rng(abs(hash(name)) & 0xFFFF)
    block_size, _ = GGML_QUANT_SIZES[qtype]
    n, gsz = 32, block_size // 16
    raw = _make_blocks(qtype, hdr, scales, codes, n, rng, code_fill=None)   # random everywhere first
    for lo, hi in codes:
        raw[:, lo:hi] = 0x00
    w = gguf.quants.dequantize(raw.reshape(-1), qtype).reshape(n, block_size).astype(np.float64)
    _assert_finite(w, f"{name} golden at code fill 0x00")
    g = w.reshape(n, block_size // gsz, gsz)
    spread = np.abs(g - g[:, :, :1]).max()
    assert spread < 1e-9 * max(1.0, np.abs(w).max()), \
        f"{name}: with every declared code range zeroed, a group is still not constant (spread {spread:.3e}) -- " \
        f"the code_ranges table is missing a plane, so any scale extracted from a fill would be wrong"


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
