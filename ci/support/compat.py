"""Independent GGUF fixtures and arithmetic bounds for native typed execution."""
import numpy as np
import gguf

from .fixture import bf16_bits, bf16_float

FORMATS = (7, 8, 10, 11, 12, 13, 14)
# Logical N,K from target/mtp-index.json, GGUF [K,N,E] transposed explicitly.
DENSE_Q8 = ((320, 10240), (10240, 320), (4, 10240), (2560, 640), (512, 2560),
            (640, 2560), (2560, 2560), (6144, 2560), (10240, 2560), (12288, 2560),
            (248320, 2560), (2560, 6144), (2560, 5120))
EXPERT_SHAPES = ((7, 2560, 640), (8, 2560, 640), (8, 640, 2560),
                 (12, 640, 2560), (13, 640, 2560))


def raw_blocks(q, e, n, k, seed=0):
    block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(q)]
    rng = np.random.default_rng(22000 + q + seed)
    raw = rng.integers(0, 256, (e, n, k // block, size), dtype='u1')
    offsets = {7: (0, 2), 8: (0,), 10: (80, 82), 11: (108,), 12: (0, 2), 13: (0, 2), 14: (208,)}[q]
    for at in offsets:
        scales = rng.uniform(-.003, .003, raw.shape[:-1]).astype('<f2')
        raw[..., at:at+2] = scales[..., None].view('u1')
    return raw


def encode(values, dtype):
    return bf16_bits(values) if dtype else np.asarray(values, dtype='<f2').view('<u2')


def decode(values, dtype):
    return bf16_float(values) if dtype else np.asarray(values, dtype='<u2').view('<f2').astype('f4')


def weight(raw, q, dtype):
    # This oracle is the independent GGUF package, not the C++ reader/producer.
    result = gguf.dequantize(raw.reshape(-1, raw.shape[-1]), gguf.GGMLQuantizationType(q))
    return decode(encode(result.reshape(*raw.shape[:2], -1), dtype), dtype)


def assert_dot(got, a, w, dtype):
    a, w = np.asarray(a, 'f8'), np.asarray(w, 'f8')
    gold = a @ w.T
    magnitude = np.abs(a) @ np.abs(w.T)
    # FP32 multiply-accumulate + reduction, one native output RNE, including
    # absolute subnormal rounding. No near-zero relative-error division.
    u32 = 2**-24
    gamma = (a.shape[-1] + 32)*u32 / (1 - (a.shape[-1] + 32)*u32)
    uout = 2**(-8 if dtype else -11)
    tiny = 2**(-134 if dtype else -25)
    bound = gamma*magnitude*(1+uout) + np.abs(gold)*uout + tiny
    assert np.isfinite(got).all()
    bad = np.abs(got-gold) > bound
    if bad.any():
        at = tuple(np.argwhere(bad)[0])
        raise AssertionError(f'FP32 accumulation/native output bound: index={at} got={got[at]} gold={gold[at]} bound={bound[at]}')
