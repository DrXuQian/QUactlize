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


def assert_merge(got, values, weights, *, context=''):
    """Native-16 expert outputs x F32 routing weights, accumulated into F32.

    Use the ACTUAL projection outputs/weights to isolate this boundary. HGCC
    contracts the slot loop to FMA; NumPy rounds the products before summing.
    Relative error in the final sum is not a useful bound under cancellation.
    Each term sees at most topk FP32 roundings (with or without contraction),
    giving gamma_topk * sum(abs(products)); no arbitrary absolute tolerance.
    """
    got, values, weights = (np.asarray(x, 'f8') for x in (got, values, weights))
    assert values.ndim == 3 and weights.shape == values.shape[:2]
    assert got.shape == (values.shape[0], values.shape[2])
    assert all(np.isfinite(x).all() for x in (got, values, weights)), 'nonfinite weighted merge'
    topk = values.shape[1]
    assert 1 <= topk <= 32
    products = values * weights[..., None]
    gold = products.sum(axis=1)
    magnitude = np.abs(products).sum(axis=1)
    u32, u64 = 2**-24, 2**-53
    gamma32 = topk*u32 / (1-topk*u32)
    gamma64 = topk*u64 / (1-topk*u64)
    # Include FP64 reference summation and FP32 gradual-underflow rounding.
    # Native-16 x F32 products are exactly representable in FP64.
    bound = (gamma32+gamma64)/(1-gamma64)*magnitude + topk*2**-149/(1-topk*u32)
    error = np.abs(got-gold)
    bad = error > bound
    if bad.any():
        row, col = map(int, np.argwhere(bad)[0])
        raise AssertionError(
            f'FP32 weighted merge bound: {context} index=({row}, {col}) '
            f'got={got[row,col]:.17g} gold={gold[row,col]:.17g} '
            f'error={error[row,col]:.9g} bound={bound[row,col]:.9g} '
            f'sum_abs={magnitude[row,col]:.9g} topk={topk} '
            f'values={values[row,:,col].tolist()} weights={weights[row].tolist()}')
