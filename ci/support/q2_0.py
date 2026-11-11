"""Independent ID42 oracle: literal GGUF byte order, not the production reader.

Source: llama.cpp 356a231c1, dequantize_row_q2_0. Q2_K is not this encoding.
"""
import numpy as np


def dequantize(raw):
    blocks = np.ascontiguousarray(raw, dtype='u1').reshape(-1, 18)
    scale = blocks[:, :2].copy().view('<f2').astype('f4')
    codes = ((blocks[:, 2:, None] >> np.array([0, 2, 4, 6], 'u1')) & 3)
    return (codes.reshape(-1, 64).astype('f4') - 1) * scale


def recover(low, units, e, n, k):
    """Independent inverse: [E,K/8,N] words -> original 18-byte blocks."""
    words = np.asarray(low, 'u1').view('<u2').reshape(e, k//64, 8, n)
    codes = ((words[..., None] >> (2*np.arange(8, dtype='u2'))) & 3)
    codes = codes.transpose(0, 3, 1, 4, 2).reshape(e, n, k//64, 16, 4)
    raw = np.empty((e, n, k//64, 18), 'u1')
    raw[..., :2] = np.asarray(units, 'u1').reshape(e, k//64, n, 2).transpose(0, 2, 1, 3)
    raw[..., 2:] = np.bitwise_or.reduce(codes << (2*np.arange(4, dtype='u2')), axis=-1)
    return raw
