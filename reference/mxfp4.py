"""Lossless GGUF MXFP4 <-> Sail W4FA16 layout oracle. No quantization."""
# Layout permutation attribution and MIT terms: ../LICENSE-DeepGEMM-layout.
import numpy as np

MAPPING_ID = 0x4D58463441313601


def permutations():
    # DeepGEMM-for-sail tests/math_utils.py::_get_perms, expressed as an
    # independent coordinate enumeration rather than the device bit formula.
    values = []
    for lane in range(32):
        rows = [2*(lane % 4), 2*(lane % 4)+1, 2*(lane % 4)+8, 2*(lane % 4)+9]
        for tile in range(4):
            fragment = [256*tile+16*r+lane//4+half*8 for half in range(2) for r in rows]
            values.extend(fragment[i] for i in (0, 2, 4, 6, 1, 3, 5, 7))
    scale = np.array([i+8*j for i in range(8) for j in range(8)])
    scale = scale.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(-1)
    return np.array(values), scale


def raw_parts(raw):
    raw = np.asarray(raw)
    if raw.dtype != np.uint8 or raw.ndim != 4 or raw.shape[-1] != 17:
        raise ValueError('expected uint8 GGUF [E,N,K/32,17]')
    e, n, kb, _ = raw.shape
    if e < 1 or n < 1 or n % 64 or kb < 1:
        raise ValueError('MXFP4 requires E>0, N%64=0, K%32=0')
    codes = np.concatenate((raw[..., 1:] & 15, raw[..., 1:] >> 4), axis=-1).reshape(e, n, kb*32)
    return codes, raw[..., 0].copy()


def pack(raw):
    codes, scales = raw_parts(raw)
    e, n, k = codes.shape
    perm, sp = permutations()
    tiled = codes.transpose(0, 2, 1).reshape(e, k//16, 16, n//16, 16)
    tiled = tiled.transpose(0, 1, 3, 2, 4).copy().reshape(-1, 1024)[:, perm]
    nibs = tiled.reshape(e, k//16, 2*n, 8).astype('<u4')
    weights = np.bitwise_or.reduce(nibs << np.arange(0, 32, 4, dtype='<u4'), axis=-1)
    scales = scales.transpose(0, 2, 1).copy().reshape(-1, 64)[:, sp].reshape(e, k//32, n).copy()
    return np.ascontiguousarray(weights, dtype='<u4'), np.ascontiguousarray(scales)


def recover(weights, scales):
    weights, scales = np.asarray(weights), np.asarray(scales)
    if weights.dtype != np.dtype('<u4') or weights.ndim != 3 or scales.dtype != np.uint8:
        raise ValueError('expected uint32 weights and uint8 E8M0 scales')
    e, kt, twice_n = weights.shape
    n, k = twice_n//2, kt*16
    if twice_n % 2 or n % 64 or k % 32 or e < 1 or n < 1 or k < 1 or scales.shape != (e, k//32, n):
        raise ValueError('W4FA16 plane shape mismatch')
    perm, sp = permutations()
    codes = ((weights[..., None] >> np.arange(0, 32, 4, dtype='<u4')) & 15).astype('u1')
    codes = codes.reshape(-1, 1024)[:, np.argsort(perm)]
    codes = codes.reshape(e, k//16, n//16, 16, 16).transpose(0, 1, 3, 2, 4)
    codes = codes.reshape(e, k, n).transpose(0, 2, 1).reshape(e, n, k//32, 32)
    sf = scales.reshape(-1, 64)[:, np.argsort(sp)].reshape(e, k//32, n).transpose(0, 2, 1)
    return np.concatenate((sf[..., None], codes[..., :16] | (codes[..., 16:] << 4)), axis=-1)


def dequantize(raw):
    codes, scales = raw_parts(raw)
    lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype='f8')
    s = np.exp2(scales.astype('i4')-127).astype('f8')
    s[scales == 255] = np.nan
    return lut[codes]*np.repeat(s, 32, axis=-1)
