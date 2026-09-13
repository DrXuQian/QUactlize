"""Exact row domains and a CPU factorized oracle for resident BF16 GEMM."""
import hashlib
import numpy as np

from dev.gemv_ppu.decode_sweep import real_ids
from tools.kpack_dequant_fixture import bf16


def as_float(bits):
    return (np.asarray(bits, dtype='<u2').astype('<u4') << 16).view('<f4')


def row_domain(tokens, experts):
    if experts == 1:
        rows = np.asarray([tokens], dtype='<i4')
        return rows, np.zeros(tokens, dtype='<i4'), None
    if experts != 256:
        raise ValueError('grouped provider uses the previous E256/top8 real router')
    routes = real_ids(tokens)
    rows = np.bincount(routes.reshape(-1), minlength=experts).astype('<i4')
    indices = np.repeat(np.arange(experts, dtype='<i4'), rows)
    return rows, indices, hashlib.sha256(routes.tobytes()).hexdigest()


class Oracle:
    def __init__(self, bf16_weights):
        e, n, k = bf16_weights.shape
        self.categories = np.random.default_rng(60413+k).integers(0, 4, k)
        self.sums = np.empty((e,4,n),dtype='<f8')
        self.absolute = np.empty_like(self.sums)
        for expert in range(e):
            weights = as_float(bf16_weights[expert])
            for c in range(4):
                part = weights[:,self.categories==c]
                self.sums[expert,c] = part.sum(axis=1,dtype='f8')
                self.absolute[expert,c] = np.abs(part).sum(axis=1,dtype='f8')

    def activations(self, m):
        rng = np.random.default_rng(76121+m)
        coefficients = as_float(bf16(rng.uniform(-.25,.25,(m,4)).astype('f4')))
        return bf16(coefficients[:,self.categories]), coefficients.astype('f8')

    def error(self, output_bits, coefficients, indices):
        out = as_float(output_bits)
        if not np.isfinite(out).all() or out.shape != (len(indices),self.sums.shape[-1]):
            raise ValueError('BF16 output is incomplete or nonfinite')
        worst = 0.
        for expert in range(self.sums.shape[0]):
            idx = np.flatnonzero(indices==expert)
            for start in range(0,len(idx),256):
                selected = idx[start:start+256]
                wanted = coefficients[selected] @ self.sums[expert]
                denom = np.abs(coefficients[selected]) @ self.absolute[expert]
                err = np.abs(out[selected].astype('f8')-wanted)/np.maximum(denom,np.finfo('f8').tiny)
                worst = max(worst,float(err.max(initial=0)))
        return worst
