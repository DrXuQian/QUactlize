"""Vectorized benchmark fixture, byte-checked against the scalar reference.

This is not a new production packer. Official GGUF dequantization is the
numeric oracle. Four random K categories make full-output reference evaluation
O(E*N*K + M*N) rather than a second full CPU GEMM. A is dense, FP16-exact and
row-dependent; every output is checked. Timed kernels use those same buffers.
"""

import numpy as np
from reference import gguf_kpack as ref


def pack_plane(codes, bits, *, first_n=0, q5_high=False):
    nc, k = codes.shape
    pack = 16 // bits
    if not q5_high:
        values = codes.reshape(nc, k // (8 * pack), pack, 8).astype(np.uint16)
        shifts = (np.arange(pack, dtype=np.uint16) * bits)[None, None, :, None]
        words = np.bitwise_or.reduce(values << shifts, axis=2)
        return words.reshape(nc, k // pack).T.copy()
    # Inverse coordinates of the separately specified Q5 high-plane transpose.
    pn = np.arange(first_n, first_n + nc)[None, :]
    kg = np.arange(k // 16)[:, None]
    words = np.zeros((k // 16, nc), dtype="<u2")
    for slot in range(16):
        n = (pn & ~15) | (pn & 7) | ((slot >> 3) << 3)
        kk = (
            (kg // 16) * 256
            | ((pn & 8) << 4)
            | ((kg & 8) << 3)
            | ((slot & 7) << 3)
            | (kg & 7)
        )
        words |= codes[n - first_n, kk].astype(np.uint16) << slot
    return words


def prepare_expert(raw, q, n, k):
    """Return canonical low/high/units plus resident half scale/zero planes."""
    spec = ref.SPECS[q]
    ref._validate_geometry(spec, n, k)
    raw = np.asarray(raw, dtype=np.uint8).reshape(n, k // 256, spec.raw_bytes)
    sb = k // 256
    low = np.empty((k // (16 // spec.low_bits), n), dtype="<u2")
    high = (
        np.empty((k // (16 // spec.high_bits), n), dtype="<u2")
        if spec.high_bits
        else np.empty(0, dtype="<u2")
    )
    units = np.zeros(
        (sb // spec.superblocks_per_unit, n, spec.unit_bytes), dtype=np.uint8
    )
    scales = np.empty((k // spec.group_size, n), dtype="<f2")
    zeros = np.empty_like(scales)
    for start in range(0, n, 256):
        stop = min(n, start + 256)
        nc = stop - start
        blocks = raw[start:stop].reshape(-1, spec.raw_bytes)
        byte_major = blocks.T
        lo = np.empty((nc, sb, 256), dtype=np.uint8)
        hi = np.empty_like(lo) if spec.high_bits else None
        for i in range(256):
            l, h = ref._raw_code_planes(byte_major, 0, spec, i)
            lo[:, :, i] = l.reshape(nc, sb)
            if hi is not None:
                hi[:, :, i] = h.reshape(nc, sb)
        low[:, start:stop] = pack_plane(lo.reshape(nc, k), spec.low_bits)
        if hi is not None:
            high[:, start:stop] = pack_plane(
                hi.reshape(nc, k), spec.high_bits, first_n=start, q5_high=q == 13
            )
        packed = np.zeros((nc, sb, spec.sb_bytes), dtype=np.uint8)
        packed[:, :, :2] = raw[start:stop, :, spec.d_offset : spec.d_offset + 2]
        if spec.has_min:
            packed[:, :, 2:4] = raw[
                start:stop, :, spec.dmin_offset : spec.dmin_offset + 2
            ]
        d = (
            blocks[:, spec.d_offset : spec.d_offset + 2]
            .copy()
            .view("<f2")
            .reshape(nc, sb)
        )
        dmin = (
            blocks[:, spec.dmin_offset : spec.dmin_offset + 2]
            .copy()
            .view("<f2")
            .reshape(nc, sb)
            if spec.has_min
            else np.zeros_like(d)
        )
        for g in range(spec.groups):
            sc, mn = ref._metadata_codes(byte_major, 0, spec, g)
            sc = sc.reshape(nc, sb)
            mn = np.broadcast_to(mn, (nc * sb,)).reshape(nc, sb)
            for which, value in ((0, sc), (1, mn)):
                if which and not spec.has_min:
                    continue
                bit = ref._unit_bit(spec, g, which)
                shift, byte = bit & 7, bit // 8
                value = value.astype(np.uint16) << shift
                packed[:, :, byte] |= (value & 255).astype(np.uint8)
                width = spec.min_bits if which else spec.scale_bits
                if shift + width > 8:
                    packed[:, :, byte + 1] |= (value >> 8).astype(np.uint8)
            code = sc.astype(np.int16)
            if q == 11:
                code -= 32
            elif q == 14:
                code = sc.view(np.int8)
            scale = np.multiply(d, code.astype(np.float16), dtype=np.float16)
            zero = -np.multiply(dmin, mn.astype(np.float16), dtype=np.float16)
            zmul = {10: 0, 11: -4, 12: 8, 13: 8, 14: -24}[q]
            if zmul:
                zero = np.add(
                    np.multiply(np.float16(zmul), scale, dtype=np.float16),
                    zero,
                    dtype=np.float16,
                )
            scales[g :: spec.groups, start:stop] = scale.T
            zeros[g :: spec.groups, start:stop] = zero.T
        units[:, start:stop, :] = packed.reshape(nc, -1, spec.unit_bytes).transpose(
            1, 0, 2
        )
    return dict(low=low, high=high, units=units, scale=scales, zero=zeros)


def activation_values(global_rows):
    # Bijective 16-bit row tag: neighboring rows are not nearly equal. Half-
    # integer coefficients avoid an all-zero row. Sums divided by 512 are
    # exactly representable in FP16, so the factorized oracle is not rounded A.
    tag = (np.asarray(global_rows, dtype=np.int64) * 40503 + 17389) & 65535
    c0 = (tag & 255) - 127.5
    c1 = (tag >> 8) - 127.5
    signs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float64)
    return (c0[:, None] * signs[None, :, 0] + c1[:, None] * signs[None, :, 1]) / 512


class Weights:
    def __init__(self, q, n, k, experts, progress=None):
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize

        self.q, self.n, self.k, self.experts = q, n, k, experts
        spec = ref.SPECS[q]
        ref._validate_geometry(spec, n, k)
        self.planes = {
            "low": np.empty((experts, k // (16 // spec.low_bits), n), dtype="<u2"),
            "high": (
                np.empty((experts, k // (16 // spec.high_bits), n), dtype="<u2")
                if spec.high_bits
                else np.empty(0, dtype="<u2")
            ),
            "units": np.empty(
                (experts, k // 256 // spec.superblocks_per_unit, n, spec.unit_bytes),
                dtype=np.uint8,
            ),
            "scale": np.empty((experts, k // spec.group_size, n), dtype="<f2"),
            "zero": np.empty((experts, k // spec.group_size, n), dtype="<f2"),
        }
        self.categories = np.empty((experts, k), dtype=np.uint8)
        self.sums = np.empty((experts, 4, n), dtype=np.float64)
        self.abs_sums = np.empty_like(self.sums)
        for e in range(experts):
            rng = np.random.default_rng(np.random.SeedSequence([95171, q, n, k, e]))
            raw = rng.integers(
                0, 256, size=(n * (k // 256), spec.raw_bytes), dtype=np.uint8
            )
            for offset in (spec.d_offset, spec.dmin_offset):
                if offset >= 0:
                    h = (rng.random(raw.shape[0]) * 0.025 + 0.005).astype("<f2")
                    raw[:, offset : offset + 2] = h.view(np.uint8).reshape(-1, 2)
            placed = prepare_expert(raw, q, n, k)
            for name, array in placed.items():
                if array.size:
                    self.planes[name][e] = array
            category = rng.integers(0, 4, size=k, dtype=np.uint8)
            self.categories[e] = category
            official = dequantize(raw.reshape(-1), GGMLQuantizationType(q)).reshape(
                n, k
            )
            if not np.isfinite(official).all():
                raise ValueError("nonfinite official GGUF oracle")
            for g in range(4):
                subset = official[:, category == g]
                self.sums[e, g] = subset.sum(axis=1, dtype=np.float64)
                self.abs_sums[e, g] = np.abs(subset).sum(axis=1, dtype=np.float64)
            if progress and (e + 1 == experts or (e + 1) % 32 == 0):
                progress(e + 1, experts)

    def activation(self, request):
        if (
            request.qtype,
            request.n,
            request.k,
            len(request.rows) if request.grouped else 1,
        ) != (self.q, self.n, self.k, self.experts):
            raise ValueError("request/weight fixture mismatch")
        a = np.empty((request.m, self.k), dtype="<f2")
        golden = np.empty((request.m, self.n), dtype=np.float64)
        denom = np.empty_like(golden)
        offset = 0
        for e, rows in enumerate(request.rows or (request.m,)):
            values = activation_values(np.arange(offset, offset + rows))
            a[offset : offset + rows] = values[:, self.categories[e]]
            golden[offset : offset + rows] = values @ self.sums[e]
            denom[offset : offset + rows] = np.abs(values) @ self.abs_sums[e]
            offset += rows
        if not np.isfinite(golden).all() or not np.any(golden) or not np.all(denom > 0):
            raise ValueError("degenerate full-output fixture")
        return a, golden, denom
