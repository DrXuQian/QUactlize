"""Shared logical affine group64 weights in three different native layouts.

Group64 is a declared common subset, not arbitrary GGUF group32 support in
Machete. Canonical K-pack repeats each scale/min across two group32 entries.
All scalar operations are exact for both FP16 and BF16 in this fixture.
"""
import numpy as np
import torch


def canonical(q, d, scale_codes, min_codes):
    k, n = q.shape
    # Canonical word holds k={0,8,16,24} within each K32 group.
    lanes = q.reshape(k // 32, 4, 8, n).astype(np.uint16)
    words = np.zeros((k // 32, 8, n), dtype=np.uint16)
    for slot in range(4):
        words |= lanes[:, slot] << (4 * slot)
    units = np.zeros((k // 256, n, 16), dtype=np.uint8)
    for header, value in ((0, d), (2, d / 2)):
        bits = value.astype('<f2').view('<u2')
        units[:, :, header] = bits & 255
        units[:, :, header+1] = bits >> 8
    for which, codes in enumerate((scale_codes, min_codes)):
        fields = np.repeat(codes, 2, axis=0).reshape(k // 256, 8, n)
        for group in range(8):
            bit = 32 + (group // 4) * 48 + which * 24 + (group % 4) * 6
            byte, shift = divmod(bit, 8)
            value = fields[:, group].astype(np.uint16) << shift
            units[:, :, byte] |= (value & 255).astype(np.uint8)
            if shift + 6 > 8:
                units[:, :, byte+1] |= (value >> 8).astype(np.uint8)
    return words.view(np.uint8).reshape(-1), units.reshape(-1)


def unpack_canonical(low, units, n, k):
    """Independent logical-index decode, including all metadata bits."""
    words = low.view('<u2').reshape(k // 4, n)
    kk = np.arange(k)
    offsets = kk // 32 * 8 + kk % 8
    q = ((words[offsets] >> (4 * (kk % 32 // 8))[:, None]) & 15).astype(np.uint8)
    unit = units.reshape(k // 256, n, 16)
    d = (unit[:, :, 0].astype(np.uint16) | unit[:, :, 1].astype(np.uint16) << 8).view('<f2').astype(np.float32)
    dmin = (unit[:, :, 2].astype(np.uint16) | unit[:, :, 3].astype(np.uint16) << 8).view('<f2').astype(np.float32)
    fields = []
    for which in range(2):
        groups = []
        for group in range(8):
            bit = 32 + (group // 4) * 48 + which * 24 + (group % 4) * 6
            byte, shift = divmod(bit, 8)
            value = unit[:, :, byte].astype(np.uint16)
            if shift + 6 > 8:
                value |= unit[:, :, byte+1].astype(np.uint16) << 8
            groups.append((value >> shift) & 63)
        fields.append(np.stack(groups, axis=1).reshape(k // 32, n))
    scale = np.repeat(d, 8, axis=0) * fields[0]
    minimum = np.repeat(dmin, 8, axis=0) * fields[1]
    return q, scale, minimum


def make_fixture(m, n, k, dtype):
    if n % 128 or k % 256 or m < 1:
        raise ValueError('common fixture dimensions')
    rng = np.random.default_rng(1731 + n + 3*k + m)
    q = rng.integers(0, 16, (k, n), dtype=np.uint8)
    col = np.arange(n)
    group = np.arange(k // 64)[:, None]
    d = np.exp2(-10 + col % 3).astype(np.float32)
    sc = (1 << ((group + col) % 4)).astype(np.uint8)
    mc = ((3*group + col) % 4).astype(np.uint8)
    scale = sc * d
    minimum = mc * d / 2
    weight = q.astype(np.float32) * np.repeat(scale, 64, axis=0) - np.repeat(minimum, 64, axis=0)
    typed = torch.from_numpy(weight).to(dtype)
    if not torch.equal(typed.float(), torch.from_numpy(weight)):
        raise ValueError('fixture weights are not exact in the compute dtype')
    low, units = canonical(q, d, sc, mc)
    rq, rs, rm = unpack_canonical(low, units, n, k)
    if not (np.array_equal(q, rq) and np.array_equal(rs, np.repeat(scale, 2, axis=0))
            and np.array_equal(rm, np.repeat(minimum, 2, axis=0))):
        raise ValueError('canonical packing roundtrip failed')
    # Verify centered-code typed multiply/add semantics too (Q4 converter bias8).
    centered = torch.from_numpy(q.astype(np.int16)-8).to(dtype)
    ts = torch.from_numpy(np.repeat(scale, 64, axis=0)).to(dtype)
    tz = torch.from_numpy(np.repeat(8*scale-minimum, 64, axis=0)).to(dtype)
    reconstructed = (centered * ts).to(dtype) + tz
    if not torch.equal(reconstructed, typed):
        raise ValueError('typed reconstruction differs across representations')
    a = torch.from_numpy(rng.integers(-8, 9, (m, k)).astype(np.float32) / 16).to(dtype)
    expected = (a.double() @ typed.double()).to(dtype).float()
    return {'q': q, 'low': low, 'units': units, 'a': a, 'expected': expected,
            'scale': torch.from_numpy(scale).to(dtype),
            'minimum': torch.from_numpy(minimum).to(dtype)}


def marlin_metadata(value):
    perm = torch.tensor([i+8*j for i in range(8) for j in range(8)])
    return value.reshape(-1, 64)[:, perm].reshape(value.shape).contiguous()
