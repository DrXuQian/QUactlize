"""Product correctness-test support; performance runners are on experiments/ppu-tuning."""
import numpy as np
from reference import gguf_kpack as ref
from ci.support.fixture import bf16_bits, bf16_float
from ci.support.weights import prepare_expert
from ci.support.gemv import metadata_oracle
def fixture(q, n=256, k=512, experts=2):
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    s = ref.SPECS[q]
    rng = np.random.default_rng(916001 + q)
    raw = rng.integers(0, 256, (experts, n, k // 256, s.raw_bytes), dtype="u1")
    for offset in (s.d_offset, s.dmin_offset):
        if offset >= 0:
            headers = rng.uniform(.005, .025, raw.shape[:-1]).astype("<f2")
            headers.flat[:4] = np.array([1, 0x203, 0x1357, 0x33ab], dtype="<u2").view("<f2")
            raw[..., offset:offset+2] = headers.view("u1").reshape(*headers.shape, 2)
    scales, minima = np.zeros((2, experts, k // s.group_size, n), dtype="f4")
    for e in range(experts):
        for col in range(n):
            for sb in range(k // 256):
                block = raw[e, col, sb].tobytes()
                d = np.frombuffer(block[s.d_offset:s.d_offset+2], dtype="<f2").astype("f4")[0]
                dm = np.frombuffer(block[s.dmin_offset:s.dmin_offset+2], dtype="<f2").astype("f4")[0] if s.has_min else np.float32(0)
                for g in range(s.groups):
                    sc, mn = ref._metadata_codes(block, 0, s, g)
                    sc = sc-32 if q == 11 else sc-256 if q == 14 and sc >= 128 else sc
                    scales[e, sb*s.groups+g, col] = d*np.float32(sc)
                    minima[e, sb*s.groups+g, col] = -dm*np.float32(mn) if s.has_min else 0
    scale = bf16_float(bf16_bits(scales))
    zero = bf16_float(bf16_bits(minima))
    zmul = {10: 0, 11: -4, 12: 8, 13: 8, 14: -24}[q]
    if zmul:
        zero = bf16_float(bf16_bits(zero + bf16_float(bf16_bits(np.float32(zmul)*scale))))
    golden = np.stack((bf16_bits(scale), bf16_bits(zero)))
    items = [prepare_expert(raw[e].reshape(-1, s.raw_bytes), q, n, k) for e in range(experts)]
    planes = {name: np.stack([p[name] for p in items]) for name in ("low", "high", "units")}
    full = bf16_bits(dequantize(raw.reshape(-1), GGMLQuantizationType(q)).reshape(experts, n, k))
    legacy = np.stack(metadata_oracle(planes["units"], q, n, k, experts)).view("<u2")
    if np.array_equal(bf16_bits(legacy[0].view("<f2").astype("f4")), golden[0]):
        raise ValueError("fixture cannot reject a decoded FP16 intermediate")
    return planes, golden, full, legacy
