"""Q6-only independent GGUF fixture with varying signed scales in every group."""
import hashlib
import numpy as np

from dev.smallm_closure.fixture import Weights as BaseWeights
from dev.bf16_compute.fixture import planes


class Weights(BaseWeights):
    # Reuse activation/routing/golden composition; do not reuse the old
    # common-dyadic fixture's all-one Q6 scales to validate a scale reader.
    def __init__(self, n, k, experts):
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        from reference.gguf_kpack import SPECS
        spec = SPECS[14]
        self.q, self.n, self.k, self.experts = 14, n, k, experts
        self.categories = np.random.default_rng(81811+k).integers(0, 4, k)
        self.planes = {}
        count = min(experts, 17)
        self.sums = np.empty((count, 4, n), dtype='f8')
        self.absolute = np.empty_like(self.sums)
        hashes = []
        for expert in range(count):
            hasher = hashlib.sha256()
            for first in range(0, n, 256):
                rows = min(256, n-first)
                rng = np.random.default_rng(710+expert*31+first)
                raw = rng.integers(0, 256, (rows, k//256, spec.raw_bytes), dtype='u1')
                # Random signed int8 scale bytes, including -128, 0 and127.
                scales = raw[..., spec.scale_offset:spec.scale_offset+16]
                tags = np.arange(rows)[:, None]+np.arange(k//256)[None, :]+expert+first
                scales[..., :3] = np.asarray([128, 0, 127], dtype='u1')
                d = np.ldexp(np.where(tags&1, -1., 1.), -(tags%5+12)).astype('<f2')
                raw[..., spec.d_offset:spec.d_offset+2] = d.view('u1').reshape(rows, -1, 2)
                hasher.update(raw.tobytes())
                gold = dequantize(raw.reshape(-1), GGMLQuantizationType.Q6_K).reshape(rows, k).astype('f4')
                if not np.isfinite(gold).all() or not np.any(gold):
                    raise ValueError('invalid independent Q6 fixture')
                packed = planes(raw, 14)
                for name in ('low', 'high', 'units'):
                    arr = packed[name]
                    if name not in self.planes:
                        shape = list(arr.shape);shape[1] = n
                        self.planes[name] = np.empty((experts, *shape), dtype=arr.dtype)
                    self.planes[name][expert, :, first:first+rows] = arr
                for group in range(4):
                    part = gold[:, self.categories == group]
                    self.sums[expert, group, first:first+rows] = part.sum(axis=1, dtype='f8')
                    self.absolute[expert, group, first:first+rows] = np.abs(part).sum(axis=1, dtype='f8')
            hashes.append(hasher.hexdigest())
        for arr in self.planes.values():
            for expert in range(count, experts):
                arr[expert] = arr[expert % count]
        self.record = dict(q=14, n=n, k=k, experts=experts, raw_sha256=hashes,
                           unique_patterns=count, oracle='OFFICIAL_GGUF_Q6_F32_DOT',
                           metadata='VARYING_SIGNED_GROUP_SCALES_AND_SUPERSCALE_SIGNS',
                           activation='FOUR_K_CATEGORIES_ROUNDED_TO_REQUESTED_COMPUTE')
