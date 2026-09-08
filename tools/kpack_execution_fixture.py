"""Indexed GEMV fixture with shared K categories across experts.

The original calibrated warmup fixture remains byte-for-byte unchanged.
Only the new indexed-input experiment uses this factorization; packing and
the activation oracle are reused, and weights still come from official GGUF.
"""

import numpy as np

from reference import gguf_kpack as ref
from tools.kpack_warmup_fixture import Weights, prepare_expert


class IndexedWeights(Weights):
    def __init__(self, q, n, k, experts, progress=None):
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize

        self.q, self.n, self.k, self.experts = q, n, k, experts
        spec = ref.SPECS[q]
        ref._validate_geometry(spec, n, k)
        self.planes = {}
        category = np.random.default_rng(81811 + k).integers(0, 4, k, dtype=np.uint8)
        self.categories = np.tile(category, (experts, 1))
        self.sums = np.empty((experts, 4, n), dtype=np.float64)
        self.abs_sums = np.empty_like(self.sums)
        for e in range(experts):
            rng = np.random.default_rng(np.random.SeedSequence([81923, q, n, k, e]))
            raw = rng.integers(0, 256, (n * (k // 256), spec.raw_bytes), dtype=np.uint8)
            for offset in (spec.d_offset, spec.dmin_offset):
                if offset >= 0:
                    h = rng.uniform(0.005, 0.03, len(raw)).astype("<f2")
                    raw[:, offset : offset + 2] = h.view("u1").reshape(-1, 2)
            placed = prepare_expert(raw, q, n, k)
            for name, array in placed.items():
                if e == 0:
                    self.planes[name] = np.empty(
                        (experts, *array.shape), dtype=array.dtype
                    )
                self.planes[name][e] = array
            official = dequantize(raw.reshape(-1), GGMLQuantizationType(q)).reshape(
                n, k
            )
            if not np.isfinite(official).all():
                raise ValueError("nonfinite independent GGUF oracle")
            for g in range(4):
                subset = official[:, category == g]
                self.sums[e, g] = subset.sum(axis=1, dtype=np.float64)
                self.abs_sums[e, g] = np.abs(subset).sum(axis=1, dtype=np.float64)
            if progress and (e + 1 == experts or (e + 1) % 32 == 0):
                progress(e + 1, experts)
