"""Chunked original-GGUF oracle; never materialize a vocabulary-sized FP32 B."""
import hashlib
import math
import numpy as np

from dev.bf16_compute.fixture import raw_weight, planes, round_compute


class Weights:
    def __init__(self, q, n, k, experts):
        self.q, self.n, self.k, self.experts = q, n, k, experts
        self.categories = np.random.default_rng(81811 + k).integers(0, 4, k)
        self.planes = {}
        count = min(experts, 17)
        sums = np.empty((count, 4, n), dtype="f8")
        absolutes = np.empty_like(sums)
        hashes = []
        for e in range(count):
            hasher = hashlib.sha256()
            for first in range(0, n, 256):
                stop = min(n, first + 256)
                raw, gold = raw_weight(q, stop - first, k, 710 + e * 31 + q * 1009 + first)
                hasher.update(raw)
                packed = planes(raw, q)
                for name in ("low", "high", "units"):
                    arr = packed[name]
                    if not arr.size:
                        self.planes[name] = np.empty(0, "u1")
                        continue
                    if name not in self.planes:
                        shape = list(arr.shape)
                        # All plane producers expose N as axis 1, including
                        # [superunit,N,unit_bytes] and Q8 [K/32,N] metadata.
                        shape[1] = n
                        self.planes[name] = np.empty((experts, *shape), dtype=arr.dtype)
                    self.planes[name][e, :, first:stop] = arr
                for g in range(4):
                    part = gold[:, self.categories == g]
                    sums[e, g, first:stop] = part.sum(axis=1, dtype="f8")
                    absolutes[e, g, first:stop] = np.abs(part).sum(axis=1, dtype="f8")
            hashes.append(hasher.hexdigest())
        for arr in self.planes.values():
            if arr.size:
                for e in range(count, experts):
                    arr[e] = arr[e % count]
        self.sums, self.absolute = sums, absolutes
        self.record = dict(q=q, n=n, k=k, experts=experts, raw_sha256=hashes,
                           unique_patterns=count, oracle="OFFICIAL_GGUF_COMMON_EXACT_DYADIC_WEIGHTS",
                           activation="DENSE_FOUR_RANDOM_K_CATEGORIES_NONZERO_ROW_TAGS")

    def inputs(self, p, repeat=0):
        m, ch, top = p["tokens"], p["channels"], p["topk"]
        values = np.random.default_rng(931 + repeat).uniform(-.5, .5, (m * ch, 4)).astype("f4")
        values = round_compute(values, p["compute"])
        a = np.ascontiguousarray(values[:, self.categories])
        if not p["mode"]:
            owners, from_rows, ids = np.zeros(m, dtype="i4"), np.arange(m), None
        else:
            router = p["router"]
            buckets = np.arange(m)
            if router == "cluster":
                buckets[:] = 0
            elif router.startswith("repeat"):
                buckets //= int(router[6:])
            if router == "real":
                rng = np.random.default_rng(4013 + repeat)
                weight = np.ones(self.experts)
                weight[:min(16, self.experts)] = 4
                ids = np.stack([rng.choice(self.experts, top, replace=False, p=weight/weight.sum()) for _ in range(m)]).astype("i4")
            else:
                ids = ((np.arange(top)[None, :] + 13 * buckets[:, None] + repeat * 7) % self.experts).astype("i4")
            owners = ids.ravel()
            from_rows = np.arange(m * top) // top * ch + np.arange(m * top) % top % ch
        v = values[from_rows].astype("f8")
        patterns = owners % len(self.sums)
        gold = np.einsum("rg,rgn->rn", v, self.sums[patterns])
        denom = np.einsum("rg,rgn->rn", np.abs(v), self.absolute[patterns])
        return dict(a=a, ids=ids, owners=owners, gold=gold, denom=denom)


def copies_for_l2(expert_bytes, active, l2_bytes):
    if min(expert_bytes, active, l2_bytes) <= 0:
        raise ValueError("verified positive L2 and active bytes required")
    return max(1, math.ceil(2.25 * l2_bytes / (expert_bytes * active)))
