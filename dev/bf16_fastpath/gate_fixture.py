"""Factorized activation oracle; only eight distinct experts are decoded on CPU."""
import ctypes as C

import numpy as np

from dev.bf16_compute.fixture import raw_weight, planes, round_compute, digest, bf16_bits
from dev.gemv_simt.native import checked

OUTLIER = np.float32(243383.484375)
POISON = np.uint32(0xa5a5a5a5)


class Buffer:
    def __init__(self, rt, size):
        self.rt, self.size = rt, int(size)
        self.base = rt.allocate(self.size+32)
        self.ptr = self.base+16
        self.poison()

    def poison(self):
        self.rt.fill(self.base, self.size+32)

    def upload(self, value):
        value = np.ascontiguousarray(value)
        if value.nbytes != self.size:
            raise ValueError("fixture byte width differs")
        self.rt.copy(self.ptr, value)

    def read(self, dtype, shape=None):
        self.rt.sync()
        image = self.rt.download(self.base, self.size+32)
        if not np.all(image[:16] == 0xa5) or not np.all(image[-16:] == 0xa5):
            raise ValueError("buffer boundary changed")
        value = image[16:-16].copy().view(dtype)
        return value.reshape(shape) if shape is not None else value

    def guard(self):
        self.rt.sync()
        if (not np.all(self.rt.download(self.base, 16) == 0xa5) or
                not np.all(self.rt.download(self.ptr+self.size, 16) == 0xa5)):
            raise ValueError("input/weight boundary changed")


class Weights:
    def __init__(self, n, k, experts, seed=1709):
        self.n, self.k, self.experts = n, k, experts
        self.pool = min(experts, 8)
        self.category = np.random.default_rng(seed+k).integers(0, 4, k)
        self.outlier_index = 5613 if k > 5613 else k//3
        # Keep the large activation separate from category sums.
        self.category[self.outlier_index] = 4
        self.sums, self.absolute, self.columns, self.samples, self.hashes = [], [], [], [], []
        for p in range(self.pool):
            raw, golden = raw_weight(12, n, k, seed + p*31)
            packed = planes(raw, 12)
            sums = np.stack([golden[:, self.category == j].sum(1, dtype="f8") for j in range(4)])
            absolute = np.stack([np.abs(golden[:, self.category == j]).sum(1, dtype="f8") for j in range(4)])
            self.sums.append(sums)
            self.absolute.append(absolute)
            self.columns.append(golden[:, self.outlier_index].astype("f8"))
            self.samples.append({name: np.ascontiguousarray(packed[name]) for name in ("low", "units")})
            self.hashes.append(dict(raw=digest(raw), golden=digest(golden),
                low=digest(packed["low"]), units=digest(packed["units"])))

    def upload(self, rt):
        result = {}
        for name in ("low", "units"):
            size = self.samples[0][name].nbytes
            buffer = Buffer(rt, size*self.experts)
            for p in range(self.pool):
                rt.copy(buffer.ptr+p*size, self.samples[p][name])
            # Broadcast only on device. The host never expands 256 expert planes.
            for expert in range(self.pool, self.experts):
                checked(rt.MemcpyAsync(buffer.ptr+expert*size, buffer.ptr+(expert%self.pool)*size,
                    size, 3, rt.stream), "expert pattern D2D")
            rt.sync()
            result[name] = buffer
        return result

    def inputs(self, point, repeat, large=False):
        tokens, channels = point["tokens"], point["channels"]
        row_stride, token_stride = self.k+8, (self.k+8)*channels+8
        if point["mode"] == 0:
            token_stride = row_stride
        # Padded elements must not affect the dot. Logical inputs vary across
        # tokens/channels and cross BF16 rounding ties (not FP16-pre-rounded).
        image = np.full(tokens*token_stride, np.float32(19.25))
        coeff = np.empty((tokens, channels, 5), dtype="f4")
        for t in range(tokens):
            for ch in range(channels):
                values = np.array([0.13791, -0.28437, 0.41911, 0.68793, 0.21473], "f4")
                values *= np.float32(1 + (repeat*3+t*5+ch*7)%11/16)
                if large:
                    values[4] = OUTLIER if (t+ch)%2 == 0 else -OUTLIER
                coeff[t, ch] = values
                at = t*token_stride+ch*row_stride
                image[at:at+self.k] = values[self.category]
        ids = None
        if point["mode"] == 2:
            ids = np.full((tokens, 11), -177, dtype="i4")
            for t in range(tokens):
                # Distinct IDs, including high expert addresses. Pattern order
                # changes on replay, so retaining old IDs cannot pass.
                ids[t, :8] = (255 + np.arange(8)*17 + t*13 + repeat*19) % self.experts
        return image, coeff, ids

    def truth(self, point, coeff, ids, compute):
        rounded = round_compute(coeff, compute).astype("f8")
        output, denom = [], []
        for row in range(point["rows"]):
            token, slot = divmod(row, 8) if point["mode"] == 2 else (row, 0)
            expert = int(ids[token, slot]) if ids is not None else 0
            p = expert % self.pool
            a = rounded[token, slot % point["channels"]]
            output.append(a[:4] @ self.sums[p] + a[4]*self.columns[p])
            denom.append(np.abs(a[:4]) @ self.absolute[p] + abs(a[4])*np.abs(self.columns[p]))
        return np.asarray(output), np.asarray(denom)

    def record(self):
        return dict(n=self.n, k=self.k, experts=self.experts, unique_expert_patterns=self.pool,
            expert_pattern="expert modulo pool; remaining planes copied D2D", pool=self.hashes,
            category_sha256=digest(self.category), outlier_index=self.outlier_index,
            oracle="OFFICIAL_GGUF_F64_CATEGORY_SUMS_AND_SEPARATE_OUTLIER_COLUMN")


def storage(image, kind):
    return image.astype("<f4", copy=False) if kind == 1 else bf16_bits(image)


def output(buffer, point):
    value = buffer.read("<f4", (point["rows"], point["n"]+8))
    if not np.all(value[:, point["n"]:].view("<u4") == POISON):
        raise ValueError("output row padding changed")
    return value[:, :point["n"]].copy()
