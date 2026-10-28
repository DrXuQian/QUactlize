"""Product correctness-test support; performance runners are on experiments/ppu-tuning."""
import ctypes as C
import math
import numpy as np
from quactlize.runtime.native import checked
from ci.support.weights import activation_values
CONFIGS = [(c, w, s) for c in (16, 32) for w in (4, 8) for s in (1, 4)]


def plan(real, dense_real=False):
    # Every format: small M, empty experts, indexed broadcast and per-slot A.
    weights = [
        dict(
            q=q,
            n=256,
            k=512,
            e=4,
            cases=[dict(mode=0, rows=m, channels=1, topk=1) for m in (1, 7)]
            + [
                dict(mode=1, rows=6, channels=1, topk=1),
                dict(mode=2, rows=8, channels=1, topk=2),
                dict(mode=2, rows=8, channels=2, topk=2),
            ],
        )
        for q in range(10, 15)
    ]
    if real:
        weights += [
            dict(
                q=q,
                n=n,
                k=k,
                e=256,
                cases=[
                    dict(mode=2, rows=t * 8, channels=ch, topk=8)
                    for t in (1, 4, 16)
                    for ch in (1, 8)
                ],
            )
            for q, n, k in ((12, 512, 2048), (13, 2048, 512))
        ]
    if dense_real:
        weights += [
            dict(
                q=q,
                n=n,
                k=k,
                e=1,
                cases=[dict(mode=0, rows=m, channels=1, topk=1) for m in (1, 4)],
            )
            for q in range(10, 15)
            for n, k in (
                (1024, 5120),
                (5120, 8192),
                (5120, 25600),
                (8192, 5120),
                (25600, 5120),
            )
        ]
    return weights


class Resources:
    def __init__(self, sdk):
        self.sdk = sdk
        self.allocations = []
        self.events = []
        self.stream = C.c_void_p()
        bindings = {
            "hggcMemsetAsync": [C.c_void_p, C.c_int, C.c_size_t, C.c_void_p],
            "hggcStreamCreateWithFlags": [C.POINTER(C.c_void_p), C.c_uint],
            "hggcStreamDestroy": [C.c_void_p],
            "hggcEventCreate": [C.POINTER(C.c_void_p)],
            "hggcEventRecord": [C.c_void_p, C.c_void_p],
            "hggcEventSynchronize": [C.c_void_p],
            "hggcEventElapsedTime": [C.POINTER(C.c_float), C.c_void_p, C.c_void_p],
            "hggcEventDestroy": [C.c_void_p],
        }
        for name, args in bindings.items():
            fn = getattr(sdk.lib, name)
            fn.argtypes, fn.restype = args, C.c_int
        checked(
            sdk.lib.hggcStreamCreateWithFlags(C.byref(self.stream), 1), "stream create"
        )
        for _ in range(2):
            p = C.c_void_p()
            checked(sdk.lib.hggcEventCreate(C.byref(p)), "event create")
            self.events.append(p)

    def alloc(self, bytes_):
        p = self.sdk.allocate(bytes_)
        self.allocations.append(p)
        return p

    def upload(self, a):
        p = self.sdk.upload(a.tobytes())
        self.allocations.append(p)
        return p

    def fill(self, pointer, byte, size):
        # Device memset need not finish before its host call returns. Queue
        # poison on the consumer's nonblocking stream, outside timed graphs.
        checked(
            self.sdk.lib.hggcMemsetAsync(pointer, byte, size, self.stream),
            "same-stream fixture poison",
        )

    def samples(self, fn, count):
        samples = []
        for _ in range(count):
            checked(
                self.sdk.lib.hggcEventRecord(self.events[0], self.stream),
                "record begin",
            )
            checked(fn(), "enqueue")
            checked(
                self.sdk.lib.hggcEventRecord(self.events[1], self.stream), "record end"
            )
            checked(
                self.sdk.lib.hggcEventSynchronize(self.events[1]), "timing completion"
            )
            elapsed = C.c_float()
            checked(
                self.sdk.lib.hggcEventElapsedTime(C.byref(elapsed), *self.events),
                "elapsed",
            )
            if not math.isfinite(elapsed.value) or elapsed.value <= 0:
                raise ValueError("invalid timing")
            samples.append(elapsed.value * 1000)
        return samples

    def close(self):
        self.sdk.synchronize(self.stream)
        for p in reversed(self.allocations):
            self.sdk.free(p)
        for p in self.events:
            checked(self.sdk.lib.hggcEventDestroy(p), "destroy event")
        checked(self.sdk.lib.hggcStreamDestroy(self.stream), "destroy stream")


def fixture(weights, case):
    m, mode = case["rows"], case["mode"]
    n, k, e = weights.n, weights.k, weights.experts
    topk, channels = case["topk"], case["channels"]
    ids = offsets = None
    if mode == 0:
        expert = np.zeros(m, dtype=np.int32)
        arows = np.arange(m)
        acount = m
    elif mode == 1:
        rows = np.array([2, 0, 3, 1], dtype=np.int32)
        offsets = np.r_[0, rows.cumsum()].astype(np.int32)
        expert = np.repeat(np.arange(e), rows)
        arows = np.arange(m)
        acount = m
    else:
        tokens = m // topk
        ids = np.stack(
            [(np.arange(topk) * 17 + t * 13) % e for t in range(tokens)]
        ).astype(np.int32)
        expert = ids.reshape(-1)
        arows = np.arange(m) // topk * channels + np.arange(m) % topk % channels
        acount = tokens * channels
    values = activation_values(np.arange(acount))
    a = values[:, weights.categories[0]].astype("<f2")
    golden = np.stack([values[arows[r]] @ weights.sums[expert[r]] for r in range(m)])
    denom = np.stack(
        [np.abs(values[arows[r]]) @ weights.abs_sums[expert[r]] for r in range(m)]
    )
    return dict(
        a=a,
        expert=expert,
        arows=arows,
        ids=ids,
        offsets=offsets,
        golden=golden,
        denom=denom,
    )


def compare(got, data):
    if not np.isfinite(got).all():
        raise ValueError("nonfinite output / unwritten poison")
    error = float(
        np.max(
            np.abs(got.astype("f8") - data["golden"]) / np.maximum(data["denom"], 1e-30)
        )
    )
    if error >= 0.005:
        raise ValueError(f"independent GGUF dot oracle failed: {error:.6g}")
    return error


def metadata_oracle(units, q, n, k, e):
    from reference import gguf_kpack as ref

    spec = ref.SPECS[q]
    sb = k // 256
    packed = units.reshape(
        e, sb // spec.superblocks_per_unit, n, spec.superblocks_per_unit, spec.sb_bytes
    )
    packed = packed.transpose(0, 1, 3, 2, 4).reshape(e, sb, n, spec.sb_bytes)
    d = packed[..., :2].copy().view("<f2")[..., 0]
    dm = (
        packed[..., 2:4].copy().view("<f2")[..., 0]
        if spec.has_min
        else np.zeros_like(d)
    )
    scale = np.empty((e, sb * spec.groups, n), dtype="<f2")
    zero = np.empty_like(scale)
    for g in range(spec.groups):
        codes = []
        for which, bits in ((0, spec.scale_bits), (1, spec.min_bits)):
            if not bits:
                codes.append(np.zeros_like(d, dtype="i2"))
                continue
            bit = ref._unit_bit(spec, g, which)
            byte, shift = divmod(bit, 8)
            word = packed[..., byte].astype("u2")
            if shift + bits > 8:
                word |= packed[..., byte + 1].astype("u2") << 8
            codes.append(((word >> shift) & ((1 << bits) - 1)).astype("i2"))
        sc, mn = codes
        if q == 11:
            sc = sc - 32
        if q == 14:
            sc = np.where(sc >= 128, sc - 256, sc)
        s = (d.astype("f4") * sc).astype("<f2")
        z = (
            (-(dm.astype("f4") * mn)).astype("<f2")
            if spec.has_min
            else np.zeros_like(s)
        )
        zmul = {10: 0, 11: -4, 12: 8, 13: 8, 14: -24}[q]
        if zmul:
            z = (z.astype("f4") + np.float32(zmul) * s.astype("f4")).astype("<f2")
        scale[:, g :: spec.groups] = s
        zero[:, g :: spec.groups] = z
    return scale, zero
