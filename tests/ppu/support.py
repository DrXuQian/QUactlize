"""Small device harness shared by public-pack/dense and low-level MoE tests.

The oracle decodes original GGUF, not device-produced planes. No timers.
"""
import ctypes as C

import numpy as np

from ci.support.fixture import compare, planes, raw_weight, round_compute
from ci.support.dense import Arrangement, Candidate, Info, Options, Params, Sizes
from quactlize.runtime.native import checked
from quactlize.execution.native import arrangement

QTYPES = (8, 10, 11, 12, 13, 14)
DECODE_M = tuple(range(1, 9))
DENSE_M = DECODE_M + (9, 16, 128, 256)


def bind(library, name, args, result=C.c_int):
    fn = getattr(library, name)
    fn.argtypes, fn.restype = args, result
    return fn


class Buffer:
    def __init__(self, arena, size):
        self.arena, self.size = arena, int(size)
        self.base = arena.sdk.allocate(self.size + 512)
        self.ptr = self.base + 256
        arena.buffers.append(self)
        assert self.ptr % 256 == 0
        arena.sdk.fill(self.base, 0xA5, self.size + 512)

    def write(self, value):
        value = np.ascontiguousarray(value)
        assert value.nbytes == self.size
        checked(self.arena.sdk.lib.hggcMemcpy(self.ptr, value.ctypes.data, value.nbytes, 1), "test upload")

    def guard(self):
        sdk = self.arena.sdk
        assert sdk.download(self.base, 256) == b"\xa5" * 256
        assert sdk.download(self.ptr + self.size, 256) == b"\xa5" * 256

    def bytes(self):
        self.guard()
        return self.arena.sdk.download(self.ptr, self.size)

    def poison(self):
        self.arena.sdk.fill(self.ptr, 0xff, self.size)


class Arena:
    def __init__(self, sdk):
        self.sdk, self.buffers, self.streams = sdk, [], []

    def buffer(self, size):
        return Buffer(self, size)

    def upload(self, value):
        result = self.buffer(value.nbytes)
        result.write(value)
        return result

    def stream(self):
        stream = C.c_void_p()
        checked(self.sdk.lib.hggcStreamCreateWithFlags(C.byref(stream), 1), "CI stream")
        self.streams.append(stream)
        return stream

    def sync(self):
        # Setup H2D uses the default stream; make its dependency explicit.
        self.sdk.synchronize(None)
        for stream in self.streams:
            self.sdk.synchronize(stream)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.sync()
        for buf in reversed(self.buffers):
            self.sdk.free(buf.base)
        for stream in reversed(self.streams):
            checked(self.sdk.lib.hggcStreamDestroy(stream), "destroy CI stream")


class Public:
    def __init__(self, package):
        self.lib = C.CDLL(str(package / "libquactlize_ppu.so"), mode=C.RTLD_LOCAL)
        def api(name, args, result=C.c_int):
            return bind(self.lib, "quactlize_ppu_" + name, args, result)
        self.canonical = api("kpack_canonical_arrangement_v1", [C.c_int, C.POINTER(Arrangement)])
        self.sizes = api("kpack_sizes_for_arrangement_v1", [C.c_int] * 4 + [C.POINTER(Arrangement), C.POINTER(Sizes)])
        self.pack = api("prepare_fully_quantized_dev_for_arrangement_v2",
                        [C.c_void_p] * 4 + [C.c_int] * 4 + [C.POINTER(Arrangement), C.c_void_p])
        self.pack_pair = api("prepare_gate_up_dev_for_arrangement_v1",
                             [C.c_void_p] * 5 + [C.c_int] * 4 + [C.POINTER(Arrangement), C.c_void_p])
        self.supported = api("dense_supported_v1", [C.POINTER(Params)])
        self.create = api("dense_create_with_options_v1", [C.c_int, C.POINTER(Options), C.POINTER(C.c_void_p)])
        prep = [C.c_void_p, C.POINTER(Params)] + [C.c_void_p] * 3
        self.prepare = api("dense_prepare_v1", prep + [C.POINTER(C.c_void_p), C.POINTER(Info)])
        self.candidate = api("dense_prepare_candidate_v1", prep + [C.POINTER(Candidate), C.POINTER(C.c_void_p), C.POINTER(Info)])
        self.run = api("dense_run_v1", [C.c_void_p] * 4 + [C.c_uint64, C.c_void_p])
        self.destroy = api("dense_destroy_v1", [C.c_void_p], None)
        self.error = api("dense_error_v1", [], C.c_char_p)

    def check(self, rc):
        if rc:
            raise AssertionError(f"public dense rc={rc}: {self.error().decode(errors='replace')}")


def packed(arena, lib, q, n=256, k=512, experts=1, pair=False, bit_patterns=False):
    """Pack on device, compare every byte, then reuse those planes in GEMM."""
    raw, golden, expected = [], [], {name: [] for name in ("low", "high", "units")}
    gates, ups = [], []
    for expert in range(experts):
        source, gold = raw_weight(q, n, k, 913 + expert * 17 + q)
        if bit_patterns:
            source = np.random.default_rng(1201 + expert + q).integers(0, 256, source.shape, dtype='u1')
        if pair:
            up, up_gold = raw_weight(q, n, k, 719 + expert * 13 + q)
            if bit_patterns:
                up = np.random.default_rng(1401 + expert + q).integers(0, 256, up.shape, dtype='u1')
            gates.append(source); ups.append(up)
            source, gold = np.concatenate((source, up)), np.concatenate((gold, up_gold))
        raw.append(source); golden.append(gold)
        # Arbitrary FP16 metadata bits (including NaNs) must round-trip in the
        # pack-only gate. Derived arithmetic planes are not used in that gate.
        with np.errstate(all='ignore'):
            p = planes(source, q)
        for name in expected:
            expected[name].append(p[name].tobytes())
    arr, sizes = Arrangement(), Sizes()
    checked(lib.canonical(q, C.byref(arr)), "canonical arrangement")
    assert bytes(arr) == bytes(arrangement(q))
    final_n = n * (2 if pair else 1)
    checked(lib.sizes(final_n, k, experts, q, C.byref(arr), C.byref(sizes)), "canonical sizes")
    assert sizes.raw_bytes == sum(x.nbytes for x in raw)
    assert sizes.low_bytes + sizes.high_bytes + sizes.units_bytes == sizes.raw_bytes
    buffers = {name: arena.buffer(getattr(sizes, name + "_bytes")) if getattr(sizes, name + "_bytes") else None
               for name in expected}
    pointers = [buffers[name].ptr if buffers[name] else None for name in expected]
    stream = arena.stream()
    if pair:
        sources = [arena.upload(np.stack(values)) for values in (gates, ups)]
        arena.sync()
        checked(lib.pack_pair(*[s.ptr for s in sources], *pointers, n, k, experts, q, C.byref(arr), stream), "pair pack")
    else:
        source = arena.upload(np.stack(raw))
        arena.sync()
        checked(lib.pack(source.ptr, *pointers, n, k, experts, q, C.byref(arr), stream), "pack")
        assert lib.pack(source.ptr, source.ptr, pointers[1], pointers[2], n, k, experts, q, C.byref(arr), stream) != 0
        bad = Arrangement.from_buffer_copy(arr); bad.mapping_id ^= 1
        assert lib.pack(source.ptr, *pointers, n, k, experts, q, C.byref(bad), stream) == 38
    arena.sync()
    for name, buffer in buffers.items():
        want = b"".join(expected[name])
        assert (buffer.bytes() if buffer else b"") == want, f"q{q} {name} packed byte mismatch"
    # Never expose the finite oracle as a reference for arbitrary bit patterns.
    return arr, sizes, pointers, None if bit_patterns else np.stack(golden)


def check_output(got, source, weight, compute):
    a = round_compute(source, "bf16" if compute else "f16").astype("f8")
    w = weight.astype("f8")
    return compare(got, a @ w.T, np.abs(a) @ np.abs(w).T)
