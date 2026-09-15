"""Shared call ABI with a small, real CUDA/PPU timing adapter for development."""
import ctypes as C
import json
import math
from pathlib import Path

import numpy as np

from quactlize.execution.native import Call, Config as LegacyConfig, Sizes, Arrangement, arrangement, SimtConfig
from dev.gemv_simt.build import sha
from dev.gemv_simt.spec import Config, SCHEMA


class NativeConfig(SimtConfig):
    def __init__(self, c):
        super().__init__(c.variant, c.columns, c.warps, c.values, c.split)


def checked(rc, operation):
    if rc:
        raise RuntimeError(f"{operation}: runtime/ABI rc={rc}")


class Runtime:
    def __init__(self, sdk, platform):
        sdk = Path(sdk)
        path = sdk / ("lib/libhggc_wrapper.so" if platform == "ppu" else "lib64/libcudart.so")
        self.lib = C.CDLL(str(path.resolve(strict=True)), mode=C.RTLD_GLOBAL)
        prefix = "hggc" if platform == "ppu" else "cuda"
        bindings = {
            "Malloc": [C.POINTER(C.c_void_p), C.c_size_t], "Free": [C.c_void_p],
            "Memcpy": [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int],
            "MemcpyAsync": [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_void_p],
            "MemsetAsync": [C.c_void_p, C.c_int, C.c_size_t, C.c_void_p],
            "StreamCreateWithFlags": [C.POINTER(C.c_void_p), C.c_uint],
            "StreamSynchronize": [C.c_void_p], "StreamDestroy": [C.c_void_p],
            "StreamBeginCapture": [C.c_void_p, C.c_int],
            "StreamEndCapture": [C.c_void_p, C.POINTER(C.c_void_p)],
            "GraphInstantiateWithFlags": [C.POINTER(C.c_void_p), C.c_void_p, C.c_uint64],
            "GraphLaunch": [C.c_void_p, C.c_void_p], "GraphDestroy": [C.c_void_p],
            "GraphExecDestroy": [C.c_void_p], "EventCreate": [C.POINTER(C.c_void_p)],
            "EventRecord": [C.c_void_p, C.c_void_p], "EventSynchronize": [C.c_void_p],
            "EventElapsedTime": [C.POINTER(C.c_float), C.c_void_p, C.c_void_p],
            "EventDestroy": [C.c_void_p],
            "DeviceGetAttribute": [C.POINTER(C.c_int), C.c_int, C.c_int],
        }
        for name, signature in bindings.items():
            fn = getattr(self.lib, prefix + name)
            fn.argtypes, fn.restype = signature, C.c_int
            setattr(self, name, fn)
        self.stream = C.c_void_p()
        checked(self.StreamCreateWithFlags(C.byref(self.stream), 1), "create stream")
        self.allocations = []

    def allocate(self, size):
        p = C.c_void_p()
        checked(self.Malloc(C.byref(p), max(1, size)), "allocate")
        self.allocations.append(p.value)
        return p.value

    def upload(self, a):
        a = np.ascontiguousarray(a)
        if not a.size:
            return None
        p = self.allocate(a.nbytes)
        self.copy(p, a)
        return p

    def copy(self, p, a):
        a = np.ascontiguousarray(a)
        # Explicitly order fixture writes with the nonblocking consumer.
        # Keep pageable source storage alive through completion. This is
        # untimed setup, never an inference-side host wait.
        checked(self.MemcpyAsync(p, a.ctypes.data, a.nbytes, 1, self.stream), "same-stream H2D")
        self.sync()

    def fill(self, p, n, byte=0xa5):
        checked(self.MemsetAsync(p, byte, n, self.stream), "poison")

    def sync(self):
        checked(self.StreamSynchronize(self.stream), "synchronize")

    def download(self, p, n):
        result = np.empty(n, dtype=np.uint8)
        checked(self.Memcpy(result.ctypes.data, p, n, 2), "D2H")
        return result

    def attribute(self, code):
        value = C.c_int()
        checked(self.DeviceGetAttribute(C.byref(value), code, 0), "device attribute")
        return value.value

    def release_after(self, count):
        self.sync()
        for p in reversed(self.allocations[count:]):
            checked(self.Free(p), "free")
        del self.allocations[count:]

    def close(self):
        self.release_after(0)
        checked(self.StreamDestroy(self.stream), "destroy stream")


class Library:
    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        path = directory / self.manifest["library"]
        if self.manifest["schema"] != SCHEMA or sha(path) != self.manifest["library_sha256"]:
            raise ValueError("candidate manifest/library identity differs")
        self.lib = C.CDLL(str(path.resolve()), mode=C.RTLD_LOCAL)
        self.query = self.lib.quactlize_kpack_simt_query_v1
        self.query.argtypes, self.query.restype = [C.POINTER(Call), C.POINTER(NativeConfig),
            C.POINTER(Arrangement), C.POINTER(Sizes)], C.c_int
        self.run = self.lib.quactlize_kpack_simt_run_v1
        self.run.argtypes, self.run.restype = self.query.argtypes[:-1], C.c_int

    def probe(self):
        fn = self.lib.simt_candidate_probe
        fn.argtypes, fn.restype = [C.POINTER(C.c_int), C.c_char_p], C.c_int
        info, name = (C.c_int*6)(), C.create_string_buffer(256)
        checked(fn(info, name), "same-image launch/identity probe")
        return dict(zip(("ordinal", "sm", "l2_bytes", "warp", "major", "minor"), list(info))) | {
            "name": name.value.decode(), "marker": "PASS"}

    def candidates(self, q):
        return [Config(**{k: r[k] for k in Config.__dataclass_fields__})
                for r in self.manifest["configs"][str(q)]]

    def prepare(self, call, config):
        f, a, s = NativeConfig(config), arrangement(call.qtype), Sizes()
        checked(self.query(C.byref(call), C.byref(f), C.byref(a), C.byref(s)), "SIMT query")
        if s.workspace_bytes > call.workspace_bytes:
            raise ValueError("candidate workspace capacity differs")
        def invoke():
            return self.run(C.byref(call), C.byref(f), C.byref(a))
        return invoke


class Graph:
    def __init__(self, rt, calls):
        self.rt, self.calls = rt, len(calls)
        self.graph, self.instance = C.c_void_p(), C.c_void_p()
        self.events = [C.c_void_p(), C.c_void_p()]
        for event in self.events:
            checked(rt.EventCreate(C.byref(event)), "create timer")
        rt.sync()
        checked(rt.StreamBeginCapture(rt.stream, 0), "capture begin")
        for call in calls:
            checked(call(), "captured complete call")
        checked(rt.StreamEndCapture(rt.stream, C.byref(self.graph)), "capture end")
        checked(rt.GraphInstantiateWithFlags(C.byref(self.instance), self.graph, 0), "instantiate")
        checked(rt.GraphLaunch(self.instance, rt.stream), "excluded upload/first replay")
        rt.sync()

    def sample(self):
        rt = self.rt
        checked(rt.EventRecord(self.events[0], rt.stream), "start timer")
        checked(rt.GraphLaunch(self.instance, rt.stream), "replay")
        checked(rt.EventRecord(self.events[1], rt.stream), "end timer")
        checked(rt.EventSynchronize(self.events[1]), "timer wait")
        ms = C.c_float()
        checked(rt.EventElapsedTime(C.byref(ms), *self.events), "timer interval")
        value = ms.value * 1000 / self.calls
        if not math.isfinite(value) or value <= 0:
            raise ValueError("nonfinite or nonpositive sample")
        return value

    def close(self):
        rt = self.rt
        rt.sync()
        checked(rt.GraphExecDestroy(self.instance), "destroy instance")
        checked(rt.GraphDestroy(self.graph), "destroy graph")
        for event in self.events:
            checked(rt.EventDestroy(event), "destroy timer")
