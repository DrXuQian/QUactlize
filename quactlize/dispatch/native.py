"""Offline device-gate bindings for the production C++ selector."""

import ctypes as C
from pathlib import Path

from quactlize.runtime.native import Call, checked


class Request(C.Structure):
    _fields_ = (
        [("version", C.c_uint32), ("size", C.c_uint32)]
        + [
            (x, C.c_int32)
            for x in ("qtype", "route", "m", "n", "k", "experts", "max_rows")
        ]
        + [("mapping_id", C.c_uint64)]
    )


class Choice(C.Structure):
    _fields_ = (
        [("version", C.c_uint32), ("size", C.c_uint32)]
        + [(x, C.c_uint64) for x in ("ticket", "workspace_bytes", "shared_bytes")]
        + [
            (x, C.c_int32)
            for x in ("policy", "algorithm", "split", "grid", "device", "compute_units")
        ]
        + [("parent", C.c_char * 192), ("build_key", C.c_char * 65)]
    )


class JitOptions(C.Structure):
    _fields_ = [("version", C.c_uint32), ("size", C.c_uint32)] + [
        (name, C.c_char_p) for name in ("python", "helper", "sdk", "cache")]


class IndexedIO(C.Structure):
    _fields_ = [("version",C.c_uint32),("size",C.c_uint32)] + [
        (name,C.c_int32) for name in ("tokens","topk","channels","reserved")] + [
        (name,C.c_int64) for name in ("ids_stride","a_row_stride","a_token_stride","out_row_stride")] + [
        (name,C.c_void_p) for name in ("ids","a","output","row_ids")]


class Router(C.Structure):
    _fields_ = [("version",C.c_uint32),("size",C.c_uint32)] + [
        (name,C.c_int32) for name in ("use_sigmoid","with_norm","delayed_softmax","reserved")] + [
        ("clamp",C.c_float),("scale",C.c_float)] + [
        (name,C.c_void_p) for name in ("logits","bias","weights")]


class Dispatch:
    def __init__(self, root, jit=None):
        self.lib = C.CDLL(
            str(Path(root).resolve() / "libquactlize_kpack_dispatch.so"),
            mode=C.RTLD_LOCAL,
        )
        types = {
            "open": ([C.c_char_p, C.POINTER(C.c_void_p)], C.c_int),
            "query": ([C.c_void_p, C.POINTER(Request), C.POINTER(Choice)], C.c_int),
            "prepare": (
                [C.c_void_p, C.POINTER(Choice), C.POINTER(Call), C.POINTER(C.c_void_p)],
                C.c_int,
            ),
            "run": ([C.c_void_p, C.c_void_p], C.c_int),
            "destroy": ([C.c_void_p], None),
            "close": ([C.c_void_p], None),
            "error": ([], C.c_char_p),
        }
        self.fn = {}
        for name, (args, result) in types.items():
            f = getattr(self.lib, "quactlize_kpack_dispatch_" + name + "_v1")
            f.argtypes, f.restype = args, result
            self.fn[name] = f
        self.runtime = C.c_void_p()
        self.handles = []
        self.chains = []
        checked(
            self.fn["open"](str(Path(root).resolve()).encode(), C.byref(self.runtime)),
            "dispatch open",
        )
        if jit:
            enable = self.lib.quactlize_kpack_dispatch_enable_jit_v1
            enable.argtypes = [C.c_void_p, C.POINTER(JitOptions)]
            enable.restype = C.c_int
            options = JitOptions(1, C.sizeof(JitOptions), *[
                str(Path(jit[name]).resolve()).encode() for name in ("python", "helper", "sdk", "cache")])
            if enable(self.runtime, C.byref(options)):
                self.close()
                raise ValueError("JIT setup: " + self.fn["error"]().decode())

    def query(self, q, route, m, n, k, experts, max_rows, mapping):
        r = Request(1, C.sizeof(Request), q, route, m, n, k, experts, max_rows, mapping)
        choice = Choice()
        rc = self.fn["query"](self.runtime, C.byref(r), C.byref(choice))
        if rc == 1:
            return None
        if rc:
            raise ValueError("native query: " + self.fn["error"]().decode())
        return choice

    def prepare(self, choice, call, indexed=None):
        h = C.c_void_p()
        rc = self.fn["prepare"](
            self.runtime, C.byref(choice), C.byref(call), C.byref(h)
        )
        if rc:
            raise ValueError("native prepare: " + self.fn["error"]().decode())
        self.handles.append(h)
        if indexed is not None:
            bind=self.lib.quactlize_kpack_dispatch_bind_llama_indexed_v1
            bind.argtypes=[C.c_void_p,C.POINTER(IndexedIO)]; bind.restype=C.c_int
            rc=bind(h,C.byref(indexed))
            if rc:
                self.fn["destroy"](h); self.handles.pop()
                raise ValueError(f"native indexed binding failed rc={rc}")
        return lambda: self.fn["run"](h, call.stream)

    def chain(self, gate, up, down, stream, router=None):
        create=self.lib.quactlize_kpack_dispatch_moe_create_v1
        create.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.POINTER(C.c_void_p)]
        create.restype=C.c_int
        chain=C.c_void_p()
        rc=create(gate,up,down,C.byref(chain))
        if rc: raise ValueError(f"native MoE chain creation failed rc={rc}: "+self.fn['error']().decode())
        self.chains.append(chain)
        if router is not None:
            run=self.lib.quactlize_kpack_dispatch_moe_run_router_v1
            run.argtypes=[C.c_void_p,C.POINTER(Router),C.c_void_p]; run.restype=C.c_int
            return lambda: run(chain,C.byref(router),stream)
        run=self.lib.quactlize_kpack_dispatch_moe_run_v1
        run.argtypes=[C.c_void_p,C.c_void_p]; run.restype=C.c_int
        return lambda: run(chain,stream)

    def close(self):
        if self.chains:
            destroy=self.lib.quactlize_kpack_dispatch_moe_destroy_v1
            destroy.argtypes=[C.c_void_p]; destroy.restype=None
            for chain in self.chains: destroy(chain)
            self.chains.clear()
        for h in self.handles:
            self.fn["destroy"](h)
        self.handles.clear()
        if self.runtime:
            self.fn["close"](self.runtime)
            self.runtime = None


def receipt(choice):
    return {
        k: (
            getattr(choice, k).decode()
            if k in ("parent", "build_key")
            else getattr(choice, k)
        )
        for k in (
            "parent",
            "build_key",
            "policy",
            "algorithm",
            "split",
            "grid",
            "shared_bytes",
            "workspace_bytes",
        )
    }
