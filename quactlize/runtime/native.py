"""Resident-pointer backend for compile-only K-pack parent modules.

The embedding application supplies A/weight/output and grouped row pointers.
This helper owns only reusable workspace. The optional SDK allocation helpers
are for initialization and device gates, never part of a prepared launch.
"""

import ctypes as C
from functools import lru_cache
from pathlib import Path

from .compiler import LIBRARIES, sha
from .identity import ROUTES, digest


@lru_cache(maxsize=16)
def _sdk_digest(signature):
    return digest([(relative, sha(path)) for path, relative, _, _ in signature])


def sdk_identity(sdk):
    root = Path(sdk).resolve()
    paths = [root / "bin/hgcc", root / "bin/hgobjdump"] + [
        root / "lib" / f"lib{name}.so" for name in LIBRARIES
    ]
    # Hash once per SDK revision, not once per weight's warmup context.
    signature = tuple(
        (str(p), str(p.relative_to(root)), p.stat().st_mtime_ns, p.stat().st_size)
        for p in paths
    )
    return _sdk_digest(signature)


class Call(C.Structure):
    _fields_ = (
        [("version", C.c_uint32), ("size", C.c_uint32)]
        + [
            (x, C.c_int32)
            for x in ("m", "n", "k", "experts", "group_size", "device", "compute_units")
        ]
        + [("mapping_id", C.c_uint64)]
        + [
            (x, C.c_void_p)
            for x in (
                "a",
                "low",
                "high",
                "metadata",
                "zero",
                "output",
                "rows_host",
                "rows_device",
                "offsets_device",
                "workspace",
            )
        ]
        + [("workspace_bytes", C.c_uint64), ("stream", C.c_void_p)]
    )


class Recipe(C.Structure):
    _fields_ = [("version", C.c_uint32), ("size", C.c_uint32)] + [
        (x, C.c_int32) for x in ("algorithm", "split", "grid")
    ]


class Resources(C.Structure):
    _fields_ = [
        ("version", C.c_uint32),
        ("size", C.c_uint32),
        ("workspace_bytes", C.c_uint64),
        ("shared_bytes", C.c_uint64),
    ] + [(x, C.c_int32) for x in ("occupancy", "runtime_status", "cutlass_status")]


class Identity(C.Structure):
    _fields_ = (
        [("version", C.c_uint32), ("size", C.c_uint32)]
        + [
            (x, C.c_int32)
            for x in (
                "qtype",
                "route",
                "tm",
                "tn",
                "tk",
                "wm",
                "wn",
                "stages",
                "ap",
                "delivery_n",
            )
        ]
        + [
            ("mapping_id", C.c_uint64),
            ("parent", C.c_char_p),
            ("build_key", C.c_char_p),
        ]
    )


def checked(status, operation):
    if status:
        raise RuntimeError(f"{operation} failed: status={status}")


class SDK:
    def __init__(self, path):
        self.lib = C.CDLL(
            str(Path(path) / "lib/libhggc_wrapper.so"), mode=C.RTLD_GLOBAL
        )
        bindings = {
            "hggcMalloc": ([C.POINTER(C.c_void_p), C.c_size_t], C.c_int),
            "hggcFree": ([C.c_void_p], C.c_int),
            "hggcMemcpy": ([C.c_void_p, C.c_void_p, C.c_size_t, C.c_int], C.c_int),
            "hggcMemset": ([C.c_void_p, C.c_int, C.c_size_t], C.c_int),
            "hggcStreamSynchronize": ([C.c_void_p], C.c_int),
            "hggcStreamIsCapturing": ([C.c_void_p, C.POINTER(C.c_int)], C.c_int),
        }
        for name, (args, result) in bindings.items():
            fn = getattr(self.lib, name)
            fn.argtypes = args
            fn.restype = result

    def allocate(self, size):
        pointer = C.c_void_p()
        checked(self.lib.hggcMalloc(C.byref(pointer), size), "allocation")
        return pointer.value

    def free(self, pointer):
        if pointer:
            checked(self.lib.hggcFree(pointer), "free")

    def upload(self, data):
        data = bytes(data)
        pointer = self.allocate(len(data))
        host = C.create_string_buffer(data)
        checked(self.lib.hggcMemcpy(pointer, host, len(data), 1), "H2D")
        return pointer

    def download(self, pointer, size):
        host = C.create_string_buffer(size)
        checked(self.lib.hggcMemcpy(host, pointer, size, 2), "D2H")
        return host.raw

    def fill(self, pointer, byte, size):
        checked(self.lib.hggcMemset(pointer, byte, size), "memset")

    def synchronize(self, stream):
        checked(self.lib.hggcStreamSynchronize(stream), "synchronize")


class Module:
    def __init__(self, record):
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("module payload differs from its build receipt")
        self.lib = C.CDLL(record["path"], mode=C.RTLD_LOCAL)
        self.record = record
        bindings = {
            "identity": ([], C.POINTER(Identity)),
            "device": (
                [C.c_char_p, C.c_int, C.POINTER(C.c_int32), C.POINTER(C.c_int32)],
                C.c_int,
            ),
            "query": (
                [C.POINTER(Call), C.POINTER(Recipe), C.POINTER(Resources)],
                C.c_int,
            ),
            "prepare": (
                [C.POINTER(Call), C.POINTER(Recipe), C.POINTER(C.c_void_p)],
                C.c_int,
            ),
            "run": ([C.c_void_p, C.c_void_p], C.c_int),
            "destroy": ([C.c_void_p], None),
            "measure": (
                [C.c_void_p, C.c_void_p, C.c_int, C.POINTER(C.c_double)],
                C.c_int,
            ),
        }
        for short, (args, result) in bindings.items():
            fn = getattr(self.lib, f"quactlize_kpack_{short}_v1")
            fn.argtypes = args
            fn.restype = result
            setattr(self, short, fn)
        identity = self.identity().contents
        p = record["parent"]
        if (
            identity.version != 1
            or identity.size != C.sizeof(Identity)
            or identity.parent.decode() != p["symbol"]
            or identity.build_key.decode() != record["key"]
            or identity.route != ROUTES.index(p["route"])
            or any(
                getattr(identity, f) != p[f]
                for f in ("qtype", "tm", "tn", "tk", "wm", "wn", "stages", "ap")
            )
            or identity.delivery_n != p["dn"]
        ):
            raise ValueError("module ABI/parent identity differs")

    def device_identity(self):
        name = C.create_string_buffer(256)
        device = C.c_int32()
        cu = C.c_int32()
        checked(
            self.device(name, len(name), C.byref(device), C.byref(cu)),
            "device identity",
        )
        return dict(
            device=name.value.decode(), compute_units=cu.value, ordinal=device.value
        )
