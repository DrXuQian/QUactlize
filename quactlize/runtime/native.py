"""Resident-pointer backend for compile-only K-pack parent modules.

The embedding application supplies A/weight/output and grouped row pointers.
This helper owns only reusable workspace. The optional SDK allocation helpers
are for initialization and device gates, never part of a prepared launch.
"""

import ctypes as C
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .compiler import LIBRARIES, sha
from .tuning import GROUPS, ROUTES, UnsupportedTactic, digest


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


@dataclass
class Prepared:
    module: Module
    pointer: C.c_void_p
    call: Call
    rows: object


class NativeBackend:
    def __init__(self, sdk, records, buffers, correctness, stream=0, catalog=None):
        if not records:
            raise ValueError("compile/load parents before constructing a runtime")
        installed_sdk = sdk_identity(sdk)
        if any(r["identity"]["sdk"] != installed_sdk for r in records):
            raise ValueError("runtime SDK differs from compiled parent SDK")
        self.sdk = SDK(sdk)
        self.modules = {r["parent"]["symbol"]: Module(r) for r in records}
        if not self.modules:
            raise ValueError("compile/load parents before constructing a runtime")
        devices = [m.device_identity() for m in self.modules.values()]
        if any(d != devices[0] for d in devices):
            raise ValueError("modules do not target one device")
        identities = [m.record["identity"] for m in self.modules.values()]
        if any(i != identities[0] for i in identities):
            raise ValueError("parent modules have different source/SDK contracts")
        self.identity = devices[0] | {k: identities[0][k] for k in ("sdk", "kernel")}
        if catalog is not None and any(
            catalog.get(r["parent"]["symbol"]) != r["parent"] for r in records
        ):
            raise ValueError("compiled parent differs from the admitted catalog")
        self.identity["inventory"] = digest(
            (
                identities[0],
                catalog if catalog is not None else sorted(r["key"] for r in records),
                [
                    (name, sha(Path(__file__).with_name(name)))
                    for name in ("native.py", "tuning.py", "candidates.py")
                ],
            )
        )
        self.buffers = dict(buffers)
        self.correctness = correctness
        self.stream = stream
        self.workspace = 0
        self.workspace_bytes = 0

    def is_capturing(self):
        status = C.c_int()
        checked(
            self.sdk.lib.hggcStreamIsCapturing(self.stream, C.byref(status)),
            "capture query",
        )
        return bool(status.value)

    def arguments(self, request, tactic):
        module = self.modules.get(tactic.parent)
        if module is None:
            raise UnsupportedTactic("parent has not been compiled/loaded")
        p = module.record["parent"]
        if request.qtype != p["qtype"] or request.route != p["route"]:
            raise UnsupportedTactic("parent route/format differs")
        rows = (C.c_int32 * len(request.rows))(*request.rows) if request.rows else None
        call = Call(
            version=1,
            size=C.sizeof(Call),
            m=request.m,
            n=request.n,
            k=request.k,
            experts=len(request.rows) if rows is not None else 1,
            group_size=GROUPS[request.qtype],
            device=self.identity["ordinal"],
            compute_units=self.identity["compute_units"],
            mapping_id=request.mapping,
            rows_host=C.cast(rows, C.c_void_p) if rows is not None else None,
            workspace=self.workspace,
            workspace_bytes=self.workspace_bytes,
            stream=self.stream,
            **{
                k: self.buffers.get(k, 0)
                for k in (
                    "a",
                    "low",
                    "high",
                    "metadata",
                    "zero",
                    "output",
                    "rows_device",
                    "offsets_device",
                )
            },
        )
        grid = 0
        persistent = tactic.algorithm in ("PERSISTENT", "GROUPED_PERSISTENT")
        allowed = (
            {"GROUPED_PERSISTENT", "GROUPED_NONPERSISTENT"}
            if request.grouped
            else (
                {f"TC_S{s}" for s in (1, 2, 4, 8)}
                if request.route == "fq-dense"
                else {
                    "PERSISTENT",
                    "NONPERSISTENT",
                    "SPLITK_S2",
                    "SPLITK_S4",
                    "SPLITK_S8",
                }
            )
        )
        if tactic.algorithm not in allowed:
            raise UnsupportedTactic("unknown runtime algorithm")
        if request.route == "fq-dense" and tactic.algorithm != f"TC_S{tactic.split}":
            raise UnsupportedTactic("algorithm/split mismatch")
        if request.route != "fq-dense":
            expected_split = (
                int(tactic.algorithm[-1])
                if tactic.algorithm.startswith("SPLITK_S")
                else 1
            )
            if tactic.split != expected_split:
                raise UnsupportedTactic("algorithm/split mismatch")
        if not persistent and (
            tactic.grid_mode not in ("implicit", "ordinary") or tactic.grid_b != 0
        ):
            raise UnsupportedTactic("ordinary launch cannot have a persistent grid")
        if persistent:
            q = sum(
                (m + p["tm"] - 1) // p["tm"] for m in (request.rows or (request.m,))
            ) * ((request.n + p["tn"] - 1) // p["tn"])
            capacity = self.identity["compute_units"] * tactic.grid_b
            if capacity <= 0 or tactic.grid_mode not in ("capacity", "balanced"):
                raise UnsupportedTactic("missing source-owned grid recipe")
            waves = (q + capacity - 1) // capacity
            grid = (
                min(q, capacity)
                if tactic.grid_mode == "capacity"
                else (q + waves - 1) // waves
            )
        recipe = Recipe(1, C.sizeof(Recipe), int(persistent), tactic.split, grid)
        resources = Resources()
        status = module.query(C.byref(call), C.byref(recipe), C.byref(resources))
        if status in (1, 2):
            raise UnsupportedTactic(f"query rejected: {status}")
        checked(status, "resource query")
        if persistent and tactic.grid_b > resources.occupancy:
            raise UnsupportedTactic("grid recipe exceeds actual kernel occupancy")
        return module, call, recipe, resources, rows

    def admissible(self, request, tactic):
        try:
            self.arguments(request, tactic)
            return True
        except UnsupportedTactic:
            return False

    def prepare(self, request, tactic):
        module, call, recipe, resources, rows = self.arguments(request, tactic)
        if resources.workspace_bytes > self.workspace_bytes:
            self.synchronize()
            self.sdk.free(self.workspace)
            self.workspace = self.sdk.allocate(resources.workspace_bytes)
            self.workspace_bytes = resources.workspace_bytes
        call.workspace, call.workspace_bytes = self.workspace, self.workspace_bytes
        pointer = C.c_void_p()
        status = module.prepare(C.byref(call), C.byref(recipe), C.byref(pointer))
        if status in (1, 2):
            raise UnsupportedTactic(f"prepare rejected: {status}")
        checked(status, "prepare")
        return Prepared(module, pointer, call, rows)

    def check(self, handle):
        self.run(handle)
        self.synchronize()
        if self.correctness is None or not self.correctness(self):
            raise RuntimeError("candidate correctness check failed")

    def run(self, handle):
        checked(handle.module.run(handle.pointer, self.stream), "kernel launch")

    def measure(self, handle, repeats):
        value = C.c_double()
        checked(
            handle.module.measure(handle.pointer, self.stream, repeats, C.byref(value)),
            "kernel timing",
        )
        return value.value

    def synchronize(self):
        self.sdk.synchronize(self.stream)

    def close(self, handle):
        handle.module.destroy(handle.pointer)

    def release(self):
        self.synchronize()
        self.sdk.free(self.workspace)
        self.workspace, self.workspace_bytes = 0, 0
