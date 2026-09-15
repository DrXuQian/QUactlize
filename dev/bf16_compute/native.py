"""Private gate bindings: explicit low-level compute APIs, no dispatcher."""
import ctypes as C
from pathlib import Path

import numpy as np

from quactlize.decode.native_compute import (
    DenseIO, DenseComputeCall, GroupedDeviceCall, GroupedComputeCall, ComputeIdentity, bind_compute,
)
from quactlize.dispatch.native import IndexedIO, Router, MoeFinish
from quactlize.execution.native import (
    Call as SimtCall, Sizes, SimtCallV2, SimtConfig, arrangement, bind_simt_compute,
)
from quactlize.runtime.compiler import sha
from quactlize.runtime.native import Call, Recipe, Resources as QueryResources, checked
from tools.run_kpack_gemv_gate import Resources
from dev.bf16_compute.fixture import bf16_bits, bf16_float


class Projection(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ("version", "size", "shape_size", "stride_size")]
    _fields_ += [("shape_offsets", C.c_uint32 * 3), ("stride_offset", C.c_uint32)]
    _fields_ += [(name, C.c_int32) for name in ("m", "n", "k", "experts", "tile_m", "splits", "device", "reserved")]
    _fields_ += [(name, C.c_void_p) for name in ("a", "output", "partials", "shapes", "outputs", "strides",
        "offsets", "rows", "directory_header", "directory_entries")]
    _fields_ += [("directory_capacity", C.c_uint64), ("workspace", C.c_void_p),
                 ("workspace_bytes", C.c_uint64), ("io", IndexedIO)]


class ProjectionV2(C.Structure):
    _fields_ = [("version", C.c_uint32), ("size", C.c_uint32),
                ("projection", Projection), ("compute_type", C.c_int32)]


class MoePlan(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ("version", "size", "merged", "reserved")]
    _fields_ += [("gate", Projection), ("up", Projection), ("down", Projection), ("router", Router)]


class MoePlanV2(C.Structure):
    _fields_ = [("version", C.c_uint32), ("size", C.c_uint32),
                ("plan", MoePlan), ("compute_type", C.c_int32)]


class MixedPlanV2(C.Structure):
    _fields_ = [("version", C.c_uint32), ("size", C.c_uint32), ("plan", MoePlan),
                ("simt_mask", C.c_uint32), ("compute_type", C.c_int32)]


def load(root, record):
    path = (Path(root) / record["path"]).resolve(strict=True)
    if Path(root).resolve() not in path.parents or sha(path) != record["sha256"]:
        raise ValueError("gate module path/payload differs from manifest")
    return C.CDLL(str(path), mode=C.RTLD_LOCAL)


def function(lib, name, arguments, result=C.c_int):
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = arguments, result
    return fn


class Buffer:
    def __init__(self, r, size):
        self.r, self.size = r, int(size)
        self.base = r.alloc(self.size + 32)
        self.ptr = self.base + 16
        self.poison()

    def poison(self):
        self.r.fill(self.base, 0xa5, self.size + 32)

    def upload(self, data):
        data = np.ascontiguousarray(data)
        if data.nbytes != self.size:
            raise ValueError(f"fixture storage width mismatch {data.nbytes}!={self.size}")
        self.r.sdk.synchronize(self.r.stream)
        checked(self.r.sdk.lib.hggcMemcpy(self.ptr, data.ctypes.data, data.nbytes, 1), "fixture H2D")
        self.r.sdk.synchronize(None)

    def read(self, dtype="<f4", shape=None):
        image = self.r.sdk.download(self.base, self.size + 32)
        if image[:16] != b"\xa5" * 16 or image[-16:] != b"\xa5" * 16:
            raise ValueError("input/output/workspace guard changed")
        data = np.frombuffer(image[16:-16], dtype=dtype)
        return data.reshape(shape) if shape else data

    def guard(self):
        first = self.r.sdk.download(self.base, 16)
        last = self.r.sdk.download(self.ptr + self.size, 16)
        if first != b"\xa5" * 16 or last != first:
            raise ValueError("workspace boundary changed")


class TensorCore:
    def __init__(self, root, record, sdk, r, weights, m, maximum, algorithm=0, split=1, storage=1):
        self.lib = load(root, record)
        self.record, self.r, self.weights = record, r, weights
        self.grouped = record["parent"]["route"].endswith("grouped")
        self.compute = record["spec"]["compute"]
        self.tag = int(self.compute == "bf16")
        self.storage = storage
        self.identity, self.query, self.prepare, self.run_fn, self.destroy = bind_compute(self.lib, self.grouped)
        identity = self.identity().contents
        parent = identity.parent.contents
        if (identity.version != (3 if self.grouped else 2) or identity.size != C.sizeof(ComputeIdentity) or
                identity.compute_type != self.tag or parent.build_key.decode() != record["key"] or
                parent.qtype != weights.q or any(getattr(parent, name) != record["parent"][name]
                    for name in ("tm", "tn", "tk", "wm", "wn", "stages", "ap"))):
            raise ValueError("explicit compute/parent identity mismatch")
        device_name = C.create_string_buffer(256)
        self.device, self.cu = C.c_int32(), C.c_int32()
        suffix = "device_v1" if self.grouped else "decode_dense_device_v1"
        dev = function(self.lib, "quactlize_kpack_" + suffix,
                       [C.c_char_p, C.c_int, C.POINTER(C.c_int32), C.POINTER(C.c_int32)])
        checked(dev(device_name, 256, C.byref(self.device), C.byref(self.cu)), "module device identity")
        self.arr = arrangement(weights.q)
        self.planes = {name: r.upload(value) if value.size else None for name, value in weights.planes.items()}
        sdk.synchronize(None)
        self.offsets = Buffer(r, (weights.experts + 1) * 4) if self.grouped else None
        width = 2 if self.grouped or storage == 2 else 4
        self.a, self.output = Buffer(r, m * weights.k * width), Buffer(r, m * weights.n * width)
        sf = record["parent"]["route"].startswith("sf")
        call = Call(version=1, size=C.sizeof(Call), m=m, n=weights.n, k=weights.k,
            experts=weights.experts, group_size=self.arr.group_size, device=self.device.value,
            compute_units=self.cu.value, mapping_id=self.arr.mapping_id, a=self.a.ptr,
            low=self.planes["low"], high=self.planes["high"],
            metadata=self.planes["scale" if sf else "units"],
            zero=self.planes["zero"] if sf and weights.q != 8 else None,
            output=self.output.ptr, offsets_device=self.offsets.ptr if self.offsets else None,
            stream=r.stream.value)
        self.call = call
        self.recipe = Recipe(1, C.sizeof(Recipe), algorithm, split, 1 if algorithm else 0)
        if self.grouped:
            self.request = GroupedComputeCall(3, C.sizeof(GroupedComputeCall),
                GroupedDeviceCall(2, C.sizeof(GroupedDeviceCall), call, maximum, 0), self.tag)
        else:
            self.request = DenseComputeCall(2, C.sizeof(DenseComputeCall),
                DenseIO(1, C.sizeof(DenseIO), call, storage, storage), self.tag)
        resource = QueryResources()
        if self.tag:
            if self.grouped:
                legacy = function(self.lib, "quactlize_kpack_grouped_query_v2",
                    [C.POINTER(GroupedDeviceCall), C.c_void_p, C.c_void_p])
                old_request = self.request.device_call
            else:
                legacy = function(self.lib, "quactlize_kpack_decode_dense_query_v1",
                    [C.POINTER(DenseIO), C.c_void_p, C.c_void_p])
                old_request = self.request.dense
            if legacy(C.byref(old_request), C.byref(self.recipe), C.byref(resource)) != 1:
                raise ValueError("BF16 module accepted a legacy compute query")
        checked(self.query(C.byref(self.request), C.byref(self.recipe), C.byref(resource)), "required capability query")
        if algorithm:
            self.recipe.grid = self.cu.value
            checked(self.query(C.byref(self.request), C.byref(self.recipe), C.byref(resource)), "persistent grid query")
        self.workspace = Buffer(r, resource.workspace_bytes)
        call.workspace, call.workspace_bytes = self.workspace.ptr, resource.workspace_bytes
        if self.grouped:
            self.request.device_call.call = call
        else:
            self.request.dense.call = call
        wrong = type(self.request).from_buffer_copy(self.request)
        wrong.compute_type ^= 1
        if self.query(C.byref(wrong), C.byref(self.recipe), C.byref(QueryResources())) == 0:
            raise ValueError("wrong compute type was admitted")
        self.handle = C.c_void_p()
        checked(self.prepare(C.byref(self.request), C.byref(self.recipe), C.byref(self.handle)), "typed prepare")
        self.receipt = dict(parent=record["parent"], compute=self.compute, algorithm=algorithm, split=split,
            grid=self.recipe.grid, shared_bytes=resource.shared_bytes, workspace_bytes=resource.workspace_bytes,
            occupancy=resource.occupancy, actual_schedule="persistent" if algorithm else
                "compact" if self.grouped and weights.experts <= 1024 else "ordinary")

    def upload_a(self, values):
        if not self.grouped and self.storage == 1:
            data = np.asarray(values, dtype="<f4")
        elif self.compute == "bf16" or (not self.grouped and self.storage == 2):
            data = bf16_bits(values)
        else:
            data = np.asarray(values, dtype="<f2")
        self.a.upload(data)

    def read(self):
        shape = (self.call.m, self.call.n)
        if not self.grouped and self.storage == 1:
            return self.output.read("<f4", shape)
        if self.compute == "bf16" or (not self.grouped and self.storage == 2):
            return bf16_float(self.output.read("<u2", shape))
        return self.output.read("<f2", shape).astype("f4")

    def run(self):
        return self.run_fn(self.handle, self.r.stream)

    def bind_indexed(self, io):
        bind = function(self.lib, "quactlize_kpack_bind_llama_indexed_v1", [C.c_void_p, C.POINTER(IndexedIO)])
        checked(bind(self.handle, C.byref(io)), "bind fused indexed compute")
        project = function(self.lib, "quactlize_kpack_moe_projection_v2", [C.c_void_p, C.POINTER(ProjectionV2)])
        p = ProjectionV2()
        checked(project(self.handle, C.byref(p)), "typed MoE projection")
        if p.compute_type != self.tag:
            raise ValueError("MoE projection compute identity differs")
        self.stage = function(self.lib, "quactlize_kpack_moe_stage_v2",
            [C.c_void_p, C.POINTER(MoePlanV2), C.c_int, C.c_void_p])
        return p.projection

    def close(self):
        self.r.sdk.synchronize(self.r.stream)
        self.destroy(self.handle)
