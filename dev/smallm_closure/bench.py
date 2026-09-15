"""Same-stream, same-weight-ring F32 endpoints for SIMT and typed TC."""
import ctypes as C
import math
from pathlib import Path

import numpy as np

from dev.smallm_closure.fixture import Weights, copies_for_l2
from dev.bf16_compute.fixture import compare
from dev.bf16_compute.native import function
from quactlize.decode.native_compute import bind_compute, DenseIO, DenseComputeCall, GroupedDeviceCall, GroupedComputeCall
from quactlize.dispatch.native import IndexedIO
from quactlize.execution.native import Call as VecCall, SimtCallV2, SimtConfig, Sizes, Arrangement, arrangement, bind_simt_compute
from quactlize.runtime.native import Call, Recipe, Resources as Query, checked
from quactlize.runtime.compiler import sha
from tools.run_kpack_gemv_gate import Resources
from tools.run_kpack_grouped_decode_probe import Replay


class Unsupported(ValueError):
    pass


class Buffer:
    """Preserve the 128-byte Split-K alignment as well as canaries."""
    def __init__(self, r, size):
        self.r, self.size = r, int(size)
        self.base = r.alloc(self.size + 256)
        if self.base % 128:
            raise ValueError("allocator did not supply 128-byte alignment")
        self.ptr = self.base + 128
        self.poison()

    def poison(self):
        self.r.fill(self.base, 0xa5, self.size + 256)
        # Scratch can be owned by a different allocation stream. This wait
        # is setup-only, outside every timed replay and production entry.
        self.r.sdk.synchronize(self.r.stream)

    def upload(self, array):
        array = np.ascontiguousarray(array)
        if array.nbytes != self.size:
            raise ValueError("upload shape/width differs")
        self.r.sdk.synchronize(self.r.stream)
        checked(self.r.sdk.lib.hggcMemcpy(self.ptr, array.ctypes.data, self.size, 1), "H2D fixture")
        self.r.sdk.synchronize(None)

    def guard(self):
        s = self.r.sdk
        if s.download(self.base, 128) != b"\xa5" * 128 or s.download(self.ptr + self.size, 128) != b"\xa5" * 128:
            raise ValueError("workspace/input/output redzone differs")

    def read(self, shape):
        self.guard()
        return np.frombuffer(self.r.sdk.download(self.ptr, self.size), dtype="<f4").reshape(shape)


class WeightRing:
    def __init__(self, sdk, point, l2_bytes):
        self.r = Resources(sdk)
        self.w = Weights(point["q"], point["n"], point["k"], point["experts"])
        resident = sum(v.nbytes for v in self.w.planes.values())
        self.expert_bytes = resident // self.w.experts
        # Minimum unique experts across all profiles/replays. The full
        # E-sized allocation is never mistaken for bytes actually touched.
        minimum_active = point["topk"] if point["mode"] else 1
        self.copies = copies_for_l2(self.expert_bytes, minimum_active, l2_bytes)
        self.planes = []
        bases = {name: self.r.alloc(v.nbytes * self.copies) if v.size else None for name, v in self.w.planes.items()}
        for i in range(self.copies):
            row = {}
            for name, value in self.w.planes.items():
                row[name] = bases[name] + i * value.nbytes if value.size else None
                if value.size:
                    checked(sdk.lib.hggcMemcpy(row[name], value.ctypes.data, value.nbytes, 1), "weight ring upload")
            self.planes.append(row)
        sdk.synchronize(None)
        self.receipt = dict(l2_bytes=l2_bytes, copies=self.copies, resident_bytes=resident * self.copies,
            minimum_active=minimum_active, active_ring_bytes=self.expert_bytes * minimum_active * self.copies,
            cache="ROTATING_ACTIVE_WEIGHT_ADDRESSES", fixture=self.w.record)

    def close(self):
        self.r.close()


class Context:
    def __init__(self, sdk, ring, point):
        self.sdk, self.ring, self.p = sdk, ring, point
        self.r = Resources(sdk)
        self.rows = point["tokens"] * (point["topk"] if point["mode"] else 1)
        self.data = ring.w.inputs(point)
        self.a = Buffer(self.r, self.data["a"].nbytes)
        self.ids = Buffer(self.r, self.data["ids"].nbytes) if self.data["ids"] is not None else None
        self.out = Buffer(self.r, self.rows * point["n"] * 4)
        self.update(0)
        self.arr = arrangement(point["q"])

    def update(self, repeat):
        self.data = self.ring.w.inputs(self.p, repeat)
        self.a.upload(self.data["a"])
        if self.ids:
            self.ids.upload(self.data["ids"])

    def check(self):
        self.sdk.synchronize(self.r.stream)
        out = self.out.read((self.rows, self.p["n"]))
        self.a.guard()
        if self.ids:
            self.ids.guard()
        return compare(out, self.data["gold"], self.data["denom"])

    def close(self):
        self.r.close()


class Simt:
    def __init__(self, context, library, candidate):
        self.b, self.c = context, candidate
        self.r = Resources(context.sdk)
        self.graph, self.proof_graph = None, None
        b, p = context, context.p
        self.library = C.CDLL(str(Path(library).resolve()), mode=C.RTLD_LOCAL)
        base = VecCall(version=1, size=C.sizeof(VecCall), qtype=p["q"], n=p["n"], k=p["k"],
            experts=p["experts"], rows=b.rows, mode=p["mode"], input_type=1, channels=p["channels"],
            topk=p["topk"], a_row_stride=p["k"], a_token_stride=p["channels"] * p["k"],
            ids_stride=p["topk"], out_row_stride=p["n"], a=b.a.ptr,
            ids=b.ids.ptr if b.ids else None, output=b.out.ptr, stream=b.r.stream.value,
            **b.ring.planes[0])
        sizes = Sizes()
        if candidate["kind"] == "q4":
            class Config(C.Structure):
                _fields_ = [("version", C.c_uint32), ("size", C.c_uint32)] + [(n, C.c_int32) for n in ("reader", "variant", "warps", "values", "columns")]
            self.cfg = Config(1, C.sizeof(Config), **candidate["recipe"])
            selected = Config()
            query = function(self.library, "quactlize_kpack_q4_decode_select_v1", [C.POINTER(VecCall), C.POINTER(Arrangement), C.POINTER(Config), C.POINTER(Sizes)])
            checked(query(C.byref(base), C.byref(b.arr), C.byref(selected), C.byref(sizes)), "Q4 incumbent selection")
            if bytes(selected) != bytes(self.cfg):
                raise ValueError("Q4 incumbent differs from public selector")
            self.run_fn = function(self.library, "quactlize_kpack_q4_decode_run_v1", [C.POINTER(VecCall), C.POINTER(Config), C.POINTER(Arrangement)])
            self.calls = []
            for plane in b.ring.planes:
                call = VecCall.from_buffer_copy(base)
                for name, ptr in plane.items():
                    setattr(call, name, ptr)
                self.calls.append(call)
        else:
            self.cfg = SimtConfig(**candidate["recipe"])
            query, self.run_fn = bind_simt_compute(self.library)
            request = SimtCallV2(base, int(p["compute"] == "bf16"))
            rc = query(C.byref(request), C.byref(self.cfg), C.byref(b.arr), C.byref(sizes))
            if rc == 24:
                raise Unsupported(f"SIMT_QUERY_RC_{rc}")
            checked(rc, "SIMT ABI/arrangement query")
            self.calls = []
            for plane in b.ring.planes:
                call = SimtCallV2.from_buffer_copy(request)
                for name, ptr in plane.items():
                    setattr(call.call, name, ptr)
                self.calls.append(call)
        self.scratch = Buffer(self.r, sizes.workspace_bytes)
        for call in self.calls:
            target = call.call if candidate["kind"] != "q4" else call
            target.workspace, target.workspace_bytes = self.scratch.ptr, sizes.workspace_bytes
        self.receipt = dict(kind=candidate["kind"], split=candidate["split"], workspace_bytes=sizes.workspace_bytes,
                            scope="F32_INPUT_REAL_SPLIT_REDUCER_F32_OUTPUT_NO_HOST_ROUTING")

    def launch(self, index=0):
        return self.run_fn(C.byref(self.calls[index]), C.byref(self.cfg), C.byref(self.b.arr))

    def close(self):
        self.b.sdk.synchronize(self.b.r.stream)
        if self.graph:
            self.graph.close()
        if self.proof_graph:
            self.proof_graph.close()
        self.r.close()


class TensorCore:
    def __init__(self, context, record, candidate):
        self.b, self.c = context, candidate
        self.r = Resources(context.sdk)
        self.handles, self.graph, self.proof_graph = [], None, None
        b, p = context, context.p
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("TC payload differs")
        self.library = C.CDLL(str(record["path"]), mode=C.RTLD_LOCAL)
        grouped = bool(p["mode"])
        identity, query, prepare, self.run_fn, self.destroy = bind_compute(self.library, grouped)
        actual = identity().contents
        if actual.compute_type != int(p["compute"] == "bf16") or actual.parent.contents.build_key.decode() != record["key"]:
            raise ValueError("TC compute/build identity differs")
        dev, cu, name = C.c_int32(), C.c_int32(), C.create_string_buffer(256)
        getter = function(self.library, "quactlize_kpack_" + ("device_v1" if grouped else "decode_dense_device_v1"),
                          [C.c_char_p, C.c_int, C.POINTER(C.c_int32), C.POINTER(C.c_int32)])
        checked(getter(name, 256, C.byref(dev), C.byref(cu)), "TC device identity")
        if name.value.decode() != "PPU-ZW810" or cu.value != 72:
            raise ValueError("TC device is not the measured 72-CU PPU-ZW810")
        self.recipe = Recipe(1, C.sizeof(Recipe), candidate["algorithm"], candidate["split"],
                             1 if candidate["algorithm"] else 0)
        if grouped:
            self.compact_a = Buffer(self.r, b.rows * p["k"] * 2)
            self.compact_out = Buffer(self.r, b.rows * p["n"] * 2)
            self.offsets = Buffer(self.r, (p["experts"] + 1) * 4)
            self.row_ids = Buffer(self.r, b.rows * 4)
            # Mutable GPU offsets are produced by the indexed binding; the
            # host never copies/counts device routing inside the timed call.
            self.offsets.upload(np.zeros(p["experts"] + 1, dtype="i4"))
        call = Call(version=1, size=C.sizeof(Call), m=b.rows, n=p["n"], k=p["k"], experts=p["experts"],
            group_size=b.arr.group_size, device=dev.value, compute_units=cu.value, mapping_id=b.arr.mapping_id,
            a=self.compact_a.ptr if grouped else b.a.ptr,
            output=self.compact_out.ptr if grouped else b.out.ptr,
            offsets_device=self.offsets.ptr if grouped else None, stream=b.r.stream.value,
            low=b.ring.planes[0]["low"], high=b.ring.planes[0]["high"], metadata=b.ring.planes[0]["units"])
        if grouped:
            req = GroupedComputeCall(3, C.sizeof(GroupedComputeCall),
                GroupedDeviceCall(2, C.sizeof(GroupedDeviceCall), call, p["tokens"], 0), int(p["compute"] == "bf16"))
        else:
            req = DenseComputeCall(2, C.sizeof(DenseComputeCall), DenseIO(1, C.sizeof(DenseIO), call, 1, 1), int(p["compute"] == "bf16"))
        resources = Query()
        rc = query(C.byref(req), C.byref(self.recipe), C.byref(resources))
        if rc == 1:
            raise Unsupported("TC_RESOURCE_OR_SHAPE_UNSUPPORTED")
        checked(rc, "TC explicit query")
        if self.recipe.algorithm:
            par = candidate["parent"]
            mt = (b.rows + par["tm"] - 1) // par["tm"]
            if grouped:
                mt = min(p["experts"] * ((p["tokens"] + par["tm"] - 1) // par["tm"]),
                         (b.rows + min(b.rows, p["experts"]) * (par["tm"] - 1)) // par["tm"])
            tiles = mt * ((p["n"] + par["tn"] - 1) // par["tn"]) * (self.recipe.split if grouped else 1)
            cap = cu.value * max(1, min(candidate["grid_b"], resources.occupancy))
            waves = (tiles + cap - 1) // cap
            self.recipe.grid = (tiles + waves - 1) // waves if candidate["grid_mode"] == 3 else min(tiles, cap)
            checked(query(C.byref(req), C.byref(self.recipe), C.byref(resources)), "bounded persistent recipe query")
        self.scratch = Buffer(self.r, resources.workspace_bytes)
        call.workspace, call.workspace_bytes = self.scratch.ptr, resources.workspace_bytes
        bind = function(self.library, "quactlize_kpack_bind_llama_indexed_v1", [C.c_void_p, C.POINTER(IndexedIO)]) if grouped else None
        io = IndexedIO(version=1, size=C.sizeof(IndexedIO), tokens=p["tokens"], topk=p["topk"], channels=p["channels"],
            ids_stride=p["topk"], a_row_stride=p["k"], a_token_stride=p["channels"] * p["k"], out_row_stride=p["n"],
            ids=b.ids.ptr if b.ids else None, a=b.a.ptr, output=b.out.ptr,
            row_ids=self.row_ids.ptr if grouped else None)
        for plane in b.ring.planes:
            call.low, call.high, call.metadata = plane["low"], plane["high"], plane["units"]
            if grouped:
                req.device_call.call = call
            else:
                req.dense.call = call
            handle = C.c_void_p()
            checked(prepare(C.byref(req), C.byref(self.recipe), C.byref(handle)), "TC prepare")
            self.handles.append(handle)
            if bind:
                checked(bind(handle, C.byref(io)), "TC indexed prepare/finish binding")
        self.receipt = dict(kind="tc", parent=record["parent"], build_key=record["key"],
            module_sha256=record["sha256"], split=self.recipe.split, algorithm=self.recipe.algorithm,
            grid=self.recipe.grid, shared_bytes=resources.shared_bytes, workspace_bytes=resources.workspace_bytes,
            compute_units=cu.value, device=dev.value,
            scope="F32_INDEXED_PREPARE_TC_REAL_REDUCER_INDEXED_FINISH" if grouped else "F32_TYPED_TC_REAL_REDUCER_F32_OUTPUT")

    def launch(self, index=0):
        return self.run_fn(self.handles[index], self.b.r.stream)

    def close(self):
        self.b.sdk.synchronize(self.b.r.stream)
        if self.graph:
            self.graph.close()
        if self.proof_graph:
            self.proof_graph.close()
        for h in self.handles:
            self.destroy(h)
        self.r.close()


def correctness(kernel, full=False):
    b = kernel.b
    b.out.poison()
    checked(kernel.launch(), "correctness full call")
    proof = b.check()
    kernel.scratch.guard()
    if hasattr(kernel, "row_ids"):
        for buf in (kernel.row_ids, kernel.offsets, kernel.compact_a, kernel.compact_out):
            buf.guard()
    if not full:
        return proof
    kernel.proof_graph = Replay(b.sdk, b.r.stream, kernel.launch, 1)
    proofs = []
    for repeat in (1, 2, 3):
        b.update(repeat)
        b.out.poison()
        checked(kernel.proof_graph(), "mutable input and IDs replay")
        proofs.append(b.check())
        kernel.scratch.guard()
    b.a.upload(np.zeros_like(b.data["a"]))
    b.out.poison()
    checked(kernel.proof_graph(), "zero-A negative")
    b.sdk.synchronize(b.r.stream)
    zero = b.out.read((b.rows, b.p["n"]))
    if not np.isfinite(zero).all() or np.any(zero != 0):
        raise ValueError("zero-A replay not zero")
    try:
        compare(zero, b.data["gold"], b.data["denom"])
    except ValueError:
        pass
    else:
        raise ValueError("independent oracle accepted planted zero A")
    b.update(0)
    checked(kernel.proof_graph(), "restore after negative")
    b.check()
    # Prove the last handle/call uses every plane from the last ring slot.
    # Identical replicas alone would not catch accidentally timing slot zero
    # repeatedly through an aliased prepared handle (a false cold result).
    last = b.ring.copies - 1
    checked(kernel.launch(last), "last weight-ring slot")
    b.check()
    positive = b.out.read((b.rows, b.p["n"])).copy()
    checked_planes = []
    for name, array in b.ring.w.planes.items():
        if not array.size:
            continue
        ptr = b.ring.planes[last][name]
        try:
            b.r.fill(ptr, 0, array.nbytes)
            checked(kernel.launch(last), "ring plane pointer negative")
            b.sdk.synchronize(b.r.stream)
            negative = b.out.read(positive.shape)
            if np.array_equal(positive.view("u4"), negative.view("u4")):
                raise ValueError("ring slot plane was not consumed: " + name)
            checked_planes.append(name)
        finally:
            b.sdk.synchronize(b.r.stream)
            checked(b.sdk.lib.hggcMemcpy(ptr, array.ctypes.data, array.nbytes, 1), "restore ring plane")
            b.sdk.synchronize(None)
    checked(kernel.launch(last), "restored ring slot")
    b.check()
    return dict(eager=proof, graph_replays=proofs, zero_a_negative="RED", guards="PASS",
                ring_pointer_negatives=checked_planes)


def measure(kernel, count):
    b = kernel.b
    if kernel.graph is None:
        index = 0
        repeats = b.ring.copies * max(1, math.ceil(16 / b.ring.copies))
        def next_call():
            nonlocal index
            rc = kernel.launch(index % b.ring.copies)
            index += 1
            return rc
        kernel.graph = Replay(b.sdk, b.r.stream, next_call, repeats)
        kernel.repeats = repeats
        checked(kernel.graph(), "excluded graph upload/first launch")
        b.sdk.synchronize(b.r.stream)
    samples = [t / kernel.repeats for t in b.r.samples(kernel.graph, count)]
    b.check()
    kernel.scratch.guard()
    return samples
