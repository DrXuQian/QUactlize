"""Resident full-call comparison. No setup, H2D or CPU routing inside timing."""

import ctypes as C
import hashlib
import math
from pathlib import Path

import numpy as np

from dev.bf16_compute.fixture import raw_weight, planes, round_compute
from dev.gemv_simt.native import Graph, checked
from dev.gemv_simt.production import Library as ProductionSimt
from quactlize.execution.native import Call, arrangement
from quactlize.execution.simt_codegen import Config as SimtConfig
from quactlize.fusion.native import Library, FusionCall, Sizes
from quactlize.decode.native_compute import (
    bind_compute,
    DenseIO,
    DenseComputeCall,
    GroupedDeviceCall,
    GroupedMetadataCall,
)
from quactlize.dispatch.native import IndexedIO
from quactlize.runtime.native import Call as TcCall, Recipe, Resources
from dev.gate_up_perf.plan import ring_copies


def routing(point):
    m = point["tokens"]
    if point["mode"] == 0:
        return np.empty((0, 8), dtype="i4"), np.zeros(m, dtype="i4"), np.arange(m)
    ids = ((13 + np.arange(m)[:, None] * 37 + np.arange(8)[None, :] * 29) % 256).astype(
        "i4"
    )
    return ids, ids.reshape(-1), np.arange(m * 8) // 8


def physical(raw_gate, raw_up, paired):
    if not paired:
        return np.concatenate((raw_gate, raw_up), axis=0)
    n = len(raw_gate)
    return np.stack(
        [x.reshape(n // 4, 4, *x.shape[1:]) for x in (raw_gate, raw_up)], axis=1
    ).reshape(2 * n, *raw_gate.shape[1:])


class Buffer:
    def __init__(self, rt, size):
        self.rt, self.size = rt, int(size)
        # Preserve production-like plane alignment; a 16-byte guard prefix
        # alone would shift every 64/128-byte transaction boundary.
        self.base = rt.allocate(self.size + 512)
        self.ptr = (self.base + 16 + 255) // 256 * 256
        rt.fill(self.base, self.size + 512)

    def check_guard(self):
        if not np.all(self.rt.download(self.ptr - 16, 16) == 0xA5) or not np.all(
            self.rt.download(self.ptr + self.size, 16) == 0xA5
        ):
            raise ValueError("buffer guard changed")

    def read(self):
        self.check_guard()
        return self.rt.download(self.ptr, self.size)

    def poison(self):
        self.rt.fill(self.ptr, self.size)


class WeightRing:
    def __init__(self, rt, payloads, experts, copies):
        self.rt, self.experts, self.copies = rt, experts, copies
        sample = next(iter(payloads.values()))
        self.widths = {k: sample[k].nbytes for k in ("low", "high", "units")}
        self.buffers = {
            k: Buffer(rt, size * experts * copies)
            for k, size in self.widths.items()
            if size
        }
        # Only active expert slices need source data, but their public E256
        # address space is real and disjoint. Inactive slices stay zero.
        for b in self.buffers.values():
            rt.fill(b.ptr, b.size, 0)
        for e, values in payloads.items():
            for name, width in self.widths.items():
                if width:
                    if values[name].nbytes != width:
                        raise ValueError("expert slice size differs")
                    rt.copy(self.buffers[name].ptr + e * width, values[name].view("u1"))
        for copy in range(1, copies):
            for e in payloads:
                for name, width in self.widths.items():
                    if width:
                        ptr = self.buffers[name].ptr
                        checked(
                            rt.MemcpyAsync(
                                ptr + (copy * experts + e) * width,
                                ptr + e * width,
                                width,
                                3,
                                rt.stream,
                            ),
                            "untimed same-stream ring D2D",
                        )
        rt.sync()
        # Verify first and last physical copies, including every active expert.
        for copy in (0, copies - 1):
            for e, values in payloads.items():
                for name, width in self.widths.items():
                    if width and not np.array_equal(
                        rt.download(
                            self.buffers[name].ptr + (copy * experts + e) * width, width
                        ),
                        values[name].view("u1").reshape(-1),
                    ):
                        raise ValueError("weight ring byte verification failed")
        for b in self.buffers.values():
            b.check_guard()

    def at(self, copy):
        return {
            name: (
                self.buffers[name].ptr + copy * self.experts * width if width else None
            )
            for name, width in self.widths.items()
        }


class Bench:
    def __init__(self, rt, bundle, manifest, point, l2):
        self.rt, self.bundle, self.manifest, self.point = (
            rt,
            Path(bundle),
            manifest,
            point,
        )
        p = point
        self.rows = p["tokens"] * (8 if p["mode"] else 1)
        self.ids_host, self.owners, self.a_rows = routing(p)
        self.active = sorted(set(map(int, self.owners)))
        self.input = Buffer(rt, p["tokens"] * p["k"] * 4)
        self.ids = Buffer(rt, self.ids_host.nbytes) if p["mode"] else None
        if self.ids:
            rt.copy(self.ids.ptr, self.ids_host)
        self.gold_weights = {}
        raw_hash = hashlib.sha256()
        fused = {}
        baseline = [{}, {}] if not p["mode"] else [{}]
        for ordinal, e in enumerate(self.active):
            raws = []
            gold = []
            for side in (0, 1):
                raw, w = raw_weight(p["q"], p["n"], p["k"], 17 + e * 73 + side * 419)
                raws.append(raw)
                gold.append(w)
                raw_hash.update(raw.tobytes())
            self.gold_weights[e] = gold
            fused[e] = planes(physical(*raws, True), p["q"])
            if p["mode"]:
                baseline[0][e] = planes(physical(*raws, False), p["q"])
            else:
                for side in (0, 1):
                    baseline[side][e] = planes(raws[side], p["q"])
            if (ordinal + 1) % 8 == 0:
                print(
                    f"GATE_UP_PERF_FIXTURE point={p['key']} experts={ordinal+1}/{len(self.active)}",
                    flush=True,
                )
        self.weight_bytes = sum(
            next(iter(fused.values()))[name].nbytes for name in ("low", "high", "units")
        ) * len(self.active)
        self.copies = ring_copies(l2, self.weight_bytes)
        self.fused = WeightRing(rt, fused, p["experts"], self.copies)
        self.baseline = [WeightRing(rt, b, p["experts"], self.copies) for b in baseline]
        self.fusion_lib = Library(self.bundle / "libquactlize_ppu_gate_up.so")
        self.layout = self.fusion_lib.arrangement(p["q"])
        self.postlib = C.CDLL(
            str(self.bundle / "libgate_up_perf_postop.so"), mode=C.RTLD_LOCAL
        )
        self.post = self.postlib.gate_up_perf_postop
        self.post.argtypes = [C.c_void_p] * 3 + [C.c_int] * 5 + [C.c_void_p]
        self.post.restype = C.c_int
        self.proof = dict(
            raw_sha256=raw_hash.hexdigest(),
            ids_sha256=hashlib.sha256(self.ids_host.tobytes()).hexdigest(),
            active_experts=self.active,
            weight_bytes=self.weight_bytes,
            copies=self.copies,
            cold_weight_bytes=self.copies * self.weight_bytes,
            l2_bytes=l2,
            weight_alignment={k: v.ptr % 256 for k, v in self.fused.buffers.items()},
            oracle="OFFICIAL_GGUF_DYADIC_PAIRED_DOT_PLUS_SWIGLU",
        )
        self.update(0)

    def update(self, repeat):
        p = self.point
        self.a = (
            np.random.default_rng(132 + repeat)
            .integers(-7, 8, (p["tokens"], p["k"]))
            .astype("f4")
            / 32
        )
        self.rt.copy(self.input.ptr, self.a)
        rounded = round_compute(self.a, "bf16" if p["compute"] else "f16")
        dot = []
        for side in (0, 1):
            dot.append(
                np.stack(
                    [
                        rounded[self.a_rows[r]].astype("f8")
                        @ self.gold_weights[int(e)][side].astype("f8").T
                        for r, e in enumerate(self.owners)
                    ]
                ).astype("f4")
            )
        if p["rounding"]:
            dot = [round_compute(x, "bf16") for x in dot]
        g, u = dot
        with np.errstate(over="ignore"):
            self.gold = g / (np.float32(1) + np.exp(-g)) * u
        if not np.isfinite(self.gold).all() or not np.any(self.gold):
            raise ValueError("degenerate oracle")

    def call(self, n, planes, output, workspace):
        p = self.point
        return Call(
            version=1,
            size=C.sizeof(Call),
            qtype=p["q"],
            n=n,
            k=p["k"],
            experts=p["experts"],
            rows=self.rows,
            mode=p["mode"],
            input_type=1,
            channels=1,
            topk=8 if p["mode"] else 1,
            a_row_stride=p["k"],
            a_token_stride=p["k"],
            ids_stride=8,
            out_row_stride=n,
            a=self.input.ptr,
            ids=self.ids.ptr if self.ids else None,
            output=output,
            workspace=workspace.ptr,
            workspace_bytes=workspace.size,
            stream=self.rt.stream.value,
            **planes,
        )

    def check(self, arm):
        got = arm.output.read().view("f4").reshape(self.gold.shape)
        self.input.check_guard()
        if self.ids:
            self.ids.check_guard()
        for b in arm.guards:
            b.check_guard()
        error = float(
            np.max(np.abs(got.astype("f8") - self.gold))
            / max(1e-20, float(np.max(np.abs(self.gold))))
        )
        if not np.isfinite(got).all() or not math.isfinite(error) or error >= 0.005:
            raise ValueError(f"independent GGUF+SwiGLU error={error}")
        return error


class Arm:
    def __init__(self, b, key, config=None):
        self.b, self.key, self.config = b, key, config
        rt, p = b.rt, b.point
        n = p["n"]
        self.handles = []
        self.guards = []
        self.graph = None
        self.output = Buffer(rt, b.rows * n * 4)
        self.work = Buffer(rt, b.rows * 8 * 2 * n * 4)
        self.guards.append(self.work)
        self.calls = []
        self.receipt = dict(key=key)
        if key == "incumbent":
            choice = p["incumbent"]["config"]
            self.receipt.update(p["incumbent"])
            self.receipt["call_scope"] = (
                "SELECTED_INDEXED_TC_PREPARE_GEMM_FINISH_PLUS_MINIMAL_SWIGLU"
                if choice["kind"] == "tc" and p["mode"]
                else "SELECTED_PROJECTIONS_INCLUDING_REDUCERS_PLUS_MINIMAL_SWIGLU"
            )
            self.projections = [
                Buffer(rt, b.rows * (2 * n if p["mode"] else n) * 4) for _ in b.baseline
            ]
            self.guards += self.projections
            self.post_call = lambda: b.post(
                self.projections[0].ptr,
                (
                    self.projections[0].ptr + n * 4
                    if p["mode"]
                    else self.projections[1].ptr
                ),
                self.output.ptr,
                b.rows,
                n,
                2 * n if p["mode"] else n,
                n,
                p["rounding"],
                rt.stream,
            )
            if choice["kind"] == "simt":
                lib = ProductionSimt(
                    b.bundle / "libquactlize_ppu_execution.so", p["compute"]
                )
                c = SimtConfig(
                    **{
                        k: choice[k]
                        for k in ("variant", "columns", "warps", "values", "split")
                    }
                )
                for copy in range(b.copies):
                    runs = [
                        lib.prepare(
                            b.call(
                                2 * n if p["mode"] else n,
                                ring.at(copy),
                                out.ptr,
                                self.work,
                            ),
                            c,
                        )
                        for ring, out in zip(b.baseline, self.projections)
                    ]
                    self.calls.append(self.combine([*runs, self.post_call]))
            else:
                record = next(
                    r
                    for r in b.manifest["modules"]
                    if r["parent"]["symbol"] == choice["symbol"]
                )
                for copy in range(b.copies):
                    runs = [
                        self.tc(record, ring.at(copy), out.ptr)
                        for ring, out in zip(b.baseline, self.projections)
                    ]
                    self.calls.append(self.combine([*runs, self.post_call]))
        else:
            self.receipt["config"] = {
                k: getattr(config, k) for k in ("backend", "split", "tile_m", "warps")
            }
            self.receipt["call_scope"] = "PAIRED_FUSED_PRODUCER_INCLUDING_FINAL_REDUCER"
            for copy in range(b.copies):
                call = FusionCall(
                    b.call(n, b.fused.at(copy), self.output.ptr, self.work),
                    p["compute"],
                    1,
                    p["rounding"],
                )
                size = Sizes()
                checked(
                    b.fusion_lib.query(
                        C.byref(call), C.byref(config), C.byref(b.layout), C.byref(size)
                    ),
                    "fusion query",
                )
                if size.workspace_bytes > self.work.size:
                    raise ValueError("fusion workspace mismatch")
                self.calls.append(
                    lambda call=call: b.fusion_lib.run(
                        C.byref(call), C.byref(config), C.byref(b.layout)
                    )
                )

    @staticmethod
    def combine(runs):
        def launch():
            for fn in runs:
                rc = fn()
                if rc:
                    return rc
            return 0

        return launch

    def tc(self, record, planes, output):
        b = self.b
        rt, p = b.rt, b.point
        grouped = bool(p["mode"])
        n = p["n"] * (2 if grouped else 1)
        lib = C.CDLL(str(b.bundle / record["path"]), mode=C.RTLD_LOCAL)
        identity, query, prepare, run, destroy = bind_compute(lib, grouped)
        ident = identity().contents
        parent = ident.parent.contents
        if (
            ident.compute_type != p["compute"]
            or parent.build_key.decode() != record["key"]
        ):
            raise ValueError("TC compute/build identity differs")
        device, cu = C.c_int(), C.c_int()
        name = C.create_string_buffer(256)
        dev = getattr(
            lib,
            (
                "quactlize_kpack_device_v1"
                if grouped
                else "quactlize_kpack_decode_dense_device_v1"
            ),
        )
        dev.argtypes = [C.c_char_p, C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_int)]
        dev.restype = C.c_int
        checked(dev(name, 256, C.byref(device), C.byref(cu)), "TC device identity")
        call = TcCall(
            version=1,
            size=C.sizeof(TcCall),
            m=b.rows,
            n=n,
            k=p["k"],
            experts=p["experts"],
            group_size=arrangement(p["q"]).group_size,
            device=device.value,
            compute_units=cu.value,
            mapping_id=arrangement(p["q"]).mapping_id,
            a=b.input.ptr,
            low=planes["low"],
            high=planes["high"],
            metadata=planes["units"],
            output=output,
            stream=rt.stream.value,
        )
        if grouped:
            a = Buffer(rt, b.rows * p["k"] * 2)
            out = Buffer(rt, b.rows * n * 2)
            offset = Buffer(rt, (p["experts"] + 1) * 4)
            self.guards += [a, out, offset]
            rt.fill(offset.ptr, offset.size, 0)
            call.a, call.output, call.offsets_device = a.ptr, out.ptr, offset.ptr
            typed = GroupedMetadataCall(
                4,
                C.sizeof(GroupedMetadataCall),
                GroupedDeviceCall(2, C.sizeof(GroupedDeviceCall), call, p["tokens"], 0),
                p["compute"],
                p["compute"],
            )
        else:
            typed = DenseComputeCall(
                2,
                C.sizeof(DenseComputeCall),
                DenseIO(1, C.sizeof(DenseIO), call, 1, 1),
                p["compute"],
            )
        choice = p["incumbent"]["config"]
        recipe = Recipe(1, C.sizeof(Recipe), 0, choice["split"], 0)
        resources = Resources()
        checked(
            query(C.byref(typed), C.byref(recipe), C.byref(resources)),
            "TC resource query",
        )
        work = Buffer(rt, resources.workspace_bytes)
        self.guards.append(work)
        target = typed.device_call.call if grouped else typed.dense.call
        target.workspace, target.workspace_bytes = work.ptr, work.size
        handle = C.c_void_p()
        checked(prepare(C.byref(typed), C.byref(recipe), C.byref(handle)), "TC prepare")
        self.handles.append((lib, handle, destroy))
        if grouped:
            rows = Buffer(rt, b.rows * 4)
            self.guards.append(rows)
            io = IndexedIO(
                1,
                C.sizeof(IndexedIO),
                p["tokens"],
                8,
                1,
                0,
                8,
                p["k"],
                p["k"],
                n,
                b.ids.ptr,
                b.input.ptr,
                output,
                rows.ptr,
            )
            bind = lib.quactlize_kpack_bind_llama_indexed_v1
            bind.argtypes = [C.c_void_p, C.POINTER(IndexedIO)]
            bind.restype = C.c_int
            checked(bind(handle, C.byref(io)), "TC indexed binding")
        return lambda: run(handle, rt.stream)

    def correctness(self):
        b = self.b
        errors = []
        for repeat, copy in ((0, 0), (1, b.copies - 1)):
            b.update(repeat)
            self.output.poison()
            checked(self.calls[copy](), "full-call correctness")
            b.rt.sync()
            errors.append(b.check(self))
        b.rt.fill(b.input.ptr, b.input.size, 0)
        self.output.poison()
        checked(self.calls[0](), "zero-A negative")
        b.rt.sync()
        try:
            b.check(self)
        except ValueError as e:
            if not str(e).startswith("independent GGUF"):
                raise
        else:
            raise ValueError("zero-A negative was not detected")
        b.update(0)
        self.graph = Graph(
            b.rt, self.calls
        )  # complete ring; first upload/replay excluded
        b.check(self)
        return dict(
            errors=errors,
            zero_a_negative="RED",
            ring_endpoints_checked=[0, b.copies - 1],
        )

    def close(self):
        self.b.rt.sync()
        if self.graph:
            self.graph.close()
            self.graph = None
        for _, handle, destroy in reversed(self.handles):
            destroy(handle)
        self.handles.clear()
