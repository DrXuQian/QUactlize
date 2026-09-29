"""Whole-call providers and numerical admission for one resident fixture."""

import ctypes as C
from dataclasses import asdict
import numpy as np

from dev.gemv_simt.native import Graph, checked
from dev.gemv_simt.production import Library as ShippingSimt
from dev.gemv_model.plan import Candidate, candidates
from quactlize.execution.native import Call, SimtCallV2, arrangement
from quactlize.execution.simt_codegen import Config as SimtConfig
from quactlize.fusion.native import Library as FusionLibrary, MappedCall, FusionCall, Config as FusionConfig, Layout, integration_entries
from quactlize.decode.native_compute import bind_compute, DenseIO, DenseComputeCall
from quactlize.runtime.native import Call as TcCall, Recipe, Resources
from dev.gate_up_perf.bench import Buffer


class Provider:
    def __init__(self, bench, bundle, record, arm):
        self.b, self.arm, self.handles = bench, arm, []
        p = bench.point
        configs = [Candidate(**c) for c in record['candidates']] if 'candidates' in record else candidates(p)
        self.config = configs[0 if arm == "incumbent" else int(arm)]
        self.fusion = FusionLibrary(bundle / "libquactlize_ppu_gate_up.so") if p.paired else None
        if self.fusion:
            self.layout = self.fusion.arrangement(p.q)
        if arm != "incumbent":
            self.lib = C.CDLL(str(bundle / record["library"]), mode=C.RTLD_LOCAL)
            self.fn = self.lib.model_gemv_run
            self.fn.argtypes = [C.POINTER(SimtCallV2), C.POINTER(MappedCall), C.POINTER(Layout), C.c_int]
            self.fn.restype = C.c_int
        elif p.paired:
            self.fn = integration_entries(self.fusion)["run"]
        elif not p.tc:
            self.shipping = ShippingSimt(bundle / "libquactlize_ppu_execution.so", p.compute)
        else:
            self.record = record["tc"]
            self.lib = C.CDLL(str(bundle / self.record["path"]), mode=C.RTLD_LOCAL)
            self.identity, self.query, self.prepare_tc, self.run_tc, self.destroy = bind_compute(self.lib, False)
            ident = self.identity().contents
            if ident.compute_type != p.compute or ident.parent.contents.build_key.decode() != self.record["key"]:
                raise ValueError("TC control build/compute identity differs")
            actual = ident.parent.contents
            for attr, key in (("tm", "tm"), ("tn", "tn"), ("tk", "tk"), ("wm", "wm"), ("wn", "wn"),
                              ("stages", "stages"), ("ap", "ap"), ("delivery_n", "dn"), ("qtype", "qtype")):
                if getattr(actual, attr) != p.parent[key]:
                    raise ValueError("frozen TC parent differs: " + key)

    @property
    def receipt(self):
        p = self.b.point
        return dict(arm=self.arm, name="incumbent" if self.arm == "incumbent" else self.config.name,
                    kind="tc" if self.arm == "incumbent" and p.tc else "simt",
                    config=p.parent | dict(split=p.tc[-1]) if self.arm == "incumbent" and p.tc else asdict(self.config),
                    scope="PAIRED_GATE_UP_PLUS_SWIGLU" if p.paired else "GEMV_INCLUDING_SPLITK_REDUCTION")

    def prepare(self, copy=0):
        b, p = self.b, self.b.point
        call = b.call(copy)
        if p.paired:
            d = MappedCall(FusionCall(call, p.compute, 1, p.compute),
                           b.mapping.ptr if b.mapping and b.mapped_call else None,
                           b.status.ptr if b.status else None)
            if self.arm == "incumbent":
                cfg = FusionConfig(0, 1, 0, 8)
                return lambda: self.fn(C.byref(d), C.byref(cfg), C.byref(self.layout))
            return lambda: self.fn(None, C.byref(d), C.byref(self.layout), int(self.arm))
        d = SimtCallV2(call, p.compute)
        if self.arm != "incumbent":
            return lambda: self.fn(C.byref(d), None, None, int(self.arm))
        if not p.tc:
            cfg = SimtConfig(**{k: getattr(self.config, k) for k in ("variant", "columns", "warps", "values", "split")})
            return self.shipping.prepare(call, cfg)
        device, cu, name = C.c_int(), C.c_int(), C.create_string_buffer(256)
        probe = self.lib.quactlize_kpack_decode_dense_device_v1
        probe.argtypes = [C.c_char_p, C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_int)]
        probe.restype = C.c_int
        checked(probe(name, 256, C.byref(device), C.byref(cu)), "TC device identity")
        c = TcCall(version=1, size=C.sizeof(TcCall), m=b.rows, n=p.n, k=p.k, experts=1,
                   group_size=arrangement(p.q).group_size, device=device.value, compute_units=cu.value,
                   mapping_id=arrangement(p.q).mapping_id, a=call.a, low=call.low, high=call.high,
                   metadata=call.units, output=call.output, workspace=call.workspace,
                   workspace_bytes=call.workspace_bytes, stream=call.stream)
        typed = DenseComputeCall(2, C.sizeof(DenseComputeCall), DenseIO(1, C.sizeof(DenseIO), c, 1, 1), p.compute)
        recipe = Recipe(1, C.sizeof(Recipe), 0, p.tc[-1], 0)
        resources = Resources()
        checked(self.query(C.byref(typed), C.byref(recipe), C.byref(resources)), "TC control query")
        if resources.workspace_bytes > b.workspace.size:
            raise ValueError("TC control workspace too small")
        handle = C.c_void_p()
        checked(self.prepare_tc(C.byref(typed), C.byref(recipe), C.byref(handle)), "TC control prepare")
        self.handles.append(handle)
        return lambda: self.run_tc(handle, b.rt.stream)

    def close(self):
        for handle in self.handles:
            self.destroy(handle)
        self.handles.clear()


def correctness(provider, repeat_controls=True, token_controls=None):
    b, p, rt = provider.b, provider.b.point, provider.b.rt
    records, snapshots = [], {}
    b.mapped_call = True
    for tokens in (token_controls if token_controls is not None else ((1, 2, 8) if repeat_controls else (1,))):
        b.update(tokens, 0)
        launch = provider.prepare()
        graph = Graph(rt, [launch])
        try:
            for repeat in (0, 1):
                b.update(tokens, repeat, mapped=bool(p.paired and p.mode and repeat))
                b.poison()
                graph.sample()
                error = b.check()
                records.append(dict(tokens=tokens, repeat=repeat, error=error))
                if tokens == 1 and repeat == 0:
                    snapshots["m1"] = b.result().view("u4").copy()
            if p.compute:
                b.update(tokens, 1, large=True)
                b.poison(); graph.sample()
                records.append(dict(tokens=tokens, repeat="bf16-range", error=b.check()))
        finally:
            graph.close()
            provider.close()
    # A planted zero input must disagree with the nonzero independent oracle.
    b.update(1, 0)
    launch = provider.prepare()
    rt.fill(b.a.ptr, b.a.size, 0)
    b.poison(); checked(launch(), "zero-input negative"); rt.sync()
    zero = b.result()
    if not np.isfinite(zero).all() or np.any(zero != 0) or not np.any(np.abs(b.gold) > 0):
        raise ValueError("zero-input negative did not reject the nonzero oracle")
    if p.mode:
        b.update(1, 0)
        for bad_value in (-1, p.experts):
            ids = b.ids_host.copy(); ids[0, 0] = bad_value
            rt.copy(b.ids.ptr, ids)
            b.poison(); checked(launch(), "invalid expert negative"); rt.sync()
            if not np.isnan(b.result()[0]).all():
                raise ValueError("invalid expert ID was not rejected")
    if b.mapping:
        b.update(1, 0)
        bad = np.arange(b.rows, dtype="i4"); bad[0] = b.rows
        rt.copy(b.mapping.ptr, bad)
        b.poison(); checked(launch(), "invalid mapped row negative"); rt.sync()
        if not np.isnan(b.result()[0]).all():
            raise ValueError("invalid compacted row was not rejected")
    if b.status:
        b.update(1, 0); rt.fill(b.status.ptr, 4, 1)
        b.poison(); checked(launch(), "upstream status negative"); rt.sync()
        if not np.isnan(b.result()).all():
            raise ValueError("upstream error status was not propagated")
    provider.close()
    if provider.arm!='incumbent' and provider.config.vector_reduce:
        records.extend(vector_reducer_controls(provider))
    b.update(1, 0)
    return records, snapshots


def vector_reducer_controls(provider):
    """Exercise the row-vector path and its public four-byte fallback untimed."""
    b,p,rt=provider.b,provider.b.point,provider.b.rt
    original=b.call
    mark=len(rt.allocations)
    records=[]
    try:
        for pad,out_offset,partial_offset in ((2,0,0),(2,4,0),(2,0,4),(1,0,0)):
            stride=p.n+pad
            b.update(3,0)
            rows=b.rows
            output=Buffer(rt,rows*stride*4+4)
            partials=Buffer(rt,rows*p.n*provider.config.split*4+4)
            def call(copy=0):
                c=original(copy)
                c.output=output.ptr+out_offset;c.out_row_stride=stride
                c.workspace=partials.ptr+partial_offset;c.workspace_bytes=partials.size-partial_offset
                return c
            b.call=call
            b.update(3,0)
            launch=provider.prepare()
            for repeat in (0,1):
                b.update(3,repeat);output.poison();partials.poison()
                checked(launch(),'row reducer stride/alignment control');rt.sync()
                raw=output.read();words=raw.view('f4')
                positions=out_offset//4+np.arange(rows)[:,None]*stride+np.arange(p.n)
                got=words[positions]
                error=float(np.max(np.abs(got.astype('f8')-b.gold)/np.maximum(b.denom,1e-20)))
                if not np.isfinite(got).all() or not np.isfinite(error) or error>=.005:
                    raise ValueError('row reducer independent oracle differs')
                owned=np.zeros(raw.size,dtype=bool)
                owned[(positions[...,None]*4+np.arange(4)).reshape(-1)]=True
                if not np.all(raw[~owned]==0xA5):
                    raise ValueError('row reducer wrote output padding')
                partials.check_guard();b.a.check_guard()
                raw_partials=partials.read()
                end=partial_offset+rows*p.n*provider.config.split*4
                if not np.all(raw_partials[:partial_offset]==0xA5) or not np.all(raw_partials[end:]==0xA5):
                    raise ValueError('producer wrote outside partial extent')
                records.append(dict(tokens=3,repeat=repeat,error=error,reducer_pad=pad,
                                    output_offset=out_offset,partial_offset=partial_offset))
            provider.close()
    finally:
        b.call=original
        provider.close()
        rt.release_after(mark)
    return records


def timing_graph(provider):
    b = provider.b
    b.update(1, 0)
    b.mapped_call = False
    calls = [provider.prepare(i) for i in range(b.copies)]
    # Every loop still traverses the full cold working set in order.
    return Graph(b.rt, calls * max(1, (32 + len(calls) - 1) // len(calls)))
