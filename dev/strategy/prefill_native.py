"""Current native F16 dense and v4 BF16 grouped TC, never decode-only bind."""
import ctypes as C

from dev.smallm_closure.bench import Buffer
from dev.strategy.resources import NativeProbe, request_abi
from quactlize.runtime.native import Recipe as Launch, Resources as Query, checked
from tools.run_kpack_gemv_gate import Resources


class TensorCore:
    def __init__(self, b, arm, record, query, sf=None):
        self.b, self.r = b, Resources(b.sdk)
        self.handles, self.buffers = [], []
        self.sf = sf if arm['recipe']['route'] == 'sf' and b.w.qtype != 8 else None
        try:
            probe = NativeProbe(arm['spec'], record)
            self.library = probe.lib
            grouped = arm['spec']['endpoint'] == 'grouped'
            if not grouped and arm['spec']['endpoint'] != 'native':
                raise ValueError('prefill TC must use native F16 or general typed grouped endpoint')
            request = request_abi(query['request'], probe.device)
            call = request.device_call.call if grouped else request
            launch = arm['binding']['launch']
            self.recipe = Launch(1, C.sizeof(Launch), launch['algorithm'], launch['split'], launch['grid'])
            get = probe._bind('grouped_query_v4' if grouped else 'query_v1',
                [C.POINTER(type(request)), C.POINTER(Launch), C.POINTER(Query)])
            actual = Query()
            checked(get(C.byref(request), C.byref(self.recipe), C.byref(actual)), 'prefill TC resources')
            for field in ('workspace_bytes', 'shared_bytes', 'occupancy'):
                if getattr(actual, field) != query[field]: raise ValueError('TC resource changed: '+field)
            def alloc(size):
                out = Buffer(self.r, size); self.buffers.append(out); return out
            self.a = alloc(b.rows*b.w.k*2)
            self.out = alloc(b.rows*b.w.n*2)
            scratch = alloc(actual.workspace_bytes)
            call.a, call.output = self.a.ptr, self.out.ptr
            call.offsets_device = b.offsets.ptr if grouped else None
            call.workspace, call.workspace_bytes, call.stream = scratch.ptr, scratch.size, b.r.stream.value
            prepare = probe._bind('grouped_prepare_v4' if grouped else 'prepare_v1',
                [C.POINTER(type(request)), C.POINTER(Launch), C.POINTER(C.c_void_p)])
            self.run_fn = probe._bind('run_v1', [C.c_void_p, C.c_void_p])
            self.destroy = probe._bind('destroy_v1', [C.c_void_p], None)
            for plane in (self.sf.planes if self.sf else b.ring.planes):
                call.low, call.high, call.metadata = plane['low'], plane['high'], plane['units']
                call.zero = plane.get('zero')
                h = C.c_void_p()
                checked(prepare(C.byref(request), C.byref(self.recipe), C.byref(h)), 'prefill TC prepare')
                self.handles.append(h)
            self.receipt = dict(endpoint=arm['spec']['endpoint'], compute=b.w.compute,
                module_sha256=record['sha256'], build_key=record['key'], launch=launch,
                metadata_type=request.metadata_type if grouped else 0,
                shared_bytes=actual.shared_bytes, workspace_bytes=actual.workspace_bytes,
                occupancy=actual.occupancy, compute_units=probe.device['compute_units'],
                scope='GENERAL_GPU_OFFSETS_METADATA_NOT_HOST_COMPACT_NOT_DECODE_BIND',
                row_maps='RESIDENT_INPUTS_CREATION_NOT_TIMED')
        except BaseException:
            self.close(); raise

    def launch(self, copy=0):
        checked(self.b.input(self.a.ptr, self.b.w.compute == 'bf16'), 'prefill input adapter')
        checked(self.run_fn(self.handles[copy], self.b.r.stream), 'prefill TC call')
        return self.b.finish(self.out.ptr, self.b.w.compute == 'bf16')

    def setup(self, config=0, *, validate=True):
        if self.sf:
            for i in range(self.b.ring.copies): checked(self.sf.launch(i, config), 'untimed SF preparation')
            if validate:
                self.sf.check(0); self.sf.check(self.b.ring.copies-1)

    def close(self):
        self.b.sdk.synchronize(self.b.r.stream)
        for h in self.handles: self.destroy(h)
        self.handles = []
        self.r.close()
