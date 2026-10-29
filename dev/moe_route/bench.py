"""Exact integer checks and isolated explicit-stage graphs; no kernel-parameter API."""
import ctypes as C
from contextlib import ExitStack
from dataclasses import asdict

from dev.moe_route.protocol import Call, Case, Config, Sizes, ids_for, oracle
from quactlize.runtime.native import SDK, checked


def bind(lib, name, args):
    fn = getattr(lib, name); fn.argtypes = args; fn.restype = C.c_int
    return fn


class Runtime(SDK):
    def __init__(self, sdk):
        super().__init__(sdk)
        signatures = {
            'StreamCreateWithFlags': [C.POINTER(C.c_void_p), C.c_uint],
            'StreamDestroy': [C.c_void_p], 'StreamBeginCapture': [C.c_void_p, C.c_int],
            'StreamEndCapture': [C.c_void_p, C.POINTER(C.c_void_p)],
            'GraphInstantiateWithFlags': [C.POINTER(C.c_void_p), C.c_void_p, C.c_uint64],
            'GraphLaunch': [C.c_void_p, C.c_void_p], 'GraphDestroy': [C.c_void_p],
            'GraphExecDestroy': [C.c_void_p], 'ProfilerStart': [], 'ProfilerStop': [],
            'MemsetAsync': [C.c_void_p, C.c_int, C.c_size_t, C.c_void_p],
            'MemcpyAsync': [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_void_p],
        }
        for name, args in signatures.items(): setattr(self, name, bind(self.lib, 'hggc'+name, args))
        self.stream = C.c_void_p(); checked(self.StreamCreateWithFlags(C.byref(self.stream), 1), 'route stream')

    def sync(self): self.synchronize(self.stream)

    def write(self, ptr, data):
        data = bytes(data)
        if data:
            host = C.create_string_buffer(data)
            checked(self.MemcpyAsync(ptr, host, len(data), 1, self.stream), 'route H2D')
            self.sync()  # Keep host bytes alive until the same-stream transfer completes.

    def close(self):
        self.sync(); checked(self.StreamDestroy(self.stream), 'destroy route stream')


class Buffer:
    def __init__(self, rt, size):
        self.rt, self.size = rt, size
        self.base = rt.allocate(size+512); self.ptr = self.base+256
        self.poison()

    def poison(self): checked(self.rt.MemsetAsync(self.base, 0xa5, self.size+512, self.rt.stream), 'route poison')
    def upload(self, data): self.rt.write(self.ptr, data)
    def read(self): return self.rt.download(self.ptr, self.size) if self.size else b''

    def guard(self):
        for ptr in (self.base, self.ptr+self.size):
            if self.rt.download(ptr, 256) != b'\xa5'*256: raise ValueError('route buffer guard changed')

    def close(self): self.rt.free(self.base)


class Graph:
    def __init__(self, rt, launch, repeats):
        self.rt, self.graph, self.executable = rt, C.c_void_p(), C.c_void_p()
        checked(rt.StreamBeginCapture(rt.stream, 0), 'route capture begin')
        try:
            for _ in range(repeats): checked(launch(), 'route capture launch')
        except BaseException:
            rt.StreamEndCapture(rt.stream, C.byref(self.graph))
            if self.graph.value: rt.GraphDestroy(self.graph)
            raise
        checked(rt.StreamEndCapture(rt.stream, C.byref(self.graph)), 'route capture end')
        rc = rt.GraphInstantiateWithFlags(C.byref(self.executable), self.graph, 0)
        if rc: rt.GraphDestroy(self.graph)
        checked(rc, 'route instantiate')

    def run(self): checked(self.rt.GraphLaunch(self.executable, self.rt.stream), 'route graph launch')
    def close(self):
        self.rt.sync()
        checked(self.rt.GraphExecDestroy(self.executable), 'route executable free')
        checked(self.rt.GraphDestroy(self.graph), 'route graph free')


class Library:
    def __init__(self, candidate, native):
        self.lib = C.CDLL(str(candidate), mode=C.RTLD_LOCAL)
        self.query = bind(self.lib, 'quactlize_ppu_moe_route_query_v1', [C.POINTER(Call), C.POINTER(Config), C.POINTER(Sizes)])
        self.run = bind(self.lib, 'quactlize_ppu_moe_route_run_v1', [C.POINTER(Call), C.POINTER(Config)])
        self.stage = bind(self.lib, 'qzr_test_stage_v1', [C.POINTER(Call), C.POINTER(Config), C.c_int])
        self.register = bind(self.lib, 'qzr_test_register_v1', [C.POINTER(Call)])
        self.ballot = bind(self.lib, 'qzr_test_ballot_v1', [C.POINTER(Call), C.c_int])
        other = C.CDLL(str(native), mode=C.RTLD_LOCAL)
        self.native = bind(other, 'qzr_native_run_v1', [C.POINTER(Call), C.c_int])
        name, ordinal, cu, warp = C.create_string_buffer(256), C.c_int(), C.c_int(), C.c_int()
        get_device = bind(self.lib, 'qzr_test_device_v1', [C.c_void_p]+[C.POINTER(C.c_int)]*3)
        checked(get_device(name, C.byref(ordinal), C.byref(cu), C.byref(warp)), 'route device query')
        self.device = dict(name=name.value.decode(), ordinal=ordinal.value, compute_units=cu.value, warp_size=warp.value)
        if warp.value != 32: raise ValueError('route candidate requires a 32-lane physical warp')


class Fixture:
    def __init__(self, rt, lib, case, algorithm, tile):
        self.rt, self.lib, self.case = rt, lib, case
        self.config = Config(1, C.sizeof(Config), algorithm, tile)
        self.call = Call(version=1, size=C.sizeof(Call), tokens=case.tokens, experts=case.experts,
                         topk=case.topk, channels=case.channels, inverse=case.inverse,
                         ids_stride=case.topk+case.padding, source_stride=case.channels+case.padding,
                         stream=rt.stream.value)
        self.sizes = Sizes(); checked(lib.query(C.byref(self.call), C.byref(self.config), C.byref(self.sizes)), 'route query')
        rows = case.tokens*case.topk
        self.buffers = {}
        try:
            for field, size in [('ids', case.tokens*(case.topk+case.padding)*4), ('src_rows', rows*4),
                                ('dst_rows', rows*4), ('offsets', (case.experts+1)*4), ('workspace', self.sizes.workspace_bytes)]:
                b = Buffer(rt, size); self.buffers[field] = b; setattr(self.call, field, b.ptr)
            self.call.workspace_bytes = self.sizes.workspace_bytes
        except BaseException:
            self.close(); raise

    def update(self, seed):
        self.ids = ids_for(self.case, seed)
        for buf in self.buffers.values(): buf.poison()
        self.buffers['ids'].upload(self.ids.tobytes())
        self.expected = oracle(self.ids, self.case)

    def check(self):
        import numpy as np
        self.rt.sync()
        for name, expected in zip(('src_rows', 'dst_rows', 'offsets'), self.expected):
            actual = np.frombuffer(self.buffers[name].read(), dtype=np.int32)
            if not np.array_equal(actual, expected):
                bad = np.flatnonzero(actual != expected)
                raise ValueError(f'route {name} differs at {int(bad[0])}; case={asdict(self.case)}')
        if self.buffers['ids'].read() != self.ids.tobytes(): raise ValueError('IDs input changed')
        for b in self.buffers.values(): b.guard()

    def launch(self, stage=-1): return self.lib.stage(C.byref(self.call), C.byref(self.config), stage)
    def close(self):
        self.rt.sync()
        for b in self.buffers.values(): b.close()


def stream_gate(sdk, lib):
    with ExitStack() as stack:
        fixtures, graphs = [], []
        for variant, tokens in (('tiled', 513), ('register', 17), ('ballot', 17), ('scalar', 17)):
            for seed in (3, 19):
                rt = Runtime(sdk); stack.callback(rt.close)
                case = Case(tokens, 33, 6, 3, padding=5, inverse=1)
                b = Fixture(rt, lib, case, 2 if variant=='tiled' else 1, 512); stack.callback(b.close)
                if variant=='tiled': launch=b.launch
                elif variant=='register': launch=lambda b=b: lib.register(C.byref(b.call))
                else: launch=lambda b=b, scalar=int(variant=='scalar'): lib.ballot(C.byref(b.call),scalar)
                b.update(seed); checked(launch(), 'two-stream initialize')
                g = Graph(rt, launch, 1); stack.callback(g.close)
                fixtures.append(b); graphs.append(g)
        for replay in range(3):
            for i, b in enumerate(fixtures): b.update(replay+41*i)
            for g in graphs: g.run()
            for b in fixtures: b.check()


def measure(sdk, receipt, cases, rounds=4, samples=3, repeats=32):
    with ExitStack() as stack:
        rt = Runtime(sdk); stack.callback(rt.close)
        lib = Library(receipt['candidate'], receipt['native'])
        stream_gate(sdk, lib)
        points, components, graphs, fixtures = [], {}, {}, []
        for case in cases:
            entries = [('tile256', 2, 256), ('tile512', 2, 512), ('tile1024', 2, 1024)]
            if case.tokens <= 8192: entries.insert(0, ('warp', 1, 512))
            if 0 < case.tokens*case.topk <= 256: entries.insert(1, ('register', 1, 512))
            if 0 < case.tokens*case.topk <= 512: entries[1:1]=[('ballot',1,512),('scalar',1,512)]
            # AUTO aliases one tested body, not another timed copy.
            p = dict(case=asdict(case), key=case.key, components={},
                     auto='warp' if case.tokens*case.topk <= 1024 else 'tile512', native_status='SUPPORTED')
            for arm, algorithm, tile in [*entries, ('native', 2, 512)]:
                b = Fixture(rt, lib, case, algorithm, tile); stack.callback(b.close)
                if arm == 'native':
                    rc = lib.native(C.byref(b.call), 0)
                    if rc == 37:
                        p['native_status'] = 'UNSUPPORTED_NATIVE_RESOURCE_OR_INDEX_LIMIT'; continue
                    checked(rc, 'native capability')
                    launch = lambda b=b: lib.native(C.byref(b.call), 1)
                elif arm == 'register':
                    launch = lambda b=b: lib.register(C.byref(b.call))
                elif arm in ('ballot','scalar'):
                    launch = lambda b=b, scalar=int(arm=='scalar'): lib.ballot(C.byref(b.call),scalar)
                else:
                    launch = b.launch
                for seed in range(3):
                    b.update(seed); checked(launch(), 'route numerical'); b.check()
                replay = Graph(rt, launch, 1); stack.callback(replay.close)
                b.update(37); replay.run(); b.check()
                # Leave all producer planes initialized for repeated isolated stages.
                b.update(0); checked(launch(), 'route timing initialization'); b.check()
                keys = []
                stages = 1 if arm == 'native' else b.sizes.stages
                for stage in range(stages):
                    key = f'{case.key}/{arm}/{stage}'; keys.append(key)
                    call = launch if arm in ('native','register','ballot','scalar') else lambda b=b, s=stage: b.launch(s)
                    g = Graph(rt, call, repeats); stack.callback(g.close); graphs[key] = g
                    g.run(); rt.sync()
                    if arm == 'native': kernel, grid, block = 'mm_ids_helper<', case.experts, 32
                    elif arm == 'register': kernel, grid, block = 'warp_register<', case.experts, 32
                    elif arm in ('ballot','scalar'): kernel, grid, block = 'warp_'+arm+'<', case.experts, 32
                    elif not case.tokens: kernel, grid, block = 'empty_offsets(', (case.experts+256)//256, 256
                    elif algorithm == 1: kernel, grid, block = 'warp_compact<', case.experts, 32
                    elif stage == 0: kernel, grid, block = 'histogram_rank<', b.sizes.tiles, 128
                    elif stage == 1: kernel, grid, block = 'prefix_chunks(', 1, ((case.experts+31)//32)*32
                    else: kernel, grid, block = 'scatter_maps<', (case.tokens*case.topk+255)//256, 256
                    components[key] = dict(component=key, kernel=kernel, grid=[grid, 1, 1], block=[block, 1, 1], repeats=repeats)
                p['components'][arm] = keys
                b.check()
                fixtures.append(b)
            points.append(p)
            print('MOE_ROUTE_READY '+case.key+' M='+str(case.tokens), flush=True)
        rt.sync(); schedule = []
        checked(rt.ProfilerStart(), 'route profiler start')
        try:
            keys = list(graphs)
            for r in range(rounds):
                order = keys if r % 2 == 0 else keys[::-1]
                for key in order:
                    for s in range(samples):
                        graphs[key].run(); rt.sync()
                        schedule.append(dict(components[key], round=r, sample=s))
        finally:
            rt.sync(); checked(rt.ProfilerStop(), 'route profiler stop')
        for b in fixtures: b.check()
        return dict(points=points, schedule=schedule, device=lib.device, correctness='PASS',
                    protocol=dict(rounds=rounds, samples=samples, repeats=repeats),
                    timing_status='CAPTURED_NOT_TIMED', production_changed=False)
