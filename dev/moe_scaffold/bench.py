"""Fused helpers vs same-arithmetic decompositions; GEMM producers are fixtures."""
import ctypes as C
from contextlib import ExitStack
from dataclasses import asdict

import numpy as np

from dev.moe_route.bench import Runtime, Buffer, Graph, bind
from dev.moe_scaffold.protocol import (Call, Config, Sizes, Value, Prepare, Transition, Finish,
    Case, ids_for, oracle, encode, decode, rounded, projected, activation, weighted)
from quactlize.runtime.native import checked


class Library:
    def __init__(self, path):
        self.lib = C.CDLL(str(path), mode=C.RTLD_LOCAL)
        for name, suffix, args in (
            ('query', 'prepare_query_v1', [C.POINTER(Prepare), C.POINTER(Sizes)]),
            ('prepare', 'prepare_run_v1', [C.POINTER(Prepare)]),
            ('transition_query', 'transition_query_v1', [C.POINTER(Transition), C.POINTER(C.c_uint)]),
            ('transition', 'transition_run_v1', [C.POINTER(Transition)]),
            ('finish', 'finish_run_v1', [C.POINTER(Finish)])):
            setattr(self, name, bind(self.lib, 'quactlize_ppu_moe_'+suffix, args))
        for name, symbol, args in (
            ('prepare_stage', 'qzm_test_prepare_stage_v1', [C.POINTER(Prepare), C.c_int]),
            ('prepare_unfused', 'qzm_test_prepare_unfused_v1', [C.POINTER(Prepare), C.c_int]),
            ('materialize', 'qzm_test_materialize_v1', [C.POINTER(Value), C.c_int, C.c_int, C.c_void_p, C.c_void_p]),
            ('cast', 'qzm_test_cast_v1', [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int64, C.c_void_p]),
            ('weight', 'qzm_test_weight_v1', [C.POINTER(Finish), C.c_void_p, C.c_void_p]),
            ('sum', 'qzm_test_sum_v1', [C.POINTER(Finish), C.c_void_p])):
            setattr(self, name, bind(self.lib, symbol, args))
        name, ordinal, cu, warp = C.create_string_buffer(256), C.c_int(), C.c_int(), C.c_int()
        query = bind(self.lib, 'qzr_test_device_v1', [C.c_void_p]+[C.POINTER(C.c_int)]*3)
        checked(query(name, C.byref(ordinal), C.byref(cu), C.byref(warp)), 'helper device')
        self.device = dict(name=name.value.decode(), ordinal=ordinal.value, compute_units=cu.value, warp_size=warp.value)
        if warp.value != 32: raise ValueError('helper requires warp size 32')


class Fixture:
    def __init__(self, rt, lib, case):
        self.rt, self.lib, self.case, self.buffers = rt, lib, case, {}
        self.rows = case.tokens*case.topk
        self.gcols = case.hidden*(2 if case.merged and case.state != 2 else 1)
        route = Call(version=1, size=C.sizeof(Call), tokens=case.tokens, experts=case.experts,
                     topk=case.topk, channels=1, ids_stride=case.topk+3, source_stride=4, stream=rt.stream.value)
        self.p = Prepare(version=1, size=C.sizeof(Prepare), route=route,
                         config=Config(1, C.sizeof(Config), 2, 512), columns=case.columns,
                         dtype=case.dtype, input_row_stride=case.columns+3)
        self.sizes = Sizes(); checked(lib.query(C.byref(self.p), C.byref(self.sizes)), 'helper query')
        try:
            for name, n in [('ids', case.tokens*(case.topk+3)*4), ('src_rows', self.rows*4),
                            ('dst_rows', self.rows*4), ('offsets', (case.experts+1)*4), ('workspace', self.sizes.workspace_bytes)]:
                setattr(self.p.route, name, self.alloc(name, n))
            self.p.route.workspace_bytes = self.sizes.workspace_bytes
            self.input_rows = (case.tokens-1)*4+1 if case.tokens else 0
            for name, n in [('input', self.input_rows*(case.columns+3)*4),
                            ('output', self.rows*case.columns*(4 if case.dtype == 0 else 2)),
                            ('inverse_rows', self.rows*4), ('expert_rows', case.experts*4),
                            ('active_experts', case.experts*4), ('active_count', 4)]:
                setattr(self.p, name, self.alloc('prepare_'+name, n))
            self.t = Transition(version=1, size=C.sizeof(Transition), tokens=case.tokens, topk=case.topk,
                hidden=case.hidden, merged=case.merged, output_dtype=case.dtype, output_order=case.output_order,
                output_stride=case.hidden+3, row_ids=self.p.route.dst_rows, inverse_rows=self.p.inverse_rows,
                output=self.alloc('activation', self.rows*(case.hidden+3)*(4 if case.dtype==0 else 2)), stream=rt.stream.value)
            self.f = Finish(version=1, size=C.sizeof(Finish), tokens=case.tokens, topk=case.topk,
                columns=case.output, inverse_rows=self.p.inverse_rows, weights_stride=case.topk+3,
                output_stride=case.output+3, weights=self.alloc('weights', case.tokens*(case.topk+3)*4),
                output=self.alloc('finish', case.tokens*(case.output+3)*4), stream=rt.stream.value)
            for name, n in [('tmp_gate', self.rows*self.gcols*4), ('tmp_up', self.rows*case.hidden*4),
                            ('tmp_act', self.rows*case.hidden*4), ('tmp_down', self.rows*case.output*4),
                            ('tmp_weighted', self.rows*case.output*4)]: self.alloc(name, n)
            for name, nc, state, splits, order in [
                    ('gate', self.gcols, case.state, case.splits if case.state == 1 else 1, case.gate_order),
                    ('up', case.hidden, 1 if case.state == 1 else 0, case.splits if case.state == 1 else 1, case.up_order),
                    ('down', case.output, int(case.down_splits > 1), case.down_splits, case.down_order)]:
                dtype = 0 if state == 1 else case.dtype
                pitch = self.rows*(nc+3)+5
                v = Value(self.alloc(name, splits*pitch*(4 if dtype == 0 else 2)), dtype, order, state,
                          splits, nc+3, pitch, 0 if state == 2 else case.dtype, 0)
                if name == 'down': self.f.down = v
                else: setattr(self.t, name, v)
        except BaseException:
            self.close(); raise

    def alloc(self, name, size):
        b = Buffer(self.rt, size); self.buffers[name] = b; return b.ptr

    def ptr(self, name): return self.buffers[name].ptr

    def update(self, seed):
        c = self.case; rng = np.random.default_rng(seed)
        for b in self.buffers.values(): b.poison()
        ids = ids_for(c.route_case, seed)
        src, dst, offsets = oracle(ids, c.route_case)
        inverse = np.empty(self.rows, dtype=np.int32); inverse[dst] = np.arange(self.rows, dtype=np.int32)
        self.maps = src, dst, offsets, inverse
        self.buffers['ids'].upload(ids.tobytes())
        a = rng.uniform(-3, 3, (self.input_rows, c.columns+3)).astype(np.float32)
        if a.size and c.dtype == 2: a[0, 0] = 243383.484
        self.buffers['prepare_input'].upload(a.tobytes())
        self.prepared_expected = encode(a[src, :c.columns], c.dtype)
        values = {}
        for name, v, nc in [('gate', self.t.gate, self.gcols), ('up', self.t.up, c.hidden), ('down', self.f.down, c.output)]:
            planes = rng.uniform(-2, 2, (v.partial_count, self.rows, nc)).astype(np.float32)
            if planes.size and c.dtype == 2 and name == 'up': planes[0, 0, 0] = 100000
            # Independent CPU oracle sums partials in declared order; source is
            # then physically permuted, not generated by the device algorithm.
            logical = projected(planes, c.dtype)
            if v.state == 2: logical = rounded(planes[0], c.dtype)
            values[name] = logical
            raw = np.zeros(v.partial_count*v.partial_stride, dtype=np.float32)
            for i in range(v.partial_count):
                rows = planes[i, dst] if v.order == 0 else planes[i]
                raw[i*v.partial_stride:i*v.partial_stride+self.rows*v.row_stride].reshape(self.rows, v.row_stride)[:, :nc] = rows
            self.buffers[name].upload(encode(raw, v.dtype).tobytes())
        if c.state == 2: act = values['gate']
        else:
            gate = values['gate'][:, :c.hidden]
            up = values['gate'][:, c.hidden:] if c.merged else values['up']
            act = activation(gate, up)
        self.activation_expected = rounded(act[dst] if c.output_order == 0 else act, c.dtype)
        w = rng.uniform(0, 1, (c.tokens, c.topk+3)).astype(np.float32)
        self.buffers['weights'].upload(w.tobytes())
        self.finish_expected = weighted(values['down'].reshape(c.tokens, c.topk, c.output), w[:, :c.topk])

    def prepare_check(self):
        self.rt.sync(); c = self.case
        for name, expected in zip(('src_rows', 'dst_rows', 'offsets', 'prepare_inverse_rows'), self.maps):
            np.testing.assert_array_equal(np.frombuffer(self.buffers[name].read(), np.int32), expected)
        counts = np.diff(self.maps[2]); active = np.flatnonzero(counts).astype(np.int32)
        np.testing.assert_array_equal(np.frombuffer(self.buffers['prepare_expert_rows'].read(), np.int32), counts)
        np.testing.assert_array_equal(np.frombuffer(self.buffers['prepare_active_count'].read(), np.int32), [len(active)])
        np.testing.assert_array_equal(np.frombuffer(self.buffers['prepare_active_experts'].read(), np.int32)[:len(active)], active)
        want = self.prepared_expected.tobytes()
        if self.buffers['prepare_output'].read() != want: raise ValueError('prepare dtype/gather bits differ')

    def values(self, name, dtype, rows, stride, columns):
        storage = np.float32 if dtype == 0 else np.float16 if dtype == 1 else np.uint16
        return decode(np.frombuffer(self.buffers[name].read(), storage).reshape(rows, stride)[:, :columns], dtype)

    def numerical(self):
        c = self.case
        checked(self.lib.prepare(C.byref(self.p)), 'fused prepare'); self.prepare_check()
        for stage in range(1 if not c.tokens else 5): checked(self.lib.prepare_unfused(C.byref(self.p), stage), 'unfused prepare')
        self.prepare_check()
        checked(self.lib.transition(C.byref(self.t)), 'fused transition'); self.rt.sync()
        act = self.buffers['activation'].read()
        actual = self.values('activation', c.dtype, self.rows, c.hidden+3, c.hidden)
        # expf/libm differences may move the last BF16/FP16 bit. The device
        # decomposition below must still match the fused device result EXACTLY.
        np.testing.assert_allclose(actual, self.activation_expected, rtol=(3e-6,.0011,.009)[c.dtype], atol=2e-6)
        for call, _, _ in self.transition_control(): checked(call(), 'unfused transition')
        self.rt.sync()
        if self.buffers['activation'].read() != act: raise ValueError('fused vs unfused transition bits differ')
        checked(self.lib.finish(C.byref(self.f)), 'fused finish'); self.rt.sync()
        finish = self.buffers['finish'].read()
        np.testing.assert_array_equal(self.values('finish', 0, c.tokens, c.output+3, c.output), self.finish_expected)
        for call, _, _ in self.finish_control(): checked(call(), 'unfused finish')
        self.rt.sync()
        if self.buffers['finish'].read() != finish: raise ValueError('fused vs unfused finish bits differ')
        for b in self.buffers.values(): b.guard()
        if c.state == 2:
            noop = Transition.from_buffer_copy(self.t)
            noop.output=noop.gate.data;noop.output_dtype=noop.gate.dtype
            noop.output_order=noop.gate.order;noop.output_stride=noop.gate.row_stride
            ops=C.c_uint();checked(self.lib.transition_query(C.byref(noop),C.byref(ops)), 'activated no-op query')
            if ops.value != 16: raise ValueError('already-activated view was not a no-op')
            checked(self.lib.transition(C.byref(noop)), 'activated no-op')

    def transition_control(self):
        c = self.case; result=[]
        def materialize(name, v, nc):
            return (lambda: self.lib.materialize(C.byref(v),self.rows,nc,self.ptr('tmp_'+name),self.rt.stream),
                    'materialize(', self.flat(self.rows,nc))
        result.append(materialize('gate', self.t.gate, self.gcols))
        if not c.merged and c.state != 2: result.append(materialize('up',self.t.up,c.hidden))
        t=Transition.from_buffer_copy(self.t)
        t.gate=Value(self.ptr('tmp_gate'),0,c.gate_order,2 if c.state==2 else 0,1,self.gcols,0,0,0)
        t.up=Value(self.ptr('tmp_up'),0,c.up_order,0,1,c.hidden,0,0,0)
        t.output=self.ptr('tmp_act');t.output_dtype=0;t.output_stride=c.hidden
        result.append((lambda: self.lib.transition(C.byref(t)),'transition_kernel(', self.row_grid(self.rows,c.hidden)))
        result.append((lambda: self.lib.cast(self.ptr('tmp_act'),self.t.output,self.rows,c.hidden,c.dtype,
                       self.t.output_stride,self.rt.stream),'cast_materialized(',self.flat(self.rows,c.hidden)))
        return result

    def finish_control(self):
        c=self.case
        return [(lambda:self.lib.materialize(C.byref(self.f.down),self.rows,c.output,self.ptr('tmp_down'),self.rt.stream),
                 'materialize(',self.flat(self.rows,c.output)),
                (lambda:self.lib.weight(C.byref(self.f),self.ptr('tmp_down'),self.ptr('tmp_weighted')),
                 'weight_slots(',self.flat(self.rows,c.output)),
                (lambda:self.lib.sum(C.byref(self.f),self.ptr('tmp_weighted')),
                 'sum_slots(',self.flat(c.tokens,c.output))]

    @staticmethod
    def flat(rows, columns): return [min((rows*columns+255)//256,65535),1,1]
    @staticmethod
    def row_grid(rows, columns): return [min(rows*((columns+255)//256),65535),1,1]

    def component_groups(self):
        c=self.case
        if not c.tokens:return {}
        grouped={}
        for arm, count, method in [('fused',3,self.lib.prepare_stage),('unfused',5,self.lib.prepare_unfused)]:
            items=[]
            for stage in range(count):
                if stage==0: symbol,grid,threads='histogram_rank<',[self.sizes.tiles,1,1],128
                elif stage==1: symbol,grid,threads=('prepare_prefix<' if arm=='fused' else 'prefix_chunks('),[1,1,1],((c.experts+31)//32)*32
                elif arm=='unfused' and stage==2: symbol,grid,threads='scatter_maps<',[(self.rows+255)//256,1,1],256
                elif arm=='unfused' and stage==3: symbol,grid,threads='prepare_prefix<',[1,1,1],((c.experts+31)//32)*32
                else: symbol,grid,threads='prepare_gather<',self.row_grid(self.rows,c.columns),256
                items.append((lambda s=stage,m=method:m(C.byref(self.p),s),symbol,grid,[threads,1,1]))
            grouped['prepare/'+arm]=items
        grouped['transition/fused']=[(lambda:self.lib.transition(C.byref(self.t)),'transition_kernel(',self.row_grid(self.rows,c.hidden),[256,1,1])]
        grouped['transition/unfused']=[(*x,[256,1,1]) for x in self.transition_control()]
        grouped['finish/fused']=[(lambda:self.lib.finish(C.byref(self.f)),'finish_kernel(',self.row_grid(c.tokens,c.output),[256,1,1])]
        grouped['finish/unfused']=[(*x,[256,1,1]) for x in self.finish_control()]
        return grouped

    def close(self):
        self.rt.sync()
        for b in self.buffers.values(): b.close()


def measure(sdk, receipt, cases, rounds=4, samples=2, repeats=32):
    points,schedule=[],[]
    with ExitStack() as root:
        rt=Runtime(sdk);root.callback(rt.close);lib=Library(receipt['candidate'])
        # Two independent per-stream scratch owners, changing IDs and graph
        # replay. No graph cloning or KernelNodeGetParams is used.
        with ExitStack() as gate:
            fixtures=[];graphs=[]
            for i in range(2):
                r=Runtime(sdk);gate.callback(r.close)
                b=Fixture(r,lib,Case(17,experts=33,topk=6,state=1,splits=2,down_splits=4))
                gate.callback(b.close);b.update(i);b.numerical()
                def call(b=b):
                    for fn,arg in ((lib.prepare,b.p),(lib.transition,b.t),(lib.finish,b.f)):
                        rc=fn(C.byref(arg))
                        if rc:return rc
                    return 0
                g=Graph(r,call,1);gate.callback(g.close);fixtures.append(b);graphs.append(g)
            for seed in (23,41):
                for i,b in enumerate(fixtures):b.update(seed+i)
                for g in graphs:g.run()
                for b in fixtures:
                    b.prepare_check()
                    np.testing.assert_allclose(b.values('activation',b.case.dtype,b.rows,b.case.hidden+3,b.case.hidden),
                                               b.activation_expected,rtol=.009,atol=2e-6)
                    np.testing.assert_array_equal(b.values('finish',0,b.case.tokens,b.case.output+3,b.case.output),b.finish_expected)
        for case in cases:
            with ExitStack() as stack:
                b=Fixture(rt,lib,case);stack.callback(b.close)
                for seed in (7,19):b.update(seed);b.numerical()
                # Independently exercise the bounded WARP algorithm, then
                # restore the scalable path before isolated stage profiling.
                if case.tokens<=8192:
                    b.p.config.algorithm=1
                    checked(lib.prepare(C.byref(b.p)),'WARP prepare');b.prepare_check()
                    b.p.config.algorithm=2
                b.update(31);b.numerical()
                groups=b.component_groups();graphs={};entries={};case_points={}
                for group,components in groups.items():
                    phase,arm=group.split('/')
                    p=case_points.setdefault(phase,dict(key=case.key+'/'+phase,case=asdict(case),phase=phase,components={}))
                    keys=[]
                    for i,(launch,symbol,grid,block) in enumerate(components):
                        key=case.key+'/'+group+'/'+str(i);keys.append(key)
                        graph=Graph(rt,launch,repeats);stack.callback(graph.close);graphs[key]=graph
                        graph.run();rt.sync()
                        entries[key]=dict(component=key,kernel=symbol,grid=grid,block=block,repeats=repeats)
                    p['components'][arm]=keys
                rt.sync();checked(rt.ProfilerStart(),'scaffold profiler begin')
                try:
                    keys=list(graphs)
                    for r in range(rounds):
                        for key in (keys if r%2==0 else keys[::-1]):
                            for s in range(samples):
                                graphs[key].run();rt.sync();schedule.append(dict(entries[key],round=r,sample=s))
                finally:
                    rt.sync();checked(rt.ProfilerStop(),'scaffold profiler end')
                for buffer in b.buffers.values():buffer.guard()
                points.extend(case_points.values())
                print('MOE_SCAFFOLD_POINT '+case.key+' M='+str(case.tokens)+' exact_device_controls=PASS',flush=True)
        return dict(points=points,schedule=schedule,device=lib.device,correctness='PASS',
                    cases=[asdict(c) for c in cases],production_changed=False,
                    scope='HELPERS_ONLY_SYNTHETIC_PROJECTION_OUTPUTS_NO_GEMM_OR_MODEL')
