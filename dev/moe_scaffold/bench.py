"""Fused helpers vs same-arithmetic decompositions; GEMM producers are fixtures."""
import ctypes as C
from contextlib import ExitStack
from dataclasses import asdict
import json

import numpy as np

from dev.moe_route.bench import Runtime, Buffer, Graph, bind
from dev.moe_scaffold.protocol import (Call, Config, Sizes, Value, Prepare, Transition, Finish,
    Case, Plan, PreparePlanV2, ids_for, oracle, encode, decode, rounded, activation, weighted)
from quactlize.runtime.native import checked
from dev.strategy.refresh_clock import ADVISORY, observe, pci_key, request_lock


class Library:
    def __init__(self, path):
        self.lib = C.CDLL(str(path), mode=C.RTLD_LOCAL)
        for name, suffix, args in (
            ('query', 'prepare_query_v1', [C.POINTER(Prepare), C.POINTER(Sizes)]),
            ('prepare', 'prepare_run_v1', [C.POINTER(Prepare)]),
            ('transition_query', 'transition_query_v1', [C.POINTER(Transition), C.POINTER(C.c_uint)]),
            ('transition', 'transition_run_v1', [C.POINTER(Transition)]),
            ('finish', 'finish_run_v1', [C.POINTER(Finish)]),
            ('prepare_select', 'prepare_select_v1', [C.POINTER(Prepare), C.c_int, C.POINTER(Plan)]),
            ('prepare_execute', 'prepare_execute_v1', [C.POINTER(Prepare), C.POINTER(Plan)]),
            ('prepare_select_v2', 'prepare_select_v2', [C.POINTER(Prepare), C.POINTER(PreparePlanV2)]),
            ('prepare_execute_v2', 'prepare_execute_v2', [C.POINTER(Prepare), C.POINTER(PreparePlanV2)]),
            ('finish_select', 'finish_select_v1', [C.POINTER(Finish), C.c_int, C.POINTER(Plan)]),
            ('finish_execute', 'finish_execute_v1', [C.POINTER(Finish), C.POINTER(Plan), C.c_void_p, C.c_uint64])):
            setattr(self, name, bind(self.lib, 'quactlize_ppu_moe_'+suffix, args))
        for name, symbol, args in (
            ('prepare_stage', 'qzm_test_prepare_stage_v1', [C.POINTER(Prepare), C.c_int]),
            ('prepare_unfused', 'qzm_test_prepare_unfused_v1', [C.POINTER(Prepare), C.c_int]),
            ('prepare_hybrid', 'qzm_test_prepare_hybrid_v1', [C.POINTER(Prepare), C.c_int]),
            ('finish_parallel', 'qzm_test_finish_parallel_v1', [C.POINTER(Finish), C.c_int]),
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
        if case.padding<0 or case.projection_dtype not in (-1,0) or case.phase not in ('all','prepare','transition','finish'):
            raise ValueError('invalid helper layout/storage/phase')
        padding = case.padding
        self.rows = case.tokens*case.topk
        self.gcols = case.hidden*(2 if case.merged and case.state != 2 else 1)
        route = Call(version=1, size=C.sizeof(Call), tokens=case.tokens, experts=case.experts,
                     topk=case.topk, channels=1, ids_stride=case.topk+padding, source_stride=1+padding, stream=rt.stream.value)
        self.p = Prepare(version=1, size=C.sizeof(Prepare), route=route,
                         config=Config(1, C.sizeof(Config), 2, 512), columns=case.columns,
                         dtype=case.dtype, input_row_stride=case.columns+padding)
        self.sizes = Sizes(); checked(lib.query(C.byref(self.p), C.byref(self.sizes)), 'helper query')
        try:
            for name, n in [('ids', case.tokens*(case.topk+padding)*4), ('src_rows', self.rows*4),
                            ('dst_rows', self.rows*4), ('offsets', (case.experts+1)*4), ('workspace', self.sizes.workspace_bytes)]:
                setattr(self.p.route, name, self.alloc(name, n))
            self.p.route.workspace_bytes = self.sizes.workspace_bytes
            self.input_rows = (case.tokens-1)*(1+padding)+1 if case.tokens else 0
            for name, n in [('input', self.input_rows*(case.columns+padding)*4),
                            ('output', self.rows*case.columns*(4 if case.dtype == 0 else 2)),
                            ('inverse_rows', self.rows*4), ('expert_rows', case.experts*4),
                            ('active_experts', case.experts*4), ('active_count', 4)]:
                setattr(self.p, name, self.alloc('prepare_'+name, n))
            self.t = Transition(version=1, size=C.sizeof(Transition), tokens=case.tokens, topk=case.topk,
                hidden=case.hidden, merged=case.merged, output_dtype=case.dtype, output_order=case.output_order,
                output_stride=case.hidden+padding, row_ids=self.p.route.dst_rows, inverse_rows=self.p.inverse_rows,
                output=self.alloc('activation', self.rows*(case.hidden+padding)*(4 if case.dtype==0 else 2)), stream=rt.stream.value)
            self.f = Finish(version=1, size=C.sizeof(Finish), tokens=case.tokens, topk=case.topk,
                columns=case.output, inverse_rows=self.p.inverse_rows, weights_stride=case.topk+padding,
                output_stride=case.output+padding, weights=self.alloc('weights', case.tokens*(case.topk+padding)*4),
                output=self.alloc('finish', case.tokens*(case.output+padding)*4), stream=rt.stream.value)
            for name, n in [('tmp_gate', self.rows*self.gcols*4), ('tmp_up', self.rows*case.hidden*4),
                            ('tmp_act', self.rows*case.hidden*4), ('tmp_down', self.rows*case.output*4),
                            ('tmp_weighted', self.rows*case.output*4)]: self.alloc(name, n)
            for name, nc, state, splits, order in [
                    ('gate', self.gcols, case.state, case.splits if case.state == 1 else 1, case.gate_order),
                    ('up', case.hidden, 1 if case.state == 1 else 0, case.splits if case.state == 1 else 1, case.up_order),
                    ('down', case.output, int(case.down_splits > 1), case.down_splits, case.down_order)]:
                dtype = 0 if state == 1 else case.dtype if case.projection_dtype == -1 else case.projection_dtype
                pitch = self.rows*(nc+padding)+(5 if padding else 0)
                v = Value(self.alloc(name, splits*pitch*(4 if dtype == 0 else 2)), dtype, order, state,
                          splits, nc+padding, pitch, 0 if state == 2 else case.dtype, 0)
                if name == 'down': self.f.down = v
                else: setattr(self.t, name, v)
            self.prepare_plan,self.finish_plan=Plan(),Plan()
            checked(lib.prepare_select(C.byref(self.p),0,C.byref(self.prepare_plan)),'public prepare selection')
            auto=Prepare.from_buffer_copy(self.p);auto.config.algorithm=auto.config.tile_rows=0
            self.combined_plan=PreparePlanV2()
            checked(lib.prepare_select_v2(C.byref(auto),C.byref(self.combined_plan)),'combined prepare selection')
            if self.combined_plan.helper.workspace_bytes>self.sizes.workspace_bytes:
                raise ValueError('combined prepare needs a larger route workspace')
            checked(lib.finish_select(C.byref(self.f),0,C.byref(self.finish_plan)),'public finish selection')
            self.split_plan=Plan()
            checked(lib.finish_select(C.byref(self.f),19,C.byref(self.split_plan)),'public split finish query')
            self.alloc('selected_scratch',self.split_plan.workspace_bytes)
        except BaseException:
            self.close(); raise

    def alloc(self, name, size):
        b = Buffer(self.rt, size); self.buffers[name] = b; return b.ptr

    def ptr(self, name): return self.buffers[name].ptr

    def update_inputs(self, seed):
        """Refresh caller inputs only; real GEMM producers own their outputs."""
        c = self.case; rng = np.random.default_rng(seed)
        for b in self.buffers.values(): b.poison()
        ids = ids_for(c.route_case, seed)
        src, dst, offsets = oracle(ids, c.route_case)
        inverse = np.empty(self.rows, dtype=np.int32); inverse[dst] = np.arange(self.rows, dtype=np.int32)
        self.maps = src, dst, offsets, inverse
        self.buffers['ids'].upload(ids.tobytes())
        a = rng.uniform(-3, 3, (self.input_rows, self.p.input_row_stride)).astype(np.float32)
        if a.size and c.dtype == 2: a[0, 0] = 243383.484
        self.buffers['prepare_input'].upload(a.tobytes())
        self.prepared_expected = encode(a[src, :c.columns], c.dtype)
        return rng

    def update_weights(self, rng):
        c = self.case
        w = rng.uniform(0, 1, (c.tokens, self.f.weights_stride)).astype(np.float32)
        self.buffers['weights'].upload(w.tobytes())
        return w

    def update(self, seed):
        # Synthetic helper-only fixtures retain their original RNG sequence.
        # Do not call this after binding a real final/partial producer value.
        c = self.case; rng = self.update_inputs(seed); dst = self.maps[1]
        values = {}
        for name, v, nc in [('gate', self.t.gate, self.gcols), ('up', self.t.up, c.hidden), ('down', self.f.down, c.output)]:
            # Stream planes so host RAM does not scale with Split-K. Keep the
            # same RNG order, independent slot order and explicit rounding.
            logical = np.zeros((self.rows, nc), dtype=np.float32)
            for i in range(v.partial_count):
                plane = rng.uniform(-2, 2, (self.rows, nc)).astype(np.float32)
                if plane.size and not i and c.dtype == 2 and name == 'up': plane[0, 0] = 100000
                if v.state == 1: np.add(logical, plane, out=logical)
                else: logical = rounded(plane, v.dtype)
                raw = np.zeros(v.partial_stride, dtype=np.float32)
                raw[:self.rows*v.row_stride].reshape(self.rows, v.row_stride)[:, :nc] = plane[dst] if v.order == 0 else plane
                data = encode(raw, v.dtype).tobytes()
                self.rt.write(self.ptr(name)+i*v.partial_stride*(4 if v.dtype==0 else 2), data)
                del plane, raw, data
            values[name] = logical if v.state == 2 else rounded(logical, v.round_type)
        if c.state == 2: act = values['gate']
        else:
            gate = values['gate'][:, :c.hidden]
            up = values['gate'][:, c.hidden:] if c.merged else values['up']
            act = activation(gate, up)
        self.activation_expected = rounded(act[dst] if c.output_order == 0 else act, c.dtype)
        w = self.update_weights(rng)
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

    def reset_prepare(self):
        for name in ('src_rows', 'dst_rows', 'offsets', 'workspace', 'prepare_output',
                     'prepare_inverse_rows', 'prepare_expert_rows', 'prepare_active_experts', 'prepare_active_count'):
            self.buffers[name].poison()

    def selected_prepare(self):
        return self.lib.prepare_execute(C.byref(self.p),C.byref(self.prepare_plan))

    def combined_prepare(self,plan=None):
        call=Prepare.from_buffer_copy(self.p);call.config.algorithm=call.config.tile_rows=0
        return self.lib.prepare_execute_v2(C.byref(call),C.byref(self.combined_plan if plan is None else plan))

    def selected_finish(self,plan=None):
        return self.lib.finish_execute(C.byref(self.f),C.byref(self.finish_plan if plan is None else plan),
            self.ptr('selected_scratch'),self.buffers['selected_scratch'].size)

    def numerical(self):
        c = self.case
        self.reset_prepare()
        checked(self.lib.prepare(C.byref(self.p)), 'fused prepare'); self.prepare_check()
        self.reset_prepare()
        for stage in range(1 if not c.tokens else 5): checked(self.lib.prepare_unfused(C.byref(self.p), stage), 'unfused prepare')
        self.prepare_check()
        self.reset_prepare()
        checked(self.lib.prepare_hybrid(C.byref(self.p), -1), 'hybrid prepare'); self.prepare_check()
        self.reset_prepare()
        checked(self.selected_prepare(),'public selected prepare'); self.prepare_check()
        self.reset_prepare()
        checked(self.combined_prepare(),'combined route/helper prepare'); self.prepare_check()
        bad=PreparePlanV2.from_buffer_copy(self.combined_plan);bad.helper.stages+=1
        if self.combined_prepare(bad)!=20:raise ValueError('combined prepare accepted a damaged plan')
        if c.phase=='all' and c.tokens<=8192:
            alternate=Prepare.from_buffer_copy(self.p);alternate.config.algorithm=1
            plan=Plan();checked(self.lib.prepare_select(C.byref(alternate),3,C.byref(plan)),'public WARP route plan')
            self.reset_prepare()
            checked(self.lib.prepare_execute(C.byref(alternate),C.byref(plan)),'public WARP route prepare')
            self.prepare_check()
        checked(self.lib.transition(C.byref(self.t)), 'fused transition'); self.rt.sync()
        act = self.buffers['activation'].read()
        actual = self.values('activation', c.dtype, self.rows, self.t.output_stride, c.hidden)
        # expf/libm differences may move the last BF16/FP16 bit. The device
        # decomposition below must still match the fused device result EXACTLY.
        np.testing.assert_allclose(actual, self.activation_expected, rtol=(3e-6,.0011,.009)[c.dtype], atol=2e-6)
        for call, _, _ in self.transition_control(): checked(call(), 'unfused transition')
        self.rt.sync()
        if self.buffers['activation'].read() != act: raise ValueError('fused vs unfused transition bits differ')
        checked(self.lib.finish(C.byref(self.f)), 'fused finish'); self.rt.sync()
        finish = self.buffers['finish'].read()
        np.testing.assert_array_equal(self.values('finish', 0, c.tokens, self.f.output_stride, c.output), self.finish_expected)
        for call, _, _ in self.finish_control(): checked(call(), 'unfused finish')
        self.rt.sync()
        if self.buffers['finish'].read() != finish: raise ValueError('fused vs unfused finish bits differ')
        for warps in (4, 8):
            self.buffers['finish'].poison()
            checked(self.lib.finish_parallel(C.byref(self.f), warps), f'parallel finish {warps}')
            self.rt.sync()
            if self.buffers['finish'].read() != finish: raise ValueError(f'parallel finish {warps} bits differ')
        self.buffers['finish'].poison();self.buffers['selected_scratch'].poison()
        checked(self.selected_finish(),'public selected finish');self.rt.sync()
        if self.buffers['finish'].read()!=finish:raise ValueError('public selected finish bits differ')
        if c.phase=='all':
            self.buffers['finish'].poison();self.buffers['selected_scratch'].poison()
            checked(self.selected_finish(self.split_plan),'public separate finish');self.rt.sync()
            if self.buffers['finish'].read()!=finish:raise ValueError('public separate finish bits differ')
            if c.tokens:
                self.buffers['finish'].poison();self.rt.sync()
                poison=self.buffers['finish'].read()
                for ptr,size,want in ((self.ptr('selected_scratch'),self.split_plan.workspace_bytes-1,37),
                                      (self.f.output,self.split_plan.workspace_bytes,20)):
                    rc=self.lib.finish_execute(C.byref(self.f),C.byref(self.split_plan),ptr,size)
                    if rc!=want:raise ValueError(f'public finish scratch rejection differs: {rc}/{want}')
                bad=Plan.from_buffer_copy(self.split_plan);bad.stages+=1
                if self.selected_finish(bad)!=20:raise ValueError('public finish accepted a damaged plan')
                self.rt.sync()
                if self.buffers['finish'].read()!=poison:raise ValueError('rejected public finish launched or wrote output')
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
        for arm, count, method in [('fused',3,self.lib.prepare_stage),('unfused',5,self.lib.prepare_unfused),
                                   ('hybrid',4,self.lib.prepare_hybrid)]:
            items=[]
            for stage in range(count):
                if stage==0: symbol,grid,threads='histogram_rank<',[self.sizes.tiles,1,1],128
                elif stage==1: symbol,grid,threads=('prefix_chunks(' if arm=='unfused' else 'prepare_prefix<'),[1,1,1],((c.experts+31)//32)*32
                elif arm!='fused' and stage==2: symbol,grid,threads='scatter_maps<',[(self.rows+255)//256,1,1],256
                elif arm=='unfused' and stage==3: symbol,grid,threads='prepare_prefix<',[1,1,1],((c.experts+31)//32)*32
                else: symbol,grid,threads='prepare_gather<',self.row_grid(self.rows,c.columns),256
                items.append((lambda s=stage,m=method:m(C.byref(self.p),s),symbol,grid,[threads,1,1]))
            grouped['prepare/'+arm]=items
        grouped['transition/fused']=[(lambda:self.lib.transition(C.byref(self.t)),'transition_kernel(',self.row_grid(self.rows,c.hidden),[256,1,1])]
        grouped['transition/unfused']=[(*x,[256,1,1]) for x in self.transition_control()]
        grouped['finish/fused']=[(lambda:self.lib.finish(C.byref(self.f)),'finish_kernel(',self.row_grid(c.tokens,c.output),[256,1,1])]
        grouped['finish/unfused']=[(*x,[256,1,1]) for x in self.finish_control()]
        for warps in (4, 8):
            grouped[f'finish/slots{warps}']=[(lambda w=warps:self.lib.finish_parallel(C.byref(self.f),w),
                f'finish_slots<{warps}>', [min(c.tokens*((c.output+31)//32),65535),1,1], [warps*32,1,1])]
        return {key:items for key,items in grouped.items() if c.phase=='all' or key.split('/')[0]==c.phase}

    def close(self):
        self.rt.sync()
        for b in self.buffers.values(): b.close()


def start_clock(rt, device, smi, mhz):
    # Resolve the masked runtime ordinal to the physical card. Never feed a
    # CUDA_VISIBLE_DEVICES logical ordinal to the system-wide management CLI.
    pci=C.create_string_buffer(32)
    query=bind(rt.lib,'hggcDeviceGetPCIBusId',[C.c_void_p,C.c_int,C.c_int])
    checked(query(pci,len(pci),device['ordinal']),'helper PCI identity')
    device['pci']=pci.value.decode();pci_key(device['pci'])
    lock=request_lock(smi,device['pci'],mhz)
    print('MOE_SCAFFOLD_CLOCK '+json.dumps(lock),flush=True)
    if lock['returncode']:raise RuntimeError('assigned-device clock request failed; see MOE_SCAFFOLD_CLOCK')
    return dict(policy=ADVISORY,request=lock,smi=str(smi),
                before=observe(smi,device['pci'],mhz,advisory=True))


def measure(sdk, receipt, cases, rounds=4, samples=2, repeats=32, *, smi, clock_mhz=1700, progress=None):
    points,schedule=[],[]
    with ExitStack() as root:
        rt=Runtime(sdk);root.callback(rt.close);lib=Library(receipt['candidate'])
        clock=start_clock(rt,lib.device,smi,clock_mhz)
        # Two independent per-stream scratch owners, changing IDs and graph
        # replay. No graph cloning or KernelNodeGetParams is used.
        with ExitStack() as gate:
            fixtures=[];graphs=[]
            for i in range(10):
                warps=(0,4,8,-1,-2)[i//2]
                r=Runtime(sdk);gate.callback(r.close)
                case=(Case(16,columns=2048,hidden=31,output=33,padding=0,state=1,splits=2,down_splits=4)
                      if warps==-2 else Case(17,experts=33,topk=6,state=1,splits=2,down_splits=4))
                b=Fixture(r,lib,case)
                gate.callback(b.close);b.update(i);b.numerical()
                if warps==-1:
                    checked(lib.prepare_select(C.byref(b.p),2,C.byref(b.prepare_plan)),'graph hybrid plan')
                    if i==7:
                        # Reuse the exact same address-free plans on independent
                        # storage/streams; only the call descriptors are rebound.
                        b.prepare_plan=fixtures[-1].prepare_plan;b.split_plan=fixtures[-1].split_plan
                if i==9:b.combined_plan=fixtures[-1].combined_plan
                def call(b=b,w=warps):
                    if w in (-1,-2):
                        for fn in (b.combined_prepare if w==-2 else b.selected_prepare,
                                   lambda:lib.transition(C.byref(b.t)),lambda:b.selected_finish(b.split_plan)):
                            rc=fn()
                            if rc:return rc
                        return 0
                    for fn in (lambda:lib.prepare_hybrid(C.byref(b.p),-1) if w else lib.prepare(C.byref(b.p)),
                               lambda:lib.transition(C.byref(b.t)),
                               lambda:lib.finish_parallel(C.byref(b.f),w) if w else lib.finish(C.byref(b.f))):
                        rc=fn()
                        if rc:return rc
                    return 0
                g=Graph(r,call,1);gate.callback(g.close);fixtures.append(b);graphs.append(g)
            for seed in (23,41):
                for i,b in enumerate(fixtures):b.update(seed+i)
                for g in graphs:g.run()
                for b in fixtures:
                    b.prepare_check()
                    np.testing.assert_allclose(b.values('activation',b.case.dtype,b.rows,b.t.output_stride,b.case.hidden),
                                               b.activation_expected,rtol=.009,atol=2e-6)
                    np.testing.assert_array_equal(b.values('finish',0,b.case.tokens,b.f.output_stride,b.case.output),b.finish_expected)
        for index,case in enumerate(cases):
            if progress:progress(index,len(cases),case)
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
                    selected={1:'fused',2:'hybrid',3:'unfused',16:'fused',17:'slots4',18:'slots8',19:'unfused'}
                    chosen=selected[b.prepare_plan.implementation] if phase=='prepare' else (
                        selected[b.finish_plan.implementation] if phase=='finish' else 'fused')
                    p=case_points.setdefault(phase,dict(key=case.key+'/'+phase,case=asdict(case),phase=phase,
                        selected_arm=chosen,components={}))
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
                if progress:progress(index+1,len(cases),None)
                print('MOE_SCAFFOLD_POINT '+case.key+' M='+str(case.tokens)+' phase='+case.phase+
                      f' completed={index+1}/{len(cases)} exact_device_controls=PASS',flush=True)
        clock['after']=observe(smi,lib.device['pci'],clock_mhz,advisory=True)
        return dict(points=points,schedule=schedule,device=lib.device,clock=clock,correctness='PASS',
                    cases=[asdict(c) for c in cases],production_changed=False,
                    public_selected_correctness='PASS',
                    scope='HELPERS_ONLY_SYNTHETIC_PROJECTION_OUTPUTS_NO_GEMM_OR_MODEL')
