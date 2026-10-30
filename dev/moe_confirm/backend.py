"""Same projection contracts with public Quactlize preparation, no llama."""
import ctypes as C
import numpy as np

from dev.moe_route.bench import bind
from dev.moe_scaffold.bench import Library
from dev.moe_scaffold.protocol import Call,Config,Prepare,PreparePlanV2
from dev.native_components.kernel_nodes import KernelNodes
from dev.strategy.moe_prefill_native import check_maps
from dev.strategy.prefill_fixture import Context as BaseContext
from dev.strategy.prefill_native import TensorCore as BaseTC
from dev.strategy.prefill_full import Full as BaseFull
from dev.strategy.vector import Vector as BaseVector
from quactlize.runtime.native import checked

ROW_MAPS='PUBLIC_QUACTLIZE_PREPARE_OR_DIRECT_INDEXED_SIMT'


def timing_fixture(core):
    from dev.strategy.fixture_identity import fingerprint
    b=core.b;b.rt.sync()
    return fingerprint(np.frombuffer(b.sdk.download(b.a.ptr,b.a.size),dtype='<f4'),
                       np.frombuffer(b.sdk.download(b.ids.ptr,b.ids.size),dtype='<i4'))


def prepare_receipt(plan):
    return dict(version=plan.version,route=dict(algorithm=plan.config.algorithm,tile_rows=plan.config.tile_rows),
        helper=dict(implementation=plan.helper.implementation,stages=plan.helper.stages,
                    workspace_bytes=plan.helper.workspace_bytes))


class Context(BaseContext):
    def __init__(self,sdk,ring,w,helpers,receipt):
        self.ids=None;self.prepare_call=None
        self.qz=Library(receipt['scaffold']['candidate'])
        self.bridge=C.CDLL(receipt['bridge'],mode=C.RTLD_LOCAL)
        self.route_run=bind(self.qz.lib,'quactlize_ppu_moe_route_run_v1',[C.POINTER(Call),C.POINTER(Config)])
        self.provider_rows=bind(self.bridge,'qzm_test_provider_rows',[C.c_void_p]*3+[C.c_int,C.c_void_p])
        super().__init__(sdk,ring,w,helpers)
        self.rt.kernel_nodes=KernelNodes(self.bridge)
        self.ids=self.allocate(self.rows*4)
        self.inverse=self.allocate(self.rows*4)
        self.expert_rows=self.allocate(w.experts*4)
        self.active_ids=self.allocate(w.experts*4)
        self.active_count=self.allocate(4)
        route=Call(version=1,size=C.sizeof(Call),tokens=w.m,experts=w.experts,topk=w.topk,
            channels=w.channels,ids_stride=w.topk,source_stride=w.channels,ids=self.ids.ptr,
            src_rows=self.source.ptr,dst_rows=self.destination.ptr,offsets=self.offsets.ptr,stream=self.r.stream.value)
        self.prepare_call=Prepare(version=1,size=C.sizeof(Prepare),route=route,config=Config(1,C.sizeof(Config),0,0),
            columns=w.k,dtype=2,input_row_stride=w.k,input=self.a.ptr,inverse_rows=self.inverse.ptr,
            expert_rows=self.expert_rows.ptr,active_experts=self.active_ids.ptr,active_count=self.active_count.ptr)
        self.prepare_plan=PreparePlanV2()
        checked(self.qz.prepare_select_v2(C.byref(self.prepare_call),C.byref(self.prepare_plan)),'combined prepare plan')
        self.route_scratch=self.allocate(self.prepare_plan.helper.workspace_bytes)
        self.prepare_call.route.workspace=self.route_scratch.ptr
        self.prepare_call.route.workspace_bytes=self.route_scratch.size
        self.update(0)

    def launch_route(self):
        return self.route_run(C.byref(self.prepare_call.route),C.byref(self.prepare_plan.config))

    def prepare(self,ptr,*,rows=None,active_ids=None,active_count=None):
        call=Prepare.from_buffer_copy(self.prepare_call);call.output=ptr
        if rows is not None:call.expert_rows=rows
        if active_ids is not None:call.active_experts=active_ids
        if active_count is not None:call.active_count=active_count
        return self.qz.prepare_execute_v2(C.byref(call),C.byref(self.prepare_plan))

    def validate_routes(self):
        self.rt.sync()
        arrays=[np.frombuffer(self.sdk.download(b.ptr,b.size),dtype='<i4') for b in
                (self.source,self.destination,self.offsets)]
        check_maps(self.w,self.route,*arrays)
        for b in (self.ids,self.source,self.destination,self.offsets,self.route_scratch):b.guard()

    def update(self,repeat,*,zero=False,amplitude=1):
        super().update(repeat,zero=zero,amplitude=amplitude)
        if self.prepare_call is not None:
            self.ids.upload(self.route['owners'])
            for b in (self.source,self.destination,self.offsets):b.poison()
            checked(self.launch_route(),'Quactlize GPU row maps')
            self.validate_routes()

    def check(self,*,zero=False):
        self.validate_routes()
        return super().check(zero=zero)


class TensorCore(BaseTC):
    timing_fixture=timing_fixture
    def launch(self,copy=0):
        checked(self.b.prepare(self.a.ptr),'combined Quactlize prepare')
        checked(self.run_fn(self.handles[copy],self.b.r.stream),'confirmed TC call')
        return self.b.finish(self.out.ptr,True)


class Full(BaseFull):
    timing_fixture=timing_fixture

    def __init__(self,b,*args,**kwargs):
        from quactlize.prefill.native import Call as FullCall
        from quactlize.dequant.native import Call as Weight
        from quactlize.execution.native import Arrangement
        w=b.w
        shape=FullCall(version=1,size=C.sizeof(FullCall),m=b.rows,device=0,a_rows=w.m*w.channels,
            a_stride=w.k,output_stride=w.n,weight=Weight(1,C.sizeof(Weight),w.qtype,w.n,w.k,w.experts,1,4))
        query=bind(b.bridge,'qzm_test_full_workspace',[C.POINTER(FullCall),C.POINTER(Arrangement),C.POINTER(C.c_uint64)])
        size=C.c_uint64();checked(query(C.byref(shape),C.byref(b.arr),C.byref(size)),'shipping Full workspace')
        super().__init__(b,*args,**kwargs)
        self.receipt['shipping_workspace_bytes']=size.value
        self.receipt['shipping_routed_rows']=b.rows

    def update(self):
        # CPU sorted expert IDs are an oracle only, never provider input.
        checked(self.b.launch_route(),'full setup row maps')
        checked(self.active(),'full setup active experts')
        self.b.r.fill(self.indices.data_ptr(),0xa5,self.b.rows*4)
        self.ids_ready=False

    def launch(self,copy=0):
        b=self.b
        checked(b.prepare(self.a.data_ptr(),rows=self.rows.data_ptr(),active_ids=self.ids.ptr,
                          active_count=self.count.ptr),'full combined prepare')
        checked(b.provider_rows(b.ids.ptr,b.destination.ptr,self.indices.data_ptr(),b.rows,b.r.stream),
                'provider GPU expert IDs')
        self.ids_ready=True
        self.matmul(copy)
        return b.finish(self.out.data_ptr(),True)

    def guards(self):
        super().guards()
        if self.ids_ready:
            got=np.frombuffer(self.b.sdk.download(self.indices.data_ptr(),self.b.rows*4),dtype='<i4')
            np.testing.assert_array_equal(got,self.b.route['sorted_owners'])


class Vector(BaseVector):
    def setup(self,config=0,*,validate=True):
        pass  # Direct indexed SIMT consumes current raw IDs/F32 A, no sorting.
