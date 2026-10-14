"""Reuse the independent matched MoE fixture and real typed module entry."""
import ctypes as C
from pathlib import Path


def point(w):
    return dict(q=w.qtype, n=w.n, k=w.k, tokens=w.m, mode=2, experts=w.experts,
                topk=w.topk, channels=w.channels, compute=w.compute, router=w.router or 'real')


def config(recipe, w):
    p = dict(recipe.parameters)
    if recipe.kind in ('simt', 'q4'):
        fields = ('variant', 'columns', 'warps', 'values', 'split') if recipe.kind == 'simt' else (
            'reader', 'variant', 'columns', 'warps', 'values')
        return dict(kind=recipe.kind, recipe={k:p[k] for k in fields}, split=p.get('split',1))
    if recipe.kind != 'tc':
        raise ValueError('grouped full-dequant requires the separate current GPU routing-map component; not silently omitted')
    parent = {k:p[k] for k in ('symbol','qtype','tm','tn','tk','wm','wn','stages','ap','dn')}
    parent.update(route=recipe.route+'-grouped', persistent=p['persistent'])
    return dict(kind='tc', parent=parent, split=p['split'], algorithm=int(p['grid_mode'] >= 2),
                grid_mode=p['grid_mode'], grid_b=p['grid_b'])


class RuntimeView:
    """Use the SAME fixture stream; no second stream dependency is introduced."""
    def __init__(self, context):
        self.lib, self.stream, self.context = context.sdk.lib, context.r.stream, context
        signatures = {
            'StreamBeginCapture':[C.c_void_p,C.c_int],
            'StreamEndCapture':[C.c_void_p,C.POINTER(C.c_void_p)],
            'GraphInstantiateWithFlags':[C.POINTER(C.c_void_p),C.c_void_p,C.c_uint64],
            'GraphLaunch':[C.c_void_p,C.c_void_p], 'GraphDestroy':[C.c_void_p],
            'GraphExecDestroy':[C.c_void_p], 'EventCreate':[C.POINTER(C.c_void_p)],
            'EventRecord':[C.c_void_p,C.c_void_p], 'EventSynchronize':[C.c_void_p],
            'EventElapsedTime':[C.POINTER(C.c_float),C.c_void_p,C.c_void_p],
            'EventDestroy':[C.c_void_p],
        }
        for name,args in signatures.items():
            fn=getattr(self.lib,'hggc'+name);fn.argtypes,fn.restype=args,C.c_int
            setattr(self,name,fn)

    def sync(self):
        self.context.sdk.synchronize(self.stream)


class Provider:
    def __init__(self, package, context, w, recipe, module=None):
        from dev.smallm_closure.bench import Simt
        c = config(recipe,w)
        # The old matched carrier expands no SF planes. Q8 metadata is already
        # the exact resident FP16 scale plane; K-quant SF is not that contract.
        if recipe.route=='sf' and w.qtype!=8:
            raise ValueError('indexed K-quant SF needs an explicit typed expansion component; retained as pending')
        self.impl = (TensorCore(context,module,recipe,w) if recipe.kind=='tc' else
                     Simt(context,Path(package)/'quactlize-runtime/libquactlize_ppu_execution.so',c))
        self.receipt=self.impl.receipt

    def launch(self, copy=0):
        return self.impl.launch(copy)

    def close(self):
        self.impl.close()


class TensorCore:
    """Current v4 metadata ABI; historical fixture/arithmetic oracle reused.

    The older matched benchmark's TC carrier sends v3 requests. Do not cast
    that shorter struct into the BF16 v4 ABI or quietly measure a fallback.
    This carrier is experiment-only and performs no production selection.
    """
    def __init__(self, context, record, recipe, w):
        import numpy as np
        from dev.smallm_closure.bench import Buffer,Unsupported
        from tools.run_kpack_gemv_gate import Resources
        from quactlize.dispatch.native import IndexedIO
        from quactlize.runtime.native import Recipe as Launch,Resources as Query,checked
        from dev.strategy.resources import NativeProbe,module_spec,probe_spec,request_abi
        from dev.strategy.space import resolved_grid
        if w.operation!='indexed' or w.storage!='f32' or w.m>8:
            raise ValueError('indexed carrier requires F32 endpoints and tokens1..8')
        self.b=context;self.r=Resources(context.sdk)
        self.graph=self.proof_graph=None;self.handles=[]
        try:
            probe=NativeProbe(module_spec(w,recipe),record);self.library=probe.lib
            algorithm=int(dict(recipe.parameters)['grid_mode']>=2)
            spec=probe_spec(w,recipe,algorithm);resources=probe.query(spec)
            if resources['status']!='READY':raise Unsupported('TC_RESOURCE_OR_SHAPE_UNSUPPORTED')
            resolved=resolved_grid(w,recipe,probe.device['compute_units'],resources['occupancy'])
            self.recipe=Launch(1,C.sizeof(Launch),resolved['algorithm'],resolved['split'],resolved['grid'])
            self.compact_a=Buffer(self.r,context.rows*w.k*2)
            self.compact_out=Buffer(self.r,context.rows*w.n*2)
            self.offsets=Buffer(self.r,(w.experts+1)*4)
            self.row_ids=Buffer(self.r,context.rows*4)
            self.offsets.upload(np.zeros(w.experts+1,dtype='i4'))
            self.scratch=Buffer(self.r,resources['workspace_bytes'])
            request=request_abi(spec,probe.device);call=request.device_call.call
            call.a=self.compact_a.ptr;call.output=self.compact_out.ptr;call.offsets_device=self.offsets.ptr
            call.workspace=self.scratch.ptr;call.workspace_bytes=self.scratch.size;call.stream=context.r.stream.value
            query=probe._bind('grouped_query_v4',[C.POINTER(type(request)),C.POINTER(Launch),C.POINTER(Query)])
            final=Query()
            checked(query(C.byref(request),C.byref(self.recipe),C.byref(final)),'exact resolved grid query')
            if (final.occupancy!=resources['occupancy'] or final.workspace_bytes!=resources['workspace_bytes'] or
                    final.shared_bytes!=resources['shared_bytes']):
                raise ValueError('resolved grid changed compiled resources')
            prepare=probe._bind('grouped_prepare_v4',[C.POINTER(type(request)),C.POINTER(Launch),C.POINTER(C.c_void_p)])
            bind=probe._bind('bind_llama_indexed_v1',[C.c_void_p,C.POINTER(IndexedIO)])
            self.run_fn=probe._bind('run_v1',[C.c_void_p,C.c_void_p])
            self.destroy=probe._bind('destroy_v1',[C.c_void_p],None)
            io=IndexedIO(version=1,size=C.sizeof(IndexedIO),tokens=w.m,topk=w.topk,channels=w.channels,
                ids_stride=w.topk,a_row_stride=w.k,a_token_stride=w.channels*w.k,out_row_stride=w.n,
                ids=context.ids.ptr,a=context.a.ptr,output=context.out.ptr,row_ids=self.row_ids.ptr)
            for plane in context.ring.planes:
                call.low,call.high,call.metadata=plane['low'],plane['high'],plane['units']
                handle=C.c_void_p()
                checked(prepare(C.byref(request),C.byref(self.recipe),C.byref(handle)),'TC v4 prepare')
                self.handles.append(handle)
                checked(bind(handle,C.byref(io)),'TC indexed prepare/finish binding')
            self.receipt=dict(kind='tc',parent=record['parent'],build_key=record['key'],module_sha256=record['sha256'],
                split=self.recipe.split,algorithm=self.recipe.algorithm,grid=self.recipe.grid,
                shared_bytes=resources['shared_bytes'],workspace_bytes=resources['workspace_bytes'],
                compute_units=probe.device['compute_units'],device=probe.device['ordinal'],
                compute_type=w.compute,metadata_type=request.metadata_type,
                scope='F32_INDEXED_PREPARE_TC_REAL_REDUCER_INDEXED_FINISH')
        except BaseException:
            self.close();raise

    def launch(self,index=0):
        return self.run_fn(self.handles[index],self.b.r.stream)

    def close(self):
        self.b.sdk.synchronize(self.b.r.stream)
        for graph in (self.graph,self.proof_graph):
            if graph:graph.close()
        for handle in self.handles:self.destroy(handle)
        self.handles=[];self.graph=self.proof_graph=None
        self.r.close()
