"""Production dense helper/library entries, not copied diagnostic kernels."""
import ctypes as C
from dataclasses import asdict

import numpy as np

from dev.bf16_compute.fixture import raw_weight, bf16_bits
from dev.prefill_point.bench import call, ptr, trace_contract
from dev.strategy.journal import atomic_json
from quactlize.dequant.native import Call as Weight
from quactlize.execution.native import Arrangement
from quactlize.prefill.native import Call
from quactlize.runtime.native import checked


class Options(C.Structure):
    _fields_=[('version',C.c_uint32),('size',C.c_uint32),('sdk',C.c_char_p)]+[
        (x,C.c_int32)for x in ('storage','vector_io','dequant_variant','math_mode','algorithm')]


def bind(lib,name,args,restype=C.c_int):
    fn=getattr(lib,name);fn.argtypes=args;fn.restype=restype;return fn


class Guard:
    def __init__(self,r,size):
        self.r,self.size=r,int(size);self.base=r.alloc(self.size+512);self.ptr=self.base+256
        if self.ptr%256:raise ValueError('workspace alignment differs')
        r.fill(self.base,0xa5,self.size+512);r.sdk.synchronize(r.stream)
    def guard(self):
        for ptr in (self.base,self.ptr+self.size):
            if self.r.sdk.download(ptr,256)!=b'\xa5'*256:raise ValueError('redzone overwritten')


class DenseLibrary:
    def __init__(self, path):
        self.lib=C.CDLL(str(path),mode=C.RTLD_LOCAL)
        self.policy=bind(self.lib,'quactlize_kpack_dense_helper_policy_v1',
            [C.POINTER(Call),C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_int)])
        self.io=bind(self.lib,'quactlize_dense_io_v1',
            [C.c_int]*3+[C.c_void_p]*2+[C.c_int]*2+[C.c_int64,C.c_void_p,C.c_void_p])
        self.query=bind(self.lib,'quactlize_kpack_dense_full_workspace_v1',
            [C.POINTER(Call),C.POINTER(Arrangement),C.POINTER(C.c_uint64)])
        self.prepare=bind(self.lib,'quactlize_kpack_dense_full_prepare_v1',
            [C.POINTER(Call),C.POINTER(Arrangement),C.POINTER(Options),C.POINTER(C.c_void_p)])
        self.run=bind(self.lib,'quactlize_kpack_dense_full_run_v1',[C.c_void_p,C.POINTER(Call),C.c_int])
        self.destroy=bind(self.lib,'quactlize_kpack_dense_full_destroy_v1',[C.c_void_p],None)
        self.error=bind(self.lib,'quactlize_kpack_dense_full_error_v1',[],C.c_char_p)

    def shape(self,w):
        c=Call();c.version=1;c.size=C.sizeof(Call);c.m=c.a_rows=w.m
        c.a_stride=w.k;c.output_stride=w.n;c.device=0
        # Legacy config remains in its own namespace; typed DQ is in Options.
        c.weight=Weight(1,C.sizeof(Weight),w.qtype,w.n,w.k,1,1,5)
        return c

    def choice(self,w,storage,allow_miss=False):
        c=self.shape(w);dq,io=C.c_int(-1),C.c_int(-1)
        rc=self.policy(C.byref(c),storage,C.byref(dq),C.byref(io))
        if allow_miss and rc==24: return None  # QKG_SHAPE only; runtime errors still fail.
        checked(rc,
                'production dense helper policy (requires measured ZW810 shape)')
        return dict(dequant_variant=dq.value,vector_io=io.value,storage=storage)

    def io_call(self,b,src,dst,*,input,storage,vector):
        w=b.w;width=w.k if input else w.n
        return call(self.io,5,[src,dst,None,b.r.stream],
                    [storage,int(input),vector,w.m,width],width)


class TypedFull:
    def __init__(self,b,library,sdk_path,storage,options=None):
        self.b,self.library,self.storage=b,library,storage;self.handle=C.c_void_p()
        self.workspaces=[];w=b.w;self.shape=library.shape(w)
        choice=library.choice(w,storage) if options is None else dict(options,storage=storage)
        self.options=Options(1,C.sizeof(Options),str(sdk_path).encode(),
                            **choice,math_mode=16,algorithm=-1)
        size=C.c_uint64()
        checked(library.query(C.byref(self.shape),C.byref(b.arr),C.byref(size)),'typed full workspace')
        rc=library.prepare(C.byref(self.shape),C.byref(b.arr),C.byref(self.options),C.byref(self.handle))
        if rc:raise ValueError('typed full prepare: '+library.error().decode(errors='replace'))
        self.calls=(Call*b.ring.copies)()
        try:
            for i,plane in enumerate(b.ring.planes):
                c=self.calls[i];C.memmove(C.byref(c),C.byref(self.shape),C.sizeof(c))
                for name in ('low','high','units'):setattr(c.weight,name,plane[name])
                for name,field in (('low','low_bytes'),('high','high_bytes'),('units','unit_bytes')):
                    setattr(c.weight,field,b.ring.w.planes[name].nbytes)
                ws=Guard(b.r,size.value);b.buffers.append(ws);self.workspaces.append(ws)
                c.workspace=ws.ptr;c.workspace_bytes=size.value;c.a=b.a.ptr;c.output=b.out.ptr
                c.weight.stream=b.r.stream.value
        except BaseException:
            self.close();raise
        self.receipt=dict(kind='typed-full',helper_policy=choice,
            arithmetic=dict(a='BF16' if storage else 'F16',b='BF16' if storage else 'F16',
                            c='BF16' if storage else 'F16',accumulator='FP32'),
            compute_type=68,math_mode=16,algorithm=-1,workspace_bytes=size.value,
            cublas_workspace_bytes=4<<20,entry='quactlize_kpack_dense_full_run_v1',
            scope='SAME_PREPARE_AND_RUN_AS_PRODUCTION_FULL; ROUTE_WINNER_NOT_ADMITTED')

    def descriptors(self):
        return [call(self.library.run,4,[self.handle,ptr(C.byref(c))],[stage])
                for c in self.calls for stage in (4,0,1,2,3)]

    def launch(self,copy,stage=-1):
        rc=self.library.run(self.handle,C.byref(self.calls[copy]),stage)
        if rc:raise ValueError('typed full run: '+self.library.error().decode(errors='replace'))

    def check_weights(self,copy):
        b,w=self.b,self.b.w;b.rt.sync();count=0
        for first in range(0,w.n,256):
            n=min(256,w.n-first)
            _,gold=raw_weight(w.qtype,n,w.k,710+w.qtype*1009+first)
            want=bf16_bits(gold) if self.storage else gold.astype('<f2').view('<u2')
            got=np.frombuffer(b.sdk.download(self.workspaces[copy].ptr+first*w.k*2,n*w.k*2),dtype='<u2').reshape(n,w.k)
            if np.any((got!=want)&(((got|want)&0x7fff)!=0)):
                raise ValueError('typed full independent raw GGUF weight bits differ')
            count+=got.size
        return dict(bits=count,bad=0,storage=self.storage)

    def close(self):
        self.b.rt.sync()
        if self.handle.value:self.library.destroy(self.handle);self.handle=C.c_void_p()


def measure_full(args,b,timer,arm,recipe,library):
    from dev.prefill_point.bench import Call as Descriptor, summarize
    from tools.kpack_bf16_providers import loaded_images
    storage=int(arm=='full-bf16-typed');core=None
    try:
        core=TypedFull(b,library,args.sdk,storage,getattr(args,'helper_options',None))
        calls=(Descriptor*(5*b.ring.copies))(*core.descriptors())
        names=['error_reset','input','expansion','main','output'];proof=[];bits=[]
        for repeat in (0,1):
            b.update(repeat)
            for i in (0,b.ring.copies-1):
                timer.run(calls,5,i);proof.append(b.check())
                bits.append(core.check_weights(i))
        # Omitted main must fail rather than reuse the preceding output.
        # Complete workspaces are poisoned, then reset and finish only.
        b.update(1);b.r.fill(core.workspaces[0].ptr,0,core.workspaces[0].size);b.rt.sync()
        core.launch(0,3)
        try:b.check()
        except ValueError as e:
            if not str(e).startswith('prefill oracle:'):raise
        else:raise ValueError('typed full omitted producer not detected')
        b.update(0,zero=True);timer.run(calls,5,0);b.check(zero=True)
        b.update(0)
        for i in range(b.ring.copies):timer.run(calls,5,i)
        proof.append(b.check())
        if args.trace_only:
            timer.profile(calls,5,b.ring.copies,b.r.stream);rows=[]
        else:rows=timer.measure(calls,5,b.ring.copies,b.r.stream,args.rounds,args.samples)
        proof.append(b.check())
        for buf in b.ring.buffers+b.buffers:buf.guard()
        value=dict(status='PASS',arm=arm,workload=asdict(b.w),recipe=asdict(recipe),correctness=proof,
            helper_mode='experimental' if getattr(args,'helper_options',None) is not None else 'production',
            helper_policy=core.receipt['helper_policy'],
            weight_bits=bits,omitted_producer='DETECTED',zero_a='PASS',receipt=core.receipt,
            fixture=b.ring.receipt,stages=names,point_timings=rows,loaded_images=loaded_images(),
            trace=trace_contract(b.ring.copies),summary=None if args.trace_only else summarize(rows,names),
            timing='ASYS_GPU_ACTIVITIES_COMPLETE_CALLS_NO_INNER_EVENTS',
            sequence='RESET_INPUT_DEQUANT_GEMM_OUTPUT_EVERY_CALL',
            allocation='ROTATING_PACKED_AND_EXPANDED_B_WITH_PER_COPY_SCRATCH',
            production_changed=False,model_admission=False)
        atomic_json(args.output/'results'/(args.case_id+'.json'),value)
        return value
    finally:
        if core:core.close()
