"""One-helper controls, exact device parity and Asys-only ABBA. No GEMM sweep."""
import ctypes as C
from contextlib import ExitStack
from dataclasses import asdict, replace
import json

import numpy as np

from dev.moe_route.bench import Runtime, Graph, bind
from dev.moe_scaffold.bench import Fixture as BaseFixture, Library as BaseLibrary, start_clock
from dev.moe_scaffold.protocol import Case, Finish, Transition, weighted
from dev.strategy.refresh_clock import observe
from quactlize.runtime.native import checked

BASELINE='CURRENT_GENERIC_HELPER_SAME_DTYPE_AND_ORDER_NOT_OLD_FUSED_GEMV'


def case(m,phase,**kwargs):
    return replace(Case(m,columns=1,hidden=1,output=1,dtype=0,projection_dtype=0,
                        gate_order=1,up_order=1,output_order=1,down_order=1,
                        padding=0,phase=phase),**kwargs)


def workload_plan(smoke=False,phase='all'):
    ms=(1,8) if smoke else range(1,9)
    cases=[case(m,'finish',output=n) for n in (2048,3072) for m in ms]
    cases += [case(m,'transition',hidden=n,topk=topk) for n in (512,1024) for topk in (1,8) for m in ms]
    if not smoke:
        cases += [case(m,'finish',output=n) for m in (16,128) for n in (2048,3072)]
        cases += [case(7,'finish',output=2051,padding=3,dtype=d) for d in (0,1,2)]
        cases += [case(17,'transition',hidden=513,padding=3,topk=k) for k in (1,8)]
    if phase!='all':cases=[c for c in cases if c.phase==phase]
    if not cases:raise ValueError('decode-kernels supports finish/transition, not prepare')
    assert len({c.key for c in cases})==len(cases)
    return cases,dict(cohort='decode-kernels',cases=len(cases),baseline=BASELINE,
        phases=sorted({c.phase for c in cases}),
        priority='M1..8_F32_FINISH_N2048_N3072_TRANSITION_H512_H1024_TOP1_TOP8',
        timing='WARM_HELPER_ABBA_ASYS_ACTUAL_INTERVAL_MEANS',
        scope='HELPER_ONLY_SYNTHETIC_PROJECTION_OUTPUTS_NOT_GEMM_OR_MODEL',
        production_changed=False)


class Library(BaseLibrary):
    def __init__(self,path):
        super().__init__(path)
        self.decode_finish=bind(self.lib,'qzm_decode_finish',[C.POINTER(Finish),C.c_int])
        self.decode_transition=bind(self.lib,'qzm_decode_transition',
            [C.POINTER(Transition),C.c_int,C.c_void_p,C.c_int64,C.c_void_p])


class Fixture(BaseFixture):
    def __init__(self,rt,lib,c):
        super().__init__(rt,lib,c)
        self.alloc('retained',self.rows*4)

    def update_weights(self,rng):
        # Signed weights also exercise ordered cancellation, without changing
        # the production contract (router weights normally remain positive).
        c=self.case
        self.weights_host=rng.uniform(-1,1,(c.tokens,self.f.weights_stride)).astype(np.float32)
        self.buffers['weights'].upload(self.weights_host.tobytes())
        return self.weights_host

    def update(self,seed):
        super().update(seed)
        # Only grouped fallback controls need maps. CPU oracle supplies them;
        # no prepare/gather launch participates in this helper experiment.
        for name,values in zip(('src_rows','dst_rows','offsets','prepare_inverse_rows'),self.maps):
            self.buffers[name].upload(values.tobytes())

    def launch(self,phase,arm):
        if phase=='finish':return self.lib.decode_finish(C.byref(self.f),arm)
        snapshot=phase=='snapshot'
        return self.lib.decode_transition(C.byref(self.t),arm,self.f.weights if snapshot else None,
                                           self.f.weights_stride,self.ptr('retained') if snapshot else None)

    def output_name(self,phase):return 'finish' if phase=='finish' else 'activation'

    def reset_output(self,phase):
        self.buffers[self.output_name(phase)].poison();self.buffers['retained'].poison()

    def reference(self,phase):
        self.reset_output(phase);checked(self.launch(phase,0),'decode helper control');self.rt.sync()
        c=self.case
        if phase=='finish':
            np.testing.assert_array_equal(self.values('finish',0,c.tokens,self.f.output_stride,c.output),self.finish_expected)
        else:
            np.testing.assert_allclose(self.values('activation',c.dtype,self.rows,self.t.output_stride,c.hidden),
                self.activation_expected,rtol=(3e-6,.0011,.009)[c.dtype],atol=2e-6)
        self.check_snapshot(phase)
        return self.buffers[self.output_name(phase)].read()

    def check_snapshot(self,phase):
        if phase=='snapshot':
            expected=np.ascontiguousarray(self.weights_host[:,:self.case.topk]).tobytes()
            if self.buffers['retained'].read()!=expected:raise ValueError('routing-weight snapshot differs')

    def exact(self,phase,reference):
        self.rt.sync()
        if self.buffers[self.output_name(phase)].read()!=reference:
            raise ValueError(f'device helper bits/padding differ: {self.case.key} {phase}')
        self.check_snapshot(phase)
        for b in self.buffers.values():b.guard()

    def validate(self,phase):
        ref=self.reference(phase)
        for arm in range(1,4 if phase=='finish' else 3):
            self.reset_output(phase);checked(self.launch(phase,arm),'decode helper candidate');self.exact(phase,ref)

    def geometry(self,phase,arm):
        c=self.case
        if phase=='finish':
            fast=c.topk==8 and self.f.down.dtype==0 and c.down_order==1 and c.down_splits==1
            if arm==1:return 'finish_slots<4>',[min(c.tokens*((c.output+31)//32),65535),1,1],[128,1,1]
            if arm>=2 and fast:
                threads=128 if arm==2 else 64
                return f'finish_top8<{threads},',[(c.output+threads-1)//threads,min(c.tokens,65535),1],[threads,1,1]
            return 'finish_kernel(',self.row_grid(c.tokens,c.output),[256,1,1]
        fast=c.merged and c.dtype==0 and c.state==0 and c.gate_order==1 and c.output_order==1
        if arm and fast:
            threads=128 if arm==1 else 256
            return f'transition_direct<{threads},',[(c.hidden+threads-1)//threads,min(self.rows,65535),1],[threads,1,1]
        return 'transition_kernel',self.row_grid(self.rows,c.hidden),[256,1,1]


def boundary_gate(rt,lib):
    tests=[case(0,'finish'),case(1,'finish',output=1),case(7,'finish',output=31,padding=3),
           case(17,'finish',output=37,topk=6,padding=3,down_order=0,down_splits=4),
           case(9,'transition',hidden=31,padding=3,merged=0,gate_order=0,up_order=1,output_order=0),
           case(3,'transition',hidden=33,state=1,splits=4,dtype=2,gate_order=0,output_order=0),
           case(3,'transition',hidden=17,state=2,merged=0),
           case(65536,'finish',output=1),case(8193,'transition',hidden=1)]
    tests += [case(3,'finish',output=37,dtype=d,projection_dtype=-1,down_order=0) for d in (1,2)]
    for c in tests:
        b=Fixture(rt,lib,c)
        try:
            b.update(37)
            if c.tokens==1 and c.phase=='finish' and c.output==1:
                values=np.array([1e20,1,-1e20,3,0,0,0,0],np.float32).reshape(1,8,1)
                b.buffers['down'].upload(values.tobytes())
                b.weights_host[:]=1;b.buffers['weights'].upload(b.weights_host.tobytes())
                b.finish_expected=weighted(values,b.weights_host)
            b.validate(c.phase)
            if c.phase=='transition' and c.state!=2:b.validate('snapshot')
        finally:b.close()
    if lib.decode_finish(None,2)!=20 or lib.decode_transition(None,1,None,0,None)!=20:
        raise ValueError('null call accepted')
    print('DECODE_HELPER_BOUNDARIES PASS cases='+str(len(tests)),flush=True)


def stream_gate(sdk,lib):
    with ExitStack() as stack:
        entries=[]
        for phase in ('finish','snapshot'):
            for lane in range(2):
                rt=Runtime(sdk);stack.callback(rt.close)
                b=Fixture(rt,lib,case(3,'finish' if phase=='finish' else 'transition',output=37,hidden=31,padding=3))
                stack.callback(b.close);b.update(19+lane)
                arm=lane+(2 if phase=='finish' else 1)
                graph=Graph(rt,lambda b=b,p=phase,a=arm:b.launch(p,a),1);stack.callback(graph.close)
                entries.append((b,phase,graph))
        for seed in (71,83):
            refs=[]
            for b,phase,_ in entries:
                b.update(seed+len(refs));refs.append(b.reference(phase));b.reset_output(phase)
            for _,_,g in entries:g.run()
            for (b,phase,_),ref in zip(entries,refs):b.exact(phase,ref)
    print('DECODE_HELPER_REPLAY PASS streams=2_per_phase changing_inputs=1',flush=True)


def alias_gate(rt,lib):
    # Model allocator case: routing weights live inside the later finish
    # destination. The transition snapshot, not an extra copy kernel, retains
    # them. Changing-input graph replay must still equal the disjoint control.
    with ExitStack() as stack:
        b=Fixture(rt,lib,case(3,'transition',output=37,hidden=31,padding=3));stack.callback(b.close)
        f=Finish.from_buffer_copy(b.f);f.weights=b.ptr('retained');f.weights_stride=b.case.topk
        for arm in (1,2):
            def chain(a=arm):
                rc=lib.decode_transition(C.byref(b.t),a,b.f.output,b.f.weights_stride,b.ptr('retained'))
                return rc if rc else lib.decode_finish(C.byref(f),2)
            g=Graph(rt,chain,1)
            try:
                for seed in (47,53):
                    b.update(seed);ref=b.reference('finish');b.reset_output('finish')
                    # 33 F32 values fit in row0's 37 live columns, so this does
                    # not change the output's padding sentinel.
                    rt.write(b.f.output,b.weights_host.tobytes())
                    g.run();b.exact('finish',ref);b.check_snapshot('snapshot')
            finally:g.close()
    print('DECODE_HELPER_ALIAS PASS weights_inside_finish_output=1 extra_copy_kernel=0',flush=True)


def measure(sdk,receipt,cases,rounds=4,samples=2,repeats=32,*,smi,clock_mhz=1700,progress=None):
    points,schedule=[],[]
    with ExitStack() as root:
        rt=Runtime(sdk);root.callback(rt.close);lib=Library(receipt['candidate'])
        clock=start_clock(rt,lib.device,smi,clock_mhz)
        boundary_gate(rt,lib);stream_gate(sdk,lib);alias_gate(rt,lib)
        for index,c in enumerate(cases):
            if progress:progress(index,len(cases),c)
            with ExitStack() as stack:
                b=Fixture(rt,lib,c);stack.callback(b.close)
                phases=('finish',) if c.phase=='finish' else ('transition','snapshot')
                for seed in (7,19):
                    b.update(seed)
                    for phase in phases:b.validate(phase)
                graphs={};entries={}
                for phase in phases:
                    arms=('control','slots4','typed128','typed64') if phase=='finish' else ('control','typed128','typed256')
                    point=dict(key=c.key+'/'+phase,case=asdict(c),phase=phase,components={})
                    for arm,label in enumerate(arms):
                        key=point['key']+'/'+label;point['components'][label]=[key]
                        launch=lambda p=phase,a=arm:b.launch(p,a)
                        g=Graph(rt,launch,repeats);stack.callback(g.close)
                        for _ in range(3):g.run()
                        rt.sync();graphs[key]=g
                        symbol,grid,block=b.geometry(phase,arm)
                        entries[key]=dict(component=key,kernel=symbol,grid=grid,block=block,repeats=repeats)
                    points.append(point)
                rt.sync();checked(rt.ProfilerStart(),'decode helpers profile begin')
                try:
                    keys=list(graphs)
                    # Forward/reverse/reverse/forward gives ABBA for each pair.
                    for r in range(rounds):
                        for key in (keys if r%4 in (0,3) else keys[::-1]):
                            for s in range(samples):
                                graphs[key].run();rt.sync();schedule.append(dict(entries[key],round=r,sample=s))
                finally:
                    rt.sync();checked(rt.ProfilerStop(),'decode helpers profile end')
                for phase in phases:b.validate(phase)
                if progress:progress(index+1,len(cases),None)
                print('DECODE_HELPER_POINT '+json.dumps(dict(key=c.key,phase=c.phase,m=c.tokens,
                      completed=index+1,total=len(cases),correctness='PASS')),flush=True)
        clock['after']=observe(smi,lib.device['pci'],clock_mhz,advisory=True)
        return dict(points=points,schedule=schedule,device=lib.device,clock=clock,correctness='PASS',
                    cases=[asdict(c) for c in cases],baseline=BASELINE,production_changed=False,
                    scope='HELPER_ONLY_NO_GEMM_OR_MODEL_TPOT',protocol=dict(rounds=rounds,samples=samples,repeats=repeats))
