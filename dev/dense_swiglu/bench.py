"""One dense SwiGLU candidate, not a GEMM/helper sweep or model prediction."""
import ctypes as C
from contextlib import ExitStack
from dataclasses import asdict
import json

from dev.moe_decode_helpers.bench import case as helper_case, Fixture as BaseFixture
from dev.moe_scaffold.bench import Library as BaseLibrary, start_clock
from dev.moe_scaffold.protocol import Transition
from dev.moe_route.bench import Runtime, Graph, bind
from dev.strategy.refresh_clock import observe
from quactlize.runtime.native import checked

BASELINE='SHIPPING_7112F58D_GENERIC_F32_TRANSITION_SAME_MATH'


def case(m,h,**kw):
    return helper_case(m,'transition',hidden=h,topk=1,experts=1,**kw)


def workload_plan(smoke=False,phase='all'):
    if phase not in ('all','transition'):raise ValueError('dense-swiglu measures only transition')
    cases=[case(m,h) for h in ((25600,) if smoke else (17408,25600))
           for m in ((1,2048) if smoke else (1,8,512,2048,5120))]
    if not smoke:cases += [case(m,h) for m in (1,8) for h in (512,1024)]
    return cases,dict(cohort='dense-swiglu',cases=len(cases),baseline=BASELINE,phases=['transition'],
        scope='ONE_DENSE_F32_SWIGLU_HELPER_NO_GEMM_NO_CAST_NO_REDUCER',production_changed=False)


class Library(BaseLibrary):
    def __init__(self,path):
        super().__init__(path)
        self.dense=bind(self.lib,'qdf_swiglu',[C.POINTER(Transition),C.c_int])


class Fixture(BaseFixture):
    def launch(self,phase,arm):return self.lib.dense(C.byref(self.t),arm)

    def validate(self,phase='transition'):
        ref=self.reference(phase)
        self.reset_output(phase);checked(self.launch(phase,1),'dense SwiGLU');self.exact(phase,ref)

    def geometry(self,arm):
        c=self.case
        if not arm:
            if 1<=c.tokens<=8 and c.hidden in (512,1024):
                return 'transition_direct<256, false>',[(c.hidden+255)//256,c.tokens,1],[256,1,1]
            return 'transition_kernel',self.row_grid(c.tokens,c.hidden),[256,1,1]
        u=self.t.gate.data+c.hidden*4 if c.merged else self.t.up.data
        stride=self.t.gate.row_stride if c.merged else self.t.up.row_stride
        vector=not (c.hidden%4 or self.t.gate.row_stride%4 or stride%4 or self.t.output_stride%4 or
                    (self.t.gate.data|u|self.t.output)%16)
        width,threads=(512,128) if vector else (256,256)
        return ('quactlize::ffn::dense_swiglu_vec4' if vector else 'quactlize::ffn::dense_swiglu_scalar'),[
            (c.hidden+width-1)//width,min(c.tokens,65535),1],[threads,1,1]


def boundary_gate(rt,lib):
    cases=[case(0,4),case(1,1),case(3,33,padding=3),case(17,516,padding=4),
           case(3,32,merged=0),case(7,31,padding=3,merged=0),case(65536,4)]
    for c in cases:
        with ExitStack() as stack:
            b=Fixture(rt,lib,c);stack.callback(b.close);b.update(37);b.validate()
    print('DENSE_SWIGLU_BOUNDARY PASS cases='+str(len(cases)),flush=True)


def stream_gate(sdk,lib):
    with ExitStack() as stack:
        entries=[]
        for lane in range(2):
            rt=Runtime(sdk);stack.callback(rt.close)
            b=Fixture(rt,lib,case(3,516,padding=4,merged=lane));stack.callback(b.close);b.update(19+lane)
            graph=Graph(rt,lambda b=b:b.launch('transition',1),1);stack.callback(graph.close)
            entries.append((b,graph))
        for seed in (71,83):
            refs=[]
            for b,_ in entries:
                b.update(seed+len(refs));refs.append(b.reference('transition'));b.reset_output('transition')
            for _,g in entries:g.run()
            for (b,_),ref in zip(entries,refs):b.exact('transition',ref)


def measure(sdk,receipt,cases,rounds=4,samples=2,repeats=32,*,smi,clock_mhz=1700,progress=None):
    points,schedule=[],[]
    with ExitStack() as root:
        rt=Runtime(sdk);root.callback(rt.close);lib=Library(receipt['candidate'])
        clock=start_clock(rt,lib.device,smi,clock_mhz)
        boundary_gate(rt,lib);stream_gate(sdk,lib)
        for index,c in enumerate(cases):
            if progress:progress(index,len(cases),c)
            with ExitStack() as stack:
                b=Fixture(rt,lib,c);stack.callback(b.close)
                for seed in (7,19):b.update(seed);b.validate()
                graphs=[];entries=[];point=dict(key=c.key,case=asdict(c),phase='transition',components={})
                for arm,label in enumerate(('control','dense-vec4')):
                    key=c.key+'/'+label;point['components'][label]=[key]
                    graph=Graph(rt,lambda arm=arm:b.launch('transition',arm),repeats);stack.callback(graph.close)
                    for _ in range(3):graph.run()
                    rt.sync();graphs.append(graph)
                    symbol,grid,block=b.geometry(arm)
                    entries.append(dict(component=key,kernel=symbol,grid=grid,block=block,repeats=repeats))
                rt.sync();checked(rt.ProfilerStart(),'dense SwiGLU profile begin')
                try:
                    for r in range(rounds):
                        for arm in ((0,1) if r%4 in (0,3) else (1,0)):
                            for s in range(samples):
                                graphs[arm].run();rt.sync();schedule.append(dict(entries[arm],round=r,sample=s))
                finally:
                    rt.sync();checked(rt.ProfilerStop(),'dense SwiGLU profile end')
                b.validate();points.append(point)
                if progress:progress(index+1,len(cases),None)
                print('DENSE_SWIGLU_POINT '+json.dumps(dict(m=c.tokens,h=c.hidden,correctness='PASS')),
                      flush=True)
        clock['after']=observe(smi,lib.device['pci'],clock_mhz,advisory=True)
        return dict(points=points,schedule=schedule,device=lib.device,clock=clock,correctness='PASS',
            cases=[asdict(c) for c in cases],baseline=BASELINE,production_changed=False,
            scope='DENSE_SWIGLU_ONLY_NOT_MODEL',protocol=dict(rounds=rounds,samples=samples,repeats=repeats))
