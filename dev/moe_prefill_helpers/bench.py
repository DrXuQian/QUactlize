"""Bounded BF16 prefill delivery experiments, each helper timed independently."""
import ctypes as C
from contextlib import ExitStack
from dataclasses import asdict, replace
import json

import numpy as np

from dev.moe_route.bench import Runtime, Graph, bind
from dev.moe_scaffold.bench import Library as BaseLibrary, start_clock
from dev.moe_scaffold.protocol import Case, Prepare, Finish, Transition, encode, rounded, weighted
from dev.moe_decode_helpers.bench import Fixture as DecodeFixture
from dev.strategy.refresh_clock import observe
from quactlize.runtime.native import checked

BASELINE='SHIPPING_D8F9A260_BF16_GENERIC_BODIES_SAME_ROUNDING_NOT_MODEL_TIMING'


def case(m, phase, **kwargs):
    return replace(Case(m, columns=1, hidden=1, output=1, dtype=2, padding=0, phase=phase), **kwargs)


def workload_plan(smoke=False, phase='all'):
    cases=[]
    for n,h in ((2048,512), (3072,512)):
        for m,profile in ((2048,'random'), (2048,'cluster'), (512,'random'), (5120,'random')):
            if smoke and (n!=2048 or m!=2048 or profile!='random'):continue
            cases += [case(m,'prepare',columns=n,profile=profile),
                      case(m,'transition',hidden=h,profile=profile),
                      case(m,'finish',output=n,profile=profile)]
    # Both routed models have hidden512: do not time identical transitions twice.
    cases=list({c.key:c for c in cases}.values())
    if phase!='all':cases=[c for c in cases if c.phase==phase]
    assert cases and len({c.key for c in cases})==len(cases)
    return cases,dict(cohort='prefill-kernels',cases=len(cases),baseline=BASELINE,
        phases=sorted({c.phase for c in cases}),
        priority='BF16_GROUPED_PROJECTED_M2048_TOP8_KN2048_H512',
        extension='KN3072_H512_M512_M5120_CLUSTER_ROUTES',
        timing='WARM_HELPER_ABBA_ASYS_ACTUAL_INTERVAL_MEANS',
        scope='HELPER_ONLY_SYNTHETIC_PRODUCERS_NOT_GEMM_OR_MODEL', production_changed=False)


class Library(BaseLibrary):
    def __init__(self,path):
        super().__init__(path)
        self.fast=bind(self.lib,'qzm_prefill_fast',[C.c_int,C.c_void_p])
        self.prefill_prepare=bind(self.lib,'qzm_prefill_prepare',[C.POINTER(Prepare),C.c_int,C.c_int])
        self.prefill_finish=bind(self.lib,'qzm_prefill_finish',[C.POINTER(Finish),C.c_int])
        self.prefill_transition=bind(self.lib,'qzm_prefill_transition',
            [C.POINTER(Transition),C.c_int,C.c_void_p,C.c_int64,C.c_void_p])


class Fixture(DecodeFixture):
    def update(self,seed):
        super().update(seed)
        if self.case.phase=='prepare' and self.rows:
            # Completed route prefix/ranks are inputs to every gather arm.
            # They are not rerun inside an isolated gather timer.
            for stage in (0,1):checked(self.lib.prefill_prepare(C.byref(self.p),0,stage),'prepare metadata')
            self.rt.sync()

    def launch(self,phase,arm):
        if phase=='prepare':return self.lib.prefill_prepare(C.byref(self.p),arm,2 if self.rows else -1)
        if phase=='finish':return self.lib.prefill_finish(C.byref(self.f),arm)
        snapshot=phase=='snapshot'
        return self.lib.prefill_transition(C.byref(self.t),arm,self.f.weights if snapshot else None,
            self.f.weights_stride,self.ptr('retained') if snapshot else None)

    def output_name(self,phase):
        return 'prepare_output' if phase=='prepare' else super().output_name(phase)

    def reset_output(self,phase):
        super().reset_output(phase)
        if phase=='prepare':
            for name in ('src_rows','dst_rows','prepare_inverse_rows'):self.buffers[name].poison()

    def reference(self,phase):
        if phase!='prepare':return super().reference(phase)
        self.reset_output(phase);checked(self.launch(phase,0),'prepare gather control');self.prepare_check()
        return self.buffers['prepare_output'].read()

    def exact(self,phase,reference):
        super().exact(phase,reference)
        if phase=='prepare':self.prepare_check()

    def validate(self,phase):
        ref=self.reference(phase)
        for arm in (1,2):
            self.reset_output(phase);checked(self.launch(phase,arm),'prefill helper candidate');self.exact(phase,ref)

    def geometry(self,phase,arm):
        c=self.case
        slot,call=(0,self.p) if phase=='prepare' else (2,self.f) if phase=='finish' else (1,self.t)
        fast=bool(self.lib.fast(slot,C.byref(call)))
        if phase=='prepare':
            if arm and fast:
                rows=c.tokens if arm==2 else self.rows
                return 'prepare_vec4<',[(c.columns+1023)//1024,min(rows,65535),1],[256,1,1]
            return 'prepare_gather<',self.row_grid(self.rows,c.columns),[256,1,1]
        if arm and fast:
            threads=128 if arm==1 else 256
            rows,cols=(c.tokens,c.output) if phase=='finish' else (self.rows,c.hidden)
            symbol='finish_vec4' if phase=='finish' else 'transition_vec4'
            return f'{symbol}<{threads}',[(cols+threads*4-1)//(threads*4),min(rows,65535),1],[threads,1,1]
        cols=c.output if phase=='finish' else c.hidden
        return ('finish_kernel(' if phase=='finish' else 'transition_kernel'),self.row_grid(
            c.tokens if phase=='finish' else self.rows,cols),[256,1,1]


def boundary_gate(rt,lib):
    cases=[case(0,'prepare'),case(1,'prepare',columns=4),case(8193,'prepare',columns=4),
           case(17,'prepare',columns=2052,padding=4),case(3,'prepare',columns=2051,padding=3),
           case(17,'prepare',columns=32,topk=6,experts=33),
           case(1,'finish',output=4),case(3,'finish',output=36,padding=4),case(65536,'finish',output=4),
           case(3,'finish',output=37,padding=3),case(9,'finish',output=36,down_splits=4),
           case(3,'finish',output=32,topk=6,experts=33),
           case(8193,'transition',hidden=4),case(3,'transition',hidden=516,padding=4),
           case(3,'transition',hidden=33,padding=3),case(3,'transition',hidden=32,state=1,splits=4),
           case(3,'transition',hidden=32,merged=0),case(3,'transition',hidden=32,output_order=1)]
    cases += [case(3,p,dtype=d,columns=32,hidden=32,output=32)
              for p in ('prepare','transition','finish') for d in (0,1)]
    for c in cases:
        with ExitStack() as stack:
            b=Fixture(rt,lib,c);stack.callback(b.close);b.update(37)
            if c.phase=='finish' and c.tokens==1:
                values=np.broadcast_to(np.array([1e20,1,-1e20,3,0,0,0,0],np.float32)[:,None],(8,4)).copy()
                b.buffers['down'].upload(encode(values[b.maps[1]],2).tobytes())
                b.weights_host[:]=1;b.buffers['weights'].upload(b.weights_host.tobytes())
                b.finish_expected=weighted(rounded(values,2).reshape(1,8,4),b.weights_host)
            b.validate(c.phase)
            if c.phase=='transition':b.validate('snapshot')
    if lib.prefill_prepare(None,1,-1)!=20 or lib.prefill_finish(None,1)!=20:
        raise ValueError('null prepare/finish accepted')
    if lib.prefill_transition(None,1,None,0,None)!=20:raise ValueError('null transition accepted')
    print('PREFILL_HELPER_BOUNDARIES PASS cases='+str(len(cases)),flush=True)


def stream_gate(sdk,lib):
    with ExitStack() as stack:
        entries=[]
        for phase in ('prepare','finish','snapshot'):
            for lane in range(2):
                rt=Runtime(sdk);stack.callback(rt.close)
                c=case(17,'transition' if phase=='snapshot' else phase,
                       columns=516,hidden=516,output=516,padding=4)
                b=Fixture(rt,lib,c);stack.callback(b.close);b.update(19+lane)
                arm=1+lane
                launch=(lambda b=b,a=arm:lib.prefill_prepare(C.byref(b.p),a,-1)) if phase=='prepare' else (
                    lambda b=b,a=arm,p=phase:b.launch(p,a))
                graph=Graph(rt,launch,1);stack.callback(graph.close);entries.append((b,phase,graph))
        for seed in (71,83):
            refs=[]
            for b,phase,_ in entries:
                b.update(seed+len(refs));refs.append(b.reference(phase));b.reset_output(phase)
            for _,_,g in entries:g.run()
            for (b,phase,_),ref in zip(entries,refs):b.exact(phase,ref)
    print('PREFILL_HELPER_REPLAY PASS streams=2_per_phase changing_inputs=1',flush=True)


def alias_gate(rt,lib):
    with ExitStack() as stack:
        b=Fixture(rt,lib,case(3,'transition',hidden=32,output=36));stack.callback(b.close)
        f=Finish.from_buffer_copy(b.f);f.weights=b.ptr('retained');f.weights_stride=b.case.topk
        for arm in (1,2):
            def chain(a=arm):
                rc=lib.prefill_transition(C.byref(b.t),a,b.f.output,b.f.weights_stride,b.ptr('retained'))
                return rc if rc else lib.prefill_finish(C.byref(f),a)
            g=Graph(rt,chain,1)
            try:
                for seed in (47,53):
                    b.update(seed);ref=b.reference('finish');b.reset_output('finish')
                    rt.write(b.f.output,b.weights_host.tobytes());g.run()
                    b.exact('finish',ref);b.check_snapshot('snapshot')
            finally:g.close()
    print('PREFILL_HELPER_ALIAS PASS weights_inside_finish_output=1 extra_copy_kernel=0',flush=True)


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
                phases=('transition','snapshot') if c.phase=='transition' else (c.phase,)
                for seed in (7,19):
                    b.update(seed)
                    for phase in phases:b.validate(phase)
                graphs={};entries={}
                for phase in phases:
                    labels=('control','row_vec4','token_vec4') if phase=='prepare' else ('control','vec4_128','vec4_256')
                    point=dict(key=c.key+'/'+phase,case=asdict(c),phase=phase,components={})
                    for arm,label in enumerate(labels):
                        key=point['key']+'/'+label;point['components'][label]=[key]
                        graph=Graph(rt,lambda p=phase,a=arm:b.launch(p,a),repeats);stack.callback(graph.close)
                        for _ in range(3):graph.run()
                        rt.sync();graphs[key]=graph
                        symbol,grid,block=b.geometry(phase,arm)
                        entries[key]=dict(component=key,kernel=symbol,grid=grid,block=block,repeats=repeats)
                    points.append(point)
                rt.sync();checked(rt.ProfilerStart(),'prefill helpers profile begin')
                try:
                    keys=list(graphs)
                    for r in range(rounds):
                        for key in (keys if r%4 in (0,3) else keys[::-1]):
                            for s in range(samples):
                                graphs[key].run();rt.sync();schedule.append(dict(entries[key],round=r,sample=s))
                finally:
                    rt.sync();checked(rt.ProfilerStop(),'prefill helpers profile end')
                for phase in phases:b.validate(phase)
                if progress:progress(index+1,len(cases),None)
                print('PREFILL_HELPER_POINT '+json.dumps(dict(key=c.key,phase=c.phase,m=c.tokens,
                    completed=index+1,total=len(cases),correctness='PASS')),flush=True)
        clock['after']=observe(smi,lib.device['pci'],clock_mhz,advisory=True)
        return dict(points=points,schedule=schedule,device=lib.device,clock=clock,correctness='PASS',
            cases=[asdict(c) for c in cases],baseline=BASELINE,production_changed=False,
            scope='HELPER_ONLY_NO_GEMM_OR_MODEL_TPOT',protocol=dict(rounds=rounds,samples=samples,repeats=repeats))
