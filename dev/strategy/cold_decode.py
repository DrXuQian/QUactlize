"""One cold fixture per point, separate producer/reducer/helper timers."""
import ctypes as C
from contextlib import contextmanager
import math
import time
from types import SimpleNamespace

import numpy as np

from dev.smallm_closure.bench import Buffer
from dev.smallm_closure.fixture import Weights
from dev.strategy.indexed import RuntimeView, TensorCore
from dev.strategy.pilot_native import DenseCore, scale_activation_data
from dev.strategy.isolated import Captured, checked
from dev.strategy.components import component_key, stage_role, statistics_checked
from dev.strategy.vector import Vector
from dev.strategy.journal import atomic_json
from quactlize.dispatch.table import Recipe
from quactlize.runtime.tuning import digest
from tools.run_kpack_gemv_gate import Resources


def ring_geometry(w, plane_bytes, l2_bytes, has_sf=False, *, metadata_only=False):
    if type(l2_bytes) is not int or l2_bytes<=0:
        raise ValueError('verified positive L2 bytes required')
    active=w.experts if metadata_only else w.topk if w.operation=='indexed' else 1
    packed=sum(plane_bytes.values())//w.experts*active
    # SF reads ONLY metadata. Size the ring for the SMALLEST actual read set,
    # not total allocated E weights or code planes an SF prepass never reads.
    if packed<=0 or any(type(v) is not int or v<0 for v in plane_bytes.values()):
        raise ValueError('invalid cold plane sizes')
    if (has_sf or metadata_only) and (plane_bytes['units']<=0 or has_sf and w.operation!='dense'):
        raise ValueError('standalone SF ring requires positive dense metadata')
    read=plane_bytes['units'] if metadata_only else min(packed,plane_bytes['units']) if has_sf else packed
    copies=max(2,math.ceil(2.25*l2_bytes/read))
    return dict(copies=copies,l2_bytes=l2_bytes,minimum_active=active,
        minimum_read_bytes=read,active_ring_bytes=read*copies,
        packed_resident_bytes=(plane_bytes['units'] if metadata_only else sum(plane_bytes.values()))*copies,
        cache='ROTATING_ACTIVE_READ_SET_AT_LEAST_2P25_L2',has_sf=has_sf,
        domain='ALL_EXPERT_METADATA' if metadata_only else 'ACTIVE_PACKED_WEIGHTS')


class Plane:
    def __init__(self, owner, ptr, size): self.owner,self.ptr,self.size=owner,ptr,size
    def upload(self, values):
        values=np.ascontiguousarray(values)
        if values.nbytes!=self.size: raise ValueError('plane restore size differs')
        checked(self.owner.sdk.lib.hggcMemcpy(self.ptr,values.ctypes.data,self.size,1),'plane upload')
        self.owner.sdk.synchronize(None)


class Ring:
    def __init__(self,sdk,w,l2_bytes,has_sf=False,*,weights=None,metadata_only=False):
        self.r=Resources(sdk);self.buffers=[];self.planes=[];self.objects=[]
        try:
            self.w=weights if weights is not None else Weights(w.qtype,w.n,w.k,w.experts)
            if (self.w.q,self.w.n,self.w.k,self.w.experts)!=(w.qtype,w.n,w.k,w.experts):
                raise ValueError('shared fixture weight family differs')
            self.receipt=ring_geometry(w,{k:v.nbytes for k,v in self.w.planes.items()},l2_bytes,has_sf,
                                       metadata_only=metadata_only)
            self.copies=self.receipt['copies']
            self.receipt['fixture']=self.w.record
            bases={}
            for name,v in self.w.planes.items():
                if v.size and (not metadata_only or name=='units'):
                    b=Buffer(self.r,v.nbytes*self.copies);self.buffers.append(b);bases[name]=b.ptr
            for i in range(self.copies):
                planes,objects={},{}
                for name,v in self.w.planes.items():
                    planes[name]=bases[name]+i*v.nbytes if name in bases else None
                    if name in bases:
                        view=Plane(self.r,planes[name],v.nbytes);view.upload(v);objects[name]=view
                self.planes.append(planes);self.objects.append(objects)
        except BaseException:
            self.close();raise
    def close(self): self.r.close()


class Expansion:
    def __init__(self,b,bundle,*,gold=None):
        from quactlize.dequant.native import bind,Call,TypedCall
        from dev.strategy.sf import metadata_bits
        self.b,self.r=b,Resources(b.sdk);self.buffers=[];self.calls=[];self.planes=[]
        p=b.p
        self.typed=bool(p['mode'] and p['compute']=='bf16')
        self.bound=True
        self.family=(p['q'],p['n'],p['k'],p['experts'],self.typed,b.ring)
        try:
            self.lib,self.fn,_=bind(bundle)
            self.gold=gold if gold is not None else metadata_bits(b.ring.w.planes['units'],p['q'],p['n'],p['k'],p['experts'],self.typed)
            if len(self.gold)!=2 or any(v.dtype!=np.dtype('<u2') or v.shape!=(p['experts'],p['k']//b.arr.group_size,p['n']) for v in self.gold):
                raise ValueError('typed expansion oracle layout differs')
            if self.typed:
                self.fn=self.lib.quactlize_kpack_dequant_v2
                self.fn.argtypes=[C.POINTER(TypedCall),C.POINTER(type(b.arr))];self.fn.restype=C.c_int
            self.width=self.gold[0].nbytes
            scale,zero=(Buffer(self.r,self.width*b.ring.copies) for _ in range(2))
            self.buffers=[scale,zero]
            for i,plane in enumerate(b.ring.planes):
                c=Call(1,C.sizeof(Call),p['q'],p['n'],p['k'],p['experts'],0,0,
                    plane['low'],plane['high'],plane['units'],
                    b.ring.w.planes['low'].nbytes,b.ring.w.planes['high'].nbytes,b.ring.w.planes['units'].nbytes,
                    scale.ptr+i*self.width,zero.ptr+i*self.width,self.width,b.r.stream.value)
                self.calls.append(TypedCall(2,C.sizeof(TypedCall),c,1) if self.typed else c)
                self.planes.append(dict(low=plane['low'],high=plane['high'],units=c.output,zero=c.zero))
        except BaseException:
            self.close();raise
    def launch(self,index=0,config=0):
        if not self.bound:raise ValueError('expansion has no live consumer stream')
        call=self.calls[index].call if self.typed else self.calls[index]
        call.config=config
        return self.fn(C.byref(self.calls[index]),C.byref(self.b.arr))
    def check(self,index=0):
        from tools.kpack_dequant_fixture import compare
        self.b.sdk.synchronize(self.b.r.stream)
        result=[]
        call=self.calls[index].call if self.typed else self.calls[index]
        for ptr,gold in zip((call.output,call.zero),self.gold):
            got=np.frombuffer(self.b.sdk.download(ptr,self.width),dtype='<u2').reshape(gold.shape)
            result.append(compare(got,gold.view('<u2')))
        return result
    def poison(self):
        for buffer in self.buffers: buffer.poison()
    def detach(self):
        if self.bound:self.b.sdk.synchronize(self.b.r.stream)
        self.bound=False
    def rebind(self,b):
        p=b.p
        if self.family!=(p['q'],p['n'],p['k'],p['experts'],bool(p['mode'] and p['compute']=='bf16'),b.ring):
            raise ValueError('expansion rebind changed weight/precision/ring')
        self.detach();self.b=b;self.bound=True
        for request in self.calls:
            (request.call if self.typed else request).stream=b.r.stream.value
    def close(self):
        self.detach();self.r.close()


class Core:
    def __init__(self,b,w,arm,record,query,sf=None):
        r=Recipe.from_dict(arm['recipe']);self.b=b;self.sf=sf if r.route=='sf' and w.qtype!=8 else None
        if r.kind!='tc':
            self.inner=Vector(b,record,arm['spec'],dict(r.parameters).get('split',1))
        else:
            view=b if self.sf is None else SimpleNamespace(**(vars(b)|{'ring':SimpleNamespace(planes=self.sf.planes)}))
            self.inner=TensorCore(view,record,r,w) if w.operation=='indexed' else DenseCore(view,record,arm,query)
        self.receipt=self.inner.receipt
    def launch(self,index=0):
        if self.sf: checked(self.sf.launch(index),'untimed/separately captured SF expansion')
        return self.inner.launch(index)
    def close(self): self.inner.close()


def load_component(path,identity):
    from tools.run_kpack_strategy import record_load
    return record_load(path,identity)


def component_capture(rt,capture,w,r,authority,source,output,proof,rounds=3,samples=3,*,separate_expansion=False):
    """Each timing graph has ONE kernel. No API spans or subtraction."""
    rows=[];pending=[]
    split=dict(r.parameters).get('split',1) if r else 1
    nodes=capture.descriptions[:len(capture.groups)]
    kinds=[stage_role(n['name'],split) for n in nodes]
    if r and (kinds.count('producer')!=1 or 'unclassified' in kinds or
              split>1 and not any(k in ('reducer','reducer_finish') for k in kinds)):
        raise ValueError('cold captured stage inventory differs: '+repr(list(zip(kinds,nodes))))
    if not r and kinds!=['expansion']: raise ValueError('SF diagnostic must contain exactly one expansion')
    # Independent oracle on the extracted serial sequence, before any timing.
    for stage in range(len(nodes)):
        graph=capture.isolate(stage,weight_ring=False,validation_only=True);graph.close()
    proof()
    for stage,(node,kind) in enumerate(zip(nodes,kinds)):
        rotating=kind in ('producer','expansion')
        cache='ROTATING_ACTIVE_READ_SET_AT_LEAST_2P25_L2' if rotating else 'REPEATED_RESIDENT_SCRATCH'
        node=dict(node,split=split)
        if separate_expansion and kind=='expansion':node['expert_domain']='ALL'
        cid,identity=component_key(w,r,node,authority=authority,source=source,cache=cache)
        path=output/'components'/(cid+'.json')
        if path.exists(): value=load_component(path,identity)
        else:
            if separate_expansion and r is not None and kind=='expansion':
                raise ValueError('SF must be timed on its independent metadata ring before GEMM')
            graph=capture.isolate(stage,weight_ring=rotating,repeats=16)
            try:
                values=[[graph.sample() for _ in range(samples)] for _ in range(rounds)]
                value=statistics_checked(values)|dict(identity=identity,component_id=cid,kind=kind,
                    timed_kernel_types=1,calls_per_graph=graph.count,first_use_excluded=True,
                    host_launch_cost_added=False,cache=cache)
                value['record_sha256']=digest(value)
                pending.append((path,value))
            finally: graph.close()
        rows.append(dict(id=cid,role=kind,median_us=value['median_us'],spread_pct=value['spread_pct']))
    # Timed producers may leave a different copy's partials. Revalidate the
    # extracted sequence on copy0 before publishing any reusable timings.
    for stage in range(len(nodes)):
        graph=capture.isolate(stage,weight_ring=False,validation_only=True);graph.close()
    proof()
    # The caller must also pass its final full-call oracle/redzones and close
    # the GPU objects. A late failure cannot publish reusable timing evidence.
    return rows,pending


def publish_components(pending):
    for path,value in pending: atomic_json(path,value)


def replay_extracted(capture,proof):
    for stage in range(len(capture.groups)):
        graph=capture.isolate(stage,weight_ring=False,validation_only=True);graph.close()
    return proof()


@contextmanager
def prepared_arm(b,w,arm,record,query,sf,*,expansion_proven=False):
    """Keep exact handles/arguments alive for alternating single-stage timers."""
    core=None;capture=None;rt=RuntimeView(b);started=time.monotonic()
    r=Recipe.from_dict(arm['recipe'])
    try:
        core=Core(b,w,arm,record,query,sf)
        proofs=[]
        for repeat in (0,1):
            b.update(repeat)
            for copy in (0,b.ring.copies-1):
                b.out.poison();checked(core.launch(copy),'cold complete-call correctness');proofs.append(b.check())
        if w.compute=='bf16':
            b.update(2);scale_activation_data(b.data,2**20);b.a.upload(b.data['a'])
            if float(np.max(np.abs(b.data['a'])))<=65504: raise ValueError('missing BF16 large activation')
            checked(core.launch(),'BF16 large activation');proofs.append(b.check())
        b.update(0);b.a.upload(np.zeros_like(b.data['a']));b.out.poison()
        checked(core.launch(),'zero A');rt.sync()
        zero=b.out.read((b.rows,w.n))
        if not np.isfinite(zero).all() or np.any(zero): raise ValueError('zero A did not produce finite zeros')
        try: b.check()
        except ValueError as exc:
            if not str(exc).startswith('typed oracle mismatch:'): raise
        else: raise ValueError('independent oracle accepted deliberately wrong zero output')
        b.update(0)
        for i in range(b.ring.copies): checked(core.launch(i),'excluded whole-ring setup')
        if core.sf and not expansion_proven:
            core.sf.check(0);core.sf.check(b.ring.copies-1)
        b.check();pointer_checks=[]
        for name,buf in b.ring.objects[-1].items():
            try:
                b.ring.r.fill(buf.ptr,0,buf.size);b.sdk.synchronize(b.ring.r.stream)
                checked(core.launch(b.ring.copies-1),'last-copy negative')
                try: b.check()
                except ValueError as exc:
                    if not str(exc).startswith('typed oracle mismatch:'): raise
                else: raise ValueError('last-copy plane not consumed: '+name)
                checked(core.launch(0),'first-copy isolation');b.check();pointer_checks.append(name)
            finally:
                rt.sync();buf.upload(b.ring.w.planes[name])
        for i in range(b.ring.copies): checked(core.launch(i),'restored ring')
        capture=Captured(rt,[lambda i=i:core.launch(i) for i in range(b.ring.copies)])
        split=dict(r.parameters).get('split',1)
        kinds=[stage_role(n['name'],split) for n in capture.descriptions[:len(capture.groups)]]
        if kinds.count('producer')!=1: raise ValueError('missing captured producer: '+repr(kinds))
        b.update(1);b.out.poison()
        for stage,kind in enumerate(kinds):
            if kind!='producer':
                graph=capture.isolate(stage,weight_ring=False,validation_only=True);graph.close()
        try: b.check()
        except ValueError as exc:
            if not str(exc).startswith('typed oracle mismatch:'): raise
        else: raise ValueError('omitted producer incorrectly passed')
        # Dense typed modules own conversion scratch per handle. Update all
        # copies, not only copy0, before timing producers with the changed A.
        for i in range(b.ring.copies): checked(core.launch(i),'changed-input ring setup')
        b.out.poison()
        replay_extracted(capture,b.check)
        result=dict(correctness='PENDING',numerical=proofs,pointer_negatives=pointer_checks,
            omitted_producer='DETECTED',zero_input='PASS',receipt=core.receipt,fixture=b.ring.receipt,
            performance_admitted=False,cost_scope='ISOLATED_KERNEL_SUM_SCREEN_NOT_MODEL')
        yield SimpleNamespace(capture=capture,core=core,rt=rt,proof=b.check,data=result,recipe=r)
        # Another finalist may have used the shared output. Verify this arm's
        # own extracted sequence, then its complete call, before admitting it.
        b.out.poison();replay_extracted(capture,b.check)
        checked(core.launch(),'post-timer complete correctness');proofs.append(b.check())
        owned=getattr(core.inner,'buffers',[v for v in vars(core.inner).values() if isinstance(v,Buffer)])
        for buffer in [*b.ring.buffers,*owned,*(sf.buffers if sf else [])]: buffer.guard()
        result.update(correctness='PASS',wall_seconds=time.monotonic()-started)
    finally:
        if capture: capture.close()
        if core: core.close()


def measure_arm(b,w,arm,record,query,sf,authority,source,output,*,separate_expansion=False):
    with prepared_arm(b,w,arm,record,query,sf,expansion_proven=separate_expansion) as session:
        rows,pending=component_capture(session.rt,session.capture,w,session.recipe,authority,source,output,
                                      session.proof,separate_expansion=separate_expansion)
    session.data.update(components=[x['id'] for x in rows],stages=rows)
    publish_components(pending)
    return session.data


def measure_expansion(b,w,sf,config,authority,source,output,*,separate_expansion=False):
    rt=RuntimeView(b)
    # Do not let a no-op/new config inherit correct output from a previous one.
    sf.poison()
    for i in range(b.ring.copies): checked(sf.launch(i,config),'independent SF setup')
    proof=[sf.check(0),sf.check(b.ring.copies-1)]
    for buffer in sf.buffers: buffer.guard()
    capture=Captured(rt,[lambda i=i:sf.launch(i,config) for i in range(b.ring.copies)])
    try:
        options=dict(separate_expansion=True) if separate_expansion else {}
        rows,pending=component_capture(rt,capture,w,None,authority,source,output,lambda:sf.check(0),**options)
        for buffer in [*b.ring.buffers,*sf.buffers]: buffer.guard()
        result=dict(correctness='PASS',components=[x['id'] for x in rows],stages=rows,
                    numerical=proof,config=config,scope='SF_EXPANSION_ONLY_NO_GEMM',
                    fixture=getattr(b.ring,'receipt',None))
    finally: capture.close()
    publish_components(pending)
    return result
