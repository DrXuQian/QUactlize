"""Matched F32 endpoints for the incremental dense/indexed decode sweep."""
import ctypes as C
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from dev.gemv_ppu import decode_sweep as spec, smallm, moe_s1
from dev.gemv_ppu.moe_s1_bench import fixture as moe_fixture, error, Config as OldConfig
from dev.gemv_ppu.moe_compare_bench import TensorCore as GroupedTC, Unsupported, upload
from dev.gemv_ppu.run import query_l2_attribute, resolve_l2
from quactlize.execution.native import Call as VecCall, Arrangement, arrangement
from quactlize.runtime.native import SDK, Module, Call, Recipe, Resources as Query, checked
from tools.kpack_execution_fixture import IndexedWeights
from tools.run_kpack_gemv_gate import Resources
from tools.run_kpack_grouped_device_gate import graph_bind
from tools.run_kpack_grouped_decode_probe import Replay
from tools.run_kpack_pack_gate import device_identity


def truth(w, data, ids=None):
    expert = data['expert'] if ids is None else ids[:, :8].reshape(-1)
    positions = [int(np.flatnonzero(w.categories[0] == g)[0]) for g in range(4)]
    values = data['ah'][:, positions].astype('f8')
    return dict(golden=np.stack([values[data['arows'][i]] @ w.sums[e] for i,e in enumerate(expert)]),
                denom=np.stack([np.abs(values[data['arows'][i]]) @ w.abs_sums[e] for i,e in enumerate(expert)]))


def fixture(w, workload):
    t, n, k = workload['tokens'], w.n, w.k
    if workload['operator'] == 'grouped':
        data = moe_fixture(w, t, workload['channels'])
        data['ids'][:, :8] = spec.routed_ids(t, workload['router'])
        data['expert'] = data['ids'][:, :8].reshape(-1)
    else:
        values = np.random.default_rng(811+101*t+7).normal(0, .2, (t, 4)).astype('<f4')
        a = np.full((t, k+8), np.nan, dtype='<f4'); a[:, :k] = values[:, w.categories[0]]
        data = dict(a=a, ah=a[:, :k].astype('<f2'), ids=None, expert=np.zeros(t, dtype=int),
                    arows=np.arange(t), tokens=t, channels=1, rows=t, n=n, k=k)
    data.update(truth(w, data))
    if not np.isfinite(data['golden']).all() or not np.all(data['denom'] > 0):
        raise ValueError('invalid independent original-GGUF fixture')
    return data


def probe(sdk, library, l2_bytes):
    fn = library.q4_ppu_probe
    fn.argtypes = [C.POINTER(C.c_int)]*3+[C.c_char_p]; fn.restype=C.c_int
    l2, sm, warp, name = C.c_int(), C.c_int(), C.c_int(), C.create_string_buffer(256)
    checked(fn(C.byref(l2), C.byref(sm), C.byref(warp), name), 'image marker/device probe')
    if warp.value != 32:
        raise ValueError('reader requires 32-lane warps')
    return device_identity(sdk) | resolve_l2(l2.value, l2_bytes, query_l2_attribute(sdk.lib)) | dict(
        name=name.value.decode(), properties_sm=sm.value, warp=warp.value)


class Base:
    def __init__(self, args, workload):
        self.workload=workload;self.bundle=args.bundle;self.n=workload['n'];self.k=workload['k']
        self.sdk=SDK(args.sdk);graph_bind(self.sdk)
        self.r=Resources(self.sdk);self.case_r=Resources(self.sdk);self.graphs={};self.old=None
        n,k=self.n,self.k;e=workload['experts'];rows=workload['rows']
        self.library=C.CDLL(str(args.bundle/f'libq4_decode_n{n}_k{k}.so'),mode=C.RTLD_LOCAL)
        self.run=self.library.q4_decode_sweep_run
        self.run.argtypes=[C.POINTER(VecCall),C.c_int,C.POINTER(Arrangement)];self.run.restype=C.c_int
        self.ids_by_key={c.key:i for i,c in enumerate(spec.simt_inventory(n,k))}
        self.device=probe(self.sdk,self.library,args.l2_bytes)
        self.w=IndexedWeights(12,n,k,e,progress=lambda done,total:print(
            f'Q4_DECODE_FIXTURE case={workload["id"]} experts={done}/{total}',flush=True))
        self.data=fixture(self.w,workload)
        self.expert_bytes=n*k*9//16;self.active_minimum=8 if e>1 else 1
        self.copies=max(2,math.ceil(2.25*self.device['l2_bytes']/(self.active_minimum*self.expert_bytes)))
        self.calls_per_graph=max(2,math.ceil(32/self.copies))*self.copies
        self.planes=[];self.plane_sizes={key:self.w.planes[key].nbytes for key in ('low','units')}
        bases={key:self.r.alloc(size*self.copies) for key,size in self.plane_sizes.items()}
        for i in range(self.copies):
            record={}
            for key,size in self.plane_sizes.items():
                record[key]=bases[key]+i*size
                checked(self.sdk.lib.hggcMemcpy(record[key],self.w.planes[key].ctypes.data,size,1),'weight ring upload')
            self.planes.append(record)
        self.zero_low=self.r.alloc(self.plane_sizes['low']);self.r.fill(self.zero_low,0,self.plane_sizes['low'])
        self.weight_sha256={key:hashlib.sha256(self.w.planes[key]).hexdigest() for key in self.plane_sizes}
        self.arr=arrangement(12);self.output_stride=n+8;self.output_bytes=rows*self.output_stride*4
        self.out_base=self.case_r.alloc(self.output_bytes+32);self.out=self.out_base+16
        self.a=self.case_r.upload(self.data['a'])
        self.ids=self.case_r.upload(self.data['ids']) if e>1 else None
        self.base=VecCall(version=1,size=C.sizeof(VecCall),qtype=12,n=n,k=k,experts=e,rows=rows,
            mode=2 if e>1 else 0,input_type=1,channels=workload['channels'],topk=8 if e>1 else 1,
            a_row_stride=k+8,a_token_stride=workload['channels']*(k+8),ids_stride=11 if e>1 else 0,
            out_row_stride=self.output_stride,a=self.a,ids=self.ids,output=self.out,stream=self.case_r.stream.value,
            low=self.planes[0]['low'],units=self.planes[0]['units'])
        self.case_r.fill(self.out_base,0xa5,self.output_bytes+32)
        self.sdk.synchronize(None)

    def invoke(self,key,call=None):
        return self.run(C.byref(self.base if call is None else call),self.ids_by_key[key],C.byref(self.arr))

    def poison(self):
        self.case_r.fill(self.out_base,0xa5,self.output_bytes+32)

    def read(self):
        self.sdk.synchronize(self.case_r.stream)
        raw=np.frombuffer(self.sdk.download(self.out_base,self.output_bytes+32),dtype='u1')
        out=raw[16:-16].view('<f4').reshape(self.workload['rows'],self.output_stride)
        if not np.all(raw[:16]==0xa5) or not np.all(raw[-16:]==0xa5) or not np.all(out[:,self.n:].view('u1')==0xa5):
            raise ValueError('output guard/padding differs')
        return out[:,:self.n].copy()

    def old_control(self,key,got):
        """Untimed immutable-image checks preserve historical reader semantics."""
        n,k=self.n,self.k;config=spec.recipe(n,k,key)
        if self.workload['operator']=='dense' and (n,k) in smallm.SHAPES:
            previous={spec.old_dense_recipe(n,k,c.key):i for i,c in enumerate(smallm.inventory(n,k))}
            if config not in previous:return 'UNMEASURED_RECIPE_EXTENSION'
            path=spec.ROOT/'prebuilt/ppu0010/q4-smallm-v1'/smallm.payload(n,k)
            manifest=json.loads((path.parent/'manifest.json').read_text())
            if sha_file(path)!=manifest['payloads'][path.name]['sha256']:raise ValueError('immutable dense image differs')
            lib=C.CDLL(str(path),mode=C.RTLD_LOCAL);fn=lib.q4_smallm_run
            fn.argtypes=[C.c_int]*3+[C.c_void_p]*5;fn.restype=C.c_int
            a=self.case_r.upload(np.ascontiguousarray(self.data['ah']))
            out=self.case_r.alloc(got.nbytes)
            checked(fn(previous[config],self.workload['rows'],0,a,self.base.low,self.base.units,out,self.case_r.stream),'old dense body')
            self.sdk.synchronize(self.case_r.stream)
            old=np.frombuffer(self.sdk.download(out,got.nbytes),dtype='<f4').reshape(got.shape)
        elif self.workload['operator']=='grouped' and config.columns==4 and (
                moe_s1.Recipe(config.reader,config.variant,config.warps,config.values) in moe_s1.inventory()):
            path=spec.moe_compare.SIMT_BUNDLE/moe_s1.payload(n,k)
            manifest=json.loads((path.parent/'manifest.json').read_text())
            if sha_file(path)!=manifest['payloads'][path.name]:raise ValueError('immutable indexed image differs')
            lib=C.CDLL(str(path),mode=C.RTLD_LOCAL);fn=lib.quactlize_q4_s1_run_v1
            fn.argtypes=[C.POINTER(VecCall),C.POINTER(OldConfig),C.POINTER(Arrangement)];fn.restype=C.c_int
            c=OldConfig(moe_s1.Recipe(config.reader,config.variant,config.warps,config.values))
            self.poison();checked(fn(C.byref(self.base),C.byref(c),C.byref(self.arr)),'old indexed body');old=self.read()
        else:return 'NEW_SHAPE_NO_OLD_IMAGE'
        if not np.array_equal(got.view('u4'),old.view('u4')):raise ValueError('transferred reader differs from immutable old image')
        return 'EXACT_BITS'

    def correctness(self,key):
        self.poison();checked(self.invoke(key),'SIMT eager');got=self.read();err=error(got,self.data)
        if err>=.005:raise ValueError(f'SIMT independent GGUF error {err:.9g}')
        c=VecCall.from_buffer_copy(self.base)
        c.a=self.case_r.upload(self.data['a'].astype('<f2'));c.input_type=0
        self.poison();checked(self.invoke(key,c),'SIMT F16 control')
        if not np.array_equal(got.view('u4'),self.read().view('u4')):raise ValueError('F32-rounded/F16 differs')
        old=self.old_control(key,got)
        c=VecCall.from_buffer_copy(self.base);c.low=self.zero_low
        self.poison();checked(self.invoke(key,c),'SIMT zero-code plant')
        if error(self.read(),self.data)<=.005:raise ValueError('zero-code plant escaped oracle')
        if self.ids:
            bad=self.data['ids'].copy();bad[:,:8]=256;c=VecCall.from_buffer_copy(self.base)
            c.ids=self.case_r.upload(bad);self.poison();checked(self.invoke(key,c),'SIMT invalid IDs')
            if not np.isnan(self.read()).all():raise ValueError('invalid expert ID was not rejected')
        replay=Replay(self.sdk,self.case_r.stream,lambda:self.invoke(key),1)
        try:
            if self.ids:
                changed=self.data['ids'].copy();changed[:,:8]=np.roll(changed[:,:8],1,axis=1)
                upload(self.sdk,self.ids,changed);target=truth(self.w,self.data,changed)
            else:
                upload(self.sdk,self.a,-self.data['a']);target=dict(golden=-self.data['golden'],denom=self.data['denom'])
            self.poison();checked(replay(),'mutable input graph');new=self.read()
            if error(new,target)>=.005 or error(new,self.data)<=.005:raise ValueError('mutable-input graph replay failed')
            if self.ids:upload(self.sdk,self.ids,self.data['ids'])
            upload(self.sdk,self.a,self.data['a'])
            self.poison();checked(replay(),'restored graph')
            if not np.array_equal(got.view('u4'),self.read().view('u4')):raise ValueError('eager/restored graph differs')
            upload(self.sdk,self.a,np.zeros_like(self.data['a']))
            self.poison();checked(replay(),'zero A')
            if np.any(self.read()!=0):raise ValueError('zero A failed')
        finally:
            if self.ids:upload(self.sdk,self.ids,self.data['ids'])
            upload(self.sdk,self.a,self.data['a']);replay.close()
        self.poison();checked(self.invoke(key),'restored eager');self.read()
        return dict(error=err,zero_codes='PASS',zero_a='PASS',output_guard='PASS',
                    mutable_input_replay='PASS',eager_graph_bits='PASS',f16_f32_bits='PASS',immutable_reader=old)

    def graph(self,key):
        if key not in self.graphs:
            index=0
            def launch():
                nonlocal index
                c=VecCall.from_buffer_copy(self.base);plane=self.planes[index%self.copies];index+=1
                c.low,c.units=plane['low'],plane['units'];return self.invoke(key,c)
            self.graphs[key]=Replay(self.sdk,self.case_r.stream,launch,self.calls_per_graph)
            checked(self.graphs[key](),'SIMT graph upload excluded');self.sdk.synchronize(self.case_r.stream)
        return self.graphs[key]

    def measure(self,key,count):
        values=[x/self.calls_per_graph for x in self.case_r.samples(self.graph(key),count)]
        if error(self.read(),self.data)>=.005:raise ValueError('SIMT timed output differs')
        return values

    def close(self):
        self.sdk.synchronize(self.case_r.stream)
        for graph in self.graphs.values():graph.close()
        self.graphs.clear();self.case_r.close();self.r.close()


def sha_file(path):
    from quactlize.runtime.compiler import sha
    return sha(path)


class DenseTC:
    def __init__(self,base,bundle,record,candidate):
        self.base=base;self.candidate=candidate;self.handles=[];self.graph=None;self.r=Resources(base.sdk);self.module=None
        try:
            self.module=Module(record|dict(path=str((bundle/record['path']).resolve())))
            dev=self.module.device_identity()
            if dev['ordinal']!=base.device['ordinal'] or dev['compute_units']!=72:raise ValueError('TC device differs')
            rows=base.workload['rows'];n,k=base.n,base.k
            self.recipe=Recipe(1,C.sizeof(Recipe),0,candidate['split'],0);self.resources=Query()
            self.compact_guard=self.r.alloc(rows*n*2+32);self.output_bytes=rows*n*2
            self.call=Call(version=1,size=C.sizeof(Call),m=rows,n=n,k=k,experts=1,group_size=32,
                device=dev['ordinal'],compute_units=dev['compute_units'],mapping_id=base.arr.mapping_id,
                a=self.r.alloc(rows*k*2),low=base.planes[0]['low'],metadata=base.planes[0]['units'],
                output=self.compact_guard+16,stream=self.r.stream.value)
            rc=self.module.query(C.byref(self.call),C.byref(self.recipe),C.byref(self.resources))
            if rc==1:raise Unsupported('DEVICE_QUERY_UNSUPPORTED')
            checked(rc,'dense TC query')
            self.workspace_guard=self.r.alloc(self.resources.workspace_bytes+256)
            self.r.fill(self.workspace_guard,0xa5,self.resources.workspace_bytes+256)
            self.call.workspace=self.workspace_guard+128;self.call.workspace_bytes=self.resources.workspace_bytes
            self.adapter=C.CDLL(str((bundle/'libq4_dense_io.so').resolve()),mode=C.RTLD_LOCAL)
            self.cast=self.adapter.q4_decode_dense_cast
            self.cast.argtypes=[C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_void_p];self.cast.restype=C.c_int
            for i in range(base.copies):self.handles.append(self.handle(i))
            self.negative=self.handle(0,base.zero_low);self.handles.append(self.negative)
            self.poison();base.sdk.synchronize(self.r.stream)
        except BaseException:
            self.close();raise

    def handle(self,index,low=None):
        c=Call.from_buffer_copy(self.call);c.low=low if low is not None else self.base.planes[index]['low']
        c.metadata=self.base.planes[index]['units'];h=C.c_void_p()
        checked(self.module.prepare(C.byref(c),C.byref(self.recipe),C.byref(h)),'dense TC prepare')
        return h

    def launch(self,index=0,negative=False):
        b=self.base;r=self.r.stream;m=b.workload['rows']
        rc=self.cast(1,b.a,self.call.a,m,b.k,b.k+8,r)
        if rc:return rc
        rc=self.module.run(self.negative if negative else self.handles[index],r)
        if rc:return rc
        return self.cast(0,self.call.output,b.out,m,b.n,b.output_stride,r)

    def poison(self):
        self.r.fill(self.base.out_base,0xa5,self.base.output_bytes+32)
        self.r.fill(self.compact_guard,0xa5,self.output_bytes+32)

    def read(self):
        b=self.base;b.sdk.synchronize(self.r.stream)
        for p,size in ((self.workspace_guard,128),(self.call.workspace+self.call.workspace_bytes,128),
                       (self.compact_guard,16),(self.call.output+self.output_bytes,16)):
            if b.sdk.download(p,size)!=b'\xa5'*size:raise ValueError('dense scratch guard differs')
        return b.read()

    def correctness(self):
        b=self.base;self.poison();checked(self.launch(),'dense eager');got=self.read();err=error(got,b.data)
        if err>=.005:raise ValueError(f'dense TC independent GGUF error {err:.9g}')
        self.poison();checked(self.launch(negative=True),'dense zero-code plant')
        if error(self.read(),b.data)<=.005:raise ValueError('dense zero-code escaped oracle')
        replay=Replay(b.sdk,self.r.stream,self.launch,1)
        try:
            upload(b.sdk,b.a,-b.data['a']);self.poison();checked(replay(),'dense graph changed A')
            changed=self.read()
            if error(changed,dict(golden=-b.data['golden'],denom=b.data['denom']))>=.005 or error(changed,b.data)<=.005:
                raise ValueError('dense mutable-A replay failed')
            upload(b.sdk,b.a,b.data['a']);self.poison();checked(replay(),'dense restored graph')
            if not np.array_equal(self.read().view('u4'),got.view('u4')):raise ValueError('dense eager/graph differs')
            upload(b.sdk,b.a,np.zeros_like(b.data['a']));self.poison();checked(replay(),'dense zero A')
            if np.any(self.read()!=0):raise ValueError('dense zero A failed')
        finally:
            upload(b.sdk,b.a,b.data['a']);replay.close()
        self.poison();checked(self.launch(),'dense restore');self.read()
        return dict(error=err,zero_codes='PASS',zero_a='PASS',output_guard='PASS',
                    mutable_input_replay='PASS',eager_graph_bits='PASS')

    def measure(self,count):
        if self.graph is None:
            index=0
            def launch():
                nonlocal index
                rc=self.launch(index%self.base.copies);index+=1;return rc
            self.graph=Replay(self.base.sdk,self.r.stream,launch,self.base.calls_per_graph)
            checked(self.graph(),'dense TC upload excluded');self.base.sdk.synchronize(self.r.stream)
        values=[v/self.base.calls_per_graph for v in self.r.samples(self.graph,count)]
        if error(self.read(),self.base.data)>=.005:raise ValueError('dense rotating output differs')
        return values

    def receipt(self):
        return dict(scope='DENSE_F32_CAST_TC_REAL_REDUCER_F32_CAST',split=self.recipe.split,grid=0,
                    shared_bytes=self.resources.shared_bytes,workspace_bytes=self.resources.workspace_bytes,
                    occupancy=self.resources.occupancy,output_rounding='F16_THEN_F32',host_routing=False,weights_moved=False)

    def close(self):
        if self.r:
            self.base.sdk.synchronize(self.r.stream)
            if self.graph:self.graph.close();self.graph=None
            if self.module:
                for h in self.handles:self.module.destroy(h)
            self.handles=[];self.r.close();self.r=None
