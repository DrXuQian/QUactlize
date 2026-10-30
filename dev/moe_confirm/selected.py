"""Exercise the public C++ selector/ticket, not a second Python recipe choice."""
import ctypes as C
import numpy as np
from pathlib import Path
import shutil
import subprocess

from dev.moe_confirm import backend
from dev.moe_confirm.backend import ROW_MAPS,prepare_receipt,Full,Vector
from dev.strategy.resources import request_abi
from quactlize.dispatch.native import Dispatch,Request,Choice
from quactlize.runtime.native import checked
from quactlize.runtime.compiler import sha
from tools.build_kpack_dispatch import catalog

ROOT=Path(__file__).resolve().parents[2]


def build(output,images):
    output=Path(output);output.mkdir()
    records=list(images['tc'].values())
    (output/'catalog.inc').write_text(catalog(records))
    for r in records:
        target=output/'modules'/r['key'];target.mkdir(parents=True,exist_ok=True)
        # A fresh private package; never move/replace an existing measured DSO.
        shutil.copy2(r['path'],target/'kernel.so')
    library=output/'libquactlize_kpack_dispatch.so'
    command=['g++','-std=c++17','-O2','-fPIC','-shared','-pthread','-Wl,-Bsymbolic',
             '-I'+str(output),str(ROOT/'quactlize/dispatch/binding.cpp'),'-ldl','-o',str(library)]
    with (output/'build.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('selected dispatcher build failed: '+(output/'build.log').read_text()[-8000:])
    return dict(root=str(output),files={str(p):sha(p) for p in output.rglob('*') if p.is_file() and p.suffix=='.so'},
                command=command)


class Context(backend.Context):
    def __init__(self,sdk,ring,w,helpers,receipt):
        super().__init__(sdk,ring,w,helpers,receipt)
        self.dispatch_root=receipt['selected_dispatch']['root']


class TensorCore(backend.TensorCore):
    def __init__(self,b,arm,record,query,sf=None):
        self.dispatch=None;self.dynamic=None
        super().__init__(b,arm,record,query,sf)
        try:
            self.dispatch=Dispatch(b.dispatch_root)
            w=b.w;p=dict(arm['recipe']['parameters']);route=2 if arm['recipe']['route']=='fq' else 3
            req=Request(1,C.sizeof(Request),w.qtype,route,w.m*w.topk,w.n,w.k,w.experts,w.m,w.mapping)
            selected=self.dispatch.prefill(req,1,w.channels,3)
            if (selected is None or selected.policy!=22 or selected.base.route!=route-2 or
                    selected.base.dequant_config!=p['dequant_config']):
                raise ValueError('public MoE prefill route/helper differs from confirmed row')
            choice=self.dispatch.query_prefill(req,1,w.channels)
            launch=arm['binding']['launch']
            if (choice.policy!=22 or choice.parent.decode()!=p['symbol'] or choice.build_key.decode()!=record['key'] or
                    (choice.algorithm,choice.split,choice.grid)!=(launch['algorithm'],launch['split'],launch['grid']) or
                    choice.workspace_bytes!=query['workspace_bytes']):
                raise ValueError('materialized MoE parent/split/grid/workspace differs from confirmation')
            lib=self.dispatch.lib
            prepare=lib.quactlize_kpack_dispatch_prepare_grouped_dynamic_v1
            prepare.argtypes=[C.c_void_p,C.POINTER(Choice),C.POINTER(C.c_void_p)];prepare.restype=C.c_int
            self.dynamic=C.c_void_p()
            checked(prepare(self.dispatch.runtime,C.byref(choice),C.byref(self.dynamic)),'selected grouped prepare')
            self.dispatch.handles.append(self.dynamic)
            self.current=request_abi(query['request'],dict(ordinal=choice.device,compute_units=choice.compute_units))
            c=self.current.device_call.call
            c.a,c.output,c.offsets_device=self.a.ptr,self.out.ptr,b.offsets.ptr
            c.workspace,c.workspace_bytes=self.buffers[-1].ptr,self.buffers[-1].size
            self.dynamic_run=lib.quactlize_kpack_dispatch_run_grouped_dynamic_v1
            self.dynamic_run.argtypes=[C.c_void_p,C.POINTER(type(self.current)),C.c_int];self.dynamic_run.restype=C.c_int
            self.receipt.update(selection='PUBLIC_CPP_POLICY22',policy=choice.policy,
                execution='ADDRESS_FREE_DYNAMIC_GROUPED_COMPLETE',dequant_config=selected.base.dequant_config)
        except BaseException:self.close();raise

    def launch(self,copy=0):
        checked(self.b.prepare(self.a.ptr),'selected public prepare v2')
        plane=(self.sf.planes if self.sf else self.b.ring.planes)[copy]
        c=self.current.device_call.call
        c.a,c.output,c.offsets_device=self.a.ptr,self.out.ptr,self.b.offsets.ptr
        c.workspace,c.workspace_bytes=self.buffers[-1].ptr,self.buffers[-1].size
        c.low,c.high,c.metadata=plane['low'],plane['high'],plane['units'];c.zero=plane.get('zero')
        c.stream=self.b.r.stream.value
        checked(self.dynamic_run(self.dynamic,C.byref(self.current),0),'selected complete grouped call')
        return self.b.finish(self.out.ptr,True)

    def timing_fixture(self):
        # Once, outside profiling: bind the native comparison to the actual
        # restored F32 input and raw IDs, not just a router profile name.
        from dev.strategy.fixture_identity import fingerprint
        b=self.b;b.rt.sync()
        return fingerprint(np.frombuffer(b.sdk.download(b.a.ptr,b.a.size),dtype='<f4'),
                           np.frombuffer(b.sdk.download(b.ids.ptr,b.ids.size),dtype='<i4'))

    def close(self):
        if self.dispatch:
            self.b.rt.sync();self.dispatch.close();self.dispatch=None
        super().close()
