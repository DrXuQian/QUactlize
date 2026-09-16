"""Exact selected dense decode endpoint for a bounded profiler invocation."""
import ctypes as C
import sys

import numpy as np

from dev.bf16_compute.fixture import round_compute
from dev.gemv_simt.fixture import weights
from quactlize.dispatch.native import Dispatch, Request
from quactlize.execution.native import arrangement, Call as SimtCall
from quactlize.runtime.native import Call
from tools.kpack_warmup_fixture import activation_values
from tools.run_kpack_decode_io import output_check


def check_choice(choice, point):
    if choice is None:
        raise ValueError('observed TC recipe is no longer selected')
    actual=dict(parent=choice.parent.decode(),build=choice.build_key.decode(),
                split=choice.split,algorithm=choice.algorithm,grid=choice.grid,policy=choice.policy)
    expected={key:point[key] for key in actual}
    if actual!=expected:
        raise ValueError(f'TC recipe changed since Asys: {actual} != {expected}')


def select_choice(dispatch, point, arr):
    """Use the caller's selector, not the older decode-only TC policy."""
    p=point
    if p['policy'] in (12,13,14):
        call=SimtCall(version=1,size=C.sizeof(SimtCall),qtype=p['q'],n=p['n'],k=p['k'],
            experts=1,rows=1,mode=0,input_type=1,channels=1,topk=1,
            a_row_stride=p['k'],a_token_stride=0,ids_stride=0,out_row_stride=p['n'])
        selected=dispatch.query_smallm_matched(call,arr,p['compute'])
        choice=selected.base.tc if selected is not None and selected.base.kind==0 else None
    else:
        request=Request(1,C.sizeof(Request),p['q'],p['route'],1,p['n'],p['k'],1,1,arr.mapping_id)
        choice=dispatch.query_compute(request,p['compute'],p['endpoint'],True)
    check_choice(choice,p)
    return choice


class DenseBench:
    def __init__(self, rt, bundle, sdk, cache, point):
        from tools.profile_kpack_model_decode import ROOT
        self.rt,self.dispatch=rt,None
        self.start=len(rt.allocations)
        p=point
        if p['mode']!=0 or p['tokens']!=1 or p['experts']!=1 or p['endpoint']!=1:
            raise ValueError('TC profiling requires the observed M1 dense F32 endpoint')
        try:
            self.dispatch=Dispatch(bundle,jit=dict(python=sys.executable,
                helper=ROOT/'tools/kpack_jit.py',sdk=sdk,cache=cache))
            arr=arrangement(p['q'])
            choice=select_choice(self.dispatch,p,arr)
            print(f'KPACK_MODEL_ACU_FIXTURE kind=tc q={p["q"]} n={p["n"]} k={p["k"]}',flush=True)
            w=weights(p['q'],p['n'],p['k'],1)
            values=activation_values(np.array([0])).astype('f4')
            rounded=round_compute(values,'bf16' if p['compute'] else 'f16').astype('f8')
            self.gold=rounded@w.sums[0]
            self.denom=np.abs(rounded)@w.abs_sums[0]
            a=rt.upload(np.ascontiguousarray(values[:,w.categories[0]]))
            planes={key:rt.upload(w.planes[key]) for key in ('low','high','units')}
            if p['route']==1 and p['q']!=8:
                if p['compute']:
                    raise ValueError('BF16 SF metadata must use its explicit prepass gate')
                metadata=rt.upload(w.planes['scale']);zero=rt.upload(w.planes['zero'])
            else:
                metadata,zero=planes['units'],None
            self.output_bytes=4*p['n']+32
            self.output=rt.allocate(self.output_bytes)
            self.workspace_bytes=choice.workspace_bytes+32
            self.workspace=rt.allocate(self.workspace_bytes)
            rt.fill(self.output,self.output_bytes)
            rt.fill(self.workspace,self.workspace_bytes)
            call=Call(version=1,size=C.sizeof(Call),m=1,n=p['n'],k=p['k'],experts=1,
                      group_size=arr.group_size,device=choice.device,compute_units=choice.compute_units,
                      mapping_id=arr.mapping_id,a=a,low=planes['low'],high=planes['high'],
                      metadata=metadata,zero=zero,output=self.output+16,
                      workspace=self.workspace+16 if choice.workspace_bytes else None,
                      workspace_bytes=choice.workspace_bytes,stream=rt.stream.value)
            self.launch=self.dispatch.prepare_dense_io(choice,call,1,p['compute'])
            rt.sync()
        except BaseException:
            self.close()
            raise

    def check(self):
        image=self.rt.download(self.output,self.output_bytes)
        work=self.rt.download(self.workspace,self.workspace_bytes)
        for name,data in (('output',image),('workspace',work)):
            if not (np.all(data[:16]==0xa5) and np.all(data[-16:]==0xa5)):
                raise ValueError('TC '+name+' guard changed')
        got=image[16:-16].view('<f4').reshape(self.gold.shape)
        error=output_check(got,self.gold,self.denom)
        return dict(status='PASS',oracle='INDEPENDENT_GGUF_FACTOR_DOT',error=error,
                    guards='PASS',storage='F32',adapters=0)

    def close(self):
        if self.dispatch is not None:
            self.rt.sync()
            self.dispatch.close()
            self.dispatch=None
        self.rt.release_after(self.start)
