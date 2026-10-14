"""Host carrier/ABI regression only; mock pointers are never dereferenced."""
import ctypes as C
from types import SimpleNamespace

import pytest

from dev.strategy.indexed import TensorCore,config
from quactlize.dispatch.table import Recipe,Workload
from quactlize.decode.native_compute import GroupedMetadataCall
from quactlize.runtime.native import Resources as Query
from test_strategy_prescreen import tc


def test_sf_ordinary_grid_alias_is_not_persistent():
    w=Workload(12,1024,2048,1,'indexed','bf16',experts=256,topk=8)
    r=Recipe.make('tc','sf',**(dict(tc().parameters)|dict(persistent=-1,grid_mode=1,grid_b=0)))
    assert config(r,w)['algorithm']==0


def test_current_v4_prepare_and_ring_binding(monkeypatch):
    pytest.importorskip('torch')  # existing independent fixture dependency
    import dev.smallm_closure.bench as fixture
    import tools.run_kpack_gemv_gate as gate
    import dev.strategy.resources as resources
    w=Workload(12,1024,2048,8,'indexed','bf16',experts=256,topk=8)
    r=Recipe.make('tc','fq',**(dict(tc().parameters)|dict(persistent=0,grid_mode=0,grid_b=0)))
    calls=[];expected=[0x30000,0x40000]
    class Buffer:
        address=0x10000
        def __init__(self,owner,size):
            self.ptr=Buffer.address;Buffer.address+=max(256,size);self.size=size
        def upload(self,value):pass
    class Owner:
        def __init__(self,sdk):self.stream=C.c_void_p(17)
        def close(self):pass
    class Probe:
        device=dict(ordinal=0,compute_units=64)
        lib=object()
        def __init__(self,spec,record):pass
        def query(self,spec):return dict(status='READY',occupancy=3,workspace_bytes=4096,shared_bytes=8192)
        def _bind(self,name,args,result=C.c_int):
            def invoke(*values):
                calls.append(name)
                if name in ('grouped_query_v4','grouped_prepare_v4'):
                    req=C.cast(values[0],C.POINTER(GroupedMetadataCall)).contents
                    assert req.version==4 and req.size==C.sizeof(GroupedMetadataCall)
                    assert req.compute_type==1 and req.metadata_type==1
                    call=req.device_call.call
                    assert call.compute_units==64 and call.m==64 and req.device_call.max_rows==8
                    assert call.a and call.output and call.offsets_device and call.workspace and call.stream==17
                    if name=='grouped_query_v4':
                        out=C.cast(values[2],C.POINTER(Query)).contents
                        out.occupancy=3;out.workspace_bytes=4096;out.shared_bytes=8192
                    else:
                        assert call.low==expected.pop(0) and call.metadata and not call.zero
                        C.cast(values[2],C.POINTER(C.c_void_p))[0]=C.c_void_p(42)
                return 0
            return invoke
    monkeypatch.setattr(fixture,'Buffer',Buffer)
    monkeypatch.setattr(gate,'Resources',Owner)
    monkeypatch.setattr(resources,'NativeProbe',Probe)
    b=SimpleNamespace(sdk=SimpleNamespace(synchronize=lambda stream:None),r=Owner(None),rows=64,
        ring=SimpleNamespace(planes=[dict(low=p,high=None,units=p+128) for p in expected]),
        a=Buffer(None,4),ids=Buffer(None,4),out=Buffer(None,4))
    record=dict(parent=resources.module_spec(w,r)['parent'],key='HOST_MOCK',sha256='HOST_MOCK')
    kernel=TensorCore(b,record,r,w)
    assert not expected
    assert calls.count('grouped_prepare_v4')==2 and calls.count('bind_llama_indexed_v1')==2
    assert kernel.receipt['compute_units']==64 and kernel.receipt['metadata_type']==1
    assert kernel.launch(1)==0
    kernel.close()
    assert calls.count('destroy_v1')==2
