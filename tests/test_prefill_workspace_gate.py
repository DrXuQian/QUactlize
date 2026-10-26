"""Full preflight is host-only and must cover every scheduled large-M point."""
import ctypes as C
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from dev.prefill_point.typed import DenseLibrary, workspace_gate
from dev.prefill_asys.m_span import MS
from quactlize.dispatch.table import Workload
from quactlize.execution.native import arrangement


def point(m):
    w=Workload(8,256,512,m)
    return dict(id=w.key,workload=asdict(w),cases=[
        dict(arm='sf',recipe=dict(kind='tc')),
        dict(arm='full-fp16',recipe=dict(kind='full')),
        dict(arm='full-fp16',recipe=dict(kind='full')),
        dict(arm='full-bf16-typed',recipe=dict(kind='full')),
        dict(arm='full-bf16',recipe=dict(kind='full'))])


def test_gate_queries_unique_typed_and_legacy_geometries():
    queries=[]
    def workspace(w,arr,*,legacy):
        queries.append((w.m,legacy,arr.mapping_id))
        return 4096
    result=workspace_gate([point(m) for m in MS],SimpleNamespace(workspace=workspace))
    assert result['status']=='PASS' and result['cases']==2*len(MS)
    assert [(m,legacy) for m,legacy,_ in queries]==[(m,b) for m in MS for b in (False,True)]
    assert all(mapping==arrangement(8).mapping_id for _,_,mapping in queries)
    assert all(r['workspace_bytes']==4096 for r in result['records'])


def test_gate_does_not_relabel_old_shape_limit_as_resource_unavailable():
    def workspace(w,arr,*,legacy):
        if w.m>4096:raise RuntimeError('status=24 M='+str(w.m))
        return 4096
    with pytest.raises(RuntimeError,match='status=24 M=5120'):
        workspace_gate([point(m) for m in MS],SimpleNamespace(workspace=workspace))


def test_old_plan_defers_queries_until_cases_are_resolved():
    p=point(5120);p.pop('cases')
    def unexpected(*args,**kwargs):raise AssertionError('no declared Full case yet')
    result=workspace_gate([p],SimpleNamespace(workspace=unexpected))
    assert result['records']==[]


def test_typed_workspace_error_includes_exact_shape_before_prepare():
    lib=DenseLibrary.__new__(DenseLibrary)
    lib.query=lambda *args:24
    with pytest.raises(RuntimeError,match=r'status=24 q=8 M=5120 N=256 K=512'):
        lib.workspace(Workload(8,256,512,5120),arrangement(8))


@pytest.mark.parametrize('size',[0,1,4096])
def test_query_requires_nonzero_aligned_workspace(size):
    lib=DenseLibrary.__new__(DenseLibrary)
    def query(shape,arr,out):
        C.cast(out,C.POINTER(C.c_uint64)).contents.value=size
        return 0
    lib.query=query
    if size==4096:
        assert lib.workspace(Workload(8,256,512,8192),arrangement(8))==4096
    else:
        with pytest.raises(ValueError,match='workspace extent'):
            lib.workspace(Workload(8,256,512,8192),arrangement(8))
