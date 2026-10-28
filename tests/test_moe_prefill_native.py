"""Host contracts for a device-dependent pilot; synthetic timing is not evidence."""
from types import SimpleNamespace

import numpy as np
import pytest

from dev.strategy.asys_components import Collector, parse_component_rows
from dev.strategy.moe_prefill_native import check_maps, AsysStages
from dev.strategy.prefill_fixture import routing
from quactlize.dispatch.table import Workload, Recipe


@pytest.mark.parametrize('m',[16,128,5120])
@pytest.mark.parametrize('channels',[1,8])
@pytest.mark.parametrize('router',['spread','cluster','real'])
def test_gpu_map_oracle_accepts_any_within_expert_order(m,channels,router):
    w=Workload(12,512,2048,m,'indexed','bf16',experts=256,topk=8,channels=channels,router=router)
    r=routing(w,1)
    order=r['order'].copy()
    for begin,end in zip(r['offsets'][:-1],r['offsets'][1:]):
        order[begin:end]=order[begin:end][::-1]
    check_maps(w,r,r['source'][order],order,r['offsets'])
    for which in ('source','destination','offsets'):
        src,dst,bounds=r['source'][order].copy(),order.copy(),r['offsets'].copy()
        {'source':src,'destination':dst,'offsets':bounds}[which][0]=-1
        with pytest.raises(ValueError,match='GPU routing'):
            check_maps(w,r,src,dst,bounds)


def test_asys_only_separate_stage_collection_and_exact_coverage(monkeypatch):
    clock=[0];active=[False];rows=[];events=[]
    w=Workload(12,512,2048,128,'indexed','bf16',experts=256,topk=8,channels=1,router='spread')
    recipe=Recipe.make('tc','fq',split=1)
    nodes=[dict(name=name,symbol=name,grid=[256,1,1],block=[32,1,1],shared_bytes=0) for name in
           ('mm_ids_helper<8>','gather_bf16','cutlass::device_kernel<GemmUniversal>','finish_bf16')]

    class Graph:
        count=16
        rt=SimpleNamespace(sync=lambda:None)
        def __init__(self,node):self.node=node
        def run(self):
            if active[0]:
                for _ in range(self.count):
                    rows.append(dict(self.node,kind='kernel',start=clock[0],end=clock[0]+1000))
                    clock[0]+=1100
        def sample(self):pytest.fail('event timer forbidden')
        def close(self):events.append(('close',active[0]))

    class Capture:
        groups=[[i] for i in range(4)]
        descriptions=nodes
        rt=Graph.rt
        def isolate(self,i,*,weight_ring,**kw):
            events.append(('isolate',active[0],i,weight_ring))
            return Graph(nodes[i])

    def start(self,rt):
        assert not active[0];active[0]=True;self.rt=rt
    def stop(self):active[0]=False;self.rt=None
    monkeypatch.setattr(Collector,'start',start);monkeypatch.setattr(Collector,'stop',stop)
    timer=AsysStages();proofs=[]
    def proof():
        assert not active[0];proofs.append('PASS')
    stages,pending=timer.collect(Capture(),w,recipe,'host-test','test-only',proof)
    assert pending==[] and len(stages)==len(timer.records)==4 and len(proofs)==2
    assert len(timer.collector.schedule)==4*6*5
    assert all(not e[1] for e in events)
    parsed=parse_component_rows(timer.records,timer.collector.schedule,rows,dict(host_test=True))
    assert all(r['mean_us']==1 for r in parsed.values())
    # Same helper identity can be reused in this same process, not from old timing.
    count=len(rows)
    timer.collect(Capture(),w,recipe,'host-test','test-only',proof)
    assert len(rows)==count
    with pytest.raises(ValueError,match='coverage'):
        parse_component_rows(timer.records,timer.collector.schedule,rows[:-1],{})
    Capture.descriptions=nodes[1:];Capture.groups=[[0],[1],[2]]
    with pytest.raises(ValueError,match='inventory incomplete'):
        AsysStages().collect(Capture(),w,recipe,'test','test',proof)
