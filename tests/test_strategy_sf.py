"""Independent metadata rounding and cold domains, without a PPU."""
from dataclasses import replace
import numpy as np
import pytest

from quactlize.dispatch.table import Workload
from dev.strategy.components import component_key,role


@pytest.mark.parametrize('q',[10,11,12,13,14])
def test_direct_bf16_oracle_matches_original_gguf_not_fp16_then_cast(q):
    pytest.importorskip('torch')
    from dev.bf16_compute.check_metadata import fixture
    from dev.strategy.sf import metadata_bits
    planes,want,_,legacy=fixture(q,n=256,k=512,experts=2)
    got=np.stack(metadata_bits(planes['units'],q,256,512,2,True))
    assert np.array_equal(got,want)
    assert np.array_equal(np.stack(metadata_bits(planes['units'],q,256,512,2)),legacy)


def test_independent_rings_do_not_force_every_gemm_to_use_sf_metadata_size():
    pytest.importorskip('torch')
    from dev.strategy.cold_decode import ring_geometry
    w=Workload(12,512,2048,1)
    size=dict(low=524288,high=0,units=65536);l2=64*1024**2
    producer=ring_geometry(w,size,l2)
    helper=ring_geometry(w,size,l2,metadata_only=True)
    assert producer['copies']==256 and helper['copies']==2304
    assert producer['active_ring_bytes']>=2.25*l2 and helper['active_ring_bytes']>=2.25*l2
    assert helper['packed_resident_bytes']==helper['active_ring_bytes']
    moe=replace(w,operation='indexed',compute='bf16',experts=256,topk=8)
    sizes={k:v*256 for k,v in size.items()}
    sf=ring_geometry(moe,sizes,l2,metadata_only=True)
    assert sf['minimum_active']==256 and sf['minimum_read_bytes']==sizes['units']
    assert sf['domain']=='ALL_EXPERT_METADATA'


def test_explicit_all_expert_sf_dedup_is_independent_of_tokens_not_active_only():
    w=Workload(12,512,2048,1,'indexed','bf16',experts=256,topk=8)
    n=dict(name='sf_bfloat_units<Format>(Params)',symbol='bf16_sf',expert_domain='ALL')
    key=lambda w,n:component_key(w,None,n,authority='device',source='code',cache='cold')[0]
    assert role(n['name'])=='expansion'
    assert key(w,n)==key(replace(w,m=8,channels=8,router='cluster'),n)
    active={k:v for k,v in n.items() if k!='expert_domain'}
    assert key(w,active)!=key(replace(w,m=8),active)
    assert key(w,n)!=key(replace(w,compute='f16'),n)


@pytest.mark.parametrize('typed',[False,True])
def test_shared_expansion_rebind_owns_only_live_consumer_stream(typed):
    pytest.importorskip('torch')
    from types import SimpleNamespace as NS
    from dev.strategy.cold_decode import Expansion
    events=[];ring=object()
    p=dict(q=12,n=512,k=2048,experts=256 if typed else 1,mode=2 if typed else 0,
           compute='bf16' if typed else 'f16')
    sdk=NS(synchronize=lambda s:events.append(('sync',s.value)))
    ctx=lambda stream:NS(p=p.copy(),ring=ring,sdk=sdk,r=NS(stream=NS(value=stream)))
    sf=Expansion.__new__(Expansion);sf.b=ctx(10);sf.typed=typed;sf.bound=True
    sf.r=NS(close=lambda:events.append(('close',0)))
    sf.family=(12,512,2048,p['experts'],typed,ring)
    calls=[NS(stream=10),NS(stream=10)]
    sf.calls=[NS(call=c) for c in calls] if typed else calls
    sf.detach()
    with pytest.raises(ValueError,match='live consumer'):sf.launch()
    sf.rebind(ctx(20));assert all(c.stream==20 for c in calls)
    wrong=ctx(30);wrong.p['k']=4096
    with pytest.raises(ValueError,match='weight/precision/ring'):sf.rebind(wrong)
    sf.detach();sf.close()
    assert events==[('sync',10),('sync',20),('close',0)]
