"""Host-only gate inventory, output binding and trace accounting tests."""
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dev.moe_grouped.bench import Result,value,parse
from dev.moe_grouped.build import parent
from quactlize.runtime.compiler import validate_parent
from tools.run_moe_grouped import plan

ROOT=Path(__file__).resolve().parents[1]


def test_model_gate_not_a_new_sweep():
    rows=plan(False)
    assert len(rows)==76 and len({r['id'] for r in rows})==76
    assert {r['m'] for r in rows}==set(range(1,10))|{16,32,64,128,256,512,1024,2048,4096,5120}
    for r in rows:
        assert r['k'] in (2048,3072) and r['profile'] in ('spread','cluster')
        assert r['k']%(256*r['gate_split'])==0
        assert 512%(256*r['down_split'])==0
    assert len(plan(True))==24
    for q in (8,10,11,12,13,14):validate_parent(parent(q))


def test_gate_cli_plan_requires_no_sdk():
    out=subprocess.check_output([sys.executable,str(ROOT/'tools/run_moe_grouped.py'),'--plan'],text=True)
    data=json.loads(out)
    assert data['points']==76 and data['modules']==6 and data['contract_points']==28
    assert not data['production_changed'] and not data['config_search']


@pytest.mark.parametrize('compute',[0,1])
@pytest.mark.parametrize('split',[1,2,4,8])
def test_actual_result_drives_helper_state(compute,split):
    r=Result(1,C.sizeof(Result),0x1000,2 if split>1 else compute,split,compute,0,512,8192 if split>1 else 0)
    v=value(r)
    assert v.state==int(split>1) and v.order==0 and v.partial_count==split
    assert v.dtype==(0 if split>1 else compute+1) and v.round_type==compute+1


def fixture(count=16):
    capture=dict(points=[dict(job={'id':'p'},components={'control':['c'],'selected':['s']})],
        schedule=[dict(component=x,kernel='producer',repeats=16) for x in ('c','s')]+
                 [dict(component='p/chain/'+x,kernel=None,repeats=1) for x in ('control','selected')])
    rows=[];calls=[]
    for i in range(4):
        calls.append(dict(name='hggcGraphLaunch',correlation_id=i,return_value=0))
        for j in range(count if i<2 else 3):
            rows.append(dict(correlation_id=i,kind='kernel',name='producer',grid=[8,1,1],block=[256,1,1],
                             start=i*10000+j*100,end=i*10000+j*100+50))
    return capture,rows,calls


def test_component_and_actual_chain_are_distinct_metrics():
    out=parse(*fixture())
    assert out['component_us']['c']==.05
    p=out['points'][0]
    assert p['isolated_component_sum_us']=={'control':.05,'selected':.05}
    assert p['chain_activity_us']=={'control':.15,'selected':.15}
    assert p['chain_span_us']=={'control':.25,'selected':.25}
    assert not out['missing_intervals'] and not out['imputed_intervals']


def test_actual_activity_counts_not_fabricated():
    out=parse(*fixture(12))
    assert out['component_us']['c']==pytest.approx(.05) and out['missing_intervals']==8
    capture,rows,calls=fixture();rows[0]['name']='wrong'
    with pytest.raises(ValueError,match='unexpected'):parse(capture,rows,calls)
    capture,rows,calls=fixture();rows[0]['grid']=[9,1,1]
    with pytest.raises(ValueError,match='launch changed'):parse(capture,rows,calls)


@pytest.mark.parametrize('m',[1,2,3,8,9,16])
@pytest.mark.parametrize('mode',[0,1])
def test_real_chain_refresh_does_not_fill_producer_outputs(monkeypatch,m,mode):
    """Exercise actual Chain/Fixture allocations without loading a GPU runtime."""
    from dev.moe_grouped import bench
    from dev.moe_route.bench import Buffer
    from dev.moe_scaffold.protocol import Prepare,Sizes,Value

    class Memory:
        stream=C.c_void_p(1)
        def __init__(self):self.allocations={}
        def allocate(self,size):
            buf=C.create_string_buffer(size);ptr=C.addressof(buf);self.allocations[ptr]=buf;return ptr
        def free(self,ptr):del self.allocations[ptr]
        def sync(self):pass
        def MemsetAsync(self,ptr,value,size,stream):C.memset(ptr,value,size);return 0
        def write(self,ptr,data):
            assert any(start<=ptr and ptr+len(data)<=start+len(b) for start,b in self.allocations.items())
            C.memmove(ptr,data,len(data))
        def download(self,ptr,size):return C.string_at(ptr,size)

    class Query:
        def query(self,call,out):
            c=C.cast(call,C.POINTER(Prepare)).contents
            s=C.cast(out,C.POINTER(Sizes)).contents
            s.tiles=(c.route.tokens*c.route.topk+511)//512
            s.workspace_bytes=s.tiles*c.route.experts*8+c.route.tokens*c.route.topk*4
            s.algorithm=2;s.tile_rows=512;s.stages=3
            return 0
        def __getattr__(self,name):return lambda *args:0

    class Producer:
        def __init__(self,rt,spec,record,weights,m,split,a,output,offsets,shared=None):
            self.n,self.rows,self.split,self.final=weights,m*8,split,output
            self.partial=Buffer(rt,self.rows*self.n*split*4)
            self.partial.upload(b'\x5a'*self.partial.size)
        def output(self,mode):
            partial=mode==1 and self.split>1
            return Value(self.partial.ptr if partial else self.final,0 if partial else 2,0,
                         int(partial),self.split if partial else 1,self.n,self.rows*self.n if partial else 0,2,0)
        def close(self):self.partial.close()

    monkeypatch.setattr(bench,'Producer',Producer)
    rt=Memory();job=dict(m=m,k=2048,profile='spread',gate_split=4,down_split=2)
    b=bench.Chain(rt,Query(),job,{'12':({},{}),'13':({},{})},[512,512,2048])
    try:
        b.bind(mode)
        descriptors=[bytes(x) for x in (b.b.t.gate,b.b.t.up,b.b.f.down)]
        inputs=[]
        for seed in (7,19,31):
            b.update(seed)
            inputs.append(b.b.buffers['ids'].read())
            assert descriptors==[bytes(x) for x in (b.b.t.gate,b.b.t.up,b.b.f.down)]
            assert b.b.prepared_expected.shape==(m*8,2048)
            weights=np.frombuffer(b.b.buffers['weights'].read(),'<f4')
            assert weights.size==m*8 and np.all((weights>=0)&(weights<1))
            for name in ('gate','up','down'):
                buf=b.b.buffers[name]
                assert buf.read()==b'\xa5'*buf.size  # Only the producer may fill this.
            for p in b.producers:assert p.partial.read()==b'\x5a'*p.partial.size
            for buf in b.b.buffers.values():buf.guard()
        assert len(set(inputs))==3
    finally:b.close()
    assert not rt.allocations


def reuse_fixture(tmp_path,monkeypatch):
    from dev.moe_grouped import reuse as reuse_module,build as build_module
    from dev.moe_scaffold import build as helper_module
    from quactlize.runtime.compiler import sha
    root=tmp_path/'source';root.mkdir()
    for module in (reuse_module,build_module,helper_module):monkeypatch.setattr(module,'ROOT',root)
    sources={}
    for name in (*reuse_module.HARNESS_FILES,'quactlize/moe/scaffold.cu'):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('original')
        sources[name]=sha(path)
    (root/'dev/moe_grouped/reuse.py').write_text('reuse')
    previous=tmp_path/'old-build';previous.mkdir()
    library=previous/'module.so';library.write_bytes(b'original device bytes')
    helper=dict(sources=sources.copy(),libraries={},candidate=str(library),candidate_sha256=sha(library))
    receipt=dict(compiler={'kernel':'unchanged'},helper=helper,sources=sources.copy(),
                 modules={'12':[{},dict(path=str(library),sha256=sha(library))]})
    (previous/'receipt.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(reuse_module,'Compiler',lambda *args:SimpleNamespace(identity=receipt['compiler']))
    (root/'dev/moe_grouped/bench.py').write_text('fixed fixture')
    return reuse_module,previous,root,library


def test_fixture_only_retry_keeps_compiled_identity_and_original_receipt(tmp_path,monkeypatch):
    module,previous,root,library=reuse_fixture(tmp_path,monkeypatch)
    old=(previous/'receipt.json').read_bytes()
    receipt=module.reuse('/unused-sdk',previous,tmp_path/'new-build')
    assert (previous/'receipt.json').read_bytes()==old
    assert receipt['helper']['candidate']==str(library)
    assert receipt['reused_build']['compiled_artifacts_unchanged']
    assert set(receipt['reused_build']['harness_changes'])=={'dev/moe_grouped/bench.py'}
    assert json.loads((tmp_path/'new-build/reused-receipt.json').read_text())==json.loads(old)


@pytest.mark.parametrize('change',['kernel','binary','sdk'])
def test_fixture_retry_still_rejects_compiled_changes(tmp_path,monkeypatch,change):
    module,previous,root,library=reuse_fixture(tmp_path,monkeypatch)
    if change=='kernel':(root/'quactlize/moe/scaffold.cu').write_text('changed kernel')
    elif change=='binary':library.write_bytes(b'changed binary')
    else:monkeypatch.setattr(module,'Compiler',lambda *args:SimpleNamespace(identity={'kernel':'changed'}))
    with pytest.raises(ValueError):module.reuse('/unused-sdk',previous,tmp_path/'new-build')
    assert not (tmp_path/'new-build').exists()
