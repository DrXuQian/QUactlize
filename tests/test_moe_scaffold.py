import ctypes as C
from pathlib import Path
import subprocess

import numpy as np
import pytest

from dev.moe_scaffold.protocol import (Value,Prepare,Transition,Finish,Plan,Case,workloads,plan_calls,
    encode,decode,rounded,projected,activation,weighted,ids_for,oracle,parse)

ROOT=Path(__file__).resolve().parents[1]


def test_host_contract(tmp_path):
    target=tmp_path/'contract'
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-fsanitize=undefined',
                    '-I'+str(ROOT),str(ROOT/'tests/moe_scaffold_host.cpp'),
                    str(ROOT/'quactlize/moe/selection.cpp'),'-o',str(target)],check=True)
    assert 'MOE_SCAFFOLD_HOST PASS' in subprocess.check_output([target],text=True)
    assert [C.sizeof(t) for t in (Value,Prepare,Transition,Finish)]==[48,192,168,120]
    assert C.sizeof(Plan)==24
    header='#include "quactlize/moe/api.h"\nint main(void) { return 0; }\n'
    subprocess.run(['gcc','-std=c11','-Wall','-Wextra','-Werror','-I'+str(ROOT),'-x','c','-fsyntax-only','-'],
                   input=header,text=True,check=True)


@pytest.mark.parametrize('m',[0,1,4,8,9,17,65,128,257,5120])
@pytest.mark.parametrize('topk',[1,6,8,32])
def test_forward_inverse_and_reused_group_order(m,topk):
    case=Case(m,experts=max(33,topk),topk=topk)
    ids=ids_for(case.route_case,7)
    src,dst,offsets=oracle(ids,case.route_case)
    inverse=np.empty(m*topk,dtype=np.int32);inverse[dst]=np.arange(m*topk,dtype=np.int32)
    data=np.arange(m*topk*3).reshape(m*topk,3)
    np.testing.assert_array_equal(data[dst][inverse],data)
    np.testing.assert_array_equal(np.diff(offsets),np.bincount(ids[:,:topk].ravel(),minlength=case.experts))
    # GateUp input has one row/token; Down already consumes grouped rows.
    np.testing.assert_array_equal(src,(dst//topk)*4)


@pytest.mark.parametrize('splits',[1,2,4,8])
@pytest.mark.parametrize('dtype',[1,2])
def test_round_after_reduce_before_activation(splits,dtype):
    rng=np.random.default_rng(31)
    g=rng.normal(size=(splits,17,9)).astype(np.float32)
    u=rng.normal(size=(splits,17,9)).astype(np.float32)
    right=rounded(activation(projected(g,dtype),projected(u,dtype)),dtype)
    if splits>1:
        wrong=projected(np.stack([activation(rounded(a,dtype),rounded(b,dtype)) for a,b in zip(g,u)]),dtype)
        assert not np.array_equal(right,wrong)
    # An already-activated recipe's output is consumed, never activated twice.
    assert not np.array_equal(right,rounded(activation(right,np.ones_like(right)),dtype))


def test_bf16_large_finite_and_weighted_slot_order():
    v=np.array([243383.484,100000,-100000],np.float32)
    assert np.isfinite(rounded(v,2)).all()
    x=np.array([[[1e20],[1],[-1e20],[3]]],np.float32)
    w=np.ones((1,4),np.float32)
    assert weighted(x,w)[0,0]==3
    assert weighted(x[:,[0,2,1,3]],w)[0,0]==4
    bits=np.array([0x3f808000,0x3f818000],np.uint32).view(np.float32)
    np.testing.assert_array_equal(encode(bits,2),[0x3f80,0x3f82])
    np.testing.assert_array_equal(decode(encode(bits,2),2),rounded(bits,2))


def test_plan_has_boundary_and_recipe_coverage():
    cases=workloads()
    assert len({c.key for c in cases})==len(cases)
    assert set(range(10))<={c.tokens for c in cases}
    assert {16,32,64,128,512,1024,2048,4096,5120,8193}<={c.tokens for c in cases}
    assert {0,1,2}=={c.state for c in cases}
    assert {0,1,2}=={c.dtype for c in cases}
    assert {1,2,4,8}<={c.down_splits for c in cases}
    source=(ROOT/'quactlize/moe/scaffold.cu').read_text()
    assert 'atomicAdd' not in source and 'GraphKernelNodeGetParams' not in source
    assert 'm>8' not in source and 'tokens>8' not in source
    builder=(ROOT/'tools/build_ppu_library.py').read_text()
    assert "'quactlize/moe/scaffold.cu'" in builder and "'quactlize/moe/api.h'" in builder


def trace_fixture():
    schedule=[];rows=[];apis=[]
    for cid,(component,ns) in enumerate([('fused',1000),('unfused0',700),('unfused1',700)],1):
        schedule.append(dict(component=component,kernel=component,grid=[1,1,1],block=[256,1,1],repeats=32))
        apis.append(dict(name='hggcGraphLaunch',return_value=0,correlation_id=cid))
        rows.extend(dict(kind='kernel',name=component,grid=[1,1,1],block=[256,1,1],
                         start=i*ns,end=(i+1)*ns,correlation_id=cid) for i in range(32))
    capture=dict(schedule=schedule,points=[dict(key='case/finish',case={},components=
                 dict(fused=['fused'],unfused=['unfused0','unfused1']))],device={'name':'host-only'})
    return capture,rows,apis


def test_profile_means_actual_intervals_no_native_claim():
    c,r,a=trace_fixture();r.pop(0)
    data=parse(c,r,a)
    assert data['missing_intervals']==1 and data['imputed_intervals']==0
    assert data['points'][0]['arms_us']==dict(fused=1,unfused=1.4)
    assert data['baseline']=='SAME_MATH_UNFUSED_HELPERS_NOT_LLAMA_GEMM'
    assert not data['model_tpot_admitted'] and not data['no_regression_admitted']


def test_candidate_costs_compare_both_measured_controls():
    c,r,a=trace_fixture()
    c['schedule'].append(dict(component='new',kernel='new',grid=[1,1,1],block=[128,1,1],repeats=32))
    a.append(dict(name='hggcGraphLaunch',return_value=0,correlation_id=4))
    r.extend(dict(kind='kernel',name='new',grid=[1,1,1],block=[128,1,1],
                  start=i*800,end=(i+1)*800,correlation_id=4) for i in range(32))
    c['points'][0]['components']['slots4']=['new']
    p=parse(c,r,a)['points'][0]
    assert p['arms_us']['slots4']==pytest.approx(.8)
    assert p['versus_fused_pct']['slots4']==pytest.approx(-20)
    assert p['versus_unfused_pct']['slots4']==pytest.approx(100*(.8/1.4-1))


@pytest.mark.parametrize('warps',[4,8])
@pytest.mark.parametrize('topk',[1,6,8,17,32])
def test_slot_parallel_ownership_preserves_order(warps,topk):
    # Every shared cell has exactly one producer. Ordered slots must not be
    # replaced with a warp/tree sum (the cancellation example is order-sensitive).
    values=np.zeros((2,topk,37),np.float32)
    values[:,0,:]=1e20
    if topk>=4:
        values[:,1,:]=1;values[:,2,:]=-1e20;values[:,3,:]=3
    expected=weighted(values,np.ones((2,topk),np.float32))
    got=np.empty_like(expected)
    for token in range(2):
        for tile in range(2):
            scratch=np.full((topk,32),np.nan,np.float32)
            writes=np.zeros((topk,32),np.int32)
            for warp in range(warps):
                for slot in range(warp,topk,warps):
                    for lane in range(32):
                        col=tile*32+lane
                        scratch[slot,lane]=values[token,slot,col] if col<37 else 0
                        writes[slot,lane]+=1
            assert np.all(writes==1)
            for lane in range(32):
                if tile*32+lane>=37:continue
                value=scratch[0,lane]
                for slot in range(1,topk):value=np.float32(value+scratch[slot,lane])
                got[token,tile*32+lane]=value
    np.testing.assert_array_equal(got,expected)


def test_clock_uses_physical_identity_and_throttling_is_advisory(monkeypatch):
    from dev.moe_scaffold import bench
    class Runtime:
        lib=object()
    def fake_bind(lib,name,args):
        assert name=='hggcDeviceGetPCIBusId'
        def query(ptr,length,ordinal):
            assert ordinal==0 and length==32
            C.memmove(ptr,b'0000:08:00.0\0',13)
            return 0
        return query
    calls=[]
    monkeypatch.setattr(bench,'bind',fake_bind)
    monkeypatch.setattr(bench,'request_lock',lambda binary,pci,mhz:
        calls.append((str(binary),pci,mhz)) or dict(returncode=0))
    def observed(binary,pci,mhz,*,advisory):
        assert advisory
        return dict(cu_mhz=1500,affects_verdict=False)
    monkeypatch.setattr(bench,'observe',observed)
    device=dict(ordinal=0)
    receipt=bench.start_clock(Runtime(),device,'ppu-smi',1700)
    assert calls==[('ppu-smi','0000:08:00.0',1700)]
    assert device['pci']=='0000:08:00.0'
    assert receipt['before']['cu_mhz']==1500 and not receipt['before']['affects_verdict']


@pytest.mark.parametrize('fault',['empty','grid','symbol','rc','few'])
def test_profile_diagnostics(fault):
    c,r,a=trace_fixture()
    if fault=='empty':r=[]
    if fault=='grid':r[0]['grid']=[2,1,1]
    if fault=='symbol':r[0]['name']='wrong'
    if fault=='rc':a[0]['return_value']=1
    if fault=='few':r=r[20:]
    with pytest.raises(ValueError):parse(c,r,a)


@pytest.mark.parametrize('state',[0,1,2])
@pytest.mark.parametrize('dtype',[0,1,2])
@pytest.mark.parametrize('tokens',[0,9])
@pytest.mark.parametrize('padding,projection_dtype',[(3,-1),(0,-1),(0,0)])
def test_fixture_storage_and_recipe_wiring(state,dtype,tokens,padding,projection_dtype):
    # Exercise the real Box fixture's allocation/strides and encoded source
    # bytes on host. This does not simulate a GPU launch or admit performance.
    from dev.moe_scaffold.bench import Fixture
    from dev.moe_scaffold.protocol import Sizes
    class Memory:
        stream=C.c_void_p(1)
        def __init__(self):self.allocations={}
        def allocate(self,size):
            buf=C.create_string_buffer(size);ptr=C.addressof(buf);self.allocations[ptr]=buf;return ptr
        def free(self,ptr):del self.allocations[ptr]
        def sync(self):pass
        def MemsetAsync(self,ptr,value,size,stream):C.memset(ptr,value,size);return 0
        def write(self,ptr,data):C.memmove(ptr,data,len(data))
        def download(self,ptr,size):return C.string_at(ptr,size)
    class Query:
        def query(self,call,out):
            c=C.cast(call,C.POINTER(Prepare)).contents
            n=c.route.tokens*c.route.topk;tiles=(n+511)//512
            prefix=((tiles*c.route.experts*4+15)//16)*16
            size=prefix*2+((n*4+15)//16)*16 if n else 0
            s=C.cast(out,C.POINTER(Sizes)).contents
            s.workspace_bytes=size;s.algorithm=2;s.tile_rows=512;s.tiles=tiles;s.stages=3 if n else 1
            return 0
        def __getattr__(self,name):return lambda *args:0
    rt=Memory();c=Case(tokens,columns=7,hidden=5,output=9,dtype=dtype,state=state,
        splits=4 if state==1 else 1,down_splits=4,merged=0,gate_order=1,up_order=0,output_order=1,down_order=0,
        padding=padding,projection_dtype=projection_dtype)
    b=Fixture(rt,Query(),c)
    try:
        b.update(31)
        def read_value(name,v,nc):
            raw=np.frombuffer(b.buffers[name].read(),np.float32 if v.dtype==0 else np.float16 if v.dtype==1 else np.uint16)
            raw=decode(raw,v.dtype)
            planes=np.stack([raw[i*v.partial_stride:i*v.partial_stride+b.rows*v.row_stride]
                .reshape(b.rows,v.row_stride)[:,:nc] for i in range(v.partial_count)])
            values=planes[0] if v.state==2 else projected(planes,v.round_type)
            return values[b.maps[3]] if v.order==0 else values
        g=read_value('gate',b.t.gate,b.gcols);u=read_value('up',b.t.up,c.hidden)
        a=g if state==2 else activation(g,u)
        np.testing.assert_array_equal(rounded(a,dtype),b.activation_expected)
        down=read_value('down',b.f.down,c.output).reshape(tokens,c.topk,c.output)
        w=np.frombuffer(b.buffers['weights'].read(),np.float32).reshape(tokens,c.topk+padding)
        np.testing.assert_array_equal(weighted(down,w[:,:c.topk]),b.finish_expected)
        assert b.p.route.source_stride==1+padding and b.p.route.ids_stride==c.topk+padding
        assert b.t.output_stride==c.hidden+padding and b.f.output_stride==c.output+padding
        p,f=plan_calls(c)
        assert (p.columns,p.dtype,p.input_row_stride,p.route.tokens,p.route.ids_stride,p.route.source_stride)==(
            b.p.columns,b.p.dtype,b.p.input_row_stride,b.p.route.tokens,b.p.route.ids_stride,b.p.route.source_stride)
        assert (f.tokens,f.topk,f.columns,f.down.dtype,f.down.order,f.down.state,f.down.partial_count,
                f.down.row_stride,f.down.partial_stride,f.down.round_type,f.weights_stride,f.output_stride)==(
            b.f.tokens,b.f.topk,b.f.columns,b.f.down.dtype,b.f.down.order,b.f.down.state,b.f.down.partial_count,
            b.f.down.row_stride,b.f.down.partial_stride,b.f.down.round_type,b.f.weights_stride,b.f.output_stride)
        groups=b.component_groups()
        if tokens:
            assert len(groups['prepare/fused'])==3 and len(groups['prepare/unfused'])==5
            assert len(groups['prepare/hybrid'])==4
            assert [item[1] for item in groups['prepare/hybrid']]==[
                'histogram_rank<','prepare_prefix<','scatter_maps<','prepare_gather<']
            assert len(groups['transition/fused'])==1 and len(groups['finish/fused'])==1
            for warps in (4,8):
                component=groups[f'finish/slots{warps}'][0]
                assert component[1]==f'finish_slots<{warps}>'
                assert component[2]==[tokens,1,1] and component[3]==[warps*32,1,1]
        else:assert not groups
    finally:b.close()
    assert not rt.allocations


def test_qwen_model_phase_inventory_and_deduplication():
    from dev.moe_scaffold.models import model_plan,MS,ROUTERS,MODELS,assignments,estimated_work
    from collections import Counter
    cases,coverage=model_plan()
    assert len(cases)==1284 and len({c.key for c in cases})==1284
    assert coverage['phases']==dict(finish=896,prepare=172,transition=216)
    assert len(model_plan(True)[0])==12
    assert all(c.padding==0 and c.dtype==2 for c in cases)
    for name,_,k in MODELS:
        for m in MS:
            for router in (('spread',) if m==1 else ROUTERS):
                prep=[c for c in cases if c.phase=='prepare' and c.columns==k and c.tokens==m and c.profile==router]
                assert len(prep)==1 and coverage['consumers'][prep[0].key]==[name]
                finish=[c for c in cases if c.phase=='finish' and c.output==k and c.tokens==m and c.profile==router and c.down_order==0]
                assert {(c.down_splits,c.projection_dtype) for c in finish}=={(1,-1),(1,0),(2,-1),(4,-1),(8,-1)}
            direct=[c for c in cases if c.phase=='finish' and c.output==k and c.tokens==m and c.down_order==1]
            assert len(direct)==1 and direct[0].projection_dtype==0 and direct[0].down_splits==1
    for m in MS:
        transition=[c for c in cases if c.phase=='transition' and c.tokens==m]
        assert len(transition)==12
        assert all(c.hidden==512 and c.gate_order==c.up_order==c.output_order==0 for c in transition)
        assert all(len(coverage['consumers'][c.key])==2 for c in transition)
    assigned=assignments(cases,4)
    flat=[Case(**c) for shard in assigned for c in shard]
    assert Counter(c.key for c in flat)==Counter(c.key for c in cases)
    loads=[sum(estimated_work(Case(**c)) for c in shard) for shard in assigned]
    assert max(loads)-min(loads)<=max(map(estimated_work,cases))
    assert Case(2048,columns=2048,hidden=512,output=2048).key=='ddaf4c78b0815ff0b07f'
    assert Case(2048,columns=2048,hidden=512,output=2048,padding=0).key!='ddaf4c78b0815ff0b07f'


def test_model_cli_plan_without_gpu_or_sdk():
    import json,sys
    output=subprocess.check_output([sys.executable,str(ROOT/'tools/run_moe_scaffold.py'),
        '--cohort','qwen-models','--smoke','--plan'],text=True)
    plan=json.loads(output)
    assert plan['count']==12 and plan['coverage']['full_cases']==1284
    assert not plan['coverage']['actual_model_weights_used'] and not plan['production_changed']


def test_batched_worker_retains_pass_and_retries_failed_batch(tmp_path,monkeypatch):
    from argparse import Namespace
    from dataclasses import asdict
    from tools import run_moe_scaffold as runner
    from dev.prefill_asys import trace
    from dev.strategy.journal import atomic_json
    from quactlize.runtime.compiler import sha
    (tmp_path/'build').mkdir()
    atomic_json(tmp_path/'build/receipt.json',{})
    cases=[asdict(Case(m,phase='finish')) for m in (1,2)]
    atomic_json(tmp_path/'run.json',dict(devices=['3'],assignments=[cases],batch_size=1))
    monkeypatch.setattr(runner,'verify',lambda receipt:None)
    monkeypatch.setattr(trace,'activities',lambda *args,**kwargs:[])
    monkeypatch.setattr(trace,'runtime_activities',lambda *args,**kwargs:[])
    monkeypatch.setattr(runner,'parse',lambda capture,*args:dict(status='PASS',points=[]))
    visited=[];fail_once=[True]
    def execute(command,log):
        if command[1]!='profile':return
        attempt=Path(command[command.index('--batch')+1]);job=runner.read(attempt/'job.json')
        visited.append(job['batch'])
        if job['batch']==1 and fail_once[0]:
            fail_once[0]=False;raise RuntimeError('intentional first-batch failure')
        (attempt/'trace.asysrep').write_bytes(b'host-test-not-a-real-trace')
        atomic_json(attempt/'capture.json',dict(cases=job['cases'],clock={},
            latency_provenance=dict(build_sha256=sha(tmp_path/'build/receipt.json'))))
    monkeypatch.setattr(runner,'run',execute)
    args=Namespace(output=tmp_path,worker=0,sdk=Path('/unused-sdk'),asys=Path('/unused-asys'),resume=False)
    assert runner.worker(args)==1 and visited==[1,2]
    first=tmp_path/'gpu-3/batch-0/result.json';second=tmp_path/'gpu-3/batch-1/result.json'
    assert runner.read(first)['status']=='FAIL' and runner.read(second)['status']=='PASS'
    before=second.read_bytes();args.resume=True
    assert runner.worker(args)==0 and visited==[1,2,1]
    assert second.read_bytes()==before
    with pytest.raises(ValueError,match='does not match'):
        runner.reusable(runner.read(second),cases[:1],tmp_path)
