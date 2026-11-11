"""Same contract on separate PPU/NVIDIA binaries, never a fake device fallback."""
import ctypes as C
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from quactlize.serving import Serving, Matmul
from quactlize.compat.native import problem
from quactlize.compat.serving_native import Rows, Router
from ci.support.serving import row_blocks, dequant_rows, router_reference
from ci.support.compat import raw_blocks, weight, assert_dot, assert_merge


@pytest.fixture(scope='module')
def owner():
    path=os.environ.get('QUACTLIZE_SERVING_LIBRARY')
    if not path:
        from quactlize import get_library_path
        path=get_library_path()
    if not torch.cuda.is_available():pytest.fail('this gate requires device-enabled platform Torch and a real device')
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        obj=Serving(path,torch.cuda.current_device(),stream)
        print('SERVING_DEVICE',torch.cuda.get_device_name(),str(Path(path).resolve()),flush=True)
        yield obj
        stream.synchronize();obj.close()


def dev(a,dtype=None):return torch.tensor(a,device='cuda',dtype=dtype)
def cpu(t):return t.detach().float().cpu().numpy()


def guarded(shape,dtype):
    n=int(np.prod(shape));store=torch.full((n+64,),37,device='cuda',dtype=dtype)
    view=store[32:32+n].view(shape)
    return view,lambda: (torch.testing.assert_close(store[:32],torch.full_like(store[:32],37),rtol=0,atol=0),
                        torch.testing.assert_close(store[-32:],torch.full_like(store[-32:],37),rtol=0,atol=0))


@pytest.mark.parametrize('m',[1,2,4,8,16,32])
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32])
def test_router_flare_contract_and_replay(owner,m,dtype):
    logits=torch.empty((m,521),device='cuda',dtype=dtype)[:,:512]
    selected=torch.empty((m,13),device='cuda')[:,:10]
    ids,check_i=guarded((m,10),torch.int32);weights,check_w=guarded((m,10),torch.float32)
    error,check_e=guarded((1,),torch.int32);count=torch.zeros(1,device='cuda',dtype=torch.int32)
    logits.zero_();error.zero_();owner.router(logits,selected,ids,weights,error,count)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=torch.cuda.current_stream()):
        # Status is deliberately NOT cleared here; caller owns chain reset.
        owner.router(logits,selected,ids,weights,error,count,multiplier=.75)
    rng=np.random.default_rng(m)
    for live in (0,m//2,m,-1,m+1):
        x=rng.normal(size=(m,512)).astype('f4')
        x[0]=0 # complete tie
        if m>1:
            x[1,:]=-10000;x[1,:3]=[0,-90,-91] # select before underflow/rounding
        if m>2:x[2,:16]=1.0002 # near BF16 rounding boundary/partial tie
        x[max(live,0):]=np.nan
        logits.copy_(dev(x,dtype));count.fill_(live);error.fill_(16)
        graph.replay();torch.cuda.current_stream().synchronize()
        gold=router_reference(cpu(logits),10,live,.75)
        np.testing.assert_array_equal(cpu(selected),gold[0]);np.testing.assert_array_equal(ids.cpu().numpy(),gold[1])
        np.testing.assert_allclose(cpu(weights),gold[2],rtol=3e-6,atol=2e-7)
        assert error.item()==(gold[3]|16)
    for bad in (float('nan'),float('inf'),-float('inf')):
        logits.fill_(bad);count.fill_(m);error.zero_();graph.replay()
        assert error.item()==2 and torch.all(ids==-1) and torch.all(weights==0)
    count.zero_();error.zero_()
    p=Router(2,C.sizeof(Router),m,512,10,1 if dtype==torch.bfloat16 else 2,0,1,1.,0,
        512,selected.stride(0),10,10,m*512*logits.element_size(),m*13*4,m*10*4,m*10*4,
        0x100000,selected.data_ptr(),ids.data_ptr(),weights.data_ptr(),count.data_ptr(),error.data_ptr(),
        torch.cuda.current_stream().cuda_stream)
    owner.lib.check(owner.lib.router(C.byref(p)));torch.cuda.current_stream().synchronize()
    assert error.item()==0 and torch.all(ids==-1) and torch.all(selected==0)
    check_i();check_w();check_e()


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('merged',[False,True])
@pytest.mark.parametrize('round_silu',[False,True])
def test_swiglu_rounding_and_strides(owner,dtype,merged,round_silu):
    rows,h=4,640
    g=torch.linspace(-4,4,rows*h,device='cuda').reshape(rows,h).to(dtype)
    u=torch.full_like(g,1.5703125)
    storage=torch.zeros((rows,h*(2 if merged else 1)+7),device='cuda',dtype=dtype)
    gate=storage[:,:h*(2 if merged else 1)];gate[:,:h].copy_(g)
    if merged:gate[:,h:].copy_(u);up=None
    else:up=torch.empty((rows,h+3),device='cuda',dtype=dtype)[:,:h];up.copy_(u)
    output=torch.full((rows,h+5),37,device='cuda',dtype=dtype);out=output[:,:h]
    owner.swiglu(gate,out,up,round_silu=round_silu)
    a=g.float()*torch.sigmoid(g.float())
    expected=((a.to(dtype).float() if round_silu else a)*u.float()).to(dtype)
    # expf implementations can differ at a target rounding midpoint. At most
    # one target ULP; the OTHER mode must still differ on many witness values.
    ulp=torch.nextafter(expected,torch.full_like(expected,float('inf'))).float()-expected.float()
    assert torch.all((out.float()-expected.float()).abs()<=ulp.abs())
    other=((a if round_silu else a.to(dtype).float())*u.float()).to(dtype)
    assert torch.count_nonzero(out!=other)>100
    assert torch.all(output[:,h:]==37)


@pytest.mark.parametrize('dtype,tiny',[(torch.float16,2**-24),(torch.bfloat16,2**-133)])
def test_swiglu_intermediate_subnormal_rounding(owner,dtype,tiny):
    gate=dev([[tiny,3*tiny,-tiny,-3*tiny]],dtype);up=torch.full_like(gate,2)
    out=torch.empty_like(gate)
    for rounded in (True,False):
        owner.swiglu(gate,out,up,round_silu=rounded)
        a=gate.float()*torch.sigmoid(gate.float())
        expected=((a.to(dtype).float() if rounded else a)*up.float()).to(dtype)
        torch.testing.assert_close(out,expected,rtol=0,atol=0)


@pytest.mark.parametrize('q,columns',[(8,2560),(20,160),(42,640),(42,2560)])
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_rows_raw_format_counts_errors_and_canaries(owner,q,columns,dtype):
    raw=row_blocks(q,7,columns);table=dev(raw,torch.uint8);ids=dev([0,6,6,-1,-2,7,2],torch.int64)
    out,check_o=guarded((7,columns),dtype);error,check_e=guarded((1,),torch.int32)
    count=dev([0],torch.int32)
    owner.gather_rows(table,ids,out,error,qtype=q,table_rows=7,count=count)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=torch.cuda.current_stream()):
        owner.gather_rows(table,ids,out,error,qtype=q,table_rows=7,count=count)
    decoded=dev(dequant_rows(raw,q),dtype)
    for live in (0,3,4,7,-1,8):
        count.fill_(live);error.fill_(16);graph.replay()
        expected=torch.zeros_like(out);bad=0
        if 0<=live<=7:
            for r,index in enumerate([0,6,6,-1,-2,7,2][:live]):
                if 0<=index<7:expected[r]=decoded[index]
                elif index!=-1:bad=1
        else:bad=4
        torch.testing.assert_close(out,expected,rtol=0,atol=0)
        assert error.item()==(16|bad)
        check_o();check_e()
    # Strict inactive-input probe: live=0 with deliberately inaccessible IDs
    # and raw pointers. Host validates extents only; kernel must never read them.
    count.zero_();error.zero_()
    p=Rows(1,C.sizeof(Rows),q,1 if dtype==torch.bfloat16 else 0,7,columns,7,
        raw.nbytes,56,out.numel()*2,0x100000,0x200000,out.data_ptr(),count.data_ptr(),error.data_ptr(),
        torch.cuda.current_stream().cuda_stream)
    owner.lib.check(owner.lib.rows(C.byref(p)));torch.cuda.current_stream().synchronize()
    assert torch.all(out==0) and error.item()==0


def test_zero_capacity_validates_device_count(owner):
    raw=dev(row_blocks(20,1,160),torch.uint8);ids=torch.empty(0,device='cuda',dtype=torch.int64)
    output=torch.empty((0,160),device='cuda',dtype=torch.bfloat16)
    status=dev([16],torch.int32);count=dev([0],torch.int32)
    owner.gather_rows(raw,ids,output,status,qtype=20,table_rows=1,count=count)
    assert status.item()==16
    count.fill_(1)
    owner.gather_rows(raw,ids,output,status,qtype=20,table_rows=1,count=count)
    assert status.item()==20


@pytest.mark.parametrize('m,e,t',[(1,512,10),(2,512,10),(4,512,10),(8,512,10),(16,4,2),(32,4,2)])
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_whole_chain_dynamic_graph_and_shared_scratch(owner,m,e,t,dtype):
    # Real 512/top10 route, with compact synthetic GEMMs. Actual 640/2560
    # projection geometry is covered separately by test_compat_library.
    k,h=64,64;dt=int(dtype==torch.bfloat16)
    raw_up=raw_blocks(8,e,2*h,k);raw_down=raw_blocks(7,e,k,h)
    wup=dev(raw_up,torch.uint8);wdown=dev(raw_down,torch.uint8)
    embed_np=row_blocks(8,7,k);embedding=dev(embed_np,torch.uint8);lookup_ids=dev(np.arange(m)%7,torch.int64)
    ple=dev(row_blocks(20,7,160),torch.uint8);ple_out=dev(np.zeros((m,160)),dtype)
    x=dev(np.zeros((m,1,k)),dtype);gu=dev(np.zeros((m,t,2*h)),dtype)
    mid=dev(np.zeros((m,t,h)),dtype);down=dev(np.zeros((m,t,k)),dtype)
    out,canary=guarded((m,k),torch.float32)
    logits=dev(np.zeros((m,e)),torch.bfloat16);sel=dev(np.zeros((m,t)),torch.float32)
    ids=dev(np.zeros((m,t)),torch.int32);weights=dev(np.zeros((m,t)),torch.float32)
    live=dev([m],torch.int32);error=dev([0],torch.int32)
    pu=problem(8,m,2*h,k,experts=e,topk=t,dtype=dt,device=owner.device)
    pd=problem(7,m,k,h,experts=e,topk=t,channels=t,dtype=dt,device=owner.device)
    su=Matmul.query(owner,pu);sd=Matmul.query(owner,pd)
    workspace=torch.empty(max(su.workspace_bytes,sd.workspace_bytes),device='cuda',dtype=torch.uint8)
    up=Matmul(owner,pu,input=x,output=gu,workspace=workspace,error=error,ids=ids,raw=wup)
    dn=Matmul(owner,pd,input=mid,output=down,workspace=workspace,error=error,ids=ids,raw=wdown)
    def chain():
        owner.gather_rows(embedding,lookup_ids,x.view(m,k),error,qtype=8,table_rows=7,count=live)
        owner.gather_rows(ple,lookup_ids,ple_out,error,qtype=20,table_rows=7,count=live)
        owner.router(logits,sel,ids,weights,error,live)
        up(x,gu,workspace,error,ids=ids,raw=wup)
        owner.swiglu(gu.view(m*t,2*h),mid.view(m*t,h),round_silu=True)
        dn(mid,down,workspace,error,ids=ids,raw=wdown)
        owner.merge(down,weights,out)
    try:
        chain();torch.cuda.current_stream().synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=torch.cuda.current_stream()):chain()
        for iteration,valid in enumerate((m,m//2,0,m)):
            host=np.random.default_rng(iteration).normal(size=(m,e)).astype('f4')
            host[valid:]=np.nan
            logits.copy_(dev(host,torch.bfloat16));live.fill_(valid);error.fill_(16)
            lookup_ids.copy_(dev((np.arange(m)+iteration)%7,torch.int64))
            graph.replay();torch.cuda.current_stream().synchronize()
            gold=router_reference(cpu(logits),t,valid)
            np.testing.assert_array_equal(ids.cpu().numpy(),gold[1]);np.testing.assert_allclose(cpu(weights),gold[2],rtol=3e-6,atol=2e-7)
            assert error.item()==16
            # Check BOTH real matmuls against independent GGUF decode, then
            # rounding/merge using the actual preceding projection boundary.
            ids_np=ids.cpu().numpy();a=cpu(x)[:,0];g=cpu(gu);d=cpu(down);activation=cpu(mid)
            for expert in np.unique(ids_np):
                r,s=np.where(ids_np==expert)
                if expert==-1:
                    assert not g[r,s].any() and not d[r,s].any();continue
                assert_dot(g[r,s],a[r],weight(raw_up[expert:expert+1],8,dt)[0],dt)
                assert_dot(d[r,s],activation[r,s],weight(raw_down[expert:expert+1],7,dt)[0],dt)
            activated=(gu[...,:h].float()*torch.sigmoid(gu[...,:h].float())).to(dtype)
            expected_mid=(activated.float()*gu[...,h:].float()).to(dtype)
            ulp=torch.nextafter(expected_mid,torch.full_like(expected_mid,float('inf'))).float()-expected_mid.float()
            assert torch.all((mid.float()-expected_mid.float()).abs()<=ulp.abs())
            assert_merge(cpu(out),d,cpu(weights),context=f'serving dtype={dtype} m={m} replay={iteration}')
            canary()
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();end.record();end.synchronize() # materialize events outside timing
        start.record()
        for _ in range(10):graph.replay()
        end.record();end.synchronize()
        print('SERVING_GRAPH',json.dumps(dict(m=m,experts=e,topk=t,dtype=str(dtype),
            workspace_bytes=workspace.numel(),replay_us=start.elapsed_time(end)*100.,
            scope='WARM_SYNTHETIC_GRAPH_NOT_MODEL_OR_PPU_CROSS_DEVICE_ADMISSION')),flush=True)
        # Error persists across subsequent valid work; inactive tokens never
        # turn malformed logits or IDs into a later host branch.
        live.fill_(m+1);error.zero_();graph.replay()
        assert error.item()==4 and torch.all(out==0)
        live.fill_(m);graph.replay();assert error.item()==4
        del graph
    finally:
        torch.cuda.current_stream().synchronize();up.close();dn.close()


def test_disjoint_streams_and_status(owner):
    streams=[torch.cuda.Stream(),torch.cuda.Stream()]
    holders=[]
    for idx,s in enumerate(streams):
        with torch.cuda.stream(s):
            other=Serving(os.environ['QUACTLIZE_SERVING_LIBRARY'],owner.device,s)
            raw=dev(row_blocks(20,4,160),torch.uint8);ids=dev([idx,idx,idx],torch.int64)
            out=dev(np.zeros((3,160)),torch.bfloat16);error=dev([16<<idx],torch.int32)
            other.gather_rows(raw,ids,out,error,qtype=20,table_rows=4)
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=s):other.gather_rows(raw,ids,out,error,qtype=20,table_rows=4)
            holders.append((other,raw,ids,out,error,g))
    for repeat in range(3):
        for s,entry in zip(streams,holders):
            with torch.cuda.stream(s):entry[-1].replay()
    for idx,(s,entry) in enumerate(zip(streams,holders)):
        s.synchronize();other,raw,ids,out,error,g=entry
        gold=dev(dequant_rows(row_blocks(20,4,160),20)[[idx]*3],torch.bfloat16)
        torch.testing.assert_close(out,gold,rtol=0,atol=0);assert error.item()==16<<idx
        g.reset();other.close()


@pytest.mark.parametrize('q',[7,8,10,11,12,13,14,42])
@pytest.mark.parametrize('m',[1,16])
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_matmul_binding_formats_and_strided_slots(owner,q,m,dtype):
    # Shared composition regression also exercises the separate bounded CUDA
    # implementation. The main PPU suite covers the production-size catalog.
    e,t,n,k=3,2,37,512;dt=int(dtype==torch.bfloat16)
    raw_np=raw_blocks(q,e,n,k);raw=dev(raw_np,torch.uint8)
    p=problem(q,m,n,k,experts=e,topk=t,channels=t,dtype=dt,padding=3,device=owner.device)
    input_store=torch.randn((m,t,k+3),device='cuda').to(dtype)*.03;x=input_store[...,:k]
    output_store=torch.full((m,t,n+3),37,device='cuda',dtype=dtype);out=output_store[...,:n]
    ids_store=torch.zeros((m,t+3),device='cuda',dtype=torch.int32);ids=ids_store[:,:t]
    status=dev([16],torch.int32);s=Matmul.query(owner,p)
    scratch=torch.empty(s.workspace_bytes,device='cuda',dtype=torch.uint8)
    plan=Matmul(owner,p,input=x,output=out,workspace=scratch,error=status,ids=ids,raw=raw)
    try:
        for selection in ((2,2),(-1,0),(3,-2)):
            ids[:,0]=selection[0];ids[:,1]=selection[1];status.fill_(16)
            plan(x,out,scratch,status,ids=ids,raw=raw);torch.cuda.current_stream().synchronize()
            for slot,expert in enumerate(selection):
                if 0<=expert<e:assert_dot(cpu(out[:,slot]),cpu(x[:,slot]),weight(raw_np[expert:expert+1],q,dt)[0],dt)
                else:assert torch.all(out[:,slot]==0)
            assert status.item()==(17 if selection==(3,-2) else 16)
            assert torch.all(output_store[...,n:]==37)
    finally:
        torch.cuda.current_stream().synchronize();plan.close()


def test_concurrent_prepared_matmuls_with_independent_scratch(owner):
    # M16 uses real scratch on both backends, rather than only concurrency of
    # zero-workspace helpers. Shared immutable raw weights, separate handles.
    raw_np=raw_blocks(8,3,64,64);raw=dev(raw_np,torch.uint8)
    torch.cuda.current_stream().synchronize()
    owners=[]
    for i in range(2):
        stream=torch.cuda.Stream()
        with torch.cuda.stream(stream):
            ops=Serving(os.environ['QUACTLIZE_SERVING_LIBRARY'],owner.device,stream)
            p=problem(8,16,64,64,experts=3,topk=2,device=owner.device)
            x=dev(np.full((16,1,64),.1*(i+1),'f4'),torch.bfloat16)
            out=dev(np.zeros((16,2,64)),torch.bfloat16);ids=dev(np.full((16,2),i),torch.int32)
            error=dev([16<<i],torch.int32)
            scratch=torch.empty(Matmul.query(ops,p).workspace_bytes,device='cuda',dtype=torch.uint8)
            assert scratch.numel()>0
            plan=Matmul(ops,p,input=x,output=out,workspace=scratch,error=error,ids=ids,raw=raw)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):plan(x,out,scratch,error,ids=ids,raw=raw)
            owners.append((ops,stream,x,out,ids,error,scratch,plan,graph))
    try:
        for _ in range(3):
            for ops,stream,x,out,ids,error,scratch,plan,graph in owners:
                with torch.cuda.stream(stream):graph.replay()
        for i,(ops,stream,x,out,ids,error,scratch,plan,graph) in enumerate(owners):
            stream.synchronize()
            for slot in range(2):assert_dot(cpu(out[:,slot]),cpu(x[:,0]),weight(raw_np[i:i+1],8,1)[0],1)
            assert error.item()==16<<i
    finally:
        for ops,stream,x,out,ids,error,scratch,plan,graph in owners:
            stream.synchronize();graph.reset();plan.close();ops.close()
