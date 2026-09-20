"""Exercise the production prepare snapshot and weighted finish, without GEMM."""
import ctypes as C
import json
from pathlib import Path
import traceback

import numpy as np

from dev.bf16_compute.native import Projection, MoePlan, MixedPlanV2, function
from dev.bf16_compute.fixture import round_compute
from dev.gate_up_perf.bench import Buffer
from dev.gemv_simt.native import Runtime, Graph, checked
from quactlize.dispatch.native import Router, MoeFinish
from quactlize.execution.native import Call


def case(rt, lib, tokens, compute, alias, router):
    n, k, hidden, top = 3072, 3072, 512, 8
    query=function(lib,'quactlize_kpack_moe_simt_query_v1',[C.POINTER(Call),C.POINTER(C.c_uint64)])
    bind=function(lib,'quactlize_kpack_moe_simt_bind_v1',
                  [C.POINTER(Call),C.c_int,C.c_void_p,C.c_uint64,C.POINTER(Projection)])
    prepare=function(lib,'quactlize_kpack_moe_prepare_finish_v1',
                     [C.POINTER(MixedPlanV2),C.POINTER(MoeFinish),C.c_void_p])
    finish=function(lib,'quactlize_kpack_moe_weighted_finish_v2',
                    [C.POINTER(MixedPlanV2),C.POINTER(MoeFinish),C.c_void_p])
    mark=len(rt.allocations)
    graph=None
    try:
        arena=Buffer(rt,tokens*n*4+64)
        route_weights=Buffer(rt,tokens*11*4)
        route_ids=Buffer(rt,tokens*13*4)
        logits=Buffer(rt,tokens*256*4)
        weights_ptr=arena.ptr+(16 if alias=='weights-offset' else 0) if alias.startswith('weights') else route_weights.ptr
        ids_ptr=arena.ptr if alias=='ids' else route_ids.ptr
        stride=8 if router else 11
        parts=[]
        buffers=[]
        for q,width,depth,channels in ((12,2*hidden,k,1),(13,n,hidden,8)):
            act=Buffer(rt,tokens*channels*depth*4)
            output=Buffer(rt,tokens*top*width*4)
            call=Call(version=1,size=C.sizeof(Call),qtype=q,n=width,k=depth,experts=256,
                      rows=tokens*top,mode=2,input_type=1,channels=channels,topk=top,
                      a_row_stride=depth,a_token_stride=channels*depth,ids_stride=13,
                      out_row_stride=width,a=act.ptr,output=output.ptr,ids=ids_ptr,stream=rt.stream)
            size=C.c_uint64()
            checked(query(C.byref(call),C.byref(size)),'snapshot scratch query')
            scratch=Buffer(rt,size.value)
            part=Projection()
            checked(bind(C.byref(call),-1,scratch.ptr,size.value,C.byref(part)),'snapshot production bind')
            parts.append(part);buffers.extend((act,output,scratch))
        plan=MoePlan(1,C.sizeof(MoePlan),1,0,parts[0],Projection(),parts[1],Router())
        if router:
            plan.router=Router(1,C.sizeof(Router),0,1,0,0,2**-14,1.25,logits.ptr,None,weights_ptr)
        typed=MixedPlanV2(2,C.sizeof(MixedPlanV2),plan,5,compute)
        request=MoeFinish(1,C.sizeof(MoeFinish),stride,n,weights_ptr,arena.ptr)
        retained=MoeFinish(1,C.sizeof(MoeFinish),8,n,plan.down.outputs,arena.ptr)
        launch=lambda:prepare(C.byref(typed),C.byref(request),rt.stream)
        complete=lambda:finish(C.byref(typed),C.byref(retained),rt.stream)
        graph=Graph(rt,[launch,complete])
        worst=0.0
        for repeat in range(4):
            arena.poison()
            ids=np.full((tokens,13),-91,'i4')
            weights=np.full((tokens,stride),-37,'f4')
            for token in range(tokens):
                ids[token,:8]=(np.arange(8)*29+repeat*7+token)%256
                weights[token,:8]=(np.arange(8)+1+token+repeat)/64
            if router:
                scores=-((np.arange(tokens*256).reshape(tokens,256)*29+repeat*7)%256).astype('f4')/64
                selected=np.argsort(-scores,axis=1,kind='stable')[:,:8]
                vals=np.take_along_axis(scores,selected,axis=1).astype('f8')
                ex=np.exp(vals-vals.max(axis=1,keepdims=True))
                weights[:,:8]=(ex/ex.sum(axis=1,keepdims=True)*1.25).astype('f4')
                rt.copy(logits.ptr,scores)
            rt.copy(ids_ptr,ids)
            if not router:rt.copy(weights_ptr,weights)
            values=((np.arange(tokens*8*n).reshape(tokens,8,n)*7+repeat*13)%53-26).astype('f4')/16
            if compute and repeat==3:values*=65536
            rt.copy(plan.down.io.output,values)
            rounded=round_compute(values,'bf16') if compute else values
            products=(rounded*weights[:,:8,None]).astype('f4')
            gold=products[:,0].copy()
            for slot in range(1,8):gold=np.add(gold,products[:,slot],dtype='f4')
            graph.sample()
            result=arena.read()[:tokens*n*4].view('f4').reshape(tokens,n)
            denom=np.maximum(np.abs(products.astype('f8')).sum(axis=1),1e-20)
            error=float(np.max(np.abs(result.astype('f8')-gold)/denom))
            if not np.isfinite(result).all() or not np.isfinite(error) or error>2e-6:
                raise ValueError(f'finish snapshot oracle differs: {error}')
            worst=max(worst,error)
            snapshot=rt.download(plan.down.outputs,tokens*8*4).view('f4').reshape(tokens,8)
            if not np.allclose(snapshot,weights[:,:8],rtol=2e-6,atol=1e-7):
                raise ValueError('retained route weights differ')
            for buf in (arena,route_weights,route_ids,logits,*buffers):buf.check_guard()
        if not router:
            rt.copy(weights_ptr,weights)
            ids[0,0]=256;rt.copy(ids_ptr,ids)
            graph.sample()
            result=arena.read()[:n*4].view('f4')
            if not np.isnan(result).all():raise ValueError('invalid ID did not poison finish')
        return dict(tokens=tokens,compute=compute,alias=alias,router=router,
                    repeats=4,error=worst,status='PASS')
    finally:
        if graph:graph.close()
        rt.release_after(mark)


def run(sdk, bundle, output):
    result=dict(status='FAIL',scope='PRODUCTION_PREPARE_SNAPSHOT_AND_FINISH_NO_GEMM',timing_valid=False,cases=[])
    rt=None
    try:
        rt=Runtime(sdk,'ppu')
        lib=C.CDLL(str(Path(bundle)/'libquactlize_ppu_execution.so'),mode=C.RTLD_LOCAL)
        for compute in (0,1):
            for tokens in range(1,9):
                for alias in ('none','weights','weights-offset','ids'):
                    for router in (False,True):
                        result['cases'].append(case(rt,lib,tokens,compute,alias,router))
            print(f'MOE_FINISH_SNAPSHOT compute={compute} cases={len(result["cases"])} status=PASS',flush=True)
        result['status']='PASS'
        return 0
    except Exception as exc:
        result['error']=str(exc);traceback.print_exc();return 1
    finally:
        if rt:rt.close()
        (Path(output)/'moe-finish-snapshot.json').write_text(json.dumps(result,indent=2)+'\n')
