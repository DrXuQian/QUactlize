"""One same-image, all-SIMT prepare call with an independent router oracle."""
import ctypes as C

import numpy as np

from dev.bf16_compute.native import Projection, MoePlan, MixedPlanV2, Router, function
from dev.gemv_simt.native import checked
from quactlize.execution.native import Call


def prepare(rt, library, point):
    tokens, k = point['tokens'], point['k']
    if tokens != 1 or k not in (512,2048) or point['simt_mask'] != 5:
        raise ValueError('profile requires the observed all-SIMT M1 prepare domain')
    query = function(library, 'quactlize_kpack_moe_simt_query_v1', [C.POINTER(Call), C.POINTER(C.c_uint64)])
    bind = function(library, 'quactlize_kpack_moe_simt_bind_v1',
                    [C.POINTER(Call), C.c_int, C.c_void_p, C.c_uint64, C.POINTER(Projection)])
    stage = function(library, 'quactlize_kpack_moe_mixed_stage_v2',
                     [C.POINTER(MixedPlanV2), C.c_int, C.c_void_p])
    ids_host = np.full((tokens,11), -71, dtype='<i4')
    ids = rt.upload(ids_host)
    source = rt.upload(np.linspace(-1,1,tokens*k,dtype='<f4'))
    tails = []

    def projection(n, kk, channels):
        c = Call(version=1,size=C.sizeof(Call),qtype=12,n=n,k=kk,experts=256,
                 rows=tokens*8,mode=2,input_type=1,channels=channels,topk=8,
                 a_row_stride=kk,a_token_stride=channels*kk,ids_stride=11,out_row_stride=n,
                 a=source if channels==1 else rt.allocate(tokens*channels*kk*4),
                 ids=ids,output=rt.allocate(tokens*8*n*4),stream=rt.stream.value)
        size = C.c_uint64()
        checked(query(C.byref(c),C.byref(size)), 'MoE scratch query')
        scratch = rt.allocate(size.value+32)
        rt.fill(scratch,size.value+32)
        result = Projection()
        checked(bind(C.byref(c),0,scratch,size.value,C.byref(result)), 'MoE projection bind')
        tails.append(scratch+size.value)
        return result

    gate, down = projection(1024,k,1), projection(2048,512,8)
    # Distinct scores avoid ambiguous near-ties. Check full softmax/top8/norm,
    # even though the selected-score normalization cancels its denominator.
    logits = np.random.default_rng(71933).normal(0,2,(tokens,256)).astype('<f4')
    score = np.exp(logits.astype('f8')-logits.max(axis=1,keepdims=True))
    score /= score.sum(axis=1,keepdims=True)
    expected_ids = np.argsort(-score,axis=1,kind='stable')[:,:8]
    chosen = np.take_along_axis(score,expected_ids,axis=1)
    expected_weights = chosen/np.maximum(chosen.sum(axis=1,keepdims=True),6.103515625e-5)*1.25
    weights = rt.allocate(tokens*8*4)
    router = Router(1,C.sizeof(Router),0,1,0,0,6.103515625e-5,1.25,rt.upload(logits),None,weights)
    plan = MoePlan(1,C.sizeof(MoePlan),1,0,gate,gate,down,router)
    request = MixedPlanV2(2,C.sizeof(MixedPlanV2),plan,5,point['compute'])

    def check():
        got = rt.download(ids,ids_host.nbytes).view('<i4').reshape(tokens,11)
        if not np.array_equal(got[:,:8],expected_ids) or not np.all(got[:,8:]==-71):
            raise ValueError('production prepare top8/ID guard differs')
        got_weights = rt.download(weights,tokens*8*4).view('<f4').reshape(tokens,8)
        if not np.allclose(got_weights,expected_weights,atol=2e-6,rtol=2e-6):
            raise ValueError('production prepare weights differ')
        for p in (gate,down):
            rows = rt.download(p.io.row_ids,tokens*8*4).view('<i4')
            header = rt.download(p.directory_header,16).view('<i4')
            if not np.array_equal(rows,np.arange(tokens*8)) or list(header)!=[0,0,8,256]:
                raise ValueError('production prepare identity map/header differs')
        if any(not np.all(rt.download(tail,32)==0xa5) for tail in tails):
            raise ValueError('production prepare scratch guard changed')
        return dict(status='PASS',oracle='INDEPENDENT_SOFTMAX_TOP8_NORM_IDENTITY_MAP',
                    max_weight_error=float(np.max(np.abs(got_weights-expected_weights))))

    return lambda: stage(C.byref(request),0,rt.stream), check
