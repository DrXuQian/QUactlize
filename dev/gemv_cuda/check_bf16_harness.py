#!/usr/bin/env python3
"""NVIDIA device check of the BF16 oracle/stream harness, NOT DeepGEMM admission.

Use resident BF16 random weights and real CUDA cuBLAS calls per expert. There
is no PPU library, no GGUF conversion, no timing policy and no provider fallback
in this test. It exercises the shared zero-A diagnostic and changed-input
graph checks before the installed PPU Python-JIT provider is tested on box.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.kpack_bf16_fixture import Oracle, row_domain
from tools.kpack_dequant_fixture import bf16
from tools.run_kpack_bf16_gate import Weights
from quactlize.runtime.native import SDK


class CudaNames:
    """Bind real CUDA runtime functions, not host-only no-op PPU stubs."""
    def __init__(self,path):self.runtime=C.CDLL(str(path))
    def __getattr__(self,name):
        if not name.startswith('hggc'):raise AttributeError(name)
        return getattr(self.runtime,'cuda'+name[4:])


def case(folder,runtime,n,k,tokens):
    experts=256
    case_id=f'bf16-e{experts}-n{n}-k{k}-t{tokens}'
    rng=np.random.default_rng(983771+n+k)
    bits=bf16(rng.uniform(-.25,.25,(experts,n,k)).astype('f4'))
    oracle=Oracle(bits);counts,indices,route_hash=row_domain(tokens,experts)
    m=len(indices);abits,coeff=oracle.activations(m)
    offsets=np.r_[0,np.cumsum(counts)]
    stream=torch.cuda.Stream()
    sdk=SDK.__new__(SDK);sdk.lib=CudaNames(runtime)
    sdk.lib.hggcMemcpy.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int]
    sdk.lib.hggcMemcpy.restype=C.c_int
    sdk.lib.hggcStreamSynchronize.argtypes=[C.c_void_p]
    sdk.lib.hggcStreamSynchronize.restype=C.c_int
    checker=Weights.__new__(Weights)
    checker.sdk=sdk;checker.stream=stream;checker.execution_stream=C.c_void_p(stream.cuda_stream)
    checker.output_folder=folder;checker.w=dict(id=case_id,n=n,k=k,experts=experts)
    checker.provider=type('Control',(),{'identity':dict(provider='NVIDIA_CUBLAS_HARNESS_CONTROL',
        actual_deepgemm=False,ppu_admission=False)})()
    guards=64
    with torch.cuda.stream(stream):
        b=torch.from_numpy(bits.view('i2')).view(torch.bfloat16).cuda()
        a=torch.from_numpy(abits.view('i2')).view(torch.bfloat16).cuda()
        storage=torch.full((m*n+2*guards,),-123.,dtype=torch.bfloat16,device='cuda')
        out=storage[guards:-guards].reshape(m,n)
        def launch():
            for e,(start,end) in enumerate(zip(offsets[:-1],offsets[1:])):
                if start!=end:torch.mm(a[start:end],b[e].T,out=out[start:end])
        def proof(c):
            stream.synchronize()
            assert bool((storage[:guards]==-123).all()) and bool((storage[-guards:]==-123).all())
            value=checker.read_bits(out)
            err=oracle.error(value,c,indices)
            if err>=.005:raise ValueError(f'CUDA independent BF16 oracle error {err}')
            return err
        launch();error=proof(coeff)
        checker.check_zero_a(a,out,launch,indices,tokens)
        a.copy_(torch.from_numpy(abits.view('i2')).view(torch.bfloat16));launch();error=max(error,proof(coeff))
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            launch();launch()
        graph.replay();graph.replay();stream.synchronize()
        a.neg_();out.fill_(float('nan'));graph.replay();error=max(error,proof(-coeff))
        a.zero_();out.fill_(float('nan'));graph.replay();stream.synchronize()
        zero=checker.read_bits(out)
        if np.any(zero&0x7fff):raise ValueError('zero-A graph replay left output residuals')
        planted=oracle.error(zero,coeff,indices)
        if planted<=.005:raise ValueError('missing-compute negative escaped the independent oracle')
        # Reassigning every row to expert 0 must not match the real route.
        a.copy_(torch.from_numpy(abits.view('i2')).view(torch.bfloat16))
        for start,end in zip(offsets[:-1],offsets[1:]):
            if start!=end:torch.mm(a[start:end],b[0].T,out=out[start:end])
        stream.synchronize()
        wrong_expert=oracle.error(checker.read_bits(out),coeff,indices)
        if wrong_expert<=.005:raise ValueError('wrong-expert negative escaped the independent oracle')
        graph.replay();error=max(error,proof(coeff))
        result=dict(case=case_id,status='PASS',scope='CUDA_BF16_HARNESS_NOT_PPU_DEEPGEMM',
            n=n,k=k,tokens=tokens,experts=experts,total_rows=m,active=int(np.count_nonzero(counts)),
            empty=int(np.count_nonzero(counts==0)),stream=int(stream.cuda_stream),
            weight_sha256=hashlib.sha256(bits).hexdigest(),routes_sha256=route_hash,error=error,
            zero_a='PASS',nan_overwrite='PASS',changed_a_graph='PASS',guards='PASS',
            zero_output_negative=planted,wrong_expert_negative=wrong_expert,
            graph_first_launch_excluded=True,performance_admission=False,ppu_admission=False)
        del graph
    print('BF16_CUDA_HARNESS '+json.dumps(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runtime',type=Path,default=Path('/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'result.json').exists():raise ValueError('result already exists')
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    device=torch.cuda.get_device_properties(0)
    rows=[]
    for n,k in ((256,512),(512,2048)):
        for tokens in (128,2048,4096):
            rows.append(case(a.output,a.runtime,n,k,tokens))
    result=dict(status='PASS',scope='NVIDIA_CUDA_CONTROL_ONLY',deepgemm_executed=False,
        device=device.name,capability=[device.major,device.minor],torch=torch.__version__,cuda=torch.version.cuda,
        cases=rows,ppu_admission=False,performance_admission=False)
    (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'BF16_CUDA_HARNESS_DONE status=PASS cells={len(rows)} deepgemm_executed=0 ppu_admission=0',flush=True)


if __name__=='__main__':main()
