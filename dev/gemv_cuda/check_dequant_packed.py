#!/usr/bin/env python3
"""Exact-header CUDA numerical check; PPU timing/admission is separate."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
CONFIGS=(4,5,10,11,12)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def emit(folder):
    from tools.kpack_dequant_fixture import fixture
    folder.mkdir(parents=True,exist_ok=False)
    rows=[]
    for q in (12,13):
        for n,k,e in ((256,512,3),(512,768,3),(512,2048,3),(1024,3072,1)):
            planes,gold=fixture(q,n,k,e,1)
            path=folder/f'q{q}-n{n}-k{k}-e{e}.npz'
            np.savez_compressed(path,low=planes['low'],high=planes['high'],units=planes['units'],gold=gold)
            rows.append(dict(q=q,n=n,k=k,experts=e,path=path.name,sha256=sha(path)))
    (folder/'manifest.json').write_text(json.dumps(dict(scope='ORIGINAL_GGUF_BF16_ORACLE',cases=rows),indent=2)+'\n')


def check(folder,output,nvcc):
    import torch
    from tools.kpack_dequant_fixture import compare
    output.mkdir(parents=True,exist_ok=False)
    includes=[ROOT/'dev/gemv_cuda/compat',ROOT,ROOT/'quactlize/include',ROOT/'third_party/actlize/include']
    device=torch.cuda.get_device_properties(0)
    library=output/'dequant_packed_cuda.so'
    command=[str(nvcc),'-std=c++17','-O3','--fmad=false','--expt-relaxed-constexpr',
        f'-arch=sm_{device.major}{device.minor}','-shared','--cudart=shared','-Xcompiler=-fPIC',
        '-DCUTLASS_USE_PACKED_TUPLE=1','-DCUTE_USE_PACKED_TUPLE=1',
        '-include',str(ROOT/'dev/gemv_cuda/compat/compiler_bridge.h'),
        *[f'-I{p}' for p in includes],str(ROOT/'dev/gemv_cuda/dequant_packed_check.cu'),'-o',str(library)]
    inputs={str(p.relative_to(ROOT)):sha(p) for d in
        (ROOT/'quactlize/dequant',ROOT/'quactlize/execution',ROOT/'quactlize/include',
         ROOT/'third_party/actlize/include',ROOT/'dev/gemv_cuda/compat')
        for p in d.rglob('*') if p.is_file() and p.suffix in ('.hpp','.h','.cuh','.cu','.inc')}
    for name in ('dev/gemv_cuda/check_dequant_packed.py','dev/gemv_cuda/dequant_packed_check.cu'):
        inputs[name]=sha(ROOT/name)
    with (output/'build.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    lib=C.CDLL(str(library));fn=lib.dequant_packed_cuda
    fn.argtypes=[C.c_int]*2+[C.c_void_p]*4+[C.c_int]*3+[C.c_void_p];fn.restype=C.c_int
    manifest=json.loads((folder/'manifest.json').read_text());results=[]
    stream=torch.cuda.Stream()
    for w in manifest['cases']:
        path=folder/w['path']
        if sha(path)!=w['sha256']:raise ValueError('fixture checksum differs')
        with np.load(path,allow_pickle=False) as data:
            gold=data['gold'];plane={n:np.ascontiguousarray(data[n]) for n in ('low','high','units')}
        with torch.cuda.stream(stream):
            gpu={name:torch.from_numpy(v.copy()).cuda() for name,v in plane.items() if v.size}
            storage=torch.full((gold.size+128,),0x25a5,dtype=torch.int16,device='cuda')
            out=storage[64:-64]
            def launch(c):
                rc=fn(w['q'],c,*[gpu[n].data_ptr() if n in gpu else None for n in ('low','high','units')],
                      out.data_ptr(),w['n'],w['k'],w['experts'],stream.cuda_stream)
                if rc:raise ValueError(f'CUDA launch failed rc={rc}')
                stream.synchronize()
            for c in CONFIGS:
                out.fill_(0x7fff);launch(c)
                got=out.cpu().numpy().view('u2').reshape(gold.shape)
                proof=compare(got,gold)
                if not bool((storage[:64]==0x25a5).all()) or not bool((storage[-64:]==0x25a5).all()):
                    raise ValueError('output guard changed')
                # Both code planes matter. Q5's high-bit plane must also fail independently.
                negatives={}
                for name in ('low','high'):
                    if name not in gpu:continue
                    gpu[name].zero_();launch(c)
                    bad=int(np.count_nonzero(out.cpu().numpy().view('u2').reshape(gold.shape)!=gold))
                    if bad<gold.size//100:raise ValueError('blank plane escaped oracle: '+name)
                    negatives[name]=bad
                    gpu[name].copy_(torch.from_numpy(plane[name]));launch(c)
                    compare(out.cpu().numpy().view('u2').reshape(gold.shape),gold)
                result=dict(workload=w,config=c,status='PASS',proof=proof,negatives=negatives,guards='PASS')
                results.append(result);print('DEQUANT_PACKED_CUDA '+json.dumps(result),flush=True)
    if any(sha(ROOT/p)!=h for p,h in inputs.items()):raise ValueError('source changed during CUDA check')
    record=dict(status='PASS',scope='EXACT_HEADERS_NVIDIA_NUMERICAL_ONLY',device=device.name,
        cases=len(manifest['cases']),cells=len(results),results=results,command=command,
        source_hashes=inputs,fixture_manifest_sha256=sha(folder/'manifest.json'),library_sha256=sha(library),
        ppu_admission=False,performance_admission=False)
    (output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'DEQUANT_PACKED_CUDA_DONE status=PASS cells={len(results)} ppu_admission=0',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--emit',action='store_true')
    p.add_argument('--output',type=Path);p.add_argument('--nvcc',type=Path,default=Path('/usr/local/cuda-12.8/bin/nvcc'))
    a=p.parse_args()
    if a.emit:emit(a.fixtures)
    elif a.output:check(a.fixtures,a.output,a.nvcc)
    else:p.error('--output is required for the device check')


if __name__=='__main__':main()
