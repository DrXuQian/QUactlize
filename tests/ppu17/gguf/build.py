"""Build the isolated GGUF group32 collective; never a native PPU receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PPU_CUTLASS = '023e82d03e80b4d5982f664925e15499033df314'
VLLM = '11ba93f3646d4c5476c3b3fd56835589701f0fb1'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pin(path, revision):
    actual = subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True)
    if actual != revision or dirty:
        raise ValueError('dependency must be clean and pinned: '+str(path))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--cutlass',type=Path,required=True)
    p.add_argument('--vllm',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    for key in ('compiler','cutlass','vllm'):
        setattr(a,key,getattr(a,key).resolve(strict=True))
    pin(a.cutlass,PPU_CUTLASS); pin(a.vllm,VLLM)
    out=a.output.resolve()
    if out==ROOT or ROOT in out.parents or out.exists():
        raise ValueError('fresh external build directory required')
    out.mkdir(parents=True)
    sources=list(HERE.glob('*.cu'))+list(HERE.glob('*.h'))+[Path(__file__)]
    sources+=list((ROOT/'quactlize/backends/ppu17').glob('gguf*.cuh'))
    sources.append(ROOT/'quactlize/backends/ppu17/target.cuh')
    bound={str(x):sha(x) for x in sources}
    common=[str(a.compiler),'-std=c++17','-O3','-DNDEBUG',
            '--expt-relaxed-constexpr','--extended-lambda',
            '-gencode=arch=compute_90a,code=sm_90a','-lineinfo',
            '-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1',
            '-I'+str(ROOT),'-I'+str(a.cutlass/'include'),
            '-I'+str(HERE.parent/'reference/shim'),'-I'+str(a.vllm/'csrc')]
    receipt={'scope':'GGUF_Q4_K_GROUP32_SF_HOPPER_NOT_NATIVE_PPU17',
             'config_count':6,
             'contract':'GGUF_Q4_K_G32_SF_PRIVATE_MMA_V1','cutlass':PPU_CUTLASS,
             'vllm':VLLM,'source_sha256':bound,'compiler_sha256':sha(a.compiler),
             'compiler_version':subprocess.check_output([str(a.compiler),'--version'],text=True),
             'builds':{}}
    for name in ('metadata','adapter','smoke'):
        dest=out/{'metadata':'metadata-host','adapter':'libgguf_q4_sf.so','smoke':'smoke'}[name]
        command=common+[str(HERE/(name+'.cu')),'-o',str(dest)]
        if name=='adapter':command+=['-shared','-Xcompiler=-fPIC,-fvisibility=hidden']
        if name=='smoke':command+=['-L'+str(out),'-lgguf_q4_sf','-Xlinker','-rpath','-Xlinker','$ORIGIN']
        command+=['-L'+str(a.compiler.parent.parent/'lib64/stubs'),'-lcuda']
        (out/(name+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
        start=time.monotonic()
        with (out/(name+'.build.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                           env={**os.environ,'TMPDIR':str(out)})
        receipt['builds'][name]={'file':dest.name,'sha256':sha(dest),'seconds':time.monotonic()-start}
        if name=='metadata':
            result=subprocess.check_output([str(dest)],text=True)
            (out/'metadata-host.log').write_text(result)
            print(result,flush=True,end='')
        print('GGUF_Q4_BUILD part='+name+' status=PASS',flush=True)
    if any(sha(path)!=digest for path,digest in bound.items()):
        raise ValueError('source changed during build')
    pin(a.cutlass,PPU_CUTLASS);pin(a.vllm,VLLM)
    (out/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
