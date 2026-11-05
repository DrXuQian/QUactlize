#!/usr/bin/env python3
"""Separate NVIDIA compatibility DSO. No pip, Torch build, submodule fetch or dependency installation."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cuda',type=Path,default=Path(os.environ.get('CUDA_HOME','/usr/local/cuda')))
    p.add_argument('--cutlass',type=Path,required=True,help='existing stock NVIDIA CUTLASS checkout, not actlize')
    p.add_argument('--arch',default='80',help='CUDA SM number; PTX is included for forward compatibility')
    p.add_argument('--output',type=Path,required=True,help='new directory; never overwrite PPU binaries')
    a=p.parse_args()
    if not a.arch.isdigit():p.error('--arch must be numeric')
    nvcc=a.cuda.resolve()/'bin/nvcc'
    inc=a.cutlass.resolve()/'include'
    if not nvcc.is_file() or not (inc/'cutlass/cutlass.h').is_file():p.error('CUDA compiler or stock CUTLASS missing')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    objects=[]
    for source in ('quactlize/compat/serving.cu','quactlize/compat/cuda_matmul.cu','quactlize/packing/sizes.cpp'):
        obj=a.output/(Path(source).stem+'.o');objects.append(obj)
        command=[str(nvcc),'-std=c++17','-O3','--expt-relaxed-constexpr','-Xcompiler=-fPIC','-DQPM_CUDA',
            '-gencode',f'arch=compute_{a.arch},code=[sm_{a.arch},compute_{a.arch}]',
            '-I'+str(ROOT/'quactlize/include'),'-I'+str(inc),'-x','cu','-c',str(ROOT/source),'-o',str(obj)]
        subprocess.run(command,check=True)
    target=a.output/'libquactlize_cuda_serving.so'
    subprocess.run([str(nvcc),'-shared',*map(str,objects),'-lcublas','-Xlinker=-z,defs','-o',str(target)],check=True)
    for src,dst in (('compat/api.h','quactlize/compat/api.h'),('compat/serving.h','quactlize/compat/serving.h'),
                    ('packing/api.h','quactlize/packing/api.h'),('include/quactlize_ppu_config.h','quactlize_ppu_config.h')):
        dest=a.output/'include'/dst;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'quactlize'/src,dest)
    print(f'CUDA_SERVING_BUILT {target} device_validation=PENDING',flush=True)


if __name__=='__main__':main()
