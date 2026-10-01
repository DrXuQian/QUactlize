#!/usr/bin/env python3
"""Resolve the installed Python BF16 grouped entry once; never benchmark it.

Meta tensors carry shapes only. The installed provider chooses and compiles
its own tactic; its generated launch ABI is checked before native handoff.
"""
import argparse
import contextlib
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys


def resolve(args):
    import torch
    if args.experts!=256 or not 1024<=args.m<=32768 or args.m%8 or args.n<=0 or args.k<=0:
        raise ValueError('outside measured prefill domain')
    torch.cuda.set_device(args.device)
    prop=torch.cuda.get_device_properties(args.device)
    if prop.multi_processor_count!=72 or 'PPU-ZW810' not in prop.name:
        raise ValueError('DeepGEMM prefill requires measured PPU class')
    mod=importlib.import_module('deep_gemm.jit_kernels.m_grouped_gemm')
    fn=mod.m_grouped_gemm_bf16_bf16_bf16_nt_nopad
    if not inspect.isfunction(fn) or fn.__module__!=mod.__name__:
        raise ValueError('expected installed Python grouped BF16 entry')
    rt=importlib.import_module('deep_gemm.jit.runtime')
    template=importlib.import_module('deep_gemm.jit.template')
    seen=[]
    original=mod.jit_tuner.compile_and_tune
    expected=('lhs','rhs','out','grouped_layout','block_m_info','m','expected_m','stream','num_sms','smem_size','signal')

    def compile_only(*a,**kw):
        if a or kw['space']!=() or kw['name']!='m_grouped_gemm_bf16_bf16_bf16_nt':
            raise ValueError('provider changed to tuning or another kernel path')
        defs=kw['arg_defs']
        types=(torch.bfloat16,torch.bfloat16,torch.bfloat16,torch.int32,torch.int32,int,int,torch.cuda.Stream,int,int,torch.int32)
        if tuple(defs)!=tuple(zip(expected,types)):
            raise ValueError('provider launch ABI changed')
        runtime=original(**kw)
        root=Path(runtime.path).resolve(strict=True)
        declared=', '.join(f"('{name}', {template.typename_map[dtype]})" for name,dtype in defs)
        generated=template.generate(kw['includes'],defs,template.cpp_format(kw['template'],kw['keys']))
        if (root/'kernel.args').read_text()!=declared or (root/'kernel.cu').read_text()!=generated:
            raise ValueError('cached provider source or launch arguments differ')
        actual=kw['args']
        if actual[5]!=args.m or actual[6]!=(args.m+args.experts-1)//args.experts or actual[8]!=72:
            raise ValueError('provider launch scalars changed')
        seen.append((runtime,actual[8],actual[9],actual[4].numel()*4,kw['keys']))
        return runtime

    # Use the provider's public compile-only switch; do not replace its
    # runtime implementation or introduce a second tactic selector.
    previous_mode=rt.get_compile_mode()
    mod.jit_tuner.compile_and_tune=compile_only
    rt.set_compile_mode(rt.CompileMode.ONLY_COMPILE.value)
    try:
        fn(torch.empty((args.m,args.k),dtype=torch.bfloat16,device='meta'),
           torch.empty((args.experts,args.n,args.k),dtype=torch.bfloat16,device='meta'),
           torch.empty((args.m,args.n),dtype=torch.bfloat16,device='meta'),
           torch.empty(args.m,dtype=torch.int32,device='meta'),
           m_rows=torch.empty(args.experts,dtype=torch.int32,device='meta'))
    finally:
        mod.jit_tuner.compile_and_tune=original
        rt.set_compile_mode(previous_mode)
    if len(seen)!=1:
        raise ValueError('provider did not resolve exactly one module')
    runtime,sms,smem,scratch,keys=seen[0]
    root=Path(runtime.path).resolve(strict=True)
    files={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('kernel.cu','kernel.args','kernel.so')}
    receipt=dict(schema='quactlize.deepgemm-launch.v1',shape=[args.m,args.n,args.k,args.experts],
        device=prop.name,compute_units=sms,smem=smem,directory_bytes=scratch,keys=keys,files=files,
        python_entry=fn.__module__+'.'+fn.__name__,python_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
        scope='INSTALLED_PROVIDER_HEURISTIC_COMPILE_ONLY_NO_TUNING_NO_LAUNCH')
    # Trusted provider cache, same lifetime/permissions as its kernel.so.
    (root/f'quactlize-launch-m{args.m}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return (f'QKP_DEEPGEMM_V1 {args.m} {args.n} {args.k} {args.experts} {sms} {smem} {scratch}\n'
            f'{root / "kernel.so"}\n{files["kernel.so"]}\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('m','n','k','experts','device'):p.add_argument('--'+name,type=int,required=True)
    p.add_argument('--sdk',type=Path,required=True)
    args=p.parse_args()
    sdk=args.sdk.resolve(strict=True)
    if not (sdk/'bin/hgcc').is_file():raise ValueError('PPU SDK compiler missing')
    os.environ['PPU_SDK']=str(sdk)
    os.environ['PATH']=str(sdk/'bin')+os.pathsep+os.environ.get('PATH','')
    # Native SDK/compiler diagnostics may use the OS fd, not Python stdout.
    protocol=os.dup(1)
    try:
        os.dup2(2,1)
        with contextlib.redirect_stdout(sys.stderr):reply=resolve(args)
    finally:
        os.dup2(protocol,1);os.close(protocol)
    sys.stdout.write(reply)


if __name__=='__main__':main()
