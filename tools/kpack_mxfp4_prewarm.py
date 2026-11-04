#!/usr/bin/env python3
"""Compile one installed Sail W4FA16 recipe, then hand off its dynamic ABI.

No quantization, GPU tensor allocation, timing sweep or GEMM launch. The
provider's heuristic is a proposal, NOT measured Q4_K/ZW810 table evidence.
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

ROOT = Path(__file__).resolve().parents[1]


def resolve(args):
    import torch
    if (args.m <= 0 or args.n <= 0 or args.n % 256 or args.k <= 0 or args.k % 256 or
            not 1 <= args.experts <= 1024 or args.m*max(args.n,args.k) > 2147483647):
        raise ValueError('outside MXFP4 grouped shape contract')
    torch.cuda.set_device(args.device)
    prop = torch.cuda.get_device_properties(args.device)
    if not prop.name.startswith('PPU-') or (prop.major,prop.minor) not in ((8,0),(8,9)) or prop.multi_processor_count <= 0:
        raise ValueError('Sail W4FA16 requires a PPU1.0/1.5 device')
    mod = importlib.import_module('deep_gemm.jit_kernels.m_grouped_gemm_w4a16')
    fn = mod.m_grouped_gemm_w4a16_nopad
    if not inspect.isfunction(fn) or fn.__module__ != mod.__name__:
        raise ValueError('expected installed Sail Python W4FA16 entry')
    rt = importlib.import_module('deep_gemm.jit.runtime')
    template = importlib.import_module('deep_gemm.jit.template')
    utils = importlib.import_module('deep_gemm.jit_kernels.utils')
    # Some installed releases cap ZW810E to 20 here. Use actual queried CUs.
    utils.set_num_sms(prop.multi_processor_count)
    headers = [ROOT/'quactlize/mxfp4'/name for name in ('deepgemm.cuh','e8m0.hpp')]
    header_hash = hashlib.sha256(b''.join(p.read_bytes() for p in headers)).hexdigest()
    seen = []
    original = mod.jit_tuner.compile_and_tune
    expected = ('lhs','rhs','rhs_scales','out','m','expected_m','stream','num_sms','m_rows','block_m_info')

    def compile_only(*a, **kw):
        if a or kw['space'] != () or kw['name'] != 'm_grouped_gemm_w4a16':
            raise ValueError('provider changed to tuning or another kernel path')
        defs = kw['arg_defs']
        types = (torch.bfloat16,torch.int32,torch.uint8,torch.bfloat16,int,int,torch.cuda.Stream,int,torch.int32,torch.int32)
        keys = kw['keys']
        if (tuple(defs) != tuple(zip(expected,types)) or kw['jit_include_dir'] != 'actlize_v1.0.0' or
                any(keys.get(k) != v for k,v in dict(N=args.n,K=args.k,NUM_GROUPS=args.experts,
                    GROUP_SIZE=32,GEMM_TYPE='GroupedNoPad',ELEMENT_B='cutlass::float4_t',ELEMENT_SCALE='uint8_t').items())):
            raise ValueError('provider W4FA16 launch ABI/layout changed')
        actual = kw['args']
        if actual[4] != args.m or actual[5] != (args.m+args.experts-1)//args.experts or actual[7] != prop.multi_processor_count:
            raise ValueError('provider launch scalars changed')
        # Source identity covers our small endpoint correction as well as the
        # installed provider's header hash; never reuse an uncorrected image.
        kw = dict(kw, includes=('"'+str(headers[0])+'"',),
                  template='// Quactlize E8M0 endpoints '+header_hash+'\n'+kw['template'])
        runtime = original(**kw)
        root = Path(runtime.path).resolve(strict=True)
        declared = ', '.join(f"('{name}', {template.typename_map[dtype]})" for name,dtype in defs)
        generated = template.generate(kw['includes'],defs,template.cpp_format(kw['template'],keys))
        if (root/'kernel.args').read_text() != declared or (root/'kernel.cu').read_text() != generated:
            raise ValueError('cached provider source or launch arguments differ')
        seen.append((runtime,actual[9].numel()*4,keys))
        return runtime

    previous_mode = rt.get_compile_mode()
    mod.jit_tuner.compile_and_tune = compile_only
    rt.set_compile_mode(rt.CompileMode.ONLY_COMPILE.value)
    try:
        fn(torch.empty((args.m,args.k),dtype=torch.bfloat16,device='meta'),
           (torch.empty((args.experts,args.k//16,args.n*2),dtype=torch.int32,device='meta'),
            torch.empty((args.experts,args.k//32,args.n),dtype=torch.uint8,device='meta')),
           torch.empty((args.m,args.n),dtype=torch.bfloat16,device='meta'),
           torch.empty(args.m,dtype=torch.int32,device='meta'),
           m_rows=torch.empty(args.experts,dtype=torch.int32,device='meta'))
    finally:
        mod.jit_tuner.compile_and_tune = original
        rt.set_compile_mode(previous_mode)
    if len(seen) != 1:
        raise ValueError('provider did not resolve exactly one module')
    runtime,scratch,keys = seen[0]
    root = Path(runtime.path).resolve(strict=True)
    config = 'w4fa16:tm{BLOCK_M}:tn{BLOCK_N}:tk{BLOCK_K}:wm{WARP_M}:wn{WARP_N}:wk{WARP_K}:s{NUM_STAGES}'.format(**keys)
    receipt = dict(schema='quactlize.mxfp4-w4fa16.v1',shape=[args.m,args.n,args.k,args.experts],
        mapping_id='4d58463441313601',device=prop.name,compute_units=prop.multi_processor_count,
        directory_bytes=scratch,keys=keys,config=config,adapter_sha256=header_hash,
        files={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('kernel.cu','kernel.args','kernel.so')},
        python_entry=fn.__module__+'.'+fn.__name__,python_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
        scope='INSTALLED_PROVIDER_HEURISTIC_COMPILE_ONLY; DEVICE_ACCURACY_AND_PERFORMANCE_PENDING')
    (root/f'quactlize-mxfp4-m{args.m}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return (f'QZ_MXFP4_V1 {args.m} {args.n} {args.k} {args.experts} {prop.multi_processor_count} {scratch}\n'
            f'{root/"kernel.so"}\n{config}\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('m','n','k','experts','device'):
        p.add_argument('--'+name,type=int,required=True)
    p.add_argument('--sdk',type=Path,required=True)
    args = p.parse_args()
    sdk = args.sdk.resolve(strict=True)
    if not (sdk/'bin/hgcc').is_file():
        raise ValueError('PPU SDK compiler missing')
    os.environ['PPU_SDK'] = str(sdk)
    os.environ['DG_JIT_HGCC_COMPILER'] = str(sdk/'bin/hgcc')
    os.environ['PATH'] = str(sdk/'bin')+os.pathsep+os.environ.get('PATH','')
    protocol = os.dup(1)
    try:
        os.dup2(2,1)
        with contextlib.redirect_stdout(sys.stderr):
            reply = resolve(args)
    finally:
        os.dup2(protocol,1)
        os.close(protocol)
    sys.stdout.write(reply)


if __name__ == '__main__':
    main()
