"""Build the isolated GGUF group32 collective; never a native PPU receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'quactlize/backends/ppu17'))
from dependency import REVISION as PPU_CUTLASS, require_cutlass
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
    p.add_argument('--enable-pdl',action='store_true',help='opt in the existing cooperative GDC protocol')
    a=p.parse_args()
    for key in ('compiler','cutlass','vllm'):
        setattr(a,key,getattr(a,key).resolve(strict=True))
    dependency = require_cutlass(a.cutlass)
    pin(a.vllm,VLLM)
    out=a.output.resolve()
    if out==ROOT or ROOT in out.parents or out.exists():
        raise ValueError('fresh external build directory required')
    out.mkdir(parents=True)
    # Keep the dependency checkout immutable. Materialize one exact optional
    # coordinate hook in this build only; preserve the scheduler and pipeline.
    relative=Path('cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp')
    original=a.cutlass/'include'/relative
    text=original.read_text()
    seam='          collective_mainloop.mma(\n'
    if text.count(seam)!=1:raise ValueError('pinned scheduler/MMA seam differs')
    hook=('          quactlize::ppu17::detail::KTileCoordinate<CollectiveMainloop>::set(\n'
          '              collective_mainloop, work_tile_info.K_idx);\n')
    overlay=out/'overlay'
    generated=overlay/relative;generated.parent.mkdir(parents=True)
    generated.write_text('#define QUACTLIZE_PPU17_GGUF_K_COORD_HOOK 1\n'
        '#include "quactlize/backends/ppu17/gguf_kernel_hook.cuh"\n'+text.replace(seam,hook+seam))
    sources=list(HERE.glob('*.cu'))+list(HERE.glob('*.h'))+[Path(__file__)]
    sources+=list((ROOT/'quactlize/backends/ppu17').glob('gguf*.cuh'))
    sources.append(ROOT/'quactlize/backends/ppu17/target.cuh')
    sources.append(ROOT/'quactlize/backends/ppu17/launch.cuh')
    sources.append(ROOT/'quactlize/backends/ppu17/format.cuh')
    sources.append(ROOT/'quactlize/backends/ppu17/dependency.py')
    sources += [ROOT/'quactlize/include'/name for name in (
        'gguf_packed_unit.hpp','gguf_scale_layout.hpp','gguf_scale_decode.hpp',
        'ppu_format_config.hpp','ppu_format_config.inc','kquant_kpack_offline.hpp')]
    bound={str(x):sha(x) for x in sources}
    common=[str(a.compiler),'-std=c++17','-O3','-DNDEBUG',
            '--expt-relaxed-constexpr','--extended-lambda',
            '-gencode=arch=compute_90a,code=sm_90a','-lineinfo',
            '-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1',
            '-I'+str(ROOT),'-I'+str(overlay),'-I'+str(a.cutlass/'include'),
            '-I'+str(ROOT/'quactlize/include'),
            '-I'+str(HERE.parent/'reference/shim'),'-I'+str(a.vllm/'csrc')]
    if a.enable_pdl:
        common += ['-DQ17_ENABLE_PDL=1','-DCUTLASS_ENABLE_GDC_FOR_SM90=1']
    receipt={'scope':'GGUF_Q4_K_GROUP32_SF_AND_FQ_HOPPER_NOT_NATIVE_PPU17',
             'pdl_compiled':a.enable_pdl,
             'config_count':6,
             'metadata_modes':2,
             'fq_contract':'GGUF_Q4_K_G32_PACKED_UNITS_PRIVATE_MMA_V1',
             'contract':'GGUF_Q4_K_G32_SF_PRIVATE_MMA_V1','cutlass':PPU_CUTLASS,
             'cutlass_identity':dependency,
             'vllm':VLLM,'source_sha256':bound,'compiler_sha256':sha(a.compiler),
             'coordinate_hook':{'original_sha256':sha(original),
                                'generated_file':str(generated.relative_to(out)),
                                'generated_sha256':sha(generated),'seams':1},
             'compiler_version':subprocess.check_output([str(a.compiler),'--version'],text=True),
             'builds':{}}
    for name in ('metadata','packed_abi','adapter','smoke','fq_smoke','split_start'):
        dest=out/{'metadata':'metadata-host','packed_abi':'packed-abi-host',
                  'adapter':'libgguf_q4.so','smoke':'smoke','fq_smoke':'fq-smoke',
                  'split_start':'split-start'}[name]
        command=common+[str(HERE/(name+'.cu')),'-o',str(dest)]
        if name=='adapter':command+=['-shared','-Xcompiler=-fPIC,-fvisibility=hidden']
        if name in ('smoke','fq_smoke','split_start'):
            command+=['-L'+str(out),'-lgguf_q4','-Xlinker','-rpath','-Xlinker','$ORIGIN']
        command+=['-L'+str(a.compiler.parent.parent/'lib64/stubs'),'-lcuda']
        (out/(name+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
        start=time.monotonic()
        with (out/(name+'.build.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                           env={**os.environ,'TMPDIR':str(out)})
        receipt['builds'][name]={'file':dest.name,'sha256':sha(dest),'seconds':time.monotonic()-start}
        if name in ('metadata','packed_abi'):
            result=subprocess.check_output([str(dest)],text=True)
            (out/(name+'-host.log')).write_text(result)
            print(result,flush=True,end='')
        print('GGUF_Q4_BUILD part='+name+' status=PASS',flush=True)
    if any(sha(path)!=digest for path,digest in bound.items()):
        raise ValueError('source changed during build')
    require_cutlass(a.cutlass);pin(a.vllm,VLLM)
    (out/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
