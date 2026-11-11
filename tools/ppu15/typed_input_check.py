"""Raw-bit typed-A/AIU closure; no GEMM timing or tolerance relaxation."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from ci.support.fixture import bf16_bits
from tools.ppu15.check import Arena, bind, setup
from tools.run_zw890_bringup import command
from quactlize.runtime.compiler import FLAGS, LIBRARIES, sha
from quactlize.runtime.target import flags_for_arch
from quactlize.runtime.native import checked


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--build-only',action='store_true')
    a=p.parse_args();sdk_path=a.sdk.resolve();output=a.output.resolve()
    work=output/'build';work.mkdir(parents=True,exist_ok=False)
    source=ROOT/'tools/ppu15/typed_input.cu';lib=work/'libqz15_probe.so'
    includes=[ROOT/'quactlize/include',ROOT/'third_party/actlize/include',
              sdk_path/'include',sdk_path/'targets/x86_64-linux/include']
    commands=[
        [str(sdk_path/'bin/hgcc'),*flags_for_arch(FLAGS,'ppu_15'),
         *['-I'+str(x) for x in includes],'-c',str(source),'-o',str(work/'typed.o')],
        ['g++','-shared','-Wl,-Bsymbolic',str(work/'typed.o'),'-o',str(lib),
         '-L'+str(sdk_path/'lib'),*['-l'+name for name in LIBRARIES]],
    ]
    for index,argv in enumerate(commands): command(argv,work/f'build-{index}.log',os.environ)
    (work/'receipt.json').write_text(json.dumps(dict(commands=commands,library_sha256=sha(lib),
        sources={str(path.relative_to(ROOT)):sha(path) for path in [source,
          ROOT/'tools/ppu15/native.cu',ROOT/'tools/ppu15/typed_input_check.py',
          ROOT/'quactlize/include/actlize_extensions/cutlass/gemm/collective/detail/ppu_decode_input.hpp']}),indent=2)+'\n')
    if a.build_only:
        print('ZW890_TYPED_INPUT_BUILD_ONLY device_validated=0',flush=True);return 0
    sdk,library,device=setup(sdk_path,output)
    run=bind(library,'qz15_typed_input',[C.c_void_p]*3+[C.c_int]*4+[C.c_void_p])
    results=[]
    for height,k in ((16,64),(16,128),(16,256),(64,128)):
        stages=3 if height==16 else 2
        for rows in (1,9,height):
            for bf16 in (0,1):
                logical=np.arange(stages*height*k,dtype='i4').reshape(stages,height,k)
                source=(((logical*37)%4093-2046)/32).astype('f4')
                rounded=bf16_bits(source) if bf16 else source.astype('f2').view('u2')
                # Independent physical expectations. The reference arm below
                # actually uses the native AIU, not either software publisher.
                expected=np.empty_like(rounded);legacy=np.empty_like(rounded)
                for stage in range(stages):
                    for cube in range(k//64):
                        for row in range(height):
                            for col in range(64):
                                value=rounded[stage,row,cube*64+col] if row<rows else 0
                                base=stage*height*k+cube*height*64
                                expected.flat[base+((row*64+col)^((row&7)<<3))]=value
                                part,half,within=col//16,(col%16)//8,col%8
                                start=(((part&1)<<1)+((part&2)>>1))*2
                                old=2*(height*8*part+(row//4)*32+(2*(row%4)+start)%8*4)
                                legacy.flat[base+old+(half^((row//4)&1))*8+within]=value
                with Arena(sdk) as arena:
                    src=arena.upload(source);rounds=arena.upload(rounded)
                    out=arena.buffer(3*rounded.nbytes);stream=arena.stream();out.poison();arena.sync()
                    checked(run(src.ptr,rounds.ptr,out.ptr,bf16,height,k,rows,stream),'typed A delivery')
                    arena.sync();got=np.frombuffer(out.bytes(),'u2').reshape(3,-1)
                    np.testing.assert_array_equal(got[0],expected.ravel(),err_msg='native AIU physical layout')
                    np.testing.assert_array_equal(got[1],legacy.ravel(),err_msg='legacy publisher negative')
                    np.testing.assert_array_equal(got[2],got[0],err_msg='typed publisher vs native AIU')
                    bad=int(np.count_nonzero(got[1]!=got[0]));assert bad==int(np.count_nonzero(legacy!=expected)) and bad>0
                    src.guard();rounds.guard()
                result=dict(height=height,k=k,rows=rows,stages=stages,compute='bf16' if bf16 else 'f16',
                            cells=expected.size,legacy_mismatches=bad,status='PASS')
                results.append(result);print('ZW890_TYPED_INPUT_CASE '+json.dumps(result),flush=True)
    assert len(results)==24
    (output/'summary.json').write_text(json.dumps(dict(status='PASS',device=device,cases=results,
        scope='RAW_BITS_AIUVS_TYPED_PUBLISHER_NOT_TIMING'),indent=2)+'\n')
    print('ZW890_TYPED_INPUT_DONE passed=24 expected=24',flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
