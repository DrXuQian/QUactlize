#!/usr/bin/env python3
"""Compile production MoE helper bodies and their bounded standalone proof."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quactlize.runtime.compiler import FLAGS,LIBRARIES,sha
from tools.build_kpack_decode_io import native_simt_source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sdk=args.sdk.resolve(strict=True);out=args.output.resolve()
    if not (sdk/'bin/hgcc').is_file():raise ValueError('SDK lacks hgcc')
    out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'tests/kpack_moe_chain_cuda.cu'
    inputs=[source,ROOT/'quactlize/runtime/moe_chain.cuh',ROOT/'quactlize/runtime/indexed.cuh',
            ROOT/'quactlize/runtime/moe_protocol.h',ROOT/'quactlize/integrations/llama/indexed.h',
            ROOT/'quactlize/integrations/llama/router.cuh',Path(__file__).resolve(),
            ROOT/'tools/build_kpack_decode_io.py']
    hashes={str(f.relative_to(ROOT)):sha(f) for f in inputs}
    generated=out/'moe-helper-proof.cu';generated.write_text(native_simt_source(source.read_text()))
    binary=out/'moe-helper-proof'
    command=[str(sdk/'bin/hgcc'),*FLAGS,f'-I{ROOT}',f'-I{ROOT/"quactlize/include"}',
             f'-I{ROOT/"third_party/actlize/include"}',f'-I{ROOT/"third_party/actlize/tools/util/include"}',
             str(generated),'-o',str(binary),'-Wl,--allow-shlib-undefined',f'-L{sdk/"lib"}',
             *[f'-l{x}' for x in LIBRARIES]]
    env=dict(os.environ);env['LD_LIBRARY_PATH']=str(sdk/'lib')+':'+env.get('LD_LIBRARY_PATH','')
    with (out/'build.log').open('w') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    if any(sha(ROOT/f)!=v for f,v in hashes.items()):raise ValueError('source changed during compilation')
    manifest=dict(schema='quactlize.moe-helper-proof.v1',binary=binary.name,sha256=sha(binary),
                  source_hashes=hashes,compiler_sha256=sha(sdk/'bin/hgcc'),command=command,
                  device_validated=False,scope='HELPERS_ONLY_NO_GEMM_OR_MODEL_ADMISSION',
                  perf_baseline='UNFUSED_STAGE_REFERENCE_NOT_LLAMA_BINARY')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'MOE_HELPER_BUILD PASS binary={binary} device_admission=PENDING')


if __name__=='__main__':main()
