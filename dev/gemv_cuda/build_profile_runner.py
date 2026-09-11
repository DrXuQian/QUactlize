#!/usr/bin/env python3
"""Build the existing CUDA profile harness with complete rotating-cache cycles.

Keep the historical source frozen for its published receipts. No kernel,
fixture, precision, warmup, sample count or profile boundary changes.
"""
import argparse
import json
from pathlib import Path
import subprocess

if __package__:
    from .build import digest, replace_once
else:
    from build import digest, replace_once

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def whole_ring_source(source):
    old='    int const copies=mode=="warm" ? 1 : int((9*uint64_t(prop.l2CacheSize)+4*one-1)/(4*one));'
    new='    int const copies=mode=="warm" ? 1 : std::max(2,int((9*uint64_t(prop.l2CacheSize)+4*one-1)/(4*one)));\n'
    new+='    int const calls=std::max(2,(32+copies-1)/copies)*copies;'
    source=replace_once(source,old,new)
    source=replace_once(source,'        int const calls=std::max(32,2*copies);\n','')
    source=replace_once(source,'copies=%d mode=%s','copies=%d calls_per_graph=%d mode=%s')
    return replace_once(source,'prop.l2CacheSize,copies,','prop.l2CacheSize,copies,calls,')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    source=HERE/'profile_xplane.cu'
    generated=args.output/'profile_xplane_ring.cu'
    generated.write_text(whole_ring_source(source.read_text()))
    binary=args.output/'profile_xplane'
    command=[str(args.cuda/'bin/nvcc'),'-std=c++17','-arch=sm_120','-O3','-lineinfo',
             '--cudart=shared',f'-I{HERE}',f'-I{ROOT}',f'-I{ROOT / "quactlize/include"}',
             str(generated),'-ldl','-o',str(binary)]
    with (args.output/'build.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    (args.output/'manifest.json').write_text(json.dumps(dict(command=command,
        source_sha256=digest(source),generator_sha256=digest(Path(__file__)),
        generated_sha256=digest(generated),binary_sha256=digest(binary),
        scope='CUDA_WHOLE_RING_PROFILE_RUNNER_NOT_DEVICE_ADMISSION'),indent=2)+'\n')
    print('Q4_RING_RUNNER COMPILED binary='+str(binary),flush=True)


if __name__=='__main__':main()
