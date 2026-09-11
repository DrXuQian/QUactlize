#!/usr/bin/env python3
"""Compile a bounded CUDA Q4 replay, including the supplied FP32 raw reader.

No AIU emulation. Does not modify the frozen CUDA or PPU control generators.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_cuda.build import digest, replace_once, q4_large_static_source
from dev.gemv_cuda.build_xplane_accum import fp32_source
from dev.gemv_cuda.build_profile_runner import whole_ring_source
from dev.gemv_ppu.build import q4_dispatch, candidate_validation
from dev.gemv_ppu.bload_source import REFERENCE_SHA256, reference_fp32, reference_wrapper


def runner_source(source):
    src=whole_ring_source(source)
    src=replace_once(src,'    auto pack=xp.symbol<Pack>("q4_xplane_pack");',
        '    using Rawrun=int(*)(int,int,int,int,void const*,void const*,void*,void*);\n'
        '    auto rawrun=xp.symbol<Rawrun>("q4_ref_fp32_run");\n'
        '    auto pack=xp.symbol<Pack>("q4_xplane_pack");')
    src=replace_once(src,'require(arm=="xplane" || arm=="kpack","arm");',
        'require(arm=="xplane" || arm=="kpack" || arm=="raw-reference","arm");')
    src=replace_once(src,'    Device low(h.lengths[1]*copies),',
        '    uint64_t const low_stride=arm=="raw-reference" ? h.lengths[0] : h.lengths[1];\n'
        '    Device low(low_stride*copies),')
    src=replace_once(src,'low.put(arm=="xplane" ? xlow.data() : data[1].data(),h.lengths[1],i*h.lengths[1]);',
        'low.put(arm=="raw-reference" ? data[0].data() : arm=="xplane" ? xlow.data() : data[1].data(),low_stride,i*low_stride);')
    src=replace_once(src,'copy*h.lengths[1];','copy*low_stride;')
    src=replace_once(src,'        int rc=arm=="xplane" ? xrun(',
        '        int rc=arm=="raw-reference" ? rawrun(cfg.columns,cfg.warps,h.n,h.k,a.ptr,call.low,call.output,stream)\n'
        '             : arm=="xplane" ? xrun(')
    src=replace_once(src,'    auto validate=[&] {','    auto validate=[&](bool positive=true) {')
    src=replace_once(src,'        require(worst<.005,"independent GGUF oracle");',
        '        if(positive) require(worst<.005,"independent GGUF oracle");')
    src=replace_once(src,'    run(0);double err=validate();', '''    run(0);double err=validate();
    std::vector<uint8_t> original=arm=="raw-reference" ? data[0] : arm=="xplane" ? xlow : data[1];
    auto fault=original;
    if(arm=="raw-reference") {
        for(size_t block=0;block<fault.size()/144;++block)
            std::fill(fault.begin()+block*144+16,fault.begin()+(block+1)*144,uint8_t(0));
    } else std::fill(fault.begin(),fault.end(),uint8_t(0));
    low.put(fault.data(),fault.size());
    check(cudaDeviceSynchronize());
    run(0);require(validate(false)>.005,"zero-code negative escaped");
    low.put(original.data(),original.size());
    check(cudaDeviceSynchronize());''')
    return src


def build(cuda, output, arch, jobs):
    if arch not in (80,90,120) or jobs<1: raise ValueError("unsupported bounded CUDA architecture/jobs")
    cuda=cuda.resolve(strict=True);output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    raw=ROOT/"dev/gemv_ppu/reference/gemv_ref.cuh"
    if digest(raw)!=REFERENCE_SHA256:raise ValueError("supplied reference source differs")
    inputs=[Path(__file__),raw,ROOT/"dev/gemv_ppu/bload_source.py",ROOT/"dev/gemv_ppu/bload_contract.hpp"]
    control=json.loads((ROOT/"prebuilt/ppu0010/q4-simt-ab-v1/manifest.json").read_text())
    hashes={p:digest(ROOT/p) for p in control["source_hashes"]}
    hashes.update({str(p.relative_to(ROOT)):digest(p) for p in inputs})
    (output/"gemv.cu").write_text(q4_large_static_source((ROOT/"quactlize/execution/gemv.cu").read_text()))
    (output/"dispatch.cpp").write_text(q4_dispatch((ROOT/"quactlize/execution/dispatch.cpp").read_text()))
    (output/"validation.hpp").write_text(candidate_validation((ROOT/"quactlize/execution/validation.hpp").read_text()))
    (output/"gguf_bc_q4_gemv.hpp").write_text(fp32_source((ROOT/"quactlize/include/gguf_bc_q4_gemv.hpp").read_text()))
    (output/"gemv_ref_fp32.cuh").write_text(reference_fp32(raw.read_text(),ppu=False))
    (output/"reference.cu").write_text(reference_wrapper(ppu=False))
    (output/"profile.cu").write_text(runner_source((ROOT/"dev/gemv_cuda/profile_xplane.cu").read_text()))
    includes=[output,ROOT/"dev/gemv_cuda/compat",ROOT/"dev/gemv_cuda",ROOT/"dev/gemv_ppu",ROOT,
              ROOT/"quactlize/execution",ROOT/"quactlize/include",ROOT/"benchmarks",
              ROOT/"third_party/actlize/include",ROOT/"third_party/actlize/tools/util/include"]
    flags=["-std=c++17",f"-arch=sm_{arch}","-O3","-lineinfo","--expt-relaxed-constexpr","-Xcompiler=-fPIC",
        "-Xptxas=-v","-DCUTLASS_USE_PACKED_TUPLE=1","-DCUTE_USE_PACKED_TUPLE=1",*[f"-I{x}" for x in includes]]
    commands=[];started=time.monotonic()
    def run(label,command):
        command=list(map(str,command));commands.append(dict(label=label,argv=command))
        with (output/(label+".log")).open("w") as log:
            rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        if rc:raise ValueError(f"{label} rc={rc}; log={output/(label+'.log')}")
        print(f"Q4_FIXED_CUDA_BUILD phase={label} status=PASS",flush=True)
    units=[("gemv",output/"gemv.cu",["-DQKG_QTYPE=12"]),("dispatch",output/"dispatch.cpp",[]),
           ("xplane",ROOT/"dev/gemv_cuda/xplane_compare.cu",[]),("reference",output/"reference.cu",[])]
    def compile_one(u):
        label,src,defs=u
        bridge=["-include",str(ROOT/"dev/gemv_cuda/compat/compiler_bridge.h")] if label in ("gemv","dispatch") else []
        run(label,[cuda/"bin/nvcc",*flags,*bridge,*defs,"-x","cu","-c",src,"-o",output/(label+".o")])
    with ThreadPoolExecutor(max_workers=min(jobs,len(units))) as pool:list(pool.map(compile_one,units))
    for name,objs in (("kpack",("gemv","dispatch")),("reference",("xplane","reference"))):
        run("link-"+name,[cuda/"bin/nvcc","-shared","--cudart=shared","-Xlinker=-Bsymbolic",
            *[output/(o+".o") for o in objs],"-o",output/("lib"+name+".so")])
    run("runner",[cuda/"bin/nvcc",*flags,"--cudart=shared",output/"profile.cu","-ldl","-o",output/"profile"])
    if any(digest(ROOT/p)!=sha for p,sha in hashes.items()):raise ValueError("source changed during build")
    manifest=dict(arch=arch,source_hashes=hashes,commands=commands,reference_original_sha256=REFERENCE_SHA256,
        precision="FP16_A_FP32_DOT_AND_REDUCTION_AND_OUTPUT",aiu_emulated=False,
        payloads={p.name:digest(p) for p in (output/"libkpack.so",output/"libreference.so",output/"profile")},
        generated={p.name:digest(p) for p in output.iterdir() if p.suffix in (".cu",".cuh",".cpp",".hpp")},
        compiler=subprocess.check_output([cuda/"bin/nvcc","--version"],text=True),seconds=time.monotonic()-started)
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Q4_FIXED_CUDA_BUILD status=COMPILED seconds={manifest['seconds']:.1f}",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cuda",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--arch",type=int,default=90);p.add_argument("--jobs",type=int,default=4)
    a=p.parse_args();build(a.cuda,a.output,a.arch,a.jobs)
