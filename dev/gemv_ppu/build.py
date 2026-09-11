#!/usr/bin/env python3
"""Compile-only PPU Q4 SIMT comparison. No CUDA emulation or production edits."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_cuda.build import q4_large_static_source, q4_wide_validation, replace_once, digest
from dev.gemv_cuda.build_xplane_accum import fp32_source
from quactlize.runtime.compiler import FLAGS, LIBRARIES


def ppu_api(source):
    # SDK-native runtime and half intrinsics. Neither the CUDA compatibility
    # headers nor CUDA libraries enter this build.
    return re.sub(r"\bcuda(?=[A-Z_])", "hggc", source)


def q4_dispatch(source):
    guard="if (!c || !f || !out) return QKG_INVALID;"
    if source.count(guard)!=2:
        raise ValueError("query guard seams changed")
    source=source.replace(guard,guard+"\n    if(c->qtype!=12) return QKG_FORMAT;")
    start=source.index("    using Launch = int(*)(qkg_call_v1 const&, qkg_config_v1 const&);")
    end=source.index("\n}",start)
    return source[:start]+"    return (pair ? qkg_pair_launch_12 : qkg_launch_12)(*c,*f);"+source[end:]


def candidate_validation(source):
    source=q4_wide_validation(source)
    source=replace_once(source,"(f.columns == 4 || f.columns == 8)",
                        "(f.columns == 1 || f.columns == 2 || f.columns == 4 || f.columns == 8)")
    return replace_once(source,"!(pair && f.warps == 2)",
        "!(pair && (f.warps == 2 || (c.qtype == 12 && (f.warps == 16 || "
        "(f.split == 1 && (f.warps == 5 || f.warps == 10))))))")


def build(sdk, output, jobs):
    sdk,output=Path(sdk).resolve(strict=True),Path(output).resolve()
    if jobs<1: raise ValueError("jobs must be positive")
    sdk_files=[sdk/"bin/hgcc",sdk/"bin/hgobjdump",*[sdk/"lib"/f"lib{x}.so" for x in LIBRARIES]]
    if not all(p.is_file() for p in sdk_files): raise ValueError("incomplete PPU SDK")
    output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ)
    env["PATH"]=str(sdk/"bin")+os.pathsep+env.get("PATH","")
    env["LD_LIBRARY_PATH"]=str(sdk/"lib")+os.pathsep+env.get("LD_LIBRARY_PATH","")
    includes=[ROOT/"quactlize/execution",ROOT/"quactlize/include",ROOT/"benchmarks",
              ROOT/"third_party/actlize/include",ROOT/"third_party/actlize/tools/util/include"]
    paths=set()
    for d in includes+[ROOT/"dev/gemv_cuda",ROOT/"dev/gemv_ppu"]:
        paths.update(p for p in d.rglob("*") if p.is_file() and p.suffix in (".hpp",".h",".cuh",".cu",".cpp",".inc",".py"))
    source_hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}
    original=(ROOT/"quactlize/execution/gemv.cu").read_text()
    validation=(ROOT/"quactlize/execution/validation.hpp").read_text()
    dispatch=q4_dispatch((ROOT/"quactlize/execution/dispatch.cpp").read_text())
    units=[]
    for arm in ("old","new"):
        out=output/arm; out.mkdir()
        body=original if arm=="old" else q4_large_static_source(original)
        if arm=="new":
            for name in ("q4_native.cuh","q4_aligned.cuh"):
                (out/name).write_text(ppu_api((ROOT/"dev/gemv_cuda"/name).read_text()))
                body=body.replace(str(ROOT/"dev/gemv_cuda"/name),str(out/name))
        body=body.replace("QKG_CONCAT(kpack_q,QKG_QTYPE)",f"QKG_CONCAT(kpack_ppu_{arm}_q,QKG_QTYPE)")
        (out/"gemv.cu").write_text(ppu_api(body))
        (out/"validation.hpp").write_text(validation if arm=="old" else candidate_validation(validation))
        (out/"dispatch.cpp").write_text(dispatch)
        units.extend((f"{arm}-{name}",out/f"{name}.{suffix}",out/f"{name}.o",[out], ["-DQKG_QTYPE=12"] if name=="gemv" else [])
                     for name,suffix in (("gemv","cu"),("dispatch","cpp")))
    out=output/"xplane";out.mkdir()
    for name in ("q4k_pdf_ab_fixture.hpp","q4k_pdf_reconstruction.cuh"):
        (out/name).write_text(ppu_api((ROOT/"benchmarks"/name).read_text()))
    (out/"gguf_bc_q4_gemv.hpp").write_text(fp32_source((ROOT/"quactlize/include/gguf_bc_q4_gemv.hpp").read_text()))
    (out/"xplane.cu").write_text(ppu_api((ROOT/"dev/gemv_cuda/xplane_compare.cu").read_text()))
    units.append(("xplane",out/"xplane.cu",out/"xplane.o",[out],[]))
    units.append(("probe",ROOT/"dev/gemv_ppu/probe.cu",output/"probe.o",[],[]))
    commands=[]
    def run(label,command):
        commands.append(dict(label=label,argv=list(map(str,command))))
        with (output/(label+".log")).open("w") as log:
            proc=subprocess.run(list(map(str,command)),env=env,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode: raise RuntimeError(f"{label} rc={proc.returncode}; log={output/(label+'.log')}")
        print(f"Q4_PPU_BUILD phase={label} status=PASS",flush=True)
    def compile_one(unit):
        label,source,obj,extra,defs=unit
        run(label,[sdk/"bin/hgcc",*FLAGS,*defs,*[f"-I{p}" for p in extra+includes],"-c",source,"-o",obj])
    started=time.monotonic()
    failures=[]
    with ThreadPoolExecutor(max_workers=min(jobs,len(units))) as pool:
        for future in as_completed([pool.submit(compile_one,u) for u in units]):
            try:future.result()
            except Exception as exc:failures.append(str(exc))
    if failures: raise RuntimeError("; ".join(failures))
    payloads={}
    for arm in ("old","new","xplane"):
        objects=[u[2] for u in units if u[0]==arm or u[0].startswith(arm+"-")]
        objects.append(output/"probe.o")
        library=output/f"libq4_ppu_{arm}.so"
        run("link-"+arm,["g++","-shared","-Wl,-Bsymbolic",*objects,"-o",library,
                          f"-L{sdk/'lib'}",*[f"-l{x}" for x in LIBRARIES]])
        # hgobjdump device disassembly, not host ELF symbol presence alone.
        run("isa-"+arm,[sdk/"bin/hgobjdump","--dump-isa",library])
        isa=(output/("isa-"+arm+".log")).read_text()
        if "q4_ppu_marker" not in isa: raise ValueError("PPU marker missing from device image: "+arm)
        if arm=="new" and "kpack_q4_large_static" not in isa:
            raise ValueError("large static specialization missing from device image")
        payloads[arm]=dict(file=library.name,sha256=digest(library),isa_sha256=digest(output/("isa-"+arm+".log")))
    if any(digest(ROOT/p)!=h for p,h in source_hashes.items()): raise ValueError("sources changed during build")
    generated={str(p.relative_to(output)):digest(p) for p in output.rglob("*") if p.suffix in (".cu",".cpp",".cuh",".hpp")}
    manifest=dict(schema="quactlize.q4-simt-ppu-comparison.v1",device_validated=False,
        source_hashes=source_hashes,generated_hashes=generated,commands=commands,payloads=payloads,
        runtime={p.name:digest(p) for p in sdk_files[2:]},compiler_sha256=digest(sdk_files[0]),
        inspector_sha256=digest(sdk_files[1]),seconds=time.monotonic()-started,
        shapes=[[1,n,k] for n,k in ((512,2048),(1024,5120),(4096,2048),(4096,4096),(5120,8192),(8192,5120))],
        precision="FP16_A_AND_WEIGHT_FP32_DOT_AND_REDUCTION",production_changed=False)
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Q4_PPU_BUILD status=COMPILED device_validated=0 seconds={manifest['seconds']:.1f}",flush=True)
    return manifest


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--jobs",type=int,default=6)
    a=p.parse_args();build(a.sdk,a.output,a.jobs)
