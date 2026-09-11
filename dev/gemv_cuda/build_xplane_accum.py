#!/usr/bin/env python3
"""Build the historical Xplane reader and an FP32-dot arithmetic control.

No producer, address, launch shape, metadata or weight-half boundary changes.
The second library changes intra-group dot accumulation only; it is not the
historical performance winner and must never be labelled as such.
"""
import argparse
import json
from pathlib import Path
import subprocess

if __package__:
    from .build import replace_once, digest
else:
    from build import replace_once, digest

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def fp32_source(original):
    src=original
    for j in range(4):
        src=replace_once(src,f"half2& d{j}",f"float2& d{j}")
        src=replace_once(src,
            f"d{j} = __hfma2(__hfma2(q.pair[{j}], scale, zero), activation_pair<Word, {j}>(logical), d{j});",
            f"d{j} = fp32_dot(__hfma2(q.pair[{j}], scale, zero), activation_pair<Word, {j}>(logical), d{j});")
        src=replace_once(src,f"half2 d{j} = __float2half2_rn(0.0f);",f"float2 d{j} = make_float2(0.f,0.f);")
    seam="template <int Word>\n__device__ __forceinline__ void dot_word"
    helper="""__device__ __forceinline__ float2 fp32_dot(half2 w, half2 a, float2 sum) {
  float2 const wf=__half22float2(w), af=__half22float2(a);
  return make_float2(fmaf(wf.x,af.x,sum.x),fmaf(wf.y,af.y,sum.y));
}

"""
    src=replace_once(src,seam,helper+seam)
    return replace_once(src,
        """      half2 const sum = __hadd2(__hadd2(d0, d1), __hadd2(d2, d3));
      acc[ii] += __half2float(__low2half(sum)) + __half2float(__high2half(sum));""",
        """      float2 const sum = make_float2((d0.x+d1.x)+(d2.x+d3.x), (d0.y+d1.y)+(d2.y+d3.y));
      acc[ii] += sum.x + sum.y;""")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cuda",type=Path,default=Path("/usr/local/cuda-12.8"))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    original=ROOT/"quactlize/include/gguf_bc_q4_gemv.hpp"
    wrapper=HERE/"xplane_compare.cu"
    source=original.read_text()
    records=[]
    for arm,body in (("fp16-original",source),("fp32-control",fp32_source(source))):
        out=a.output/arm
        out.mkdir()
        header=out/original.name
        header.write_text(body)
        includes=[out,HERE/"compat",ROOT/"quactlize/include",ROOT/"benchmarks",
                  ROOT/"third_party/actlize/include",ROOT/"third_party/actlize/tools/util/include"]
        library=out/"libq4_xplane.so"
        cmd=[str(a.cuda/"bin/nvcc"),"-std=c++17","-arch=sm_120","-O3","-lineinfo",
             "--expt-relaxed-constexpr","--shared","--cudart=shared","-Xcompiler=-fPIC",
             "-DCUTLASS_USE_PACKED_TUPLE=1","-DCUTE_USE_PACKED_TUPLE=1",
             *[f"-I{x}" for x in includes],
             str(wrapper),"-o",str(library)]
        with (out/"build.log").open("w") as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        records.append(dict(arm=arm,library=str(library),library_sha256=digest(library),
                            generated_header_sha256=digest(header),command=cmd))
        print("XPLANE_ARITHMETIC_BUILD "+arm+" PASS",flush=True)
    (a.output/"manifest.json").write_text(json.dumps(dict(original_sha256=digest(original),
        wrapper_sha256=digest(wrapper),builder_sha256=digest(Path(__file__)),
        scope="CUDA_ARITHMETIC_CONTROL_NOT_PPU",arms=records),indent=2)+"\n")


if __name__ == "__main__":
    main()
