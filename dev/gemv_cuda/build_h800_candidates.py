#!/usr/bin/env python3
"""Small CUDA-only Q4 experiments; reuse the verified control dispatcher.

No production selection, ABI, format, or PPU prebuilt changes. Each output
directory is immutable, and retains generated source, compiler logs and hashes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_cuda.build import digest, q4_large_static_source, q4_small_source, q4_shm_source, replace_once
from dev.gemv_ppu.astage_source import STAGE, stage_body, static_body

N4 = [(c, w) for c in (1, 2, 4, 8) for w in (2, 4, 8, 16)] + [(2, 10), (4, 10), (8, 10)]
SHM = [(c, w) for c in (4, 8, 16, 32) for w in (4, 8)]
WARP = [(c, w) for c in (4, 8, 16) for w in (4, 8)]
WIDE = [(c, w) for c in (1, 2, 4, 8) for w in (2, 4, 5, 8, 10, 16)]


def parse_arm(name):
    flags={}
    for suffix in ("unsigned","int","u32","a8","a4","dm","lb1","s2","skew","av","as","bare","one","cg","fast","rs"):
        flags[suffix]=name.endswith("-"+suffix)
        if flags[suffix]: name=name.removesuffix("-"+suffix)
    return name,flags


def source(arm):
    arm,flags=parse_arm(arm)
    unsigned,vector_a,stage_a,bare,one,cg,fast,scatter=(flags[s] for s in
        ("unsigned","av","as","bare","one","cg","fast","rs"))
    if arm.startswith("affine"):
        unsigned = True  # grouped affine consumes q in [0,15], never q-8.
    original = (ROOT / "quactlize/execution/gemv.cu").read_text()
    if arm == "shared-b":
        body = q4_shm_source(original)
        recipes = SHM
    elif arm in ("n2-global", "n2-static"):
        body = q4_small_source(original)
        recipes = [(c, w) for c in (1, 2, 4, 8) for w in (2, 4, 8, 16)]
        if arm == "n2-static":
            start = body.index("template<int Columns,int Warps,int Type>\n__global__ void kpack_q4_n2_tree")
            end = body.index("\n}\n", start) + 3
            kernel = body[start:end]
            kernel = kernel.replace("template<int Columns,int Warps,int Type>", "template<int Columns,int Warps,int N,int K>")
            kernel = kernel.replace("kpack_q4_n2_tree(qkg_call_v1 c,int split)", "kpack_q4_n2_static(qkg_call_v1 c)")
            kernel = replace_once(kernel, "    constexpr int Workers", "    constexpr int Type=0,split=1;\n    constexpr int Workers")
            kernel = replace_once(kernel, "row=blockIdx.z,partition=blockIdx.y", "row=0,partition=0")
            begin = kernel.index("    int const expert=expert_for(c,row);")
            finish = kernel.index("    auto low=", begin)
            kernel = kernel[:begin] + "    constexpr int expert=0;\n    constexpr int64_t a_base=0;\n" + kernel[finish:]
            kernel = replace_once(kernel, "    for(int g=partition*Workers+worker;g<c.k/32;g+=split*Workers) {",
                                  "    #pragma unroll\n    for(int pass=0;pass<(K/32+Workers-1)/Workers;++pass) {\n        int const g=pass*Workers+worker;\n        if(g>=K/32) continue;")
            kernel = kernel.replace("c.n", "N").replace("c.k", "K")
            body = body[:end] + "\n" + kernel + body[end:]
    else:
        body = q4_large_static_source(original)
        recipes = N4
    if arm == "shared-a":
        for name in ("kpack_q4_small_static", "kpack_q4_large_static"):
            begin, end, before = static_body(body, name)
            body = body[:begin] + stage_body(before) + body[end:]
        start = body.index("__global__ void kpack_q4_n4_coop")
        pos = body.index("    auto low=", start)
        stage = STAGE.replace("K / 8", "c.k / 8")
        body = body[:pos] + stage + body[pos:]
        end = body.index("\n}\n", start)
        sub = body[start:end].replace("aligned_activation<Type>(c.a,", "aligned_activation<Type>(staged_a,")
        body = body[:start] + sub + body[end:]
        body, count = re.subn(r"(kpack_q4_(?:small_static|large_static|n4_coop)<[^\n]+><<<n4_grid,Warps\*32,)0(,stream>>>)",
                             r"\g<1>size_t(c.k)*2\2", body)
        if count != 8: raise ValueError(f"A launch seams: {count}")
    if arm.startswith("warp-"):
        recipes = WARP
        seam = "template<int Columns, int Warps, bool Pair = false> int launch"
        body = replace_once(body, seam, (ROOT / "dev/gemv_cuda/q4_warp_k.cuh").read_text() + "\n" + seam)
    if arm.startswith("n8-") or arm.startswith("n16-"):
        recipes = WIDE
        seam = "template<int Columns, int Warps, bool Pair = false> int launch"
        body = replace_once(body, seam, (ROOT / "dev/gemv_cuda/q4_nwide.cuh").read_text() + "\n" + seam)
    if arm.startswith("ldmatrix-"):
        recipes = [(c, w) for c in (4, 8, 16, 32) for w in (4, 8)]
        seam = "template<int Columns, int Warps, bool Pair = false> int launch"
        body = replace_once(body, seam, (ROOT / "dev/gemv_cuda/q4_ldmatrix.cuh").read_text() + "\n" + seam)
    if arm.startswith("meta"):
        recipes = [(c, w) for c in (1, 2, 4) for w in (2, 4, 5, 8, 10, 16)]
        seam = "template<int Columns, int Warps, bool Pair = false> int launch"
        body = replace_once(body, seam, (ROOT / "dev/gemv_cuda/q4_cooperative_metadata.cuh").read_text() + "\n" + seam)
    if arm.startswith("matrix"):
        recipes = [(8,4),(16,4),(16,8)] if arm.startswith("matrix8") else [(4,4),(8,4),(8,8),(16,4),(16,8),(16,16)]
        seam = "template<int Columns, int Warps, bool Pair = false> int launch"
        matrix=(ROOT / "dev/gemv_cuda/q4_ldmatrix_v2.cuh").read_text()
        if arm=="matrix8-full":
            recipes=[(1,w) for w in (4,8,10,16)]
            matrix=replace_once(matrix,"template<int TileN,int TileK,int Warps,bool Pipeline>",
                                "template<int TileN,int TileK,int Warps,bool Pipeline,int N>")
            matrix=matrix.replace("c.k","TileK").replace("c.n","N")
        body = replace_once(body, seam, matrix + "\n" + seam)
    if arm.startswith("affine"):
        recipes=[(c,w) for c in (1,2,4) for w in (2,4,5,8,10,16)]
        seam = "template<int Columns, int Warps, bool Pair = false> int launch"
        body = replace_once(body, seam, (ROOT / "dev/gemv_cuda/q4_group_affine.cuh").read_text() + "\n" + seam)
        if arm in ("affine-coop","affine-coop2r","affine-coop2r-fp16"):
            recipes=[(c,w) for c in ((4,8) if arm=="affine-coop" else (4,)) for w in (2,4,5,8,10,16)]
            body=replace_once(body,seam,(ROOT/"dev/gemv_cuda/q4_cooperative_affine.cuh").read_text()+"\n"+seam)
            if arm!="affine-coop":
                body=replace_once(body,seam,(ROOT/"dev/gemv_cuda/q4_cooperative_residue2.cuh").read_text()+"\n"+seam)
    start = body.index('extern "C" int QKG_CONCAT(qkg_launch_,QKG_QTYPE)')
    body = body[:start] + '''extern "C" int qkg_launch_12(qkg_call_v1 const&,qkg_config_v1 const&) {return QKG_INVALID;}
extern "C" int qkg_pair_launch_12(qkg_call_v1 const& c,qkg_config_v1 const& f) {
    using namespace QKG_CONCAT(kpack_q,QKG_QTYPE);
    if(c.mode!=QKG_DENSE || c.rows!=1 || c.experts!=1 || c.input_type!=QKG_F16 || f.split!=1 ||
       (uintptr_t(c.a)&15) || (uintptr_t(c.low)&15) || (uintptr_t(c.units)&15)) return QKG_INVALID;
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
'''
    for c, w in recipes:
        if arm.startswith("affine-coop2r"):
            for n,k in ((512,2048),(1024,5120),(4096,2048),(4096,4096),(5120,8192),(8192,5120)):
                body+=f'''    if(f.columns=={c} && f.warps=={w} && c.n=={n} && c.k=={k}) {{
        q4_cooperative_residue2<{w},{n},{k},{str(not arm.endswith('fp16')).lower()}><<<c.n/4,{32*w},0,static_cast<hggcStream_t>(c.stream)>>>(c);
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }}
'''
        elif arm=="affine-coop":
            for n,k in ((512,2048),(1024,5120),(4096,2048),(4096,4096),(5120,8192),(8192,5120)):
                body += f'''    if(f.columns=={c} && f.warps=={w} && c.n=={n} && c.k=={k}) {{
        q4_cooperative_affine<{c},{w},{n},{k}><<<c.n/{c},{32*w},0,static_cast<hggcStream_t>(c.stream)>>>(c);
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }}
'''
        elif arm.startswith("affine"):
            width=int(arm.split("-")[0].removeprefix("affine"))
            early=arm.endswith("early")
            for n,k in ((512,2048),(1024,5120),(4096,2048),(4096,4096),(5120,8192),(8192,5120)):
                body += f'''    if(f.columns=={c} && f.warps=={w} && c.n=={n} && c.k=={k}) {{
        q4_group_affine<{c},{w},{width},{n},{k},{str(early).lower()}><<<c.n/{width*c},{32*w},0,static_cast<hggcStream_t>(c.stream)>>>(c);
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }}
'''
        elif arm=="matrix8-full":
            for n,k in ((512,2048),(1024,5120),(4096,2048),(4096,4096),(8192,5120)):
                body+=f'''    if(f.columns=={c} && f.warps=={w} && c.n=={n} && c.k=={k}) {{
        q4_ldmatrix_v2<8,{k},{w},false,{n}><<<c.n/8,{32*w},size_t(c.k)*2,static_cast<hggcStream_t>(c.stream)>>>(c);
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }}
'''
        elif arm.startswith("matrix"):
            tile_n=8 if arm.startswith("matrix8") else 16
            pipeline=arm.endswith("pipe")
            body += f'''    if(f.columns=={c} && f.warps=={w}) {{
        if(c.k%{c*64}) return QKG_SHAPE;
        q4_ldmatrix_v2<{tile_n},{c*64},{w},{str(pipeline).lower()}><<<c.n/{tile_n},{32*w},size_t(c.k)*2,static_cast<hggcStream_t>(c.stream)>>>(c);
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }}
'''
        elif arm.startswith("meta"):
            stage = arm.endswith("shared")
            width = (2 if arm.startswith("meta2-") else 4 if arm.startswith("meta4-") else 8) * c
            if "static" in arm:
                for n,k in ((512,2048),(1024,5120),(4096,2048),(4096,4096),(5120,8192),(8192,5120)):
                    body += f'''    if(f.columns=={c} && f.warps=={w} && c.n=={n} && c.k=={k}) {{
        q4_cooperative_metadata<{width},{w},{str(stage).lower()},{str(scatter).lower()},{n},{k}><<<c.n/{width},{32*w},{'size_t(c.k)*2' if stage else '0'},static_cast<hggcStream_t>(c.stream)>>>(c);
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }}
'''
                continue
            body += f'''    if(f.columns=={c} && f.warps=={w}) {{
        q4_cooperative_metadata<{width},{w},{str(stage).lower()},{str(scatter).lower()}><<<c.n/{width},{32*w},{'size_t(c.k)*2' if stage else '0'},static_cast<hggcStream_t>(c.stream)>>>(c);
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }}
'''
        elif arm == "n2-static":
            for n, k in ((512, 2048), (1024, 5120), (4096, 2048), (4096, 4096), (5120, 8192), (8192, 5120)):
                body += f'''    if(f.columns=={c} && f.warps=={w} && c.n=={n} && c.k=={k}) {{
        kpack_q4_n2_static<{c},{w},{n},{k}><<<c.n/{2*c},{32*w},0,static_cast<hggcStream_t>(c.stream)>>>(c);
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }}
'''
        elif arm.startswith("ldmatrix-"):
            pipeline = arm == "ldmatrix-pipeline"
            body += f'''    if(f.columns=={c} && f.warps=={w}) {{
        if(c.k%{c*64}) return QKG_SHAPE;
        q4_ldmatrix<{c*64},{w},{str(pipeline).lower()}><<<c.n/16,{32*w},size_t(c.k)*2,static_cast<hggcStream_t>(c.stream)>>>(c,1);
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }}
'''
        elif arm.startswith("n8-") or arm.startswith("n16-"):
            stage = arm.endswith("shared")
            width = 8 if arm.startswith("n8-") else 16
            if "static" in arm:
                for n, k in ((512, 2048), (1024, 5120), (4096, 2048), (4096, 4096), (5120, 8192), (8192, 5120)):
                    body += f'''    if(f.columns=={c} && f.warps=={w} && c.n=={n} && c.k=={k}) {{
        q4_nwide<{c},{w},{width},{str(stage).lower()},{n},{k},{str(scatter).lower()}><<<c.n/{width*c},{32*w},{'size_t(c.k)*2' if stage else '0'},static_cast<hggcStream_t>(c.stream)>>>(c,1);
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }}
'''
                continue
            body += f'''    if(f.columns=={c} && f.warps=={w}) {{
        q4_nwide<{c},{w},{width},{str(stage).lower()},0,0,{str(scatter).lower()}><<<c.n/{width*c},{32*w},{'size_t(c.k)*2' if stage else '0'},static_cast<hggcStream_t>(c.stream)>>>(c,1);
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }}
'''
        elif arm.startswith("warp-"):
            stage = arm != "warp-global"
            sb = arm == "warp-sb"
            body += f'''    if(f.columns=={c} && f.warps=={w}) {{
        q4_warp_k<{c},{w},{str(stage).lower()},{str(sb).lower()}><<<c.n/{2*c},{32*w},{'size_t(c.k)*2' if stage else '0'},static_cast<hggcStream_t>(c.stream)>>>(c,1);
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }}
'''
        else:
            body += f"    if(f.columns=={c} && f.warps=={w}) return launch<{c},{w},true>(c,1);\n"
    body = '#include "' + str(ROOT / "dev/gemv_cuda/q4_warp_reduce_scatter.cuh") + '"\n' + body
    if flags["int"]:
        if not arm.startswith("affine") or arm.startswith("affine-coop"): raise ValueError("integer conversion targets group affine")
        start=body.index("__global__ void q4_group_affine(qkg_call_v1 c)")
        end=body.index("\n}\n",start)+3
        kernel=body[start:end]
        first=kernel.index("                        __half2 q;")
        last=kernel.index("                        dot[p].x",first)
        kernel=kernel[:first]+'''                        uint32_t code_word=words[r][p];
                        float2 v=make_float2(float((code_word>>(4*slot))&15),
                                            float((code_word>>(16+4*slot))&15));
'''+kernel[last:]
        body=body[:start]+kernel+body[end:]
    if flags["a8"]:
        if not arm.startswith("affine") or arm.startswith("affine-coop") or stage_a: raise ValueError("A8 requires direct group A")
        start=body.index("__global__ void q4_group_affine(qkg_call_v1 c)")
        end=body.index("\n}\n",start)+3
        kernel=body[start:end]
        kernel=replace_once(kernel,"        uint4 metadata[Early ? P : 1];",'''        uint4 activation[4];
        #pragma unroll
        for(int slot=0;slot<4;++slot)
            activation[slot]=*reinterpret_cast<uint4 const*>(static_cast<__half const*>(c.a)+g*32+slot*8);
        uint4 metadata[Early ? P : 1];''')
        kernel=replace_once(kernel,"                float4 av=aligned_activation<0>(c.a,g*32+slot*8+half*4);",'''                uint4 packed_a=activation[slot];
                uint32_t lo_bits=half ? packed_a.z : packed_a.x;
                uint32_t hi_bits=half ? packed_a.w : packed_a.y;
                __half2_raw lo,hi;
                lo.x=uint16_t(lo_bits);lo.y=uint16_t(lo_bits>>16);
                hi.x=uint16_t(hi_bits);hi.y=uint16_t(hi_bits>>16);
                float2 av0=__half22float2(__half2(lo)),av1=__half22float2(__half2(hi));
                float4 av=make_float4(av0.x,av0.y,av1.x,av1.y);''')
        body=body[:start]+kernel+body[end:]
    if flags["dm"]:
        if not arm.startswith("affine") or not arm.endswith("early"): raise ValueError("decoded metadata requires early group arm")
        start=body.index("__global__ void q4_group_affine(qkg_call_v1 c)")
        end=body.index("\n}\n",start)+3
        kernel=body[start:end].replace("uint4 metadata[Early ? P : 1];","float2 metadata[Early ? P : 1];")
        kernel=replace_once(kernel,"metadata[p]=aligned_unit(c.units+(size_t(g/8)*N+col+p)*16);",
                            "metadata[p]=q4_affine_header(aligned_unit(c.units+(size_t(g/8)*N+col+p)*16),g&7);")
        first=kernel.index("            uint4 u0,u1;")
        last=kernel.index("            total[p].x",first)
        kernel=kernel[:first]+'''            float2 s0,s1;
            if constexpr(Early) {s0=metadata[2*p];s1=metadata[2*p+1];}
            else {
                s0=q4_affine_header(aligned_unit(c.units+(size_t(g/8)*N+col+2*p)*16),g&7);
                s1=q4_affine_header(aligned_unit(c.units+(size_t(g/8)*N+col+2*p+1)*16),g&7);
            }
'''+kernel[last:]
        body=body[:start]+kernel+body[end:]
    if flags["skew"]:
        if not arm.startswith("affine") or arm.startswith("affine-coop"): raise ValueError("K phase requires full groups")
        start=body.index("__global__ void q4_group_affine(qkg_call_v1 c)")
        end=body.index("\n}\n",start)+3
        kernel=replace_once(body[start:end],"""        int const g=pass*Workers+worker;
        if(g>=K/32) continue;""","""        int const logical_g=pass*Workers+worker;
        if(logical_g>=K/32) continue;
        int g=logical_g+(int(blockIdx.x)%8)*(K/32/8);
        if(g>=K/32) g-=K/32;""")
        body=body[:start]+kernel+body[end:]
    if vector_a:
        if not (arm.startswith("meta") or arm=="affine-coop"): raise ValueError("vector A requires cooperative lanes")
        body='#define Q4_COOPERATIVE_VECTOR_A 1\n#include "'+str(ROOT/"dev/gemv_cuda/q4_warp_activation.cuh")+'"\n'+body
    if stage_a:
        if not arm.startswith("affine") or arm.startswith("affine-coop"): raise ValueError("A stage requires per-thread groups")
        start=body.index("__global__ void q4_group_affine(qkg_call_v1 c)")
        end=body.index("\n}\n",start)+3
        kernel=body[start:end]
        seam="    auto low=reinterpret_cast<uint16_t const*>(c.low);"
        stage='''    extern __shared__ __align__(16) unsigned char act_stage[];
    for(int i=tid;i<K/8;i+=Warps*32)
        reinterpret_cast<uint4*>(act_stage)[i]=reinterpret_cast<uint4 const*>(c.a)[i];
    __syncthreads();
'''
        kernel=replace_once(kernel,seam,stage+seam)
        kernel=kernel.replace("aligned_activation<0>(c.a,","aligned_activation<0>(act_stage,")
        body=body[:start]+kernel+body[end:]
        body,count=re.subn(r"(q4_group_affine<[^\n]+<<<[^\n]+,)0(,static_cast<hggcStream_t>)",r"\1size_t(c.k)*2\2",body)
        if not count: raise ValueError("A stage launch seam missing")
    if bare:
        kernel_name = "q4_cooperative_residue2" if arm.startswith("affine-coop2r") else "q4_cooperative_affine" if arm=="affine-coop" else "q4_group_affine" if arm.startswith("affine") else "q4_cooperative_metadata"
        start=body.index(f"__global__ void {kernel_name}(qkg_call_v1 c)")
        end=body.index("\n}\n",start)+3
        kernel=body[start:end].replace("qkg_call_v1 c","void const* a_ptr,uint8_t const* low_ptr,uint8_t const* units_ptr,float* out_ptr")
        for old,new in (("c.a","a_ptr"),("c.low","low_ptr"),("c.units","units_ptr"),("c.output","out_ptr"),("c.n","N"),("c.k","K")):
            kernel=kernel.replace(old,new)
        body=body[:start]+kernel+body[end:]
        body,count=re.subn(r"("+kernel_name+r"<[^\n]+>>>\()c(\);)",r"\1c.a,c.low,c.units,c.output\2",body)
        if not count: raise ValueError("bare kernel launch seam missing")
    if flags["lb1"]:
        if not arm.startswith("meta"): raise ValueError("launch-bound experiment targets cooperative metadata")
        body=replace_once(body,"__global__ void q4_cooperative_metadata(",
                          "__global__ void __launch_bounds__(Warps*32,1) q4_cooperative_metadata(")
    if flags["s2"]:
        if not (arm.startswith("affine") and not arm.startswith("affine-coop") and bare) or stage_a or flags["skew"]:
            raise ValueError("S2 experiment requires the plain, bare per-thread group kernel")
        start=body.index("__global__ void q4_group_affine(")
        end=body.index("\n}\n",start)+3
        kernel=body[start:end].replace("(K/32+Workers-1)/Workers","(K/64+Workers-1)/Workers")
        kernel=replace_once(kernel,"""        int const g=pass*Workers+worker;
        if(g>=K/32) continue;""","""        int const local_g=pass*Workers+worker;
        if(local_g>=K/64) continue;
        int const g=local_g+int(blockIdx.y)*(K/64);""")
        kernel=replace_once(kernel,"out_ptr[blockIdx.x*TileN+tid]=sum;","out_ptr[blockIdx.y*N+blockIdx.x*TileN+tid]=sum;")
        reducer='''
__global__ void q4_two_part_reduce(float const* partial,float* output,int n) {
    int col=(blockIdx.x*128+threadIdx.x)*4;
    if(col>=n) return;
    float4 a=*reinterpret_cast<float4 const*>(partial+col);
    float4 b=*reinterpret_cast<float4 const*>(partial+n+col);
    *reinterpret_cast<float4*>(output+col)=make_float4(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w);
}
'''
        body=body[:start]+kernel+reducer+body[end:]
        pattern=r"q4_group_affine<([^>\n]+)><<<c.n/(\d+),(\d+),0,static_cast<hggcStream_t>\(c.stream\)>>>\(c.a,c.low,c.units,c.output\);"
        def split_launch(match):
            return f'''q4_group_affine<{match[1]}><<<dim3(c.n/{match[2]},2),{match[3]},0,static_cast<hggcStream_t>(c.stream)>>>(c.a,c.low,c.units,static_cast<float*>(c.workspace));
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        q4_two_part_reduce<<<(c.n+511)/512,128,0,static_cast<hggcStream_t>(c.stream)>>>(static_cast<float const*>(c.workspace),c.output,c.n);'''
        body,count=re.subn(pattern,split_launch,body)
        if count!=len(recipes)*6: raise ValueError("S2 launch count differs")
        body=replace_once(body,"f.split!=1","f.split!=2")
    if one:
        body=replace_once(body,"float2 accum[Pairs][2]{};","float2 accum[Pairs][1]{};")
        body=body.replace("accum[p][r%2]","accum[p][0]")
        for field in ("x","y"):
            body=re.sub(r"accum\[p\]\[0\]\."+field+r"\s*\+\s*accum\[p\]\[1\]\."+field,"accum[p][0]."+field,body)
        before='''                for (int p = 0; p < Pairs; ++p) {
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {'''
        after='''                for (int r = 0; r < 4; ++r) {
                    #pragma unroll
                    for (int p = 0; p < Pairs; ++p) {'''
        body=replace_once(body,before,after)
    if cg:
        if arm.startswith("affine") and not arm.startswith("affine-coop"):
            body,count=re.subn(r"=\s*\*reinterpret_cast<(uint(?:2|4)|uint32_t) const\*>\(ptr\);",
                                r"=__ldcg(reinterpret_cast<\1 const*>(ptr));",body)
            if count!=3: raise ValueError("affine B-only cache load seams missing")
        body, count = re.subn(r"=\s*\*reinterpret_cast<(uint(?:2|4)|uint(?:32|64)_t) const\*>\((low\s*\+[^;\n]*)\);",
                              r"= __ldcg(reinterpret_cast<\1 const*>(\2));", body)
        if not count: raise ValueError("B-only cache load seam missing")
    if unsigned or fast or flags["a4"] or flags["u32"]:
        native = (ROOT / "dev/gemv_cuda/q4_native.cuh").read_text()
        aligned = (ROOT / "dev/gemv_cuda/q4_aligned.cuh").read_text().replace('#include "q4_native.cuh"', "")
        if flags["u32"]:
            start=aligned.index("    uint64_t const run=")
            end=aligned.index("    __half2_raw codes_raw,header_raw;",start)
            aligned=aligned[:start]+"    uint2 code=q4_unit_codes(m,group);\n    unsigned const sc=code.x,mn=code.y;\n"+aligned[end:]
            if "float2 q4_affine_header" in body:
                start=body.index("    uint64_t run=",body.index("float2 q4_affine_header"))
                end=body.index("    return make_float2",start)
                body=body[:start]+"    uint2 code=q4_unit_codes(u,group);\n    float sc=float(code.x),mn=float(code.y);\n"+body[end:]
            native='#include "'+str(ROOT/"dev/gemv_cuda/q4_unit_bits.cuh")+'"\n'+native
        if flags["a4"]:
            aligned=replace_once(aligned,"""        float2 const a=__half22float2(*reinterpret_cast<__half2 const*>(p));
        float2 const b=__half22float2(*reinterpret_cast<__half2 const*>(p+2));""","""        uint2 const packed=*reinterpret_cast<uint2 const*>(p);
        __half2_raw lo,hi;
        lo.x=uint16_t(packed.x);lo.y=uint16_t(packed.x>>16);
        hi.x=uint16_t(packed.y);hi.y=uint16_t(packed.y>>16);
        float2 const a=__half22float2(__half2(lo));
        float2 const b=__half22float2(__half2(hi));""")
        if unsigned:
            native = replace_once(native, "__float2half2_rn(1032.f)", "__float2half2_rn(1024.f)")
            native = replace_once(native, "return {scale, __float2half_rn(__half2float(zero) + 8.f * __half2float(scale))};", "return {scale, zero};")
            aligned = replace_once(aligned, "return {scale,__float2half_rn(__half2float(zero)+8.f*__half2float(scale))};", "return {scale,zero};")
        if fast:
            start = native.index("template<int Slot>\n__device__ __forceinline__ __half2 codes")
            end = native.index("\n}\n", start) + 3
            fast_body = '''template<int Slot>
__device__ __forceinline__ __half2 codes(uint32_t words) {
    constexpr int Pos=(Slot&1)*4;
    uint32_t source=Slot>=2 ? words>>8 : words, bits;
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(bits)
        : "r"(source), "n"(0x000f000fu<<Pos), "n"(0x64006400u));
    __half2_raw raw; raw.x=uint16_t(bits);raw.y=uint16_t(bits>>16);
    return __hfma2(__half2(raw), __float2half2_rn(1.f/(1<<Pos)),
                   __float2half2_rn(-float(1024>>Pos)-BIAS));
}
'''.replace("BIAS", "0.f" if unsigned else "8.f")
            native = native[:start] + fast_body + native[end:]
        for name in ("q4_native.cuh", "q4_aligned.cuh"):
            body = body.replace('#include "' + str(ROOT / "dev/gemv_cuda" / name) + '"', "")
        body = native + "\n" + aligned + "\n" + body
    return body + "    return QKG_INVALID;\n}\n", recipes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cuda", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arms", default="shared-a,shared-b,warp-global,warp-shared,warp-sb")
    p.add_argument("--jobs", type=int, default=5)
    a = p.parse_args()
    a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=False)
    a.control = a.control.resolve(strict=True); a.cuda = a.cuda.resolve(strict=True)
    manifest = json.loads((a.control / "manifest.json").read_text())
    for name, sha in manifest["payloads"].items():
        if digest(a.control / name) != sha: raise ValueError("control payload changed")
    command = next(c["argv"] for c in manifest["commands"] if c["label"] == "gemv")
    flags = command[1:command.index("-c")]
    dispatch_obj=a.control/"dispatch.o"
    dispatch_command=None
    inputs = [Path(__file__), ROOT / "dev/gemv_cuda/q4_warp_k.cuh", ROOT / "dev/gemv_cuda/q4_nwide.cuh", ROOT / "dev/gemv_cuda/q4_ldmatrix.cuh", ROOT / "dev/gemv_cuda/q4_ldmatrix_v2.cuh", ROOT / "dev/gemv_cuda/q4_group_affine.cuh", ROOT / "dev/gemv_cuda/q4_cooperative_affine.cuh", ROOT / "dev/gemv_cuda/q4_cooperative_residue2.cuh", ROOT / "dev/gemv_cuda/q4_cooperative_metadata.cuh", ROOT / "dev/gemv_cuda/q4_warp_reduce_scatter.cuh", ROOT / "dev/gemv_cuda/q4_warp_activation.cuh", ROOT / "dev/gemv_cuda/q4_unit_bits.cuh", ROOT / "dev/gemv_cuda/build.py",
              ROOT / "dev/gemv_ppu/astage_source.py", ROOT / "dev/gemv_cuda/q4_shm_kernel.cuh"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    started = time.monotonic()
    def build(arm):
        text, recipes = source(arm)
        out = a.output / arm; out.mkdir()
        src = out / "kernel.cu"; src.write_text(text)
        obj = out / "kernel.o"; so = out / "libkpack.so"
        commands = [[str(a.cuda / "bin/nvcc"), *flags, "-c", str(src), "-o", str(obj)],
                    [str(a.cuda / "bin/nvcc"), "-shared", "--cudart=shared", "-Xlinker=-Bsymbolic",
                     str(obj), str(dispatch_obj), "-o", str(so)]]
        for i, cmd in enumerate(commands):
            with (out / f"build-{i}.log").open("w") as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc: raise ValueError(f"{arm} build {i}: rc={rc}")
        split=2 if parse_arm(arm)[1]["s2"] else 1
        row = dict(arm=arm, recipes=[[*r, split] for r in recipes], library=str(so), sha256=digest(so),
                   source_sha256=digest(src), commands=commands)
        row["launches_per_call"]=split
        row["weight_arithmetic"]="FP32_GROUP_AFFINE_NOT_PER_WEIGHT_FP16" if arm.startswith("affine") and "fp16" not in arm else "PER_WEIGHT_FP16_AFFINE"
        if parse_arm(arm)[0]=="matrix8-full":
            row["supported_shapes"]=[[1,n,k] for n,k in ((512,2048),(1024,5120),(4096,2048),(4096,4096),(8192,5120))]
        (out / "manifest.json").write_text(json.dumps(row, indent=2) + "\n")
        print(f"Q4_H800_BUILD arm={arm} status=COMPILED", flush=True)
        return row
    arms = a.arms.split(",")
    if len(set(arms)) != len(arms) or not {parse_arm(x)[0] for x in arms} <= {"shared-a", "shared-b", "warp-global", "warp-shared", "warp-sb", "n8-global", "n8-shared", "n8-static-global", "n8-static-shared", "n2-global", "n2-static", "n4-global", "n16-global", "n16-shared", "ldmatrix-single", "ldmatrix-pipeline", "meta-global", "meta-shared", "meta2-global", "meta4-global", "meta-static-global", "meta2-static-global", "matrix8-single", "matrix8-pipe", "matrix8-full", "matrix16-single", "matrix16-pipe", "affine2-early", "affine4-early", "affine8-early", "affine8-late", "affine-coop", "affine-coop2r", "affine-coop2r-fp16"}:
        raise ValueError("unknown or duplicate arm")
    if any(parse_arm(x)[1]["s2"] for x in arms):
        # The new S2 templates support W5/W10. The frozen control dispatcher
        # admitted those only for S1, so build a private dispatcher with the
        # exact extra predicate. The shipping dispatcher is not changed.
        validation=replace_once((a.control/"validation.hpp").read_text(),
            "f.split == 1 && (f.warps == 5 || f.warps == 10)",
            "(f.split == 1 || f.split == 2) && (f.warps == 5 || f.warps == 10)")
        (a.output/"validation.hpp").write_text(validation)
        (a.output/"dispatch.cpp").write_text((a.control/"dispatch.cpp").read_text())
        dispatch_obj=a.output/"dispatch.o"
        original_cmd=next(row["argv"] for row in manifest["commands"] if row["label"]=="dispatch")
        dispatch_command=[original_cmd[0],f"-I{a.output}",*original_cmd[1:original_cmd.index("-c")],
                          "-c",str(a.output/"dispatch.cpp"),"-o",str(dispatch_obj)]
        with (a.output/"dispatch-build.log").open("w") as log:
            subprocess.run(dispatch_command,stdout=log,stderr=subprocess.STDOUT,check=True)
        flags=[f"-I{a.output}",*flags]
    with ThreadPoolExecutor(max_workers=min(len(arms), a.jobs)) as pool:
        results = list(pool.map(build, arms))
    if any(digest(ROOT / p) != h for p, h in hashes.items()): raise ValueError("source changed during compile")
    report = dict(arms=results, control_manifest_sha256=digest(a.control / "manifest.json"), source_hashes=hashes,
                  dispatcher_sha256=digest(dispatch_obj), dispatcher_command=dispatch_command,
                  seconds=time.monotonic()-started, device_validated=False, production_changed=False)
    (a.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Q4_H800_BUILD status=COMPILED seconds={report['seconds']:.1f}", flush=True)


if __name__ == "__main__": main()
