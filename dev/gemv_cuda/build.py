#!/usr/bin/env python3
"""Development-only CUDA builds of production SIMT GEMV and reader experiments."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("production schedule seam changed: " + before[:80])
    return source.replace(before, after)


def grid_schedule(source):
    source = replace_once(
        source,
        """    int const tiles = c.n / Columns;
    int const tile = int(blockIdx.x) % tiles;
    int const outer = int(blockIdx.x) / tiles;
    int const partition = outer % split, row = outer / split;""",
        """    int const tile = int(blockIdx.x);
    int const partition = int(blockIdx.y), row = int(blockIdx.z);""",
    )
    source = replace_once(
        source,
        """    unsigned const grid = unsigned(uint64_t(c.rows) * split * (c.n / Columns));""",
        """    if (c.rows > 65535) return QKG_INVALID;
    dim3 const grid(c.n / Columns, split, c.rows);""",
    )
    source = replace_once(
        source,
        """    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < int64_t(c.rows) * c.n; i += int64_t(gridDim.x) * blockDim.x) {
        int64_t const row = i / c.n, col = i % c.n;""",
        """    int64_t const row = blockIdx.y;
    int const col = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col < c.n) {""",
    )
    return replace_once(
        source,
        """        uint64_t const count = (uint64_t(c.rows)*c.n+255)/256;
        kpack_gemv_reduce<<<unsigned(count < 65535 ? count : 65535),256,0,stream>>>(c,split);""",
        """        dim3 const reduce_grid((c.n + 127) / 128, c.rows);
        kpack_gemv_reduce<<<reduce_grid,128,0,stream>>>(c,split);""",
    )


def replace_dot(original, filename, *, content=None):
    start = "template<class Reader>\n__device__ __forceinline__ float pair_dot("
    end = "template<int Columns, int Warps, bool Pair = false>\n__global__ void kpack_gemv("
    if original.count(start) != 1 or original.count(end) != 1:
        raise ValueError("production dot boundaries changed")
    first, last = original.index(start), original.index(end)
    if first >= last:
        raise ValueError("reversed dot boundaries")
    return (
        original[:first]
        + ((HERE / filename).read_text() if content is None else content)
        + "\n"
        + original[last:]
    )


def affine_source(original):
    return replace_dot(original, "affine_dot.cuh")


def q4_n2_source(original):
    generic = replace_once((HERE / "n2_dot.cuh").read_text(),
                           "float2 pair_dot(", "float2 generic_n2_pair_dot(")
    return '#include "' + str(HERE / "q4_native.cuh") + '"\n' + n2_schedule(
        replace_dot(original, "q4_n2_dot.cuh",
                    content=generic + "\n" + (HERE / "q4_n2_dot.cuh").read_text()))


def q4_wide_source(original):
    source = q4_n2_source(original)
    seam = "    if (f.columns==16) {"
    extra = "#if QKG_QTYPE == 12\n"
    for columns in (4, 8):
        extra += f"    if (f.columns=={columns}) {{\n"
        extra += f"        if (f.warps==2) return launch<{columns},2,true>(c,f.split);\n"
        extra += f"        return f.warps==4 ? launch<{columns},4,true>(c,f.split) : launch<{columns},8,true>(c,f.split);\n"
        extra += "    }\n"
    extra += "#endif\n" + seam
    return replace_once(source, seam, extra)


def q4_wide_validation(original):
    return replace_once(original, "(f.columns != 16 && f.columns != 32) ||",
        "(f.columns != 16 && f.columns != 32 &&\n"
        "         !(pair && c.qtype == 12 && (f.columns == 4 || f.columns == 8))) ||")


def q4_shared_source(original):
    source = q4_wide_source(original)
    seam = "auto dot=pair_dot<R>(c,a_base,low,high,units,col,worker,Workers,partition,split);"
    stage = """#if QKG_QTYPE == 12
        extern __shared__ __half staged_a[];
        for (int j=threadIdx.x; j<c.k; j+=Warps*32)
            staged_a[j] = c.input_type == QKG_F32
                ? __float2half_rn(static_cast<float const*>(c.a)[a_base+j])
                : static_cast<__half const*>(c.a)[a_base+j];
        __syncthreads();
        qkg_call_v1 cached = c;
        cached.a = staged_a;
        cached.input_type = QKG_F16;
        auto dot=pair_dot<R>(cached,0,low,high,units,col,worker,Workers,partition,split);
#else
        """ + seam + "\n#endif"
    source = replace_once(source, seam, stage)
    return replace_once(source, "kpack_gemv<Columns,Warps,Pair><<<grid,Warps*32,0,stream>>>(c,split);",
        "size_t const smem = Pair && QKG_QTYPE == 12 ? size_t(c.k)*2 : 0;\n"
        "    kpack_gemv<Columns,Warps,Pair><<<grid,Warps*32,smem,stream>>>(c,split);")


def q4_half_source(original):
    source = q4_wide_source(original)
    source = '#include "' + str(HERE / "q4_half_dot.cuh") + '"\n' + source
    source = replace_once(source, "float x[4]{}, y[4]{};", "float sum_x=0.f, sum_y=0.f;")
    old = """        // Constant slots keep paired-nibble extraction as a word operation.
        dot_slot<0>(c.a, a_base + g*32, c.input_type, words, scale, zero, x, y);
        dot_slot<1>(c.a, a_base + g*32, c.input_type, words, scale, zero, x, y);
        dot_slot<2>(c.a, a_base + g*32, c.input_type, words, scale, zero, x, y);
        dot_slot<3>(c.a, a_base + g*32, c.input_type, words, scale, zero, x, y);"""
    new = """        __half2 partial[8];
        #pragma unroll
        for (int j=0;j<8;++j) partial[j]=__float2half2_rn(0.f);
        half_dot_slot<0>(c.a, a_base+g*32, c.input_type, words, scale, zero, partial);
        half_dot_slot<1>(c.a, a_base+g*32, c.input_type, words, scale, zero, partial);
        half_dot_slot<2>(c.a, a_base+g*32, c.input_type, words, scale, zero, partial);
        half_dot_slot<3>(c.a, a_base+g*32, c.input_type, words, scale, zero, partial);
        float2 const first_half=__half22float2(__hadd2(__hadd2(partial[0],partial[1]),__hadd2(partial[2],partial[3])));
        float2 const second_half=__half22float2(__hadd2(__hadd2(partial[4],partial[5]),__hadd2(partial[6],partial[7])));
        sum_x += first_half.x+second_half.x; sum_y += first_half.y+second_half.y;"""
    source=replace_once(source, old, new)
    return replace_once(source, "return make_float2((x[0]+x[1])+(x[2]+x[3]), (y[0]+y[1])+(y[2]+y[3]));",
                        "return make_float2(sum_x,sum_y);")


def q4_shm_source(original):
    source=q4_wide_source(original)
    seam="template<int Columns, int Warps, bool Pair = false> int launch"
    source=replace_once(source,seam,"#if QKG_QTYPE == 12\n"+
        (HERE/"q4_shm_kernel.cuh").read_text()+"\n#endif\n"+seam)
    return replace_once(source,"    kpack_gemv<Columns,Warps,Pair><<<grid,Warps*32,0,stream>>>(c,split);",
        "#if QKG_QTYPE == 12\n"
        "    if constexpr (Pair) kpack_q4_shm<Columns,Warps><<<grid,Warps*32,0,stream>>>(c,split);\n"
        "    else\n#endif\n"
        "    kpack_gemv<Columns,Warps,Pair><<<grid,Warps*32,0,stream>>>(c,split);")


def q4_grid_source(original):
    source=q4_wide_source(original)
    source=replace_once(source,"""    int const tiles = c.n / (Columns * (Pair ? 2 : 1));
    int const tile = int(blockIdx.x) % tiles;
    int const outer = int(blockIdx.x) / tiles;
    int const partition = outer % split, row = outer / split;""",
        """    int const tile=blockIdx.x, partition=blockIdx.y, row=blockIdx.z;""")
    return replace_once(source,
        "unsigned const grid = unsigned(uint64_t(c.rows) * split * (c.n / (Columns*(Pair ? 2 : 1))));",
        "if (c.rows>65535) return QKG_INVALID;\n"
        "    dim3 const grid(c.n/(Columns*(Pair ? 2 : 1)),split,c.rows);")


def q4_aligned_source(original):
    source=q4_grid_source(original)
    native=(HERE/"q4_n2_dot.cuh").read_text()
    aligned=replace_once(native,"template<class Reader>","template<class Reader,int Type>")
    aligned=replace_once(aligned,"float2 pair_dot(","float2 aligned_q4_pair_dot(")
    aligned=aligned.replace("load_unit(","aligned_unit(").replace("n2_word(","aligned_word(")
    for slot in range(4):
        aligned=replace_once(aligned,f"dot_slot<{slot}>(c.a, a_base + g*32, c.input_type,",
                             f"aligned_dot_slot<{slot},Type>(c.a, a_base + g*32,")
    fallback=replace_once(native,"float2 pair_dot(","float2 fallback_q4_pair_dot(")
    wrapper="""
template<class Reader>
__device__ __forceinline__ float2 pair_dot(qkg_call_v1 const& c,int64_t a_base,
    uint16_t const* low,uint16_t const* high,uint8_t const* units,
    int col,int worker,int workers,int partition,int split) {
#if QKG_QTYPE == 12
    if (!(uintptr_t(low)&3) && !(uintptr_t(units)&15)) {
        if (c.input_type==0 && !((uintptr_t(c.a)+2*a_base)&3))
            return aligned_q4_pair_dot<Reader,0>(c,a_base,low,high,units,col,worker,workers,partition,split);
        if (c.input_type==1 && !((uintptr_t(c.a)+4*a_base)&15))
            return aligned_q4_pair_dot<Reader,1>(c,a_base,low,high,units,col,worker,workers,partition,split);
    }
#endif
    return fallback_q4_pair_dot<Reader>(c,a_base,low,high,units,col,worker,workers,partition,split);
}
"""
    return '#include "'+str(HERE/"q4_aligned.cuh")+'"\n'+replace_once(source,native,aligned+fallback+wrapper)


def q4_metadata_source(original):
    source=q4_aligned_source(original)
    if source.count("scale_zero(aligned_unit(") != 2:
        raise ValueError("aligned metadata seams changed")
    return source.replace("scale_zero(aligned_unit(","aligned_scale_zero(aligned_unit(")


def q4_n4_source(original):
    source=q4_metadata_source(original)
    extra="#if QKG_QTYPE == 12\n"
    for c in (1,2):
        extra+=f"    if (f.columns=={c}) {{\n"
        extra+=f"        if (f.warps==2) return launch<{c},2,true>(c,f.split);\n"
        extra+=f"        return f.warps==4 ? launch<{c},4,true>(c,f.split) : launch<{c},8,true>(c,f.split);\n    }}\n"
    extra+="#endif\n    if (f.columns==16) {"
    source=replace_once(source,"    if (f.columns==16) {",extra)
    seam="template<int Columns, int Warps, bool Pair = false> int launch"
    source=replace_once(source,seam,"#if QKG_QTYPE == 12\n"+
        (HERE/"q4_n4_kernel.cuh").read_text()+"\n#endif\n"+seam)
    seam="    kpack_gemv<Columns,Warps,Pair><<<grid,Warps*32,0,stream>>>(c,split);"
    dispatch="""#if QKG_QTYPE == 12
    bool const aligned_b=!(uintptr_t(c.low)&7) && !(uintptr_t(c.units)&15);
    bool const aligned_a=c.input_type==0
        ? !(uintptr_t(c.a)&3) && !(c.a_row_stride&1) && !(c.a_token_stride&1)
        : !(uintptr_t(c.a)&15) && !(c.a_row_stride&3) && !(c.a_token_stride&3);
    if (Pair && aligned_b && aligned_a) {
        dim3 const n4_grid(c.n/(Columns*4),split,c.rows);
        if(c.input_type==0) kpack_q4_n4<Columns,Warps,0><<<n4_grid,Warps*32,0,stream>>>(c,split);
        else kpack_q4_n4<Columns,Warps,1><<<n4_grid,Warps*32,0,stream>>>(c,split);
    } else
#endif
"""
    return replace_once(source,seam,dispatch+seam)


def q4_unsigned_source(original):
    # Arithmetic-matched SIMT control: q is unsigned in both resident formats.
    # The signed tensor-core reader's +8 zero compensation is not needed here.
    native=(HERE/"q4_native.cuh").read_text()
    native=replace_once(native,"__float2half2_rn(1032.f)","__float2half2_rn(1024.f)")
    native=replace_once(native,"return {scale, __float2half_rn(__half2float(zero) + 8.f * __half2float(scale))};",
                        "return {scale, zero};")
    aligned=(HERE/"q4_aligned.cuh").read_text().replace('#include "q4_native.cuh"',"")
    aligned=replace_once(aligned,"return {scale,__float2half_rn(__half2float(zero)+8.f*__half2float(scale))};",
                         "return {scale,zero};")
    source=q4_n4_source(original)
    for name in ("q4_native.cuh","q4_aligned.cuh"):
        source=source.replace('#include "'+str(HERE/name)+'"',"")
    return (native+"\n"+aligned+"\n"+source).replace("quactlize::dev::q4_native","quactlize::dev::q4_unsigned_native")


def q4_tree_source(original):
    source=q4_n4_source(original)
    kernel=(HERE/"q4_n4_kernel.cuh").read_text()
    start=kernel.index("    constexpr int T=Warps*32;")
    tail=kernel[start:kernel.rfind("}")]
    reduced="""    float v0=(x[0]+x[1])+(x[2]+x[3]);
    float v1=(y[0]+y[1])+(y[2]+y[3]);
    float v2=(z[0]+z[1])+(z[2]+z[3]);
    float v3=(t[0]+t[1])+(t[2]+t[3]);
    #pragma unroll
    for(int distance=Columns;distance<32;distance*=2) {
        v0+=__shfl_xor_sync(0xffffffffu,v0,distance);
        v1+=__shfl_xor_sync(0xffffffffu,v1,distance);
        v2+=__shfl_xor_sync(0xffffffffu,v2,distance);
        v3+=__shfl_xor_sync(0xffffffffu,v3,distance);
    }
    constexpr int T=Warps*Columns;
    __shared__ float partial[4*T];
    if((tid&31)<Columns) {
        int i=(tid/32)*Columns+(tid&31);
        partial[i]=v0;partial[T+i]=v1;partial[2*T+i]=v2;partial[3*T+i]=v3;
    }
    __syncthreads();
    if(tid<Columns) {
        v0=0;v1=0;v2=0;v3=0;
        #pragma unroll
        for(int w=0;w<Warps;++w) {
            int i=w*Columns+tid;
            v0+=partial[i];v1+=partial[T+i];v2+=partial[2*T+i];v3+=partial[3*T+i];
        }
        auto dst=split==1 ? c.output+int64_t(row)*c.out_row_stride :
            static_cast<float*>(c.workspace)+(int64_t(row)*split+partition)*c.n;
        dst[col]=v0;dst[col+1]=v1;dst[col+2]=v2;dst[col+3]=v3;
    }
"""
    source=replace_once(source,tail,reduced)
    seam="    if (hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;"
    extra=seam+"\n#if QKG_QTYPE == 12\n    if (f.warps==16) {\n"
    for c in (1,2,4,8,16,32):
        extra+=f"        if(f.columns=={c}) return launch<{c},16,true>(c,f.split);\n"
    extra+="    }\n#endif"
    return replace_once(source,seam,extra)


def q4_small_source(original):
    source=q4_tree_source(original)
    begin="template<int Columns,int Warps,int Type>\n__global__ void kpack_q4_n4"
    end="\n#endif\ntemplate<int Columns, int Warps, bool Pair = false> int launch"
    if source.count(begin)!=1 or source.count(end)!=1:
        raise ValueError("N4 kernel boundary changed")
    first,last=source.index(begin),source.index(end)
    source=source[:first]+(HERE/"q4_n2_tree_kernel.cuh").read_text()+source[last:]
    source=replace_once(source,"!(uintptr_t(c.low)&7)","!(uintptr_t(c.low)&3)")
    source=replace_once(source,"n4_grid(c.n/(Columns*4),split,c.rows)",
                         "n4_grid(c.n/(Columns*2),split,c.rows)")
    return source.replace("kpack_q4_n4<", "kpack_q4_n2_tree<")


def q4_coop_source(original):
    source=q4_tree_source(original)
    start=source.index("    constexpr int T=Warps*Columns;")
    end=source.index("\n}\n\n#endif\ntemplate<int Columns",start)
    source=source[:start]+"""    __shared__ float4 partial[Warps*Columns];
    if((tid&31)<Columns)
        partial[(tid/32)*Columns+(tid&31)]=make_float4(v0,v1,v2,v3);
    __syncthreads();
    if(tid<32) {
        float4 sum=make_float4(0,0,0,0);
        #pragma unroll
        for(int w=tid/Columns;w<Warps;w+=32/Columns) {
            float4 value=partial[w*Columns+tid%Columns];
            sum.x+=value.x;sum.y+=value.y;sum.z+=value.z;sum.w+=value.w;
        }
        #pragma unroll
        for(int distance=Columns;distance<32;distance*=2) {
            sum.x+=__shfl_xor_sync(0xffffffffu,sum.x,distance);
            sum.y+=__shfl_xor_sync(0xffffffffu,sum.y,distance);
            sum.z+=__shfl_xor_sync(0xffffffffu,sum.z,distance);
            sum.w+=__shfl_xor_sync(0xffffffffu,sum.w,distance);
        }
        if(tid<Columns) {
            auto dst=split==1 ? c.output+int64_t(row)*c.out_row_stride :
                static_cast<float*>(c.workspace)+(int64_t(row)*split+partition)*c.n;
            dst[col]=sum.x;dst[col+1]=sum.y;dst[col+2]=sum.z;dst[col+3]=sum.w;
        }
    }"""+source[end:]
    return source.replace("kpack_q4_n4", "kpack_q4_n4_coop")


def q4_static_source(original):
    source=q4_coop_source(original)
    begin="template<int Columns,int Warps,int Type>\n__global__ void kpack_q4_n4_coop"
    end="\n#endif\ntemplate<int Columns, int Warps, bool Pair = false> int launch"
    first,last=source.index(begin),source.index(end)
    kernel=source[first:last]
    static=replace_once(kernel,"template<int Columns,int Warps,int Type>",
                        "template<int Columns,int Warps,int N,int K>")
    static=replace_once(static,"kpack_q4_n4_coop(qkg_call_v1 c,int split)",
                        "kpack_q4_small_static(qkg_call_v1 c)")
    static=replace_once(static,"using namespace quactlize::dev::q4_native;",
        "using namespace quactlize::dev::q4_native;\n    constexpr int Type=0,split=1;")
    static=replace_once(static,"row=blockIdx.z,partition=blockIdx.y;","row=0,partition=0;")
    a=static.index("    int const expert=expert_for(c,row);")
    b=static.index("    auto low=",a)
    static=static[:a]+"    constexpr int expert=0;\n    constexpr int64_t a_base=0;\n"+static[b:]
    static=replace_once(static,"for (int g=partition*Workers+worker;g<c.k/32;g+=split*Workers) {",
        "#pragma unroll\n    for(int pass=0;pass<(K/32+Workers-1)/Workers;++pass) {\n"
        "        int const g=pass*Workers+worker;\n        if(g>=K/32) continue;")
    static=static.replace("c.n","N").replace("c.k","K")
    source=source[:last]+"\n"+static+source[last:]
    launch="""        if(c.input_type==0 && c.mode==QKG_DENSE && c.rows==1 && split==1) {
            if(c.n==512 && c.k==2048) {
                kpack_q4_small_static<Columns,Warps,512,2048><<<n4_grid,Warps*32,0,stream>>>(c);
                return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
            }
            if(c.n==1024 && c.k==5120) {
                kpack_q4_small_static<Columns,Warps,1024,5120><<<n4_grid,Warps*32,0,stream>>>(c);
                return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
            }
        }
"""
    seam="        if(c.input_type==0) kpack_q4_n4_coop<"
    return replace_once(source,seam,launch+seam)


def q4_balanced_source(original):
    source=q4_static_source(original)
    extra="    if (f.split==1) {\n"
    for w in (5,10):
        extra+=f"        if(f.warps=={w}) {{\n"
        for c in (1,2,4,8,16,32):
            extra+=f"            if(f.columns=={c}) return launch<{c},{w},true>(c,f.split);\n"
        extra+="        }\n"
    extra+="    }\n"
    return replace_once(source,"    if (f.warps==16) {",extra+"    if (f.warps==16) {")


def q4_warp_source(original):
    source=q4_small_source(original)
    kernel=(HERE/"q4_n2_tree_kernel.cuh").read_text()
    warp=replace_once(kernel,"constexpr int Workers=Warps*32/Columns;",
                       "constexpr int Workers=32/Columns;")
    warp=replace_once(warp,"worker=tid/Columns;","worker=(tid&31)/Columns;")
    warp=replace_once(warp,"int const col=(blockIdx.x*Columns+tid%Columns)*2;",
        "int const col=(blockIdx.x*Warps*Columns+(tid/32)*Columns+tid%Columns)*2;\n"
        "    // N is a multiple of 256 and 2*Columns divides 64: a tail\n"
        "    // removes complete warps, never a lane participating in shuffle.\n"
        "    if(col>=c.n) return;")
    warp=replace_once(warp,"        if(tid<Columns)","        if((tid&31)<Columns)")
    first=warp.index("    constexpr int T=Warps*Columns;")
    warp=warp[:first]+"""    if((tid&31)<Columns) {
        auto dst=split==1 ? c.output+int64_t(row)*c.out_row_stride :
            static_cast<float*>(c.workspace)+(int64_t(row)*split+partition)*c.n;
        dst[col]=v0;dst[col+1]=v1;
    }
}
"""
    source=replace_once(source,kernel,warp)
    source=replace_once(source,"n4_grid(c.n/(Columns*2),split,c.rows)",
        "n4_grid((c.n+Columns*2*Warps-1)/(Columns*2*Warps),split,c.rows)")
    return source.replace("kpack_q4_n2_tree", "kpack_q4_n2_warp")


def n2_schedule(source):
    # Only Pair=true changes ownership. Scalar dispatch remains the exact
    # production control. The API's N divisibility also admits twice Columns.
    source = replace_once(source, "int const tiles = c.n / Columns;",
                          "int const tiles = c.n / (Columns * (Pair ? 2 : 1));")
    source = replace_once(source, "int const col = tile * Columns + threadIdx.x % Columns;",
                          "int const col = (tile * Columns + threadIdx.x % Columns) * (Pair ? 2 : 1);")
    source = replace_once(source, "float accum = 0.f;", "float accum = 0.f, accum2 = 0.f;")
    source = replace_once(source,
        "accum=pair_dot<R>(c,a_base,low,high,units,col,worker,Workers,partition,split);",
        "auto dot=pair_dot<R>(c,a_base,low,high,units,col,worker,Workers,partition,split);\n"
        "        accum=dot.x; accum2=dot.y;")
    source = replace_once(source, "__shared__ float partial[Warps * 32];",
                          "__shared__ float partial[Warps * 32 * (Pair ? 2 : 1)];")
    source = replace_once(source, "partial[threadIdx.x] = accum;",
                          "partial[threadIdx.x] = accum;\n"
                          "    if constexpr (Pair) partial[Warps * 32 + threadIdx.x] = accum2;")
    source = replace_once(source,
        "else static_cast<float*>(c.workspace)[(int64_t(row) * split + partition) * c.n + col] = value;",
        "else static_cast<float*>(c.workspace)[(int64_t(row) * split + partition) * c.n + col] = value;\n"
        "        if constexpr (Pair) {\n"
        "            float second=0.f;\n"
        "            #pragma unroll\n"
        "            for (int w=0;w<Workers;++w) second+=partial[Warps*32+w*Columns+threadIdx.x];\n"
        "            if (split==1) c.output[int64_t(row)*c.out_row_stride+col+1]=second;\n"
        "            else static_cast<float*>(c.workspace)[(int64_t(row)*split+partition)*c.n+col+1]=second;\n"
        "        }")
    # Preserve observable NaN writes for invalid expert IDs in both columns.
    source = replace_once(source, "        return;\n    }\n    int64_t const a_base",
        "        if constexpr (Pair) if (threadIdx.x<Columns) {\n"
        "            if (split==1) c.output[int64_t(row)*c.out_row_stride+col+1]=__int_as_float(0x7fc00000);\n"
        "            else static_cast<float*>(c.workspace)[(int64_t(row)*split+partition)*c.n+col+1]=__int_as_float(0x7fc00000);\n"
        "        }\n        return;\n    }\n    int64_t const a_base")
    return replace_once(source,
        "unsigned const grid = unsigned(uint64_t(c.rows) * split * (c.n / Columns));",
        "unsigned const grid = unsigned(uint64_t(c.rows) * split * (c.n / (Columns*(Pair ? 2 : 1))));")


def build(cuda, output, jobs, reader="production", schedule="production", max_q4_registers=None):
    if reader == "cuda-half2" and schedule != "production":
        raise ValueError("the half2 include wrapper requires the production schedule")
    nvcc = cuda / "bin/nvcc"
    version = subprocess.check_output([nvcc, "--version"], text=True)
    if "NVIDIA" not in version or "HGG" in version:
        raise ValueError("requires a complete NVIDIA CUDA SDK")
    output.mkdir(parents=True, exist_ok=False)
    commands = {}

    def run(name, command):
        commands[name] = [str(x) for x in command]
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name} failed: {output / f'{name}.log'}")
        print(f"CUDA_GEMV_BUILD phase={name} status=PASS", flush=True)

    started = time.monotonic()
    run(
        "probe-build",
        [nvcc, "-arch=sm_120", "-O2", HERE / "probe.cu", "-o", output / "probe"],
    )
    run("probe-run", [output / "probe"])
    includes = [
        HERE / "compat",
        ROOT / "quactlize/execution",
        ROOT / "quactlize/include",
        ROOT / "third_party/actlize/include",
        ROOT / "third_party/actlize/tools/util/include",
        ROOT / "third_party/actlize/examples/common",
    ]
    flags = [
        "-std=c++17",
        "-arch=sm_120",
        "-O3",
        "-lineinfo",
        "--expt-relaxed-constexpr",
        "-Xcompiler=-fPIC",
        "-Xptxas=-v",
        "-DCUTLASS_USE_PACKED_TUPLE=1",
        "-DCUTE_USE_PACKED_TUPLE=1",
        "-include",
        str(HERE / "compat/compiler_bridge.h"),
        *[f"-I{x}" for x in includes],
    ]
    sources = sorted(
        {
            p
            for d in includes + [HERE, ROOT / "quactlize/runtime"]
            for p in d.rglob("*")
            if p.is_file() and p.suffix in (".h", ".hpp", ".cuh", ".cu", ".cpp", ".inc")
        }
    )
    sources += [ROOT / "tests/kpack_grouped_postops_layout.hpp"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    gemv_source = (
        ROOT / "quactlize/execution/gemv.cu"
        if reader == "production"
        else HERE / "gemv_half2.cu"
    )
    original = gemv_source.read_text()
    if reader == "cuda-affine":
        original = replace_dot((ROOT / "quactlize/execution/gemv.cu").read_text(), "affine_dot.cuh")
    if reader == "cuda-vector":
        original = "#include <cuda_fp16.h>\n" + replace_dot(
            (ROOT / "quactlize/execution/gemv.cu").read_text(), "vector_dot.cuh")
    if reader == "cuda-n2":
        original = "#include <cuda_fp16.h>\n" + n2_schedule(replace_dot(
            (ROOT / "quactlize/execution/gemv.cu").read_text(), "n2_dot.cuh"))
        if schedule != "production":
            raise ValueError("N2 owns its launch mapping; use the production schedule option")
    q4_readers = ("cuda-q4-n2", "cuda-q4-n2-wide", "cuda-q4-n2-shared", "cuda-q4-n2-half-control", "cuda-q4-shm-fp32", "cuda-q4-n2-grid", "cuda-q4-n2-aligned", "cuda-q4-n2-metadata", "cuda-q4-n4", "cuda-q4-n4-unsigned", "cuda-q4-n4-tree", "cuda-q4-n2-tree", "cuda-q4-n2-warp", "cuda-q4-n4-coop", "cuda-q4-small-static", "cuda-q4-small-balanced")
    n4_readers = ("cuda-q4-n4","cuda-q4-n4-unsigned","cuda-q4-n4-tree","cuda-q4-n4-coop","cuda-q4-small-static","cuda-q4-small-balanced")
    narrow_readers=n4_readers+("cuda-q4-n2-tree","cuda-q4-n2-warp")
    tree_readers=("cuda-q4-n4-tree","cuda-q4-n2-tree","cuda-q4-n2-warp","cuda-q4-n4-coop","cuda-q4-small-static","cuda-q4-small-balanced")
    wide_readers = q4_readers[1:]
    n2_readers = ("cuda-n2",) + q4_readers
    if reader in q4_readers:
        if schedule != "production":
            raise ValueError("Q4 N2 owns its launch mapping; use the production schedule option")
        make_source = {"cuda-q4-n2": q4_n2_source, "cuda-q4-n2-wide": q4_wide_source,
                       "cuda-q4-n2-shared": q4_shared_source,
                       "cuda-q4-n2-half-control": q4_half_source,
                       "cuda-q4-shm-fp32": q4_shm_source,
                       "cuda-q4-n2-grid": q4_grid_source,
                       "cuda-q4-n2-aligned": q4_aligned_source,
                       "cuda-q4-n2-metadata": q4_metadata_source,
                       "cuda-q4-n4": q4_n4_source,
                       "cuda-q4-n4-unsigned": q4_unsigned_source,
                       "cuda-q4-n4-tree": q4_tree_source,
                       "cuda-q4-n2-tree": q4_small_source,
                       "cuda-q4-n2-warp": q4_warp_source,
                       "cuda-q4-n4-coop": q4_coop_source,
                       "cuda-q4-small-static": q4_static_source,
                       "cuda-q4-small-balanced": q4_balanced_source}[reader]
        original = make_source((ROOT / "quactlize/execution/gemv.cu").read_text())
    if schedule == "cuda-grid":
        original = grid_schedule(original)
    if reader in ("cuda-affine", "cuda-vector") + n2_readers or schedule != "production":
        gemv_source = output / "gemv_experiment.cu"
        gemv_source.write_text(original)
    q8_source = gemv_source if reader in ("cuda-vector",) + n2_readers else ROOT / "quactlize/execution/gemv.cu"
    dispatch_source = ROOT / "quactlize/execution/dispatch.cpp"
    if reader in wide_readers:
        validation=q4_wide_validation((ROOT / "quactlize/execution/validation.hpp").read_text())
        if reader in narrow_readers:
            validation=replace_once(validation,"(f.columns == 4 || f.columns == 8)",
                                    "(f.columns == 1 || f.columns == 2 || f.columns == 4 || f.columns == 8)")
        if reader in tree_readers:
            validation=replace_once(validation,"!(pair && f.warps == 2)",
                                    "!(pair && (f.warps == 2 || (c.qtype == 12 && f.warps == 16)))")
        if reader=="cuda-q4-small-balanced":
            validation=replace_once(validation,"c.qtype == 12 && f.warps == 16",
                "c.qtype == 12 && (f.warps == 16 || (f.split == 1 && (f.warps == 5 || f.warps == 10)))")
        (output / "validation.hpp").write_text(validation)
        (output / "dispatch.cpp").write_text(dispatch_source.read_text())
        dispatch_source = output / "dispatch.cpp"
    entries = [("q8", q8_source, ["-DQKG_QTYPE=8"])] + [
        (f"q{q}", gemv_source, [f"-DQKG_QTYPE={q}"]) for q in range(10, 15)] + [
        ("dispatch", dispatch_source, []),
        ("reducer", HERE / "reducer.cu", []),
        (
            "direct",
            HERE / "direct.cu",
            ["-DPPU_PACKED_SCALE=1", "-DPPU_PACKED_FORMAT=0"],
        ),
    ]

    def compile_one(item):
        name, source, defs = item
        if name=="q12" and max_q4_registers is not None:
            defs=defs+[f"-maxrregcount={max_q4_registers}"]
        run(
            name,
            [
                nvcc,
                *flags,
                *defs,
                "-x",
                "cu",
                "-c",
                source,
                "-o",
                output / f"{name}.o",
            ],
        )

    with ThreadPoolExecutor(max_workers=min(jobs, len(entries))) as pool:
        list(pool.map(compile_one, entries))
    library = output / "libkpack_gemv_cuda.so"
    run(
        "link",
        [
            nvcc,
            "-shared",
            "--cudart=shared",
            "-Xlinker=-Bsymbolic",
            *[output / f"{name}.o" for name, _, _ in entries],
            "-o",
            library,
        ],
    )
    if any(digest(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError("source changed during compilation")
    manifest = dict(
        schema="quactlize.dev-gemv-cuda.v1",
        compiler=version,
        compiler_sha256=digest(nvcc),
        source_hashes=hashes,
        builder_sha256=digest(Path(__file__)),
        generated_source_sha256=digest(gemv_source),
        commands=commands,
        library=library.name,
        library_sha256=digest(library),
        seconds=time.monotonic() - started,
        production_kernel="quactlize/execution/gemv.cu",
        pair_affine={
            "production": "CUDA_SCALAR_FALLBACK_NOT_PPU_F16X2_ASM",
            "cuda-half2": "CUDA_HFMA2_EXPERIMENT",
            "cuda-affine": "FP32_GROUP_AFFINE_NO_METADATA_WEIGHT_OR_A_FP16_ROUNDING",
            "cuda-vector": "FP16_PAIR_AFFINE_VECTOR_A_FOUR_FP32_DOT_CHAINS",
            "cuda-n2": "FP16_PAIR_AFFINE_TWO_COLUMNS_PER_THREAD_VECTOR_A",
            "cuda-q4-n2": "FP16_PAIR_AFFINE_Q4_NATIVE_WORDS_SAME_N2_FP32_ORDER",
            "cuda-q4-n2-wide": "FP16_PAIR_AFFINE_Q4_NATIVE_WORDS_WIDE_N_DOMAIN_FP32_DOT",
            "cuda-q4-n2-shared": "FP16_A_CTA_SHARED_Q4_NATIVE_WORDS_WIDE_N_DOMAIN_FP32_DOT",
            "cuda-q4-n2-half-control": "ARITHMETIC_CONTROL_FP16_GROUP_PARTIAL_FP32_CROSS_GROUP_NOT_PRODUCTION",
            "cuda-q4-shm-fp32": "KPACK_B16_ON_CHIP_TRANSPOSE_FP16_AFFINE_FP32_DOT",
            "cuda-q4-n2-grid": "N2_FP32_DOT_SAME_OWNERSHIP_THREE_DIMENSIONAL_GRID",
            "cuda-q4-n2-aligned": "N2_FP32_DOT_THREE_DIMENSIONAL_GRID_HOIST_TYPE_ALIGNMENT_GUARDS",
            "cuda-q4-n2-metadata": "N2_FP32_DOT_ALIGNED_PACKED_HALF_METADATA_PRODUCTS",
            "cuda-q4-n4": "N4_B64_LOAD_SAME_N2_FP32_DOT_ORDER_GUARDED_N2_FALLBACK",
            "cuda-q4-n4-unsigned": "N4_UNSIGNED_Q4_ZMUL0_HALF_WEIGHT_FP32_DOT_XPLANE_MATCHED_CONTROL",
            "cuda-q4-n4-tree": "N4_FP32_WARP_CTA_TREE_REDUCTION_DIFFERENT_ADD_ORDER_W16_ADDED",
            "cuda-q4-n2-tree": "N2_SMALL_N_FP32_WARP_CTA_TREE_REDUCTION_W16_ADDED",
            "cuda-q4-n2-warp": "N2_WARP_OWNED_OUTPUT_FP32_REDUCTION_NO_CTA_BARRIER",
            "cuda-q4-n4-coop": "N4_FP32_WARP_COOPERATIVE_CTA_REDUCTION_FLOAT4_SHARED",
            "cuda-q4-small-static": "N4_FP32_S1_F16_M1_KN_STATIC_SMALL_SHAPES_COOP_FALLBACK",
            "cuda-q4-small-balanced": "N4_FP32_SMALL_STATIC_S1_W5_W10_BALANCED_K_GROUPS",
        }[reader],
        reader=reader,
        pair_column_values_per_thread=2 if reader in n2_readers else 1,
        q4_aligned_column_values_per_thread=4 if reader in n4_readers else 2,
        q4_n_positions=[1,2,4,8,16,32] if reader in narrow_readers else [4,8,16,32] if reader in wide_readers else [16,32],
        q4_warps=[2,4,8,16] if reader in tree_readers else [2,4,8],
        q4_s1_extra_warps=[5,10] if reader=="cuda-q4-small-balanced" else [],
        query_source_sha256=digest(output / "validation.hpp") if reader in wide_readers else digest(ROOT / "quactlize/execution/validation.hpp"),
        q8_reader=("cuda-n2" if reader in q4_readers else reader)
                  if reader in ("cuda-vector",) + n2_readers else "production",
        schedule=schedule,
        max_q4_registers=max_q4_registers,
        ppu_admission=False,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"CUDA_GEMV_BUILD status=COMPLETE seconds={manifest['seconds']:.3f} library={library}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda-12.8"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--max-q4-registers",type=int,
                        help="development-only q12 register-allocation control")
    parser.add_argument(
        "--reader",
        choices=("production", "cuda-half2", "cuda-affine", "cuda-vector", "cuda-n2", "cuda-q4-n2", "cuda-q4-n2-wide", "cuda-q4-n2-shared", "cuda-q4-n2-half-control", "cuda-q4-shm-fp32", "cuda-q4-n2-grid", "cuda-q4-n2-aligned", "cuda-q4-n2-metadata", "cuda-q4-n4", "cuda-q4-n4-unsigned", "cuda-q4-n4-tree", "cuda-q4-n2-tree", "cuda-q4-n2-warp", "cuda-q4-n4-coop", "cuda-q4-small-static", "cuda-q4-small-balanced"),
        default="production",
    )
    parser.add_argument(
        "--schedule", choices=("production", "cuda-grid"), default="production"
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.max_q4_registers is not None and not 16<=args.max_q4_registers<=128:
        parser.error("--max-q4-registers must be between 16 and 128")
    build(
        args.cuda.resolve(),
        args.output.resolve(),
        args.jobs,
        args.reader,
        args.schedule,
        args.max_q4_registers,
    )
