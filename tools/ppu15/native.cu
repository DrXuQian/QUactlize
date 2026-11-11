// PPU1.5 bring-up probes only. Not a production collective or selector.
#include <hggc_runtime.h>
#include "ppu_include.hpp"
#include <cstdio>
#include <cstring>

#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ != 150
#error "Build this probe with -arch=ppu_15"
#endif

using namespace cute;

extern "C" int qz15_device(char* name, char* pci, int64_t* values) {
    int count=0, dev=0, driver=0, runtime=0;
    hggcDeviceProp p{};
    auto rc=hggcGetDeviceCount(&count); if(rc!=hggcSuccess) return int(rc);
    if(count!=1) return -1;
    rc=hggcGetDevice(&dev); if(rc!=hggcSuccess) return int(rc);
    rc=hggcGetDeviceProperties(&p,dev); if(rc!=hggcSuccess) return int(rc);
    rc=hggcDeviceGetPCIBusId(pci,64,dev); if(rc!=hggcSuccess) return int(rc);
    rc=hggcDriverGetVersion(&driver); if(rc!=hggcSuccess) return int(rc);
    rc=hggcRuntimeGetVersion(&runtime); if(rc!=hggcSuccess) return int(rc);
    std::snprintf(name,128,"%s",p.name);
    int64_t data[]={dev,p.major,p.minor,p.multiProcessorCount,p.warpSize,
        p.maxThreadsPerBlock,p.maxThreadsPerMultiProcessor,p.sharedMemPerBlock,
        p.sharedMemPerBlockOptin,p.sharedMemPerMultiprocessor,p.regsPerBlock,
        p.regsPerMultiprocessor,driver,runtime};
    std::memcpy(values,data,sizeof(data));
    return 0;
}

template<bool BF16> using NativeAtom=std::conditional_t<BF16,
    PPU0015_16x16x16_F32BF16BF16F32_TN,PPU0015_16x16x16_F32F16F16F32_TN>;

template<bool BF16>
__global__ void native_mma(uint16_t const* a,uint16_t const* b,float const* c,
                           float* d,int m,int n,int k) {
    using Atom=NativeAtom<BF16>;
    using Traits=MMA_Traits<Atom>;
    int lane=threadIdx.x, row=int(blockIdx.y)*16, col=int(blockIdx.x)*16;
    float acc[8];
    #pragma unroll
    for(int v=0;v<8;++v) {
        int p=typename Traits::CLayout{}(make_coord(lane,v));
        acc[v]=(row+p%16<m && col+p/16<n) ? c[(row+p%16)*n+col+p/16] : 0.f;
    }
    for(int start=0;start<k;start+=16) {
        uint32_t av[4],bv[4];
        #pragma unroll
        for(int r=0;r<4;++r) {
            av[r]=bv[r]=0;
            #pragma unroll
            for(int h=0;h<2;++h) {
                int ap=typename Traits::ALayout{}(make_coord(lane,2*r+h));
                int bp=typename Traits::BLayout{}(make_coord(lane,2*r+h));
                uint32_t x=row+ap%16<m ? a[(row+ap%16)*k+start+ap/16] : 0;
                uint32_t y=col+bp%16<n ? b[(col+bp%16)*k+start+bp/16] : 0;
                av[r]|=x<<(16*h); bv[r]|=y<<(16*h);
            }
        }
        Atom::fma(acc[0],acc[1],acc[2],acc[3],acc[4],acc[5],acc[6],acc[7],
                  av[0],av[1],av[2],av[3],bv[0],bv[1],bv[2],bv[3],
                  acc[0],acc[1],acc[2],acc[3],acc[4],acc[5],acc[6],acc[7]);
    }
    #pragma unroll
    for(int v=0;v<8;++v) {
        int p=typename Traits::CLayout{}(make_coord(lane,v));
        if(row+p%16<m && col+p/16<n) d[(row+p%16)*n+col+p/16]=acc[v];
    }
}

extern "C" int qz15_mma(void const* a,void const* b,void const* c,void* d,
                        int m,int n,int k,int bf16,void* stream) {
    if(!a || !b || !c || !d || m<=0 || n<=0 || k<=0 || k%16 || (bf16!=0 && bf16!=1)) return -1;
    dim3 grid((n+15)/16,(m+15)/16);
    if(bf16) native_mma<true><<<grid,32,0,static_cast<hggcStream_t>(stream)>>>(
        static_cast<uint16_t const*>(a),static_cast<uint16_t const*>(b),
        static_cast<float const*>(c),static_cast<float*>(d),m,n,k);
    else native_mma<false><<<grid,32,0,static_cast<hggcStream_t>(stream)>>>(
        static_cast<uint16_t const*>(a),static_cast<uint16_t const*>(b),
        static_cast<float const*>(c),static_cast<float*>(d),m,n,k);
    return int(hggcGetLastError());
}

// Test the vendor pair as a unit: opaque b16 tags, padded global stride,
// three disjoint shared stages, and both A and transposed-B orientations.
// No dequantization or arithmetic is allowed to hide a wrong word mapping.
template<bool Trans,int Width,bool LegacyA=false>
__global__ void native_delivery(uint16_t const* source,uint16_t* output,int leading) {
    constexpr int Rows=Trans?Width:16, Cols=Trans?16:Width, Stages=3;
    using E=cutlass::half_t;
    // PPU0015 GemmConfig uses Swap=false for A and true for B. Here Trans
    // also selects the B test. The old unconditional true swapped two 8x8
    // quadrants of every A fragment (50% of the opaque tags).
    constexpr bool Swap=Trans || LegacyA;
    using Operand=cutlass::gemm::config::DefaultGemm_AIU_Operand<
        cutlass::arch::PPU0015,E,Trans,Int<Rows>,Int<Cols>,Swap>;
    using SL=decltype(tile_to_shape(typename Operand::SmemLayoutAtom{},Shape<Int<Rows>,Int<Cols>,Int<Stages>>{}));
    // Bind the minimal host CuTe oracle to these exact factory results.
    static_assert(is_same_v<typename Operand::SmemLayoutAtom,
        Layout<Shape<Int<Rows>,Int<Cols>>,Stride<_1,Int<Rows>>>>);
    static_assert(is_same_v<typename Operand::SmemCopyAtom,
        Copy_Atom<PPU0015_TSM_LD_SWZL<E,16,Width,Swap,Trans,1>,E>>);
    __shared__ __align__(128) E shared[cosize_v<SL>];
    constexpr int H=Trans?Cols:Rows, W=Trans?Rows:Cols;
    AiuDesc desc{};
    desc.init(reinterpret_cast<uint8_t const*>(source),H,W,leading);
    if(threadIdx.x==0) {
        #pragma unroll
        for(int s=0;s<Stages;++s)
            PPU0015_AIU_LOAD<Int<H*W*16>,E,Trans,H,W>::copy(
                shared+s*H*W,source+s*H*leading,desc,0,0);
        cp_async_fence();
        cp_async_wait<0>();
    }
    __syncthreads();
    using TM=TiledMMA<MMA_Atom<NativeAtom<false>>,Layout<Shape<_1,_1,_1>>>;
    TM mma;
    auto thread=mma.get_thread_slice(threadIdx.x);
    auto tensor=make_tensor(make_smem_ptr(shared),SL{});
    auto identity=make_identity_tensor(Shape<Int<Rows>,Int<Cols>>{});
    if constexpr(Trans) {
        auto fragment=thread.partition_fragment_B(tensor(_,_,0));
        auto coords=thread.partition_B(identity);
        auto copy_op=make_tiled_copy_B(typename Operand::SmemCopyAtom{},mma);
        copy_op.smem_base_=shared;
        auto slice=copy_op.get_thread_slice(0); // warp-uniform descriptor coordinates
        auto src=slice.partition_S(make_tsm_ld_tensor<SL>());
        auto dst=slice.retile_D(fragment);
        #pragma unroll
        for(int s=0;s<Stages;++s) {
            copy(copy_op,src(_,_,_,s),dst);
            #pragma unroll
            for(int i=0;i<size(fragment);++i)
                output[s*Rows*Cols+get<0>(coords(i))*Cols+get<1>(coords(i))]=
                    reinterpret_cast<uint16_t const*>(fragment.data())[i];
        }
    } else {
        auto fragment=thread.partition_fragment_A(tensor(_,_,0));
        auto coords=thread.partition_A(identity);
        auto copy_op=make_tiled_copy_A(typename Operand::SmemCopyAtom{},mma);
        copy_op.smem_base_=shared;
        auto slice=copy_op.get_thread_slice(0);
        auto src=slice.partition_S(make_tsm_ld_tensor<SL>());
        auto dst=slice.retile_D(fragment);
        #pragma unroll
        for(int s=0;s<Stages;++s) {
            copy(copy_op,src(_,_,_,s),dst);
            #pragma unroll
            for(int i=0;i<size(fragment);++i)
                output[s*Rows*Cols+get<0>(coords(i))*Cols+get<1>(coords(i))]=
                    reinterpret_cast<uint16_t const*>(fragment.data())[i];
        }
    }
}

extern "C" int qz15_delivery(void const* src,void* dst,int width,int trans,int leading,void* stream) {
    if(!src || !dst || (width!=32 && width!=64) || leading<width || (trans!=0 && trans!=1)) return -1;
    auto s=static_cast<hggcStream_t>(stream);
    auto a=static_cast<uint16_t const*>(src);auto b=static_cast<uint16_t*>(dst);
    if(trans && width==32) native_delivery<true,32><<<1,32,0,s>>>(a,b,leading);
    if(trans && width==64) native_delivery<true,64><<<1,32,0,s>>>(a,b,leading);
    if(!trans && width==32) native_delivery<false,32><<<1,32,0,s>>>(a,b,leading);
    if(!trans && width==64) native_delivery<false,64><<<1,32,0,s>>>(a,b,leading);
    return int(hggcGetLastError());
}

// Exact old A16x32 reader for the raw-bit negative; never a production path.
extern "C" int qz15_delivery_legacy(void const* src,void* dst,int leading,void* stream) {
    if(!src || !dst || leading<32) return -1;
    native_delivery<false,32,true><<<1,32,0,static_cast<hggcStream_t>(stream)>>>(
        static_cast<uint16_t const*>(src),static_cast<uint16_t*>(dst),leading);
    return int(hggcGetLastError());
}
