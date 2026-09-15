#include <hggc_runtime.h>
dim3 threadIdx{0,0,0};
#include "cutlass/numeric_types.h"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_decode_input.hpp"
#include "quactlize/execution/q4_s1_validation.hpp"
#include "quactlize/execution/q4_decode.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstring>

static uint4 shuffle_words[32];
static int shuffle_word=0,shuffle_base=0;
inline float __uint_as_float(uint32_t bits) { float value;std::memcpy(&value,&bits,4);return value; }
inline uint32_t __shfl_sync(unsigned,uint32_t,int owner) {
    auto const& v=shuffle_words[owner];
    uint32_t words[]={v.x,v.y,v.z,v.w};
    return words[shuffle_base+(shuffle_word++%2)];
}
template<class T> T __shfl_xor_sync(unsigned,T,int) {assert(false);return T{};}
inline uint32_t __byte_perm(uint32_t,uint32_t,int) {assert(false);return 0;}
// Dequant arithmetic is not executed by this host-only transport test.
__half2 __hfma2(__half2,__half2,__half2);
#include "quactlize/execution/q4_s1_helpers.cuh"

using Bf16=cutlass::bfloat16_t;
using Half=cutlass::half_t;
namespace detail=cutlass::gemm::collective::detail;

template<int Columns,int Input> void register_transport() {
    namespace q4=quactlize::execution::q4_s1;
    using A=q4::ComputeActivation<Input,1>;
    alignas(16) typename A::Scalar storage[32];
    float expected[32];
    for (int i=0;i<32;++i) {
        float value=i%5==0 ? 243383.484375f : (float(i)-16.f)*1.00390625f;
        if constexpr(Input==1) storage[i]=value;
        else storage[i]=Bf16(value).raw();
        expected[i]=float(Bf16(value));
    }
    A a{storage};
    for (int lane=0;lane<32;++lane) shuffle_words[lane]=q4::q4_cooperative_a_chunk<Columns>(a,0,lane);
    int wrong_type=0;
    for (int lane=0;lane<32;++lane)
        for (int offset=0;offset<32;offset+=4) {
            shuffle_word=0;shuffle_base=Columns==4 && (offset&4) ? 2 : 0;
            float4 v=q4::q4_cooperative_a_read<Columns,1>(shuffle_words[lane],offset,lane);
            float values[]={v.x,v.y,v.z,v.w};
            for (int j=0;j<4;++j) assert(values[j]==expected[offset+j]);
            shuffle_word=0;
            v=q4::q4_cooperative_a_read<Columns,0>(shuffle_words[lane],offset,lane);
            wrong_type+=v.x!=expected[offset];
        }
    assert(wrong_type>128);
    for (int residue=0;residue<8;++residue) {
        uint4 packed=q4::latency_residue_a<1,Input,1>(a,0,residue);
        float4 v=q4::latency_residue_values<1,1>(packed,residue);
        float values[]={v.x,v.y,v.z,v.w};
        for (int j=0;j<4;++j) assert(values[j]==expected[residue+8*j]);
    }
    for (int offset=0;offset<32;offset+=4) {
        float4 v=q4::aligned_activation<0>(a,offset);
        float values[]={v.x,v.y,v.z,v.w};
        for (int j=0;j<4;++j) assert(values[j]==expected[offset+j]);
    }
}

extern "C" int qkg_q4_decode_launch(qkg_call_v1 const&,qkg_q4_decode_config_v1 const&) {return QKG_OK;}
extern "C" int qkg_q4_decode_launch_bf16(qkg_call_v1 const&,qkg_q4_decode_config_v1 const&) {return QKG_OK;}

template<class Source,int TileK> struct Tensor {
    using value_type=Source;
    Source const* data;
    Source const& operator()(int row,int column,int tile) const {
        assert(row==0);
        return data[tile*TileK+column];
    }
};

template<class Source,class Compute,int TileK,int Stages,int Threads>
int writer_reader() {
    constexpr int Cubes=TileK/64,Pitch=64;
    constexpr int StagePitch=detail::aPackStagePitchHalfs(Pitch,Cubes,1024);
    constexpr int Span=StagePitch*(Stages-1)+Pitch*(Cubes-1)+1024;
    using Read=cute::PPU0010_TSM_LD_SWZL_M8<Compute,16,64,true,false,Cubes,Pitch,StagePitch>;
    using HalfRead=cute::PPU0010_TSM_LD_SWZL_M8<Half,16,64,true,false,Cubes,Pitch,StagePitch>;
    static_assert(Read::kLogicalRegisters==HalfRead::kLogicalRegisters);
    std::vector<Source> input(TileK*Stages);
    std::vector<Compute> output(Span+16);
    std::vector<int> owners(Span+16);
    for (int i=0;i<int(input.size());++i)
        input[i]=Source(i%7==0 ? 243383.484375f : float(i-401)*1.00390625f);
    for (auto& value:output) value=Compute::bitcast(0x7fc1);
    for (int stage=0;stage<Stages;++stage)
        for (int thread=0;thread<Threads;++thread)
            detail::copy_decode_a<16,TileK,Threads,true,1,Pitch,StagePitch>(
                Tensor<Source,TileK>{input.data()},output.data(),stage,stage,thread,1);
    int cells=0,wrong_type=0,nonfinite=0;
    for (int stage=0;stage<Stages;++stage)
        for (int cube=0;cube<Cubes;++cube)
            for (int slice=0;slice<4;++slice)
                for (int lane=0;lane<4;++lane)
                    for (int vreg=0;vreg<2;++vreg)
                        for (int half=0;half<2;++half) {
                            // Independently calibrated M8 physical reader; not the writer's run-offset helper.
                            int word=Read::logical_word_offset(lane,vreg,slice*16,0);
                            assert(word==HalfRead::logical_word_offset(lane,vreg,slice*16,0));
                            int offset=stage*StagePitch+cube*Pitch+2*word+half;
                            int k=cube*64+slice*16+vreg*8+lane*2+half;
                            auto expected=Compute(float(input[stage*TileK+k]));
                            assert(output[offset].raw()==expected.raw());
                            wrong_type+=Half(float(input[stage*TileK+k])).raw()!=output[offset].raw();
                            nonfinite+=!std::isfinite(float(output[offset]));
                            ++owners[offset];++cells;
                        }
    for (int i=0;i<int(output.size());++i) {
        assert(owners[i]<=1);
        if (!owners[i]) assert(output[i].raw()==0x7fc1);
    }
    if constexpr(std::is_same_v<Compute,Bf16>) {
        assert(nonfinite==0 && wrong_type>cells/2);
    } else assert(nonfinite>0);
    return cells;
}

void queries() {
    auto arrangement=ppu_arrangements::q4_kpack4_transpose_v1();
    qkg_call_v1 c{};
    c.version=1;c.size=sizeof(c);c.qtype=12;c.mode=QKG_DENSE;c.input_type=QKG_F32;
    c.n=512;c.k=2048;c.rows=1;c.experts=1;c.channels=1;c.topk=1;
    c.a_row_stride=c.k;c.out_row_stride=c.n;
    qkg_simt_call_v2 typed{2,sizeof(typed),c,QKG_COMPUTE_BF16};
    qkg_q4_decode_config_v1 selected{};qkg_sizes_v1 sizes{};
    assert(quactlize_kpack_q4_decode_select_v2(&typed,&arrangement,&selected,&sizes)==QKG_OK);
    auto original=selected;
    typed.call.input_type=QKG_SIMT_BF16;
    assert(quactlize_kpack_q4_decode_select_v2(&typed,&arrangement,&selected,&sizes)==QKG_OK);
    assert(original.reader==selected.reader && original.variant==selected.variant);
    typed.compute_type=QKG_COMPUTE_F16;
    assert(quactlize_kpack_q4_decode_select_v2(&typed,&arrangement,&selected,&sizes)==QKG_INVALID);
    typed.compute_type=QKG_COMPUTE_BF16;
    typed.call.a_row_stride+=4;
    assert(quactlize_kpack_q4_decode_select_v2(&typed,&arrangement,&selected,&sizes)==QKG_SHAPE);
    typed.call.a_row_stride=c.k;typed.size--;
    assert(quactlize_kpack_q4_decode_select_v2(&typed,&arrangement,&selected,&sizes)==QKG_INVALID);
    typed.size++;typed.call.input_type=QKG_F32;
    for (auto recipe:{qkg_q4_s1_config_v1{1,sizeof(qkg_q4_s1_config_v1),0,0,8,1},
                      qkg_q4_s1_config_v1{1,sizeof(qkg_q4_s1_config_v1),1,2,20,4},
                      qkg_q4_s1_config_v1{1,sizeof(qkg_q4_s1_config_v1),2,7,8,8}}) {
        assert(quactlize::execution::q4_s1::validate_v2(typed,recipe,&arrangement,sizes)==QKG_OK);
        typed.call.input_type=QKG_SIMT_BF16;
        assert(quactlize::execution::q4_s1::validate_v2(typed,recipe,&arrangement,sizes)==QKG_OK);
        assert(quactlize::execution::q4_s1::validate(typed.call,recipe,&arrangement,sizes)==QKG_INVALID);
        typed.call.input_type=QKG_F32;
    }
}

int main() {
    using H=cute::MMA_Traits<cute::PPU0010_8x16x16_F32F16F16F32_TN>;
    using B=cute::MMA_Traits<cute::PPU0010_8x16x16_F32BF16BF16F32_TN>;
    static_assert(std::is_same_v<H::ALayout,B::ALayout> && std::is_same_v<H::BLayout,B::BLayout>);
    static_assert(std::is_same_v<H::CLayout,B::CLayout> && std::is_same_v<B::ValTypeA,Bf16>);
    int cells=0;
    cells+=writer_reader<float,Bf16,64,2,32>();
    cells+=writer_reader<float,Bf16,128,3,128>();
    cells+=writer_reader<float,Bf16,256,4,256>();
    cells+=writer_reader<Bf16,Bf16,256,3,128>();
    cells+=writer_reader<float,Half,256,3,128>();
    queries();
    register_transport<4,1>();register_transport<8,1>();
    register_transport<4,2>();register_transport<8,2>();
    std::printf("BF16_FASTPATH_HOST PASS packed_cells=%d typed_m8_source_destination, range, guards, selectors\n",cells);
}
