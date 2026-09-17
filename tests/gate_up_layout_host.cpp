#include "cute/tensor.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "../quactlize/fusion/paired_n4.hpp"

template<class Atom,int TM,int TN>
int check() {
    using namespace cute;
    constexpr int AM=size<0>(typename MMA_Traits<Atom>::Shape_MNK{});
    using Mma=TiledMMA<MMA_Atom<Atom>,Layout<Shape<Int<TM/AM>,Int<TN/16>,_1>>>;
    int errors=0,counts[TM*TN/2]{};
    for(int thread=0;thread<size(Mma{});++thread) {
        auto coords=Mma{}.get_thread_slice(thread).partition_C(make_identity_tensor(make_shape(Int<TM>{},Int<TN>{})));
        for(int i=0;i<size(coords);i+=2) {
            auto g=coords(i),u=coords(i+1);
            errors+=(get<0>(g)!=get<0>(u) || get<1>(u)!=get<1>(g)+4 || (get<1>(g)&4));
            int n=quactlize::fusion::PairedN4::channel(get<1>(g));
            ++counts[get<0>(g)*(TN/2)+n];
        }
    }
    for(auto count:counts) errors+=count!=1;
    return errors;
}
extern "C" int gate_up_ownership() {
    using namespace cute;
    return check<PPU0010_8x16x16_F32F16F16F32_TN,8,64>()+
           check<PPU0010_16x16x16_F32F16F16F32_TN,16,64>()+
           check<PPU0010_8x16x16_F32BF16BF16F32_TN,16,128>()+
           check<PPU0010_16x16x16_F32BF16BF16F32_TN,32,128>();
}
