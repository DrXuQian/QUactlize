#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_exact_code_geometry.hpp"
#include <cassert>
#include <set>
#include <utility>
#include <vector>

using namespace cutlass::gemm::collective::exact_code_ppu_detail;

template<int Tokens> void rotation() {
    using R=Rot<Tokens>;
    for(int active=1;active<=Tokens;++active) {
        std::set<std::pair<int,int>> owned;
        for(int t=0;t<4;++t) for(int v=0;v<2;++v) {
            int token=R::token(t,v),group=R::group(t,v);
            if(token>=active) continue;
            assert(owned.emplace(token,group).second);
            // Native B row which supplies this C output. Only its own group
            // is enabled during that group's MMA; other groups supply zero.
            int row=group+R::WG*token;
            assert(row==t+4*v && row<8);
            assert(row/R::WG==token && row%R::WG==group);
        }
        assert(owned.size()==size_t(active*R::WG));
    }
}

template<int Q,int N,int SB> void stage() {
    using L=StageLayout<Q,N,SB>;
    static_assert(L::Bytes%128==0 && L::Pitch%16==0);
    std::set<size_t> copies;
    auto rows=[&](int off,int count) {
        for(int r=0;r<count;++r) for(int q=0;q<N/8;++q) {
            size_t address=off+r*L::Pitch+q*16;
            assert(address%16==0 && address+16<=L::Bytes);
            assert(copies.insert(address).second);
        }
    };
    rows(L::LowOff,SB*L::LowPerSb);
    rows(L::HighOff,SB*L::HighPerSb);
    rows(L::MetaOff,SB*L::MetaPerSb);
    for(int p=0;p<L::UnitBytes;p+=16) {
        size_t address=L::UnitOff+p;
        assert(address%16==0 && address+16<=L::Bytes);
        assert(copies.insert(address).second);
    }
    size_t const expected=size_t(SB)*(L::LowPerSb+L::HighPerSb+L::MetaPerSb)*N*2+L::UnitBytes;
    assert(copies.size()*16==expected);
}

template<int T,int WN,int WK,int Tokens> void geometry() {
    using G=Geometry<12,T,WN,WK,Tokens>;
    std::vector<int> columns(G::CtaN);
    for(int wn=0;wn<WN;++wn) for(int m=0;m<8;++m) for(int i=0;i<T;++i)
        ++columns[wn*G::TileN+T*m+i];
    for(int count:columns) assert(count==1);
    for(int superblocks=1;superblocks<=20;++superblocks) {
        std::vector<int> coverage(superblocks);
        for(int wk=0;wk<WK;++wk)
            for(int b=superblocks*wk/WK;b<superblocks*(wk+1)/WK;++b) ++coverage[b];
        for(int count:coverage) assert(count==1);
    }
}

int main() {
    rotation<1>(); rotation<2>(); rotation<4>(); rotation<8>();
    stage<12,64,1>(); stage<12,128,2>(); stage<12,256,4>();
    stage<13,64,1>(); stage<13,128,2>(); stage<13,256,4>();
    stage<14,64,2>(); stage<14,128,4>(); stage<14,256,8>();
    stage<8,64,1>(); stage<8,128,2>(); stage<8,256,4>();
    geometry<2,1,16,1>(); geometry<4,4,2,2>();
    geometry<4,1,4,4>(); geometry<8,2,2,8>();
    static_assert(StageLayout<12,128,1>::Bytes*2==38912);
    static_assert(StageLayout<12,256,1>::Bytes*2==75776);
    static_assert(Geometry<12,4,2,4,1>::direct_shared_bytes(512)==1024);
    static_assert(Geometry<12,4,2,4,8>::direct_shared_bytes(512)==8704);
    static_assert(Geometry<8,4,2,4,8>::direct_shared_bytes(512)==8192);
    static_assert(kStageSb<14,1> == 2 && kStageSb<14,4> == 4 && kUnitSb<14> == 2);
    static_assert(Geometry<14,4,2,4,1>::metadata_bytes(512*256)==9216);
}
