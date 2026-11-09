#pragma once
// Host-testable ownership and storage geometry of the exact-code PPU collective.
// Logical GEMV (tokens, N, K) is NOT the native MMA (8 columns, 16 group/token rows, 16 K).
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HGGC__) || defined(__HGGCCC__)
#define QZ_EXACT_HD __host__ __device__
#else
#define QZ_EXACT_HD
#endif

namespace cutlass::gemm::collective {

// These tags belong to Quactlize. Neither overrides an actlize dispatch policy
// nor opts an existing mixed-input collective into a different arithmetic contract.
template<int Lookahead_> struct ExactCodeRegisterDirectPPU {
    static constexpr int Lookahead = Lookahead_;
};
template<int Stages_> struct ExactCodeCpAsyncPPU {
    static constexpr int Stages = Stages_;
};

namespace exact_code_ppu_detail {

template<int Tokens> struct Rot {
    static_assert(Tokens==1 || Tokens==2 || Tokens==4 || Tokens==8);
    static constexpr int TP=Tokens, WG=8/TP, V=TP==1 ? 1 : 2;
    static constexpr bool Direct=TP<=2;
    QZ_EXACT_HD static constexpr int group(int t,int v) { return (t+4*v)%WG; }
    QZ_EXACT_HD static constexpr int token(int t,int v) { return (t+4*v)/WG; }
};

template<int Q,int T,int WN,int WK,int Tokens> struct Geometry {
    static_assert(Q==8 || Q==12 || Q==13 || Q==14, "exact-code format not implemented");
    static_assert(T==2 || T==4 || T==8);
    static_assert(WN>0 && (WN&(WN-1))==0 && WK>0 && (WK&(WK-1))==0);
    static_assert(WN*WK<=32, "CTA exceeds 1024 threads");
    using Rotation=Rot<Tokens>;
    static constexpr int TileN=8*T, CtaN=WN*TileN, Threads=WN*WK*32;
    static constexpr int GroupK=Q==14 ? 16 : 32;
    static constexpr int LowBits=Q==8 ? 8 : 4;
    static constexpr int HighBits=Q==13 ? 1 : Q==14 ? 2 : 0;
    QZ_EXACT_HD static constexpr uint64_t metadata_bytes(uint64_t nk) {
        return Q==8 ? nk/16 : nk/256*(Q==14 ? 18 : 16);
    }
    static constexpr size_t reduction_bytes=WK>1 ? size_t(WK)*Tokens*CtaN*4 : 0;
    static constexpr size_t direct_shared_bytes(int k_len) {
        size_t const stage=Rotation::Direct ? 0 :
            size_t(Tokens)*(k_len/2)*4 +
            ((Q==12 || Q==13) ? size_t(Tokens)*(k_len/32)*4 : 0);
        return stage>reduction_bytes ? stage : reduction_bytes;
    }
};

// Q6's 36-byte canonical metadata unit spans two 256-K superblocks.
template<int Q> constexpr int kUnitSb=Q==14 ? 2 : 1;
template<int Q,int WK> constexpr int kStageSb=Q==14 ? (WK==1 ? 2 : WK) : WK;

template<int Q,int CtaN,int SB> struct StageLayout {
    static_assert(Q==8 || Q==12 || Q==13 || Q==14);
    static_assert(CtaN>0 && CtaN%16==0 && SB>0 && SB%kUnitSb<Q> == 0);
    static constexpr int Pitch=CtaN*2+16;
    static constexpr int LowPerSb=Q==8 ? 128 : 64;
    static constexpr int HighPerSb=Q==13 ? 16 : Q==14 ? 32 : 0;
    static constexpr int MetaPerSb=Q==8 ? 8 : 0;
    static constexpr int LowOff=0;
    static constexpr int HighOff=SB*LowPerSb*Pitch;
    static constexpr int MetaOff=HighOff+SB*HighPerSb*Pitch;
    static constexpr int UnitOff=MetaOff+SB*MetaPerSb*Pitch;
    static constexpr int UnitBytes=(Q==12 || Q==13) ? SB*CtaN*16 : Q==14 ? (SB/2)*CtaN*36 : 0;
    static constexpr int Bytes=(UnitOff+UnitBytes+127)/128*128;
};

} // namespace exact_code_ppu_detail
} // namespace cutlass::gemm::collective
#undef QZ_EXACT_HD
