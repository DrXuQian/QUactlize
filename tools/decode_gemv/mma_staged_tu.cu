// Standalone prototype entry for the STAGED exact-code MMA reader (not a public ABI).
// One translation unit per (QQ, MODE, COMPUTE, TT); see tools/build_mma_proto.sh STAGED=1.
// code = wn*1000 + wk*10 + stages.
#include "mma_staged.cuh"

using namespace quactlize::execution;

#define QGEMV_NAME2(a,b) a##b
#define QGEMV_NAME(a,b) QGEMV_NAME2(a,b)

template<int WN,int WK,int Tokens,int S>
static int one(qkg_call_v1 const& c,int split) {
    constexpr int CtaN=WN*8*TT;
    if constexpr(mma_reader::staged_shared_bytes<QQ,CtaN,WK,S>()<=256*1024 &&
                 mma_reader::staged_shared_bytes<QQ,CtaN,WK,S>()>=size_t(WK)*Tokens*CtaN*4)
        return mma_reader::launch_staged<QQ,MODE,COMPUTE,TT,WN,WK,Tokens,S>(c,split);
    else return QKG_INVALID;
}
template<int WN,int WK,int Tokens>
static int by_stages(qkg_call_v1 const& c,int s,int split) {
    if(s==2) return one<WN,WK,Tokens,2>(c,split);
    if(s==3) return one<WN,WK,Tokens,3>(c,split);
    if(s==4) return one<WN,WK,Tokens,4>(c,split);
    return QKG_INVALID;
}
template<int WN,int WK>
static int by_tokens(qkg_call_v1 const& c,int s,int split) {
    int const tokens=MODE==QKG_INDEXED ? 1 : c.rows;
    if(tokens<=1) return by_stages<WN,WK,1>(c,s,split);
    if constexpr(MODE!=QKG_INDEXED) {
        if(tokens<=2) return by_stages<WN,WK,2>(c,s,split);
    }
    return QKG_SHAPE;
}
// CTA tiles of 64/128/256 columns (128/256/512 contiguous bytes per staged plane row).
#if TT==2
#define QGEMV_COMBOS(X) X(8,1) X(8,2) X(16,1)
#elif TT==4
#define QGEMV_COMBOS(X) X(2,2) X(2,4) X(4,1) X(4,2) X(4,4) X(8,1) X(8,2)
#else
#define QGEMV_COMBOS(X) X(1,2) X(1,4) X(2,1) X(2,2) X(2,4) X(4,1) X(4,2)
#endif
extern "C" int QGEMV_NAME(qgemv_mma_tu_,QGEMV_TU)(qkg_call_v1 const* c,int wn,int wk,int s,int split) {
#define QGEMV_W(A,B) if(wn==A && wk==B) return by_tokens<A,B>(*c,s,split);
    QGEMV_COMBOS(QGEMV_W)
#undef QGEMV_W
    return QKG_INVALID;
}
