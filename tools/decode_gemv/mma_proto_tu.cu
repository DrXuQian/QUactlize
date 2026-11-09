// Standalone prototype entry for the register-direct MMA reader (not a public ABI).
// One translation unit per (QQ, MODE, COMPUTE, TT); see tools/build_mma_proto.sh.
#include "mma_reader.cuh"

using namespace quactlize::execution;

#define QGEMV_NAME2(a,b) a##b
#define QGEMV_NAME(a,b) QGEMV_NAME2(a,b)

template<int WN,int WK,int Tokens,int NB>
static int one(qkg_call_v1 const& c,int split) {
    constexpr int GPSB=8/mma_reader::Kind<QQ>::BPG;
    // The ring must divide a superblock; eight T=8 granules would not fit the register file.
    if constexpr(GPSB%NB==0 && !(TT==8 && NB==8))
        return mma_reader::launch_typed<QQ,MODE,COMPUTE,TT,WN,WK,Tokens,NB>(c,split);
    else return QKG_INVALID;
}
template<int WN,int WK,int Tokens>
static int by_nb(qkg_call_v1 const& c,int nb,int split) {
    if(nb==2) return one<WN,WK,Tokens,2>(c,split);
    if(nb==4) return one<WN,WK,Tokens,4>(c,split);
    if(nb==8) return one<WN,WK,Tokens,8>(c,split);
    return QKG_INVALID;
}
template<int WN,int WK>
static int by_tokens(qkg_call_v1 const& c,int nb,int split) {
    int const tokens=MODE==QKG_INDEXED ? 1 : c.rows;
    if(tokens<=1) return by_nb<WN,WK,1>(c,nb,split);
    if constexpr(MODE!=QKG_INDEXED) {
        if(tokens<=2) return by_nb<WN,WK,2>(c,nb,split);
        if(tokens<=4) return by_nb<WN,WK,4>(c,nb,split);
        if(tokens<=8) return by_nb<WN,WK,8>(c,nb,split);
    }
    return QKG_SHAPE;
}
// Warp layouts per column tile: narrow tiles take more k-slice warps.
#if TT==2
#define QGEMV_COMBOS(X) X(1,4) X(1,8) X(1,16) X(2,4) X(2,8) X(4,2) X(4,4)
#elif TT==4
#define QGEMV_COMBOS(X) X(1,4) X(1,8) X(2,2) X(2,4) X(4,1) X(4,2)
#else
#define QGEMV_COMBOS(X) X(1,2) X(1,4) X(2,1) X(2,2) X(2,4) X(4,1) X(4,2) X(8,1)
#endif
extern "C" int QGEMV_NAME(qgemv_mma_tu_,QGEMV_TU)(qkg_call_v1 const* c,int wn,int wk,int nb,int split) {
#define QGEMV_W(A,B) if(wn==A && wk==B) return by_tokens<A,B>(*c,nb,split);
    QGEMV_COMBOS(QGEMV_W)
#undef QGEMV_W
    return QKG_INVALID;
}
