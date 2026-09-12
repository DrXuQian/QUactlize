// Exact-order CTA fold for the bounded medium Q4 experiment.
// Called by the first warp only, after the existing CTA barrier.
// Host-callable as well, so the actual helper can be compared bitwise with
// the old loop without a device. It introduces no CUDA intrinsic or barrier.
template<int Round,int Warps,int TileN>
__host__ __device__ __forceinline__ void q4_medium_fold(
        float& sum,float const* partial,unsigned lane) {
    static_assert(TileN>0 && TileN<=32 && 32%TileN==0);
    constexpr unsigned Stripes=32/TileN;
    unsigned const first=(lane&31u)/TileN;
    unsigned const w=first+Round*Stripes;
    if constexpr((Round+1)*Stripes<=Warps) {
        sum+=partial[w*TileN+(lane&(TileN-1))];
    } else if(w<Warps) {
        sum+=partial[w*TileN+(lane&(TileN-1))];
    }
    if constexpr((Round+1)*Stripes<Warps)
        q4_medium_fold<Round+1,Warps,TileN>(sum,partial,lane);
}
