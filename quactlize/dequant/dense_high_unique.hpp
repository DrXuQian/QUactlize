#pragma once
#include "dense_cooperative.hpp"

namespace quactlize::dequant {

// Ownership follows PHYSICAL 16B high-plane vectors, not the logical K32
// tasks. Each loaded word feeds two K32 tasks; Q5 also feeds two N8 halves.
// Keep the established consumer-side high layout and halfword publication.
// N64 gives eight Q6 vectors/four Q5 vectors per residue. Residue parity is
// outside the innermost four residues to avoid same-warp halfword writers.
template<KType T,unsigned TileK> struct DenseUniqueHigh {
    static_assert(T==KType::Q5_K || T==KType::Q6_K);
    static_assert(TileK==128 || TileK==256);
    static constexpr unsigned NSides=T==KType::Q5_K?2:1;
    static constexpr unsigned VectorsN=64/(8*NSides);
    static constexpr unsigned Loads=(TileK/64)*8*VectorsN;
    static constexpr unsigned Fans=2*NSides;
    struct Owner { unsigned col,residue,group64; };
    struct Target { unsigned col,group32; };
    CUTLASS_HOST_DEVICE static constexpr Owner owner(unsigned i) {
        unsigned t=i/VectorsN;
        if constexpr(T==KType::Q5_K) {
            // Four physical N16 cohorts: use two K64 groups to fill a warp
            // before switching residue parity, rather than writing the two
            // halfwords of one shared word from different lanes together.
            return {(i%VectorsN)*16,((t&3u)<<1)|((t>>3)&1u),((t>>2)&1u)|((t>>4)<<1)};
        }
        return {(i%VectorsN)*(8*NSides),((t&3u)<<1)|((t>>2)&1u),t>>3};
    }
    CUTLASS_HOST_DEVICE static constexpr Target target(Owner p,unsigned fan) {
        // Q5's K64 group parity fills the bank bit missing from its physical
        // N16 cohort. Both K32 destinations still receive exactly one writer.
        unsigned phase=T==KType::Q5_K?(p.group64&1u):0u;
        return {p.col+8*(fan/2),2*p.group64+((fan&1u)^phase)};
    }
};
} // namespace quactlize::dequant
