#pragma once
#include "dense_plane_pair.hpp"

namespace quactlize::dequant {
// Keep the old 2N*K8 output ownership. Adjacent lanes jointly load the eight
// packed K residues: each reads one aligned uint4 and exchanges it with xor1.
// Each word contains both K8 phases; only the byte shift differs afterwards.
// No extra shared buffer, barrier, output transpose or arithmetic change.
template<unsigned TileK> struct DenseQ8Exchange : DensePlanePair<8,TileK> {
    using Base=DensePlanePair<8,TileK>;
    using Low=typename Base::Low;
    CUTLASS_HOST_DEVICE static constexpr unsigned pair(unsigned task) {return task/(TileK/8);}
    CUTLASS_HOST_DEVICE static constexpr unsigned k(unsigned task) {return task%(TileK/8)*8;}
    CUTLASS_HOST_DEVICE static constexpr unsigned partner(unsigned task) {return task^1u;}
    CUTLASS_HOST_DEVICE static constexpr unsigned source(unsigned task) {
        // Apply the K-residue displacement BEFORE swizzle. Pair bit2 swaps
        // the two uint4s; adding4 to the already-swizzled base is incorrect.
        return Low::shared(pair(task),Low::row(k(task))+(task&1u)*4);
    }
    CUTLASS_HOST_DEVICE static uint32_t word(unsigned task,unsigned residue,uint32_t own,uint32_t other) {
        return (residue/4)==(task&1u)?own:other;
    }
};
} // namespace quactlize::dequant
