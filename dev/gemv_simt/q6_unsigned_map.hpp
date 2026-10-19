#pragma once
#include <cstdint>

// Internal reader coordinates only: the unchanged public validator requires
// positive int32 N/K, whole Q6 units and a bounded grid. Signed scale values,
// expert IDs and the canonical offline maps are deliberately not changed.
namespace quactlize::execution::q6_unsigned {
template<unsigned Bits> struct Plane {
    static_assert(Bits==4 || Bits==2);
    static constexpr unsigned kPack=16u/Bits;
    static constexpr unsigned kLogicalKPerDelivery=8u*kPack;

    __host__ __device__ static constexpr unsigned word_slot(unsigned k) {
        return (k%kLogicalKPerDelivery)/8u;
    }
    __host__ __device__ static constexpr uint64_t word_index(unsigned n,unsigned k,unsigned extent) {
        unsigned const kg=(k/kLogicalKPerDelivery)*8u+(k%8u);
        return uint64_t(kg)*extent+n;
    }
};

__host__ __device__ constexpr uint64_t unit_offset(unsigned n,unsigned col,unsigned group) {
    return (uint64_t(group/32u)*n+col)*36u;
}
} // namespace quactlize::execution::q6_unsigned
