#pragma once
#include <cstdint>

// Experiment: canonical coordinates, not a new packed representation. All
// coordinates are validated nonnegative before launch. Signed expert IDs and
// signed scales must not use these types; address products remain 64 bit.
#if defined(__CUDACC__) || defined(__HGGC__)
#define QG_HD __host__ __device__
#else
#define QG_HD
#endif
namespace quactlize::execution::general_map {
template<unsigned Bits> struct Plane {
    static constexpr unsigned kPack = 16 / Bits;
    static constexpr unsigned kLogicalKPerDelivery = 8 * kPack;
    QG_HD static constexpr unsigned word_slot(unsigned k) {
        return (k % kLogicalKPerDelivery) / 8;
    }
    QG_HD static constexpr uint64_t word_index(unsigned n, unsigned k, unsigned extent) {
        return uint64_t((k / kLogicalKPerDelivery) * 8 + k % 8) * extent + n;
    }
};

// Q5's high plane interchanges N bit3 and K bit7; it is NOT Plane<1>.
struct Q5High {
    QG_HD static constexpr unsigned word_slot(unsigned n, unsigned k) {
        return (((n >> 3) & 1) << 3) | ((k >> 3) & 7);
    }
    QG_HD static constexpr uint64_t word_index(unsigned n, unsigned k, unsigned extent) {
        unsigned const pn = (n & ~15u) | (n & 7u) | (((k >> 7) & 1u) << 3);
        unsigned const kg = (k / 256) * 16 | (((k >> 6) & 1) << 3) | (k & 7);
        return uint64_t(kg) * extent + pn;
    }
};
}
#undef QG_HD
