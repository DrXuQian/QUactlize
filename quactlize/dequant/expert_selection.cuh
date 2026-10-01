#pragma once

namespace quactlize::dequant {
struct ExpertSelection {
    int const* ids = nullptr;
    int const* count = nullptr;
    int* error = nullptr;
    int capacity = 0;
};

template<bool Indexed>
__device__ __forceinline__ int select_expert(int slot, ExpertSelection const& selection) {
    if constexpr (!Indexed) return slot;
    // Uniform per CTA, before any collective load or barrier. Input IDs are
    // a unique active list owned by the router; no host readback is needed.
    int count = *selection.count;
    if (count < 0 || count > selection.capacity) {
        if (threadIdx.x == 0) atomicExch(selection.error, 1);
        return -1;
    }
    if (slot >= count) return -1;
    int e = selection.ids[slot];
    if (e < 0 || e >= selection.capacity) {
        if (threadIdx.x == 0) atomicExch(selection.error, 2);
        return -1;
    }
    return e;
}
} // namespace quactlize::dequant
