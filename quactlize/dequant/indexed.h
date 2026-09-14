#pragma once
#include "api.h"

// Experimental active-expert variant of operation=1. Original [E,N,K]
// output strides are preserved; unselected experts are not touched.
// IDs has capacity E int32 elements; IDs[0:*count] must be unique and in [0,E).
// count and device_error are resident int32 GPU pointers. The caller zeros
// device_error before launch; invalid count/ID writes error=1/2 respectively.
// All input/output/selection buffers must be disjoint. No host readback,
// allocation, synchronization, or construction of the unique ID list here.
// Supports full configs4/5 for Q2--Q6 and configs10/11 for Q4/Q5.
#ifdef __cplusplus
extern "C" {
#endif
int quactlize_kpack_dequant_indexed_v1(qzd_call_v1 const*,
    quactlize_ppu_placed_arrangement_v2 const*, int const* ids,
    int const* count, int* device_error);
#ifdef __cplusplus
}
#endif
