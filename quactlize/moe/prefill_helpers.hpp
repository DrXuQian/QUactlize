#pragma once
#include "api.h"
#include <cstdint>

namespace quactlize::moe::prefill_helpers {
// ZW810/1700MHz, source a09f0126, exact GPU parity and separate Asys means.
// Review SHA256: 02b5a5efb296f8aa0cc7ce5743e761459ba0c7f8f7626c0b5dd58bc15a805b5f.
// One configuration per helper, no routing-ID readback or sub-us shape ties.
// These are admission bounds, not algorithm limits; arbitrary M/strides and
// other producer contracts retain the generic implementation.
inline bool measured_tokens(int m) {return m==512 || m==2048 || m==5120;}
inline bool measured_columns(int n) {return n==2048 || n==3072;}
inline bool aligned(void const* p,int bytes) {return p && !(reinterpret_cast<uintptr_t>(p)%bytes);}
inline bool bf16_projected(qzm_value_v1 const& v) {
    return v.dtype==QZM_BF16 && v.round_type==QZM_BF16 && v.order==QZM_GROUPED &&
        v.state==QZM_PROJECTED && v.partial_count==1 && v.row_stride%4==0 && aligned(v.data,8);
}
// Pointer-free: AUTO queries happen before workspace addresses are bound.
inline bool prepare_shape(qzm_prepare_v1 const& c) {
    return measured_tokens(c.route.tokens) && measured_columns(c.columns) && c.route.experts==256 &&
        c.route.topk==8 && c.route.channels==1 && c.config.algorithm==QZR_TILED && c.config.tile_rows==512 &&
        c.dtype==QZM_BF16 && c.input_row_stride%4==0;
}
inline bool prepare_fast(qzm_prepare_v1 const& c) {
    return prepare_shape(c) && aligned(c.input,16) && aligned(c.output,8);
}
inline bool finish_fast(qzm_finish_v1 const& c) {
    return measured_tokens(c.tokens) && measured_columns(c.columns) && c.topk==8 &&
        c.output_stride%4==0 && bf16_projected(c.down) && aligned(c.output,16);
}
inline bool transition_fast(qzm_transition_v1 const& c) {
    return measured_tokens(c.tokens) && c.topk==8 && c.hidden==512 && c.merged==1 &&
        bf16_projected(c.gate) && c.output_order==QZM_GROUPED && c.output_dtype==QZM_BF16 &&
        c.output_stride%4==0 && aligned(c.output,8);
}
} // namespace quactlize::moe::prefill_helpers
