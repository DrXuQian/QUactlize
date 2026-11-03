#pragma once
#include "../../quactlize/moe/api.h"
#include <cstdint>

namespace quactlize::moe::prefill_helpers {
inline bool aligned(void const* p,int bytes) {return p && !(reinterpret_cast<uintptr_t>(p)%bytes);}
inline bool bf16_projected(qzm_value_v1 const& v) {
    return v.dtype==QZM_BF16 && v.round_type==QZM_BF16 && v.order==QZM_GROUPED &&
        v.state==QZM_PROJECTED && v.partial_count==1 && v.row_stride%4==0 && aligned(v.data,8);
}
inline bool prepare_fast(qzm_prepare_v1 const& c) {
    return c.route.tokens>0 && c.route.topk==8 && c.route.channels==1 &&
        c.config.algorithm==QZR_TILED && c.config.tile_rows==512 && c.dtype==QZM_BF16 &&
        c.columns%4==0 && c.input_row_stride%4==0 && aligned(c.input,16) && aligned(c.output,8);
}
inline bool finish_fast(qzm_finish_v1 const& c) {
    return c.tokens>0 && c.topk==8 && c.columns%4==0 && c.output_stride%4==0 &&
        bf16_projected(c.down) && aligned(c.output,16);
}
inline bool transition_fast(qzm_transition_v1 const& c) {
    return c.tokens>0 && c.merged==1 && c.hidden%4==0 && bf16_projected(c.gate) &&
        c.output_order==QZM_GROUPED && c.output_dtype==QZM_BF16 &&
        c.output_stride%4==0 && aligned(c.output,8);
}
} // namespace quactlize::moe::prefill_helpers
