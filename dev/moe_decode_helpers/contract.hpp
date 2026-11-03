#pragma once
#include "../../quactlize/moe/api.h"

namespace quactlize::moe::decode_helpers {
// These predicates run after the existing public descriptor validation.
// No measured-model dimensions or M cutoff are part of kernel legality.
inline bool finish_fast(qzm_finish_v1 const& c) {
    return c.topk==8 && c.down.dtype==QZM_F32 && c.down.order==QZM_TOKEN_SLOT &&
           c.down.state==QZM_PROJECTED && c.down.partial_count==1;
}
inline bool transition_fast(qzm_transition_v1 const& c) {
    return c.merged==1 && c.gate.dtype==QZM_F32 && c.gate.order==QZM_TOKEN_SLOT &&
           c.gate.state==QZM_PROJECTED && c.gate.partial_count==1 &&
           c.gate.round_type==QZM_F32 && c.output_dtype==QZM_F32 && c.output_order==QZM_TOKEN_SLOT;
}
}
