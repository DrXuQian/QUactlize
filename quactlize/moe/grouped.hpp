#pragma once
#include "api.h"
#include "../runtime/abi.h"

namespace quactlize::moe {
// The producer, not a caller guess from its original split setting, owns the
// output-state declaration. This bridge introduces no cast/reducer/activation.
inline int grouped_value(qk_grouped_result_v1 const& r,qzm_value_v1& v) {
    v={};
    if(r.version!=1 || r.size!=sizeof(r) || r.reserved || !r.data || r.row_stride<=0 ||
       (r.compute_type!=QK_COMPUTE_F16 && r.compute_type!=QK_COMPUTE_BF16) ||
       (r.partial_count!=1 && r.partial_count!=2 && r.partial_count!=4 && r.partial_count!=8))
        return QZR_INVALID;
    bool partial=r.partial_count>1;
    int typed=r.compute_type==QK_COMPUTE_BF16?QZM_BF16:QZM_F16;
    if(partial?(r.dtype!=QK_RESULT_F32 || r.partial_stride<r.row_stride):
       (r.dtype!=(r.compute_type==QK_COMPUTE_BF16?QK_RESULT_BF16:QK_RESULT_F16) || r.partial_stride))
        return QZR_INVALID;
    v={r.data,partial?QZM_F32:typed,QZM_GROUPED,partial?QZM_PARTIALS:QZM_PROJECTED,
       r.partial_count,r.row_stride,r.partial_stride,typed,0};
    return QZR_OK;
}
} // namespace quactlize::moe
