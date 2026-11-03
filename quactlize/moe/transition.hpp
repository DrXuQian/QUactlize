#pragma once
#include "api.h"

namespace quactlize::moe {
// Private FFN endpoint: retain token/slot weights in the existing transition
// before the final output can overwrite their graph-allocated storage.
int transition_snapshot(qzm_transition_v1 const*,float const* weights,int64_t stride,float* retained);
}
