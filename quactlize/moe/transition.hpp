#pragma once
#include "api.h"

namespace quactlize::moe {
int transition_snapshot(qzm_transition_v1 const*,float const* weights,int64_t stride,float* retained);
}
