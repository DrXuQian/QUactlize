#pragma once
#include "dense_api.h"

namespace quactlize::prefill {
// Private library-to-library entry, not a new caller ABI. Validate dynamic
// addresses once, then enqueue the admitted ordering of independent helpers.
// The public v1 entry retains its original order and stage numbering.
int run_dense_full(void* handle, qkp_call_v1 const* call, int stage, bool dequant_first);
}
