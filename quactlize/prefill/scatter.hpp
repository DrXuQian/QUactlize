#pragma once
#include <stdint.h>
// Private projection endpoint: launches the existing finish_bf16 body.
extern "C" int quactlize_prefill_scatter_bf16(void const*,float*,int32_t const*,
    int,int,int64_t,int32_t*,void*);
