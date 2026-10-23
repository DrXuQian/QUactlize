// Host-only ABI sentinel, not a dequant reference or performance surrogate.
#include <cstdint>
enum ggml_type { GGML_TYPE_Q8_0=8 };
using converter=void(*)(void const*,uint16_t*,int64_t,void*);
static void sentinel(void const* input,uint16_t* output,int64_t elements,void* stream) {
    output[0]=static_cast<uint16_t const*>(input)[0];
    output[1]=uint16_t(uint64_t(elements)>>32);
    output[2]=uint16_t(reinterpret_cast<uintptr_t>(stream));
}
#ifndef OMIT_GETTER
converter ggml_get_to_fp16_cuda(ggml_type type) {
    return type==GGML_TYPE_Q8_0?sentinel:nullptr;
}
#endif
