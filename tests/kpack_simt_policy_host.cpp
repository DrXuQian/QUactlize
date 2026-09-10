#include "quactlize-execution-lib.h"
#include <cstdio>

int main() {
    qkg_call_v1 call{};
    call.qtype=8; call.n=512; call.k=2048; call.experts=1;
    call.mode=QKG_DENSE; call.rows=call.channels=call.topk=1; call.input_type=QKG_F32;
    qkg_config_v1 config{};
    if (!ggml_quactlize_gemv_config(call,&config) || config.columns!=32 || config.warps!=4 || config.split!=4)
        return 1;
    call.n=1024;
    if (ggml_quactlize_gemv_config(call,&config)) return 2;
    call.n=512; call.qtype=12;
    if (!ggml_quactlize_gemv_config(call,&config) || config.columns!=16 || config.warps!=8 || config.split!=1)
        return 3;
    call.rows=4;
    if (ggml_quactlize_gemv_config(call,&config)) return 4;
    std::puts("SIMT_POLICY_HOST PASS Q8+Q4 exact hits and shape/row misses");
}
