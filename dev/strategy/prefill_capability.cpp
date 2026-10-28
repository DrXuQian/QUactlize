// CPU-only invocation of the shipping Full-BF16 layout validator.
// Token M is not grouped row M; do not reproduce its limits in Python.
#include "quactlize/prefill/layout.hpp"
#include <iostream>

int main() {
    int q, n, k, tokens, experts, topk, channels, config;
    while (std::cin >> q >> n >> k >> tokens >> experts >> topk >> channels >> config) {
        const auto arr = q == 8 ? q8_kpack2::arrangement() : q == 12 ?
            ppu_arrangements::q4_kpack4_transpose_v1() :
            ppu_arrangements::kquant_kpack_transpose_v1(q);
        qkp_call_v1 c{};
        c.version = 1;
        c.size = sizeof(c);
        c.m = tokens * topk;
        c.a_rows = tokens * channels;
        c.a_stride = k;
        c.output_stride = n;
        c.weight = {1, sizeof(qzd_call_v1), q, n, k, experts, 1, config};
        quactlize::prefill::Layout layout{};
        const int rc = quactlize::prefill::layout(c, &arr, layout);
        std::cout << "{\"request\":[" << q << ',' << n << ',' << k << ',' << tokens << ','
                  << experts << ',' << topk << ',' << channels << ',' << config
                  << "],\"rows\":" << c.m << ",\"rc\":" << rc
                  << ",\"workspace_bytes\":" << (rc ? 0 : layout.bytes) << "}\n";
    }
    return std::cin.eof() ? 0 : 1;
}
