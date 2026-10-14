// Host-only control selection. Never inject this result into the blind rules.
#include "quactlize/dispatch/selection.hpp"
#include "quactlize/dispatch/cost.hpp"
#include <iomanip>
#include <iostream>

int main() {
    using namespace quactlize::dispatch;
    int q, n, k, tokens, experts, topk, compute;
    while (std::cin >> q >> n >> k >> tokens >> experts >> topk >> compute) {
        const int route = experts == 1 ? 0 : 2;
        qks_request_v1 request{1, sizeof(request), q, route + (q == 8), tokens * topk,
            n, k, experts, tokens, q == 8 ? q8_kpack2::kMappingId :
            q == 12 ? UINT64_C(0x51344b5034540001) : UINT64_C(0x514b504b54000001)};
        qks_prefill_choice_v1 full{};
        const int cost_status = cost::query(request, 7, full);
        const bool expanded = cost_status == QKS_OK && full.route == 2;
        if (cost_status == QKS_OK && !expanded) request.route = route + full.route;
        auto selected = expanded ? TcDecision{} : experts == 1 ?
            select_dense_dynamic(request, compute) : select_tc(request, false, compute);
        std::cout << "{\"request\":[" << q << ',' << n << ',' << k << ',' << tokens << ','
            << experts << ',' << topk << ',' << compute << "],\"status\":"
            << (expanded || selected ? 0 : QKS_MISS) << ",\"route\":"
            << (expanded ? 2 : request.route % 2) << ",\"dequant_config\":"
            << (cost_status == QKS_OK ? full.dequant_config : -1)
            << ",\"cost_policy_status\":" << cost_status;
        if (selected) {
            auto c = selected.config();
            std::cout << ",\"policy\":" << selected.policy << ",\"parent\":{\"symbol\":"
                << std::quoted(c.symbol) << ",\"qtype\":" << c.qtype << ",\"route\":" << c.route
                << ",\"tm\":" << c.tm << ",\"tn\":" << c.tn << ",\"tk\":" << c.tk
                << ",\"wm\":" << c.wm << ",\"wn\":" << c.wn << ",\"stages\":" << c.stages
                << ",\"ap\":" << c.ap << ",\"dn\":" << c.dn
                << ",\"persistent\":" << c.parent_persistent << "},\"split\":" << c.split
                << ",\"grid_mode\":" << c.grid_mode << ",\"grid_b\":" << c.grid_b;
        }
        std::cout << "}\n";
    }
    return std::cin.eof() ? 0 : 1;
}
