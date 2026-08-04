// Host gate for the pinned token->top-k routing fixture. Its Mmax ladder is part of the sweep definition because it
// decides whether compact A capacity 1, 2, 4, or none is reachable.
#include <cstdio>

#include "moe_router_fixture.hpp"

int main() {
  int const tokens[] = {1, 2, 4, 64, 2048, 4096};
  int const want_max[] = {1, 2, 3, 12, 239, 447};
  int const want_capacity[] = {1, 2, 4, 0, 0, 0};
  int bad = 0;
  for (int i = 0; i < 6; ++i) {
    moe_router_fixture::Rows r;
    char why[96] = "";
    bool const ok = moe_router_fixture::route(tokens[i], 8, 256, r, why, sizeof why);
    int const cap = moe_router_fixture::minimum_compact_capacity(r.max);
    if (!ok || r.total != tokens[i] * 8 || r.max != want_max[i] || cap != want_capacity[i]) ++bad;
    std::printf("T=%-4d total=%-5d active=%-3d Mmax=%-3d compact=%d%s\n",
                tokens[i], r.total, r.active, r.max, cap, ok ? "" : why);
  }
  std::printf("%s bad=%d\n", moe_router_fixture::kName, bad);
  return bad ? 1 : 0;
}
