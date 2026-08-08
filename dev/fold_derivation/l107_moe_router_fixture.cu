// Host gate for the pinned token->top-k routing fixture. Its active-expert and Mmax ladders are part of the sweep
// definition; compact-A was deleted in 3fdc155, so this gate must not retain that retired policy's API or verdicts.
#include <cstdio>

#include "moe_router_fixture.hpp"

int main() {
  int const tokens[] = {1, 2, 4, 64, 2048, 4096};
  int const want_max[] = {1, 2, 3, 12, 239, 447};
  int const want_active[] = {8, 15, 30, 212, 256, 256};
  int bad = 0;
  for (int i = 0; i < 6; ++i) {
    moe_router_fixture::Rows r;
    char why[96] = "";
    bool const ok = moe_router_fixture::route(tokens[i], 8, 256, r, why, sizeof why);
    if (!ok || r.total != tokens[i] * 8 || r.active != want_active[i] || r.max != want_max[i]) ++bad;
    std::printf("T=%-4d total=%-5d active=%-3d Mmax=%-3d%s\n",
                tokens[i], r.total, r.active, r.max, ok ? "" : why);
  }
  std::printf("%s bad=%d\n", moe_router_fixture::kName, bad);
  return bad ? 1 : 0;
}
