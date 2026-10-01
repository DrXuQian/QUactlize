#pragma once

#include "cute/layout.hpp"

// WHERE PLANE 2's BITS COME FROM, as one object. This rule existed in TWO
// copies -- the unchunked converter and the chunked one -- and only the first
// was fixed for the per-plane fold. The stale copy read vregs {0,2} and never
// touched {1,3} whenever P2_DIV == 1, so half the tile's high bits could not
// arrive; it cost a box round and showed up as bad ~= 15000/32768.
//
// Written as LAYOUTS over (ii, k_block) rather than arithmetic, because
// arithmetic is what gets copied:
//   slot(ii)     = ii % N2
//   base(ii,kb)  = kb % P2_DIV + P2_DIV * (ii / N2)
// The converter then indexes hi[base + 2*(v>>1)]. Gated for both formulas in
// fold_derivation/l63 and consumed independently by l123's host-only topology
// oracle, so the probe and production cannot drift into two transcriptions.
template <int N2, int CPY_N, int P2_DIV>
struct HiPlaneSrc {
  static_assert(N2 >= 1 && CPY_N % N2 == 0, "plane 2's N extent must divide the delivery count");
  using SlotL = cute::Layout<cute::Shape <cute::Shape<cute::Int<N2>, cute::Int<CPY_N / N2>>>,
                             cute::Stride<cute::Stride<cute::_1,    cute::_0>>>;
  using BaseL = cute::Layout<cute::Shape <cute::Shape<cute::Int<N2>, cute::Int<CPY_N / N2>>, cute::Int<P2_DIV>>,
                             cute::Stride<cute::Stride<cute::_0,     cute::Int<P2_DIV>>,     cute::_1>>;
  static constexpr int slot(int ii)         { return int(SlotL{}(ii)); }
  static constexpr int base(int ii, int kb) { return int(BaseL{}(ii, kb % P2_DIV)); }
};
