// GENERATED-INCLUDED per-unit body for the GEMV perf sweep. One unit = one (format, layout, StepK, Threads,
// CtaN, Chunk). The generator defines UNIT_* and includes this.
//
// The body must be a TEMPLATE. `if constexpr (false)` around a call does NOT suppress instantiation in a plain
// function -- only inside a templated entity -- and an illegal (StepK, Threads) pair would then trip a
// static_assert in a unit that is never meant to run. Same trap the MoE sweep's unit body documents.
#include "gemv_perf_common.hpp"

template <typename D, bool Grouped, int CtaM, int LastCtaM>
static void unit_ctam_range(Shape const& sh, Bufs const& b, Best& best, SweepRuntime& sweep) {
  run_row_exact<D, Grouped, CtaM, UNIT_CN, UNIT_CH>(sh, b, best, sweep);
  if constexpr (CtaM < LastCtaM)
    unit_ctam_range<D, Grouped, CtaM + 1, LastCtaM>(sh, b, best, sweep);
}

template <int Dummy>
static void unit_body(Shape const& sh, Bufs const& b, Best& best, SweepRuntime& sweep) {
  using D = KernelDetails<FP16DetailsA, UNIT_FMT, UNIT_LAY, UNIT_SK, UNIT_TH, UNIT_TS>;
  static_assert(tactic_format_of(UNIT_FMT) == ppu_gemv::tactic_space::Format::UNIT_AUTH_FORMAT,
                "CMake authority format and KernelDetails format disagree");
  static_assert(tactic_layout_of(UNIT_LAY) == ppu_gemv::tactic_space::Layout::UNIT_AUTH_LAYOUT,
                "CMake authority layout and KernelDetails layout disagree");
  static_assert((UNIT_LAY == WLayout::Native ? 0 : UNIT_TS) == UNIT_ARTIFACT_TK,
                "native identity must be TileK=0; tiled identity must name the artifact TileK");
  if (sh.experts > 0)
    unit_ctam_range<D, true, 1, GEMV_GROUPED_CTAM_MAX>(sh, b, best, sweep);
  else
    unit_ctam_range<D, false, 1, GEMV_CTAM_MAX>(sh, b, best, sweep);
}

void UNIT_FN(Shape const& sh, Bufs const& b, Best& best, SweepRuntime& sweep) {
  unit_body<0>(sh, b, best, sweep);
}
