// Host dispatch. Structured as the reference's nested check_* chain so each runtime axis collapses to one
// template argument, with two deliberate differences:
//
//   * the group-size list is a MACRO, so a build that only ships gs=32 and gs=128 can drop the rest. Every
//     entry is a full kernel instantiation and this project has already paid for a sweep that instantiated
//     more than it needed.
//   * every refusal goes through gemv_refuse(), which bumps a counter. A harness that times a launch which
//     never happened will otherwise rank it fastest -- that has happened here twice.
#pragma once

#include "gemv_kernel.hpp"

namespace ppu_gemv {

// Which group sizes get instantiated. 16 is Q3_K's, 32 is the fine Q4_K/Q6_K one, 128 is GPTQ's, 0 is
// per-column. Narrow this in a build that does not need all of them.
#ifndef GEMV_GS_LIST
#define GEMV_GS_LIST(EMIT) EMIT(0) EMIT(16) EMIT(32) EMIT(64) EMIT(128)
#endif

// Dense rows per CTA. TRT-LLM's CUDA-core tactic uses one exact CtaM specialization for every M in [1,15]:
// the quantized weight load/conversion is then shared by every activation row in that small-M launch. Keep M=1
// as its own specialization -- it is the measured decode winner, not a tail of a larger fixed-register kernel.
// Larger dense problems remain legal and tile this maximum; the tuner owns TRT-LLM's profiling-cost prune at M>=16.
#ifndef GEMV_CTAM_MAX
#define GEMV_CTAM_MAX 15
#endif
static_assert(GEMV_CTAM_MAX >= 1 && GEMV_CTAM_MAX <= 15,
              "GEMV_CTAM_MAX must select a compiled TRT-style CtaM specialization in [1,15]");

// Grouped routing gets its parallelism from grid.z and retains the proven four-row tile. This is a compile-volume
// boundary as well as a policy: extending dense M must not instantiate eleven unused grouped variants per format.
#ifndef GEMV_GROUPED_CTAM_MAX
#define GEMV_GROUPED_CTAM_MAX 4
#endif
static_assert(GEMV_GROUPED_CTAM_MAX >= 1 && GEMV_GROUPED_CTAM_MAX <= GEMV_CTAM_MAX,
              "grouped CtaM maximum must name one of the compiled dense specializations");

// The shipping scale-first ABI has no bias pointer. Correctness sweeps leave this enabled to cover the generic
// library; a narrow shipping/perf translation unit may define it to zero and avoid dead bias instantiations.
#ifndef GEMV_ENABLE_BIAS
#define GEMV_ENABLE_BIAS 1
#endif

// Which quant ops get instantiated, for the same reason as GEMV_GS_LIST: every entry is a full set of kernels
// and a perf sweep that only ever calls two of the three should not pay for the third. Default is all three.
#ifndef GEMV_QUANT_LIST
#define GEMV_QUANT_LIST(EMIT, G)                                                                   \
  EMIT(QuantOp::PerColScaleOnly, G) EMIT(QuantOp::FinegrainedScaleOnly, G) EMIT(QuantOp::FinegrainedScaleZero, G)
#endif

// ---------------------------------------------------------------------------------------------------
// Compile-time legality of one (GS, QuantOp, StepK) triple. Checked with if constexpr inside a TEMPLATE so
// the illegal combinations are never instantiated -- an if constexpr in a plain function does not suppress
// instantiation, which is a trap this codebase has hit before.
template <typename Details, int GS, QuantOp QOp>
constexpr bool gemv_combo_ok() {
  if constexpr (QOp == QuantOp::PerColScaleOnly) return GS == 0;
  else if constexpr (GS == 0) return false;
  else return gs_step_ok<GS, Details::kStepK, Details::kCtaK>();
}

template <typename Details>
bool gemv_quant_compiled(Params const& p) {
#define GEMV_MATCH_QGS(QO, G)                                                                    \
  if (p.quant == (QO) && p.groupsize == (G)) return gemv_combo_ok<Details, (G), (QO)>();
#define GEMV_MATCH_GS(G) GEMV_QUANT_LIST(GEMV_MATCH_QGS, G)
  GEMV_GS_LIST(GEMV_MATCH_GS)
#undef GEMV_MATCH_GS
#undef GEMV_MATCH_QGS
  return false;
}

// Host-only dry run used by the public config-valid ABI. Keeping these checks here means the query asks the same
// template instantiation that launches; it does not transcribe CtaK, CtaN, the compiled GS list, or the row ceiling.
template <typename Details, int CtaN>
char const* gemv_config_invalid_reason(Params const& p) {
  if (p.format != Details::kFormat) return "format mismatch";
  if (p.layout != Details::kLayout) return "layout mismatch";
  if (p.is_bf16 != Details::ADetails::kIsBF16) return "A type mismatch";
  if (p.act_scale) return "act_scale is not instantiated";
  if (!GEMV_ENABLE_BIAS && p.bias) return "bias is not instantiated";
  if (p.n % CtaN) return "n must be a multiple of CtaN";
  if (p.k <= 0 || p.k % Details::kStepK) return "k must contain whole per-thread StepK accesses";
  if (p.groupsize > 0 && p.k % p.groupsize) return "k must contain whole quantization groups";
  if constexpr (Details::kLayout == WLayout::TileK) {
    if (p.k % Details::kTileSizeK) return "k must contain whole TileK layout tiles";
  }
  if (is_two_plane(Details::kFormat) && !p.weight_hi) return "two-plane format needs weight_hi";
  if (has_zero(p.quant) && !p.zeros) return "ScaleZero needs zeros";
  if (!gemv_quant_compiled<Details>(p)) return "quant/group-size combination is not compiled";

  int const rows_max = p.num_experts > 0 ? p.max_rows : p.m;
  if (p.num_experts > 0 && !p.row_offsets) return "grouped needs row_offsets";
  if (rows_max <= 0) return "no rows";
  int const max_cta_m = p.num_experts > 0 ? GEMV_GROUPED_CTAM_MAX : GEMV_CTAM_MAX;
  if (rows_max > max_cta_m * 4096) return "rows out of range";
  return nullptr;
}

template <typename Details, int CtaM, int CtaN, int Chunk, int GS, QuantOp QOp, bool EnableBias, bool Grouped>
void gemv_exec(Params const& p, KernelArgs const& args, int grid_m, gemv_stream_t s) {
  dim3 const grid(grid_m, p.n / CtaN, Grouped ? p.num_experts : 1);
  dim3 const block(Details::kThreads);
  if (p.k % Details::kCtaK) {
    gemv_kernel<Details, CtaM, CtaN, Chunk, GS, QOp, /*EnableActScale=*/false, EnableBias,
                /*ApplyAlphaInAdvance=*/false, /*PredicatedKTail=*/true, Grouped><<<grid, block, 0, s>>>(args);
  } else {
    // Keep the measured M=1/divisible-K specialization free of tail predicates and their loop branch.
    gemv_kernel<Details, CtaM, CtaN, Chunk, GS, QOp, /*EnableActScale=*/false, EnableBias,
                /*ApplyAlphaInAdvance=*/false, /*PredicatedKTail=*/false, Grouped><<<grid, block, 0, s>>>(args);
  }
}

// ---- CtaM ----
template <typename Details, int CtaN, int Chunk, int GS, QuantOp QOp, bool EnableBias, bool Grouped>
bool gemv_dispatch_ctam(Params const& p, KernelArgs const& args, int rows_max, gemv_stream_t s) {
#define GEMV_DISPATCH_MAX (Grouped ? GEMV_GROUPED_CTAM_MAX : GEMV_CTAM_MAX)
#define GEMV_TRY_CTAM(CM)                                                                        \
  if constexpr ((CM) <= GEMV_DISPATCH_MAX) {                                                       \
    if (rows_max <= (CM) || (CM) == GEMV_DISPATCH_MAX) {                                           \
      constexpr int _cm = (CM);                                                                    \
      int const gm = (rows_max + _cm - 1) / _cm;                                                   \
      gemv_exec<Details, _cm, CtaN, Chunk, GS, QOp, EnableBias, Grouped>(p, args, gm, s);           \
      return true;                                                                                 \
    }                                                                                              \
  }
  GEMV_TRY_CTAM(1)
#if GEMV_CTAM_MAX >= 2
  GEMV_TRY_CTAM(2)
#endif
#if GEMV_CTAM_MAX >= 3
  GEMV_TRY_CTAM(3)
#endif
#if GEMV_CTAM_MAX >= 4
  GEMV_TRY_CTAM(4)
#endif
#if GEMV_CTAM_MAX >= 5
  GEMV_TRY_CTAM(5)
#endif
#if GEMV_CTAM_MAX >= 6
  GEMV_TRY_CTAM(6)
#endif
#if GEMV_CTAM_MAX >= 7
  GEMV_TRY_CTAM(7)
#endif
#if GEMV_CTAM_MAX >= 8
  GEMV_TRY_CTAM(8)
#endif
#if GEMV_CTAM_MAX >= 9
  GEMV_TRY_CTAM(9)
#endif
#if GEMV_CTAM_MAX >= 10
  GEMV_TRY_CTAM(10)
#endif
#if GEMV_CTAM_MAX >= 11
  GEMV_TRY_CTAM(11)
#endif
#if GEMV_CTAM_MAX >= 12
  GEMV_TRY_CTAM(12)
#endif
#if GEMV_CTAM_MAX >= 13
  GEMV_TRY_CTAM(13)
#endif
#if GEMV_CTAM_MAX >= 14
  GEMV_TRY_CTAM(14)
#endif
#if GEMV_CTAM_MAX >= 15
  GEMV_TRY_CTAM(15)
#endif
#undef GEMV_TRY_CTAM
#undef GEMV_DISPATCH_MAX
  return false;
}

// ---- bias ----
template <typename Details, int CtaN, int Chunk, int GS, QuantOp QOp, bool Grouped>
bool gemv_dispatch_bias(Params const& p, KernelArgs const& args, int rows_max, gemv_stream_t s) {
#if GEMV_ENABLE_BIAS
  if (p.bias) return gemv_dispatch_ctam<Details, CtaN, Chunk, GS, QOp, true, Grouped>(p, args, rows_max, s);
#endif
  return gemv_dispatch_ctam<Details, CtaN, Chunk, GS, QOp, false, Grouped>(p, args, rows_max, s);
}

// ---- grouped ----
template <typename Details, int CtaN, int Chunk, int GS, QuantOp QOp>
bool gemv_dispatch_grouped(Params const& p, KernelArgs const& args, int rows_max, gemv_stream_t s) {
  if (p.num_experts > 0)
    return gemv_dispatch_bias<Details, CtaN, Chunk, GS, QOp, true>(p, args, rows_max, s);
  return gemv_dispatch_bias<Details, CtaN, Chunk, GS, QOp, false>(p, args, rows_max, s);
}

// ---- (quant op, group size) ----
template <typename Details, int CtaN, int Chunk>
bool gemv_dispatch_quant(Params const& p, KernelArgs const& args, int rows_max, gemv_stream_t s) {
#define GEMV_TRY_QGS(QO, G)                                                                        \
  if (p.quant == (QO) && p.groupsize == (G)) {                                                     \
    if constexpr (gemv_combo_ok<Details, (G), (QO)>())                                             \
      return gemv_dispatch_grouped<Details, CtaN, Chunk, (G), (QO)>(p, args, rows_max, s);         \
    else { gemv_refuse("group size illegal for this StepK"); return false; }                       \
  }
#define GEMV_EMIT_GS(G) GEMV_QUANT_LIST(GEMV_TRY_QGS, G)
  GEMV_GS_LIST(GEMV_EMIT_GS)
#undef GEMV_EMIT_GS
#undef GEMV_TRY_QGS
  gemv_refuse("group size not instantiated (see GEMV_GS_LIST)");
  return false;
}

// ---------------------------------------------------------------------------------------------------
// Entry point for one compiled shape. Returns false (and bumps gemv_fail_count) without launching if the
// problem does not fit the instantiation.
template <typename Details, int CtaN, int Chunk>
bool launch_gemv(Params const& p, gemv_stream_t s) {
  // The buffer's own record wins over anything the caller claims in Params.
  if (p.record) {
    char const* why = "";
    if (!wfmt_matches<Details>(*reinterpret_cast<WeightFormatRecord const*>(p.record),
                              p.n, p.k, p.groupsize, p.quant, &why)) {
      char msg[128];
      std::snprintf(msg, sizeof(msg), "weight format record disagrees: %s", why);
      gemv_refuse(msg);
      return false;
    }
  }
  if (char const* why = gemv_config_invalid_reason<Details, CtaN>(p)) {
    gemv_refuse(why);
    return false;
  }
  int const rows_max = p.num_experts > 0 ? p.max_rows : p.m;

  KernelArgs args{};
  args.act = p.act;  args.act_scale = p.act_scale;
  args.w_lo = p.weight;  args.w_hi = p.weight_hi;
  args.scales = p.scales;  args.zeros = p.zeros;  args.bias = p.bias;
  args.out = p.out;  args.alpha = p.alpha;
  args.n = p.n;  args.k = p.k;  args.rows = p.m;
  args.row_offsets = p.row_offsets;
  args.w_lo_stride_e = p.w_bytes_per_expert;
  args.w_hi_stride_e = p.w_hi_bytes_per_expert;
  args.scale_stride_e = p.scale_elems_per_expert;
  args.lo_s = Details::LoLayout::strides(p.n, p.k);
  args.hi_s = Details::HiLayout::strides(p.n, p.k);

  return gemv_dispatch_quant<Details, CtaN, Chunk>(p, args, rows_max, s);
}

}  // namespace ppu_gemv
