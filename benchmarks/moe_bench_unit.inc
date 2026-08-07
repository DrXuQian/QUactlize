// ONE RUNTIME SWEEP UNIT = ONE SHAPE, all of its stage counts. A generated moe_units/*.cu now includes this file for
// several shapes with the same PPU_B_CHUNK policy, so every helper and static-init object below must be keyed by UNIT_FN
// and every per-shape macro must be cleared at the end of the inclusion.
//
// WHY THE UNIT IS A SHAPE AND NOT A FORMAT. Splitting by translation unit is the only way to make the sweep compile in
// parallel (cutlass_build_dev_kernels emits one add_custom_command per .cu), and the granularity is chosen by MEASUREMENT
// rather than taste. Locally, the header alone costs ~5 s to parse and each two-plane kernel adds ~2.5 s, so wall clock
// for a unit is roughly 5 + 2.5*kernels: 42 kernels is ~109 s, 3 kernels is ~12 s.
//
// The axis that must NOT be split is the stage count, and that is a RUNTIME cost, not a compile one: the offline pack is
// per SHAPE, and stages do not change the shape, so all three stage counts share one pack. Measured on this machine, one
// shape costs 31 ms to pack for a two-plane format (19.2 ms int2 + 11.8 ms int1) and 24 ms single-plane; at ~112 shapes
// that is ~3 s, already the same order as the GPU time, so splitting stages into their own units would triple it for
// nothing. TileM, TileN, WarpM and WarpN are all part of the shape, so splitting on those is free.
//
// The (TileM, WarpM) L-shape is NOT enumerated anywhere: moe_ok's WM <= TM rejects the illegal corner, so a unit for
// (TM=32, WM=64) simply compiles to an empty function. One less relation written down by hand.
#include "lowbit_moe_bench.hpp"
// THE OPTIONAL COLLECTIVES, BECAUSE THIS FILE IS WHERE THE KERNELS ARE INSTANTIATED. quactlize_actlize.hpp carries the
// base and deliberately leaves these two out so a consumer names what it builds; test_lowbit_moe_bench.cu was given
// them and this file was not, which is the wrong way round -- the bench TU only DECLARES the unit functions (through
// the generated moe_bench_units.inc) while every actual CollectiveMma comes from here. The build died at
// collective_mma_decl.hpp:61 with "Could not find a mainloop specialization" for MainloopPPUAiuFold, which is exactly
// the loud incomplete-type failure the umbrella's design predicts: no silent fallback, but no protection either.
//
// BOTH, unconditionally, matching the sibling bench. UNIT_2PLANE selects between MOE1 and MOE2 so the plane count is
// known here, but FOLD is not a property of the format -- it comes out of the tactic (ArtifactLowFold), which this
// file cannot see. Picking one header per unit would need the fold predicate at preprocessor level; the sweep builds
// every shape anyway, so there is nothing to save.
#include "quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_mixed_input_2plane.hpp"

#define MOE_UNIT_STR2_(x) #x
#define MOE_UNIT_STR_(x)  MOE_UNIT_STR2_(x)
#define MOE_UNIT_CAT2_(a, b) a##b
#define MOE_UNIT_CAT_(a, b) MOE_UNIT_CAT2_(a, b)

#if !defined(UNIT_NAME) || !defined(UNIT_FN) || !defined(UNIT_B_CHUNK) || !defined(UNIT_2PLANE) || \
    !defined(UNIT_LOELEM) || !defined(UNIT_HIELEM) || !defined(UNIT_LOB) || !defined(UNIT_HIB) || \
    !defined(UNIT_TM) || !defined(UNIT_TN) || !defined(UNIT_TK) || !defined(UNIT_WM) || !defined(UNIT_WN)
#error "the MoE generator must define every UNIT_* field before including moe_bench_unit.inc"
#endif

// THE BODY MUST BE A TEMPLATE, and this is not stylistic. `if constexpr (false)` suppresses instantiation of the
// discarded branch ONLY inside a templated entity; in a plain function the branch is still instantiated, so moe_ok
// returning false does NOT make an illegal shape compile away. Measured: of four probe units, the three legal ones built
// clean and (TileM=32, WarpM=64) produced 99 errors headed by `gemm_operands.hpp: division by zero` -- warpOnM = TM/WM =
// 0, so the collective builder degrades to `int`, which is the failure already recorded in the skill for TM=16/WM=32.
//
// It is the same rule that made a static_assert fire from inside a discarded branch earlier in this work, so the evidence
// for it was already in hand; only the consequence was missed.
// TileK COMES FROM THE ROW, NOT FROM THE BUILD KNOB. It used to be MOE_TK (64 by default), while the shapes come from
// the emitted grouped tables whose tactic TileK is 256 for Q3/Q5, 128 for Q6/i2 and 64 only for i4. moe_ok<> takes TK,
// so four formats out of five were asking a legality question about a TileK their shape did not come from: the
// `if constexpr` discarded the body and 312 of 415 units compiled to EMPTY FUNCTIONS -- a sweep that looks covered and
// measures nothing. The tell is compile time, since an empty unit still parses the headers: 5.4 s for a discarded q3
// unit against 30.4 s for a live i4 one.
//
// MOE_TK REMAINS for the non-sharded callers of this header, and this file no longer reads it.
#ifndef UNIT_TK
#error "UNIT_TK is not defined -- the generator must supply the row's TileK. Falling back to MOE_TK is what silently emptied the sweep."
#endif
template <int Dummy>
static void MOE_UNIT_CAT_(UNIT_FN, _body)(const Band& bd, Best& best) {
#if UNIT_2PLANE
  MOE2(bd, best, MOE_UNIT_STR_(UNIT_NAME), UNIT_LOELEM, UNIT_HIELEM, UNIT_LOB, UNIT_HIB,
       UNIT_TM, UNIT_TN, UNIT_TK, UNIT_WM, UNIT_WN);
#else
  MOE1(bd, best, MOE_UNIT_STR_(UNIT_NAME), UNIT_LOELEM, UNIT_LOB,
       UNIT_TM, UNIT_TN, UNIT_TK, UNIT_WM, UNIT_WN);
#endif
}
void UNIT_FN(const Band& bd, Best& best) { MOE_UNIT_CAT_(UNIT_FN, _body)<0>(bd, best); }

// Static-init tally of the requested row axis; effectiveness is reported from Policy::Descriptor in every row tag.
namespace {
const int MOE_UNIT_CAT_(UNIT_FN, _vote) = (moe_chunk_vote(UNIT_B_CHUNK), 0);
}

#undef MOE_UNIT_CAT_
#undef MOE_UNIT_CAT2_
#undef MOE_UNIT_STR_
#undef MOE_UNIT_STR2_
#undef UNIT_WN
#undef UNIT_WM
#undef UNIT_TK
#undef UNIT_TN
#undef UNIT_TM
#undef UNIT_HIB
#undef UNIT_LOB
#undef UNIT_HIELEM
#undef UNIT_LOELEM
#undef UNIT_2PLANE
#undef UNIT_B_CHUNK
#undef UNIT_FN
#undef UNIT_NAME
