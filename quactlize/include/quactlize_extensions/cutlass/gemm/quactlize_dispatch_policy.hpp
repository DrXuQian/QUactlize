/***************************************************************************************************
 * quactlize dispatch policies. These lived as +130 lines inside actlize's dispatch_policy.hpp until
 * 2026-08-06; they are quactlize's, and this is where they belong.
 **************************************************************************************************/
// WHAT MAKES THESE OURS RATHER THAN actlize's. Upstream actlize (v1.0.0) has no such template: no fold schedule,
// no second B plane, no gs=32 arm. They were written for GGUF k-quant weights and they encode quactlize policy.
// Keeping them in the vendor's header meant every actlize build carried them and no boundary said otherwise.
//
// WHY MainloopQuactlizeMixedInput IS A NEW NAME AND THE OTHER TWO ARE NOT. quactlize's mixed-input collective
// used to specialise CollectiveMma on actlize's OWN MainloopPPUAiuMixedInput, for every Schedule -- which is a
// redefinition of actlize's specialisation, not an addition. Both headers in one translation unit do not
// compile, and the workaround (a quactlize umbrella that swaps actlize's collective out of the include list)
// fails for an unrelated reason: actlize's umbrella is cyclic through gemm_configs.hpp, so the original comes
// back in regardless. Under a distinct tag our collective is purely additive, actlize's specialisation stays
// reachable for actlize's own callers, and the umbrella can include the vendor's list unmodified.
//
// MainloopPPUAiuMixedInput2Plane and MainloopPPUAiuFold keep their spelling because they never collided --
// upstream has no template of either name. They are ours by origin, not by prefix. ci/check_actlize_pristine.py
// carries that fact as a list, since the names themselves do not say it.
//
// WHAT IS DELIBERATELY NOT COPIED HERE. actlize's own MainloopPPUAiuMixedInput and its four Schedule arms stay
// in the vendor header and stay in use by actlize. This file mirrors their SHAPE (same Schedule tags, same
// StaticGroupSize encoding: 0 default, -1 per-column, otherwise the group size) because quactlize's collective
// reads those members, not because the two are meant to be interchangeable.
#pragma once

#include "cutlass/gemm/dispatch_policy.hpp"   // actlize's, for the Schedule tags we specialise on

//////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm {

//////////////////////////////////////////////////////////////////////////////
// Schedules
//////////////////////////////////////////////////////////////////////////////

struct KernelAiuMultistageMixedInputFinegrainedGs32 { };  // gs=32 (Q4_0/Q4_1/Q4_K-as-AWQ)
struct KernelAiuMultistageMixedInputFinegrainedGs16 { };  // gs=16 (Q2/Q3/Q6 k-quants)

// Artifact-fold schedule wrapper. The folds describe the resident B planes, not the consumer TileShape.K: a tactic
// with a larger TileK may read the same bytes, but it must keep both physical (N/F, F*K) descriptors. The low fold
// selects the ordinary-vs-folded collective; the independent high fold sizes the second plane. ArtifactTileK is the
// resident delivery width that cannot be recovered from F when F==1. Keep BaseSchedule_ in the middle and append the
// new value with a zero default so existing KernelAiuFold<F[, Base[, HighFold]]> spellings remain source compatible.
template<int ArtifactLowFold_, class BaseSchedule_ = KernelAiuMultistageMixedInputFinegrainedGs32,
         int ArtifactHighFold_ = 0, int ArtifactTileK_ = 0>
struct KernelAiuFold {
  static constexpr int FoldF = ArtifactLowFold_;  // compatibility for downstream users of the old name
  static constexpr int ArtifactLowFold = ArtifactLowFold_;
  static constexpr int ArtifactHighFold = ArtifactHighFold_;
  static constexpr int ArtifactTileK = ArtifactTileK_;
  using BaseSchedule = BaseSchedule_;
};

// A provider choice is orthogonal to the resident-B fold contract.  Keep it in a distinct wrapper instead of
// adding another defaulted argument to KernelAiuFold: every existing schedule type (and therefore every M>1
// shipping kernel type) remains exactly the type it was before the M==1 packed-row provider existed.  The wrapper
// is introduced only by ppu_mixed_policy::PackedAMainloopPolicy.
template<int Rows_, class WrappedSchedule_>
struct KernelAiuPackedA {
  static_assert(Rows_ > 0, "KernelAiuPackedA requires a positive compile-time row count");
  static constexpr int Rows = Rows_;
  using WrappedSchedule = WrappedSchedule_;
};

template<class T> struct a_provider_schedule_traits {
  static constexpr int Rows = 0;
  using Wrapped = T;
};
template<int Rows_, class WrappedSchedule_>
struct a_provider_schedule_traits<KernelAiuPackedA<Rows_, WrappedSchedule_>> {
  static constexpr int Rows = Rows_;
  using Wrapped = WrappedSchedule_;
};
// A zero fold means that no artifact contract was supplied. The builder retains its legacy derivation only for such
// direct CollectiveBuilder users; the shared quactlize policy always supplies both folds when either plane is folded.
template<class T> struct fold_schedule_traits {
  static constexpr int FoldF = 0;
  static constexpr int ArtifactLowFold = 0;
  static constexpr int ArtifactHighFold = 0;
  static constexpr int ArtifactTileK = 0;
  using Base = T;
};
template<int LowFold, class B, int HighFold, int ArtifactTileK_>
struct fold_schedule_traits<KernelAiuFold<LowFold, B, HighFold, ArtifactTileK_>> {
  static constexpr int FoldF = LowFold;
  static constexpr int ArtifactLowFold = LowFold;
  static constexpr int ArtifactHighFold = HighFold;
  static constexpr int ArtifactTileK = ArtifactTileK_;
  using Base = B;
};
template<int Rows_, class WrappedSchedule_>
struct fold_schedule_traits<KernelAiuPackedA<Rows_, WrappedSchedule_>> {
private:
  using WrappedTraits = fold_schedule_traits<WrappedSchedule_>;
public:
  static constexpr int FoldF = WrappedTraits::FoldF;
  static constexpr int ArtifactLowFold = WrappedTraits::ArtifactLowFold;
  static constexpr int ArtifactHighFold = WrappedTraits::ArtifactHighFold;
  static constexpr int ArtifactTileK = WrappedTraits::ArtifactTileK;
  // Preserve the A-provider wrapper after removing the artifact-fold wrapper.  This lets the ordinary one-plane
  // dispatch keep its exact group-size schedule while CollectiveMma can still see Rows at compile time.
  using Base = KernelAiuPackedA<Rows_, typename WrappedTraits::Base>;
};

//////////////////////////////////////////////////////////////////////////////
// Single-plane mixed input, quactlize's collective
//////////////////////////////////////////////////////////////////////////////
// Same four Schedule arms as actlize's MainloopPPUAiuMixedInput plus the gs=32 one it lacks. The arms exist to
// carry StaticGroupSize as a compile-time constant: 0 means "runtime group size", -1 means per-column, and any
// other value is the static group. quactlize's collective branches on it, so an arm that is merely absent selects
// the primary and silently runs the runtime-group path.

template<int Stages_, class kContinous_, typename Schedule_ = KernelAiuMultistageMixedInput>
struct MainloopQuactlizeMixedInput {
  constexpr static int Stages = Stages_;
  constexpr static int StaticGroupSize = 0;  // default value
  using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_, class kContinous_>
struct MainloopQuactlizeMixedInput<Stages_, kContinous_, KernelAiuMultistageMixedInputPerCol> {
  constexpr static int Stages = Stages_;
  constexpr static int StaticGroupSize = -1;
  using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_, class kContinous_>
struct MainloopQuactlizeMixedInput<Stages_, kContinous_, KernelAiuMultistageMixedInputFinegrainedGs128> {
  constexpr static int Stages = Stages_;
  constexpr static int StaticGroupSize = 128;
  using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_, class kContinous_>
struct MainloopQuactlizeMixedInput<Stages_, kContinous_, KernelAiuMultistageMixedInputFinegrainedGs64> {
  constexpr static int Stages = Stages_;
  constexpr static int StaticGroupSize = 64;
  using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_, class kContinous_>
struct MainloopQuactlizeMixedInput<Stages_, kContinous_, KernelAiuMultistageMixedInputFinegrainedGs32> {
  constexpr static int Stages = Stages_;
  constexpr static int StaticGroupSize = 32;
  using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_, class kContinous_>
struct MainloopQuactlizeMixedInput<Stages_, kContinous_, KernelAiuMultistageMixedInputFinegrainedGs16> {
  constexpr static int Stages = Stages_;
  constexpr static int StaticGroupSize = 16;
  using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};

// Forward every ordinary schedule property (notably StaticGroupSize) and add only the independent A-provider
// identity.  This remains a MainloopQuactlizeMixedInput specialisation, so the existing collective is reused rather
// than copied and no folded/two-plane collective is touched.
template<int Stages_, class kContinous_, int Rows_, class WrappedSchedule_>
struct MainloopQuactlizeMixedInput<Stages_, kContinous_, KernelAiuPackedA<Rows_, WrappedSchedule_>>
    : MainloopQuactlizeMixedInput<Stages_, kContinous_, WrappedSchedule_> {
  static constexpr int AProviderRows = Rows_;
};

//////////////////////////////////////////////////////////////////////////////
// B BIT-PLANE CONCAT (Q3 = int2+int1, Q5 = int4+int1, Q6 = int4+int2)
//////////////////////////////////////////////////////////////////////////////
// A DISTINCT mainloop policy, so the two-B-plane collective (ppu_mma_aiu_mixed_input_2plane.hpp) is its OWN
// CollectiveMma specialization instead of competing with / being if-constexpr'd into the validated single-plane
// one. Mirrors the single-plane policy exactly -- including every per-Schedule StaticGroupSize specialization --
// because the 2-plane mainloop reuses the same scale/zero machinery (and the GGUF concats are gs=16, i.e. the
// FINE per-mma-atom scale path).
template<int Stages_, class kContinous_, int StaticGroupSize_,
         int ArtifactLowFold_, int ArtifactHighFold_, int ArtifactTileK_>
struct MainloopPPUAiuMixedInput2PlaneBase {
  constexpr static int Stages = Stages_;
  constexpr static int StaticGroupSize = StaticGroupSize_;
  constexpr static int ArtifactLowFold = ArtifactLowFold_;
  constexpr static int ArtifactHighFold = ArtifactHighFold_;
  constexpr static int ArtifactTileK = ArtifactTileK_;
  using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_, class kContinous_, typename Schedule_ = KernelAiuMultistageMixedInput,
         int ArtifactLowFold_ = 1, int ArtifactHighFold_ = 1, int ArtifactTileK_ = 0>
struct MainloopPPUAiuMixedInput2Plane
    : MainloopPPUAiuMixedInput2PlaneBase<Stages_, kContinous_, 0,
                                        ArtifactLowFold_, ArtifactHighFold_, ArtifactTileK_> {};

template<int Stages_, class kContinous_, int ArtifactLowFold_, int ArtifactHighFold_, int ArtifactTileK_>
struct MainloopPPUAiuMixedInput2Plane<Stages_, kContinous_, KernelAiuMultistageMixedInputPerCol,
                                     ArtifactLowFold_, ArtifactHighFold_, ArtifactTileK_>
    : MainloopPPUAiuMixedInput2PlaneBase<Stages_, kContinous_, -1,
                                        ArtifactLowFold_, ArtifactHighFold_, ArtifactTileK_> {};

template<int Stages_, class kContinous_, int ArtifactLowFold_, int ArtifactHighFold_, int ArtifactTileK_>
struct MainloopPPUAiuMixedInput2Plane<Stages_, kContinous_, KernelAiuMultistageMixedInputFinegrainedGs128,
                                     ArtifactLowFold_, ArtifactHighFold_, ArtifactTileK_>
    : MainloopPPUAiuMixedInput2PlaneBase<Stages_, kContinous_, 128,
                                        ArtifactLowFold_, ArtifactHighFold_, ArtifactTileK_> {};

template<int Stages_, class kContinous_, int ArtifactLowFold_, int ArtifactHighFold_, int ArtifactTileK_>
struct MainloopPPUAiuMixedInput2Plane<Stages_, kContinous_, KernelAiuMultistageMixedInputFinegrainedGs64,
                                     ArtifactLowFold_, ArtifactHighFold_, ArtifactTileK_>
    : MainloopPPUAiuMixedInput2PlaneBase<Stages_, kContinous_, 64,
                                        ArtifactLowFold_, ArtifactHighFold_, ArtifactTileK_> {};

template<int Stages_, class kContinous_, int ArtifactLowFold_, int ArtifactHighFold_, int ArtifactTileK_>
struct MainloopPPUAiuMixedInput2Plane<Stages_, kContinous_, KernelAiuMultistageMixedInputFinegrainedGs32,
                                     ArtifactLowFold_, ArtifactHighFold_, ArtifactTileK_>
    : MainloopPPUAiuMixedInput2PlaneBase<Stages_, kContinous_, 32,
                                        ArtifactLowFold_, ArtifactHighFold_, ArtifactTileK_> {};

template<int Stages_, class kContinous_, int ArtifactLowFold_, int ArtifactHighFold_, int ArtifactTileK_>
struct MainloopPPUAiuMixedInput2Plane<Stages_, kContinous_, KernelAiuMultistageMixedInputFinegrainedGs16,
                                     ArtifactLowFold_, ArtifactHighFold_, ArtifactTileK_>
    : MainloopPPUAiuMixedInput2PlaneBase<Stages_, kContinous_, 16,
                                        ArtifactLowFold_, ArtifactHighFold_, ArtifactTileK_> {};

//////////////////////////////////////////////////////////////////////////////
// N-FOLD (TK-freeing) mainloop policy
//////////////////////////////////////////////////////////////////////////////
// A DISTINCT policy so the fold collective (ppu_mma_aiu_fold.hpp) is its OWN CollectiveMma specialization, zero
// regression to the validated single-plane / 2-plane ones. FoldFactor F = 32B-elems / TileShape.K: the B plane's
// AIU contiguous run folds F adjacent N-columns to reach 32B while TileShape.K (A / MMA) stays small (A-smem =
// TileM*TK*2 shrinks by F). B smem K-extent = F * TileShape.K; mainloop runs F gemm passes (B's K-atom blocks,
// reusing one A) into an F*N-wide accumulator. Offline data prepared by nfold_column_pairs_ppu (P1.1). Mirrors the
// per-Schedule StaticGroupSize set (GGUF concats are gs=16 = FINE per-atom scale).
template<int Stages_, class kContinous_, int FoldF_ = 2, typename Schedule_ = KernelAiuMultistageMixedInput>
struct MainloopPPUAiuFold {
  constexpr static int Stages = Stages_;
  constexpr static int StaticGroupSize = 0;
  constexpr static int FoldFactor = FoldF_;
  using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};
template<int Stages_, class kContinous_, int FoldF_>
struct MainloopPPUAiuFold<Stages_, kContinous_, FoldF_, KernelAiuMultistageMixedInputPerCol> {
  constexpr static int Stages = Stages_; constexpr static int StaticGroupSize = -1;
  constexpr static int FoldFactor = FoldF_; using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput; using ClusterShape = Shape<_1,_1,_1>;
};
template<int Stages_, class kContinous_, int FoldF_>
struct MainloopPPUAiuFold<Stages_, kContinous_, FoldF_, KernelAiuMultistageMixedInputFinegrainedGs128> {
  constexpr static int Stages = Stages_; constexpr static int StaticGroupSize = 128;
  constexpr static int FoldFactor = FoldF_; using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput; using ClusterShape = Shape<_1,_1,_1>;
};
template<int Stages_, class kContinous_, int FoldF_>
struct MainloopPPUAiuFold<Stages_, kContinous_, FoldF_, KernelAiuMultistageMixedInputFinegrainedGs64> {
  constexpr static int Stages = Stages_; constexpr static int StaticGroupSize = 64;
  constexpr static int FoldFactor = FoldF_; using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput; using ClusterShape = Shape<_1,_1,_1>;
};
template<int Stages_, class kContinous_, int FoldF_>
struct MainloopPPUAiuFold<Stages_, kContinous_, FoldF_, KernelAiuMultistageMixedInputFinegrainedGs32> {
  constexpr static int Stages = Stages_; constexpr static int StaticGroupSize = 32;
  constexpr static int FoldFactor = FoldF_; using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput; using ClusterShape = Shape<_1,_1,_1>;
};
template<int Stages_, class kContinous_, int FoldF_>
struct MainloopPPUAiuFold<Stages_, kContinous_, FoldF_, KernelAiuMultistageMixedInputFinegrainedGs16> {
  constexpr static int Stages = Stages_; constexpr static int StaticGroupSize = 16;
  constexpr static int FoldFactor = FoldF_; using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput; using ClusterShape = Shape<_1,_1,_1>;
};

//////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm
