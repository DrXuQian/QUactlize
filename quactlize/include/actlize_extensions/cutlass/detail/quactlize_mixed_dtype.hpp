/***************************************************************************************************
 * quactlize's empty-zero-slot marker for the two-plane B element tuple.
 **************************************************************************************************/
// WHY THIS IS NOT IN actlize's detail/collective.hpp ANY MORE. It was, until 2026-08-06, next to the one change
// that genuinely belongs to actlize: `deduce_mixed_width_dtype`'s guard widened from index 2 to index 3, so the
// B element tuple can carry a fourth member. That widening is a generalisation of a vendor facility -- it only
// relaxes a static_assert whose body already returned void out of range -- and it stays there. NoZero is a
// convention of ours layered on top, used by nothing in actlize, so it lives here.
#pragma once

#include "cutlass/detail/collective.hpp"   // actlize's deduce_mixed_width_dtype_t, whose index 3 this relies on
#include "cute/util/type_traits.hpp"

namespace cutlass::gemm::collective {
namespace detail {

// AN EXPLICITLY EMPTY ZERO SLOT, for the 2-plane B tuple only.
//
// Everywhere else "no zero" is expressed by tuple LENGTH: <B, Scale> has none, <B, Scale, Zero> has one. The 2-plane
// path cannot use that convention, because its second B plane is deduced POSITIONALLY at index 3 -- so dropping the
// zero would leave <B, Scale, Plane>, where index 2 holds a plane and index 3 is void. Telling those two encodings
// apart from the types alone is impossible in general (a zero is fp16, a plane is a narrow int, and keying on that is
// exactly the kind of implicit rule that breaks silently later). cute::tuple cannot hold `void`, so the empty slot
// needs a real type: <B, Scale, NoZero, Plane> keeps the plane at 3 and says "no zero" out loud.
//
// This exists because QuantMode::FinegrainedScaleOnly ran as ScaleZero on the 2-plane path for as long as that path
// existed: the driver built the 4-tuple unconditionally, so every "ScaleOnly" 2-plane row -- including the ones in
// test_q3_bconcat_bench that were labelled as such -- was really an affine run whose zero cancelled the converter's
// bias. Measured, not guessed: with the bias moved to the converter the outputs came out equal to the UNBIASED golden,
// and the ratio (32-B)/(16-B) across the Q6 and Q5 rungs pinned B = 0.
struct NoZero {};

template <class T>
using strip_no_zero_t = cute::conditional_t<cute::is_same_v<T, NoZero>, void, T>;

} // namespace detail

// A SECOND B PLANE'S ATOMS, THROUGH A FIXED PARAMETER LIST -- the same problem NoZero solves, one slot over.
//
// CollectiveMma's template parameter list is fixed by its primary template, so the two-plane mainloop cannot take
// three extra B-side atoms as extra parameters. It takes them wrapped: SmemLayoutAtomB is either the ordinary
// atom or BPlanes<atom0, atom1>, and the mainloop unwraps with bplane_first_t / bplane_second_t.
//
// DO NOT REPLACE THE MARKER WITH cute::is_tuple. SmemLayoutAtomB_ is a cute Layout, and a cute Layout is itself
// tuple-like, so is_tuple answers yes for the ORDINARY single-plane case -- a false positive that selects the
// two-plane unwrap on a build that has one plane.
//
// It lives here rather than in the two-plane collective because the BUILDER names it while choosing atoms, and
// the builder is in every consumer's base while the two-plane collective is not.
template <class T0, class T1>
struct BPlanes {};

namespace bplane_detail {
  template <class T> struct first  { using type = T; };
  template <class T0, class T1> struct first<BPlanes<T0, T1>>  { using type = T0; };
  template <class T> struct second { using type = void; };
  template <class T0, class T1> struct second<BPlanes<T0, T1>> { using type = T1; };
}
template <class T> using bplane_first_t  = typename bplane_detail::first<T>::type;
template <class T> using bplane_second_t = typename bplane_detail::second<T>::type;

} // namespace cutlass::gemm::collective
