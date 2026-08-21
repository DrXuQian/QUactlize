// L129 -- DECLARED MIXED-INPUT ARGUMENTS ARE EITHER CONSUMED OR REJECTED.
//
// This is a pure host oracle for the exact admission predicate shared by the
// three collectives and every kernel wrapper.  It does not ask a second copy of
// the implementation what is valid: expected answers come from explicit
// group/fold/interleave/packed modular equalities and canonical 64-bit pitches.

#include <cstdint>
#include <cstdio>

#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_argument_contract.hpp"

namespace arg = cutlass::gemm::collective::detail;

static arg::MixedArgumentContract base() {
  arg::MixedArgumentContract c{};
  c.n = 256; c.k = 4096; c.l = 256;
  c.group_size = 32; c.tile_k = 256; c.scale_tile_k = 8;
  c.static_group_size = 32;
  c.low_bits = 4; c.interleave = 1; c.has_scales = true;
  c.dB0 = 4096; c.dB1 = 1; c.dBL = 256ll * 4096;
  c.dS0 = 1; c.dS1 = 256; c.dSL = 256ll * 128;
  return c;
}

int main() {
  int bad = 0;

  // Runtime/static gs: both values are consumed.  Equality is not optional:
  // runtime g owns scale_k while the schedule owns scale selection.
  int gs_cases = 0, gs_accept = 0, gs_expected = 0, gs_negative_red = 0;
  for (int st : {-1, 0, 16, 32, 64, 128}) {
    for (int g : {16, 32, 64, 128, 4096}) {
      auto c = base();
      c.static_group_size = st;
      c.group_size = g;
      c.scale_tile_k = (c.tile_k + g - 1) / g;
      c.dSL = c.n * ((c.k + g - 1) / g);
      bool const expected = st == 0 || (st == -1 ? g == c.k : g == st);
      bool const got = arg::mixed_arguments_supported(c);
      ++gs_cases; gs_accept += got; gs_expected += expected; bad += got != expected;
    }
  }
  {
    auto c = base();
    ++c.scale_tile_k;
    gs_negative_red += !arg::mixed_arguments_supported(c);
  }
  {
    auto c = base();
    c.group_size = 0;
    gs_negative_red += !arg::mixed_arguments_supported(c);
  }

  // Canonical interleaved B markers.  The placed bytes are anchored to
  // N*K*bits/8 per expert; the Stride marker is in logical sub-byte elements.
  int interleave_positive = 0, interleave_red = 0;
  auto ordinary = base();
  ordinary.n = 256; ordinary.k = 256; ordinary.l = 3;
  ordinary.dB0 = 256; ordinary.dB1 = 1; ordinary.dBL = 256ll * 256;
  ordinary.dS1 = 256; ordinary.dSL = 256ll * 8;
  ordinary.interleave = 256;
  interleave_positive += arg::mixed_arguments_supported(ordinary);
  for (int which = 0; which < 3; ++which) {
    auto c = ordinary;
    (which == 0 ? c.dB0 : which == 1 ? c.dB1 : c.dBL) += 1;
    interleave_red += !arg::mixed_arguments_supported(c);
  }
  auto folded = ordinary;
  folded.low_fold = 2; folded.low_bits = 2; folded.dB0 = 512;
  interleave_positive += arg::mixed_arguments_supported(folded);
  auto planes = folded;
  planes.two_plane = true; planes.high_fold = 4; planes.high_bits = 1;
  planes.dB2_valid = true; planes.dB20 = 1024; planes.dB21 = 1;
  planes.dB2L = 256ll * 256;
  interleave_positive += arg::mixed_arguments_supported(planes);
  {
    auto c = planes; c.dB2_valid = false;
    interleave_red += !arg::mixed_arguments_supported(c);
  }
  for (int which = 0; which < 3; ++which) {
    auto c = planes;
    (which == 0 ? c.dB20 : which == 1 ? c.dB21 : c.dB2L) += 1;
    interleave_red += !arg::mixed_arguments_supported(c);
  }

  // Noninterleaved dB2 is genuinely consumed rather than required canonical:
  // perturbing it changes an independently anchored address and remains legal.
  auto noninterleaved = planes;
  noninterleaved.interleave = 1;
  int64_t const before = 7 * noninterleaved.dB20 + 5 * noninterleaved.dB21 +
                         2 * noninterleaved.dB2L;
  noninterleaved.dB20 += 17;
  int64_t const after = 7 * noninterleaved.dB20 + 5 * noninterleaved.dB21 +
                        2 * noninterleaved.dB2L;
  bool const noninterleaved_consumed =
      arg::mixed_arguments_supported(noninterleaved) && before != after;
  bad += interleave_positive != 3 || interleave_red != 7 ||
         !noninterleaved_consumed;

  // Packed metadata overload: ptr_S is a raw format unit, so generic dS is a
  // tight logical marker and ptr_Z must be null.  Contradictions fail closed.
  int packed_positive = 0, packed_red = 0;
  auto packed = base();
  packed.n = 128; packed.k = 256; packed.l = 2;
  packed.dB0 = 256; packed.dB1 = 1; packed.dBL = 128ll * 256;
  packed.dS1 = 128; packed.dSL = 128ll * 8;
  packed.packed_scale = true;
  packed_positive += arg::mixed_arguments_supported(packed);
  {
    auto c = packed; c.ptr_Z_nonnull = true;
    packed_red += !arg::mixed_arguments_supported(c);
  }
  for (int which = 0; which < 3; ++which) {
    auto c = packed;
    (which == 0 ? c.dS0 : which == 1 ? c.dS1 : c.dSL) += 1;
    packed_red += !arg::mixed_arguments_supported(c);
  }
  {
    auto c = packed; c.k = 257;
    packed_red += (arg::mixed_argument_issues(c) &
                   arg::MixedArgumentPackedGroupTail) != 0;
  }
  {
    auto c = packed; c.k = 384; c.dB0 = 384; c.dBL = 128ll * 384;
    c.dSL = 128ll * 12;
    packed_red += (arg::mixed_argument_issues(c) &
                   arg::MixedArgumentPackedTileTail) != 0;
  }
  {
    auto c = packed; c.k = 768; c.dB0 = 768; c.dBL = 128ll * 768;
    c.dSL = 128ll * 24; c.packed_tiles_per_unit = 2;
    packed_red += (arg::mixed_argument_issues(c) &
                   arg::MixedArgumentPackedUnitTail) != 0;
  }
  bad += packed_positive != 1 || packed_red != 7;

  // Every integer division used to build a resident view gets both sides of
  // its boundary.  The oracle is the modulo itself, not a truncated quotient.
  int residue_positive = 0, residue_red = 0;
  for (int fold : {1, 2, 4}) {
    auto c = ordinary;
    c.low_fold = fold; c.dB0 = c.k * fold;
    residue_positive += arg::mixed_arguments_supported(c);
    if (fold > 1) {
      auto r = c; ++r.n;
      residue_red += !arg::mixed_arguments_supported(r);
    }
  }
  {
    auto c = ordinary; ++c.k; ++c.dB0; c.dBL = c.n * c.k;
    residue_red += !arg::mixed_arguments_supported(c);
  }
  {
    auto c = base(); ++c.dBL;
    residue_red += (arg::mixed_argument_issues(c) &
                    arg::MixedArgumentFractionalLowByte) != 0;
  }
  {
    auto c = base(); c.two_plane = true; c.high_bits = 1;
    c.high_fold = 1; c.dB2_valid = true;
    c.dB20 = c.k; c.dB21 = 1; c.dB2L = c.dBL + 1;
    residue_red += (arg::mixed_argument_issues(c) &
                    arg::MixedArgumentFractionalHighByte) != 0;
  }
  bad += residue_positive != 3 || residue_red != 5;
  bad += gs_accept != gs_expected || gs_negative_red != 2;

  std::printf("L129 gs cases=%d accept=%d expected=%d negative_red=%d\n",
              gs_cases, gs_accept, gs_expected, gs_negative_red);
  std::printf("L129 interleaved canonical=%d/3 perturb_red=%d/7 "
              "noninterleaved_dB2_consumed=%s\n",
              interleave_positive, interleave_red,
              noninterleaved_consumed ? "YES" : "NO");
  std::printf("L129 packed canonical=%d/1 contradictions_red=%d/7\n",
              packed_positive, packed_red);
  std::printf("L129 residues aligned=%d/3 residue_red=%d/5\n",
              residue_positive, residue_red);
  std::printf("L129 result=%s scope=gs+interleaved-B+B2+packed+divisibility\n",
              bad == 0 ? "PASS" : "FAIL");
  return bad == 0 ? 0 : 1;
}
