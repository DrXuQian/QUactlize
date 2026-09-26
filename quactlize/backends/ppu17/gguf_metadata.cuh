#pragma once
#include "cute/tensor.hpp"

namespace quactlize::ppu17 {

// The physical stage has one independent affine pair per GGUF group. The
// logical view broadcasts within that group, never across a group boundary.
template<int Columns, int TileK, int GroupK, int Stages>
struct GgufMetadataStage {
  static_assert(GroupK >= 16 && GroupK % 16 == 0 && TileK % GroupK == 0);
  static constexpr int groups = TileK / GroupK;
  using Physical = cute::Layout<
      cute::Shape<cute::Int<Columns>, cute::Int<groups>, cute::Int<Stages>>,
      cute::Stride<cute::_1, cute::Int<Columns>, cute::Int<Columns * groups>>>;
  using Logical = cute::Layout<
      cute::Shape<cute::Int<Columns>, cute::Shape<cute::Int<GroupK>, cute::Int<groups>>, cute::Int<Stages>>,
      cute::Stride<cute::_1, cute::Stride<cute::_0, cute::Int<Columns>>, cute::Int<Columns * groups>>>;
  static_assert(cute::cosize_v<Physical> == cute::cosize_v<Logical>);
};

} // namespace quactlize::ppu17
