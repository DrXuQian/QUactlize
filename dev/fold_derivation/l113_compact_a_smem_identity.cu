#include <cstdio>
#include "cute/tensor.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/detail/compact_a_smem.hpp"
using namespace cute;
template <int Cap> void show(const char* tag) {
  using TileShape = Shape<_16,_16,_256>;
  using Atom = decltype(composition(Swizzle<0,4,3>{}, Layout<Shape<_8,_64>, Stride<_64,_1>>{}));
  using S = quactlize::collective::detail::CompactASmem<TileShape, 2, Atom, Cap>;
  // the hand-written spelling this replaced, reproduced here as the INDEPENDENT reference
  auto ord = tile_to_shape(Atom{}, make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<2>{}));
  constexpr int SR = Cap > 0 ? Cap : int(shape<0>(TileShape{}));
  auto cmp = make_layout(make_shape(make_shape(Int<SR>{}, Int<int(shape<0>(TileShape{}))/SR>{}),
                                    shape<2>(TileShape{}), Int<2>{}),
                         make_stride(make_stride(shape<2>(TileShape{}), _0{}),
                                     _1{}, Int<SR*int(shape<2>(TileShape{}))>{}));
  using Want = cute::conditional_t<(Cap>0), decltype(cmp), decltype(ord)>;
  std::printf("%-10s cosize=%-8d same-as-hand-written=%s\n", tag,
              int(cosize_v<typename S::Layout>),
              std::is_same_v<typename S::Layout, Want> ? "YES" : "*** NO ***");
}
int main(){ show<0>("cap=0"); show<1>("cap=1"); show<2>("cap=2"); show<8>("cap=8"); }
