// Gate for PPU_A_CPASYNC=N's compact A layout. The logical M extent stays TileM so the MMA fragment shape is
// unchanged, while the physical allocation keeps N rows and aliases only padding rows modulo N.
#include <cstdio>

#include "cute/tensor.hpp"

using namespace cute;

template <int Rows, int TM, int TK, int Stages>
int check() {
  static_assert(TM % Rows == 0);
  using L = Layout<
      Shape<Shape<Int<Rows>, Int<TM / Rows>>, Int<TK>, Int<Stages>>,
      Stride<Stride<Int<TK>, _0>, _1, Int<Rows * TK>>>;
  static_assert(size<0>(L{}) == TM, "logical M extent must stay TileM");
  static_assert(cosize_v<L> == Rows * TK * Stages, "allocation must contain exactly Rows A rows per stage");

  int bad = 0;
  for (int s = 0; s < Stages; ++s)
    for (int m = 0; m < TM; ++m)
      for (int k : {0, TK / 2, TK - 1}) {
        int const got = int(L{}(m, k, s));
        int const want = (m % Rows) * TK + k + s * Rows * TK;
        bad += got != want;
      }
  return bad;
}

int main() {
  int const bad = check<1,16,256,2>() + check<2,16,128,3>() + check<4,64,64,4>();
  std::printf("compact A rows: logical TileM preserved, physical rows 1/2/4, bad=%d\n", bad);
  return bad ? 1 : 0;
}
