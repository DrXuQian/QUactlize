// Compose the existing physical reader, converter emission and compute
// fragment for the proposed Q8_0 K-pack2 code plane. No device launch.
#define main l236_existing_main
#include "l236_kquant_kpack_production_fragment.cu"
#undef main

namespace {
template <> struct ElementForBits<8> { using type = int8_t; };
}

int main() {
  bool const ok = plane_geometries<8, 32>("Q8_0-kpack2");
  std::printf("Q8_KPACK2_FRAGMENT %s geometries=11 activation=FP16 execution=HOST_LAYOUT_ONLY\n",
              ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
