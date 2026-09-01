// Exact type admission for the first Q4_K/A32 device-numeric failure.
//
// This is deliberately one row, not a reduced restatement of the sweep.  It
// instantiates the same generated-unit body and RowTypes used by the shipping
// ScaleFirst sweep for:
//   qtype=12 A=32 TM=64 TN=64 TK=128 WM=16 WN=32 stages=8 bchunk=0.
// A host-only map proof is insufficient here: the regression lived in the
// device collective's folded converter path.

#define PPU_PACKED_SCALE 0
#define PPU_B_CHUNK 0
#define SCALEFIRST_TYPE_ONLY 1
#define SCALEFIRST_UNIT_ROWS(X)                                      \
  X(sf_q12_a32_tm64_tn64_tk128_wm16_wn32_s8_bc0,                    \
    12, 32, 64, 64, 128, 16, 32, 8, 0)

#include "scalefirst_internal_sweep_unit.inc"

struct L214NoKPack4Marker {};
struct L214FalseKPack4Marker {
  static constexpr bool kQ4KPack4Transpose = false;
};
struct L214TrueKPack4Marker {
  static constexpr bool kQ4KPack4Transpose = true;
};

static_assert(!scalefirst_internal_sweep::MainloopUsesQ4KPack4Transpose<
              L214NoKPack4Marker>::value);
static_assert(!scalefirst_internal_sweep::MainloopUsesQ4KPack4Transpose<
              L214FalseKPack4Marker>::value);
static_assert(scalefirst_internal_sweep::MainloopUsesQ4KPack4Transpose<
              L214TrueKPack4Marker>::value);

using L214Types = scalefirst_internal_sweep::RowTypes<
    12, 32, 64, 64, 128, 16, 32, 8, 0>;
using L214Kernel = typename L214Types::ShippingKernel;

__global__ void l214_force_exact_body(L214Kernel::Params params) {
  extern __shared__ char smem[];
  L214Kernel{}(params, smem);
}

int main() { return 0; }
