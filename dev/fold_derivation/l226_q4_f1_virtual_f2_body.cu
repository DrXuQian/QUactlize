// L226 -- force the exact ScaleFirst virtual-F2 device body through the local CUDA front end.
// The local compiler cannot assemble PPU opcodes, so the runner accepts only the established vendor-asm errors and
// rejects any diagnostic in quactlize/benchmarks.  Actual opcode build and raw-bit execution remain the box gate.

#define PPU_PACKED_SCALE 0
#define PPU_B_CHUNK 0
#define PPU_Q4_F1_VIRTUAL_F2 1
#define SCALEFIRST_TYPE_ONLY 1
#define SCALEFIRST_UNIT_ROWS(X)                                      \
  X(sf_q12_a64_tm64_tn128_tk128_wm64_wn64_s3_bc0,                   \
    12, 64, 64, 128, 128, 64, 64, 3, 0)

#include "scalefirst_internal_sweep_unit.inc"

using L226Types = scalefirst_internal_sweep::RowTypes<
    12, 64, 64, 128, 128, 64, 64, 3, 0>;
using L226Kernel = typename L226Types::ShippingKernel;
static_assert(L226Types::Shipping::MainloopPolicy::Descriptor::virtual_compute_fold &&
              L226Types::Shipping::MainloopPolicy::Descriptor::compute_low_fold == 2,
              "the exact ScaleFirst row must select the opt-in virtual-fold policy");

__global__ void l226_force_exact_body(L226Kernel::Params params) {
  extern __shared__ char smem[];
  L226Kernel{}(params, smem);
}

int main() { return 0; }
