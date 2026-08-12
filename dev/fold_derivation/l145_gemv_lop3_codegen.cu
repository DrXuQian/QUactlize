// L145 -- instantiate the exact shipping int4 GEMV specialization whose converter codegen is audited.
//
// This is intentionally the full production kernel, not RawConverter in isolation. Register pressure and
// surrounding use can change instruction selection, so an isolated helper would answer a weaker question. The
// companion script also binds this tuple to ppu_backend.cu's qtype=12 dispatch before accepting the disassembly.
#include "gemv_lowbit/gemv_launcher.hpp"

using L145ShippingInt4 = ppu_gemv::KernelDetails<
    ppu_gemv::FP16DetailsA,
    ppu_gemv::WFormat::Int4,
    ppu_gemv::WLayout::Native,
    16,
    128,
    256>;

static_assert(L145ShippingInt4::kStepK == 16 &&
              L145ShippingInt4::kThreads == 128 &&
              L145ShippingInt4::kTileSizeK == 256 &&
              L145ShippingInt4::kLoBits == 4 &&
              !L145ShippingInt4::kTwoPlane,
              "L145 must remain the production int4/native s16/t128 artifact-256 specialization");

// Dense M=1, CtaN=8, Chunk=2, gs=32, affine scale+zero, divisible K, no optional feature arms.
// CtaN*StepK/2 = 64 half2 pairs are converted per thread per K-loop iteration.
template __global__ void ppu_gemv::gemv_kernel<
    L145ShippingInt4,
    1,
    8,
    2,
    32,
    ppu_gemv::QuantOp::FinegrainedScaleZero,
    false,
    false,
    false,
    false,
    false>(ppu_gemv::KernelArgs);
