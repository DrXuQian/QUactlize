/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Standalone Marlin PPU load primitives.
 *
 * These helpers deliberately live outside the generic mixed-input collective.  The cp.async and
 * ldmatrix sequence is the instruction-shape baseline shared with the local PPU-classic Marlin
 * port.  Future AIU delivery is a separate compile-time policy; it must not add a runtime branch to
 * this baseline.
 **************************************************************************************************/
#pragma once

#include <cstdint>

#include <cuda_fp16.h>

#include "cutlass/cutlass.h"

namespace cutlass::gemm::collective {

struct MarlinCpAsyncLoadPolicyPPU {};
struct MarlinAiuLoadPolicyPPU {};

namespace marlin_ppu_detail {

struct alignas(16) Vector128 {
  uint32_t x, y, z, w;
};

struct FragmentA {
  __half2 value[4];
};

CUTLASS_DEVICE void cp_async_16(void* smem_ptr, void const* global_ptr) {
  uint32_t const smem = uint32_t(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(smem), "l"(global_ptr)
               : "memory");
}

CUTLASS_DEVICE void cp_async_16_if(
    void* smem_ptr, void const* global_ptr, bool predicate) {
  // PPU lowers cp.async to an intrinsic.  Predicating the asm instruction itself can force a
  // compatibility expansion, so the classic PPU port uses a uniform C++ branch around it.
  if (predicate) {
    cp_async_16(smem_ptr, global_ptr);
  }
}

CUTLASS_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int Count>
CUTLASS_DEVICE void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" : : "n"(Count) : "memory");
}

CUTLASS_DEVICE void ldmatrix_a(FragmentA& fragment, void const* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&fragment);
  // PPU x4 returns v1/v2 in the opposite register order from the A operand.  Bind those
  // outputs directly to a2/a1: no move or temporary is emitted.
  asm volatile(
      "ppu.ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(a[0]), "=r"(a[2]), "=r"(a[1]), "=r"(a[3])
      : "l"(smem_ptr));
}

}  // namespace marlin_ppu_detail
}  // namespace cutlass::gemm::collective
