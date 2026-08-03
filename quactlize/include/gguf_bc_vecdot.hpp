#pragma once
// CUDA-CORE DECODE FROM THE MERGED BC ARTIFACT: xplane-placed code planes plus byte-neutral packed scale units.
//
// THE ADDRESS IS A LAYOUT, not a transcription of place_derived's loops. For each shipped tactic the complete
// xplane inverse is a permutation of the bits in logical (n-within-64,k-within-tile). The layouts below name that
// permutation directly: input mode i is one logical-coordinate bit and its stride is the physical bit it becomes.
// A local exhaustive gate compares all 64*TileK coordinates with xplane::{plane_map,tile_map_hi}; the point is that
// the CUDA consumer and offline producer have one reviewable mapping object rather than unrelated index arithmetic.
//
// Q6 is the only TileK=128 artifact. Two of its tiles share each 256-element physical row: map position bit 7 and
// above name the N row and therefore shift left once in the physical address, while tile%2 occupies the opened bit
// 7. Every other format has TileK=256, where map enumeration and physical-row enumeration are already identical.

#include <cstdint>
#include "gguf_vecdot.hpp"
#include "gguf_packed_unit.hpp"

namespace gguf_scale {
namespace bc_vecdot {

template <KType T> struct Traits;
template <> struct Traits<KType::Q2_K> { static constexpr int Lo=2, Hi=0, TileK=256; };
template <> struct Traits<KType::Q3_K> { static constexpr int Lo=2, Hi=1, TileK=256; };
template <> struct Traits<KType::Q4_K> { static constexpr int Lo=4, Hi=0, TileK=256; };
template <> struct Traits<KType::Q5_K> { static constexpr int Lo=4, Hi=1, TileK=256; };
template <> struct Traits<KType::Q6_K> { static constexpr int Lo=4, Hi=2, TileK=128; };

using BitShape14 = cute::Shape<cute::_2,cute::_2,cute::_2,cute::_2,cute::_2,cute::_2,cute::_2,
                               cute::_2,cute::_2,cute::_2,cute::_2,cute::_2,cute::_2,cute::_2>;
using BitShape13 = cute::Shape<cute::_2,cute::_2,cute::_2,cute::_2,cute::_2,cute::_2,cute::_2,
                               cute::_2,cute::_2,cute::_2,cute::_2,cute::_2,cute::_2>;

template <KType T, bool High>
CUTLASS_HOST_DEVICE auto xplane_bit_layout() {
  // Exhaustively derived bit maps, in input-bit order. Low and high are independently named because place_hi is a
  // cross-plane placement, not the standalone high-width plane map (Q5 even moves one N bit across the K modes).
  if constexpr (T == KType::Q2_K && !High) {
    return cute::make_layout(BitShape14{}, cute::Stride<
        cute::C<0x8>,cute::C<0x10>,cute::C<0x20>,cute::C<0x1>,cute::C<0x2>,cute::C<0x4>,cute::C<0x40>,
        cute::C<0x80>,cute::C<0x100>,cute::C<0x200>,cute::C<0x400>,cute::C<0x800>,cute::C<0x1000>,cute::C<0x2000>>{});
  } else if constexpr (T == KType::Q3_K && !High) {
    return cute::make_layout(BitShape14{}, cute::Stride<
        cute::C<0x8>,cute::C<0x10>,cute::C<0x20>,cute::C<0x1>,cute::C<0x2>,cute::C<0x4>,cute::C<0x40>,
        cute::C<0x80>,cute::C<0x100>,cute::C<0x200>,cute::C<0x400>,cute::C<0x800>,cute::C<0x1000>,cute::C<0x2000>>{});
  } else if constexpr (T == KType::Q3_K && High) {
    return cute::make_layout(BitShape14{}, cute::Stride<
        cute::C<0x10>,cute::C<0x20>,cute::C<0x40>,cute::C<0x1>,cute::C<0x2>,cute::C<0x4>,cute::C<0x8>,
        cute::C<0x80>,cute::C<0x100>,cute::C<0x200>,cute::C<0x400>,cute::C<0x800>,cute::C<0x1000>,cute::C<0x2000>>{});
  } else if constexpr ((T == KType::Q4_K || T == KType::Q5_K) && !High) {
    return cute::make_layout(BitShape14{}, cute::Stride<
        cute::C<0x4>,cute::C<0x8>,cute::C<0x10>,cute::C<0x1>,cute::C<0x2>,cute::C<0x20>,cute::C<0x40>,
        cute::C<0x80>,cute::C<0x100>,cute::C<0x200>,cute::C<0x400>,cute::C<0x800>,cute::C<0x1000>,cute::C<0x2000>>{});
  } else if constexpr (T == KType::Q5_K && High) {
    return cute::make_layout(BitShape14{}, cute::Stride<
        cute::C<0x10>,cute::C<0x20>,cute::C<0x40>,cute::C<0x1>,cute::C<0x2>,cute::C<0x4>,cute::C<0x80>,
        cute::C<0x800>,cute::C<0x100>,cute::C<0x200>,cute::C<0x400>,cute::C<0x8>,cute::C<0x1000>,cute::C<0x2000>>{});
  } else if constexpr (T == KType::Q6_K && !High) {
    return cute::make_layout(BitShape13{}, cute::Stride<
        cute::C<0x4>,cute::C<0x8>,cute::C<0x10>,cute::C<0x1>,cute::C<0x2>,cute::C<0x20>,cute::C<0x40>,
        cute::C<0x80>,cute::C<0x100>,cute::C<0x200>,cute::C<0x400>,cute::C<0x800>,cute::C<0x1000>>{});
  } else {
    static_assert(T == KType::Q6_K && High, "no xplane bit layout for this plane");
    return cute::make_layout(BitShape13{}, cute::Stride<
        cute::C<0x8>,cute::C<0x10>,cute::C<0x20>,cute::C<0x1>,cute::C<0x2>,cute::C<0x4>,cute::C<0x40>,
        cute::C<0x80>,cute::C<0x100>,cute::C<0x200>,cute::C<0x400>,cute::C<0x800>,cute::C<0x1000>>{});
  }
}

template <KType T, bool High>
CUTLASS_HOST_DEVICE int xplane_relative(int n_in_tile, int k_in_tile) {
  auto layout = xplane_bit_layout<T, High>();
  int const logical = n_in_tile * Traits<T>::TileK + k_in_tile;
  return int(layout(cute::idx2crd(logical, cute::shape(layout))));
}

template <KType T, bool High>
CUTLASS_HOST_DEVICE int64_t xplane_physical_code(int n, int k, int num_cols) {
  constexpr int TK = Traits<T>::TileK;
  int const kt = k / TK, kr = k - kt * TK;
  int const nt = n / 64, nr = n - nt * 64;
  int const rel = xplane_relative<T, High>(nr, kr);
  if constexpr (TK == 128) {
    int const row_interleaved = (rel & 127) | ((rel & ~127) << 1) | ((kt & 1) << 7);
    return int64_t(kt / 2) * num_cols * 256 + int64_t(nt) * 64 * 256 + row_interleaved;
  } else {
    return int64_t(kt) * num_cols * 256 + int64_t(nt) * 64 * 256 + rel;
  }
}

template <int Bits>
__device__ __forceinline__ int plane_code(uint8_t const* plane, int64_t physical) {
  int64_t const bit = physical * Bits;
  return (plane[bit >> 3] >> (bit & 7)) & ((1 << Bits) - 1);
}

template <KType T>
__device__ __forceinline__ int code_at(uint8_t const* low, uint8_t const* high, int n, int k, int num_cols) {
  constexpr int LB = Traits<T>::Lo, HB = Traits<T>::Hi;
  int q = plane_code<LB>(low, xplane_physical_code<T, false>(n, k, num_cols));
  if constexpr (HB != 0) q |= plane_code<HB>(high, xplane_physical_code<T, true>(n, k, num_cols)) << LB;
  if constexpr (T == KType::Q3_K) q -= 4;
  if constexpr (T == KType::Q6_K) q -= 32;
  return q;
}

template <KType T>
__device__ __forceinline__ vecdot::VecdotCode4 code4(
    uint8_t const* low, uint8_t const* high, int n, int k, int num_cols) {
  constexpr int Bias = T == KType::Q3_K ? 4 : T == KType::Q6_K ? 32 : 0;
  uint32_t bytes = 0;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < 4; ++j) bytes |= uint32_t(code_at<T>(low, high, n, k + j, num_cols) + Bias) << (8*j);
  return vecdot::vecdot_code4_from_bytes<T>(bytes);
}

// Q4's placed group is physically contiguous but its logical K is an 8x4 transpose. Vector-load logical x once,
// read the code group as four aligned words, and permute only registers. This is the measured implementation:
// per-code inverse lookups were 55x slower, while this ties raw cold latency at N=K=2048 and wins at saturated N.
__device__ __forceinline__ half q4_group(uint8_t const* low_group, vecdot::VecdotActivation const* x_group,
                                         float scale, float zero) {
  half2 logical_x[16];
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < 16; ++j) logical_x[j] = vecdot::vecdot_load_activation2(x_group + 2*j);
  half2 qx = vecdot::vecdot_half2_zero(), sx = vecdot::vecdot_half2_zero();
  CUTLASS_PRAGMA_UNROLL
  for (int word = 0; word < 4; ++word) {
    uint32_t const packed = *reinterpret_cast<uint32_t const*>(low_group + word * 4);
    auto const qlo = vecdot::vecdot_code4_from_bytes<KType::Q4_K>(packed & 0x0f0f0f0fu);
    auto const qhi = vecdot::vecdot_code4_from_bytes<KType::Q4_K>((packed >> 4) & 0x0f0f0f0fu);
    CUTLASS_PRAGMA_UNROLL
    for (int byte = 0; byte < 4; ++byte) {
      int const p0 = 8*word + 2*byte, p1 = p0 + 1;
      int const k0 = (p0 & 3) * 8 + (p0 >> 2), k1 = (p1 & 3) * 8 + (p1 >> 2);
      cutlass::half_t const h0 = qlo[byte], h1 = qhi[byte];
      half2 const hq = vecdot::vecdot_half2_from_halves(h0.to_half(), h1.to_half());
      half2 const hx = vecdot::vecdot_half2_from_halves(
          (k0 & 1) ? logical_x[k0/2].y : logical_x[k0/2].x,
          (k1 & 1) ? logical_x[k1/2].y : logical_x[k1/2].x);
      qx = __hfma2(hq, hx, qx); sx = __hadd2(sx, hx);
    }
  }
  return vecdot::vecdot_apply_group_half<true>(qx, sx, scale, -zero);
}

template <KType T>
CUTLASS_HOST_DEVICE constexpr int rows_per_warp(int rows, int blocks_per_row) {
  if constexpr (T == KType::Q4_K) return rows <= 2048 ? 4 : 8;
  return vecdot::vecdot_rows_per_warp<T>(rows, blocks_per_row);
}

template <KType T, int RowsPerWarp, bool Grouped = false>
__global__ void rows_kernel(uint8_t const* low, uint8_t const* high, uint8_t const* units,
                            vecdot::VecdotActivation const* x, float* out, int rows, int blocks_per_row,
                            int const* row_offsets = nullptr) {
  using U = packed_unit::Unit<T>;
  constexpr int LanesPerRow = 32 / RowsPerWarp;
  constexpr int Groups = U::kGroups, GroupSize = 256 / Groups;
  constexpr bool HasMin = U::kHasMin;
  int const expert = Grouped ? int(blockIdx.z) : 0;
  int const route_in_expert = Grouped ? int(blockIdx.y) : 0;
  int const route_begin = Grouped ? row_offsets[expert] : 0;
  int const route_rows = Grouped ? row_offsets[expert+1] - route_begin : 1;
  int const gathered_row = route_begin + route_in_expert;
  int const warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int const lane = threadIdx.x & 31;
  int const row_in_warp = lane / LanesPerRow, row_lane = lane & (LanesPerRow-1);
  int const r = warp * RowsPerWarp + row_in_warp;
  bool const active = r < rows && route_in_expert < route_rows;
  int const activation_row = active ? (Grouped ? gathered_row : 0) : 0;
  auto const* row_x = x + int64_t(activation_row) * blocks_per_row * 256;
  float* row_out = out + (Grouped && active ? int64_t(gathered_row) * rows : 0);
  int64_t const lo_per = int64_t(rows) * blocks_per_row * 256 * Traits<T>::Lo / 8;
  int64_t const hi_per = int64_t(rows) * blocks_per_row * 256 * Traits<T>::Hi / 8;
  int const num_units = blocks_per_row / U::kSbPerUnit;
  int64_t const unit_per = int64_t(num_units) * rows * U::kUnitTotal;
  uint8_t const* elo = low + (Grouped ? int64_t(expert) * lo_per : 0);
  uint8_t const* ehi = high;
  if constexpr (Traits<T>::Hi != 0) {
    if constexpr (Grouped) ehi += int64_t(expert) * hi_per;
  }
  uint8_t const* eu = units + (Grouped ? int64_t(expert) * unit_per : 0);
  half2 lane_acc = vecdot::vecdot_half2_zero();

  for (int sb = 0; sb < blocks_per_row; ++sb) {
    uint8_t const* unit = active
        ? eu + ((int64_t(sb / U::kSbPerUnit) * rows + r) * U::kUnitTotal)
        : eu;
    int const sb_in_unit = sb % U::kSbPerUnit;
    if constexpr (T == KType::Q4_K) {
      CUTLASS_PRAGMA_UNROLL
      for (int pair = row_lane; pair < Groups/2; pair += LanesPerRow) {
        int const g0=2*pair,g1=g0+1;
        auto const sz0=packed_unit::unit_group_sb<T,0>(unit,sb_in_unit,g0);
        auto const sz1=packed_unit::unit_group_sb<T,0>(unit,sb_in_unit,g1);
        int64_t const p0=xplane_physical_code<T,false>(r,sb*256+g0*32,rows);
        int64_t const p1=xplane_physical_code<T,false>(r,sb*256+g1*32,rows);
        half const v0=active?q4_group(elo+p0/2,row_x+sb*256+g0*32,float(sz0.scale),float(sz0.zero)):__float2half(0.f);
        half const v1=active?q4_group(elo+p1/2,row_x+sb*256+g1*32,float(sz1.scale),float(sz1.zero)):__float2half(0.f);
        lane_acc=__hadd2(lane_acc,vecdot::vecdot_half2_from_halves(v0,v1));
      }
    } else {
      int owner_slot=0;
      CUTLASS_PRAGMA_UNROLL
      for(int g=row_lane;g<Groups;g+=LanesPerRow,++owner_slot){
        auto const sz=packed_unit::unit_group_sb<T,0>(unit,sb_in_unit,g);
        half2 qx=vecdot::vecdot_half2_zero(),sx=vecdot::vecdot_half2_zero();
        if(active){
          CUTLASS_PRAGMA_UNROLL
          for(int j=0;j<GroupSize;j+=4){
            auto const q=code4<T>(elo,ehi,r,sb*256+g*GroupSize+j,rows);
            auto const hx0=vecdot::vecdot_load_activation2(row_x+sb*256+g*GroupSize+j);
            auto const hx1=vecdot::vecdot_load_activation2(row_x+sb*256+g*GroupSize+j+2);
            half2 const* q2=reinterpret_cast<half2 const*>(&q);
            qx=__hfma2(q2[0],hx0,qx); qx=__hfma2(q2[1],hx1,qx);
            if constexpr(HasMin){sx=__hadd2(sx,hx0);sx=__hadd2(sx,hx1);}
          }
        }
        half const v=vecdot::vecdot_apply_group_half<HasMin>(qx,sx,float(sz.scale),-float(sz.zero));
        half2 add=vecdot::vecdot_half2_zero(); if(owner_slot&1)add.y=v;else add.x=v;
        lane_acc=__hadd2(lane_acc,add);
      }
    }
  }
  lane_acc=vecdot::vecdot_subgroup_sum_half2<LanesPerRow>(lane_acc);
  if(active&&row_lane==0)row_out[r]=__half2float(vecdot::vecdot_half2_horizontal(lane_acc));
}

template <KType T, int RowsPerWarp, bool Grouped>
void launch_fixed(uint8_t const* low,uint8_t const* high,uint8_t const* units,
                  vecdot::VecdotActivation const* x,int const* offsets,float* out,
                  int n,int bpr,int experts,int max_rows){
  // At the real Q4 decode shape RPW=4 halves the lanes spent only on subgroup padding. Four-warp CTAs then expose
  // enough independent blocks to saturate cold delivery; eight-warp CTAs measured 5.2-5.7% behind raw over 64
  // distinct cold 2048x2048 layers. Threads=128 removed that gap while also taking warm BC 0.9-1.4% ahead of raw.
  constexpr int Threads=(T==KType::Q4_K&&RowsPerWarp==4)?128:256;
  dim3 grid(vecdot::vecdot_grid_size<T,RowsPerWarp>(n,Threads),Grouped?max_rows:1,Grouped?experts:1);
  rows_kernel<T,RowsPerWarp,Grouped><<<grid,Threads>>>(low,high,units,x,out,n,bpr,offsets);
}

template <KType T, bool Grouped>
void launch(uint8_t const* low,uint8_t const* high,uint8_t const* units,
            vecdot::VecdotActivation const* x,int const* offsets,float* out,
            int n,int bpr,int experts,int max_rows){
  switch(rows_per_warp<T>(n,bpr)){
    case 1:launch_fixed<T,1,Grouped>(low,high,units,x,offsets,out,n,bpr,experts,max_rows);break;
    case 2:launch_fixed<T,2,Grouped>(low,high,units,x,offsets,out,n,bpr,experts,max_rows);break;
    case 4:launch_fixed<T,4,Grouped>(low,high,units,x,offsets,out,n,bpr,experts,max_rows);break;
    case 8:launch_fixed<T,8,Grouped>(low,high,units,x,offsets,out,n,bpr,experts,max_rows);break;
  }
}

} // namespace bc_vecdot
} // namespace gguf_scale
