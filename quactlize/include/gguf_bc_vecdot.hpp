#pragma once
// CUDA-CORE DECODE FROM THE MERGED BC ARTIFACT: xplane-placed code planes plus byte-neutral packed scale units.
//
// THE ADDRESS IS A LAYOUT, not a transcription of place_derived's loops. For each shipped tactic the complete
// xplane inverse is a permutation of the bits in logical (n-within-64,k-within-tile). The layouts below name that
// permutation directly: input mode i is one logical-coordinate bit and its stride is the physical bit it becomes.
// A local exhaustive gate compares all 64*TileK coordinates with xplane::{plane_map,tile_map_hi}; the point is that
// the CUDA consumer and offline producer have one reviewable mapping object rather than unrelated index arithmetic.
//
// ArtifactTileK is part of the resident bytes.  The original reader hardcoded one map per qtype; that was adequate
// only while folded artifacts were rejected at the Python boundary.  The arrangement-aware reader below names the
// finite producer domain explicitly and indexes every legal map by (low bits, high bits, ArtifactTileK).  The maps
// are generated and exhaustively checked against xplane::place_from_map by l137; accepting an unknown arrangement
// is an error, never a request to reinterpret it as the registry default.

#include <cstdint>
#include "gguf_bc_q4_reader.hpp"
#if !defined(__HGGCCC__)
#if defined(__CUDACC__)
#include "gguf_bc_q4_gemv.hpp"
#endif
#endif
#include "gguf_vecdot.hpp"
#include "gguf_packed_unit.hpp"
#include "gemv_lowbit/gemv_common.hpp"
#include "quactlize_ppu_config.h"

namespace gguf_scale {
namespace bc_vecdot {

template <KType T> struct Traits;
// The unversioned BC ABI keeps the exact pre-descriptor map it shipped with. New no-tile Python artifacts may use a
// narrower scale-first placement, but they reach the arrangement-aware ABI; changing these legacy constants would
// silently reinterpret existing Q2/Q4 A256 bytes.
template <> struct Traits<KType::Q2_K> { static constexpr int Lo=2, Hi=0, DefaultArtifactTileK=256; };
template <> struct Traits<KType::Q3_K> { static constexpr int Lo=2, Hi=1, DefaultArtifactTileK=256; };
template <> struct Traits<KType::Q4_K> { static constexpr int Lo=4, Hi=0, DefaultArtifactTileK=256; };
template <> struct Traits<KType::Q5_K> { static constexpr int Lo=4, Hi=1, DefaultArtifactTileK=256; };
template <> struct Traits<KType::Q6_K> { static constexpr int Lo=4, Hi=2, DefaultArtifactTileK=128; };

template <KType T, int ArtifactTileK>
inline constexpr bool arrangement_supported_v =
    (T == KType::Q2_K || T == KType::Q4_K) ?
        (ArtifactTileK == 32 || ArtifactTileK == 64 || ArtifactTileK == 128 || ArtifactTileK == 256) :
    (T == KType::Q3_K || T == KType::Q5_K) ?
        (ArtifactTileK == 64 || ArtifactTileK == 128 || ArtifactTileK == 256) :
    (T == KType::Q6_K) ?
        (ArtifactTileK == 32 || ArtifactTileK == 64 || ArtifactTileK == 128) : false;

template <KType T, int ReaderArtifactTileK>
CUTLASS_HOST_DEVICE constexpr bool reader_accepts(
    quactlize_ppu_placed_arrangement_v1 const& arrangement) {
  return arrangement_supported_v<T, ReaderArtifactTileK> &&
         arrangement.version == QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1 &&
         arrangement.bits == Traits<T>::Lo && arrangement.high_bits == Traits<T>::Hi &&
         arrangement.artifact_tile_k == ReaderArtifactTileK;
}

// The compact object below maps a logical coordinate *inside one producer tile* (n*A+k) to place_from_map's
// flattened slot (((row*DL+dl)*8+wd)*CPW+j).  It deliberately does not encode a 256x256 physical address: folded
// artifacts use runtime-N-dependent dst_fold strides, so that tempting extension was wrong outside one block.
template <KType T, int ArtifactTileK, bool High> struct ArrangementSlotPermutation;

#define QUACTLIZE_BC_SLOT(T, AK, HIGH, S0,S1,S2,S3,S4,S5,S6,S7,S8,S9,SA,SB,SC,SD) \
  template <> struct ArrangementSlotPermutation<KType::T, AK, HIGH> {                        \
    CUTLASS_HOST_DEVICE static constexpr int get(int bit) {                                  \
      return bit==0?S0:bit==1?S1:bit==2?S2:bit==3?S3:bit==4?S4:bit==5?S5:bit==6?S6:bit==7?S7: \
             bit==8?S8:bit==9?S9:bit==10?SA:bit==11?SB:bit==12?SC:SD;                         \
    }                                                                                         \
  }

QUACTLIZE_BC_SLOT(Q2_K, 32,false, 8,16,32,1,2,128,256,512,1024,2048,4,64,0,0);
QUACTLIZE_BC_SLOT(Q2_K, 64,false, 8,16,32,1,2,4,128,256,512,1024,2048,64,0,0);
QUACTLIZE_BC_SLOT(Q2_K,128,false, 8,16,32,1,2,4,64,128,256,512,1024,2048,4096,0);
QUACTLIZE_BC_SLOT(Q2_K,256,false, 8,16,32,1,2,4,64,128,256,512,1024,2048,4096,8192);
QUACTLIZE_BC_SLOT(Q3_K, 64,false, 8,16,32,1,2,4,128,256,512,1024,2048,64,4096,0);
QUACTLIZE_BC_SLOT(Q3_K,128,false, 8,16,32,1,2,4,64,128,256,512,1024,2048,4096,0);
QUACTLIZE_BC_SLOT(Q3_K,256,false, 8,16,32,1,2,4,64,128,256,512,1024,2048,4096,8192);
QUACTLIZE_BC_SLOT(Q4_K, 32,false, 4,8,16,1,2,64,128,256,512,1024,32,0,0,0);
QUACTLIZE_BC_SLOT(Q4_K, 64,false, 4,8,16,1,2,32,64,128,256,512,1024,2048,0,0);
QUACTLIZE_BC_SLOT(Q4_K,128,false, 4,8,16,1,2,32,64,128,256,512,1024,2048,4096,0);
QUACTLIZE_BC_SLOT(Q4_K,256,false, 4,8,16,1,2,32,64,128,256,512,1024,2048,4096,8192);
QUACTLIZE_BC_SLOT(Q5_K, 64,false, 4,8,16,1,2,32,64,128,256,512,1024,2048,4096,0);
QUACTLIZE_BC_SLOT(Q5_K,128,false, 4,8,16,1,2,32,64,128,256,512,1024,2048,4096,0);
QUACTLIZE_BC_SLOT(Q5_K,256,false, 4,8,16,1,2,32,64,128,256,512,1024,2048,4096,8192);
QUACTLIZE_BC_SLOT(Q6_K, 32,false, 4,8,16,1,2,64,128,256,512,1024,32,2048,0,0);
QUACTLIZE_BC_SLOT(Q6_K, 64,false, 4,8,16,1,2,32,64,128,256,512,1024,2048,0,0);
QUACTLIZE_BC_SLOT(Q6_K,128,false, 4,8,16,1,2,32,64,128,256,512,1024,2048,4096,0);

QUACTLIZE_BC_SLOT(Q3_K, 64,true,16,32,64,1,2,4,256,512,1024,2048,4096,8,128,0);
QUACTLIZE_BC_SLOT(Q3_K,128,true,16,32,64,1,2,4,8,256,512,1024,2048,4096,128,0);
QUACTLIZE_BC_SLOT(Q3_K,256,true,16,32,64,1,2,4,8,128,256,512,1024,2048,4096,8192);
QUACTLIZE_BC_SLOT(Q5_K, 64,true,16,32,64,1,2,4,256,512,1024,8,4096,128,2048,0);
QUACTLIZE_BC_SLOT(Q5_K,128,true,16,32,64,1,2,4,128,256,512,1024,8,4096,2048,0);
QUACTLIZE_BC_SLOT(Q5_K,256,true,16,32,64,1,2,4,128,2048,256,512,1024,8,4096,8192);
QUACTLIZE_BC_SLOT(Q6_K, 32,true, 8,16,32,1,2,128,256,512,1024,2048,4,64,0,0);
QUACTLIZE_BC_SLOT(Q6_K, 64,true, 8,16,32,1,2,4,128,256,512,1024,2048,64,0,0);
QUACTLIZE_BC_SLOT(Q6_K,128,true, 8,16,32,1,2,4,64,128,256,512,1024,2048,4096,0);

#undef QUACTLIZE_BC_SLOT

template <int Bits, int ArtifactTileK> struct FoldForPlane {
  static constexpr int value = ArtifactTileK * Bits / 8 >= 32 ? 1 : 32 / (ArtifactTileK * Bits / 8);
};
template <int ArtifactTileK> struct FoldForPlane<0, ArtifactTileK> { static constexpr int value = 1; };

template <KType T, bool High, int ArtifactTileK>
#if defined(__CUDACC__) || defined(__HGGCCC__)
__host__ __device__ __forceinline__
#else
inline
#endif
int64_t xplane_physical_code(int n, int k, int num_cols) {
  static_assert(arrangement_supported_v<T, ArtifactTileK>, "BC reader instantiated for unsupported artifact map");
  constexpr int LB = Traits<T>::Lo, HB = Traits<T>::Hi;
  constexpr int Bits = High ? HB : LB;
  constexpr int F = FoldForPlane<Bits, ArtifactTileK>::value;
  constexpr int OtherF = HB == 0 ? 1 : (High ? FoldForPlane<LB,ArtifactTileK>::value
                                              : FoldForPlane<HB,ArtifactTileK>::value);
  constexpr int MaxF = F > OtherF ? F : OtherF;
  constexpr int WN = MaxF > 2 ? 16 * MaxF : 32, TN = 2 * WN;
  constexpr int CPW = 32 / Bits, DL = F * ArtifactTileK * Bits / 256;
  using P = ArrangementSlotPermutation<T, ArtifactTileK, High>;
  int const local = (n % TN) * ArtifactTileK + (k % ArtifactTileK);
  int slot = 0;
  CUTLASS_PRAGMA_UNROLL
  for (int bit = 0; bit < 14; ++bit)
    if (local & (1 << bit)) slot |= P::get(bit);
  int const j = slot % CPW; slot /= CPW;
  int const wd = slot % 8; slot /= 8;
  int const dl = slot % DL, row = slot / DL;
  int const tn = n / TN, artifact_ki = k / ArtifactTileK;
  if constexpr (F > 1) {
    constexpr int WRowOff = 256 / CPW, Runs = WRowOff / 8, R = TN / F;
    return int64_t(j) + int64_t(wd) * CPW + int64_t(row) * 256 +
           int64_t(tn) * R * 256 + int64_t(artifact_ki % Runs) * 8 * CPW +
           int64_t(artifact_ki / Runs) * (num_cols / F) * 256;
  } else {
    constexpr int Contig = ArtifactTileK * Bits / 8;
    constexpr int AiuByte = Contig > 128 ? 128 : Contig;
    constexpr int AiuElem = AiuByte * 8 / Bits, RPS = 256 / AiuElem;
    return int64_t(j) + int64_t(wd) * CPW + int64_t(dl) * 8 * CPW +
           int64_t(artifact_ki % RPS) * AiuElem + int64_t(row) * 256 +
           int64_t(tn) * TN * 256 + int64_t(artifact_ki / RPS) * num_cols * 256;
  }
}

template <int Bits>
__device__ __forceinline__ int plane_code(uint8_t const* plane, int64_t physical) {
  int64_t const bit = physical * Bits;
  return (plane[bit >> 3] >> (bit & 7)) & ((1 << Bits) - 1);
}

template <KType T, int ArtifactTileK>
__device__ __forceinline__ int code_at(uint8_t const* low, uint8_t const* high, int n, int k, int num_cols) {
  constexpr int LB = Traits<T>::Lo, HB = Traits<T>::Hi;
  int q = plane_code<LB>(low, xplane_physical_code<T, false, ArtifactTileK>(n, k, num_cols));
  if constexpr (HB != 0)
    q |= plane_code<HB>(high, xplane_physical_code<T, true, ArtifactTileK>(n, k, num_cols)) << LB;
  if constexpr (T == KType::Q3_K) q -= 4;
  if constexpr (T == KType::Q6_K) q -= 32;
  return q;
}

// The resident P4x32 Q4 group is four consecutive uint32 words for all four
// supported artifact tile widths, including folded A=32.  This is deliberately
// the closed form of the writer, not a call to xplane_physical_code: retaining
// that scalar inverse here would rerun its fourteen-position permutation once
// per group in source and make zero position terms depend on compiler
// simplification.  L187 exhausts this expression against both the production
// writer and the scalar inverse.  It is a structural bound, not a measured
// latency claim.
//
// A>=64 is the ordinary interleave-256 artifact.  A32 is folded by two and has
// a distinct row/run order; its five shift/mask terms are the F2 branch reduced
// at a 32-code group base.  Do not project either formula onto another fold.
template <int ArtifactTileK>
#if defined(__CUDACC__) || defined(__HGGCCC__)
__host__ __device__ __forceinline__
#else
inline
#endif
int64_t q4_group_byte_offset(int n, int k, int num_cols) {
  using Plan = q4_reader::Q4WordPlan<ArtifactTileK>;
  static_assert(Plan::kCodes == 32 && Plan::kWords == 4 && Plan::kCodesPerWord == 8,
                "Q4 whole-word geometry drifted");
  if constexpr (ArtifactTileK >= 64) {
    return (int64_t(k >> 8) * num_cols + n) * 128 + ((k & 255) >> 1);
  } else {
    static_assert(ArtifactTileK == 32, "unproved folded Q4 group address");
    return int64_t(k >> 7) * num_cols * 64 + int64_t(n >> 6) * 4096 +
           int64_t(n & 31) * 128 + int64_t((n >> 5) & 1) * 16 +
           int64_t((k >> 5) & 3) * 32;
  }
}

template <KType T, int ArtifactTileK>
__device__ __forceinline__ vecdot::VecdotCode4 code4(
    uint8_t const* low, uint8_t const* high, int n, int k, int num_cols) {
  constexpr int Bias = T == KType::Q3_K ? 4 : T == KType::Q6_K ? 32 : 0;
  uint32_t bytes = 0;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < 4; ++j)
    bytes |= uint32_t(code_at<T, ArtifactTileK>(low, high, n, k + j, num_cols) + Bias) << (8*j);
  return vecdot::vecdot_code4_from_bytes<T>(bytes);
}

// A placed Q4 group is physically four consecutive words while logical K is
// the fixed P4x32 permutation.  Decode each entire word through the target
// dialect (PPU `ppu.lop3/fma.rtte`, NVIDIA `lop3/fma.rn`) and permute only the
// activation register reads.  No per-code xplane address arithmetic remains.
// The natural LOP3 pairs are (physical p, p+4), rather than the old PRMT
// reader's adjacent-nibble pairs.  Both cover exactly the same 32 products;
// because the accumulator is fp16 this is nonetheless a different association,
// so the signed-activation accuracy arm is a required performance-gate output,
// not something inferred from the exact code/address proof.
template <int ArtifactTileK>
__device__ __forceinline__ half q4_group(uint8_t const* low_group, vecdot::VecdotActivation const* x_group,
                                         half scale, half zero) {
  using Plan = q4_reader::Q4WordPlan<ArtifactTileK>;
  half2 logical_x[16];
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < 16; ++j) logical_x[j] = vecdot::vecdot_load_activation2(x_group + 2*j);
  half2 qx = vecdot::vecdot_half2_zero(), sx = vecdot::vecdot_half2_zero();
  CUTLASS_PRAGMA_UNROLL
  for (int word = 0; word < 4; ++word) {
    uint32_t const packed = *reinterpret_cast<uint32_t const*>(low_group + word * 4);
    auto const quant = q4_reader::dequantize_word(packed);
    CUTLASS_PRAGMA_UNROLL
    for (int pair = 0; pair < 4; ++pair) {
      int const p0 = 8 * word + Plan::physical_nibble_from_pair_lane(pair, 0);
      int const p1 = 8 * word + Plan::physical_nibble_from_pair_lane(pair, 1);
      int const k0 = Plan::logical_k_from_physical_nibble(p0);
      int const k1 = Plan::logical_k_from_physical_nibble(p1);
      half2 const hx = vecdot::vecdot_half2_from_halves(
          (k0 & 1) ? logical_x[k0/2].y : logical_x[k0/2].x,
          (k1 & 1) ? logical_x[k1/2].y : logical_x[k1/2].x);
      qx = __hfma2(quant.pair[pair], hx, qx); sx = __hadd2(sx, hx);
    }
  }
  half2 const sums = vecdot::vecdot_half2_from_halves(
      vecdot::vecdot_half2_horizontal(qx), vecdot::vecdot_half2_horizontal(sx));
  half2 const affine = vecdot::vecdot_half2_from_halves(scale, zero);
  return vecdot::vecdot_half2_horizontal(__hmul2(sums, affine));
}

template <KType T>
CUTLASS_HOST_DEVICE constexpr int rows_per_warp(int rows, int blocks_per_row) {
  if constexpr (T == KType::Q4_K) return rows <= 2048 ? 4 : 8;
  return vecdot::vecdot_rows_per_warp<T>(rows, blocks_per_row);
}

template <KType T, int ArtifactTileK, int RowsPerWarp, bool Grouped = false>
__global__ void rows_kernel(uint8_t const* low, uint8_t const* high, uint8_t const* units,
                            vecdot::VecdotActivation const* x, float* out, int rows, int blocks_per_row,
                            int const* row_offsets = nullptr) {
  using U = packed_unit::Unit<T>;
  constexpr int LanesPerRow = 32 / RowsPerWarp;
  constexpr int Groups = U::kGroups, GroupSize = 256 / Groups;
  constexpr bool HasMin = U::kHasMin;
  int const expert = Grouped ? int(blockIdx.z) : 0;
  // Dense decode batches share one resident weight matrix.  Give each
  // activation/output row its own grid-y plane instead of paying M host
  // launches.  Grouped mode keeps its existing routed-row interpretation.
  int const dense_batch_row = Grouped ? 0 : int(blockIdx.y);
  int const route_in_expert = Grouped ? int(blockIdx.y) : dense_batch_row;
  int const route_begin = Grouped ? row_offsets[expert] : 0;
  int const route_rows = Grouped ? row_offsets[expert+1] - route_begin : int(gridDim.y);
  int const gathered_row = Grouped ? route_begin + route_in_expert : dense_batch_row;
  int const warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int const lane = threadIdx.x & 31;
  int const row_in_warp = lane / LanesPerRow, row_lane = lane & (LanesPerRow-1);
  int const r = warp * RowsPerWarp + row_in_warp;
  bool const active = r < rows && route_in_expert < route_rows;
  int const activation_row = active ? gathered_row : 0;
  auto const* row_x = x + int64_t(activation_row) * blocks_per_row * 256;
  float* row_out = out + (active ? int64_t(gathered_row) * rows : 0);
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
    // Every supported Q4 resident arrangement has the proved four-word P4x32
    // closure.  Q4 never silently falls back to per-code address arithmetic.
    if constexpr (T == KType::Q4_K) {
      static_assert(Groups == 8 && GroupSize == 32 && Traits<T>::Hi == 0,
                    "Q4 whole-word reader requires the packed-unit 8x32 single-plane contract");
      static_assert(U::kSbPerUnit == 1 && U::kUnitTotal == 16,
                    "Q4 native metadata reader requires one aligned 16-byte unit per superblock");
      auto const metadata = q4_reader::load_metadata(unit);
      CUTLASS_PRAGMA_UNROLL
      for (int pair = row_lane; pair < Groups/2; pair += LanesPerRow) {
        int const g0=2*pair,g1=g0+1;
        auto const sz0=q4_reader::decode_scale_zero(metadata,g0);
        auto const sz1=q4_reader::decode_scale_zero(metadata,g1);
        int64_t const p0=q4_group_byte_offset<ArtifactTileK>(r,sb*256+g0*32,rows);
        int64_t const p1=q4_group_byte_offset<ArtifactTileK>(r,sb*256+g1*32,rows);
        half const v0=active?q4_group<ArtifactTileK>(elo+p0,row_x+sb*256+g0*32,sz0.scale,sz0.zero):__float2half(0.f);
        half const v1=active?q4_group<ArtifactTileK>(elo+p1,row_x+sb*256+g1*32,sz1.scale,sz1.zero):__float2half(0.f);
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
            auto const q=code4<T,ArtifactTileK>(elo,ehi,r,sb*256+g*GroupSize+j,rows);
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

template <KType T, int ArtifactTileK, int RowsPerWarp, bool Grouped>
void launch_fixed(uint8_t const* low,uint8_t const* high,uint8_t const* units,
                  vecdot::VecdotActivation const* x,int const* offsets,float* out,
                  int n,int bpr,int experts,int max_rows,gemv_stream_t stream=nullptr){
  // At the real Q4 decode shape RPW=4 halves the lanes spent only on subgroup padding. Four-warp CTAs then expose
  // enough independent blocks to saturate cold delivery; eight-warp CTAs measured 5.2-5.7% behind raw over 64
  // distinct cold 2048x2048 layers. Threads=128 removed that gap while also taking warm BC 0.9-1.4% ahead of raw.
  constexpr int Threads=(T==KType::Q4_K&&RowsPerWarp==4)?128:256;
  // In dense mode max_rows is the decode batch M.  Existing M=1 callers are
  // launch compatible; M=2..7 gain native one-launch batching.
  dim3 grid(vecdot::vecdot_grid_size<T,RowsPerWarp>(n,Threads),max_rows,Grouped?experts:1);
  rows_kernel<T,ArtifactTileK,RowsPerWarp,Grouped><<<grid,Threads,0,stream>>>(low,high,units,x,out,n,bpr,offsets);
}

template <KType T, int ArtifactTileK, bool Grouped>
void launch(uint8_t const* low,uint8_t const* high,uint8_t const* units,
            vecdot::VecdotActivation const* x,int const* offsets,float* out,
            int n,int bpr,int experts,int max_rows,gemv_stream_t stream=nullptr){
#if !defined(__HGGCCC__)
#if defined(__CUDACC__)
  // Stock CUDA gets the independently measured Q4/A64 dense topology.  It
  // consumes the same resident bytes as prefill; only the reader/work
  // ownership changes.  PPU, grouped mode, other formats/arrangements and K
  // tails retain the generic route until their own device evidence exists.
  if constexpr (T == KType::Q4_K && ArtifactTileK == 64 && !Grouped) {
    if (::gguf_scale::bc_q4_gemv::launch_default(
            reinterpret_cast<half const*>(x), low, units,
            out, unsigned(max_rows), unsigned(n), unsigned(bpr * 256), stream))
      return;
  }
#endif
#endif
  switch(rows_per_warp<T>(n,bpr)){
    case 1:launch_fixed<T,ArtifactTileK,1,Grouped>(low,high,units,x,offsets,out,n,bpr,experts,max_rows,stream);break;
    case 2:launch_fixed<T,ArtifactTileK,2,Grouped>(low,high,units,x,offsets,out,n,bpr,experts,max_rows,stream);break;
    case 4:launch_fixed<T,ArtifactTileK,4,Grouped>(low,high,units,x,offsets,out,n,bpr,experts,max_rows,stream);break;
    case 8:launch_fixed<T,ArtifactTileK,8,Grouped>(low,high,units,x,offsets,out,n,bpr,experts,max_rows,stream);break;
  }
}

} // namespace bc_vecdot
} // namespace gguf_scale
