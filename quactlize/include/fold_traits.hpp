#pragma once
#include <cstddef>
#include <type_traits>

namespace fold {

// ONE SOURCE FOR THE ARTIFACT N-FOLD. The grouped consumer originally owned this expression; the offline producer
// now needs it too. The input is ArtifactTileK, never a tactic row's TileK: accepting the latter would let one kernel
// reinterpret the resident bytes. Keep the legality checks in the template so an invalid artifact never instantiates.
template <int Bits, int ArtifactTileK>
struct DeliveryFold {
  static_assert(Bits > 0 && ArtifactTileK > 0, "fold: bit width and ArtifactTileK must be positive");
  static constexpr int contig_bytes = ArtifactTileK * Bits / 8;
  static_assert(contig_bytes > 0, "fold: the K run must contain at least one byte");
  static_assert(contig_bytes >= 32 || 32 % contig_bytes == 0,
                "fold: a sub-32B K run must divide the AIU's 32B delivery");
  static constexpr int value = contig_bytes >= 32 ? 1 : 32 / contig_bytes;
};

template <int Bits, int ArtifactTileK>
inline constexpr int delivery_fold_v = DeliveryFold<Bits, ArtifactTileK>::value;

// Keep the host model on the same first-version selector as GetMmaInstForShape: only the exact TM8/WM8 family uses
// m8n16k16.  Its logical tile is eight rows, but the AIU staging cube remains physically 16 rows high.
template <int TM, int WM>
inline constexpr int selected_inst_m = (TM == 8 && WM == 8) ? 8 : 16;
template <int WM>
inline constexpr int selected_inst_m_for_warp = WM == 8 ? 8 : 16;
template <int TM>
inline constexpr int physical_a_rows = TM < 16 ? 16 : TM;

// WM/WN default to the common 32x32 warp tile. They only feed the occupancy estimate -- pass the real ones when
// analysing a config that uses something else (the Q3 B-concat sweep runs 16x32 warps at TM=16, which divides by
// zero under a hardcoded /32).
template <int Bits, int TM, int TN, int TK, int Stages = 3, int WM = 32, int WN = 32,
          int InstM = selected_inst_m<TM, WM>>
struct FoldTraits {
  // ---- fold geometry ----
  static constexpr int instruction_m = InstM;
  static constexpr int contig_bytes = TK * Bits / 8;              // bytes one N-column contributes along K
  static constexpr int F            = delivery_fold_v<Bits, TK>;    // N-columns folded into a 32B run
  static constexpr int Ng           = TN / F;                     // physical rows per tile
  static constexpr int run_codes    = F * TK;                     // codes in one 32B contiguous run
  static constexpr int codes_per_word = 32 / Bits;

  // ---- the two quantities that must agree ----
  static constexpr int delivery = 16 * 8 / Bits;                  // one swzl instruction hands a thread 16 BYTES
  static constexpr int slots    = WN * TK / 32;                   // fp16 B-fragment slots per thread
  static constexpr int b_m_reuse = WM / InstM;                    // M-direction mma uses per converted B element
  static constexpr int cvt_elems_per_mma = 8 * InstM / WM;

  // ---- occupancy ----
  static constexpr int a_smem   = physical_a_rows<TM> * TK * 2;  // physical fp16 A cube per stage
  static constexpr int b_smem   = Ng * (F * TK * Bits / 8);
  static constexpr int smem     = (a_smem + b_smem) * Stages;
  static constexpr int warps    = ((TM + WM - 1) / WM) * ((TN + WN - 1) / WN);
  static constexpr int blk_smem = 262144 / smem;                  // 256KB per CU (hard cap)
  static constexpr int blk_warp = 64 / (warps > 0 ? warps : 1);   // 64 warps per CU
  // WARNING: this omits the REGISTER limit and is therefore not the occupancy the hardware gives you. acu showed
  // shared memory allowing 10 blocks where registers allowed 4. Use fold::warps_per_cu<> below for anything
  // performance-related; this stays only because ft_check prints it alongside the fold geometry.
  static constexpr int blocks   = blk_smem < blk_warp ? blk_smem : blk_warp;

  // ---- invariants ----
  // I1 is the real one: OVER-delivery is fatal, because the surplus is never fetched into the fragment at all.
  // Under-delivery is fine -- int4 lives there (32 codes into 64 slots) and simply issues more swzl steps.
  static_assert(delivery <= slots,
      "fold: swzl delivers more codes than the mma fragment has slots; the surplus is silently never read "
      "(this is int1 F=4 at TN=64). Raise TN or TK, or lower F.");
  // Structural sanity, not fold theory.
  static_assert(TN % F == 0, "fold: TileN must divide into F groups");
  // N/K remain 16-wide for both atoms. M follows the selected instruction: the exact TM8/WM8 family is m8, while
  // every established family remains m16. The physical A cube is intentionally independent of this logical extent.
  static_assert(InstM == selected_inst_m<TM, WM>,
                "fold: InstM must match GetMmaInstForShape's exact TM8/WM8 selection");
  static_assert((InstM == 8 || InstM == 16) && TK % 16 == 0 && TN % 16 == 0 &&
                TM % InstM == 0 && WM % InstM == 0,
                "fold: tile and warp must align to the selected mma atom");
  static_assert(contig_bytes * F == 32 || F == 1, "fold: the folded run must be exactly 32B");
  // DERIVED, not asserted: how many logical columns the offline must fit into ONE 32-bit word. 1 means the
  // simple whole-word packer works; >1 means it must interleave columns at bit granularity. int1 at TK=64
  // needs 2, and that -- not any hardware limit -- is what the current packer cannot do.
  // Also measured on the real TiledMma (same file): one thread demands WN/8 columns x TK/4 k each, and one swzl
  // delivery is 4 words of 32/Bits codes. Both are functions of WN, not TN.
  static constexpr int cols_demanded = WN / 8;
  static constexpr int k_per_col     = TK / 4;
  static constexpr int words_per_dlv = 4;
};

template <int TM, int TN, int WM, int WN>
inline constexpr bool warp_shape_ok = (WM > 0 && WN > 0 && TM % WM == 0 && TN % WN == 0
                                       && TM / WM >= 1 && TN / WN >= 1 && (TM / WM) * (TN / WN) <= 16);

template <int TM, int TN, int TK, int WM, int WN, bool Zero = false>
inline constexpr int regs_per_thread = WM * WN / 32              // accumulator, fp32
                                     + WM * TK / 64              // A fragment
                                     + WN * TK / 64              // B fragment
                                     + WN * TK / 256 * (Zero ? 2 : 1);
template <int TM, int TN, int TK, int WM, int WN, bool Zero = false>
inline constexpr bool regs_ok = regs_per_thread<TM, TN, TK, WM, WN, Zero> <= 256;


// OCCUPANCY, corrected by acu. My blk formula was min(smem_limit, warp_limit) and it was MISSING THE REGISTER
// LIMIT, which is why blk kept failing to order the data. acu on ladder rungs 3 and 4:
//     rung 3  Regs=112  256 thr/blk  Block Limit Registers ...  theoretical 32 warp/CU  achieved 31.0
//     rung 4  Regs=186  128 thr/blk  Block Limit Registers 4    theoretical 16 warp/CU  achieved 15.2
// and it says so outright: "theoretical occupancy (25.0%) is limited by the number of required registers", with
// shared memory allowing 10 blocks where registers allow 4.
//
// REGISTERS ARE BILLED ROUNDED UP TO A POWER OF TWO. Reverse it from acu's own warp counts: 131072/32 warps =
// 128 reg/thread for a kernel reporting 112, and 131072/16 = 256 for one reporting 186. So 112 is billed as 128
// and 186 as 256 -- 129 registers cost exactly as much as 256. That cliff, not the raw count, is what matters,
// and it is why regs=176 and regs=104 behave like different animals while 176 vs 240 do not.
// ESTIMATE-TO-MEASURED OFFSET. regs_per_thread is consistently ~22 BELOW what acu reports, at both points measured:
//     estimate 176 -> acu 186        estimate 128 -> acu 142     (int1, (32,128,64) w32x64, unchunked / B-chunked)
// so ~10 and ~14; 12 splits them. IT DOES NOT EXTRAPOLATE RELIABLY: the chunked sweep gained +10.2 points at
// (32,128,128) w32x32, which only makes sense if that config crossed into the 128 bucket, yet 120 + 12 = 132 says it
// did not. Two anchor points cannot call a boundary. Treat the model as ordering configs, not as deciding buckets --
// only acu on the specific config does that.
// The gap is address arithmetic, pipeline temporaries and the packed source registers -- things the model does not
// count. It lives here rather than in a comment somewhere because every billing decision compares against 128 or 256,
// so the estimate is useless without it: 120 looks like it clears 128 and does not.
//
// THIRD ANCHOR, AND IT BREAKS THE OFFSET FOR CHUNKED CONFIGS (acu, 2026-07-27, the shipping best config
// (64,128,64) w64x64 s2 gs=32 ScaleOnly int1 with typed atom-at-a-time B conversion): acu reports **140 regs**,
// grid (64,32,1)x(64,1,1),
// 260.90 us on ONE cold launch. regs_per_thread_chunked predicts 224 (accum 128 + A 64 + B 16 + scale 16), so the
// error is **-84**, not +12. Both earlier anchors (176->186, 128->142) were UNCHUNKED. The compiler is evidently not
// keeping accum and the whole A fragment live simultaneously -- 140 is below accum+A = 192 on its own.
// So: regs_measured_offset applies to the UNCHUNKED model only, and even there two anchors cannot call a bucket
// boundary. For chunked configs the model is an UPPER BOUND and an ordering, nothing more. Consequence worth keeping
// in mind: at 140 regs that config's register limit is 131072/(140*32) = 29 warps = 14 blocks, while smem allows 13 --
// so it is SMEM-limited at ~13 blocks / 26 warps per CU, not register-limited at blk=4 as the model claimed.
inline constexpr int regs_measured_offset = 12;
inline constexpr int regs_per_cu = 131072;
template <int R>
inline constexpr int regs_billed = R <= 32 ? 32 : R <= 64 ? 64 : R <= 128 ? 128 : R <= 256 ? 256 : 512;
template <int TM, int TN, int TK, int WM, int WN, bool Zero = false>
inline constexpr int regs_billed_measured =
    regs_billed<regs_per_thread<TM, TN, TK, WM, WN, Zero> + regs_measured_offset>;
template <int TM, int TN, int TK, int WM, int WN, int Stages, int Bits, int Gs = 32, bool Zero = false>
inline constexpr int warps_per_cu = [] {
  constexpr int warps = (TM / WM) * (TN / WN);
  // TOTAL FOR DEGENERATE SHAPES, and that is not defensive dressing. WM > TM or WN > TN makes `warps` zero and
  // the divisions below ill-formed -- and a caller cannot protect this by putting the use inside
  // `if constexpr (legal)`: naming a constexpr VARIABLE TEMPLATE as an argument instantiates it even in a
  // discarded branch. The MoE bench did exactly that and the box build died on
  // `warps_per_cu_chunked<128,32,256,32,64,...>` with "division by zero", for a config moe_ok rejects. Third time
  // in this work that a template named in a discarded statement was instantiated anyway, so the rule is: make the
  // template TOTAL, do not rely on the guard. 0 warps is the honest answer for a shape with no legal warp tiling.
  if constexpr (warps < 1) { return 0; } else {
  constexpr int smem  = (physical_a_rows<TM> * TK * 2 + TN * TK * Bits / 8 +
                         TN * (TK / Gs) * 2 * (Zero ? 2 : 1)) * Stages;
  constexpr int blk_s = 262144 / smem;
  constexpr int blk_w = 64 / warps;
  constexpr int blk_r = regs_per_cu
                      / (regs_billed<regs_per_thread<TM, TN, TK, WM, WN, Zero>> * warps * 32);
  constexpr int blk   = blk_s < blk_w ? (blk_s < blk_r ? blk_s : blk_r) : (blk_w < blk_r ? blk_w : blk_r);
  return blk * warps; }
}();
// CHUNK-AWARE variant. The B fragment holds ONE k-atom instead of MMA_K of them, so its term is 4*MMA_N = WN/4
// instead of WN*TK/64. Added because the sweep printed the UNCHUNKED model in a chunked build -- regs=260/bill=512 next
// to a 63.7% MFU that only makes sense at bill=256 -- which is the third time in this work that a log failed to
// describe the run that produced it.
template <int TM, int TN, int TK, int WM, int WN, bool Zero = false>
inline constexpr int regs_per_thread_chunked = WM * WN / 32          // accumulator, fp32
                                             + WM * TK / 64          // A fragment, still whole
                                             + WN / 4                // B: 4*MMA_N = 4*(WN/16), one k-atom
                                             + WN * TK / 256 * (Zero ? 2 : 1);
template <int TM, int TN, int TK, int WM, int WN, int Stages, int Bits, int Gs = 32, bool Zero = false>
inline constexpr int warps_per_cu_chunked = [] {
  constexpr int warps = (TM / WM) * (TN / WN);
  // TOTAL FOR DEGENERATE SHAPES, and that is not defensive dressing. WM > TM or WN > TN makes `warps` zero and
  // the divisions below ill-formed -- and a caller cannot protect this by putting the use inside
  // `if constexpr (legal)`: naming a constexpr VARIABLE TEMPLATE as an argument instantiates it even in a
  // discarded branch. The MoE bench did exactly that and the box build died on
  // `warps_per_cu_chunked<128,32,256,32,64,...>` with "division by zero", for a config moe_ok rejects. Third time
  // in this work that a template named in a discarded statement was instantiated anyway, so the rule is: make the
  // template TOTAL, do not rely on the guard. 0 warps is the honest answer for a shape with no legal warp tiling.
  if constexpr (warps < 1) { return 0; } else {
  constexpr int smem  = (physical_a_rows<TM> * TK * 2 + TN * TK * Bits / 8 +
                         TN * (TK / Gs) * 2 * (Zero ? 2 : 1)) * Stages;
  constexpr int blk_r = regs_per_cu
      / (regs_billed<regs_per_thread_chunked<TM, TN, TK, WM, WN, Zero> + regs_measured_offset> * warps * 32);
  constexpr int blk_s = 262144 / smem, blk_w = 64 / warps;
  constexpr int blk   = blk_s < blk_w ? (blk_s < blk_r ? blk_s : blk_r) : (blk_w < blk_r ? blk_w : blk_r);
  return blk * warps; }
}();




// WM8 cannot occur in an m16 family under the tactic contract, so the one-argument form remains safe for callers
// that only carry a warp shape (the existing diagnostic probes do exactly that).
template <int WM, int InstM = selected_inst_m_for_warp<WM>>
inline constexpr int amort = WM / InstM;              // M-direction mma reuse of each B fragment element
template <int WM, int InstM = selected_inst_m_for_warp<WM>>
inline constexpr int cvt_per_mma = 8 * InstM / WM;    // B elements converted per mma instruction issued
static_assert(selected_inst_m<8, 8> == 8 && selected_inst_m<16, 16> == 16 && selected_inst_m<16, 8> == 16,
              "host atom selection must match GetMmaInstForShape's exact TM8/WM8 refinement");
static_assert(amort<8, 8> == 1 && cvt_per_mma<8, 8> == 8,
              "m8's B fragment is reused across WM/8 mma instructions");
// Below this the B convert path stops being the constraint; above it, occupancy cannot compensate.
inline constexpr int cvt_per_mma_target = 4;

template <int Bits, int TN, int TK, int WM, int WN>
inline constexpr bool deliverable = (Bits <= 0 || Bits >= 8) ? true : ((16 * 8 / Bits) <= (WN * TK / 32));

template <int Bits, int TN, int TK, int WM, int WN, class = void>
struct CheckDelivery { static constexpr bool ok = true; };

template <int Bits, int TN, int TK, int WM, int WN>
struct CheckDelivery<Bits, TN, TK, WM, WN, std::enable_if_t<(Bits > 0 && Bits < 8)>> {
  static constexpr int delivery = 16 * 8 / Bits;   // one swzl delivery, in codes
  static constexpr int slots    = WN * TK / 32;    // fp16 B-fragment slots per thread
  static_assert(delivery <= slots,
      "fold: one swzl delivery carries more codes than this thread's mma fragment has slots, and the surplus is "
      "never fetched. Need WN*TK*Bits >= 4096. Raise TileShape.K, or raise the warp N extent WN -- raising "
      "TileShape.N does NOT help, because B is split across the warps in N and slots does not depend on TN. "
      "Note that raising WN past 32 also requires a bit-granular offline packer (cols_per_word = WN/32).");
  static constexpr bool ok = true;
};

} // namespace fold
