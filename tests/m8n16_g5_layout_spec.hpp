#pragma once

// Device-free scalar spelling of the #112/G5 geometry.  The shipping
// type-level contract and L125 both consume these constants; the type-only
// L125 arm then proves that they describe the instantiated collective.
namespace m8n16_g5_layout_spec {

inline constexpr int kBits = 4;
inline constexpr int kN = 32;
inline constexpr int kTacticK = 64;
inline constexpr int kStoredRowK = 256;
inline constexpr int kK = 256;
inline constexpr int kGroupSize = 32;
inline constexpr int kScaleK = kK / kGroupSize;
inline constexpr int kScaleTileK = kTacticK / kGroupSize;
inline constexpr int kStages = 3;
inline constexpr int kExperts = 256;
inline constexpr int kCtaThreads = 32;
inline constexpr bool kAiuInterleaved = false;

static_assert(kScaleK == 8 && kScaleTileK == 2);
static_assert(kN * kScaleK == 256);

}  // namespace m8n16_g5_layout_spec
