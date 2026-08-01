// GENERATED STUB for the LOCAL front-end check only. The real one is written by CMakeLists at configure time from
// _SPLITK_CFGS; without a copy here the front-end check of test_moe_splitk_bench stops at this include and the
// file is never checked at all -- which is how a bench edit could reach the box unchecked.
#define SPLITK_UNIT_COUNT 4
void splitk_unit_16x128_256_w16x16_s2(const Band&, const SkCtx&, SkBest&, SkBest&);
void splitk_unit_16x64_256_w16x16_s2(const Band&, const SkCtx&, SkBest&, SkBest&);
void splitk_unit_16x128_128_w16x16_s2(const Band&, const SkCtx&, SkBest&, SkBest&);
void splitk_unit_16x32_64_w16x16_s4(const Band&, const SkCtx&, SkBest&, SkBest&);
static inline void splitk_run_all(const Band& bd, const SkCtx& cx, SkBest& b1, SkBest& bS) {
  (void)bd; (void)cx; (void)b1; (void)bS;
  splitk_unit_16x128_256_w16x16_s2(bd, cx, b1, bS);
  splitk_unit_16x64_256_w16x16_s2(bd, cx, b1, bS);
  splitk_unit_16x128_128_w16x16_s2(bd, cx, b1, bS);
  splitk_unit_16x32_64_w16x16_s4(bd, cx, b1, bS);
}
