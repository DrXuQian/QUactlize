#include <cstdio>

int run_ppu_m8n16_g1();
#if !defined(PPU_M8_G0_ONLY)
int run_ppu_m8n16_g2();
#endif

int main() {
  int const g1 = run_ppu_m8n16_g1();
#if defined(PPU_M8_G0_ONLY)
  // This executable must never be produced on ppu0015: the raw G1 atom is
  // expected to fail m8n16k16 intrinsic selection before link.  Keeping G2
  // out of this build ensures an unrelated ppu001-only AIU diagnostic cannot
  // satisfy G0's negative gate.
  std::printf("== [G0-only] UNEXPECTED LINK: G1=%d ==\n", g1);
  return g1;
#else
  int const g2 = run_ppu_m8n16_g2();
  std::printf("== [111] %s: G1=%d G2=%d ==\n", (g1 == 0 && g2 == 0) ? "PASS" : "FAIL", g1, g2);
  return (g1 == 0 && g2 == 0) ? 0 : 1;
#endif
}
