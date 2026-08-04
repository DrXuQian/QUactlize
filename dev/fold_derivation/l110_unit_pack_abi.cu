// The forward C ABI is the only way a .so-only integrator can produce the metadata both packed GEMMs require.
// Exercise every runtime qtype, its size query, and grouped expert addressing without a device.
#include "../../quactlize/csrc/device/ppu_unit_pack.cpp"

#include <cstdio>
#include <cstring>
#include <vector>

int main() {
  struct F { int qtype, raw, unit, supers; };
  F const formats[] = {{10,84,20,1}, {11,110,28,2}, {12,144,16,1}, {13,176,16,1}, {14,210,36,2}};
  constexpr int n = 256, k = 512, experts = 2, nsb = k / 256;
  int bad = 0;
  for (F f : formats) {
    int64_t const expected_bytes = int64_t(n) * (nsb / f.supers) * f.unit;
    int64_t const queried = quactlize_ppu_units_bytes(n, k, f.qtype);
    bad += queried != expected_bytes;

    std::vector<uint8_t> blocks(size_t(experts) * n * nsb * f.raw);
    for (int e = 0; e < experts; ++e)
      for (size_t i = 0; i < size_t(n) * nsb * f.raw; ++i)
        blocks[size_t(e) * n * nsb * f.raw + i] = uint8_t((i * 37 + e * 91 + f.qtype) & 255);
    std::vector<uint8_t> grouped(size_t(experts) * queried, 0xcc), dense0(size_t(queried), 0xdd),
                         dense1(size_t(queried), 0xee);
    bad += quactlize_ppu_prepare_units_grouped(
        blocks.data(), grouped.data(), n, k, experts, f.qtype) != 0;
    bad += quactlize_ppu_prepare_units(blocks.data(), dense0.data(), n, k, f.qtype) != 0;
    bad += quactlize_ppu_prepare_units(
        blocks.data() + size_t(n) * nsb * f.raw, dense1.data(), n, k, f.qtype) != 0;
    bad += std::memcmp(grouped.data(), dense0.data(), size_t(queried)) != 0;
    bad += std::memcmp(grouped.data() + queried, dense1.data(), size_t(queried)) != 0;
    bad += dense0 == dense1;  // the expert-base fault would make this true
  }

  std::vector<uint8_t> scratch(1024);
  bad += quactlize_ppu_units_bytes(256, 256, 11) != -1;  // Q3 pairs superblocks
  bad += quactlize_ppu_units_bytes(256, 512, 99) != -1;
  bad += quactlize_ppu_prepare_units(scratch.data(), scratch.data(), 256, 256, 11) != 24;
  bad += quactlize_ppu_prepare_units(scratch.data(), scratch.data(), 256, 512, 99) != 22;
  bad += quactlize_ppu_prepare_units(nullptr, scratch.data(), 256, 512, 12) != 20;

  std::printf("unit-pack-abi formats=5 grouped=5 bad=%d\n", bad);
  return bad ? 1 : 0;
}
