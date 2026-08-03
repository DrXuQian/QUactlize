// L104 -- DOES THE CROSS-PLANE HIGH-BIT PLACEMENT SHARE THE LOW PLANE'S TILE INVARIANCE?
//
// l61 already established the answer for the unfolded standalone/low plane across this exact 11-row tactic grid.
// This file deliberately does not repeat that result. It runs place_hi for Q3 (i2+i1), Q5 (i4+i1) and Q6 (i4+i2)
// on one common N/K buffer, records the STORED bytes, and groups equal outputs. F1/F2 are the smallest fold factors
// that give each plane a complete 32-byte delivery; the printed DL1/DL2 values are the resulting per-row delivery
// counts. Comparing rows with the same F pair but different DL pair is the experiment this file exists to make.
//
// RESULT. place_hi is NOT invariant at fixed (F1,F2). Across every compatible row, stored-byte equality is exactly
// classified by the derived tuple
//     (F1, F2, DL1, DL2, F2 > 1 ? TN/F2 : 0).
// DL1/DL2 are the low/high deliveries per physical row. TN/F2 is the folded high plane's physical row count; it is
// deliberately zeroed when F2==1 because the unfolded writer cancels TN and is invariant to it. TM, WM and WN do not
// survive into the bytes anywhere in this grid. Thus an artifact needs more than (bits,F1,F2), but not a whole tactic:
// the two delivery counts plus the folded high-row count describe every observed split, including Q6 TK128 vs TK256.
//
// BUILD/RUN (host-side cute layouts; no device required):
//   nvcc -std=c++17 -x cu -arch=sm_80 -w -I stub_inc -I ../../include \
//        -I ../../../third_party/actlize/include l104_place_hi_invariance.cu -o /tmp/l104 && /tmp/l104
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "xplane_offline.hpp"

namespace {

// One common whole-tile buffer. 256 is divisible by every TN/TK in l61's grid; larger outer repetition only repeats
// the same local placement and makes the host-side cute walk needlessly expensive.
constexpr int N = 256, K = 256;

template <int Bits, int TK>
constexpr int delivery_fold() {
  static_assert(Bits == 1 || Bits == 2 || Bits == 4, "the live planes are 1/2/4 bits");
  constexpr int run_bits = Bits * TK;
  static_assert(256 % run_bits == 0 || run_bits % 256 == 0,
                "the tactic must divide or fill one 32-byte AIU delivery");
  return run_bits < 256 ? 256 / run_bits : 1;
}

uint64_t hash_bytes(std::vector<int8_t> const& bytes) {
  uint64_t h = UINT64_C(1469598103934665603);
  for (int8_t v : bytes) {
    h ^= uint8_t(v);
    h *= UINT64_C(1099511628211);
  }
  return h;
}

size_t byte_diff(std::vector<int8_t> const& a, std::vector<int8_t> const& b) {
  if (a.size() != b.size()) return std::max(a.size(), b.size());
  size_t d = 0;
  for (size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
  return d;
}

std::vector<uint8_t> input_codes(int bits, bool planted_fault = false) {
  uint32_t s = UINT32_C(0x9e3779b9);
  std::vector<uint8_t> q(size_t(N) * K);
  uint8_t const mask = uint8_t((1 << bits) - 1);
  for (size_t i = 0; i < q.size(); ++i) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    q[i] = uint8_t(s) & mask;
    if (planted_fault) q[i] ^= mask;
  }
  return q;
}

struct Result {
  char const* config;
  int tm, tn, tk, wm, wn, f1, f2, dl1, dl2, ni1, ni2;
  bool valid;
  uint64_t hash;
  std::vector<int8_t> bytes;
};

template <int LowBits, int HiBits, int TM, int TN, int TK, int WM, int WN>
Result run(char const* config, std::vector<uint8_t> const& q) {
  constexpr int F1 = delivery_fold<LowBits, TK>();
  constexpr int F2 = delivery_fold<HiBits, TK>();
  constexpr int DL1 = F1 * TK * LowBits / 256;
  constexpr int DL2 = F2 * TK * HiBits / 256;
  constexpr int NI1 = WN / (F1 * 16);
  constexpr int NI2 = WN / (F2 * 16);
  static_assert(DL1 >= 1 && DL2 >= 1, "the derived folds must supply complete deliveries");
  if constexpr (NI1 < 1 || NI2 < 1) {
    // This is a real incompatibility in l61's grid, not a skipped comparison: int1 at TK=64 needs F2=4, but WN=32
    // gives fewer than one high-plane N instance. No other F2 both fills 32 bytes and makes the instance complete.
    return {config, TM, TN, TK, WM, WN, F1, F2, DL1, DL2, NI1, NI2, false, 0, {}};
  } else {
    std::vector<int8_t> out(size_t(N) * K * HiBits / 8);
    xplane::place_hi<LowBits, HiBits, TM, TN, TK, WM, WN, F2, F1>(out.data(), q, N, K);
    return {config, TM, TN, TK, WM, WN, F1, F2, DL1, DL2, NI1, NI2, true, hash_bytes(out), std::move(out)};
  }
}

template <int LowBits, int HiBits>
std::vector<Result> sweep(std::vector<uint8_t> const& q) {
  std::vector<Result> r;
  // The exact 11 rows under l61's "tile-invariance of the UNFOLDED placement" heading, now all applied to one format.
  r.push_back(run<LowBits, HiBits,  32,  64, 128, 32, 32>("01", q));
  r.push_back(run<LowBits, HiBits,  64, 128, 128, 32, 32>("02", q));
  r.push_back(run<LowBits, HiBits,  64, 128, 128, 32, 64>("03", q));
  r.push_back(run<LowBits, HiBits,  64, 256, 256, 64, 64>("04", q));
  r.push_back(run<LowBits, HiBits, 128, 128, 128, 64, 64>("05", q));
  r.push_back(run<LowBits, HiBits,  64,  64, 256, 32, 32>("06", q));
  r.push_back(run<LowBits, HiBits,  64, 128, 256, 64, 64>("07", q));
  r.push_back(run<LowBits, HiBits,  64,  64, 128, 32, 32>("08", q));
  r.push_back(run<LowBits, HiBits,  64, 128,  64, 32, 32>("09", q));
  r.push_back(run<LowBits, HiBits,  64, 128, 128, 64, 64>("10", q));
  r.push_back(run<LowBits, HiBits,  64, 256, 256, 64, 64>("11", q));
  return r;
}

template <int LowBits, int HiBits>
bool report(char const* format) {
  auto const q = input_codes(HiBits);
  auto const planted = input_codes(HiBits, true);
  std::vector<Result> r = sweep<LowBits, HiBits>(q);
  Result fault = run<LowBits, HiBits, 32, 64, 128, 32, 32>("01-fault", planted);

  // A HASH CONTROL BEFORE ANY EQUALITY CLAIM: different input reaches a separately instantiated/called sweep and
  // must move both stored bytes and hash. A comparison accidentally fed the same buffer twice fails here first.
  size_t const control_diff = byte_diff(r[0].bytes, fault.bytes);
  bool const control_ok = control_diff > 0 && r[0].hash != fault.hash;
  std::printf("%s planted-control config=01 hash=%016llx -> %016llx diff=%zu/%zu %s\n", format,
              (unsigned long long)r[0].hash, (unsigned long long)fault.hash, control_diff, r[0].bytes.size(),
              control_ok ? "MOVED" : "FAILED");

  std::map<uint64_t, std::vector<std::string>> groups;
  for (Result const& x : r) {
    if (!x.valid) {
      std::printf("%s config=%s tile=%dx%dx%d_w%dx%d F=%d/%d DL=%d/%d NI=%d/%d INCOMPATIBLE\n",
                  format, x.config, x.tm, x.tn, x.tk, x.wm, x.wn, x.f1, x.f2, x.dl1, x.dl2, x.ni1, x.ni2);
      continue;
    }
    std::printf("%s config=%s tile=%dx%dx%d_w%dx%d F=%d/%d DL=%d/%d NI=%d/%d hash=%016llx\n",
                format, x.config, x.tm, x.tn, x.tk, x.wm, x.wn, x.f1, x.f2, x.dl1, x.dl2, x.ni1, x.ni2,
                (unsigned long long)x.hash);
    groups[x.hash].push_back(x.config);
  }
  std::printf("%s equal-hash sets:", format);
  for (auto const& [hash, configs] : groups) {
    std::printf(" {");
    for (size_t i = 0; i < configs.size(); ++i) std::printf("%s%s", i ? "," : "", configs[i].c_str());
    std::printf("}=%016llx", (unsigned long long)hash);
  }
  std::printf("\n");

  // Hashes are only the report. Verify every same-hash pair byte-for-byte so a collision cannot manufacture an
  // agreeing set, and every different-hash pair really has at least one different byte.
  bool honest = control_ok;
  bool descriptor_exact = true;
  for (size_t i = 0; i < r.size(); ++i)
    for (size_t j = i + 1; j < r.size(); ++j)
      if (r[i].valid && r[j].valid) {
        bool const same_bytes = byte_diff(r[i].bytes, r[j].bytes) == 0;
        honest &= (r[i].hash == r[j].hash) == same_bytes;
        auto descriptor = [](Result const& x) {
          return std::make_tuple(x.f1, x.f2, x.dl1, x.dl2, x.f2 > 1 ? x.tn / x.f2 : 0);
        };
        descriptor_exact &= (descriptor(r[i]) == descriptor(r[j])) == same_bytes;
      }
  std::printf("%s derived descriptor (F1,F2,DL1,DL2,folded_R2) %s classifies stored-byte equality\n",
              format, descriptor_exact ? "EXACTLY" : "DOES NOT");
  honest &= descriptor_exact;
  return honest;
}

}  // namespace

int main() {
  bool const q3 = report<2, 1>("Q3(i2+i1)");
  bool const q5 = report<4, 1>("Q5(i4+i1)");
  bool const q6 = report<4, 2>("Q6(i4+i2)");
  return q3 && q5 && q6 ? 0 : 1;
}
