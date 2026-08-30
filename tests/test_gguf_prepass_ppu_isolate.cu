// PPU-only carrier isolate for the two production GGUF metadata prepass kernels.
//
// This is intentionally an executable, not another dlopen library and not a copied kernel body.  It launches the
// exact __global__ templates from gguf_scale_prepass.hpp with their production descriptors, while a one-store
// marker proves that this small HGCC image itself is executable.  Host goldens call the established host definitions;
// poisoned device outputs distinguish a missing launch from a numerically wrong launch.
#include <hggc_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "gguf_scale_prepass.hpp"
#include "gguf_unit_pack.hpp"

namespace {

using cutlass::half_t;
using gguf_scale::KType;
using Raw = gguf_scale::unit_pack::Raw<KType::Q2_K>;
using Unit = gguf_scale::packed_unit::Unit<KType::Q2_K>;
using Traits = gguf_scale::Traits<KType::Q2_K>;

constexpr uint16_t kPoison = 0x7bffu;
constexpr int kN = 256;
constexpr int kK = 512;
constexpr int kSuperblocks = kK / 256;
constexpr int kRawCount = kN * kSuperblocks;
constexpr int kGroups = Traits::kGroups;

template <class T>
struct DeviceBuffer {
  T* data = nullptr;
  size_t count = 0;

  explicit DeviceBuffer(size_t n) : count(n) {
    if (hggcMalloc(reinterpret_cast<void**>(&data), n * sizeof(T)) != hggcSuccess) data = nullptr;
  }
  ~DeviceBuffer() { if (data) hggcFree(data); }
  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;

  bool valid() const { return data != nullptr; }
  bool from_host(void const* src) const {
    return hggcMemcpy(data, src, count * sizeof(T), hggcMemcpyHostToDevice) == hggcSuccess;
  }
  bool to_host(void* dst) const {
    return hggcMemcpy(dst, data, count * sizeof(T), hggcMemcpyDeviceToHost) == hggcSuccess;
  }
};

__global__ void carrier_marker(uint16_t* out) {
  if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = 0x3c00u;
}

struct LaunchStatus {
  int before = 0;
  int immediate = 0;
  int synchronize = 0;
};

int begin_launch() { return int(hggcGetLastError()); }

LaunchStatus finish_launch(int before) {
  return LaunchStatus{before, int(hggcGetLastError()), int(hggcDeviceSynchronize())};
}

size_t mismatch(std::vector<uint16_t> const& got, std::vector<half_t> const& want) {
  size_t bad = 0;
  for (size_t i = 0; i < got.size(); ++i) bad += got[i] != want[i].raw();
  return bad;
}

size_t sentinel(std::vector<uint16_t> const& values) {
  return size_t(std::count(values.begin(), values.end(), kPoison));
}

size_t nonzero(std::vector<half_t> const& values) {
  return size_t(std::count_if(values.begin(), values.end(), [](half_t x) { return x.raw() != 0; }));
}

size_t xor_negative(std::vector<uint16_t> const& got, std::vector<half_t> const& want) {
  size_t bad = 0;
  for (size_t i = 0; i < got.size(); ++i) bad += got[i] != uint16_t(want[i].raw() ^ 1u);
  return bad;
}

void make_q2_fixture(std::vector<uint8_t>& full, std::vector<uint8_t>& scales,
                     std::vector<half_t>& d, std::vector<half_t>& dmin) {
  full.assign(size_t(kRawCount) * Raw::kBytes, uint8_t(0));
  scales.resize(size_t(kRawCount) * Traits::kBlockBytes);
  d.resize(kRawCount);
  dmin.resize(kRawCount);
  for (int row = 0; row < kRawCount; ++row) {
    uint8_t* block = full.data() + size_t(row) * Raw::kBytes;
    for (int g = 0; g < kGroups; ++g) {
      int const sc = 1 + ((row + 3 * g) % 15);
      int const mn = 1 + ((5 * row + g) % 15);
      block[Raw::kScaleOffset + g] = uint8_t(sc | (mn << 4));
    }
    d[size_t(row)] = half_t(0.5f + 0.125f * float(row & 3));
    dmin[size_t(row)] = half_t(0.25f + 0.0625f * float((row >> 2) & 3));
    std::memcpy(block + Raw::kDOffset, &d[size_t(row)], sizeof(half_t));
    std::memcpy(block + Raw::kDminOffset, &dmin[size_t(row)], sizeof(half_t));
    std::memcpy(scales.data() + size_t(row) * Traits::kBlockBytes,
                block + Raw::kScaleOffset, Traits::kBlockBytes);
  }
}

}  // namespace

int main() {
  std::vector<uint8_t> full, raw_scales;
  std::vector<half_t> raw_d, raw_dmin;
  make_q2_fixture(full, raw_scales, raw_d, raw_dmin);

  DeviceBuffer<uint16_t> marker(1);
  uint16_t marker_poison = kPoison, marker_got = kPoison;
  if (!marker.valid() || !marker.from_host(&marker_poison)) return 2;
  int const marker_before = begin_launch();
  carrier_marker<<<1, 32>>>(marker.data);
  LaunchStatus const marker_launch = finish_launch(marker_before);
  if (!marker.to_host(&marker_got)) return 3;

  size_t const raw_elems = size_t(kRawCount) * kGroups;
  std::vector<half_t> raw_gold_s(raw_elems), raw_gold_z(raw_elems);
  gguf_scale::prepass::BlockDesc raw_host_src{
      raw_scales.data(), raw_d.data(), raw_dmin.data(), Traits::kBlockBytes, 0, 1, 0};
  gguf_scale::prepass::PlaneDesc raw_host_dst{raw_gold_s.data(), raw_gold_z.data(), kGroups, 1};
  gguf_scale::prepass::prepass_host<KType::Q2_K, 0>(raw_host_src, raw_host_dst, kRawCount, 1);

  DeviceBuffer<uint8_t> raw_device_scales(raw_scales.size());
  DeviceBuffer<half_t> raw_device_d(raw_d.size()), raw_device_dmin(raw_dmin.size());
  DeviceBuffer<half_t> raw_device_s(raw_elems), raw_device_z(raw_elems);
  std::vector<uint16_t> poison_raw(raw_elems, kPoison), raw_got_s(raw_elems), raw_got_z(raw_elems);
  if (!raw_device_scales.valid() || !raw_device_d.valid() || !raw_device_dmin.valid() ||
      !raw_device_s.valid() || !raw_device_z.valid() ||
      !raw_device_scales.from_host(raw_scales.data()) || !raw_device_d.from_host(raw_d.data()) ||
      !raw_device_dmin.from_host(raw_dmin.data()) || !raw_device_s.from_host(poison_raw.data()) ||
      !raw_device_z.from_host(poison_raw.data())) return 4;

  gguf_scale::prepass::BlockDesc raw_device_src{
      raw_device_scales.data, raw_device_d.data, raw_device_dmin.data, Traits::kBlockBytes, 0, 1, 0};
  gguf_scale::prepass::PlaneDesc raw_device_dst{raw_device_s.data, raw_device_z.data, kGroups, 1};
  auto const raw_args = gguf_scale::prepass::make_prepass_kernel_args(
      raw_device_src, raw_device_dst, kRawCount, 1);
  int const raw_grid = gguf_scale::prepass::prepass_grid_size(kRawCount, 1, 256);
  int const raw_before = begin_launch();
  gguf_scale::prepass::prepass_kernel<KType::Q2_K, 0><<<raw_grid, 256>>>(raw_args);
  LaunchStatus const raw_launch = finish_launch(raw_before);
  if (!raw_device_s.to_host(raw_got_s.data()) || !raw_device_z.to_host(raw_got_z.data())) return 5;

  int const num_units = kSuperblocks / Unit::kSbPerUnit;
  size_t const unit_bytes = size_t(num_units) * kN * Unit::kUnitTotal;
  size_t const packed_elems = size_t(kSuperblocks) * kGroups * kN;
  std::vector<uint8_t> units(unit_bytes);
  gguf_scale::unit_pack::pack<KType::Q2_K>(full.data(), units.data(), kN, kK, 1);
  std::vector<half_t> packed_gold_s(packed_elems), packed_gold_z(packed_elems);
  gguf_scale::prepass::UnitPlaneDesc packed_host_dst{
      packed_gold_s.data(), packed_gold_z.data(), int64_t(packed_elems), kN, 1};
  gguf_scale::prepass::prepass_unit_host<KType::Q2_K, 0>(
      units.data(), packed_host_dst, 1, kN, kSuperblocks);

  DeviceBuffer<uint8_t> packed_device_units(unit_bytes);
  DeviceBuffer<half_t> packed_device_s(packed_elems), packed_device_z(packed_elems);
  std::vector<uint16_t> poison_packed(packed_elems, kPoison),
      packed_got_s(packed_elems), packed_got_z(packed_elems);
  if (!packed_device_units.valid() || !packed_device_s.valid() || !packed_device_z.valid() ||
      !packed_device_units.from_host(units.data()) || !packed_device_s.from_host(poison_packed.data()) ||
      !packed_device_z.from_host(poison_packed.data())) return 6;
  gguf_scale::prepass::UnitPlaneDesc packed_device_dst{
      packed_device_s.data, packed_device_z.data, int64_t(packed_elems), kN, 1};
  auto const packed_args = gguf_scale::prepass::make_unit_prepass_kernel_args(
      packed_device_units.data, packed_device_dst, 1, kN, kSuperblocks);
  int const packed_grid = gguf_scale::prepass::prepass_unit_grid_size<KType::Q2_K>(
      1, kN, kSuperblocks, 256);
  int const packed_before = begin_launch();
  gguf_scale::prepass::prepass_unit_kernel<KType::Q2_K, 0><<<packed_grid, 256>>>(packed_args);
  LaunchStatus const packed_launch = finish_launch(packed_before);
  if (!packed_device_s.to_host(packed_got_s.data()) || !packed_device_z.to_host(packed_got_z.data())) return 7;

  size_t const marker_bad = marker_got != uint16_t(0x3c00u);
  size_t const raw_bad = mismatch(raw_got_s, raw_gold_s) + mismatch(raw_got_z, raw_gold_z);
  size_t const packed_bad = mismatch(packed_got_s, packed_gold_s) + mismatch(packed_got_z, packed_gold_z);
  size_t const raw_sentinel = sentinel(raw_got_s) + sentinel(raw_got_z);
  size_t const packed_sentinel = sentinel(packed_got_s) + sentinel(packed_got_z);
  size_t const raw_red_bad = xor_negative(raw_got_s, raw_gold_s) + xor_negative(raw_got_z, raw_gold_z);
  size_t const packed_red_bad = xor_negative(packed_got_s, packed_gold_s) + xor_negative(packed_got_z, packed_gold_z);
  size_t const raw_gold_nonzero = nonzero(raw_gold_s) + nonzero(raw_gold_z);
  size_t const packed_gold_nonzero = nonzero(packed_gold_s) + nonzero(packed_gold_z);

  std::printf(
      "FQ_PREPASS_PPU_ISOLATE marker_bad=%zu marker=[before:%d,immediate:%d,sync:%d] "
      "raw_bad=%zu raw_sentinel=%zu raw_red_bad=%zu raw_gold_nonzero=%zu "
      "raw=[before:%d,immediate:%d,sync:%d] "
      "packed_bad=%zu packed_sentinel=%zu packed_red_bad=%zu packed_gold_nonzero=%zu "
      "packed=[before:%d,immediate:%d,sync:%d]\n",
      marker_bad, marker_launch.before, marker_launch.immediate, marker_launch.synchronize,
      raw_bad, raw_sentinel, raw_red_bad, raw_gold_nonzero,
      raw_launch.before, raw_launch.immediate, raw_launch.synchronize,
      packed_bad, packed_sentinel, packed_red_bad, packed_gold_nonzero,
      packed_launch.before, packed_launch.immediate, packed_launch.synchronize);

  bool const pass = marker_bad == 0 && marker_launch.synchronize == 0 &&
      raw_bad == 0 && raw_sentinel == 0 && raw_red_bad > 0 && raw_gold_nonzero > 0 && raw_launch.synchronize == 0 &&
      packed_bad == 0 && packed_sentinel == 0 && packed_red_bad > 0 && packed_gold_nonzero > 0 &&
      packed_launch.synchronize == 0;
  std::printf("FQ_PREPASS_PPU_ISOLATE_VERDICT verdict=%s carrier=STANDALONE-HGCC kernel_body=PRODUCTION-HEADER\n",
              pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
