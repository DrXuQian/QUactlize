#include <cuda_runtime.h>
#include "kernels.cuh"
#include "tma.cuh"
#include "host.hpp"
#include "gguf_packed_unit.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace quactlize::ppu17;
int checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { std::fprintf(stderr, "check failed %s:%d: %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (false)

q17_problem_v1 problem(int q) {
  q17_problem_v1 p{};
  p.version = 1; p.struct_bytes = sizeof(p);
  p.qtype = q; p.compute_type = Q17_BF16; p.input_type = p.output_type = Q17_F32;
  p.m = p.max_rows = 9; p.n = 256; p.k = 512; p.experts = 1;
  CHECK(q17_canonical_arrangement_v1(q, &p.arrangement) == 0);
  return p;
}

void api_contract() {
  CHECK(validate_launch_options(nullptr, false) == Q17_SUCCESS);
  q17_launch_options_v1 launch{1, sizeof(q17_launch_options_v1), 0, 0};
  CHECK(validate_launch_options(&launch, false) == Q17_SUCCESS);
  launch.programmatic_dependency = 1;
  CHECK(validate_launch_options(&launch, false) == Q17_UNSUPPORTED);
  CHECK(validate_launch_options(&launch, true) == Q17_SUCCESS);
  for (int invalid : {-1, 2}) {
    launch.programmatic_dependency = invalid;
    CHECK(validate_launch_options(&launch, true) == Q17_INVALID);
  }
  launch.programmatic_dependency = 1;
  launch.reserved = 1;
  CHECK(validate_launch_options(&launch, true) == Q17_INVALID);
  launch.reserved = 0;
  launch.struct_bytes--;
  CHECK(validate_launch_options(&launch, true) == Q17_INVALID);
  launch.struct_bytes = sizeof(launch);
  launch.version = 2;
  CHECK(validate_launch_options(&launch, true) == Q17_INVALID);
  for (int q : {8, 10, 11, 12, 13, 14}) {
    auto p = problem(q);
    q17_sizes_v1 sizes;
    CHECK(q17_sizes_query_v1(&p, &sizes) == 0);
    CHECK(sizes.low_bytes == uint64_t(p.n) * p.k * p.arrangement.bits / 8);
    CHECK(q17_config_count_v1(&p) == 6);
    for (int i = 0; i < 5; ++i) {
      q17_config_v1 c;
      CHECK(q17_config_get_v1(&p, i, &c) == 0);
      CHECK(q17_can_implement_v1(&p, &c) == 0);
      c.threads = 64;
      CHECK(q17_can_implement_v1(&p, &c) != 0);
    }
    auto bad = p;
    bad.arrangement.mapping_id ^= 1;
    CHECK(q17_sizes_query_v1(&bad, &sizes) != 0);
    CHECK(sizes.low_bytes == 0 && sizes.units_bytes == 0);
    bad = p; bad.arrangement.artifact_tile_k = 256;
    CHECK(q17_config_count_v1(&bad) < 0);
    bad = p; bad.compute_type = Q17_F32;
    CHECK(q17_config_count_v1(&bad) < 0);
    bad = p; bad.k = 513;
    CHECK(q17_config_count_v1(&bad) < 0);
    bad = p; bad.experts = 2;
    CHECK(q17_config_count_v1(&bad) < 0);
    bad.operation = Q17_GROUPED; bad.max_rows = 5;
    CHECK(q17_config_count_v1(&bad) == (q == 12 ? 11 : 6));
    bad.max_rows = 4;
    CHECK(q17_config_count_v1(&bad) < 0);
    bad = p; bad.struct_bytes--;
    CHECK(q17_config_count_v1(&bad) < 0);
  }
  quactlize_ppu_placed_arrangement_v2 arrangement{};
  for (int q : {8, 10, 11, 12, 13, 14}) {
    auto decode=problem(q);
    decode.m=decode.max_rows=8;
    CHECK(q17_config_count_v1(&decode)==(q==12 ? 11 : 6));
    if(q==12) for(int i=6;i<11;++i) {
      q17_config_v1 cached{};
      CHECK(q17_config_get_v1(&decode,i,&cached)==0);
      CHECK(cached.tile_n==cached.lanes_n && cached.tile_k==32);
      auto prefill=decode; prefill.m=prefill.max_rows=9;
      CHECK(q17_can_implement_v1(&prefill,&cached)==Q17_UNSUPPORTED);
    }
  }
  CHECK(q17_canonical_arrangement_v1(9, &arrangement) != 0);
  CHECK(arrangement.mapping_id == 0);
  CHECK(q17_config_count_v1(nullptr) < 0);
  auto p = problem(12);
  q17_call_v1 c{};
  c.version = 1; c.struct_bytes = sizeof(c); c.problem = p;
  CHECK(q17_config_get_v1(&p, 2, &c.config) == 0);
  c.a_stride = p.k; c.output_stride = p.n;
  CHECK(validate_call(&c, false) != 0);
  std::vector<float> sentinel(4);
  c.a = c.low = c.units = sentinel.data(); c.output = sentinel.data();
  CHECK(validate_call(&c, false) == 0);
  c.low = reinterpret_cast<uint8_t *>(sentinel.data()) + 1;
  CHECK(validate_call(&c, false) != 0);
  c.low = sentinel.data();
  c.high = sentinel.data(); CHECK(validate_call(&c, false) != 0); c.high = nullptr;
  c.problem.operation = Q17_GROUPED;
  CHECK(validate_call(&c, false) != 0);
  int offsets[] = {0, 9}; c.offsets = offsets;
  CHECK(validate_call(&c, false) == 0);
  CHECK(row_range(c, 0).size == 9);
  offsets[1] = 0; CHECK(row_range(c, 0).size == 0);
  offsets[1] = 10; CHECK(row_range(c, 0).size == 0);
  auto tma = problem(12);
  q17_config_v1 cfg{};
  CHECK(q17_config_get_v1(&tma, 5, &cfg) == Q17_UNSUPPORTED);
  tma.input_type = Q17_BF16;
  CHECK(q17_config_get_v1(&tma, 5, &cfg) == Q17_SUCCESS);
  c.scale = reinterpret_cast<uint8_t *>(sentinel.data()) + 1;
  c.zero = reinterpret_cast<uint8_t *>(sentinel.data()) + 6;
  CHECK(validate_call(&c, true) == Q17_INVALID);
}

template <int Q> void format_contract() {
  using F = Format<Q>;
  using Unit = gguf_scale::packed_unit::Unit<F::type>;
  constexpr int n = 32, k = 1024;
  CHECK(F::sb_bytes == Unit::kSbBytes && F::sb_per_unit == Unit::kSbPerUnit);
  std::vector<uint8_t> units(F::unit_bytes(n, k));
  std::vector<uint8_t> low(size_t(n) * k * F::low_bits / 8);
  std::vector<uint8_t> high(size_t(n) * k * F::high_bits / 8);
  for (int sb = 0; sb < k / 256; ++sb) for (int col = 0; col < n; ++col) {
    auto p = units.data() + F::unit_offset(n, col, sb * Unit::kGroups);
    auto d = cutlass::half_t(0.000137f * (col + 1) * (sb + 1));
    p[0] = d.raw() & 255; p[1] = d.raw() >> 8;
    if constexpr (Unit::kHasMin) {
      auto dm = cutlass::half_t(0.000271f * (col + 1));
      p[2] = dm.raw() & 255; p[3] = dm.raw() >> 8;
    }
    for (int g = 0; g < Unit::kGroups; ++g) {
      CHECK(F::field_bit(g, 0) == Unit::bit_of(g, 0));
      gguf_scale::packed_unit::put_code<F::type>(p, g, 0, (col * 17 + g * 3 + sb) & ((1 << Unit::kScaleBits) - 1));
      if constexpr (Unit::kHasMin) {
        CHECK(F::field_bit(g, 1) == Unit::bit_of(g, 1));
        gguf_scale::packed_unit::put_code<F::type>(p, g, 1, (col + g + 2 * sb) & ((1 << Unit::kMinBits) - 1));
      }
      auto actual = F::template affine<cutlass::half_t>(units.data(), n, col, sb * Unit::kGroups + g);
      auto expected = gguf_scale::packed_unit::unit_group<F::type, F::correction>(p, g);
      CHECK(actual.scale.raw() == expected.scale.raw());
      CHECK(actual.zero.raw() == expected.zero.raw());
      int sc = gguf_scale::packed_unit::code_of<F::type>(p, g, 0);
      if constexpr (Unit::kSigned) sc = (sc ^ 128) - 128;
      sc -= F::Traits::kScaleBias;
      auto bf = F::template affine<cutlass::bfloat16_t>(units.data(), n, col, sb * Unit::kGroups + g);
      auto expected_scale = cutlass::bfloat16_t(float(d) * sc);
      CHECK(bf.scale.raw() == expected_scale.raw());
      cutlass::bfloat16_t expected_zero(0.f);
      if constexpr (Unit::kHasMin) {
        auto dm = cutlass::half_t::bitcast(uint16_t(p[2]) | uint16_t(p[3]) << 8);
        expected_zero = cutlass::bfloat16_t(-float(dm) * gguf_scale::packed_unit::code_of<F::type>(p, g, 1));
      }
      if constexpr (F::correction != 0) {
        auto correction = cutlass::bfloat16_t(float(F::correction) * float(expected_scale));
        expected_zero = cutlass::bfloat16_t(float(expected_zero) + float(correction));
      }
      CHECK(bf.zero.raw() == expected_zero.raw());
      CHECK(std::isfinite(float(bf.scale)) && std::isfinite(float(bf.zero)));
    }
  }
  for (int col = 0; col < n; ++col) for (int kk = 0; kk < k; ++kk) {
    unsigned value = (col * 17 + kk * 7) & ((1 << (F::low_bits + F::high_bits)) - 1);
    F::Low::placed_put(low.data(), col, kk, n, value & ((1 << F::low_bits) - 1));
    if constexpr (F::high_bits) F::High::placed_put(high.data(), col, kk, n, value >> F::low_bits);
  }
  for (int col = 0; col < n; ++col) for (int kk = 0; kk < k; ++kk) {
    int expected = ((col * 17 + kk * 7) & ((1 << (F::low_bits + F::high_bits)) - 1)) - F::converter_bias;
    CHECK(F::code(reinterpret_cast<uint16_t const *>(low.data()),
                  reinterpret_cast<uint16_t const *>(high.data()), n, col, kk) == expected);
  }
  q17_call_v1 c{};
  c.problem.n = n; c.problem.k = k;
  c.low = low.data(); c.high = F::high_bits ? high.data() : nullptr; c.units = units.data();
  auto compare = [&](auto type) {
    using T = decltype(type);
    std::vector<T> scales(n * (k / F::group_size)), zeros(scales.size());
    for (int group = 0; group < k / F::group_size; ++group) for (int col = 0; col < n; ++col) {
      auto a = F::template affine<T>(units.data(), n, col, group);
      scales[group * n + col] = a.scale; zeros[group * n + col] = a.zero;
    }
    c.scale = scales.data(); c.zero = zeros.data();
    WeightReader<Q, T, false> fq(c, 0);
    WeightReader<Q, T, true> sf(c, 0);
    auto check_weight = [&](int col, int kk) {
      auto affine = F::template affine<T>(units.data(), n, col, kk / F::group_size);
      int code = F::code(reinterpret_cast<uint16_t const *>(low.data()),
                         reinterpret_cast<uint16_t const *>(high.data()), n, col, kk);
      T product(float(code) * float(affine.scale));
      T expected(float(product) + float(affine.zero));
      CHECK(fq(col, kk).raw() == expected.raw());
      CHECK(sf(col, kk).raw() == expected.raw());
    };
    for (int col = 0; col < n; ++col) for (int kk = 0; kk < k; ++kk) check_weight(col, kk);
    // Same group, different columns; then return to a previous column. A
    // group-only cache can pass monotone column-major loops but fails here.
    for (int kk = 0; kk < k; kk += 17)
      for (int col : {0,31,1,1,17,0}) check_weight(col, kk);
  };
  compare(cutlass::half_t{}); compare(cutlass::bfloat16_t{});
}

void q8_contract() {
  constexpr int n = 256, k = 256;
  std::vector<uint8_t> low(n * k);
  std::vector<cutlass::half_t> scale(n * k / 32);
  for (size_t i = 0; i < scale.size(); ++i) scale[i] = cutlass::half_t(float(1+i%13)/128.f);
  for (int col = 0; col < n; ++col) for (int kk = 0; kk < k; ++kk)
    Format<8>::Low::placed_put(low.data(), col, kk, n, (col + kk) % 256);
  for (int col = 0; col < n; ++col) for (int kk = 0; kk < k; ++kk) {
    CHECK(Format<8>::code(reinterpret_cast<uint16_t const *>(low.data()), nullptr, n, col, kk) == (col + kk) % 256 - 128);
    auto a = Format<8>::affine<cutlass::bfloat16_t>(reinterpret_cast<uint8_t const *>(scale.data()), n, col, kk / 32);
    CHECK(a.scale.raw() == cutlass::bfloat16_t(float(scale[(kk/32)*n+col])).raw() && float(a.zero) == 0.f);
  }
  q17_call_v1 c{};
  c.problem.n = n; c.problem.k = k;
  c.low = low.data(); c.units = scale.data();
  auto check_reader = [&](auto type) {
    using T = decltype(type);
    std::vector<T> expanded(scale.size()), zero(scale.size(), T(0.f));
    for (size_t i=0; i<scale.size(); ++i) expanded[i] = T(float(scale[i]));
    c.scale = expanded.data(); c.zero = zero.data();
    WeightReader<8,T,false> fq(c,0); WeightReader<8,T,true> sf(c,0);
    for (int kk=0; kk<k; ++kk) for (int col : {0,255,1,1,127,0}) {
      T product(float((col+kk)%256-128)*float(T(float(scale[(kk/32)*n+col]))));
      T expected(float(product)+float(T(0.f)));
      CHECK(fq(col,kk).raw()==expected.raw());
      CHECK(sf(col,kk).raw()==expected.raw());
    }
  };
  check_reader(cutlass::half_t{}); check_reader(cutlass::bfloat16_t{});
}

template <int Q> void tma_address_contract() {
  using F = Format<Q>;
  using Tile = TmaTile<Q, cutlass::half_t>;
  constexpr int n = 256, k = 1024;
  int noncontiguous_q5 = 0;
  for (int nb = 0; nb < n; nb += 64) for (int kb = 0; kb < k; kb += 64)
    for (int ni = 0; ni < 64; ++ni) for (int ki = 0; ki < 64; ++ki) {
      int col = nb + ni, kk = kb + ki;
      int li = F::Low::physical_kgroup(kk) - F::Low::physical_kgroup(kb);
      CHECK(li >= 0 && li < Tile::low_words);
      CHECK((size_t(F::Low::physical_kgroup(kb)) + li) * n + nb + ni == F::Low::word_index(col, kk, n));
      if constexpr (F::high_bits) {
        int hi = F::High::physical_kgroup(kk) - F::High::physical_kgroup(kb);
        int pn = ni;
        if constexpr (Q == 13) pn = F::High::physical_n(col, kk) - nb;
        CHECK(hi >= 0 && hi < Tile::high_words && pn >= 0 && pn < 64);
        CHECK((size_t(F::High::physical_kgroup(kb)) + hi) * n + nb + pn == F::High::word_index(col, kk, n));
        if constexpr (Q == 13) noncontiguous_q5 += pn != ni;
      }
    }
  if constexpr (Q == 13) CHECK(noncontiguous_q5 > 0);
}

template <class T, int N, int K> void geometry_contract() {
  using Tile = GemmTile<T, N, K>;
  std::vector<int> owners(64 * N, 0), shared_a(cute::cosize_v<typename Tile::LayoutA>, 0),
                   shared_b(cute::cosize_v<typename Tile::LayoutB>, 0);
  for (int m = 0; m < 64; ++m) for (int k = 0; k < K; ++k) ++shared_a[typename Tile::LayoutA{}(m, k)];
  for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) ++shared_b[typename Tile::LayoutB{}(n, k)];
  for (int count : shared_a) CHECK(count == 1);
  for (int count : shared_b) CHECK(count == 1);
  typename Tile::Mma mma;
  for (int thread = 0; thread < 128; ++thread) {
    auto coords = mma.get_thread_slice(thread).partition_C(cute::make_identity_tensor(cute::make_shape(cute::_64{}, cute::Int<N>{})));
    for (int i = 0; i < cute::size(coords); ++i) {
      int m = cute::get<0>(coords(i)), n = cute::get<1>(coords(i));
      CHECK(m >= 0 && m < 64 && n >= 0 && n < N);
      ++owners[m * N + n];
    }
  }
  for (int count : owners) CHECK(count == 1);
}

int main() {
  api_contract();
  format_contract<10>(); format_contract<11>(); format_contract<12>(); format_contract<13>(); format_contract<14>();
  q8_contract();
  tma_address_contract<8>(); tma_address_contract<10>(); tma_address_contract<11>();
  tma_address_contract<12>(); tma_address_contract<13>(); tma_address_contract<14>();
  geometry_contract<cutlass::half_t, 64, 64>(); geometry_contract<cutlass::half_t, 128, 64>();
  geometry_contract<cutlass::half_t, 64, 128>(); geometry_contract<cutlass::bfloat16_t, 64, 64>();
  geometry_contract<cutlass::bfloat16_t, 128, 64>(); geometry_contract<cutlass::bfloat16_t, 64, 128>();
  float large = 243383.484f;
  CHECK(std::isfinite(float(cutlass::bfloat16_t(large))));
  CHECK(!std::isfinite(float(cutlass::half_t(large))));
  std::printf("PPU17_HOST_CONTRACT PASS checks=%d formats=6 geometry=6 gpu_launches=0\n", checks);
}
