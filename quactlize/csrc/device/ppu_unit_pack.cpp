// Host-only forward producer for the packed metadata channel.  Kept separate from ppu_dense_layout.cu so callers
// which need only units do not inherit xplane templates, and so this ABI can be tested with an ordinary host build.
#include "gguf_unit_pack.hpp"
#include "quactlize_ppu_packed.h"

namespace {

using gguf_scale::KType;

template <KType T>
int prepare(uint8_t const* blocks, uint8_t* units, int n, int k, int experts) {
  using U = gguf_scale::packed_unit::Unit<T>;
  if ((k / 256) % U::kSbPerUnit) return 24;
  gguf_scale::unit_pack::pack<T>(blocks, units, n, k, experts);
  return 0;
}

int prepare_dispatch(uint8_t const* blocks, uint8_t* units, int n, int k, int experts, int qtype) {
  if (!blocks || !units || experts <= 0 || n <= 0 || n % 256 || k <= 0 || k % 256) return 20;
  switch (qtype) {
    case 10: return prepare<KType::Q2_K>(blocks, units, n, k, experts);
    case 11: return prepare<KType::Q3_K>(blocks, units, n, k, experts);
    case 12: return prepare<KType::Q4_K>(blocks, units, n, k, experts);
    case 13: return prepare<KType::Q5_K>(blocks, units, n, k, experts);
    case 14: return prepare<KType::Q6_K>(blocks, units, n, k, experts);
    default: return 22;
  }
}

}  // namespace

extern "C" int64_t quactlize_ppu_units_bytes(int n, int k, int qtype) {
  switch (qtype) {
    case 10: return gguf_scale::unit_pack::bytes<KType::Q2_K>(n, k);
    case 11: return gguf_scale::unit_pack::bytes<KType::Q3_K>(n, k);
    case 12: return gguf_scale::unit_pack::bytes<KType::Q4_K>(n, k);
    case 13: return gguf_scale::unit_pack::bytes<KType::Q5_K>(n, k);
    case 14: return gguf_scale::unit_pack::bytes<KType::Q6_K>(n, k);
    default: return -1;
  }
}

extern "C" int quactlize_ppu_prepare_units(
    uint8_t const* blocks, uint8_t* units, int n, int k, int qtype) {
  return prepare_dispatch(blocks, units, n, k, 1, qtype);
}

extern "C" int quactlize_ppu_prepare_units_grouped(
    uint8_t const* blocks, uint8_t* units, int n, int k, int experts, int qtype) {
  return prepare_dispatch(blocks, units, n, k, experts, qtype);
}
