#include "word_pack.hpp"
#include "api.h"
#include "../quactlize/fusion/paired_n4.hpp"
#include "../quactlize/fusion/validation.hpp"

template <gguf_scale::KType T, class Source>
int pack_host(Source raw, uint8_t* low, uint8_t* high, uint8_t* units,
              int n, int k, int experts) {
  using C = gguf_scale::CodeTraits<T>;
  using U = gguf_scale::packed_unit::Unit<T>;
  uint64_t const blocks = uint64_t(experts) * n * (k / 256);
  for (uint64_t i = 0; i < blocks * C::kLoBytes / 2; ++i) {
    uint16_t const word = quactlize_pack::Plane<T, false>::word(raw, n, k, i);
    low[2 * i] = uint8_t(word); low[2 * i + 1] = uint8_t(word >> 8);
  }
  if constexpr (C::kHiBytes != 0)
    for (uint64_t i = 0; i < blocks * C::kHiBytes / 2; ++i) {
      uint16_t const word = quactlize_pack::Plane<T, true>::word(raw, n, k, i);
      high[2 * i] = uint8_t(word); high[2 * i + 1] = uint8_t(word >> 8);
    }
  for (uint64_t i = 0; i < blocks / U::kSbPerUnit; ++i)
    quactlize_pack::metadata<T>(raw, units, n, k, i);
  return 0;
}

template<class Source>
int pack_all(int q, Source raw, uint8_t* low, uint8_t* high, uint8_t* units, int n, int k, int e) {
  if (q == 8) {
    uint64_t const count = uint64_t(e) * n * k;
    for (uint64_t i=0; i<count/2; ++i) {
      uint16_t w=quactlize_pack::Q8::word(raw,n,k,i);
      low[2*i]=uint8_t(w); low[2*i+1]=uint8_t(w>>8);
    }
    for (uint64_t i=0; i<count/32; ++i) quactlize_pack::Q8::metadata(raw,units,n,k,i);
    return 0;
  }
#define QP_CASE(Q, T) case Q: return pack_host<gguf_scale::KType::T>(raw, low, high, units, n, k, e)
  switch (q) {
    QP_CASE(10, Q2_K); QP_CASE(11, Q3_K); QP_CASE(12, Q4_K);
    QP_CASE(13, Q5_K); QP_CASE(14, Q6_K);
    default: return 22;
  }
#undef QP_CASE
}

extern "C" int host_pack(int q, uint8_t const* raw, uint8_t* low,
                         uint8_t* high, uint8_t* units, int n, int k, int e) {
  return pack_all(q,raw,low,high,units,n,k,e);
}
extern "C" int host_pack_pair(int q, uint8_t const* gate, uint8_t const* up,
    uint8_t* low, uint8_t* high, uint8_t* units, int n, int k, int e) {
  if (n <= 0 || n > INT32_MAX/2) return 24;
  quactlize_ppu_placed_arrangement_v2 a{};
  quactlize_ppu_kpack_sizes_v1 sizes{};
  int rc=quactlize_ppu_kpack_canonical_arrangement_v1(q,&a);
  if (rc) return rc;
  rc=quactlize_ppu_kpack_sizes_for_arrangement_v1(n,k,e,q,&a,&sizes);
  if (rc) return rc;
  return pack_all(q,quactlize_pack::PairedRows{gate,up,sizes.raw_bytes/uint64_t(e)},
                  low,high,units,2*n,k,e);
}

extern "C" int host_pack_paired_n4(int q,uint8_t const* gate,uint8_t const* up,
    uint8_t* low,uint8_t* high,uint8_t* units,int n,int k,int e) {
  qkg_gate_up_layout_v1 layout{};
  quactlize_ppu_kpack_sizes_v1 sizes{};
  int rc=quactlize_gate_up_layout_v1(q,&layout);
  if(rc) return rc;
  rc=quactlize_ppu_kpack_sizes_for_arrangement_v1(n,k,e,q,&layout.packing,&sizes);
  if(rc) return rc;
  quactlize::fusion::PairedRawRows raw{gate,up,sizes.raw_bytes/e/n,n};
  return pack_all(q,raw,low,high,units,2*n,k,e);
}

extern "C" int host_gate_up_query(qkg_gate_up_call_v1 const* d,qkg_gate_up_config_v1 const* f,
    qkg_gate_up_layout_v1 const* layout,qkg_sizes_v1* sizes) {
  qkg_sizes_v1 result{};
  int rc=quactlize::fusion::query(*d,*f,*layout,result);
  if(!rc) *sizes=result;
  return rc;
}

extern "C" int host_gate_up_buffers(qkg_gate_up_call_v1 const* d,qkg_sizes_v1 const* s) {
  return quactlize::fusion::buffers(*d,*s);
}
