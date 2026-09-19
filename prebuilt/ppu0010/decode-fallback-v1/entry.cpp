#include "quactlize/execution/simt_validation.hpp"
extern "C" int fallback_q8(qkg_simt_call_v2 const&,qkg_simt_config_v1 const&);
extern "C" int fallback_q13(qkg_simt_call_v2 const&,qkg_simt_config_v1 const&);
extern "C" int fallback_q10(qkg_simt_call_v2 const&,qkg_simt_config_v1 const&);
extern "C" int fallback_q11(qkg_simt_call_v2 const&,qkg_simt_config_v1 const&);
extern "C" int fallback_q12(qkg_simt_call_v2 const&,qkg_simt_config_v1 const&);
extern "C" int fallback_q14(qkg_simt_call_v2 const&,qkg_simt_config_v1 const&);
extern "C" int fallback_run(qkg_simt_call_v2 const* d,qkg_simt_config_v1 const* f,
quactlize_ppu_placed_arrangement_v2 const* a) {
  if(!d || !f || !a) return QKG_INVALID;
  qkg_sizes_v1 sizes{};
  int rc=quactlize::execution::simt::query_v2(*d,*f,a,sizes);if(rc)return rc;
  rc=quactlize::execution::simt::buffers_v2(*d,sizes);if(rc)return rc;
  switch(d->call.qtype) {
    case 8: return fallback_q8(*d,*f);
    case 13: return fallback_q13(*d,*f);
    case 10: return fallback_q10(*d,*f);
    case 11: return fallback_q11(*d,*f);
    case 12: return fallback_q12(*d,*f);
    case 14: return fallback_q14(*d,*f);
    default:return QKG_INVALID;
  }
}
