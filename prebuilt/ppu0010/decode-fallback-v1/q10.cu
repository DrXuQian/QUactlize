#include "quactlize/execution/simt_q8_vector.cuh"
#include "quactlize/execution/simt_validation.hpp"
extern "C" int fallback_q10(qkg_simt_call_v2 const& d,qkg_simt_config_v1 const& f) {
  if(f.variant==3 && f.columns==4 && f.warps==4 && f.values==4) return quactlize::execution::simt::launch_v2<10,3,4,4,4>(d,f.split);
  return QKG_INVALID;
}
