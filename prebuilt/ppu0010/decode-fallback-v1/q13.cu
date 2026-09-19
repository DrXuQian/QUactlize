#include "quactlize/execution/simt_q8_vector.cuh"
#include "quactlize/execution/simt_validation.hpp"
extern "C" int fallback_q13(qkg_simt_call_v2 const& d,qkg_simt_config_v1 const& f) {
  if(f.variant==3 && f.columns==4 && f.warps==2 && f.values==8) return quactlize::execution::simt::launch_v2<13,3,4,2,8>(d,f.split);
  if(f.variant==3 && f.columns==4 && f.warps==4 && f.values==4) return quactlize::execution::simt::launch_v2<13,3,4,4,4>(d,f.split);
  return QKG_INVALID;
}
