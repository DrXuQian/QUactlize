#include "quactlize/execution/simt_q8_vector.cuh"
#include "quactlize/execution/simt_validation.hpp"
extern "C" int fallback_q8(qkg_simt_call_v2 const& d,qkg_simt_config_v1 const& f) {
  if(f.variant==5 && f.columns==4 && f.warps==2 && f.values==4) return quactlize::execution::simt::q8_vector::launch_v2<8,5,4,2,4>(d,f.split);
  if(f.variant==5 && f.columns==4 && f.warps==8 && f.values==4) return quactlize::execution::simt::q8_vector::launch_v2<8,5,4,8,4>(d,f.split);
  if(f.variant==5 && f.columns==8 && f.warps==4 && f.values==4) return quactlize::execution::simt::q8_vector::launch_v2<8,5,8,4,4>(d,f.split);
  return QKG_INVALID;
}
