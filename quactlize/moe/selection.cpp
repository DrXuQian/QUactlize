#include "selection.hpp"

extern "C" int quactlize_ppu_moe_prepare_select_v1(qzm_prepare_v1 const* c,int32_t implementation,qzm_plan_v1* out) {
    if(!c || !out) return QZR_INVALID;
    return quactlize::moe::prepare_plan(*c,implementation,*out);
}
extern "C" int quactlize_ppu_moe_finish_select_v1(qzm_finish_v1 const* c,int32_t implementation,qzm_plan_v1* out) {
    if(!c || !out) return QZR_INVALID;
    return quactlize::moe::finish_plan(*c,implementation,*out);
}
extern "C" int quactlize_ppu_moe_prepare_select_v2(qzm_prepare_v1 const* c,qzm_prepare_plan_v2* out) {
    if(!c || !out) return QZR_INVALID;
    return quactlize::moe::prepare_plan_v2(*c,*out);
}
