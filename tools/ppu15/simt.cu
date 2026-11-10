#include "simt_common.cuh"
#include "simt_validation.hpp"

// Existing production bodies, selected explicitly for a bounded numeric
// probe. Keep production argument/format/buffer validation; no selector is used.
extern "C" int qz15_simt(qkg_simt_call_v2 const* c,int split,
                         quactlize_ppu_placed_arrangement_v2 const* arr) {
    if(!c || (c->call.qtype!=8 && c->call.qtype!=14) ||
       (split!=1 && split!=2 && split!=4)) return QKG_INVALID;
    qkg_simt_config_v1 config{1,sizeof(config),c->call.qtype==14?3:1,
        4,c->call.qtype==14?2:4,c->call.qtype==14?8:4,split};
    qkg_sizes_v1 sizes{};
    int rc=quactlize::execution::simt::query_v2(*c,config,arr,sizes);
    if(rc) return rc;
    rc=quactlize::execution::simt::buffers_v2(*c,sizes);
    if(rc) return rc;
    if(c->call.qtype==14)
        return quactlize::execution::simt::common_reader_launch<14,4,2,8>(*c,split);
    return quactlize::execution::simt::launch_v2<8,1,4,4,4>(*c,split);
}
