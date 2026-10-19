#include "dev/gemv_simt/general_locate.hpp"
#include "quactlize/execution/simt_validation.hpp"
#include <cstdio>
#include <cstdlib>

void require(bool b) { if(!b) std::abort(); }
template<int Mode> void check(qkg_call_v1 c,unsigned row) {
    c.mode=Mode;
    auto a=quactlize::execution::simt::general_locate<Mode>(c,row);
    auto b=quactlize::execution::q4_s1::locate(c,row);
    require(a.expert==b.expert && a.a==b.a && a.output==b.output);
}
int main() {
    int ids[96];for(int i=0;i<96;++i)ids[i]=i%16;
    ids[0]=-1;ids[13]=17;
    int offsets[]={0,0,1,1,1,2,7,7,8,16,17,20,20,32,32,40,64};
    for(int ch:{1,2,4,8})for(int64_t stride:{int64_t(512),int64_t(1)<<32}) {
        qkg_call_v1 c{};c.rows=64;c.n=512;c.k=512;c.experts=16;c.topk=8;c.channels=ch;
        c.a_row_stride=stride;c.a_token_stride=ch*stride;c.out_row_stride=stride;
        c.ids=ids;c.ids_stride=12;c.offsets=offsets;
        for(unsigned row=0;row<64;++row) {
            check<QKG_DENSE>(c,row);check<QKG_INDEXED>(c,row);check<QKG_GROUPED>(c,row);
        }
    }
    // Public grouped ABI has topk/channels=1 after pre-grouping. Do not
    // accidentally pass the fixture's original indexed topk into this ABI.
    for(int mode=0;mode<3;++mode) {
        qkg_call_v1 c{1,sizeof(c)};
        c.qtype=8;c.n=512;c.k=512;c.experts=mode?16:1;c.rows=mode?64:8;
        c.mode=mode;c.input_type=QKG_F32;c.channels=mode==QKG_INDEXED?8:1;
        c.topk=mode==QKG_INDEXED?8:1;c.a_row_stride=512;c.a_token_stride=4096;
        c.out_row_stride=512;c.ids_stride=8;
        qkg_simt_call_v2 d{2,sizeof(d),c,QKG_COMPUTE_BF16};
        qkg_simt_config_v1 cfg{1,sizeof(cfg),5,8,4,4,1};qkg_sizes_v1 sizes{};
        auto arr=q8_kpack2::arrangement();
        require(quactlize::execution::simt::query_v2(d,cfg,&arr,sizes)==0);
    }
    std::puts("GENERAL_LOCATE_HOST PASS dense indexed grouped wide-strides invalid-ids empty-experts query");
}
