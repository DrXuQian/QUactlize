// The real public v3 -> v2 -> Q4 selector chain, with resource-only device
// doubles. No GPU math, loading, JIT or performance is simulated here.
#include "quactlize/dispatch/binding.cpp"
#include "quactlize/execution/q4_decode.cpp"
#include <cassert>
#include <iomanip>
#include <iostream>

extern "C" int qkg_q4_decode_launch(qkg_call_v1 const&,qkg_q4_decode_config_v1 const&) { return QKG_RUNTIME; }
extern "C" int qkg_q4_decode_launch_bf16(qkg_call_v1 const&,qkg_q4_decode_config_v1 const&) { return QKG_RUNTIME; }

int main() {
    int q,mode,n,k,e,top,ch,m,compute;
    while(std::cin>>q>>mode>>n>>k>>e>>top>>ch>>m>>compute) {
        assert(mode==QKG_INDEXED && compute==QK_COMPUTE_BF16);
        qkg_call_v1 c{1,sizeof(c)};
        c.qtype=q;c.mode=mode;c.n=n;c.k=k;c.experts=e;c.topk=top;c.channels=ch;c.rows=m*top;
        c.input_type=QKG_F32;c.a_row_stride=k;c.a_token_stride=int64_t(k)*ch;c.ids_stride=top;c.out_row_stride=n;
        qkg_simt_call_v2 typed{2,sizeof(typed),c,compute};
        auto arr=q==8?q8_kpack2::arrangement():q==12?ppu_arrangements::q4_kpack4_transpose_v1():
            ppu_arrangements::kquant_kpack_transpose_v1(q);
        auto matched=select_smallm(typed,arr);
        TcDecision fallback;
        Config config{};
        auto legacy=smallm::select(c);
        if(matched.status==QKS_OK && matched.tc) config=*matched.tc;
        else if(matched.status==QKS_MISS && legacy.row && !smallm::data::kChoices[legacy.row->choice].simt) {
            auto const& old=smallm::data::kChoices[legacy.row->choice].tc;
            qks_request_v1 req{1,sizeof(req),q,old.route,c.rows,n,k,e,m,arr.mapping_id};
            fallback=select_tc(req,false,compute,{&old,QKS_COMPUTE_INITIAL});
            assert(fallback);config=fallback.config();
        }
        Runtime runtime;runtime.device=0;runtime.cu=72;
        runtime.moe=std::make_shared<MoeExecution>();
        runtime.moe->select_compute=quactlize_kpack_q4_decode_select_v2;
        runtime.moe->q4_compute=quactlize_kpack_q4_decode_run_v2;
        if(config.symbol) {
            auto module=std::make_shared<Module>();module->compute_type=compute;
            module->query_compute=[](qk_compute_device_call_v4 const*,qk_recipe_v1 const*,qk_resources_v1* out) -> int {
                *out={};out->occupancy=3;return QK_OK;
            };
            Image image{config.symbol,std::string(64,'a'),std::string(64,'b'),q,config.route,
                config.tm,config.tn,config.tk,config.wm,config.wn,config.stages,config.ap,config.dn,{},false,compute};
            kImages={image};runtime.modules[image.key]=module;
        }
        qks_smallm_choice_v2 out{};
        int rc=quactlize_kpack_dispatch_query_smallm_v3(&runtime,&typed,&arr,&out);
        if(rc==QKS_MISS) {
            rc=quactlize_kpack_dispatch_query_smallm_v2(&runtime,&typed,&arr,&out.base);
            if(rc==QKS_OK && out.base.kind==QKS_SMALLM_SIMT && q==12) {
                qkg_sizes_v1 size{};
                int qrc=quactlize_kpack_q4_decode_select_v2(&typed,&arr,&out.q4,&size);
                assert(qrc==QKG_OK || qrc==QKG_SHAPE);
                if(qrc==QKG_OK) out.base.kind=QKS_SMALLM_Q4;
            }
        }
        std::cout<<"{\"status\":"<<rc<<",\"kind\":"<<out.base.kind;
        if(out.base.kind==QKS_SMALLM_TC)
            std::cout<<",\"parent\":"<<std::quoted(out.base.tc.parent)<<",\"split\":"<<out.base.tc.split;
        else if(out.base.kind==QKS_SMALLM_SIMT) {
            auto f=out.base.simt;
            std::cout<<",\"config\":["<<f.variant<<','<<f.columns<<','<<f.warps<<','<<f.values<<','<<f.split<<']';
        } else {
            auto f=out.q4;
            std::cout<<",\"config\":["<<f.reader<<','<<f.variant<<','<<f.columns<<','<<f.warps<<','<<f.values<<']';
        }
        std::cout<<"}\n";
    }
}
