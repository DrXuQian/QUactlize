// Real exported selector/materializer with the UPLOADED resource-query values.
// No DSO/GPU execution: the resource callback does not claim live occupancy.
#include "quactlize/dispatch/binding.cpp"
#include <cassert>
#include <iomanip>
#include <iostream>

static qk_resources_v1 observed{};
static int resources(qk_call_v1 const*, qk_recipe_v1 const*, qk_resources_v1* out) {
    *out=observed;return QK_OK;
}

int main() {
    using namespace quactlize::dispatch;
    int q,mode,n,k,e,top,ch,m,compute,occupancy;
    uint64_t workspace,shared;
    while(std::cin>>q>>mode>>n>>k>>e>>top>>ch>>m>>compute>>occupancy>>workspace>>shared) {
        qks_request_v1 req{1,sizeof(req),q,mode,m*top,n,k,e,m,
            q==12?UINT64_C(0x51344b5034540001):UINT64_C(0x514b504b54000001)};
        qks_prefill_choice_v2 pure{};
        assert(!quactlize_kpack_dispatch_prefill_v2(&req,7,compute,ch,&pure));
        auto knot=prefill::knot(req,compute,ch);assert(knot);
        Runtime runtime;runtime.device=0;runtime.cu=72;kImages.clear();
        qks_choice_v1 out{};
        if(pure.base.route!=2) {
            assert(knot->config>=0);
            auto const& c=prefill::data::kConfigs[knot->config];req.route=c.route;
            Image image{c.symbol,std::string(64,'a'),std::string(64,'b'),c.qtype,c.route,
                c.tm,c.tn,c.tk,c.wm,c.wn,c.stages,c.ap,c.dn,{},false,compute};
            auto mod=std::make_shared<Module>();mod->compute_type=compute;
            mod->query=resources;
            mod->query_device=[](auto d,auto r,auto o){return resources(&d->call,r,o);};
            mod->query_compute=[](auto d,auto r,auto o){
                assert(d->compute_type==QK_COMPUTE_BF16 && d->metadata_type==QK_METADATA_BF16);
                return resources(&d->device_call.call,r,o);
            };
            kImages={image};runtime.modules[image.key]=mod;
            observed={1,sizeof(observed),workspace,shared,occupancy,0,0};
            assert(!quactlize_kpack_dispatch_query_prefill_v1(&runtime,&req,compute,ch,&out));
            assert(out.ticket && out.policy==19);
            auto const& p=runtime.plans.at(out.ticket-1);
            assert(p.recipe.split==out.split && p.recipe.grid==out.grid && p.compute_type==compute);
            auto ticket=out.ticket;
            assert(!quactlize_kpack_dispatch_query_prefill_v1(&runtime,&req,compute,ch,&out) && out.ticket==ticket);
        } else {
            assert(quactlize_kpack_dispatch_query_prefill_v1(&runtime,&req,compute,ch,&out)==QKS_MISS);
            assert(!out.ticket);
        }
        std::cout<<"{\"request\":["<<q<<','<<mode<<','<<n<<','<<k<<','<<e<<','<<top<<','<<ch<<','<<m<<','<<compute
            <<"],\"route\":"<<pure.base.route<<",\"dequant\":"<<pure.base.dequant_config
            <<",\"parent\":"<<std::quoted(out.parent)<<",\"algorithm\":"<<out.algorithm
            <<",\"split\":"<<out.split<<",\"grid\":"<<out.grid<<",\"workspace\":"<<out.workspace_bytes
            <<",\"shared\":"<<out.shared_bytes<<"}\n";
    }
    return std::cin.eof()?0:1;
}
