// Calls the public dispatcher. Only DSO resources/Q4 binding are stubbed;
// no GPU arithmetic or timing is simulated. Compile against either git tree.
#include "quactlize/dispatch/binding.cpp"
#include <iostream>

static quactlize::dispatch::matched::data::Choice const* current = nullptr;
static int resources(qk_resources_v1* out) {
    *out={};out->occupancy=3;out->workspace_bytes=8192;out->shared_bytes=1234;
    return QK_OK;
}
static int q4(qkg_call_v1 const*,quactlize_ppu_placed_arrangement_v2 const*,
              qkg_q4_decode_config_v1* out,qkg_sizes_v1* sizes) {
    if(!current || current->kind!=QKS_SMALLM_Q4)return QKG_INVALID;
    auto r=current->reader;*out={1,sizeof(*out),r.reader,r.variant,r.warps,r.values,r.columns};
    *sizes={};sizes->low_bytes=1024;return QKG_OK;
}
int main() {
    using namespace quactlize::dispatch;
    int q,mode,n,k,e,top,ch,t,compute,mutation;
    while(std::cin>>q>>mode>>n>>k>>e>>top>>ch>>t>>compute>>mutation) {
        qkg_call_v1 c{1,sizeof(c)};c.qtype=q;c.mode=mode;c.n=n;c.k=k;c.experts=e;
        c.topk=top;c.channels=ch;c.rows=t*(mode==QKG_INDEXED?top:1);c.input_type=QKG_F32;
        c.a_row_stride=k;c.a_token_stride=int64_t(k)*ch;c.ids_stride=top;c.out_row_stride=n;
        qkg_simt_call_v2 d{2,sizeof(d),c,compute};
        auto a=q==8?q8_kpack2::arrangement():q==12?ppu_arrangements::q4_kpack4_transpose_v1():
            q>=10 && q<=14?ppu_arrangements::kquant_kpack_transpose_v1(q):quactlize_ppu_placed_arrangement_v2{};
        if(mutation==1)d.version=99;
        if(mutation==2)a.mapping_id^=1;
        if(mutation==3)d.call.a_row_stride=k+1;
        if(mutation==4)d.compute_type=99;
        if(mutation==5)d.call.a=reinterpret_cast<void*>(4);
        auto selected=matched::select(d);
        current=selected.row?&matched::data::kChoices[selected.row->choice]:nullptr;
        Runtime runtime;runtime.device=0;runtime.cu=72;kImages.clear();
        if(current && current->kind==QKS_SMALLM_TC) {
            auto const& f=current->tc;bool dense=mode==QKG_DENSE;
            auto module=std::make_shared<Module>();module->compute_type=compute;
            module->query_dense_io=[](qkd_dense_call_v1 const*,qk_recipe_v1 const*,qk_resources_v1* o){return resources(o);};
            module->query_dense_compute=[](qkd_dense_call_v2 const*,qk_recipe_v1 const*,qk_resources_v1* o){return resources(o);};
            module->query_device=[](qk_device_call_v2 const*,qk_recipe_v1 const*,qk_resources_v1* o){return resources(o);};
            module->query_compute=[](qk_compute_device_call_v4 const*,qk_recipe_v1 const*,qk_resources_v1* o){return resources(o);};
            Image image{f.symbol,std::string(64,'a'),std::string(64,'b'),f.qtype,f.route,
                f.tm,f.tn,f.tk,f.wm,f.wn,f.stages,f.ap,f.dn,{},dense,compute};
            kImages={image};runtime.modules[image.key]=module;
        }
        runtime.moe=std::make_shared<MoeExecution>();
        runtime.moe->select=q4;
        runtime.moe->select_compute=[](qkg_simt_call_v2 const* d,quactlize_ppu_placed_arrangement_v2 const* a,
            qkg_q4_decode_config_v1* f,qkg_sizes_v1* s){return q4(&d->call,a,f,s);};
        runtime.moe->q4_compute=[](qkg_simt_call_v2 const*,qkg_q4_decode_config_v1 const*,
            quactlize_ppu_placed_arrangement_v2 const*)->int{return QKG_OK;};
        qks_smallm_choice_v2 out{};
        int rc=quactlize_kpack_dispatch_query_smallm_v3(&runtime,&d,&a,&out);
        auto const& b=out.base;auto const& f=b.simt;auto const& x=out.q4;auto const& z=b.tc;
        std::cout<<rc<<' '<<out.version<<' '<<out.size<<' '<<out.compute_type
            <<' '<<b.version<<' '<<b.size<<' '<<b.kind<<' '<<b.policy<<' '<<b.source_n<<' '<<b.source_k<<' '<<b.source_tokens
            <<' '<<f.version<<' '<<f.size<<' '<<f.variant<<' '<<f.columns<<' '<<f.warps<<' '<<f.values<<' '<<f.split
            <<' '<<b.sizes.low_bytes<<' '<<b.sizes.high_bytes<<' '<<b.sizes.units_bytes<<' '<<b.sizes.sf_plane_bytes<<' '<<b.sizes.workspace_bytes
            <<' '<<x.version<<' '<<x.size<<' '<<x.reader<<' '<<x.variant<<' '<<x.warps<<' '<<x.values<<' '<<x.columns
            <<' '<<z.version<<' '<<z.size<<' '<<z.ticket<<' '<<z.workspace_bytes<<' '<<z.shared_bytes
            <<' '<<z.policy<<' '<<z.algorithm<<' '<<z.split<<' '<<z.grid<<' '<<z.device<<' '<<z.compute_units
            <<' '<<z.parent<<' '<<z.build_key<<' '<<runtime.plans.size()<<'\n';
    }
    return std::cin.eof()?0:1;
}
