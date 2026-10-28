// Host-only final selection used by package planning and offline explain.
#include "quactlize/dispatch/selection.hpp"
#include "quactlize/execution/simt_strategy.hpp"
#include "policies/kpack_prefill_confirmation_v1.hpp"
#include <iomanip>
#include <iostream>

using namespace quactlize::dispatch;

static void emit(int q,int mode,int n,int k,int experts,int topk,int channels,int tokens,int compute,
                 bool dense_fallbacks=false,bool indexed_fallbacks=false,
                 qkg_simt_config_v1 const* explicit_simt=nullptr) {
    qkg_call_v1 c{1,sizeof(c)};
    c.qtype=q;c.mode=mode;c.n=n;c.k=k;c.experts=experts;c.topk=topk;c.channels=channels;
    int64_t rows=int64_t(tokens)*(mode==QKG_INDEXED?topk:1);
    c.rows=rows>0 && rows<=INT32_MAX ? int(rows) : 0;c.input_type=QKG_F32;
    c.a_row_stride=k;c.a_token_stride=int64_t(k)*channels;c.ids_stride=topk;c.out_row_stride=n;
    auto a=q==8?q8_kpack2::arrangement():q==12?ppu_arrangements::q4_kpack4_transpose_v1():
        q>=10 && q<=14?ppu_arrangements::kquant_kpack_transpose_v1(q):quactlize_ppu_placed_arrangement_v2{};
    auto d=select_smallm({2,sizeof(qkg_simt_call_v2),c,compute},a);
    // Offline control query only. Mirror the public dense F16 caller's
    // table-miss chain; a matched MISS is not the executed production recipe.
    TcDecision fallback;
    Config fallback_config{};
    bool native_f16=false;
    // Offline replay of the POC caller's BF16 auto chain: v3 -> v2 ->
    // typed Q4 selector. Keep the actual generic/legacy control on new E/top-k
    // families; never pretend a v3 table MISS is the executed recipe.
    if(indexed_fallbacks && d.status==QKS_MISS && mode==QKG_INDEXED &&
            compute==QK_COMPUTE_BF16 && tokens>=1 && tokens<=8) {
        auto& b=d.choice.base;
        b.kind=QKS_SMALLM_SIMT;b.policy=QKS_COMPUTE_INITIAL;
        b.source_n=n;b.source_k=k;b.source_tokens=tokens;
        b.simt={1,sizeof(b.simt),0,4,4,4,1};
        auto legacy=smallm::select(c);
        if(legacy.row) {
            auto const& row=*legacy.row;
            auto const& choice=smallm::data::kChoices[row.choice];
            b.source_n=row.n;b.source_k=row.k;b.source_tokens=row.tokens;
            if(choice.simt) {
                auto f=choice.reader;
                b.simt={1,sizeof(b.simt),f.variant,f.columns,f.warps,f.values,f.split};
            } else {
                qks_request_v1 r{1,sizeof(r),q,choice.tc.route,c.rows,n,k,experts,tokens,a.mapping_id};
                fallback=select_tc(r,false,compute,{&choice.tc,QKS_COMPUTE_INITIAL});
                if(!fallback) {std::cerr<<"indexed TC control has no host proposal\n"; std::exit(1);}
                fallback_config=fallback.config();d.tc=&fallback_config;b.kind=QKS_SMALLM_TC;
            }
        }
        d.status=quactlize::execution::simt::query_v2({2,sizeof(qkg_simt_call_v2),c,compute},
            b.simt,&a,b.sizes)==QKG_OK?QKS_OK:QKS_INVALID;
        if(d.status==QKS_OK && b.kind==QKS_SMALLM_SIMT && q==12 && experts==256 && topk==8 &&
                (channels==1 || channels==8)) {
            auto board=quactlize::decode_policy::select(mode,n,k,tokens);
            if(board && board->simt) {
                auto f=board->config;b.kind=QKS_SMALLM_Q4;
                d.choice.q4={1,sizeof(d.choice.q4),f.reader,f.variant,f.warps,f.values,f.columns};
            }
        }
    }
    if(explicit_simt) {
        auto& b=d.choice.base;b.kind=QKS_SMALLM_SIMT;b.simt=*explicit_simt;
        d.status=quactlize::execution::simt::query_v2({2,sizeof(qkg_simt_call_v2),c,compute},
            b.simt,&a,b.sizes)==QKG_OK?QKS_OK:QKS_INVALID;
    }
    if(dense_fallbacks && d.status==QKS_MISS && mode==QKG_DENSE && compute==QK_COMPUTE_F16 &&
            tokens>=1 && tokens<=8 && experts==1 && channels==1 && topk==1) {
        auto& b=d.choice.base;
        if(q==12) {
            auto board=quactlize::decode_policy::select(mode,n,k,tokens);
            if(board && board->simt) {
                auto f=board->config;
                b.kind=QKS_SMALLM_Q4;b.policy=QKS_DECODE_MEASURED;
                d.choice.q4={1,sizeof(d.choice.q4),f.reader,f.variant,f.warps,f.values,f.columns};
                d.status=QKS_OK;
            }
        } else {
            auto legacy=smallm::select(c);
            if(legacy.row) {
                auto const& choice=smallm::data::kChoices[legacy.row->choice];
                b.policy=legacy.policy;b.source_n=legacy.row->n;b.source_k=legacy.row->k;
                b.source_tokens=legacy.row->tokens;
                if(choice.simt) {
                    auto f=choice.reader;b.kind=QKS_SMALLM_SIMT;
                    b.simt={1,sizeof(b.simt),f.variant,f.columns,f.warps,f.values,f.split};
                    d.status=quactlize::execution::simt::query(c,b.simt,&a,b.sizes)==QKG_OK?QKS_OK:QKS_MISS;
                } else {
                    qks_request_v1 r{1,sizeof(r),q,choice.tc.route,tokens,n,k,1,tokens,a.mapping_id};
                    fallback=select_tc(r,false,compute,{&choice.tc,legacy.policy});
                }
            }
        }
        if(d.status!=QKS_OK && !fallback) {
            qks_request_v1 r{1,sizeof(r),q,q==8?QK_DENSE_SF:QK_DENSE_FQ,tokens,n,k,1,tokens,a.mapping_id};
            if(q==12) fallback=select_tc(r,true,compute);
            if(!fallback) fallback=select_dense_dynamic(r,compute);
            native_f16=fallback.policy==QKS_LEGACY_DENSE;
        }
        if(fallback) {
            fallback_config=fallback.config();d.tc=&fallback_config;
            b.kind=QKS_SMALLM_TC;b.policy=fallback.policy;d.status=QKS_OK;
        }
    }
    auto const& b=d.choice.base;
    std::cout<<"{\"request\":["<<q<<','<<mode<<','<<n<<','<<k<<','<<experts<<','<<topk<<','<<channels<<','<<tokens<<','<<compute
        <<"],\"status\":"<<d.status;
    if(dense_fallbacks)std::cout<<",\"control_scope\":\"PUBLIC_DENSE_F16_HOST_CHAIN\",\"native_f16\":"<<(native_f16?"true":"false");
    if(indexed_fallbacks)std::cout<<",\"control_scope\":\"POC_INDEXED_BF16_AUTO_HOST_CHAIN\"";
    if(d.status==QKS_OK) {
        std::cout<<",\"kind\":"<<b.kind<<",\"policy\":"<<b.policy
            <<",\"dequant_config\":"<<d.dequant_config
            <<",\"donor\":["<<b.source_n<<','<<b.source_k<<','<<b.source_tokens<<']'
            <<",\"workspace_bytes\":"<<b.sizes.workspace_bytes;
        if(b.kind==QKS_SMALLM_TC) {
            auto const& f=*d.tc;
            std::cout<<",\"parent\":{\"symbol\":"<<std::quoted(f.symbol)
                <<",\"qtype\":"<<f.qtype<<",\"route\":"<<f.route<<",\"tm\":"<<f.tm
                <<",\"tn\":"<<f.tn<<",\"tk\":"<<f.tk<<",\"wm\":"<<f.wm<<",\"wn\":"<<f.wn
                <<",\"stages\":"<<f.stages<<",\"ap\":"<<f.ap<<",\"dn\":"<<f.dn
                <<",\"persistent\":"<<f.parent_persistent<<"},\"split\":"<<f.split
                <<",\"grid_mode\":"<<f.grid_mode<<",\"grid_b\":"<<f.grid_b;
        } else if(b.kind==QKS_SMALLM_SIMT) {
            auto f=b.simt;
            std::cout<<",\"config\":{\"variant\":"<<f.variant<<",\"columns\":"<<f.columns
                <<",\"warps\":"<<f.warps<<",\"values\":"<<f.values<<",\"split\":"<<f.split<<'}';
            auto impl=quactlize::execution::simt::implementation({2,sizeof(qkg_simt_call_v2),c,compute},f);
            std::cout<<",\"implementation\":{\"producer\":"<<std::quoted(impl.producer)
                <<",\"measured\":"<<std::quoted(impl.measured)
                <<",\"reduction\":"<<std::quoted(impl.reduction)<<",\"changes\":"<<impl.changes
                <<",\"hoist\":"<<(impl.hoist?"true":"false")<<",\"fixed\":"<<(impl.fixed?"true":"false");
            if(impl.q8_body)std::cout<<",\"q8_body\":"<<impl.q8_body;
            if(impl.cta_tokens)std::cout<<",\"cta_tokens\":"<<impl.cta_tokens<<",\"cta_window\":"<<impl.cta_window;
            std::cout<<'}';
        } else {
            auto f=d.choice.q4;
            std::cout<<",\"config\":{\"reader\":"<<f.reader<<",\"variant\":"<<f.variant
                <<",\"columns\":"<<f.columns<<",\"warps\":"<<f.warps<<",\"values\":"<<f.values<<'}';
        }
    }
    std::cout<<"}\n";
}

// Historical inventory is an explicit offline replay fixture, never a runtime
// policy switch. Live callers and package planning always use the active table.
template<class Knot,size_t NK,class Config,size_t NC>
static void emit_prefill(Knot const (&knots)[NK],Config const (&configs)[NC]) {
        for(auto const& k:knots) {
            qks_request_v1 r{1,sizeof(r),k.q,k.experts==1?0:2,k.tokens*(k.experts==1?1:8),k.n,k.k,k.experts,k.tokens,
                k.q==12?UINT64_C(0x51344b5034540001):UINT64_C(0x514b504b54000001)};
            qks_prefill_choice_v2 out{};
            int rc=prefill::query_in(r,7,k.compute,k.channels,out,knots);
            std::cout<<"{\"request\":["<<k.q<<','<<(k.experts==1?0:2)<<','<<k.n<<','<<k.k<<','<<k.experts<<','
                <<(k.experts==1?1:8)<<','<<k.channels<<','<<k.tokens<<','<<k.compute<<"],\"status\":"<<rc
                <<",\"route\":"<<k.route<<",\"policy\":"<<out.policy<<",\"dequant_config\":"<<k.dequant;
            if(k.config>=0) {
                auto const& c=configs[k.config];
                std::cout<<",\"kind\":0,\"parent\":{\"symbol\":"<<std::quoted(c.symbol)
                    <<",\"qtype\":"<<c.qtype<<",\"route\":"<<c.route<<",\"tm\":"<<c.tm<<",\"tn\":"<<c.tn
                    <<",\"tk\":"<<c.tk<<",\"wm\":"<<c.wm<<",\"wn\":"<<c.wn<<",\"stages\":"<<c.stages
                    <<",\"ap\":"<<c.ap<<",\"dn\":"<<c.dn<<",\"persistent\":"<<c.parent_persistent
                    <<"},\"split\":"<<c.split<<",\"grid_mode\":"<<c.grid_mode<<",\"grid_b\":"<<c.grid_b;
            } else std::cout<<",\"kind\":3";
            std::cout<<"}\n";
        }
}

int main(int argc,char** argv) {
    if(argc==2 && std::string(argv[1])=="--prefill-inventory") {
        emit_prefill(prefill::data::kKnots,prefill::data::kConfigs);
        return 0;
    }
    if(argc==2 && std::string(argv[1])=="--historical-prefill-inventory") {
        emit_prefill(quactlize_prefill_confirmation_v1::kKnots,quactlize_prefill_confirmation_v1::kConfigs);
        return 0;
    }
    if(argc==2 && std::string(argv[1])=="--inventory") {
        for(auto const& r:matched::data::kExact)
            emit(r.q,r.mode,r.n,r.k,r.experts,r.topk,r.channels,r.tokens,r.compute);
        return 0;
    }
    bool dense_fallbacks=argc==2 && std::string(argv[1])=="--dense-fallbacks";
    bool indexed_fallbacks=argc==2 && std::string(argv[1])=="--indexed-bf16-fallbacks";
    bool explicit_simt=argc==2 && std::string(argv[1])=="--simt-configs";
    if(argc!=1 && !dense_fallbacks && !indexed_fallbacks && !explicit_simt)return 2;
    int q,mode,n,k,e,top,ch,m,compute;
    while(std::cin>>q>>mode>>n>>k>>e>>top>>ch>>m>>compute) {
        qkg_simt_config_v1 f{1,sizeof(f)};
        if(explicit_simt && !(std::cin>>f.variant>>f.columns>>f.warps>>f.values>>f.split))return 1;
        emit(q,mode,n,k,e,top,ch,m,compute,dense_fallbacks,indexed_fallbacks,explicit_simt?&f:nullptr);
    }
    return std::cin.eof()?0:1;
}
