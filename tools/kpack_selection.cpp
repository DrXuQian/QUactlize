// Host-only final selection used by package planning and offline explain.
#include "quactlize/dispatch/selection.hpp"
#include "quactlize/execution/simt_strategy.hpp"
#include <iomanip>
#include <iostream>

using namespace quactlize::dispatch;

static void emit(int q,int mode,int n,int k,int experts,int topk,int channels,int tokens,int compute) {
    qkg_call_v1 c{1,sizeof(c)};
    c.qtype=q;c.mode=mode;c.n=n;c.k=k;c.experts=experts;c.topk=topk;c.channels=channels;
    int64_t rows=int64_t(tokens)*(mode==QKG_INDEXED?topk:1);
    c.rows=rows>0 && rows<=INT32_MAX ? int(rows) : 0;c.input_type=QKG_F32;
    c.a_row_stride=k;c.a_token_stride=int64_t(k)*channels;c.ids_stride=topk;c.out_row_stride=n;
    auto a=q==8?q8_kpack2::arrangement():q==12?ppu_arrangements::q4_kpack4_transpose_v1():
        q>=10 && q<=14?ppu_arrangements::kquant_kpack_transpose_v1(q):quactlize_ppu_placed_arrangement_v2{};
    auto d=select_smallm({2,sizeof(qkg_simt_call_v2),c,compute},a);
    auto const& b=d.choice.base;
    std::cout<<"{\"request\":["<<q<<','<<mode<<','<<n<<','<<k<<','<<experts<<','<<topk<<','<<channels<<','<<tokens<<','<<compute
        <<"],\"status\":"<<d.status;
    if(d.status==QKS_OK) {
        std::cout<<",\"kind\":"<<b.kind<<",\"policy\":"<<b.policy
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
                <<",\"hoist\":"<<(impl.hoist?"true":"false")<<",\"fixed\":"<<(impl.fixed?"true":"false")<<'}';
        } else {
            auto f=d.choice.q4;
            std::cout<<",\"config\":{\"reader\":"<<f.reader<<",\"variant\":"<<f.variant
                <<",\"columns\":"<<f.columns<<",\"warps\":"<<f.warps<<",\"values\":"<<f.values<<'}';
        }
    }
    std::cout<<"}\n";
}

int main(int argc,char** argv) {
    if(argc==2 && std::string(argv[1])=="--inventory") {
        for(auto const& r:matched::data::kExact)
            emit(r.q,r.mode,r.n,r.k,r.experts,r.topk,r.channels,r.tokens,r.compute);
        return 0;
    }
    if(argc!=1)return 2;
    int q,mode,n,k,e,top,ch,m,compute;
    while(std::cin>>q>>mode>>n>>k>>e>>top>>ch>>m>>compute)emit(q,mode,n,k,e,top,ch,m,compute);
    return std::cin.eof()?0:1;
}
