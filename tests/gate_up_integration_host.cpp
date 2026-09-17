#include "quactlize/fusion/validation.hpp"
#include "quactlize/fusion/llama_graph.hpp"
#include <cassert>
#include <cstdio>

static void graph_case(int tokens,int fault) {
    auto * ctx=ggml_init({8*1024*1024,nullptr,true});
    assert(ctx);
    auto * a=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,2048,tokens);
    auto * gate=ggml_mul_mat(ctx,ggml_new_tensor_2d(ctx,fault==1?GGML_TYPE_Q4_K:GGML_TYPE_Q8_0,2048,512),a);
    auto * up=ggml_mul_mat(ctx,ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,2048,512),fault==2?ggml_dup_tensor(ctx,a):a);
    auto * out=fault==3?ggml_geglu_split(ctx,gate,up):ggml_swiglu_split(ctx,gate,up);
    auto * g=ggml_new_graph(ctx);ggml_build_forward_expand(g,out);
    if(fault==4) ggml_build_forward_expand(g,ggml_scale(ctx,gate,2.f));
    bool found=false;
    for(int i=0;i<g->n_nodes;++i) found|=quactlize::llama::match_shared_gate_up(g,i).count==3;
    assert(found==(tokens<=8 && fault==0));
    ggml_free(ctx);
}

int main() {
    using namespace quactlize::fusion;
    for(int tokens=1;tokens<=9;++tokens) for(int fault=0;fault<5;++fault) graph_case(tokens,fault);
    for(int q:{8,12}) for(int tokens=1;tokens<=8;++tokens) {
        qkg_gate_up_config_v1 f{};
        int e=q==8?1:256,compute=q==8?0:1;
        assert(select(q,512,2048,e,tokens,compute,&f)==0);
        if(q==8) assert(f.backend==0 && f.split==1 && f.warps==8);
        else if(tokens<=2 || tokens==4) assert(f.backend==0 && f.split==1 && f.warps==(tokens==2?4:8));
        else assert(f.backend==1 && f.split==(tokens==3?2:1) && f.tile_m==(tokens<=6?16:8));
        for(int bad:{0,9}) assert(select(q,512,2048,e,bad,compute,&f)!=0 && f.version==0);
        assert(select(q,1024,2048,e,tokens,compute,&f)!=0);
        assert(select(q,512,2048,e,tokens,1-compute,&f)!=0);
    }
    qkg_gate_up_call_v2 d{};d.version=2;d.size=sizeof(d);
    auto& c=d.call.input.call;c.rows=64;c.n=512;c.mode=QKG_INDEXED;c.out_row_stride=512;
    c.output=(float*)0x10000000;c.workspace=(void*)0x20000000;d.call.output_type=QKG_F32;
    qkg_sizes_v1 sizes{};sizes.workspace_bytes=64*1024*2*4;
    d.input_rows=(int*)0x30000000;d.status=(int*)0x30010000;
    assert(row_buffers(d,sizes)==0);
    d.input_rows=(int*)c.output;assert(row_buffers(d,sizes)!=0);
    d.input_rows=nullptr;d.status=(int*)c.workspace;assert(row_buffers(d,sizes)!=0);
    d.status=(int*)0x30000001;assert(row_buffers(d,sizes)!=0);
    d.status=nullptr;assert(row_buffers(d,sizes)==0);
    c.mode=QKG_DENSE;d.input_rows=(int*)0x30000000;assert(row_buffers(d,sizes)!=0);
    std::puts("GATE_UP_INTEGRATION_HOST PASS graph=45 selection=16 negative_arms=covered");
}
