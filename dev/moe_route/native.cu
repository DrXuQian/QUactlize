// Test-only wrapper around the linked llama kernel. No copied device body.
#include "common.cuh"
#include "mmid.cuh"
#include "quactlize/routing/layout.hpp"

extern "C" int qzr_native_run_v1(qzr_call_v1 const* c,int execute) {
    if(!c) return QZR_INVALID;
    qzr_config_v1 config{1,sizeof(config),QZR_TILED,512};
    quactlize::routing::Layout layout;
    int rc=quactlize::routing::layout(*c,config,layout);
    if(rc) return rc;
    int device=ggml_cuda_get_device();
    auto const& properties=ggml_cuda_info().devices[device];
    if(properties.warp_size!=32 || c->tokens>=(1<<22) ||
       size_t(c->tokens)*4>properties.smpbo || c->ids_stride>INT_MAX ||
       c->source_stride>INT_MAX || (c->tokens &&
       int64_t(c->tokens-1)*c->ids_stride+c->topk>INT_MAX)) return QZR_CAPACITY;
    if(!execute) return QZR_OK;
    layout.bytes=0;
    rc=quactlize::routing::buffers(*c,layout);if(rc) return rc;
    ggml_cuda_launch_mm_ids_helper(c->ids,c->src_rows,c->dst_rows,c->offsets,
        c->experts,c->tokens,c->topk,c->channels,int(c->ids_stride),int(c->source_stride),
        bool(c->inverse),static_cast<cudaStream_t>(c->stream));
    return int(cudaGetLastError());
}
