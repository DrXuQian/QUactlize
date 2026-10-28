// Test-only opaque access to the SAME Runtime/Driver node reader as Recorder.
// The captured graph owns argument storage; this holder never owns that graph.
#pragma once
#include "kernel_node.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>

struct nc_kernel_info_v1 {
    uint32_t grid[3], block[3], shared, driver;
    const char * symbol;
};
static_assert(sizeof(nc_kernel_info_v1)==40 && offsetof(nc_kernel_info_v1,symbol)==32,
              "kernel-node bridge requires the verified 64-bit host ABI");

extern "C" int nc_kernel_capture(void * node, void ** result, nc_kernel_info_v1 * info) {
    if(result) *result=nullptr;
    return nc::guard([&] {
        if(!node || !result || !info) throw std::runtime_error("invalid kernel capture request");
        auto p=std::make_unique<nc::KernelNode>();
        p->capture(reinterpret_cast<cudaGraphNode_t>(node));
        *info={}; info->symbol=p->name(); info->driver=p->driver_entry;
        if(p->driver_entry) {
            const auto & d=p->driver;
            info->grid[0]=d.gridDimX;info->grid[1]=d.gridDimY;info->grid[2]=d.gridDimZ;
            info->block[0]=d.blockDimX;info->block[1]=d.blockDimY;info->block[2]=d.blockDimZ;
            info->shared=d.sharedMemBytes;
        } else {
            const auto & r=p->runtime;
            info->grid[0]=r.gridDim.x;info->grid[1]=r.gridDim.y;info->grid[2]=r.gridDim.z;
            info->block[0]=r.blockDim.x;info->block[1]=r.blockDim.y;info->block[2]=r.blockDim.z;
            info->shared=r.sharedMemBytes;
        }
        *result=p.release();
    });
}
extern "C" int nc_kernel_add(void * handle, void ** result, void * graph, void * previous) {
    return nc::guard([&] {
        if(!handle || !result || !graph) throw std::runtime_error("invalid kernel add request");
        cudaGraphNode_t node=nullptr;
        static_cast<nc::KernelNode *>(handle)->add(&node,reinterpret_cast<cudaGraph_t>(graph),
                                                  reinterpret_cast<cudaGraphNode_t>(previous));
        *result=reinterpret_cast<void *>(node);
    });
}
extern "C" void nc_kernel_destroy(void * handle) { delete static_cast<nc::KernelNode *>(handle); }
