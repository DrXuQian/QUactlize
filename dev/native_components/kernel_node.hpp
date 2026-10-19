// Captured native kernels may come from runtime registration OR a JIT module.
// A CUfunction is not a runtime host stub; keep it in the driver API throughout.
#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace nc {
inline void driver_check(CUresult rc, const char * step) {
    if (rc != CUDA_SUCCESS) {
        const char * name = nullptr;
        cuGetErrorName(rc, &name);
        throw std::runtime_error(std::string(step) + ": driver rc=" + std::to_string(int(rc)) +
                                 (name ? " " + std::string(name) : ""));
    }
}
struct KernelNode {
    cudaKernelNodeParams runtime{};
    CUDA_KERNEL_NODE_PARAMS driver{};
    bool driver_entry = false;

    void capture(cudaGraphNode_t node) {
        const auto rc = cudaGraphKernelNodeGetParams(node, &runtime);
        if (rc == cudaSuccess) return;
        // Do not turn an unrelated capture/async error into a different API's success.
        if (rc != cudaErrorSymbolNotFound) {
            throw std::runtime_error(std::string("native kernel params: ") + cudaGetErrorString(rc));
        }
        const auto pending=cudaGetLastError();
        if(pending!=cudaSuccess && pending!=cudaErrorSymbolNotFound)
            throw std::runtime_error(std::string("native pending error before driver query: ")+cudaGetErrorString(pending));
        driver_check(cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(node), &driver),
                     "native JIT kernel params");
        if (!driver.func) throw std::runtime_error("native JIT node has no CUfunction; no timing admitted");
        driver_entry = true;
    }
    const char * name() const {
        const char * name = nullptr;
        if (driver_entry) driver_check(cuFuncGetName(&name, driver.func), "native JIT kernel name");
        else {
            const auto rc = cudaFuncGetName(&name, runtime.func);
            if (rc != cudaSuccess) throw std::runtime_error(std::string("native kernel name: ") + cudaGetErrorString(rc));
        }
        if (!name || !*name) throw std::runtime_error("missing native kernel name");
        return name;
    }
    void add(cudaGraphNode_t * node, cudaGraph_t graph, cudaGraphNode_t previous) const {
        if (driver_entry) {
            CUgraphNode added = nullptr, dependency = reinterpret_cast<CUgraphNode>(previous);
            driver_check(cuGraphAddKernelNode(&added, reinterpret_cast<CUgraph>(graph),
                previous ? &dependency : nullptr, previous ? 1 : 0, &driver), "single-stage JIT graph add");
            *node = reinterpret_cast<cudaGraphNode_t>(added);
        } else {
            const auto rc = cudaGraphAddKernelNode(node, graph, previous ? &previous : nullptr,
                                                   previous ? 1 : 0, &runtime);
            if (rc != cudaSuccess) throw std::runtime_error(std::string("single-stage graph add: ") + cudaGetErrorString(rc));
        }
    }
    // The argument arrays remain owned by the original captured graph. Recorder
    // keeps that graph (and the dispatch context/module) alive until all replays end.
};
} // namespace nc
