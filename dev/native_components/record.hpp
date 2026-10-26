// Test-only recording of the unchanged native dispatcher. Included AFTER the
// caller's real headers, compiled with its real target flags.
#pragma once
#include <functional>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include "nlohmann/json.hpp"
#include "ggml-cuda/mmid.cuh"
#include "kernel_node.hpp"
#include <cuda_profiler_api.h>

namespace nc {
using Json = nlohmann::ordered_json;
inline void check(cudaError_t rc, const char * step) {
    if (rc != cudaSuccess) throw std::runtime_error(std::string(step)+": "+cudaGetErrorString(rc));
}
struct Stage {
    KernelNode kernel;
    std::function<cublasStatus_t()> provider;
    Json description;
};
struct Recorder {
    cudaStream_t stream;
    bool capturing = false;
    std::vector<cudaGraph_t> owners;
    std::vector<Stage> stages;
    explicit Recorder(cudaStream_t s) : stream(s) {}
    void begin() {
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "native capture begin");
        capturing = true;
    }
    void end() {
        if (!capturing) throw std::runtime_error("capture is not active");
        cudaGraph_t graph = nullptr;
        auto rc = cudaStreamEndCapture(stream, &graph); capturing = false;
        if (graph) owners.push_back(graph);
        check(rc, "native capture end");
        size_t count = 0, edges = 0;
        check(cudaGraphGetNodes(graph, nullptr, &count), "native graph nodes count");
        if (!count) return;
        std::vector<cudaGraphNode_t> nodes(count);
        check(cudaGraphGetNodes(graph, nodes.data(), &count), "native graph nodes");
        check(cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &edges), "native graph edge count");
        std::vector<cudaGraphNode_t> from(edges), to(edges);
        std::vector<cudaGraphEdgeData> data(edges);
        check(cudaGraphGetEdges(graph, from.data(), to.data(), data.data(), &edges), "native graph edges");
        std::map<cudaGraphNode_t, std::set<cudaGraphNode_t>> incoming;
        for (auto node : nodes) incoming[node];
        for (size_t i=0;i<edges;++i) {
            cudaGraphEdgeData zero{};
            if (std::memcmp(&data[i], &zero, sizeof(zero)))
                throw std::runtime_error("PDL/non-default dependency is not a serial component sum");
            incoming.at(to[i]).insert(from[i]);
        }
        while (!incoming.empty()) {
            std::vector<cudaGraphNode_t> ready;
            for (const auto & item : incoming) if (item.second.empty()) ready.push_back(item.first);
            if (ready.size()!=1) throw std::runtime_error("native graph is nonlinear; no work omitted");
            auto node=ready.front(); incoming.erase(node);
            for (auto & item : incoming) item.second.erase(node);
            cudaGraphNodeType kind;
            check(cudaGraphNodeGetType(node, &kind), "native graph node type");
            if (kind!=cudaGraphNodeTypeKernel)
                throw std::runtime_error("native non-kernel work requires an explicit timer, type="+std::to_string(kind));
            Stage stage;
            stage.kernel.capture(node);
            const char * name=stage.kernel.name();
            const auto & p=stage.kernel.runtime;
            const auto & d=stage.kernel.driver;
            stage.description={{"entry","KERNEL"},{"symbol",name},
                {"grid",stage.kernel.driver_entry ? Json{d.gridDimX,d.gridDimY,d.gridDimZ} : Json{p.gridDim.x,p.gridDim.y,p.gridDim.z}},
                {"block",stage.kernel.driver_entry ? Json{d.blockDimX,d.blockDimY,d.blockDimZ} : Json{p.blockDim.x,p.blockDim.y,p.blockDim.z}},
                {"shared_bytes",stage.kernel.driver_entry ? d.sharedMemBytes : p.sharedMemBytes}};
            if(stage.kernel.driver_entry) stage.description["parameter_api"]="DRIVER_CUFUNCTION";
            stages.push_back(std::move(stage));
        }
    }
    ~Recorder() {
        if (capturing) { cudaGraph_t g=nullptr; cudaStreamEndCapture(stream,&g); if(g) owners.push_back(g); }
        cudaStreamSynchronize(stream);
        for (auto graph : owners) cudaGraphDestroy(graph);
    }
};
inline thread_local Recorder * active=nullptr;

template<class T> Json argument(T value) {
    if constexpr (std::is_enum_v<T>) return int(value);
    else if constexpr (std::is_arithmetic_v<T>) return value;
    else return "POINTER"; // no ASLR address in launch identity
}
template<class F, class... Args>
cublasStatus_t provider(const char * name, cudaStream_t stream, F fn, Args... args) {
    if (!active) return fn(args...);
    if (active->stream!=stream) throw std::runtime_error("native provider changed stream");
    active->end(); // cuBLAS NEVER runs inside graph capture
    Stage stage;
    stage.provider=[=]() {return fn(args...);};
    stage.description={{"entry","PROVIDER"},{"symbol",name},
                       {"arguments",Json::array({argument(args)...})}};
    active->stages.push_back(stage);
    // Warmup outside capture. All inputs were produced by the preceding
    // identical ordinary warmup; scratch/handles stay owned by the session.
    auto rc=stage.provider();
    active->begin();
    return rc;
}
} // namespace nc
