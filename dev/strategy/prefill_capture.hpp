// Experiment-only bridge. The carrier supplies the SDK runtime declarations.
#pragma once
#include <cstdio>
#include <vector>

namespace strategy_prefill {

inline int clone_capture(void* stream, void** result) {
    if (!result) return int(hggcErrorInvalidValue);
    *result = nullptr;
    hggcStreamCaptureStatus status = hggcStreamCaptureStatusNone;
    hggcGraph_t graph = nullptr, clone = nullptr;
    auto rc = hggcStreamGetCaptureInfo(static_cast<hggcStream_t>(stream), &status,
                                     nullptr, &graph, nullptr, nullptr, nullptr);
    std::fprintf(stderr, "STRATEGY_PROVIDER_CAPTURE stage=capture-info rc=%d capture_status=%d\n",
                 int(rc), int(status));
    if (rc != hggcSuccess) return int(rc);
    if (status != hggcStreamCaptureStatusActive || !graph) return int(hggcErrorInvalidValue);
    size_t count = 0;
    rc = hggcGraphGetNodes(graph, nullptr, &count);
    std::fprintf(stderr, "STRATEGY_PROVIDER_CAPTURE stage=node-count rc=%d nodes=%zu\n", int(rc), count);
    if (rc != hggcSuccess) return int(rc);
    if (!count || count > 65536) return int(hggcErrorInvalidValue);
    std::vector<hggcGraphNode_t> nodes(count);
    rc = hggcGraphGetNodes(graph, nodes.data(), &count);
    if (rc != hggcSuccess) {
        std::fprintf(stderr, "STRATEGY_PROVIDER_CAPTURE stage=node-list rc=%d\n", int(rc));
        return int(rc);
    }
    if (count != nodes.size()) return int(hggcErrorInvalidValue);
    size_t non_kernel = 0;
    for (size_t i = 0; i < count; ++i) {
        hggcGraphNodeType type;
        rc = hggcGraphNodeGetType(nodes[i], &type);
        if (rc != hggcSuccess) {
            std::fprintf(stderr, "STRATEGY_PROVIDER_CAPTURE stage=node-type index=%zu rc=%d\n", i, int(rc));
            return int(rc);
        }
        if (type != hggcGraphNodeTypeKernel) {
            ++non_kernel;
            std::fprintf(stderr, "STRATEGY_PROVIDER_CAPTURE stage=non-kernel index=%zu type=%d\n", i, int(type));
        }
    }
    std::fprintf(stderr, "STRATEGY_PROVIDER_CAPTURE stage=inventory nodes=%zu non_kernel=%zu\n", count, non_kernel);
    // CUDA does not permit cloning allocation/free/conditional nodes. Other
    // non-kernel nodes are outside our isolated-kernel timing contract too.
    // Never drop those nodes: later kernels may depend on their storage/work.
    if (non_kernel) {
        std::fprintf(stderr, "STRATEGY_PROVIDER_CAPTURE stage=reject-non-kernel rc=%d origin=HARNESS_NOT_CLONE_API\n",
                     int(hggcErrorNotSupported));
        return int(hggcErrorNotSupported);
    }
    rc = hggcGraphClone(&clone, graph);
    std::fprintf(stderr, "STRATEGY_PROVIDER_CAPTURE stage=clone rc=%d\n", int(rc));
    if (rc == hggcSuccess) *result = clone;
    return int(rc);
}

} // namespace strategy_prefill
