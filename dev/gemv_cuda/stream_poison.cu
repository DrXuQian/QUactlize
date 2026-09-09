// Development-only real CUDA proof of the test harness's ordering hazard.
// This does not execute, emulate or admit a PPU kernel.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#define CHECK(call) do { auto rc = (call); if (rc != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(rc)); return 2; } } while (0)

__global__ void delay_default(unsigned long long clocks) {
  auto start = clock64();
  while (clock64() - start < clocks) {}
}

__global__ void write_header(uint32_t* out) {
  if (threadIdx.x == 0) {
    out[0] = 2; out[1] = 0; out[2] = 8; out[3] = 4;
  }
}

int main() {
  uint32_t* out = nullptr;
  cudaStream_t stream;
  CHECK(cudaMalloc(&out, 16));
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  // Resolve lazy modules before deliberately delaying the default stream.
  delay_default<<<1, 32>>>(1);
  write_header<<<1, 32, 0, stream>>>(out);
  CHECK(cudaDeviceSynchronize());
  int failures = 0;
  for (int graph_mode = 0; graph_mode < 2; ++graph_mode) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (graph_mode) {
      CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      write_header<<<1, 32, 0, stream>>>(out);
      CHECK(cudaStreamEndCapture(stream, &graph));
      CHECK(cudaGraphInstantiate(&executable, graph, 0));
    }
    for (int arm = 0; arm < 3; ++arm) {
      // Same legal interleaving as a delayed fixture fill: no implicit edge
      // exists from the legacy stream to a nonblocking consumer stream.
      delay_default<<<1, 32>>>(20000000ULL);
      if (arm == 1) {
        CHECK(cudaMemsetAsync(out, 0xa5, 16, stream));
      } else {
        CHECK(cudaMemset(out, 0xa5, 16));
        if (arm == 2) CHECK(cudaStreamSynchronize(nullptr));
      }
      if (graph_mode) CHECK(cudaGraphLaunch(executable, stream));
      else write_header<<<1, 32, 0, stream>>>(out);
      CHECK(cudaGetLastError());
      CHECK(cudaStreamSynchronize(stream));
      CHECK(cudaDeviceSynchronize());
      uint32_t got[4], expected[4] = {2, 0, 8, 4};
      CHECK(cudaMemcpy(got, out, 16, cudaMemcpyDeviceToHost));
      int bad = 0, poison = 0;
      for (int i = 0; i < 4; ++i) {
        bad += got[i] != expected[i];
        poison += got[i] == 0xa5a5a5a5U;
      }
      bool const ok = arm == 0 ? bad == 4 && poison == 4 : bad == 0;
      failures += !ok;
      std::printf("CUDA_STREAM_POISON graph=%d arm=%s bad=%d poison=%d verdict=%s\n",
          graph_mode, arm == 0 ? "legacy-unordered" : arm == 1 ? "same-stream" : "explicit-default-drain",
          bad, poison, ok ? (arm == 0 ? "EXPECTED_RED" : "PASS") : "FAIL");
    }
    if (graph_mode) {
      CHECK(cudaGraphExecDestroy(executable));
      CHECK(cudaGraphDestroy(graph));
    }
  }
  CHECK(cudaStreamDestroy(stream));
  CHECK(cudaFree(out));
  return failures ? 1 : 0;
}
