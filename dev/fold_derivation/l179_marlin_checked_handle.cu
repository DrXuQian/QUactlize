// L179 -- executable lifecycle and compile-time API proof for the owned
// standalone Marlin handle.  The handle is intentionally instantiated over a
// host fake here: L169/L176 separately prove that the shipping alias reaches
// the exact generated device body.  Mixing those jobs would turn an unrelated
// fake-PPU device diagnostic into an API result.

#include <cstdio>
#include <type_traits>

#include "quactlize_extensions/cutlass/gemm/device/marlin_gemm_ppu.hpp"

struct L179Kernel {};
struct L179Tile {};
struct L179ElementA {};
struct L179ElementB {};
struct L179ElementC {};
struct L179ElementD {};
struct L179Accumulator {};
struct L179Mainloop {};
struct L179Epilogue {};
struct L179Arguments { bool supported = false; };
struct L179Params {};

struct L179FakeRaw {
  using GemmKernel = L179Kernel;
  using TileShape = L179Tile;
  using ElementA = L179ElementA;
  using ElementB = L179ElementB;
  using ElementC = L179ElementC;
  using ElementD = L179ElementD;
  using ElementAccumulator = L179Accumulator;
  using CollectiveMainloop = L179Mainloop;
  using CollectiveEpilogue = L179Epilogue;
  using Arguments = L179Arguments;
  using Params = L179Params;

  static inline int can_calls = 0;
  static inline int workspace_calls = 0;
  static inline int grid_calls = 0;
  static inline int initialize_calls = 0;
  static inline int run_calls = 0;

  static void reset() {
    can_calls = workspace_calls = grid_calls = initialize_calls = run_calls = 0;
  }
  static cutlass::Status can_implement(Arguments const& args) {
    ++can_calls;
    return args.supported ? cutlass::Status::kSuccess : cutlass::Status::kInvalid;
  }
  static size_t get_workspace_size(Arguments const&) {
    ++workspace_calls;
    return 4096;
  }
  static int maximum_active_blocks(int = -1) { return 6; }
  static dim3 get_grid_shape(Arguments const&, void* = nullptr) {
    ++grid_calls;
    return dim3(7, 1, 1);
  }
  cutlass::Status initialize(
      Arguments const& args, void* = nullptr, hggcStream_t = nullptr,
      cutlass::HostAdapter* = nullptr) {
    ++initialize_calls;
    return args.supported ? cutlass::Status::kSuccess : cutlass::Status::kInvalid;
  }
  cutlass::Status run(
      hggcStream_t = nullptr, cutlass::HostAdapter* = nullptr,
      bool = false) {
    ++run_calls;
    return cutlass::Status::kSuccess;
  }
};

using L179Gemm = cutlass::gemm::device::detail::MarlinCheckedHandlePPU<
    L179FakeRaw>;

#if defined(L179_PLANT_UPDATE)
void l179_forbidden_update() {
  L179Gemm gemm;
  L179Arguments args{};
  (void)gemm.update(args);
}
#elif defined(L179_PLANT_RAW_PARAMS_RUN)
void l179_forbidden_raw_params_run() {
  L179Params params{};
  (void)L179Gemm::run(params);
}
#elif defined(L179_PLANT_PARAMS_ACCESS)
void l179_forbidden_params_access() {
  L179Gemm gemm;
  (void)gemm.params();
}
#elif defined(L179_PLANT_UPCAST)
static_assert(std::is_base_of_v<L179FakeRaw, L179Gemm>,
              "L179_FORBIDS_RAW_ADAPTER_UPCAST");
#elif defined(L179_PLANT_PARAMS_CALL)
void l179_forbidden_params_call() {
  L179Gemm gemm;
  L179Params params{};
  (void)gemm(params);
}
#elif defined(L179_PLANT_PARAMS_GRID)
void l179_forbidden_params_grid() {
  L179Params params{};
  (void)L179Gemm::get_grid_shape(params);
}
#else
int main() {
  L179FakeRaw::reset();
  L179Gemm gemm;
  L179Arguments good{true};
  L179Arguments bad{false};

  if (gemm.run() != cutlass::Status::kErrorInvalidProblem ||
      L179FakeRaw::run_calls != 0) {
    std::fprintf(stderr, "[l179-handle] FAIL: run-before-initialize escaped\n");
    return 1;
  }
  if (L179Gemm::get_workspace_size(bad) != 0 ||
      L179FakeRaw::workspace_calls != 0 ||
      L179Gemm::get_grid_shape(bad).x != 0 ||
      L179FakeRaw::grid_calls != 0) {
    std::fprintf(stderr, "[l179-handle] FAIL: unsupported diagnostics reached raw lowering\n");
    return 1;
  }
  if (L179Gemm::get_workspace_size(good) != 4096 ||
      L179FakeRaw::workspace_calls != 1 ||
      L179Gemm::get_grid_shape(good).x != 7 ||
      L179FakeRaw::grid_calls != 1) {
    std::fprintf(stderr, "[l179-handle] FAIL: supported diagnostics were not forwarded\n");
    return 1;
  }
  if (gemm.initialize(good) != cutlass::Status::kSuccess ||
      gemm.run() != cutlass::Status::kSuccess ||
      L179FakeRaw::initialize_calls != 1 || L179FakeRaw::run_calls != 1) {
    std::fprintf(stderr, "[l179-handle] FAIL: initialize-then-run lifecycle broke\n");
    return 1;
  }
  if (gemm.initialize(bad) != cutlass::Status::kInvalid ||
      gemm.run() != cutlass::Status::kErrorInvalidProblem ||
      L179FakeRaw::initialize_calls != 2 || L179FakeRaw::run_calls != 1) {
    std::fprintf(stderr, "[l179-handle] FAIL: failed reinitialize retained stale Params\n");
    return 1;
  }
  if (gemm.run(good) != cutlass::Status::kSuccess ||
      L179FakeRaw::initialize_calls != 3 || L179FakeRaw::run_calls != 2) {
    std::fprintf(stderr, "[l179-handle] FAIL: checked one-shot run did not lower and launch\n");
    return 1;
  }
  static_assert(!std::is_base_of_v<L179FakeRaw, L179Gemm>);
  std::printf(
      "[l179-handle] PASS: preinit=blocked supported=forwarded "
      "failed-reinit=revoked one-shot=checked\n");
  return 0;
}
#endif
