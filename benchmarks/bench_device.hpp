// BIND THIS PROCESS TO ONE PPU, so a sweep can run N invocations on N cards at once.
//
//   PPU_BENCH_DEVICE=2 ./the_bench ...
//
// WHY AN ENV VAR AND NOT A FLAG. test_lowbit_dense_bench parses --m/--n/--k; test_lowbit_moe_bench and
// test_moe_splitk_bench take POSITIONAL arguments and have no flag parser at all ("--m=2048" would become
// atoi("--m=2048") == 0 and select shape zero, which build.sh's own run hint warns about). One env var works for
// both shapes of bench and for anything added later, and it shows up in the process environment where a driver
// can set it per worker without touching an argv it does not own.
//
// WHY IT DID NOT EXIST. Nothing in this repository ever called hggcSetDevice -- hggcGetDevice appears only to
// query properties for the roofline. Every process therefore took whatever the runtime picked, which is fine for
// one invocation at a time and silently wrong the moment a driver runs several: N workers would all pile onto one
// card, and the sweep would report N times the invocations at 1/N the throughput with no error anywhere.
//
// UNSET MEANS UNCHANGED, deliberately. Existing single-process runs and every recorded number must keep their
// current behaviour, so the default path calls nothing at all rather than "setting" device 0.
//
// IT SAYS WHICH CARD IT GOT, and not because it is chatty: a sweep result that cannot be attributed to a device
// is the same defect as one that cannot be attributed to a config, and this repository has already paid for that
// once (BACKTEST's D7, filed against a capacity nobody witnessed).
#pragma once

#include <cstdio>
#include <cstdlib>

namespace bench_device {

// Returns the bound device, or -1 when PPU_BENCH_DEVICE is unset and nothing was changed.
inline int bind_from_env() {
  char const* p = std::getenv("PPU_BENCH_DEVICE");
  if (p == nullptr || *p == '\0') return -1;
  char* end = nullptr;
  long const want = std::strtol(p, &end, 10);
  if (end == p || *end != '\0' || want < 0) {
    std::fprintf(stderr, "[bench_device] PPU_BENCH_DEVICE='%s' is not a device index; REFUSING to guess. "
                         "Unset it or give a non-negative integer.\n", p);
    std::exit(2);
  }
  hggcError_t const e = hggcSetDevice(int(want));
  if (e != hggcSuccess) {
    // A worker that cannot get its own card must DIE, not fall back to device 0. Falling back is how N workers
    // end up on one card while the driver reports N-way parallelism.
    std::fprintf(stderr, "[bench_device] hggcSetDevice(%d) failed (%d) -- refusing to run on a different card "
                         "than the one asked for.\n", int(want), int(e));
    std::exit(2);
  }
  int got = -1;
  hggcGetDevice(&got);
  if (got != int(want)) {
    std::fprintf(stderr, "[bench_device] asked for device %d, runtime reports %d. Not proceeding.\n",
                 int(want), got);
    std::exit(2);
  }
  std::printf("[bench_device] bound to PPU %d\n", got);
  std::fflush(stdout);
  return got;
}

}  // namespace bench_device
