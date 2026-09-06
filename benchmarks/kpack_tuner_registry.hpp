#pragma once
// Private profiler ABI, not the public inference ABI. All modules and the
// driver must come from the same source/SDK receipt. Registry rows retain the
// existing benchmark function signatures and therefore the shipping kernels.
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <cstdint>
#include <cstdio>

namespace kpack_tuner {
struct PhaseTimer {
  char const* name;
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  explicit PhaseTimer(char const* value) : name(value) {}
  ~PhaseTimer() {
    std::printf("KPACK_TUNER_PHASE phase=%s seconds=%.6f\n", name,
        std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
  }
};

// One weight geometry per process. Keep only placed weights/metadata, not
// native/recovered duplicates or A/golden. A/router and their independent
// reference are regenerated for every request. Never cache a failed oracle.
template <class Half>
struct HostWeightCache {
  int n=0, k=0, experts=0, mode=0;
  bool ready=false, isolation=false, high_covered=false;
  std::vector<std::uint8_t> low, high, units;
  std::vector<Half> scale, zero;
  bool matches(int n_, int k_, int experts_=1, int mode_=0) const {
    bool hit=ready && n==n_ && k==k_ && experts==experts_ && mode==mode_;
    std::printf("KPACK_TUNER_WEIGHT_CACHE hit=%d n=%d k=%d experts=%d\n",
                int(hit),n_,k_,experts_);
    return hit;
  }
  void bind(int n_, int k_, int experts_=1, int mode_=0) {
    n=n_; k=k_; experts=experts_; mode=mode_; ready=true;
  }
};

struct Module {
  unsigned version;
  std::size_t row_size, count;
  char const* contract;
  void const* rows;
};
using Query = Module const* (*)();

// Do not dlclose between measurements: PPU registration and row function
// pointers must outlive every request. The coordinator bounds each process
// to one qtype/route/weight geometry, not the entire kernel inventory.
inline void* open_module(std::string const& path) {
  static std::map<std::string, void*> handles;
  auto found = handles.find(path);
  if (found != handles.end()) return found->second;
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) throw std::runtime_error(std::string("dlopen: ") + dlerror());
  handles.emplace(path, handle);
  return handle;
}

template <class Row>
std::vector<Row> load_registry(char const* contract) {
  char const* path = std::getenv("KPACK_TUNER_MODULES");
  if (!path || !*path) throw std::runtime_error("KPACK_TUNER_MODULES is absent");
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open module list");
  std::vector<Row> rows;
  std::set<std::string> paths, symbols;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] != '/' || line.find_first_of("\t\r") != std::string::npos ||
        !paths.insert(line).second) throw std::runtime_error("invalid module path");
    auto query = reinterpret_cast<Query>(dlsym(open_module(line), "kpack_tuner_module_v1"));
    if (!query) throw std::runtime_error("module registry export is absent");
    Module const* module = query();
    if (!module || module->version != 1 || module->row_size != sizeof(Row) ||
        !module->contract || std::strcmp(module->contract, contract) ||
        !module->count || module->count > 32 || !module->rows)
      throw std::runtime_error("module ABI/source/route contract differs");
    auto begin = static_cast<Row const*>(module->rows);
    for (std::size_t i = 0; i < module->count; ++i) {
      if (!begin[i].symbol || !begin[i].run ||
          !symbols.insert(begin[i].symbol).second)
        throw std::runtime_error("duplicate or invalid registry row");
      rows.push_back(begin[i]);
    }
  }
  if (rows.empty()) throw std::runtime_error("empty module registry");
  return rows;
}
}  // namespace kpack_tuner
