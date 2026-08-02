#include "thop/ppu_backend.h"

#include <dlfcn.h>
#include <cstdlib>
#include <mutex>

namespace torch_ext {
namespace ppu_backend {

namespace {

struct State {
  Api api{};
  bool ok = false;
  std::string why = "not attempted";
};

// ONE ATTEMPT, and the failure is remembered. Retrying dlopen per call would turn a missing library into a per-op
// cost and, worse, into a message that appears once and then stops.
State& state() {
  static State s;
  static std::once_flag once;
  std::call_once(once, [] {
    char const* env = std::getenv("QUACTLIZE_PPU_LIB");
    char const* path = (env && *env) ? env : "libquactlize_ppu.so";
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
      // dlerror() IS CONSUMING: the first call clears the error, so a second returns nullptr and
      // `std::string + nullptr` is undefined -- it segfaulted on import. Capture it once.
      char const* err = dlerror();
      s.why = std::string("dlopen(") + path + ") failed: " + (err ? err : "no error reported");
      return;
    }
    // EVERY SYMBOL, NOT THE FIRST. A library that exports two of the three would otherwise load and then crash on
    // the third call, which is a much worse failure than not loading.
    auto sym = [&](char const* n, void** out) {
      dlerror();
      *out = dlsym(h, n);
      char const* e = dlerror();
      if (e || !*out) { s.why = std::string("missing symbol ") + n + " in " + path; return false; }
      return true;
    };
    if (!sym("quactlize_ppu_vecdot", reinterpret_cast<void**>(&s.api.vecdot))) return;
    if (!sym("quactlize_ppu_dequantize", reinterpret_cast<void**>(&s.api.dequantize))) return;
    if (!sym("quactlize_ppu_prepass", reinterpret_cast<void**>(&s.api.prepass))) return;
    s.ok = true;
    s.why = std::string("loaded ") + path;
  });
  return s;
}

}  // namespace

Api const* load(std::string* why) {
  State& s = state();
  if (why) *why = s.why;
  return s.ok ? &s.api : nullptr;
}

std::string resolved_backend() { return state().ok ? "ppu" : "cpu"; }

}  // namespace ppu_backend
}  // namespace torch_ext
