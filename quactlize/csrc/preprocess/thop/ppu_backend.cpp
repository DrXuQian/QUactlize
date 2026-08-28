#include "thop/ppu_backend.h"

#include <dlfcn.h>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

namespace torch_ext {
namespace ppu_backend {

namespace {

struct State {
  Api api{};
  bool ok = false;
  std::string why = "not attempted";
};

// -1 rather than 0: PPU_PACKED_FORMAT=0 is Q4_K, a REAL format, so zero cannot double as "unspecified"
// without making the default build and Q4_K indistinguishable.
constexpr int kDefaultFormat = -1;

// WHERE A FORMAT'S LIBRARY LIVES. PPU_PACKED_FORMAT is a COMPILE-TIME macro, so one library serves exactly one
// packed format -- and a Q4_K_M checkpoint is MIXED, carrying Q4_K tensors alongside Q6_K ones. A single handle
// therefore cannot serve a real model, and that is a fact about our build rather than about GGUF, which has
// stored a per-tensor type all along.
//
// Resolution, most specific first:
//   QUACTLIZE_PPU_LIB_FMT<k>   an explicit path for format k
//   QUACTLIZE_PPU_LIB          the default library's path, with _fmt<k> spliced before the suffix
//   libquactlize_ppu_fmt<k>.so the bare name, left to the loader search path
// Format kDefaultFormat means "the default build", which is the path QUACTLIZE_PPU_LIB names verbatim -- so every
// existing caller keeps its exact behaviour and nothing has to know about formats to keep working.
std::string library_path(int fmt) {
  char const* base_env = std::getenv("QUACTLIZE_PPU_LIB");
  std::string base = (base_env && *base_env) ? base_env : "libquactlize_ppu.so";
  if (fmt == kDefaultFormat) return base;

  std::string const key = "QUACTLIZE_PPU_LIB_FMT" + std::to_string(fmt);
  if (char const* e = std::getenv(key.c_str())) { if (*e) return e; }

  std::string const suffix = ".so";
  std::string const tag = "_fmt" + std::to_string(fmt);
  if (base.size() > suffix.size() && base.compare(base.size() - suffix.size(), suffix.size(), suffix) == 0)
    return base.substr(0, base.size() - suffix.size()) + tag + suffix;
  return base + tag;
}

// ONE ATTEMPT PER FORMAT, and each failure is remembered separately. Retrying dlopen per call would turn a
// missing library into a per-op cost and, worse, into a message that appears once and then stops -- and sharing
// one attempt across formats would let the first format's failure speak for libraries never tried.
//
// RTLD_LOCAL is what makes several of these safe at once: every library exports the same symbol names, and a
// global-scope load would let the first one answer for all of them. It was already local before any of this,
// which is the only reason per-format loading needs no change to the libraries themselves.
State& state(int fmt) {
  static std::map<int, State> states;
  static std::mutex m;
  std::lock_guard<std::mutex> guard(m);
  auto it = states.find(fmt);
  if (it != states.end()) return it->second;
  State& s = states[fmt];
  {
    std::string const path_s = library_path(fmt);
    char const* path = path_s.c_str();
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
      // dlerror() IS CONSUMING: the first call clears the error, so a second returns nullptr and
      // `std::string + nullptr` is undefined -- it segfaulted on import. Capture it once.
      char const* err = dlerror();
      s.why = std::string("dlopen(") + path + ") failed: " + (err ? err : "no error reported");
      return s;
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
    if (!sym("quactlize_ppu_vecdot", reinterpret_cast<void**>(&s.api.vecdot))) return s;
    dlerror();
    s.api.vecdot_dense = reinterpret_cast<decltype(s.api.vecdot_dense)>(
        dlsym(h, "quactlize_ppu_vecdot_dense"));
    dlerror();
    if (!sym("quactlize_ppu_vecdot_moe", reinterpret_cast<void**>(&s.api.vecdot_moe))) return s;
    if (!sym("quactlize_ppu_dequantize", reinterpret_cast<void**>(&s.api.dequantize))) return s;
    if (!sym("quactlize_ppu_prepass", reinterpret_cast<void**>(&s.api.prepass))) return s;
    dlerror();
    s.api.prepass_unit = reinterpret_cast<decltype(s.api.prepass_unit)>(
        dlsym(h, "quactlize_ppu_prepass_unit"));
    dlerror();
    s.api.units_bytes = reinterpret_cast<decltype(s.api.units_bytes)>(
        dlsym(h, "quactlize_ppu_units_bytes"));
    dlerror();
    s.api.prepare_units = reinterpret_cast<decltype(s.api.prepare_units)>(
        dlsym(h, "quactlize_ppu_prepare_units"));
    dlerror();
    s.api.prepare_units_grouped = reinterpret_cast<decltype(s.api.prepare_units_grouped)>(
        dlsym(h, "quactlize_ppu_prepare_units_grouped"));
    dlerror();
    s.api.bc_gemv = reinterpret_cast<decltype(s.api.bc_gemv)>(dlsym(h, "quactlize_ppu_bc_gemv"));
    dlerror();
    s.api.bc_gemv_for_arrangement = reinterpret_cast<decltype(s.api.bc_gemv_for_arrangement)>(
        dlsym(h, "quactlize_ppu_bc_gemv_for_arrangement_v1"));
    dlerror();
    if (!sym("quactlize_ppu_gemv_lowbit", reinterpret_cast<void**>(&s.api.gemv_lowbit))) return s;
    // Dense tensor-core symbols are hgcc-only. Do not make their absence invalidate a plain-nvcc CUDA-core GEMV
    // library; the dense op checks these pointers explicitly and refuses instead of falling back.
    dlerror();
    s.api.prepare_dense = reinterpret_cast<decltype(s.api.prepare_dense)>(dlsym(h, "quactlize_ppu_prepare_dense"));
    dlerror();
    s.api.recover_dense = reinterpret_cast<decltype(s.api.recover_dense)>(dlsym(h, "quactlize_ppu_recover_dense"));
    dlerror();
    s.api.prepare_dense_for_tile = reinterpret_cast<decltype(s.api.prepare_dense_for_tile)>(
        dlsym(h, "quactlize_ppu_prepare_dense_for_tile"));
    dlerror();
    s.api.recover_dense_for_tile = reinterpret_cast<decltype(s.api.recover_dense_for_tile)>(
        dlsym(h, "quactlize_ppu_recover_dense_for_tile"));
    dlerror();
    s.api.prepare_dense_for_arrangement_v2 =
        reinterpret_cast<decltype(s.api.prepare_dense_for_arrangement_v2)>(
            dlsym(h, "quactlize_ppu_prepare_dense_for_arrangement_v2"));
    dlerror();
    s.api.recover_dense_for_arrangement_v2 =
        reinterpret_cast<decltype(s.api.recover_dense_for_arrangement_v2)>(
            dlsym(h, "quactlize_ppu_recover_dense_for_arrangement_v2"));
    dlerror();
    s.api.dense_lowbit = reinterpret_cast<decltype(s.api.dense_lowbit)>(dlsym(h, "quactlize_ppu_dense_lowbit"));
    dlerror();
    s.api.dense_lowbit_for_arrangement_v2 =
        reinterpret_cast<decltype(s.api.dense_lowbit_for_arrangement_v2)>(
            dlsym(h, "quactlize_ppu_dense_lowbit_for_arrangement_v2"));
    dlerror();
    s.api.dense_lowbit_arrangement_valid_v2 =
        reinterpret_cast<decltype(s.api.dense_lowbit_arrangement_valid_v2)>(
            dlsym(h, "quactlize_ppu_dense_lowbit_config_valid_for_arrangement_v2"));
    dlerror();
    s.api.dense_fully_quantized = reinterpret_cast<decltype(s.api.dense_fully_quantized)>(
        dlsym(h, "quactlize_ppu_dense_fully_quantized"));
    dlerror();
    s.api.dense_fully_quantized_for_arrangement =
        reinterpret_cast<decltype(s.api.dense_fully_quantized_for_arrangement)>(
            dlsym(h, "quactlize_ppu_dense_fully_quantized_for_arrangement_v1"));
    dlerror();
    s.api.dense_fully_quantized_arrangement_valid =
        reinterpret_cast<decltype(s.api.dense_fully_quantized_arrangement_valid)>(
            dlsym(h, "quactlize_ppu_dense_fully_quantized_config_valid_for_arrangement_v1"));
    dlerror();
    s.api.dense_fully_quantized_for_arrangement_v2 =
        reinterpret_cast<decltype(s.api.dense_fully_quantized_for_arrangement_v2)>(
            dlsym(h, "quactlize_ppu_dense_fully_quantized_for_arrangement_v2"));
    dlerror();
    s.api.dense_fully_quantized_arrangement_valid_v2 =
        reinterpret_cast<decltype(s.api.dense_fully_quantized_arrangement_valid_v2)>(
            dlsym(h, "quactlize_ppu_dense_fully_quantized_config_valid_for_arrangement_v2"));
    dlerror();
    s.api.grouped_fully_quantized = reinterpret_cast<decltype(s.api.grouped_fully_quantized)>(
        dlsym(h, "quactlize_ppu_grouped_fully_quantized"));
    dlerror();
    s.api.grouped_fully_quantized_for_arrangement_v2 =
        reinterpret_cast<decltype(s.api.grouped_fully_quantized_for_arrangement_v2)>(
            dlsym(h, "quactlize_ppu_grouped_fully_quantized_for_arrangement_v2"));
    dlerror();
    s.api.grouped_fully_quantized_arrangement_valid_v2 =
        reinterpret_cast<decltype(s.api.grouped_fully_quantized_arrangement_valid_v2)>(
            dlsym(h,
                  "quactlize_ppu_grouped_fully_quantized_config_valid_for_arrangement_v2"));
    dlerror();
    s.ok = true;
    s.why = std::string("loaded ") + path;
  }
  return s;
}

}  // namespace

Api const* load_format(int fmt, std::string* why) {
  State& s = state(fmt);
  if (why) *why = s.why;
  return s.ok ? &s.api : nullptr;
}

Api const* load(std::string* why) {
  State& s = state(kDefaultFormat);
  if (why) *why = s.why;
  return s.ok ? &s.api : nullptr;
}

std::string resolved_backend() { return state(kDefaultFormat).ok ? "ppu" : "cpu"; }

}  // namespace ppu_backend
}  // namespace torch_ext
