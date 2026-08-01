// A stand-in for TensorRT-LLM's utils/cuda/cuda_utils.h, which was not copied with the port.
//
// I first wrote that the including file uses NOTHING from it, having grepped for TLLM_ and check_cuda. It uses
// FT_CHECK_WITH_INFO, the FasterTransformer assertion macro that TensorRT-LLM inherited -- so the grep was for the
// wrong names and the conclusion was wrong. The macro is provided below with its upstream SEMANTICS (abort with the
// message on a false condition) rather than its upstream text, because the two callers only need it to reject an
// invalid quant type loudly.
#pragma once
#include <cstdio>
#include <stdexcept>
#include <string>

namespace quactlize {
// Takes std::string, because FT_CHECK_WITH_INFO is called with fmtstr(...) results as often as with literals.
inline void check_or_throw(bool ok, std::string const& what, char const* file, int line) {
  if (!ok) throw std::runtime_error(what + " at " + file + ":" + std::to_string(line));
}
}  // namespace quactlize
#define QUACTLIZE_CHECK(x) ::quactlize::check_or_throw((x), #x, __FILE__, __LINE__)

// The FasterTransformer spelling the ported sources use.
#define FT_CHECK_WITH_INFO(cond, info) ::quactlize::check_or_throw((cond), (info), __FILE__, __LINE__)
#define FT_CHECK(cond)                 ::quactlize::check_or_throw((cond), #cond, __FILE__, __LINE__)

// FasterTransformer's printf-style string builder, used only to compose assertion messages here.
namespace quactlize {
template <typename... Args>
inline std::string fmtstr_impl(char const* fmt, Args... args) {
  int const n = std::snprintf(nullptr, 0, fmt, args...);
  if (n < 0) return std::string(fmt);
  std::string out(static_cast<size_t>(n) + 1, '\0');
  std::snprintf(&out[0], out.size(), fmt, args...);
  out.resize(static_cast<size_t>(n));
  return out;
}
}  // namespace quactlize
#define fmtstr(...) ::quactlize::fmtstr_impl(__VA_ARGS__)
