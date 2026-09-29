#pragma once
#include <stdexcept>
// Only the standalone scalar-type descriptor needs these host checks.
// Keep them live: malformed reference arguments must not become a timing.
#define STD_TORCH_CHECK(condition, ...) \
  do { if (!(condition)) throw std::invalid_argument("vLLM host check: " #condition); } while (0)
#define STD_TORCH_CHECK_NOT_IMPLEMENTED(condition, ...) STD_TORCH_CHECK(condition, __VA_ARGS__)
