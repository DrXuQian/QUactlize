#include "host.hpp"
#include "ppu_placed_arrangement.hpp"
#include "q8_kpack2.hpp"

#include <algorithm>
#include <limits>
#include <string>

namespace quactlize::ppu17 {
namespace {
thread_local std::string last_error;
constexpr q17_config_v1 configs[] = {
    {1, sizeof(q17_config_v1), Q17_SIMT, 1, 32, 1, 128, 8},
    {1, sizeof(q17_config_v1), Q17_SIMT, 1, 64, 1, 128, 16},
    {1, sizeof(q17_config_v1), Q17_WGMMA, 64, 64, 64, 128, 0},
    {1, sizeof(q17_config_v1), Q17_WGMMA, 64, 128, 64, 128, 0},
    {1, sizeof(q17_config_v1), Q17_WGMMA, 64, 64, 128, 128, 0},
    {1, sizeof(q17_config_v1), Q17_TMA_WGMMA, 64, 64, 64, 128, 0},
};
bool dtype(int type) { return type >= Q17_F16 && type <= Q17_F32; }
bool same_config(q17_config_v1 const &a, q17_config_v1 const &b) {
  return a.engine == b.engine && a.tile_m == b.tile_m && a.tile_n == b.tile_n &&
      a.tile_k == b.tile_k && a.threads == b.threads && a.lanes_n == b.lanes_n;
}
}

int error(int code, char const *message) { last_error = message; return code; }

int validate_problem(q17_problem_v1 const *p) {
  if (!p || p->version != 1 || p->struct_bytes != sizeof(*p))
    return error(Q17_INVALID, "problem ABI differs");
  if ((p->operation != Q17_DENSE && p->operation != Q17_GROUPED) ||
      (p->route != Q17_FQ && p->route != Q17_SF) ||
      (p->compute_type != Q17_F16 && p->compute_type != Q17_BF16) ||
      !dtype(p->input_type) || !dtype(p->output_type))
    return error(Q17_UNSUPPORTED, "operation/route/dtype differs");
  if (p->m <= 0 || p->n <= 0 || p->k <= 0 || p->experts <= 0 || p->experts > 65535 ||
      p->m > std::numeric_limits<int32_t>::max() - 256 ||
      p->n > std::numeric_limits<int32_t>::max() - 256 ||
      p->k > std::numeric_limits<int32_t>::max() - 256 ||
      p->max_rows <= 0 || p->max_rows > p->m ||
      int64_t(p->experts) * p->max_rows < p->m ||
      (p->operation == Q17_DENSE && (p->experts != 1 || p->max_rows != p->m)))
    return error(Q17_INVALID, "invalid row/expert extents");
  quactlize_ppu_placed_arrangement_v2 expected{};
  if (q17_canonical_arrangement_v1(p->qtype, &expected)) return Q17_UNSUPPORTED;
  if (!ppu_arrangements::same_descriptor(p->arrangement, expected))
    return error(Q17_UNSUPPORTED, "noncanonical K-pack arrangement");
  // Existing artifacts require whole GGUF superblocks; Q3/Q6 pair them.
  int quantum = p->qtype == 11 || p->qtype == 14 ? 512 : 256;
  if (p->k % quantum || (p->qtype == 13 && p->n % 16) ||
      (p->qtype == 8 && p->n % 256))
    return error(Q17_UNSUPPORTED, "shape does not contain whole canonical units");
  constexpr uint64_t bound = uint64_t(std::numeric_limits<int64_t>::max()) / 4;
  if (uint64_t(p->n) * p->k > bound / p->experts ||
      uint64_t(p->m) * std::max(p->n, p->k) > bound)
    return error(Q17_INVALID, "address extent overflow");
  return error(Q17_SUCCESS, "");
}

int validate_call(q17_call_v1 const *c, bool expansion) {
  if (!c || c->version != 1 || c->struct_bytes != sizeof(*c) || c->reserved != 0 || c->device < 0)
    return error(Q17_INVALID, "call ABI/device differs");
  int status = validate_problem(&c->problem);
  if (status) return status;
  if (expansion) {
    if (!c->units || !c->scale || !c->zero)
      return error(Q17_INVALID, "expansion requires units and separate output planes");
    if (c->units == c->scale || c->units == c->zero || c->scale == c->zero)
      return error(Q17_INVALID, "in-place scale expansion is not supported");
    if (uintptr_t(c->scale) % 2 || uintptr_t(c->zero) % 2)
      return error(Q17_INVALID, "expanded planes require two-byte alignment");
    return Q17_SUCCESS;
  }
  status = q17_can_implement_v1(&c->problem, &c->config);
  if (status) return status;
  if (!c->a || !c->low || !c->output ||
      (!!c->high != (c->problem.arrangement.high_bits != 0)) ||
      (c->problem.route == Q17_FQ && !c->units) ||
      (c->problem.route == Q17_SF && (!c->scale || !c->zero)) ||
      (c->problem.operation == Q17_GROUPED && !c->offsets) ||
      c->a_stride < c->problem.k || c->output_stride < c->problem.n)
    return error(Q17_INVALID, "missing planes/offsets or invalid row stride");
  constexpr uint64_t bound = uint64_t(std::numeric_limits<int64_t>::max()) / 4;
  if (uint64_t(c->a_stride) > bound / c->problem.m ||
      uint64_t(c->output_stride) > bound / c->problem.m)
    return error(Q17_INVALID, "strided address extent overflow");
  int a_alignment = c->problem.input_type == Q17_F32 ? 4 : 2;
  int c_alignment = c->problem.output_type == Q17_F32 ? 4 : 2;
  if (uintptr_t(c->a) % a_alignment || uintptr_t(c->output) % c_alignment ||
      uintptr_t(c->low) % 2 || uintptr_t(c->high) % 2 ||
      uintptr_t(c->offsets) % 4 || uintptr_t(c->scale) % 2 || uintptr_t(c->zero) % 2)
    return error(Q17_INVALID, "typed pointer alignment differs");
  if (c->config.engine == Q17_TMA_WGMMA &&
      (c->a_stride % 8 || uintptr_t(c->a) % 16 || uintptr_t(c->low) % 16 ||
       (c->high && uintptr_t(c->high) % 16) || uint64_t(c->a_stride) >= (uint64_t(1) << 39)))
    return error(Q17_UNSUPPORTED, "TMA requires 16-byte bases and row strides");
  return error(Q17_SUCCESS, "");
}
} // namespace quactlize::ppu17

extern "C" char const *q17_last_error_v1() {
  return quactlize::ppu17::last_error.c_str();
}

extern "C" int q17_canonical_arrangement_v1(int q, quactlize_ppu_placed_arrangement_v2 *out) {
  using namespace quactlize::ppu17;
  if (!out) return error(Q17_INVALID, "NULL arrangement output");
  *out = {};
  if (q == 8) *out = q8_kpack2::arrangement();
  else if (q == 12) *out = ppu_arrangements::q4_kpack4_transpose_v1();
  else if (q >= 10 && q <= 14) *out = ppu_arrangements::kquant_kpack_transpose_v1(q);
  else return error(Q17_UNSUPPORTED, "qtype is not Q2/Q3/Q4/Q5/Q6/Q8");
  return error(Q17_SUCCESS, "");
}

extern "C" int q17_sizes_query_v1(q17_problem_v1 const *p, q17_sizes_v1 *out) {
  using namespace quactlize::ppu17;
  if (!out) return error(Q17_INVALID, "NULL sizes output");
  *out = {};
  int status = validate_problem(p);
  if (status) return status;
  uint64_t count = uint64_t(p->experts) * p->n * p->k;
  out->low_bytes = count * p->arrangement.bits / 8;
  out->high_bytes = count * p->arrangement.high_bits / 8;
  // Exact byte-neutral metadata sizes, also checked against Format<Q>.
  int sb_bytes = p->qtype == 10 ? 20 : p->qtype == 11 ? 14 : p->qtype == 14 ? 18 : 16;
  out->units_bytes = p->qtype == 8 ? count / 32 * 2 : count / 256 * sb_bytes;
  out->scale_bytes = out->zero_bytes = count / p->arrangement.group_size * 2;
  return Q17_SUCCESS;
}

extern "C" int q17_config_count_v1(q17_problem_v1 const *p) {
  int status = quactlize::ppu17::validate_problem(p);
  return status ? -status : int(sizeof(quactlize::ppu17::configs) / sizeof(q17_config_v1));
}

extern "C" int q17_can_implement_v1(q17_problem_v1 const *p, q17_config_v1 const *c) {
  using namespace quactlize::ppu17;
  int status = validate_problem(p);
  if (status) return status;
  if (!c || c->version != 1 || c->struct_bytes != sizeof(*c))
    return error(Q17_INVALID, "config ABI differs");
  for (auto const &known : configs) {
    if (!same_config(known, *c)) continue;
    if (c->engine == Q17_TMA_WGMMA &&
        (p->input_type != p->compute_type || p->n % 8 ||
         int64_t(p->k) * p->experts / 2 > std::numeric_limits<int32_t>::max()))
      return error(Q17_UNSUPPORTED, "TMA A storage must match compute type; B row bytes must align to 16");
    if ((int64_t(p->max_rows) + c->tile_m - 1) / c->tile_m > 65535)
      return error(Q17_UNSUPPORTED, "row grid exceeds launch bounds");
    return error(Q17_SUCCESS, "");
  }
  return error(Q17_UNSUPPORTED, "config is not compiled in this backend");
}

extern "C" int q17_config_get_v1(q17_problem_v1 const *p, int index, q17_config_v1 *out) {
  using namespace quactlize::ppu17;
  if (!out) return error(Q17_INVALID, "NULL config output");
  *out = {};
  int count = q17_config_count_v1(p);
  if (count < 0) return -count;
  if (index < 0 || index >= count) return error(Q17_INVALID, "config index out of range");
  int status = q17_can_implement_v1(p, &configs[index]);
  if (!status) *out = configs[index];
  return status;
}
