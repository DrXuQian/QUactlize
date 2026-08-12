#!/usr/bin/env python3
"""Pin the benchmark-only exact-CtaM seam to the unchanged adaptive launcher.

The positive/negative compiles instantiate the real launcher header with a
minimal CUDA kernel stub.  The source audit is independent of that compile:
it reconstructs the old adaptive CtaM order and rejects plants which reroute
either public wrapper through the other policy.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "quactlize/include/gemv_lowbit/gemv_launcher.hpp"
NVCC = shutil.which("nvcc")


STUB = r'''#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

namespace ppu_gemv {
enum class QuantOp : int {
  PerColScaleOnly, FinegrainedScaleOnly, FinegrainedScaleZero
};
enum class WLayout : int { Native, TileK };
struct WeightFormatRecord {};
struct WStrides { int64_t col{}, thr_major{}, thr_minor{}, iter{}; };
struct Params {
  void const *act{}, *act_scale{}, *weight{}, *weight_hi{}, *scales{}, *zeros{}, *bias{}, *record{};
  void *out{};
  float alpha{1};
  int m{}, n{}, k{}, groupsize{}, format{}, num_experts{}, max_rows{};
  bool is_bf16{};
  QuantOp quant{QuantOp::FinegrainedScaleZero};
  WLayout layout{WLayout::Native};
  int const* row_offsets{};
  int64_t w_bytes_per_expert{}, w_hi_bytes_per_expert{}, scale_elems_per_expert{};
};
struct KernelArgs {
  void const *act{}, *act_scale{}, *w_lo{}, *w_hi{}, *scales{}, *zeros{}, *bias{};
  void *out{};
  float alpha{};
  int n{}, k{}, rows{};
  int const* row_offsets{};
  int64_t w_lo_stride_e{}, w_hi_stride_e{}, scale_stride_e{};
  WStrides lo_s{}, hi_s{};
};
using gemv_stream_t = cudaStream_t;
inline int& gemv_refuse_counter() { static int value = 0; return value; }
inline void gemv_refuse(char const*) { ++gemv_refuse_counter(); }
inline int gemv_fail_count() { return gemv_refuse_counter(); }
constexpr bool is_two_plane(int) { return false; }
constexpr bool has_zero(QuantOp q) { return q == QuantOp::FinegrainedScaleZero; }
template <int GS, int StepK, int CtaK>
constexpr bool gs_step_ok() {
  if constexpr (GS == 0) return true;
  return (GS < StepK ? StepK % GS == 0 : GS % StepK == 0) && CtaK % GS == 0;
}
template <typename Details>
bool wfmt_matches(WeightFormatRecord const&, int, int, int, QuantOp, char const**) { return true; }
template <typename Details, int CtaM, int CtaN, int Chunk, int GS, QuantOp QOp,
          bool EnableActScale, bool EnableBias, bool ApplyAlphaInAdvance,
          bool PredicatedKTail, bool Grouped>
__global__ void gemv_kernel(KernelArgs) {
  static_assert(!Grouped || CtaM <= 4,
                "oracle: dense exact CtaM must not instantiate a grouped kernel");
}
}  // namespace ppu_gemv
'''


TU = r'''#define GEMV_GS_LIST(EMIT) EMIT(32)
#define GEMV_QUANT_LIST(EMIT, G) EMIT(QuantOp::FinegrainedScaleZero, G)
#define GEMV_ENABLE_BIAS 0
#include "gemv_lowbit/gemv_launcher.hpp"

namespace ppu_gemv {
struct StubA { static constexpr bool kIsBF16 = false; };
struct StubLayout { static WStrides strides(int, int) { return {}; } };
struct StubDetails {
  using ADetails = StubA;
  using LoLayout = StubLayout;
  using HiLayout = StubLayout;
  static constexpr int kFormat = 4;
  static constexpr WLayout kLayout = WLayout::Native;
  static constexpr int kStepK = 16;  // per-thread K advance, never a split-K factor
  static constexpr int kCtaK = 2048;
  static constexpr int kThreads = 128;
};

void instantiate(Params const& p, gemv_stream_t s) {
  (void)launch_gemv<StubDetails, 8, 2>(p, s);
#if EXACT_CTAM >= 0
  (void)launch_gemv_exact_ctam<StubDetails, bool(EXACT_GROUPED), EXACT_CTAM, 8, 2>(p, s);
#endif
}
}  // namespace ppu_gemv

#if RUN_RUNTIME
int main() {
  ppu_gemv::Params p{};
  // Deliberately present the route opposite to the compiled route.  This
  // must refuse before either dense or grouped device type can launch.
  p.num_experts = EXACT_GROUPED ? 0 : 1;
  bool const launched = ppu_gemv::launch_gemv_exact_ctam<
      ppu_gemv::StubDetails, bool(EXACT_GROUPED), EXACT_CTAM, 8, 2>(p, nullptr);
  bool const refused = !launched && ppu_gemv::gemv_fail_count() == 1;
  std::printf("RUNTIME_%s count=%d\n", refused ? "REFUSED" : "ESCAPED",
              ppu_gemv::gemv_fail_count());
  return refused ? 0 : 1;
}
#endif
'''


def function_region(source: str, name: str) -> str:
    start = source.index(name)
    brace = source.index("{", start)
    depth = 0
    for i in range(brace, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    raise RuntimeError(f"unterminated function {name}")


def audit(source: str) -> list[str]:
    bad: list[str] = []
    dispatch = function_region(source, "bool gemv_dispatch_ctam(")
    calls = [int(x) for x in re.findall(r"GEMV_TRY_CTAM\((\d+)\)", dispatch)]
    if calls != list(range(1, 16)):
        bad.append(f"adaptive order={calls}, want 1..15")

    # Reconstruct the historical first-covering-specialization policy.  The
    # grouped ceiling is four; dense retains all fifteen.
    for grouped, ceiling in ((False, 15), (True, 4)):
        legal = [x for x in calls if x <= ceiling]
        for rows in range(1, 65):
            got = next((x for x in legal if rows <= x or x == ceiling), None)
            want = min(rows, ceiling)
            if got != want:
                bad.append(f"adaptive grouped={grouped} rows={rows}: got={got} want={want}")
                break

    expected = (
        "gemv_exec<Details, CtaM, CtaN, Chunk, GS, QOp, EnableBias, Grouped>",
        "CtaM <= (Grouped ? GEMV_GROUPED_CTAM_MAX : GEMV_CTAM_MAX)",
        "runtime Params route disagrees with benchmark exact-CtaM route",
        "gemv_config_invalid_reason<Details, CtaN>(p)",
        "wfmt_matches<Details>",
    )
    for needle in expected:
        if needle not in source:
            bad.append(f"missing seam {needle!r}")

    default = function_region(source, "bool launch_gemv(Params const&")
    exact = function_region(source, "bool launch_gemv_exact_ctam(")
    if "return gemv_dispatch_quant<Details, CtaN, Chunk>(p, args, rows_max, s);" not in default:
        bad.append("default wrapper no longer uses the historical adaptive dispatch")
    if "exact" in default.lower() or "Grouped" in default:
        bad.append("default wrapper acquired benchmark-route/exact policy")
    if "gemv_dispatch_quant_exact<Details, CtaM, CtaN, Chunk, Grouped>" not in exact:
        bad.append("exact wrapper no longer forwards route and CtaM")
    if "static_assert(gemv_exact_ctam_supported_v<CtaM, Grouped>" not in exact:
        bad.append("exact wrapper lost compile-time fail-close")
    if "if ((p.num_experts > 0) != Grouped)" not in exact:
        bad.append("exact wrapper lost runtime route fail-close")

    # The benchmark seam is additive.  Pin the complete pre-existing adaptive
    # chain against HEAD, rather than claiming "unchanged" from a few tokens.
    head = subprocess.run(
        ["git", "show", "HEAD:quactlize/include/gemv_lowbit/gemv_launcher.hpp"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if head.returncode:
        bad.append("cannot read HEAD launcher for default-path identity")
    else:
        names = (
            "bool gemv_dispatch_ctam(", "bool gemv_dispatch_bias(",
            "bool gemv_dispatch_grouped(", "bool gemv_dispatch_quant(",
            "bool launch_gemv(Params const&",
        )
        current_default = "\n".join(function_region(source, name) for name in names)
        head_default = "\n".join(function_region(head.stdout, name) for name in names)
        if current_default != head_default:
            bad.append("shipping adaptive function bodies differ from HEAD")
    return bad


def compile_header(source: str, exact: int, grouped: bool = False,
                   run_runtime: bool = False) -> subprocess.CompletedProcess[str]:
    assert NVCC is not None
    with tempfile.TemporaryDirectory(prefix="qz-gemv-exact-ctam-") as td:
        root = Path(td)
        inc = root / "gemv_lowbit"
        inc.mkdir()
        (inc / "gemv_kernel.hpp").write_text(STUB)
        (inc / "gemv_launcher.hpp").write_text(source)
        tu = root / "probe.cu"
        tu.write_text(TU)
        out = root / ("probe" if run_runtime else "probe.o")
        command = [NVCC, "-std=c++17", "-I", str(root), f"-DEXACT_CTAM={exact}",
                   f"-DEXACT_GROUPED={int(grouped)}",
                   f"-DRUN_RUNTIME={int(run_runtime)}"]
        if not run_runtime:
            command.append("-c")
        command += [str(tu), "-o", str(out)]
        build = subprocess.run(
            command,
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if build.returncode or not run_runtime:
            return build
        return subprocess.run([str(out)], cwd=ROOT, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)


def planted(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"plant seam count={source.count(old)} for {old!r}")
    return source.replace(old, new, 1)


def main() -> int:
    if not LAUNCHER.is_file():
        print(f"[gemv-exact-ctam] FAIL missing {LAUNCHER}")
        return 1
    if NVCC is None:
        print("[gemv-exact-ctam] FAIL nvcc is required for the isolated real-header compile")
        return 1
    source = LAUNCHER.read_text()
    bad = audit(source)
    if bad:
        print("[gemv-exact-ctam] FAIL " + "; ".join(bad))
        return 1

    # Default plus the full route-aware exact domains must compile.  Adjacent
    # out-of-domain values must fail at the real static_assert.
    for grouped, exacts in ((False, (-1, *range(1, 16))), (True, range(1, 5))):
      for exact in exacts:
        run = compile_header(source, exact, grouped)
        if run.returncode:
            print(f"[gemv-exact-ctam] FAIL positive grouped={grouped} exact={exact}:\n"
                  + run.stdout[-2400:])
            return 1
    for grouped, exact in ((False, 0), (False, 16), (True, 0), (True, 5)):
        run = compile_header(source, exact, grouped)
        if (run.returncode == 0 or
                "benchmark exact CtaM is outside the compiled route's range" not in run.stdout):
            print(f"[gemv-exact-ctam] FAIL negative grouped={grouped} exact={exact} "
                  "did not fail at range guard:\n"
                  + run.stdout[-2400:])
            return 1

    for grouped, exact in ((False, 15), (True, 4)):
        runtime = compile_header(source, exact, grouped, run_runtime=True)
        if runtime.returncode or "RUNTIME_REFUSED count=1" not in runtime.stdout:
            print(f"[gemv-exact-ctam] FAIL route mismatch grouped={grouped} escaped:\n"
                  + runtime.stdout[-2400:])
            return 1

    # Independent source plants prove that the audit distinguishes the old
    # adaptive policy from exact dispatch; merely compiling both is not enough.
    plants = (
        ("  GEMV_TRY_CTAM(1)\n", "  GEMV_TRY_CTAM(2)\n", "adaptive-order"),
        ("return gemv_dispatch_quant<Details, CtaN, Chunk>(p, args, rows_max, s);",
         "return gemv_dispatch_quant_exact<Details, 4, CtaN, Chunk, false>(p, args, rows_max, s);",
         "default-rerouted"),
        ("gemv_dispatch_quant_exact<Details, CtaM, CtaN, Chunk, Grouped>",
         "gemv_dispatch_quant_exact<Details, 1, CtaN, Chunk, false>", "exact-hardcoded"),
        ("if ((p.num_experts > 0) != Grouped)",
         "if (false)", "runtime-route-fail-close"),
    )
    for old, new, label in plants:
        mutation = planted(source, old, new)
        if not audit(mutation):
            print(f"[gemv-exact-ctam] FAIL source plant {label} stayed green")
            return 1

    print("[gemv-exact-ctam] PASS default=byte-unchanged adaptive dense1..15/grouped1..4; "
          "exact dense1..15 + grouped1..4 compiled; dense0/16 + grouped0/5 static-red; "
          "both runtime route mismatches refused; "
          "four source plants red")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
