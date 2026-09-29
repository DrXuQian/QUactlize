"""Host-only contract for loader admission before runtime M is known."""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest


ROOT = pathlib.Path(__file__).parents[1]
INCLUDE = ROOT / "quactlize" / "include"
BACKEND = ROOT / "quactlize" / "csrc" / "device" / "ppu_dense_backend.cu"


def _function_body(source: str, name: str) -> str:
    begin = source.index(name)
    opening = source.index("{", begin)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening:index + 1]
    raise AssertionError(f"unterminated function {name}")


@pytest.mark.parametrize("language,standard,compiler_names", [
    ("c", "c11", ("cc", "gcc", "clang")),
    ("c++", "c++17", ("c++", "g++", "clang++")),
])
def test_any_m_public_declarations_are_valid_c_abi(
        language, standard, compiler_names):
    compiler = next((shutil.which(name) for name in compiler_names
                     if shutil.which(name)), None)
    if compiler is None:
        pytest.skip(f"a host {language} compiler is required")
    source = r'''
#include "quactlize_ppu_config.h"

static int32_t (*dense_any_m)(
    int, int, int, quactlize_ppu_placed_arrangement_v2 const*) =
    &quactlize_ppu_dense_fully_quantized_any_m_valid_for_arrangement_v2;
static int32_t (*grouped_any_m)(
    int, int, int, int, quactlize_ppu_placed_arrangement_v2 const*) =
    &quactlize_ppu_grouped_fully_quantized_any_m_valid_for_arrangement_v2;

int main(void) { return dense_any_m == 0 || grouped_any_m == 0; }
'''
    result = subprocess.run(
        [compiler, f"-std={standard}", "-Wall", "-Wextra", "-Werror",
         "-fsyntax-only", "-I", str(INCLUDE), "-x", language, "-"],
        input=source, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.stdout


def test_any_m_exports_validate_the_null_selected_path_not_inventory_presence():
    source = BACKEND.read_text()
    dense = _function_body(
        source,
        "quactlize_ppu_dense_fully_quantized_any_m_valid_for_arrangement_v2")
    grouped = _function_body(
        source,
        "quactlize_ppu_grouped_fully_quantized_any_m_valid_for_arrangement_v2")

    assert "quactlize_ppu_dense_fully_quantized_selected_config_for_arrangement_v2" in dense
    assert "ppu_kquant_measured_policy::kMeasuredDynamicValues" in dense
    assert "ppu_dense_shipping::kDecodeDefault" in dense
    assert "ppu_dense_shipping::kLegacyDefault" in dense
    assert "ppu_q4_kpack4_shipping::kDecodeMaxM + 1" in dense
    assert "list_valid" not in dense

    assert "experts <= 0" in grouped
    assert "quactlize_ppu_grouped_fully_quantized_selected_config_for_arrangement_v2" in grouped
    assert "&selected, 1, n, k" in grouped
    assert "experts, 1" in grouped
    assert "list_valid" not in grouped

    for body in (dense, grouped):
        assert "QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1" in body
        assert "QUACTLIZE_PPU_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1" in body
        assert "QUACTLIZE_PPU_LAYOUT_XPLANE_V1" not in body


def test_dense_any_m_partition_is_complete_for_current_host_policies(tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++17 host compiler is required")
    source = tmp_path / "any_m_policy.cpp"
    binary = tmp_path / "any_m_policy"
    source.write_text(r'''
#include <array>
#include <cassert>

#include "ppu_grouped_shipping_policy.hpp"
#include "ppu_kquant_measured_policy.hpp"
#include "ppu_q4_kpack4_shipping_policy.hpp"

int main() {
  namespace measured = ppu_kquant_measured_policy;
  namespace q4 = ppu_q4_kpack4_shipping;

  constexpr std::array<int, 13> expected{
      1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
  static_assert(expected.size() ==
                sizeof(measured::kMeasuredDynamicValues) /
                    sizeof(measured::kMeasuredDynamicValues[0]));
  for (std::size_t i = 0; i < expected.size(); ++i) {
    assert(measured::kMeasuredDynamicValues[i] == expected[i]);
  }
  assert(!measured::measured_dynamic_value(3));
  assert(!measured::measured_dynamic_value(9));

  // Fixed N/K leaves decode values through the shared boundary and one
  // prefill region beyond it.
  for (int m = 2; m < q4::kDecodeMaxM; ++m) {
    assert(q4::default_config(m, 8192, 16384) ==
           q4::default_config(1, 8192, 16384));
  }
  assert(q4::default_config(q4::kDecodeMaxM, 8192, 16384) !=
         q4::default_config(1, 8192, 16384));
  for (int m : {q4::kDecodeMaxM + 2, 64, 512, 4096}) {
    assert(q4::default_config(m, 8192, 16384) ==
           q4::default_config(q4::kDecodeMaxM + 1, 8192, 16384));
  }

  static_assert(ppu_grouped_shipping::default_config() ==
                ppu_grouped_shipping::ConfigId::Default);
}
''')
    compile_result = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
         "-I", str(INCLUDE), str(source), "-o", str(binary)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert compile_result.returncode == 0, compile_result.stdout
    run_result = subprocess.run(
        [str(binary)], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    assert run_result.returncode == 0, run_result.stdout


def test_q4_default_keeps_valid_winners_and_rejects_invalid_split_partitions(tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a C++17 host compiler is required")
    source = r'''
#include <cassert>
#include <cstdio>
#include <initializer_list>
#include "ppu_q4_kpack4_shipping_policy.hpp"
namespace q4 = ppu_q4_kpack4_shipping;

bool valid(q4::Config const& c, int k) {
  auto p = cutlass::gemm::kernel::fixed_splitk::make_params(1, k/c.tile_k, c.split);
  return k%c.tile_k == 0 && p.is_valid() && int(p.k_tiles_per_split) >= c.stages-1;
}

q4::ConfigId previous(int m, int n, int k) {
  if (m > q4::kDecodeMaxM) return q4::ConfigId::PrefillS1;
  if (n <= 2048) return q4::ConfigId::DecodeN32S4;
  if (n >= 16384 || (m == q4::kDecodeMaxM && k >= 16384)) return q4::ConfigId::DecodeN64S1;
  return n >= 7168 ? q4::ConfigId::DecodeN128S4 : q4::ConfigId::DecodeN64S4;
}

int main() {
  int cells=0, repairs=0;
  for (int m : {1,2,3,4,5,6,7,8,9,32,128,4096})
  for (int n : {256,512,1024,2048,4096,7168,8192,16384,25600})
  for (int k : {256,512,768,1024,1280,1536,1792,2048,2304,3072,4096,5120,8192,16384,25600}) {
    auto before=previous(m,n,k), after=q4::default_config(m,n,k);
    assert(valid(q4::row(after),k));
    if (valid(q4::row(before),k)) assert(after == before);
    else { assert(after == q4::ConfigId::DecodeN64S1); ++repairs; }
    q4::ConfigId selected{};
    assert(q4::find_config(nullptr,m,n,k,selected) && selected == after);
    assert(q4::find_config("",m,n,k,selected) && selected == after);
    ++cells;
  }
  assert(!valid(q4::row(previous(1,512,512)),512));
  assert(q4::default_config(1,512,512) == q4::ConfigId::DecodeN64S1);
  assert(q4::default_config(1,256,1024) == q4::ConfigId::DecodeN64S1);
  assert(q4::default_config(1,512,2048) == q4::ConfigId::DecodeN32S4);
  q4::ConfigId explicit_choice{};
  auto name=q4::row(q4::ConfigId::DecodeN32S4).name;
  assert(q4::find_config(name,1,512,512,explicit_choice));
  assert(explicit_choice == q4::ConfigId::DecodeN32S4);
  assert(!valid(q4::row(explicit_choice),512));
  assert(!q4::find_config("not-a-config",1,512,512,explicit_choice));
  assert(repairs > 0);
  std::printf("Q4_SMALLK_POLICY PASS cells=%d repairs=%d explicit_unchanged=1\n",cells,repairs);
}
'''
    binary = tmp_path / "q4-smallk-policy"
    result = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
        "-I", str(INCLUDE), "-x", "c++", "-", "-o", str(binary)],
        input=source, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(binary)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Q4_SMALLK_POLICY PASS cells=1620" in result.stdout
