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
