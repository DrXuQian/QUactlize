"""Host-only contract for exact measured K-quant tactic selection."""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

import pytest


ROOT = pathlib.Path(__file__).parents[1]
INCLUDE = ROOT / "quactlize" / "include"


def test_kquant_measured_policy_generator_contract():
    result = subprocess.run(
        [sys.executable, "-B",
         str(ROOT / "tools/generate_fq_kquant_measured_policy.py"),
         "self-test"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.stdout
    assert "PASS deterministic generation" in result.stdout
    generated = (INCLUDE / "ppu_kquant_measured_policy_data.inc").read_text()
    assert "quactlize.ppu-kquant-dense-exact-policy.v1" in generated
    assert "compiled-default" in generated
    assert "QUACTLIZE_PPU_KQUANT_DENSE_MEASURED_FAMILIES" in generated
    assert "QUACTLIZE_PPU_KQUANT_GROUPED_MEASURED" not in generated


def test_kquant_measured_policy_is_exact_and_preserves_name_precedence(tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a C++17 host compiler is required")

    source = tmp_path / "kquant_measured_policy.cpp"
    binary = tmp_path / "kquant_measured_policy"
    source.write_text(
        r'''
#include <cassert>

#include "ppu_grouped_shipping_policy.hpp"
#include "ppu_kquant_measured_policy.hpp"

namespace measured = ppu_kquant_measured_policy;
namespace dense = ppu_dense_shipping;
namespace grouped = ppu_grouped_shipping;

int main() {
  static_assert(grouped::default_config() == grouped::ConfigId::Default);
  static_assert(grouped::minimum_tile_m() == 16);
  static_assert(grouped::minimum_tile_n() == 32);

  grouped::ConfigId grouped_id = grouped::ConfigId::Tall;
  assert(grouped::find_config(nullptr, grouped_id));
  assert(grouped_id == grouped::ConfigId::Default);
  assert(grouped::find_config("", grouped_id));
  assert(grouped_id == grouped::ConfigId::Default);
  assert(grouped::find_config("32x32:16x16:s3", grouped_id));
  assert(grouped_id == grouped::ConfigId::SmallSquare);
  grouped_id = grouped::ConfigId::Tall;
  assert(!grouped::find_config("32X32:16x16:s3", grouped_id));
  assert(grouped_id == grouped::ConfigId::Tall);
  assert(!grouped::find_config("stale-config", grouped_id));
  assert(grouped_id == grouped::ConfigId::Tall);

  // Q2_K M=1/N=256/K=3072 is an exact measured dense point.
  dense::ConfigId dense_id = dense::ConfigId::Tall;
  assert(measured::select_dense(10, 1, 256, 3072, dense_id));
  assert(dense_id == dense::ConfigId::SmallSquare);

  // A compressed interval must not turn adjacent, unmeasured dimensions into
  // observations.  Misses also leave the caller's output untouched.
  dense_id = dense::ConfigId::Tall;
  assert(!measured::select_dense(10, 3, 256, 3072, dense_id));
  assert(dense_id == dense::ConfigId::Tall);
  assert(!measured::select_dense(10, 1, 257, 3072, dense_id));
  assert(!measured::select_dense(10, 1, 256, 3073, dense_id));
  assert(!measured::select_dense(9, 1, 256, 3072, dense_id));
  assert(dense_id == dense::ConfigId::Tall);

  // Explicit names outrank measurement.  Unknown explicit names fail closed;
  // null/empty names use measurement and then the historical shape default.
  assert(measured::find_dense_config(
      "64x64:32x32:s3", 10, 1, 256, 3072, dense_id));
  assert(dense_id == dense::ConfigId::Default);
  dense_id = dense::ConfigId::Tall;
  assert(!measured::find_dense_config(
      "stale-config", 10, 1, 256, 3072, dense_id));
  assert(dense_id == dense::ConfigId::Tall);
  assert(measured::find_dense_config(
      nullptr, 10, 1, 256, 3072, dense_id));
  assert(dense_id == dense::ConfigId::SmallSquare);
  assert(measured::find_dense_config("", 10, 1, 256, 3072, dense_id));
  assert(dense_id == dense::ConfigId::SmallSquare);
  assert(measured::find_dense_config(
      nullptr, 10, 3, 256, 3072, dense_id));
  assert(dense_id == dense::ConfigId::ShortWideM8S3);
  assert(measured::find_dense_config(
      nullptr, 10, 9, 256, 3072, dense_id));
  assert(dense_id == dense::ConfigId::Default);

}
''')
    compile_result = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
         "-I", str(INCLUDE), str(source), "-o", str(binary)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert compile_result.returncode == 0, compile_result.stdout
    run_result = subprocess.run(
        [str(binary)], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert run_result.returncode == 0, run_result.stdout
