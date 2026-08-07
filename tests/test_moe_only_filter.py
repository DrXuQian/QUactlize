"""MOE_ONLY's cheap shape gate and final row gate must agree on real row tags.

This is a host-only check. It compiles the same formatter/matcher header used by the device bench, so the planted
stage-bearing exact tag crosses both production gates without needing a PPU SDK or constructing any kernels.
"""
import pathlib
import shutil
import subprocess

import pytest


ROOT = pathlib.Path(__file__).resolve().parent.parent

PROBE = r'''
#include "moe_only_filter.hpp"
#include <cstdio>
#include <cstring>

static int fail(const char* why) {
  std::fprintf(stderr, "[moe-only-filter] FAIL: %s\n", why);
  return 1;
}

int main() {
  char shape[80], tag_s6[80], tag_s4[80];
  moe_only_filter::format_shape(shape, sizeof shape, "i4", 64, 128, 64, 64, 16);
  moe_only_filter::format_tag(tag_s6, sizeof tag_s6, "i4", 64, 128, 64, 64, 16, 6, 0, 0, false);
  moe_only_filter::format_tag(tag_s4, sizeof tag_s4, "i4", 64, 128, 64, 64, 16, 4, 0, 0, false);

  const char* exact = "i4 64x128:64 w64x16 s6 bc0->0";
  const char* documented = "i4 64x128:64 w64x16 s6";
  if (std::strcmp(shape, "i4 64x128:64 w64x16") != 0)
    return fail("the per-shape identity acquired a stage or bc row field");
  if (std::strcmp(tag_s6, exact) != 0)
    return fail("the formatted row tag no longer matches the bench's public filter syntax");

  const char* tags[] = {tag_s6, tag_s4};
  int exact_selected = 0;
  for (const char* tag : tags)
    if (moe_only_filter::candidate_selected(shape, tag, exact)) ++exact_selected;
  if (exact_selected != 1)
    return fail("a filter equal to one complete tag did not select exactly that row");
  if (!moe_only_filter::candidate_selected(shape, tag_s6, documented))
    return fail("the documented stage-bearing filter did not cross both gates");
  if (!moe_only_filter::candidate_selected(shape, tag_s6, "i4"))
    return fail("the documented loose format filter did not cross both gates");
  if (moe_only_filter::candidate_selected(shape, tag_s4, documented))
    return fail("a different stage passed the final row gate");

  // NEGATIVE CONTROL FOR THE EXACT REGRESSION. Before the fix, bc was appended to this outer string. The shape then
  // had a token the stage filter lacked, while the filter had s6 which the shape lacked, so both strstr directions
  // failed. If this control starts selecting, the fixture no longer proves it can detect the bug it names.
  const char* broken_shape = "i4 64x128:64 w64x16 bc0->0";
  if (moe_only_filter::candidate_selected(broken_shape, tag_s6, documented))
    return fail("the planted pre-fix shape unexpectedly passes; the regression control is inert");

  std::printf("[moe-only-filter] PASS -- exact tag selected=%d; stage-bearing and loose filters cross both gates\n",
              exact_selected);
  return 0;
}
'''


def test_moe_only_exact_tag_crosses_shape_and_row_gates(tmp_path):
    cxx = shutil.which("c++") or shutil.which("g++")
    if not cxx:
        pytest.skip("no host C++ compiler")
    src = tmp_path / "probe.cpp"
    exe = tmp_path / "probe"
    src.write_text(PROBE)
    built = subprocess.run(
        [cxx, "-std=c++17", "-I", str(ROOT / "benchmarks"), str(src), "-o", str(exe)],
        capture_output=True, text=True,
    )
    assert built.returncode == 0, f"moe_only_filter.hpp does not compile standalone:\n{built.stderr}"
    ran = subprocess.run([str(exe)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert "exact tag selected=1" in ran.stdout
