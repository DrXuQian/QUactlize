"""The C++ grouped-row loader agrees with the Python router authority."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import fq_grouped_multi_router as router  # noqa: E402


SOURCE = r"""
#include <cstdio>
#include "kpack_grouped_fixture_rows.hpp"
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  kpack_grouped_fixture_rows::Rows rows;
  char why[128]{};
  if (!kpack_grouped_fixture_rows::load(argv[1], 256, rows, why, sizeof why)) {
    std::fprintf(stderr, "%s\n", why);
    return 3;
  }
  std::printf("%d %d %d %d 0x%016llx\n", rows.total, rows.max,
              rows.active, rows.zero,
              static_cast<unsigned long long>(rows.fnv64));
  return 0;
}
"""


@pytest.fixture(scope="module")
def loader(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("c++ is unavailable")
    root = tmp_path_factory.mktemp("grouped-rows-loader")
    source, binary = root / "probe.cpp", root / "probe"
    source.write_text(SOURCE)
    subprocess.run([
        compiler, "-std=c++17", "-O2", "-I", str(ROOT / "benchmarks"),
        str(source), "-o", str(binary),
    ], check=True, capture_output=True, text=True)
    return binary


@pytest.mark.parametrize("profile", sorted(router.profiles()))
def test_cpp_loader_matches_router_authority(
        loader: Path, tmp_path: Path, profile: str):
    authority = router.materialize()[profile]
    path = tmp_path / "rows.txt"
    path.write_text("\n".join(map(str, authority["rows"])) + "\n")
    output = subprocess.check_output([str(loader), str(path)], text=True).strip()
    total, maximum, active, zero, fnv = output.split()
    assert (int(total), int(maximum), int(active), int(zero), fnv) == (
        authority["total_rows"], authority["max_rows"],
        authority["active"], authority["zero"], authority["rows_hash"])


@pytest.mark.parametrize("contents", ["", "1\n", "-1\n" + "0\n" * 255,
                                       "0\n" * 256, "1\n" * 257])
def test_cpp_loader_rejects_malformed_histograms(
        loader: Path, tmp_path: Path, contents: str):
    path = tmp_path / "bad.txt"
    path.write_text(contents)
    result = subprocess.run([str(loader), str(path)], check=False,
                            capture_output=True, text=True)
    assert result.returncode != 0
