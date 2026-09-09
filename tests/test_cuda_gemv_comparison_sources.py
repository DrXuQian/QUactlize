"""Host-only checks for source extraction; these do not admit CUDA arithmetic."""

from pathlib import Path

import pytest

from dev.gemv_cuda.build import grid_schedule, replace_once
from dev.gemv_cuda.build_llama_reference import between

ROOT = Path(__file__).resolve().parents[1]


def test_reference_body_is_not_rewritten():
    body = "begin\n  float value = fmaf(a, b, c);\nend"
    assert between(body, "begin", "end") == "begin\n  float value = fmaf(a, b, c);\n"


@pytest.mark.parametrize(
    "source", ["begin", "end", "begin begin end", "begin end end", "end begin"]
)
def test_reference_boundary_drift_is_rejected(source):
    with pytest.raises(ValueError, match="boundaries"):
        between(source, "begin", "end")


def test_grid_experiment_keeps_the_exact_dot_body():
    original = (ROOT / "quactlize/execution/gemv.cu").read_text()
    changed = grid_schedule(original)
    begin = "template<class Reader>"
    end = "template<int Columns, int Warps, bool Pair = false>\n__global__"
    assert between(changed, begin, end) == between(original, begin, end)
    assert "dim3 const grid(c.n / Columns, split, c.rows)" in changed
    assert "if (c.rows > 65535) return QKG_INVALID" in changed
    assert "dim3 const reduce_grid((c.n + 127) / 128, c.rows)" in changed
    assert "int64_t const row = i / c.n, col = i % c.n" not in changed


@pytest.mark.parametrize("source", ["absent", "seam seam"])
def test_schedule_boundary_drift_is_rejected(source):
    with pytest.raises(ValueError, match="seam changed"):
        replace_once(source, "seam", "replacement")


def test_schedule_cannot_reapply_to_an_experiment():
    original = (ROOT / "quactlize/execution/gemv.cu").read_text()
    with pytest.raises(ValueError, match="seam changed"):
        grid_schedule(grid_schedule(original))
