"""Local constructive gate for the Q3_K/Q6_K affine-zero question.

The imported fixture builds the repository's CUDA stand-in, including the real
dense placement and packed-unit producers.  The external-plane formula is a
structural statement; the 8,192 elements are a bitwise witness, not an
exhaustive proof over fp16.  No PPU box or device result is part of this gate.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_q3_q6_external_zero_is_derived_and_packed_has_no_external_plane():
    root = Path(__file__).resolve().parent.parent
    if shutil.which("nvcc") is None:
        pytest.skip("the local producer gate needs nvcc; it does not need a runtime GPU")
    run = subprocess.run(
        ["bash", "dev/fold_derivation/run_l139_q36_zero_redundancy.sh", "--json"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads(run.stdout)
    assert result["backend"].startswith("ppu"), result
    assert result["claim"] == (
        "external-fp16-zero-is-structurally-derived; "
        "packed-unit-already-has-no-external-zero; physical-plane-removal-not-implemented"
    )
    assert result["scope"] == {
        "structural_formulas": True,
        "bitwise_witness_elements_per_arm": 8192,
        "sample_is_not_exhaustive": True,
        "packed_collective_kPackedZMul_proved": False,
    }
    assert [row["name"] for row in result["formats"]] == ["Q3_K", "Q6_K"]
    for row in result["formats"]:
        assert (row["scale_first_elements"], row["dense_elements"], row["packed_decode_elements"]) == (
            8192, 8192, 8192)
        assert (row["scale_first_bad"], row["dense_bad"], row["packed_decode_bad"]) == (0, 0, 0)
        assert row["official_max_block_relative_error"] < 1.0e-3
        assert row["wrong_bias_witnesses"] > 0
        assert row["packed_perturbation_witnesses"] == 1
        assert row["targeted_dense_actual_vs_staged_bad"] == 0
        if row["name"] == "Q6_K":
            # The separate control fixture makes every non-target group exact,
            # so this one witness cannot come from the main random fixture.
            assert row["targeted_dense_rounding_witnesses"] == 1
        else:
            assert row["targeted_dense_rounding_witnesses"] == 0


def test_q36_oracle_fails_closed_when_python_disables_assertions():
    root = Path(__file__).resolve().parent.parent
    run = subprocess.run(
        [sys.executable, "-O", "dev/fold_derivation/q36_zero_redundancy.py", "--json"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0
    assert "assertions are disabled" in run.stdout + run.stderr
