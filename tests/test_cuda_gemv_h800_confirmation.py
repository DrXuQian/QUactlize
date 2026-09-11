"""Audit frozen experiment receipts; no CUDA hardware needed here."""
import json
from pathlib import Path
import tarfile

import pytest

from dev.gemv_cuda.summarize_h800_confirmation import POLICY, read_records, summarize
from dev.gemv_cuda.build_h800_candidates import source

DATA = Path(__file__).resolve().parents[1] / "docs/measurements/q4_h800_opt_20260911"
CONFIRMATION = [DATA / f"{size}-confirmation.json" for size in ("small", "medium", "large")]
VALIDATION = [DATA / f"random-{size}-{seed}.json" for size in ("small", "medium", "large")
              for seed in (93711, 93719)]


def test_frozen_six_shapes_meet_both_control_bounds():
    report = summarize(CONFIRMATION, VALIDATION, DATA / "random-fixtures.json")
    assert report == json.loads((DATA / "final-verdict.json").read_text())
    assert report["verdict"] == "PASS"
    assert len(report["cells"]) == 12
    assert len(report["random_validation"]) == 24
    assert max(r["delta_pct"]["xplane"] for r in report["cells"]) < 1.89
    assert max(r["delta_pct"]["raw-reference"] for r in report["cells"]) < 4.41
    assert {tuple(r["shape"]) for r in report["cells"]} == set(POLICY)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "nan", "oracle", "median", "recipe", "rounds"])
def test_invalid_receipts_cannot_be_admitted(tmp_path, fault):
    data = json.loads(CONFIRMATION[0].read_text())
    name, _ = POLICY[(1, 512, 2048)]
    rows = data["cases"][0]["records"][name]
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = rows[0]
    elif fault == "nan":
        rows[0]["samples_us"][0] = float("nan")
    elif fault == "oracle":
        rows[0]["error"] = "0.006"
    elif fault == "median":
        rows[0]["median_us"] = "1.0"
    elif fault == "recipe":
        data["arms"][name]["recipes"] = [[1, 8, 1]]
    else:
        data["rounds"] = 1
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        read_records(path, 6)


def test_measured_is_not_automatically_a_performance_pass(tmp_path):
    data = json.loads(CONFIRMATION[0].read_text())
    name, _ = POLICY[(1, 512, 2048)]
    # Keep valid event medians and correctness, but make the candidate slower.
    for row in data["cases"][0]["records"][name]:
        row["samples_us"] = [10.0] * 15
        row["median_us"] = "10.0"
    path = tmp_path / "slow.json"
    path.write_text(json.dumps(data))
    rows, _ = read_records(path, 6)
    assert rows[0]["verdict"] == "OPEN"
    assert rows[1]["verdict"] == "PASS"


def test_duplicate_validation_does_not_hide_a_missing_shape():
    with pytest.raises(ValueError, match="random validation coverage"):
        summarize(CONFIRMATION, [VALIDATION[0], *VALIDATION[1:-1], VALIDATION[0]],
                  DATA / "random-fixtures.json")


@pytest.mark.parametrize("archive,build,arm", [
    ("final-evidence.tgz", "candidates-r18", "meta-static-global-rs-fast-bare-av"),
    ("final-evidence.tgz", "candidates-r22", "affine8-early-fast-bare-a4"),
    ("large-evidence.tgz", "candidates-r15", "affine4-early-fast-bare"),
])
def test_current_generator_reproduces_the_confirmed_kernel(archive, build, arm):
    with tarfile.open(DATA / archive) as evidence:
        member = evidence.getmember(f"{build}/{arm}/kernel.cu")
        assert member.isfile()
        measured = evidence.extractfile(member).read().decode()
    generated, _ = source(arm)
    remote_root = "/root/autodl-tmp/q4-ppu-replay.qC6s95/src"
    assert generated.replace(str(Path(__file__).resolve().parents[1]), remote_root) == measured
