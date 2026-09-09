import ctypes as C
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.test_kpack_gpu_compact import fake_bundle
from tools import profile_kpack_gpu_compact as profile
from quactlize.runtime.compiler import sha


def test_four_exact_arms_keep_old_payload_and_select_measured_split(tmp_path):
    manifest = fake_bundle(tmp_path)
    for case in profile.CASES:
        old = profile.selection(manifest, case, "baseline")
        new = profile.selection(manifest, case, "compact")
        assert old["module"]["parent"] == new["module"]["parent"]
        assert old["module"]["key"] != new["module"]["key"]
        assert old["tile_m"] == new["tile_m"] == 8
        for choice in (old, new):
            assert choice["module"]["parent"]["tm"] == 8
            assert choice["module"]["parent"]["wm"] == 8
            assert "_tm8_" in choice["module"]["parent"]["symbol"]
            assert "-tm8-" in profile.report_name(choice)
        assert old["split"] == 1 and not old["directory"]
        assert new["split"] == (2 if case == "q4-up" else 1) and new["directory"]
        assert old["expected_components"] == ["metadata", "gemm"]
        assert new["expected_components"] == ["metadata", "directory", "gemm"] + (
            ["reducer"] if case == "q4-up" else []
        )
        assert new["mode"] == "device-only" and new["grid_b"] == 0


def test_profile_does_not_fall_back_to_tm16(tmp_path):
    manifest = fake_bundle(tmp_path)
    manifest["groups"] = [g for g in manifest["groups"] if not g["job"].endswith("tm8")]
    with pytest.raises(ValueError, match="TM8"):
        profile.selection(manifest, "q4-up", "compact")


@pytest.mark.parametrize("case", profile.CASES)
def test_profile_rejects_a_tm16_body_behind_a_tm8_group_label(tmp_path, case):
    manifest = fake_bundle(tmp_path)
    choice = profile.selection(manifest, case, "compact")
    choice["module"]["parent"]["tm"] = 16
    with pytest.raises(ValueError, match="TM8/WM8"):
        profile.selection(manifest, case, "compact")


def test_acu_command_profiles_nodes_only_inside_api_range():
    command = profile.acu_command(
        "acu", "report", "python", "sdk", "bundle", "receipt", "q4-up", "compact"
    )
    for name, value in {
        "--set": "full",
        "--profile-from-start": "no",
        "--graph-profiling": "node",
        "--replay-mode": "kernel",
        "--kill": "no",
        "--check-exit-code": "yes",
        "--cache-control": "all",
    }.items():
        assert command[command.index(name) + 1] == value
    assert "--launch-count" not in command and "--collect" not in command
    assert "--force-overwrite" not in command and "asys" not in command


@pytest.mark.parametrize("failure", [None, "body", "start", "stop"])
def test_profile_range_lifetime_and_failure_propagation(failure):
    lib = SimpleNamespace(
        hggcProfilerStart=Mock(return_value=41 if failure == "start" else 0),
        hggcProfilerStop=Mock(return_value=41 if failure == "stop" else 0),
    )

    def run():
        with profile.AcuRange(SimpleNamespace(lib=lib)):
            if failure == "body":
                raise RuntimeError("body failure")

    if failure:
        with pytest.raises(RuntimeError):
            run()
    else:
        run()
    lib.hggcProfilerStart.assert_called_once_with()
    assert (
        lib.hggcProfilerStart.argtypes == []
        and lib.hggcProfilerStart.restype == C.c_int
    )
    assert lib.hggcProfilerStop.call_count == (0 if failure == "start" else 1)


@pytest.mark.parametrize(
    "plant",
    [
        None,
        "no-report",
        "no-receipt",
        "wrong-choice",
        "wrong-manifest",
        "no-kernels",
        "bad-exit",
    ],
)
def test_collection_preserves_other_arms_and_does_not_accept_empty_reports(
    tmp_path, monkeypatch, plant
):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = fake_bundle(bundle)
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    acu = tmp_path / "acu"
    acu.write_text("test tool")
    args = SimpleNamespace(
        bundle=bundle, sdk=tmp_path, output=tmp_path / "results", acu=acu
    )
    calls = []

    class Process:
        def __init__(self, command, stdout, stderr):
            case = command[command.index("--case") + 1]
            arm = command[command.index("--arm") + 1]
            receipt = Path(command[command.index("--output") + 1])
            report = Path(command[command.index("--export") + 1] + ".acurep")
            self.bad = not calls
            calls.append((case, arm))
            choice = profile.selection(manifest, case, arm)
            data = dict(
                status="PASS",
                selection=choice,
                manifest_sha256=sha(bundle / "manifest.json"),
            )
            if self.bad and plant == "wrong-choice":
                data["selection"] = {}
            if self.bad and plant == "wrong-manifest":
                data["manifest_sha256"] = "wrong"
            if not (self.bad and plant == "no-report"):
                report.write_bytes(b"nonempty native report")
            if not (self.bad and plant == "no-receipt"):
                receipt.write_text(json.dumps(data))
            if self.bad and plant == "no-kernels":
                stdout.write("No kernels were profiled\n")

        def wait(self, timeout):
            return int(self.bad and plant == "bad-exit")

    monkeypatch.setattr(profile.subprocess, "Popen", Process)
    assert profile.collect(args) == int(plant is not None)
    result = json.loads((args.output / "summary.json").read_text())
    assert len(calls) == len(result["captures"]) == 4
    assert sum(c["status"] == "CAPTURED" for c in result["captures"]) == (
        4 if plant is None else 3
    )
    assert result["performance_admission"] is False
    assert len((args.output / "acu-index.tsv").read_text().splitlines()) == 5
    for row in result["captures"]:
        assert row["tile_m"] == 8
        if row["report"]:
            assert "-tm8-" in row["report"]
