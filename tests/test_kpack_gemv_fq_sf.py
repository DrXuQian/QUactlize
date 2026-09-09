"""Host-only gate/recipe/receipt checks; not PPU numerical admission."""

import copy
import ctypes as C
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tools import run_kpack_gemv_fq_sf as gate
from dev.gemv_cuda.build import affine_source, grid_schedule


def complete_result():
    return dict(
        status="PASS",
        case="q4-up",
        authority={"test": "host-only"},
        first_prepass_us=10,
        prepass_proof={"status": "PASS"},
        prepass={"median_us": 3},
        screen=[
            dict(reader=r, config=list(c), error=0, samples_us=[1, 2, 3])
            for r in gate.READERS
            for c in gate.configs()
        ],
        winners={
            a: dict(
                status="PASS",
                error=0,
                correctness_checks=3,
                calls_per_graph=16,
                graph_elapsed_samples_us=[[1] * 11 for _ in range(4)],
            )
            for a in ("pair", "affine", "fq", "sf", "sf_with_prepass")
        },
    )


def test_exact_small_and_large_shapes():
    assert gate.CASES == {
        "q4-up": (12, 512, 2048, 256, 8, 1),
        "q5-down": (13, 2048, 512, 256, 8, 8),
        "q4-dense-matched": (12, 4096, 2048, 1, 1, 1),
        "q4-dense-wide": (12, 8192, 5120, 1, 1, 1),
        "q4-dense-long-k": (12, 5120, 25600, 1, 1, 1),
    }
    assert len(set(gate.configs())) == 24
    assert (16, 4, 4) in gate.configs() and (16, 4, 8) in gate.configs()
    assert set(gate.PROFILES) == {"pair", "affine", "fq", "sf", "prepass"}


def test_complete_receipt_is_reusable():
    r = complete_result()
    assert gate.case_complete(r, "q4-up", r["authority"])
    assert not gate.case_complete(r, "q5-down", r["authority"])
    assert not gate.case_complete(r, "q4-up", {})


@pytest.mark.parametrize(
    "plant",
    [
        "missing-screen",
        "duplicate-screen",
        "nan-screen",
        "bad-screen",
        "missing-arm",
        "missing-round",
        "nan-sample",
        "bad-output",
        "no-prepass",
        "zero-prepass",
        "wrong-count",
    ],
)
def test_incomplete_or_failed_result_is_not_admitted(plant):
    r = complete_result()
    if plant == "missing-screen":
        r["screen"].pop()
    if plant == "duplicate-screen":
        r["screen"][-1] = copy.deepcopy(r["screen"][0])
    if plant == "nan-screen":
        r["screen"][0]["samples_us"][0] = float("nan")
    if plant == "bad-screen":
        r["screen"][0]["error"] = 0.01
    if plant == "missing-arm":
        del r["winners"]["sf"]
    if plant == "missing-round":
        r["winners"]["affine"]["graph_elapsed_samples_us"].pop()
    if plant == "nan-sample":
        r["winners"]["fq"]["graph_elapsed_samples_us"][0][0] = float("nan")
    if plant == "bad-output":
        r["winners"]["pair"]["error"] = 0.01
    if plant == "no-prepass":
        del r["prepass_proof"]
    if plant == "zero-prepass":
        r["first_prepass_us"] = 0
    if plant == "wrong-count":
        r["winners"]["sf"]["correctness_checks"] = 1
    assert not gate.case_complete(r, "q4-up", r["authority"])


def test_all_ten_gemm_choices_are_in_existing_bundle(tmp_path):
    exe = tmp_path / "query"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-I" + str(gate.ROOT),
            str(gate.ROOT / "tools/kpack_native_policy.cpp"),
            "-o",
            str(exe),
        ],
        check=True,
    )
    queries = []
    for q, n, k, e, m, ch in gate.CASES.values():
        for route in ((2, 3) if e > 1 else (0, 1)):
            queries.append(f"{q} {route} {m} {n} {k} {e} 1")
    result = subprocess.check_output(
        [str(exe)], input="\n".join(queries) + "\n", text=True
    ).splitlines()
    manifest = json.loads(
        (gate.ROOT / "prebuilt/ppu0010/kpack-native-v1/manifest.json").read_text()
    )
    symbols = {r["parent"]["symbol"] for r in manifest["modules"]}
    assert len(result) == 10
    assert all(row.split()[0] in symbols for row in result)
    assert int(result[0].split()[12]) == 4
    assert int(result[2].split()[12]) == 1


def test_affine_reuses_tested_arithmetic_without_cuda_compatibility():
    old = (gate.ROOT / "quactlize/execution/gemv.cu").read_text()
    generated = grid_schedule(affine_source(old))
    assert "compiler_bridge" not in generated and "cuda_runtime" not in generated
    assert generated.count("QKG_CONCAT(kpack_q,QKG_QTYPE)") == 3
    assert "FP32" in generated
    assert "float4 const a" in generated
    assert "if (c.rows > 65535) return QKG_INVALID" in generated


def test_box_script_preserves_callers_shell():
    path = gate.ROOT / "tools/run_kpack_gemv_fq_sf_box.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    r = subprocess.run(
        [
            "bash",
            "-c",
            'PPU_SDK=/nonexistent bash "$1"; printf "CALLER_ALIVE\\n"',
            "test",
            str(path),
        ],
        text=True,
        capture_output=True,
    )
    assert r.returncode == 0 and "CALLER_ALIVE" in r.stdout


def test_failed_case_does_not_discard_other_cases_and_resume_is_selective(
    tmp_path, monkeypatch
):
    expected = {"test": "host-only"}
    monkeypatch.setattr(gate, "SDK", lambda _: None)
    monkeypatch.setattr(gate, "authority", lambda *_: expected)
    called = []

    def execute(command, log, label):
        case = command[command.index("--case") + 1]
        output = Path(command[command.index("--output") + 1])
        called.append(case)
        data = complete_result()
        data["case"] = case
        for row in data["winners"].values():
            row.update(median_us=1, selection={"split": 1})
        if case == "q4-up":
            data["status"] = "FAIL"
        output.write_text(json.dumps(data))
        return int(case == "q4-up")

    monkeypatch.setattr(gate, "execute", execute)
    args = SimpleNamespace(
        sdk=tmp_path,
        native_bundle=tmp_path,
        affine_bundle=tmp_path,
        output=tmp_path / "results",
        resume=False,
        skip_acu=True,
        acu=None,
        only_cases=["q4-up", "q5-down"],
    )
    assert not gate.collect(args)
    assert called == ["q4-up", "q5-down"]
    good = (args.output / "q5-down.json").read_bytes()
    args.resume = True
    assert not gate.collect(args)
    assert called == ["q4-up", "q5-down", "q4-up"]
    assert (args.output / "q5-down.json").read_bytes() == good
    assert len(list(args.output.glob("q4-up.failed.*.json"))) == 1
