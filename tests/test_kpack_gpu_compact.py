"""Host ownership proofs and bounded device-driver admission for GPU compact."""

from pathlib import Path
import subprocess
import copy
import json
from types import SimpleNamespace

import pytest

from tools import build_kpack_gpu_compact as builder
from tools import run_kpack_gpu_compact as runner
from quactlize.execution.native import Config
from quactlize.runtime.compiler import sha

ROOT = Path(__file__).resolve().parents[1]


def test_actual_directory_coordinates_and_bounded_launch(tmp_path):
    binary = tmp_path / "compact-host"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(ROOT / "quactlize/include"),
            "-I" + str(ROOT / "third_party/actlize/include"),
            str(ROOT / "tests/kpack_gpu_compact_host.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert "GPU_COMPACT_HOST PASS" in result.stdout


def test_plan_is_small_and_contains_exact_incumbents_and_split_controls():
    groups = builder.plan()
    manifest = dict(groups=groups)
    assert len(groups) == 9
    assert len(runner.jobs(manifest)) == 11
    assert sum(runner.expected_cells(j, manifest) for j in runner.jobs(manifest)) == 204
    assert {g["ordinary"]["qtype"] for g in groups} == set(range(10, 15))
    assert {g["ordinary"]["qtype"] for g in groups if g["model"]} == {12, 13}
    for g in groups:
        ordinary, persistent = g["ordinary"], g["persistent"]
        assert ordinary["tk"] == 256 and ordinary["ap"] == 0
        assert ordinary["tn"] == ordinary["dn"] == 64
        assert all(
            ordinary[k] == persistent[k]
            for k in ordinary
            if k not in ("symbol", "persistent")
        )
        if ordinary["route"] == "fq-grouped":
            assert ordinary["persistent"] == 0 and persistent["persistent"] == 1
    assert runner.split_counts(2048) == [1, 2, 4, 8]
    assert runner.split_counts(512) == [1, 2]


@pytest.mark.parametrize("q", [12, 13])
def test_simt_control_uses_the_existing_tuple_config_boundary(monkeypatch, q):
    seen = []
    monkeypatch.setattr(runner.C, "CDLL", lambda *a, **kw: object())
    monkeypatch.setattr(runner, "pair_bind", lambda lib: {})
    monkeypatch.setattr(runner, "IndexedWeights", lambda *a: object())

    def cell(args, sdk, functions, w, case, config, variant):
        # This is the real callee's construction, not a permissive mock.
        f = Config(*config)
        seen.append((variant, f.columns, f.warps, f.split))
        return dict(status="PASS")

    monkeypatch.setattr(runner, "simt_cell", cell)
    cells = []
    runner.run_simt(
        SimpleNamespace(bundle=Path("/unused")),
        None,
        dict(execution=dict(library="control.so")),
        q,
        cells,
    )
    assert len(cells) == 2
    assert seen == (
        [("scalar", 16, 8, 1), ("pair", 16, 8, 1)]
        if q == 12
        else [("scalar", 32, 8, 1), ("pair", 16, 2, 1)]
    )


def fake_bundle(root):
    groups = copy.deepcopy(builder.plan())
    modules = {}
    for g in groups:
        arms = ["ordinary", "persistent"] + (["baseline"] if g["model"] else [])
        for arm in arms:
            parent = g["ordinary" if arm == "baseline" else arm]
            key = ("old-" if arm == "baseline" else "new-") + parent["symbol"]
            path = root / (key + ".so")
            path.write_bytes(key.encode())
            modules[key] = dict(
                key=key, path=path.name, sha256=sha(path), parent=parent
            )
            g[arm + "_key"] = key
    execution = root / "execution.so"
    execution.write_bytes(b"unchanged SIMT control")
    return dict(
        schema=runner.SCHEMA,
        production_selection_changed=False,
        groups=groups,
        modules=list(modules.values()),
        execution=dict(library=execution.name, sha256=sha(execution)),
    )


@pytest.mark.parametrize(
    "plant",
    [
        None,
        "missing",
        "duplicate",
        "wrong-parent",
        "wrong-geometry",
        "extra-module",
        "payload",
        "escaping-path",
    ],
)
def test_bundle_manifest_rejects_missing_or_changed_authority(tmp_path, plant):
    manifest = fake_bundle(tmp_path)
    if plant == "missing":
        manifest["groups"].pop()
    elif plant == "duplicate":
        manifest["groups"][-1] = manifest["groups"][0]
    elif plant == "wrong-parent":
        manifest["groups"][0]["ordinary_key"] = manifest["groups"][1]["ordinary_key"]
    elif plant == "wrong-geometry":
        manifest["groups"][0]["ordinary"]["tn"] = 32
    elif plant == "extra-module":
        manifest["modules"].append({**manifest["modules"][0], "key": "unreferenced"})
    elif plant == "payload":
        (tmp_path / manifest["modules"][0]["path"]).write_bytes(b"changed")
    elif plant == "escaping-path":
        manifest["modules"][0]["path"] = str(Path(__file__).resolve())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    if plant:
        with pytest.raises((ValueError, KeyError)):
            runner.verify(tmp_path)
    else:
        assert len(runner.verify(tmp_path)["modules"]) == 20
