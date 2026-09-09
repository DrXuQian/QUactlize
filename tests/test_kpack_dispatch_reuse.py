import copy

import pytest

from tools import build_kpack_dispatch as build
from tools import run_kpack_grouped_postops as postops


@pytest.fixture
def setup(monkeypatch):
    base = {
        "modules": [
            dict(
                key="old",
                parent=dict(symbol="old"),
                path="modules/old/kernel.so",
                identity=dict(sdk="sdk"),
            )
        ]
    }
    measured = {"groups": [], "modules": []}
    for q in (12, 13):
        key = f"q{q}"
        measured["groups"].append(
            dict(job=f"fq-q{q}-tm8-ordinary", candidate=key, baseline="baseline")
        )
        measured["modules"].append(
            dict(
                key=key,
                parent=dict(symbol=key),
                path=f"modules/{key}.so",
                identity=dict(sdk="sdk"),
            )
        )
    monkeypatch.setattr(build, "verify_native", lambda root: base)
    monkeypatch.setattr(postops, "verify", lambda root: (measured, {}))
    monkeypatch.setattr(build, "sdk_identity", lambda root: "sdk")
    return base, measured


def test_reuses_old_and_only_candidate_builds(tmp_path, setup):
    base, measured = setup
    before = copy.deepcopy(base)
    rows = build.reuse_records(tmp_path, tmp_path, [], tmp_path)
    assert {r["key"] for r in rows} == {"old", "q12", "q13"}
    assert base == before


def test_missing_parent_never_silently_compiles(tmp_path, setup):
    with pytest.raises(ValueError, match="absent"):
        build.reuse_records(tmp_path, tmp_path, [dict(symbol="unmeasured")], tmp_path)


def test_reused_sdk_must_match(tmp_path, setup):
    setup[0]["modules"][0]["identity"]["sdk"] = "another-sdk"
    with pytest.raises(ValueError, match="SDK differs"):
        build.reuse_records(tmp_path, tmp_path, [], tmp_path)


def test_same_parent_other_build_is_not_overwritten(tmp_path, setup):
    setup[0]["modules"][0]["parent"]["symbol"] = "q12"
    with pytest.raises(ValueError, match="outside its measured scope"):
        build.reuse_records(tmp_path, tmp_path, [], tmp_path)


def test_catalog_rejects_ambiguous_parent_versions():
    with pytest.raises(ValueError, match="multiple builds"):
        build.catalog(
            [dict(parent=dict(symbol="same")), dict(parent=dict(symbol="same"))]
        )
