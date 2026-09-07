import copy
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import time
import tarfile

import numpy as np
import pytest

from tools import run_kpack_selected_gate as gate
from quactlize.runtime.candidates import PARENT_FIELDS
from quactlize.runtime.tuning import Request, digest, Tuner


def inputs():
    return json.loads(gate.MODEL.read_text()), json.loads(gate.BASE.read_text())


def test_plan_is_exactly_the_selected_union_not_candidate_pool():
    model, base = inputs()
    plan = gate.make_plan(model, base)
    assert len(plan["cases"]) == 90 and len(plan["parents"]) == 55
    assert len(set(gate.weight_key(c) for c in plan["cases"])) == 25
    assert plan["digest"] == digest({k: v for k, v in plan.items() if k != "digest"})
    for case in plan["cases"]:
        assert "candidates" not in case
        assert case["id"] == gate.request_of(case).exact_key
        assert case["selection"]["status"] == gate.RECENT
    model["entries"].pop()
    model["model_digest"] = digest(
        {k: v for k, v in model.items() if k != "model_digest"}
    )
    with pytest.raises(ValueError, match="denominator"):
        gate.make_plan(model, base)


def test_selected_keys_are_exact_subset_of_completed_module_gate():
    archive = gate.ROOT / "kpack-warmup-real-results.LJGzK2.tgz"
    if not archive.exists():
        pytest.skip("optional original module receipts not installed")
    model, base = inputs()
    assert gate.sha(archive) == model["authority"]["archive_sha256"]
    with tarfile.open(archive) as t:
        previous = json.load(t.extractfile("modules.json"))
    by_name = {r["parent"]["symbol"]: r for r in previous}
    plan = gate.make_plan(model, base)
    assert len(by_name) == 173 and len(plan["parents"]) == 55
    for parent in plan["parents"]:
        old = by_name[parent["symbol"]]
        assert old["parent"] == parent
        assert old["identity"] == model["authority"]["receipt"]["compiler"]
        assert old["key"] == gate.module_key(parent, old["identity"])


def test_cache_only_never_constructs_compiler_and_relocates_payload(
    tmp_path, monkeypatch
):
    model, base = inputs()
    plan = gate.make_plan(model, base)
    plan["parents"] = plan["parents"][:2]
    contract = model["authority"]["receipt"]["compiler"]

    def forbidden(*args, **kwargs):
        raise AssertionError("cache-only attempted compilation")

    monkeypatch.setattr(gate.Compiler, "__init__", forbidden)
    monkeypatch.setattr(gate.Compiler, "build", forbidden)
    monkeypatch.setattr(gate.Compiler, "compile_only", forbidden)
    records, missing = gate.cached_records(plan, contract, tmp_path)
    assert not records and missing == plan["parents"]
    parent = plan["parents"][0]
    key = gate.module_key(parent, contract)
    path = tmp_path / key
    path.mkdir()
    (path / "kernel.so").write_bytes(b"host-test-payload-not-an-ELF")
    value = dict(
        key=key,
        parent=parent,
        identity=contract,
        sha256=gate.sha(path / "kernel.so"),
        path="/wrong/path.so",
    )
    gate.save(path / "manifest.json", value)
    records, missing = gate.cached_records(plan, contract, tmp_path)
    assert len(records) == len(missing) == 1
    assert records[0]["path"] == str(path / "kernel.so")
    (path / "kernel.so").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity/payload"):
        gate.cached_records(plan, contract, tmp_path)


def test_default_cache_miss_stops_before_sdk_or_compiler(tmp_path, monkeypatch):
    model, base = inputs()
    plan = gate.make_plan(model, base)

    def forbidden(*args, **kwargs):
        raise AssertionError("cache miss attempted device work or compilation")

    monkeypatch.setattr(gate, "SDK", forbidden)
    monkeypatch.setattr(gate.Compiler, "__init__", forbidden)
    args = SimpleNamespace(
        plan_only=False,
        sdk=tmp_path / "no-sdk",
        cache=tmp_path / "empty",
        output=tmp_path,
        compile_missing=False,
        jobs=32,
    )
    with pytest.raises(ValueError, match="55 selected modules are missing"):
        gate.run_campaign(args, plan, model, argparse.ArgumentParser())
    assert len(json.loads((tmp_path / "missing-modules.json").read_text())) == 55


def small_case(route="fq-dense"):
    model, base = inputs()
    plan = gate.make_plan(model, base)
    original = next(c for c in plan["cases"] if c["request"]["route"] == route)
    r = Request(route, 12, 256, 512, 4, (1, 0, 3) if route.endswith("grouped") else ())
    selection = copy.deepcopy(original["selection"])
    selection["request"] = asdict(r) | {"rows": list(r.rows)}
    selection["grid"] = 0
    case = dict(id=r.exact_key, request=selection["request"], selection=selection)
    return case


class Memory:
    def __init__(self):
        self.cells, self.next = {}, 1

    def allocate(self, size):
        p, self.next = self.next, self.next + 1
        self.cells[p] = bytes(size)
        return p

    def upload(self, data):
        p = self.allocate(len(data))
        self.cells[p] = data
        return p

    def download(self, pointer, size):
        assert len(self.cells[pointer]) == size
        return self.cells[pointer]

    def fill(self, pointer, byte, size):
        self.cells[pointer] = bytes([byte]) * size

    def free(self, pointer):
        del self.cells[pointer]


class Backend:
    def __init__(self, case, plant=None):
        self.sdk, self.handles, self.buffers = Memory(), [], {}
        self.calls, self.plant = [], plant
        self.case = case
        chosen = case["selection"]
        c = chosen["config"]
        self.identity = dict(
            device="PPU-ZW810",
            compute_units=72,
            **{k: chosen["module_contract"][k] for k in ("sdk", "kernel")},
        )
        self.modules = {
            c["symbol"]: SimpleNamespace(
                record=dict(
                    parent={k: c[k] for k in PARENT_FIELDS},
                    identity=chosen["module_contract"],
                )
            )
        }

    def is_capturing(self):
        return False

    def arguments(self, request, tactic):
        return (
            None,
            None,
            SimpleNamespace(grid=self.case["selection"]["grid"]),
            None,
            None,
        )

    def prepare(self, request, tactic):
        handle = dict(
            buffers=dict(self.buffers), request=request, index=len(self.handles)
        )
        self.handles.append(handle)
        self.calls.append("prepare")
        return handle

    def run(self, handle):
        self.calls.append("run")
        negative = handle["buffers"]["low"] == 999
        if negative and self.plant == "negative-launch":
            raise RuntimeError("device launch failed")
        if negative and self.plant == "negative-missing-store":
            return
        if self.plant == "missing-store":
            return
        value = 0 if negative else 1
        if (self.plant == "negative-nan" and negative) or (
            self.plant == "positive-nan" and not negative
        ):
            value = float("nan")
        if self.plant == "direct-diff" and handle["index"] == 1:
            value = 1.0009765625  # within tolerance, but differs in FP16 bits
        if self.plant == "missed-negative":
            value = 1
        r = handle["request"]
        self.sdk.cells[handle["buffers"]["output"]] = np.full(
            (r.m, r.n), value, dtype="<f2"
        ).tobytes()

    def validation_sample(self, handle):
        self.calls.append("validation-timing")
        return 10.0 if self.plant != "bad-timing" else 0

    def measure(self, *args):
        raise AssertionError("online timing is forbidden")

    def synchronize(self):
        pass

    def close(self, handle):
        self.calls.append("close")


def execute(case, monkeypatch, plant=None):
    monkeypatch.setattr(gate, "select", lambda *args: case["selection"])

    def forbidden(*args, **kwargs):
        raise AssertionError("online tuner was reached")

    monkeypatch.setattr(Tuner, "warmup", forbidden)
    backend = Backend(case, plant)
    request = gate.request_of(case)
    weights = SimpleNamespace(
        activation=lambda r: (
            np.ones((r.m, r.k), dtype="<f2"),
            np.ones((r.m, r.n)),
            np.ones((r.m, r.n)),
        )
    )
    planes = dict(low=998, high=0, units=997, scale=996, zero=995, zero_low=999)
    try:
        value = gate.run_case(case, None, backend, weights, planes)
    finally:
        assert not backend.sdk.cells  # every owned allocation was released
    return value, backend


@pytest.mark.parametrize("route", ("fq-dense", "sf-dense", "fq-grouped", "sf-grouped"))
def test_real_dispatch_driver_without_device_or_online_tuner(route, monkeypatch):
    case = small_case(route)
    value, backend = execute(case, monkeypatch)
    assert value["errors"] == [0, 0, 0] and value["planted_error"] == 1
    assert value["direct_raw_equal"] and value["replay_raw_equal"]
    assert value["samples_us"] == [10, 10, 10]
    assert backend.calls.count("prepare") == backend.calls.count("close") == 3
    assert backend.calls.count("validation-timing") == 3


@pytest.mark.parametrize(
    "plant",
    (
        "positive-nan",
        "missing-store",
        "negative-launch",
        "negative-nan",
        "negative-missing-store",
        "missed-negative",
        "direct-diff",
        "bad-timing",
    ),
)
def test_bad_output_or_runtime_error_never_becomes_a_negative_pass(plant, monkeypatch):
    with pytest.raises(RuntimeError):
        execute(small_case(), monkeypatch, plant)


def test_resume_requires_complete_validated_receipt(tmp_path, monkeypatch):
    case = small_case()
    value, _ = execute(case, monkeypatch)
    result = dict(case=case, authority="a", **value)
    result["digest"] = digest(result)
    path = tmp_path / "cases" / f"{case['id']}.json"
    gate.save(path, result)
    assert gate.completed(tmp_path, case, "a") == result
    with pytest.raises(ValueError, match="stale"):
        gate.completed(tmp_path, case, "b")
    for field, bad in (
        ("errors", [0]),
        ("direct_raw_equal", False),
        ("grid", 1),
        ("negative_rejected", False),
        ("negative_raw_equal", False),
        ("samples_us", [10, 10]),
        ("validation_repeats", 1),
        ("timing_calls_during_dispatch", 1),
    ):
        modified = copy.deepcopy(result)
        modified[field] = bad
        modified["digest"] = digest(
            {k: v for k, v in modified.items() if k != "digest"}
        )
        gate.save(path, modified)
        with pytest.raises(ValueError, match="incomplete"):
            gate.completed(tmp_path, case, "a")


def test_incomplete_results_are_not_all_shape_pass(monkeypatch):
    model, base = inputs()
    plan = gate.make_plan(model, base)
    summary = gate.summarize(plan, [])
    assert summary["status"] == "INCOMPLETE" and summary["expected"] == 90
    case = small_case()
    value, _ = execute(case, monkeypatch)
    with pytest.raises(ValueError, match="foreign"):
        gate.summarize(plan, [dict(case=case, **value)])


def test_written_cases_without_clean_worker_exit_are_not_gate_pass(
    tmp_path, monkeypatch
):
    case = small_case()
    value, _ = execute(case, monkeypatch)
    result = dict(case=case, authority="a", **value)
    result["digest"] = digest(result)
    gate.save(tmp_path / "cases" / f"{case['id']}.json", result)
    plan = dict(cases=[case], parents=[case["selection"]["config"]])
    summary = gate.campaign_summary(plan, [result], tmp_path, "a", time.monotonic())
    assert summary["completed"] == 1 and summary["status"] == "INCOMPLETE"
    gate.save(
        tmp_path / "weights/0.json",
        dict(authority="a", cases=[case["id"]], process_rc=0),
    )
    summary = gate.campaign_summary(plan, [result], tmp_path, "a", time.monotonic())
    assert summary["status"] == "PASS" and summary["clean_workers"] == 1
