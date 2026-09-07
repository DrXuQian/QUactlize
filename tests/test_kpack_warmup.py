import ctypes as C
import json
from pathlib import Path
import subprocess

import pytest

from quactlize.runtime.candidates import MeasuredCandidates
from quactlize.runtime.compiler import validate_parent
from quactlize.runtime.native import Call, Recipe, Resources, Identity
from quactlize.runtime.tuning import (
    Request,
    Tactic,
    Tuner,
    TuningCache,
    UnsupportedTactic,
)

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = dict(
    device="PPU-ZW810",
    compute_units=72,
    sdk="sdk",
    kernel="kernel",
    inventory="inventory",
)
A = Tactic("a", "GROUPED_NONPERSISTENT")
B = Tactic("b", "GROUPED_NONPERSISTENT")


class Backend:
    def __init__(self, times=None, invalid=(), error=None):
        self.identity = dict(IDENTITY)
        self.times = times or {A: 10, B: 8}
        self.invalid = set(invalid)
        self.error = error
        self.counts = dict(prepare=0, run=0, measure=0, check=0, close=0)
        self.capturing = False

    def is_capturing(self):
        return self.capturing

    def admissible(self, request, tactic):
        return tactic not in self.invalid

    def prepare(self, request, tactic):
        self.counts["prepare"] += 1
        if tactic in self.invalid:
            raise UnsupportedTactic()
        return tactic

    def check(self, h):
        self.counts["check"] += 1
        if self.error:
            raise self.error

    def run(self, h):
        self.counts["run"] += 1

    def measure(self, h, r):
        self.counts["measure"] += 1
        return self.times[h]

    def synchronize(self):
        pass

    def close(self, h):
        self.counts["close"] += 1


def request(rows=(2, 0, 3, 1)):
    return Request("fq-grouped", 12, 256, 512, sum(rows), rows)


def test_warmup_then_query_is_not_reprofiling():
    b = Backend()
    t = Tuner(TuningCache(IDENTITY))
    result = t.warmup(request(), [A, B], b)
    assert result["tactic"] == B and result["candidates"] == 2
    assert [o["us"] for o in result["observations"]] == [10, 8]
    assert all(len(o["samples_us"]) == 3 for o in result["observations"])
    assert result["best_observed_us"] == 8
    assert "observations" not in t.cache.get(request())
    counts = b.counts.copy()
    assert t.select(request(), b)["tactic"] == B
    assert t.warmup(request(), [A, B], b)["tactic"] == B
    assert b.counts == counts


def test_grouped_bucket_is_hint_not_exact_rows_or_fixed_grid():
    a = request((2, 0, 3, 1))
    other = request((3, 1, 2, 0))
    assert a.bucket == other.bucket and a.exact_key != other.exact_key
    b = Backend()
    t = Tuner(TuningCache(IDENTITY))
    t.warmup(a, [A, B], b)
    assert t.select(other, b)["status"] == "BUCKET_HINT"
    assert not t.select(other, b)["performance_bound"]
    assert "grid" not in t.select(other, b)["tactic"].__dict__


def test_bucket_preserves_tiny_dense_provider_domains():
    keys = {Request("fq-dense", 12, 256, 512, m).bucket for m in range(1, 10)}
    assert len(keys) == 9


def test_request_uses_canonical_metadata_superblock_quantum():
    for q in (10, 12, 13):
        Request("fq-dense", q, 256, 256, 1)
    for q in (11, 14):
        with pytest.raises(ValueError, match="unsupported"):
            Request("fq-dense", q, 256, 256, 1)
    with pytest.raises(ValueError, match="actual expert rows"):
        Request("fq-grouped", 12, 256, 512, 1, [1])


def test_backend_identity_must_match_cache_even_without_an_entry():
    b = Backend()
    b.identity["sdk"] = "other"
    t = Tuner(TuningCache(IDENTITY))
    with pytest.raises(ValueError, match="backend differs"):
        t.select(request(), b, fallback=A)
    with pytest.raises(ValueError, match="backend differs"):
        t.warmup(request(), [A, B], b)
    assert not any(b.counts.values())


def test_close_difference_retains_incumbent():
    b = Backend({A: 10, B: 9.6})
    t = Tuner(TuningCache(IDENTITY))
    assert t.warmup(request(), [A, B], b)["tactic"] == A
    # Percentage is relative to the best, not the slower incumbent.
    t = Tuner(TuningCache(IDENTITY))
    assert t.warmup(request(), [A, B], Backend({A: 10.52, B: 10}))["tactic"] == B


def test_safe_rejection_is_distinct_from_device_or_correctness_failure():
    b = Backend(invalid=(A,))
    t = Tuner(TuningCache(IDENTITY))
    assert t.warmup(request(), [A, B], b)["tactic"] == B
    b = Backend(error=RuntimeError("numeric failure"))
    cache = TuningCache(IDENTITY)
    with pytest.raises(RuntimeError, match="numeric failure"):
        Tuner(cache).warmup(request(), [A, B], b)
    assert not cache.entries and b.counts["prepare"] == 1 and b.counts["close"] == 1


@pytest.mark.parametrize("value", [0, float("nan"), float("inf"), -1])
def test_invalid_timings_never_become_fast_winners(value):
    cache = TuningCache(IDENTITY)
    with pytest.raises(RuntimeError, match="timing"):
        Tuner(cache).warmup(request(), [A], Backend({A: value}))
    assert not cache.entries


def test_soft_budget_stops_between_candidates(monkeypatch):
    clock = iter([0, 0.2, 0.2])
    monkeypatch.setattr("quactlize.runtime.tuning.time.monotonic", lambda: next(clock))
    b = Backend()
    result = Tuner(TuningCache(IDENTITY), budget_ms=100).warmup(request(), [A, B], b)
    assert result["candidates"] == 1 and result["budget_exhausted"]


def test_persistent_cache_roundtrip_and_foreign_or_corrupt_rejection(tmp_path):
    path = tmp_path / "cache.json"
    c = TuningCache(IDENTITY, path)
    Tuner(c).warmup(request(), [A, B], Backend())
    assert (
        Tuner(TuningCache(IDENTITY, path)).select(request(), Backend())["tactic"] == B
    )
    with pytest.raises(ValueError, match="stale"):
        TuningCache(IDENTITY | {"sdk": "other"}, path)
    d = json.loads(path.read_text())
    d["entries"] = {}
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="corrupt"):
        TuningCache(IDENTITY, path)


def test_cache_merge_preserves_other_process_entries(tmp_path):
    path = tmp_path / "cache.json"
    c1 = TuningCache(IDENTITY, path)
    c2 = TuningCache(IDENTITY, path)
    c1.put(request(), A, 10, 1, 2)
    r = request((20, 0, 3, 1))
    c2.put(r, B, 8, 1, 2)
    assert len(TuningCache(IDENTITY, path).entries) == 2


def test_graph_capture_declines_tuning():
    b = Backend()
    b.capturing = True
    with pytest.raises(ValueError, match="capture"):
        Tuner(TuningCache(IDENTITY)).warmup(request(), [A], b)
    assert not any(b.counts.values())


def test_unknown_requires_an_explicit_admitted_fallback():
    t = Tuner(TuningCache(IDENTITY))
    b = Backend(invalid=(B,))
    assert t.select(request(), b)["status"] == "FALLBACK_REQUIRED"
    assert t.select(request(), b, fallback=B)["status"] == "FALLBACK_REQUIRED"
    assert t.select(request(), b, fallback=A)["tactic"] == A


def test_measured_seed_shortlists_bounded_canonical_parents():
    seed = MeasuredCandidates(ROOT / "policies/kpack_zw810_runtime_v1.json")
    for q in range(10, 15):
        for r in ("fq-dense", "sf-dense", "fq-grouped", "sf-grouped"):
            p = (
                Request(r, q, 512, 3072, 6, (2, 0, 3, 1))
                if r.endswith("grouped")
                else Request(r, q, 1024, 5120, 48)
            )
            ts = seed.shortlist(p)
            assert ts and len(ts) <= 15 and len({t.parent for t in ts}) <= 5
            for parent in seed.parent_union(ts):
                assert parent["route"] == r and parent["qtype"] == q


def test_all_frozen_recipe_kinds_bind_and_grouped_grid_uses_actual_rows():
    from types import SimpleNamespace
    from quactlize.runtime.native import NativeBackend
    from quactlize.runtime.candidates import from_config

    data = json.loads((ROOT / "policies/kpack_zw810_runtime_v1.json").read_text())
    checked_configs = set()
    for entry in data["entries"]:
        if entry["config_id"] in checked_configs:
            continue
        checked_configs.add(entry["config_id"])
        c = data["configurations"][entry["config_id"]]
        key = entry["key"]
        rows = tuple(data["row_vectors"][entry["row_vector"]]) if key[1] >= 2 else ()
        req = Request(c["route"], c["qtype"], key[2], key[3], key[5], rows)

        def query(call, recipe, resource):
            resource._obj.occupancy = max(1, c["grid_b"])
            return 0

        backend = NativeBackend.__new__(NativeBackend)
        backend.identity = IDENTITY | {"ordinal": 0}
        backend.workspace, backend.workspace_bytes = 0, 0
        backend.buffers, backend.stream = {}, 0
        backend.modules = {
            c["symbol"]: SimpleNamespace(record={"parent": c}, query=query)
        }
        _, _, recipe, _, _ = backend.arguments(req, from_config(c))
        if "PERSISTENT" in c["algorithm"] and "NONPERSISTENT" not in c["algorithm"]:
            work = sum((m + c["tm"] - 1) // c["tm"] for m in (rows or (req.m,)))
            work *= (req.n + c["tn"] - 1) // c["tn"]
            capacity = 72 * c["grid_b"]
            waves = (work + capacity - 1) // capacity
            expected = (
                min(work, capacity)
                if c["grid_mode"] == "capacity"
                else (work + waves - 1) // waves
            )
            assert recipe.grid == expected
        else:
            assert recipe.grid == 0
    assert len(checked_configs) == len(data["configurations"])


def test_parent_source_injection_and_wrong_axes_rejected():
    p = dict(
        symbol="a",
        qtype=12,
        route="fq-dense",
        tm=8,
        tn=64,
        tk=64,
        wm=8,
        wn=16,
        stages=2,
        ap=0,
        dn=16,
        persistent=-1,
    )
    validate_parent(p)
    for change in (
        {"symbol": 'a"\n#error oops'},
        {"ap": 2},
        {"route": "xplane"},
        {"dn": 7},
        {"persistent": 1},
    ):
        with pytest.raises(ValueError):
            validate_parent(p | change)
    with pytest.raises(ValueError, match="transport"):
        validate_parent(p | {"qtype": 11})


@pytest.mark.parametrize("q", range(10, 15))
def test_gate_fixture_is_nonzero_official_gguf_and_roundtrips(q):
    import numpy as np
    from tools.run_kpack_warmup_gate import fixture

    r = Request("sf-dense", q, 256, 512, 1)
    b, g, d = fixture(q, r)
    assert np.isfinite(g).all() and np.any(g) and np.all(d > 0)
    assert np.isfinite(np.frombuffer(b["metadata"], dtype=np.float16)).all()
    assert b["high"] if q in (11, 13, 14) else not b["high"]


def test_compile_only_source_is_typed_and_has_no_benchmark_dependency():
    from quactlize.runtime.compiler import Compiler

    p = dict(
        symbol="a",
        qtype=12,
        route="fq-dense",
        tm=8,
        tn=64,
        tk=64,
        wm=8,
        wn=16,
        stages=2,
        ap=0,
        dn=16,
        persistent=-1,
    )
    code = Compiler.__new__(Compiler).source(p, "abc")
    assert "#define QK_QTYPE 12" in code and '#include "module.cuh"' in code
    assert "benchmark" not in code
    text = (ROOT / "quactlize/runtime/kernel_types.cuh").read_text()
    assert "benchmarks/" not in text and "discovery.hpp" not in text


def test_compile_input_change_cannot_publish_under_old_identity(tmp_path):
    from quactlize.runtime.compiler import Compiler

    c = Compiler.__new__(Compiler)
    f = tmp_path / "source"
    f.write_text("one")
    c.input_stats = {f: (f.stat().st_mtime_ns, f.stat().st_size)}
    c.check_inputs()
    f.write_text("changed")
    with pytest.raises(RuntimeError, match="changed"):
        c.check_inputs()


def test_foreign_sdk_is_rejected_before_loading_device_code(tmp_path, monkeypatch):
    import quactlize.runtime.native as native

    monkeypatch.setattr(native, "sdk_identity", lambda _: "current")
    monkeypatch.setattr(
        native, "SDK", lambda _: pytest.fail("must reject before dlopen")
    )
    with pytest.raises(ValueError, match="runtime SDK differs"):
        native.NativeBackend(tmp_path, [{"identity": {"sdk": "foreign"}}], {}, None)


def test_sdk_hashing_is_once_per_file_revision(tmp_path, monkeypatch):
    import quactlize.runtime.native as native

    files = ["bin/hgcc", "bin/hgobjdump"] + [f"lib/lib{n}.so" for n in native.LIBRARIES]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original")
    hashed = []
    original = native.sha

    def count(path):
        hashed.append(path)
        return original(path)

    monkeypatch.setattr(native, "sha", count)
    first = native.sdk_identity(tmp_path)
    assert native.sdk_identity(tmp_path) == first and len(hashed) == len(files)
    (tmp_path / files[0]).write_bytes(b"a different compiler")
    assert native.sdk_identity(tmp_path) != first and len(hashed) == 2 * len(files)


def test_corrupted_compile_receipt_parent_is_not_a_cache_hit(tmp_path):
    from quactlize.runtime.compiler import Compiler, sha
    from quactlize.runtime.tuning import digest
    from tools.run_kpack_warmup_gate import parents

    parent = parents([12], ["fq-dense"])[0]
    compiler = Compiler.__new__(Compiler)
    compiler.input_stats = {}
    compiler.identity = {"sdk": "sdk", "kernel": "kernel"}
    compiler.cache = tmp_path
    key = digest(
        dict(
            identity=compiler.identity,
            parent=parent,
            source=compiler.source(parent, ""),
        )
    )
    directory = tmp_path / key
    directory.mkdir()
    (directory / "kernel.so").write_bytes(b"unchanged payload")
    receipt = dict(
        key=key,
        identity=compiler.identity,
        parent=parent | {"tm": 256},
        sha256=sha(directory / "kernel.so"),
    )
    (directory / "manifest.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="identity/payload differs"):
        compiler.build(parent)


def test_c_abi_layout_matches_ctypes(tmp_path):
    code = tmp_path / "abi.cpp"
    exe = tmp_path / "abi"
    code.write_text(
        '#include <cstdio>\n#include <cstddef>\n#include "abi.h"\nint main(){\n'
        + "\n".join(
            f'printf("%zu\\n",sizeof({name}));'
            for name in (
                "qk_call_v1",
                "qk_recipe_v1",
                "qk_resources_v1",
                "qk_identity_v1",
            )
        )
        + "\n"
        + "\n".join(
            f'printf("%zu\\n",offsetof(qk_call_v1,{name}));'
            for name, _ in Call._fields_
        )
        + "\n}"
    )
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Werror",
            "-I",
            str(ROOT / "quactlize/runtime"),
            str(code),
            "-o",
            str(exe),
        ],
        check=True,
    )
    actual = list(map(int, subprocess.check_output([exe], text=True).split()))
    assert actual == [C.sizeof(t) for t in (Call, Recipe, Resources, Identity)] + [
        getattr(Call, n).offset for n, _ in Call._fields_
    ]
