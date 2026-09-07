import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import calibrate_kpack_heuristic as calibrate
import generate_kpack_heuristic_header as header
import generate_kpack_runtime_header as base_header
import kpack_heuristic as h
import kpack_policy as common
import kpack_runtime_policy as rt
from test_kpack_runtime_policy import context, data, freeze, observation, parent
from quactlize.runtime.candidates import PARENT_FIELDS
from quactlize.runtime.dispatch import prepare_selected
from quactlize.runtime.tuning import Request, UnsupportedTactic


def ctx(m=8, route="fq-dense", rows=None):
    value = context(m, route, rows)
    value["problem"]["n"] = 256
    if route.endswith("dense"):
        value["rows"] = []
    return value


def fixture(route="fq-dense", contexts=None, ap=0):
    p = parent(route=route)
    p["ap"] = ap
    if route == "fq-grouped":
        p["persistent"] = 1
    contexts = contexts or [
        ctx(1 if ap else 8, route, [8, 8, 8, 8] if route.endswith("grouped") else None)
    ]
    observations = []
    for i, c in enumerate(contexts):
        oracle = copy.deepcopy(c)
        if route.endswith("dense"):
            oracle["rows"] = [oracle["problem"]["m"]]
        observations.append(observation(oracle, {"a": p}, {"a": (0, 0)}, name=str(i)))
    base, _ = freeze.freeze(data({"a": p}, observations))
    model = dict(
        schema=h.SCHEMA,
        base_policy_digest=base["policy_digest"],
        required_binding=dict(
            device="PPU-ZW810",
            compute_units=72,
            kernel="module-kernel",
            sdk="module-sdk",
        ),
        configurations={},
        entries=[],
        authority=dict(
            receipt=dict(compiler=dict(kernel="module-kernel", sdk="module-sdk"))
        ),
    )
    seal(model)
    return model, base


def seal(model):
    model["model_digest"] = common.digest(
        {k: v for k, v in model.items() if k != "model_digest"}
    )


def query(model, base, c, *, allow=True, **changes):
    binding = dict(
        device_name="PPU-ZW810",
        compute_units=72,
        kernel_source=model["required_binding"]["kernel"],
        sdk_digest=model["required_binding"]["sdk"],
        mapping_id=common.mapping(c["problem"]["qtype"]),
    )
    binding.update(changes)
    return h.Selector(model, base).select(
        c["route"],
        c["problem"],
        c["rows"] if c["route"].endswith("grouped") else None,
        allow_prediction=allow,
        **binding,
    )


def test_exact_is_single_choice_and_not_a_current_performance_bound():
    model, base = fixture()
    r = query(model, base, ctx(), allow=False)
    assert r["status"] == h.HISTORICAL
    assert r["config_id"] == base["entries"][0]["config_id"]
    assert r["runtime_admission_required"] and not r["performance_bound"]
    assert not {"candidates", "measured_us", "timing_cache"} & r.keys()


def test_transfer_is_opt_in_and_does_not_cross_weight_families():
    model, base = fixture()
    assert query(model, base, ctx(9), allow=False)["status"] == "FALLBACK_REQUIRED"
    r = query(model, base, ctx(9))
    assert r["status"] == h.PREDICTED and r["numerical_validation_required"]
    assert not r["performance_bound"] and "recent_regret_pct" not in r
    other = ctx(9)
    other["problem"]["n"] = 512
    assert query(model, base, other)["status"] == "FALLBACK_REQUIRED"


@pytest.mark.parametrize(
    "change",
    [
        {"kernel_source": "wrong"},
        {"sdk_digest": "wrong"},
        {"compute_units": 1},
        {"device_name": "other"},
        {"mapping_id": "0x0"},
    ],
)
def test_binding_failure_never_becomes_a_prediction(change):
    model, base = fixture()
    assert query(model, base, ctx(9), **change)["status"] == "BINDING_MISMATCH"


def test_ap1_and_tm8_eligibility_boundaries():
    model, base = fixture(ap=1)
    assert query(model, base, ctx(2))["status"] == "FALLBACK_REQUIRED"
    model, base = fixture("sf-dense", [ctx(4, "sf-dense")])
    assert query(model, base, ctx(8, "sf-dense"))["status"] == "FALLBACK_REQUIRED"
    model, base = fixture()
    assert query(model, base, ctx(65))["status"] == "FALLBACK_REQUIRED"


def test_grouped_uses_actual_tile_sum_not_profile_or_fixed_grid():
    model, base = fixture("fq-grouped")
    cid = base["entries"][0]["config_id"]
    c = base["configurations"][cid]
    assert c["grid_mode"] in ("capacity", "balanced")
    a = ctx(route="fq-grouped", rows=[16, 8, 8, 0])
    b = ctx(route="fq-grouped", rows=[16, 15, 1, 0])
    assert h.profile(a["route"], a["problem"], a["rows"]) == h.profile(
        b["route"], b["problem"], b["rows"]
    )
    ra, rb = query(model, base, a), query(model, base, b)
    assert ra["status"] == rb["status"] == h.PREDICTED
    assert ra["config_id"] == rb["config_id"] and (ra["grid"], rb["grid"]) == (16, 20)


def test_nonpersistent_grid_uses_module_abi_not_old_log_convention():
    model, base = fixture("sf-dense", [ctx(4, "sf-dense")])
    cid = base["entries"][0]["config_id"]
    assert base["configurations"][cid]["grid_mode"] == "ordinary"
    assert query(model, base, ctx(4, "sf-dense"))["grid"] == 0


def test_recent_correction_and_digest_guards():
    model, base = fixture()
    c = ctx()
    cid = base["entries"][0]["config_id"]
    model["entries"] = [
        dict(
            context=c,
            key=rt.query_key(c["route"], c["problem"]),
            config_id=cid,
            regret_pct=3.0,
            spread_pct=1.0,
        )
    ]
    seal(model)
    r = query(model, base, c, allow=False)
    assert r["status"] == h.RECENT and r["recent_regret_pct"] == 3
    damaged = copy.deepcopy(model)
    damaged["entries"][0]["regret_pct"] = 0
    with pytest.raises(ValueError, match="digest"):
        h.Selector(damaged, base)
    damaged = copy.deepcopy(model)
    damaged["base_policy_digest"] = "different"
    seal(damaged)
    with pytest.raises(ValueError, match="base policy"):
        h.Selector(damaged, base)


def test_prepare_dispatch_binds_one_parent_without_any_timing():
    model, base = fixture()
    selection = query(model, base, ctx(), allow=False)
    c = selection["config"]
    request = Request("fq-dense", 12, 256, 512, 8)
    calls = []
    record = dict(
        parent={k: c[k] for k in PARENT_FIELDS}, identity=selection["module_contract"]
    )
    backend = SimpleNamespace(
        modules={c["symbol"]: SimpleNamespace(record=record)},
        identity=dict(
            device="PPU-ZW810",
            compute_units=72,
            sdk="module-sdk",
            kernel="module-kernel",
        ),
        is_capturing=lambda: False,
        prepare=lambda request, tactic: calls.append((request, tactic)) or "handle",
    )
    assert prepare_selected(backend, request, selection) == "handle"
    assert len(calls) == 1 and calls[0][1].parent == c["symbol"]
    # The backend has no measure/check/tuner/compiler methods: dispatch must
    # not reach any of them, even when a module or prediction is unavailable.
    for plant in ("request", "compiler", "parent", "device", "capture", "missing"):
        bad_backend = copy.deepcopy(backend)
        bad_request = request
        if plant == "request":
            bad_request = Request("fq-dense", 12, 256, 512, 9)
        elif plant == "compiler":
            bad_backend.modules[c["symbol"]].record["identity"]["flags"] = ["wrong"]
        elif plant == "parent":
            bad_backend.modules[c["symbol"]].record["parent"]["dn"] = 32
        elif plant == "device":
            bad_backend.identity["compute_units"] = 1
        elif plant == "capture":
            bad_backend.is_capturing = lambda: True
        else:
            bad_backend.modules = {}
        with pytest.raises(UnsupportedTactic):
            prepare_selected(bad_backend, bad_request, selection)
    assert len(calls) == 1
    prediction = query(model, base, ctx(9))
    with pytest.raises(UnsupportedTactic):
        prepare_selected(backend, Request("fq-dense", 12, 256, 512, 9), prediction)


CPP = r"""
#include "heuristic.hpp"
#include <iostream>
#include <vector>
namespace h = quactlize_kpack_heuristic_v1;
int main() {
  h::Query q; int route, count, allow, bad;
  while (std::cin >> route >> q.qtype >> q.n >> q.k >> q.group_size >> q.m
         >> q.experts >> q.total_rows >> q.max_rows >> count >> allow >> bad) {
    std::vector<int> rows(count); for (auto& r : rows) std::cin >> r;
    q.route = static_cast<quactlize_kpack_runtime_v1::Route>(route);
    q.rows = count ? rows.data() : nullptr; q.rows_count = count;
    q.device_name = "PPU-ZW810"; q.compute_units = 72;
    q.mapping_id = q.qtype == 12 ? 0x51344b5034540001ULL : 0x514b504b54000001ULL;
    q.kernel_source = bad ? "wrong" : h::kKernelSource; q.sdk_digest = h::kSdkDigest;
    auto r = h::select(q, allow);
    std::cout << static_cast<int>(r.status) << ' ' << (r.config ? r.config->id : "NONE") << ' ' << r.grid << '\n';
  }
}
"""


def cpp_parity(tmp_path, model, base, contexts):
    (tmp_path / "base.hpp").write_text(base_header.generate(base))
    (tmp_path / "heuristic.hpp").write_text(header.generate(model, base, "base.hpp"))
    (tmp_path / "main.cpp").write_text(CPP)
    build = subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(tmp_path / "main.cpp"),
            "-o",
            str(tmp_path / "query"),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    lines, expected = [], []
    selector = h.Selector(model, base)
    codes = {
        "INVALID_QUERY": 0,
        "BINDING_MISMATCH": 1,
        "FALLBACK_REQUIRED": 2,
        h.RECENT: 3,
        h.HISTORICAL: 4,
        h.PREDICTED: 5,
    }
    b = model["required_binding"]
    for c in contexts:
        p, route = c["problem"], c["route"]
        rows = c["rows"] if route.endswith("grouped") else []
        for allow, bad in ((False, False), (True, False), (True, True)):
            values = [
                rt.ROUTES.index(route),
                p["qtype"],
                p["n"],
                p["k"],
                p["group_size"],
                p.get("m", 0),
                p.get("experts", 1),
                p.get("total_rows", 0),
                p.get("max_rows", 0),
                len(rows),
                int(allow),
                int(bad),
                *rows,
            ]
            lines.append(" ".join(map(str, values)))
            r = selector.select(
                route,
                p,
                rows if route.endswith("grouped") else None,
                device_name="PPU-ZW810",
                compute_units=72,
                mapping_id=common.mapping(p["qtype"]),
                kernel_source="wrong" if bad else b["kernel"],
                sdk_digest=b["sdk"],
                allow_prediction=allow,
            )
            expected.append(
                f"{codes[r['status']]} {r.get('config_id', 'NONE')} {r.get('grid', 0)}"
            )
    proc = subprocess.run(
        [str(tmp_path / "query")],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == expected


def test_sdk_free_cpp_parity(tmp_path):
    base = json.loads((ROOT / "policies/kpack_zw810_runtime_v1.json").read_text())
    model = json.loads((ROOT / "policies/kpack_zw810_heuristic_v1.json").read_text())
    contexts = []
    for e in h.Selector(model, base).entries.values():
        q, route, n, k, experts, m, maximum, _ = e["key"]
        r = rt.ROUTES[route]
        p = dict(qtype=q, n=n, k=k, group_size=common.GROUP_SIZE[q])
        p.update(
            dict(m=m)
            if route < 2
            else dict(experts=experts, total_rows=m, max_rows=maximum)
        )
        c = dict(route=r, problem=p, rows=e["rows"])
        contexts.append(c)
        shifted = copy.deepcopy(c)
        if route < 2:
            shifted["problem"]["m"] += 3
        else:
            shifted["rows"][0] += 3
            shifted["problem"].update(
                total_rows=sum(shifted["rows"]), max_rows=max(shifted["rows"])
            )
        contexts.append(shifted)
    cpp_parity(tmp_path, model, base, contexts)


def test_real_archive_replay_and_negative_plants(tmp_path):
    path = ROOT / "kpack-warmup-real-results.LJGzK2.tgz"
    if not path.is_file():
        pytest.skip("optional raw result archive not installed")
    base = json.loads((ROOT / "policies/kpack_zw810_runtime_v1.json").read_text())
    parts = calibrate.read_archive(path, base)
    model, report = calibrate.calibrate(parts, base, calibrate.sha(path))
    assert model == json.loads(
        (ROOT / "policies/kpack_zw810_heuristic_v1.json").read_text()
    )
    assert report["recent_within_both_5pct"] == report["recent_contexts"] == 90
    assert len(report["corrected_historical_regressions"]) == 9
    for plant in ("numerics", "sample", "missing", "source"):
        modified = copy.deepcopy(parts)
        if plant == "numerics":
            modified["results.json"][0]["numeric"]["max_error"] = 1
            r = modified["results.json"][0]
            r["digest"] = common.digest({k: v for k, v in r.items() if k != "digest"})
        elif plant == "sample":
            timing = modified["results.json"][0]["confirmation"]["samples_us"]
            timing[next(iter(timing))][0] = 0
        elif plant == "missing":
            modified["results.json"].pop()
        else:
            modified["authority.json"]["sources"] = {}
        archive = tmp_path / f"{plant}.tgz"
        with tarfile.open(archive, "w:gz") as t:
            for name, value in modified.items():
                content = json.dumps(value).encode()
                info = tarfile.TarInfo(name)
                info.size = len(content)
                t.addfile(info, io.BytesIO(content))
        with pytest.raises(ValueError):
            calibrate.read_archive(archive, base)
