import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import freeze_kpack_runtime_policy as freeze
import generate_kpack_runtime_header as header
import kpack_policy as common
import kpack_runtime_policy as rt
import kpack_tactic_model as recipe


def context(m=8, route="fq-dense", rows=None):
    p = dict(qtype=12, n=64, k=512, group_size=32)
    p.update(
        dict(m=m)
        if route.endswith("dense")
        else dict(total_rows=sum(rows), max_rows=max(rows), experts=len(rows))
    )
    return recipe.context(route, p, rows)


def parent(symbol="a", route="fq-dense"):
    return dict(
        symbol=symbol,
        qtype=12,
        route=route,
        tm=8,
        tn=64,
        tk=64,
        wm=8,
        wn=16,
        stages=2,
        ap=0,
        dn=16,
        persistent=-1,
        occupancy=4,
        shipping_smem=11264,
        low_bits=4,
        high_bits=0,
        metadata_bytes_per_superblock=16,
    )


def observation(ctx, parents, costs, *, name="one", status="CONFIRMED_SELECTED_SET"):
    return dict(
        id=name,
        epoch=name,
        context=ctx,
        status=status,
        cells=[
            dict(
                tactic=recipe.runtime_choices(parents[s], ctx)[0],
                cost=dict(regret_pct=v[0], spread_pct=v[1]),
            )
            for s, v in costs.items()
        ],
    )


def data(parents, observations):
    return dict(
        parents=parents,
        observations=observations,
        authority={
            o["epoch"]: dict(
                campaign_identity=dict(kernel_source="s", sdk=dict(compiler="x"))
            )
            for o in observations
        },
    )


def query(model, ctx, **kwargs):
    binding = model["required_binding"]
    options = dict(
        mapping_id=common.mapping(12),
        kernel_source=binding["kernel_source"],
        sdk_digest=binding["sdk_digest"],
    )
    options.update(kwargs)
    return rt.Selector(model).select(
        ctx["route"],
        ctx["problem"],
        ctx["rows"] if ctx["route"].endswith("grouped") else None,
        **options,
    )


def test_parent_cover_merges_small_cost_differences_without_branch_tree():
    ps = {s: parent(s) for s in ("a", "b")}
    obs = [
        observation(context(8), ps, {"a": (0, 0), "b": (2, 1)}),
        observation(context(16), ps, {"a": (2, 1), "b": (0, 0)}, name="two"),
    ]
    model, report = freeze.freeze(data(ps, obs))
    assert report["parents"] == 1 and report["contexts"] == 2
    assert report["runtime_model_coefficients"] == 0
    assert {query(model, o["context"])["status"] for o in obs} == {rt.WITHIN}


def test_noisy_winner_does_not_disqualify_proven_stable_alternative():
    ps = {s: parent(s) for s in ("a", "b")}
    ctx = context()
    obs = [
        observation(ctx, ps, {"a": (0, 7), "b": (3, 2)}, status="NOISY_CONFIRMATION")
    ]
    model, _ = freeze.freeze(data(ps, obs))
    result = query(model, ctx)
    assert result["status"] == rt.WITHIN and result["config"]["symbol"] == "b"


def test_missing_old_cost_is_not_a_pass_or_a_reason_to_keep_slow_incumbent():
    ps = {s: parent(s) for s in ("a", "b")}
    ctx = context()
    obs = [
        observation(ctx, ps, {"a": (0, 0)}),
        observation(ctx, ps, {"a": (56, 1), "b": (0, 1)}, name="two"),
    ]
    model, _ = freeze.freeze(data(ps, obs))
    r = query(model, ctx)
    assert r["config"]["symbol"] == "b" and r["status"] == rt.EXCEPTION
    assert r["reasons"] == ["MISSING_CROSS_EPOCH_COSTS"]
    assert model["entries"][0]["measured_epochs"] == ["two"]


def test_newest_epoch_choice_does_not_erase_historical_regression():
    ps = {s: parent(s) for s in ("a", "b")}
    ctx = context()
    obs = [
        observation(ctx, ps, {"a": (0, 0), "b": (12, 1)}),
        observation(ctx, ps, {"a": (20, 1), "b": (0, 1)}, name="two"),
    ]
    model, _ = freeze.freeze(data(ps, obs))
    r = query(model, ctx)
    assert r["config"]["symbol"] == "b" and r["status"] == rt.EXCEPTION
    assert (
        r["max_measured_regret_pct"] == 12
        and model["entries"][0]["latest_regret_pct"] == 0
    )


def test_correctness_failure_cannot_be_frozen_as_a_timing_exception():
    ps = {"a": parent()}
    ctx = context()
    obs = [observation(ctx, ps, {"a": (0, 0)}, status="RAW_FP16_MISMATCH")]
    with pytest.raises(ValueError, match="numerical"):
        freeze.freeze(data(ps, obs))


def test_missing_input_does_not_return_nearest_measured_tactic():
    ps = {"a": parent()}
    ctx = context()
    model, _ = freeze.freeze(data(ps, [observation(ctx, ps, {"a": (0, 0)})]))
    assert query(model, context(9))["status"] == "FALLBACK_REQUIRED"
    assert query(model, ctx, kernel_source="different")["status"] == "BINDING_MISMATCH"
    assert query(model, ctx, sdk_digest="different")["status"] == "BINDING_MISMATCH"
    assert query(model, ctx, mapping_id="0x0")["status"] == "BINDING_MISMATCH"
    assert query(model, ctx, compute_units=1)["status"] == "BINDING_MISMATCH"


def test_grouped_lookup_requires_exact_vector_and_resolves_current_grid():
    ctx = context(route="sf-grouped", rows=[9, 4, 3, 2])
    ps = {"a": parent(route="sf-grouped")}
    model, _ = freeze.freeze(data(ps, [observation(ctx, ps, {"a": (0, 0)})]))
    assert query(model, ctx)["status"] == rt.WITHIN
    # Same public total/max, different row vector.
    other = context(route="sf-grouped", rows=[9, 9, 0, 0])
    assert query(model, other)["status"] == "FALLBACK_REQUIRED"
    assert rt.rows_hash(ctx["rows"]) != rt.rows_hash(other["rows"])


def test_digest_changes_are_rejected():
    ps = {"a": parent()}
    ctx = context()
    model, _ = freeze.freeze(data(ps, [observation(ctx, ps, {"a": (0, 0)})]))
    model["entries"][0]["status"] = rt.EXCEPTION
    with pytest.raises(ValueError, match="digest"):
        rt.Selector(model)


CPP = r"""
#include "policy.hpp"
#include <iostream>
#include <vector>
using namespace quactlize_kpack_runtime_v1;
int main() {
  Query q; int route, bad_binding, count;
  while (std::cin >> route >> q.qtype >> q.n >> q.k >> q.group_size >> q.m
         >> q.experts >> q.total_rows >> q.max_rows >> bad_binding >> count) {
    std::vector<int> rows(count); for (int& n : rows) std::cin >> n;
    q.route = static_cast<Route>(route); q.rows = count ? rows.data() : nullptr; q.rows_count = count;
    q.device_name = "PPU-ZW810"; q.compute_units = 72;
    q.mapping_id = q.qtype == 12 ? 0x51344b5034540001ULL : 0x514b504b54000001ULL;
    q.kernel_source = bad_binding ? "wrong" : kKernelSource; q.sdk_digest = kSdkDigest;
    auto r = select(q);
    std::cout << static_cast<int>(r.status) << ' ' << (r.config ? r.config->id : "NONE") << ' ' << r.grid << '\n';
  }
}
"""


def cpp_parity(tmp_path, model, queries):
    # The generated header has no PPU/CUDA or Python dependencies.
    (tmp_path / "policy.hpp").write_text(header.generate(model))
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
    lines = []
    expected = []
    codes = {
        "INVALID_QUERY": 0,
        "BINDING_MISMATCH": 1,
        "FALLBACK_REQUIRED": 2,
        rt.WITHIN: 3,
        rt.EXCEPTION: 4,
    }
    selector = rt.Selector(model)
    for ctx, bad in queries:
        p = ctx["problem"]
        rows = ctx["rows"] if ctx["route"].endswith("grouped") else []
        values = [
            rt.ROUTES.index(ctx["route"]),
            p["qtype"],
            p["n"],
            p["k"],
            p["group_size"],
            p.get("m", 0),
            p.get("experts", 1),
            p.get("total_rows", 0),
            p.get("max_rows", 0),
            int(bad),
            len(rows),
            *rows,
        ]
        lines.append(" ".join(map(str, values)))
        binding = model["required_binding"]
        r = selector.select(
            ctx["route"],
            p,
            rows or None,
            mapping_id=common.mapping(p["qtype"]),
            kernel_source="wrong" if bad else binding["kernel_source"],
            sdk_digest=binding["sdk_digest"],
        )
        expected.append(
            f"{codes[r['status']]} {r.get('config_id','NONE')} {r.get('grid',0)}"
        )
    run = subprocess.run(
        [str(tmp_path / "query")],
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.splitlines() == expected


def test_sdk_free_header_parity_and_boundaries(tmp_path):
    ps = {"a": parent(), "b": parent("b", route="sf-grouped")}
    a = context()
    b = context(route="sf-grouped", rows=[129, 0, 8])
    obs = [
        observation(a, ps, {"a": (0, 0)}),
        observation(b, ps, {"b": (0, 7)}, name="two"),
    ]
    model, _ = freeze.freeze(data(ps, obs))
    cpp_parity(
        tmp_path,
        model,
        [
            (a, False),
            (a, True),
            (context(9), False),
            (b, False),
            (b, True),
            (context(route="sf-grouped", rows=[0, 129, 8]), False),
        ],
    )


def test_frozen_full_table_cpp_python_parity(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = root / "policies/kpack_zw810_runtime_v1.json"
    model = json.loads(path.read_text())
    assert (
        root / "policies/kpack_zw810_runtime_v1.hpp"
    ).read_text() == header.generate(model)
    queries = []
    for entry in model["entries"]:
        q, r, n, k, experts, total, maximum, _ = entry["key"]
        route = rt.ROUTES[r]
        p = dict(qtype=q, n=n, k=k, group_size=common.GROUP_SIZE[q])
        rows = None
        if route.endswith("dense"):
            p["m"] = total
        else:
            p.update(experts=experts, total_rows=total, max_rows=maximum)
            rows = model["row_vectors"][entry["row_vector"]]
        ctx = recipe.context(route, p, rows)
        queries.extend([(ctx, False), (ctx, True)])
        unknown = copy.deepcopy(ctx)
        unknown["problem"]["n"] += 16
        queries.append((unknown, False))
        invalid = copy.deepcopy(ctx)
        invalid["problem"]["k"] += 1
        queries.append((invalid, False))
    assert len(queries) == 4 * 2982
    cpp_parity(tmp_path, model, queries)
