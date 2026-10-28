"""Policy ABI and deterministic fallback regressions."""
import json
from pathlib import Path
import pytest
from ci.support import legacy_policy as p
def problem(m=8, route="fq-dense", **kwargs):
    result = dict(qtype=12, n=1024, k=5120, group_size=32)
    (
        result.update(m=m)
        if route.endswith("dense")
        else result.update(total_rows=m, max_rows=4, experts=256)
    )
    result.update(kwargs)
    return result


def config(name="a", route="fq-dense", **kwargs):
    result = dict(
        symbol=name,
        qtype=12,
        route=route,
        tm=16,
        tn=64,
        tk=256,
        wm=16,
        wn=16,
        stages=2,
        ap=0,
        dn=64,
        persistent=-1,
        algorithm="TC_S1",
        split=1,
        grid=0,
        grid_mode="implicit",
        layout=1,
        mapping_id=p.mapping(12),
    )
    result.update(kwargs)
    return result


def obs(
    m, costs, *, route="fq-dense", status="CONFIRMED_SELECTED_SET", name=None, **kwargs
):
    return dict(
        id=name or str(m),
        route=route,
        cell_key=f"{route}:{name or m}",
        problem=problem(m, route, **kwargs),
        status=status,
        winner_median_us=10,
        costs={
            c: {"regret_pct": r, "spread_pct": 0, "median_us": 10 * (1 + r / 100)}
            for c, r in costs.items()
        },
    )


def query(model, m=8, route="fq-dense", **kwargs):
    return p.select(
        model,
        route,
        problem(m, route, **kwargs),
        device_name="PPU-ZW810",
        compute_units=72,
        mapping_id=p.mapping(12),
    )


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"ap": 1}, "INTERPOLATED_CONFIG_INADMISSIBLE"),
        ({"tk": 512, "split": 4}, "INTERPOLATED_CONFIG_INADMISSIBLE"),
    ],
)
def test_interpolated_config_validity_is_checked(change, reason):
    family = p.fit_family(
        p.family_key("fq-dense", problem()),
        [
            {"features": (1,), "rows": [], "costs": {"a": (0, 0)}, "blocked": ""},
            {"features": (32,), "rows": [], "costs": {"a": (0, 0)}, "blocked": ""},
        ],
    )
    assert (
        p.select_family(family, {"a": config(**change)}, problem(8))["reason"] == reason
    )


def test_tm8_domains_are_route_specific():
    assert p.admissible(config(tm=8), problem(64))
    assert not p.admissible(config(tm=8), problem(65))
    assert not p.admissible(config(tm=8, route="sf-dense"), problem(8))
    assert p.admissible(
        config(tm=8, route="fq-grouped"), problem(512, "fq-grouped", max_rows=129)
    )


def test_runtime_key_keeps_provider_delivery_split_and_persistent_grid():
    base = config(route="sf-dense")
    candidate = {
        k: base[k]
        for k in (
            "symbol",
            "route",
            "qtype",
            "tm",
            "tn",
            "tk",
            "wm",
            "wn",
            "stages",
            "ap",
            "dn",
            "persistent",
        )
    }
    cell = dict(algorithm="NONPERSISTENT", split=1, grid=16)
    a = p.runtime_config(candidate, cell, problem(8))
    b = p.runtime_config(candidate, dict(cell, grid=32), problem(32))
    assert p.digest(a) == p.digest(b) and a["grid_mode"] == "ordinary"
    with pytest.raises(ValueError, match="grid"):
        p.runtime_config(candidate, dict(cell, grid=17), problem(8))
    a = p.runtime_config(
        candidate, dict(cell, algorithm="PERSISTENT", grid=72), problem(8)
    )
    b = p.runtime_config(
        candidate, dict(cell, algorithm="PERSISTENT", grid=144), problem(8)
    )
    assert p.digest(a) != p.digest(b)
    for field, value in (("ap", 1), ("dn", 32)):
        modified = p.runtime_config(
            dict(candidate, **{field: value}),
            dict(cell, algorithm="PERSISTENT", grid=72),
            problem(1),
        )
        assert p.digest(a) != p.digest(modified)


@pytest.mark.parametrize(
    "fields",
    [
        {"m": True},
        {"m": 0},
        {"group_size": 16},
        {"k": 5000},
        {"profile": "balanced"},
        {"m": 1.5},
    ],
)
def test_invalid_public_queries_rejected(fields):
    with pytest.raises(ValueError):
        p.validate_problem("fq-dense", problem(**fields))


def test_checked_in_cpp_is_the_exact_policy_export():
    from ci.support import legacy_policy_header as header

    root = Path(__file__).resolve().parents[1]
    model = json.loads((root / "policies/kpack_zw810_v1.json").read_text())
    assert header.generate(model) == (root / "policies/kpack_zw810_v1.hpp").read_text()
    assert all(p.digest(c) == cid for cid, c in model["configurations"].items())
    assert {c["qtype"] for c in model["configurations"].values()} == set(range(10, 15))
    assert {c["route"] for c in model["configurations"].values()} == set(p.ROUTES)
