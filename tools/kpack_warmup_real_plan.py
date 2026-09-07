"""Bounded real-shape warmup challenge, seeded by frozen measurements.

The exact historical choice is always included, not inferred from a geometry
string. One unmeasured M is explicitly labelled as a transfer control. This
is not a new all-format/all-shape Cartesian search or a global optimum proof.
"""

from dataclasses import asdict
from pathlib import Path

from quactlize.runtime.candidates import MeasuredCandidates, from_config
from quactlize.runtime.tuning import Request, ROUTES, digest

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "policies/kpack_zw810_runtime_v1.json"


def bounded_tactics(seeds, request, incumbent):
    proposals = [incumbent] + seeds.shortlist(request)
    selected, counts = [], {}
    for tactic in proposals:
        if tactic in selected:
            continue
        count = counts.get(tactic.parent, 0)
        if count == 3 or (count == 0 and len(counts) == 5):
            continue
        selected.append(tactic)
        counts[tactic.parent] = count + 1
    assert selected[0] == incumbent and len(selected) <= 15
    return selected


def make_plan(formats=range(10, 15), policy=POLICY):
    seeds = MeasuredCandidates(policy)
    cases, union = [], {}

    def add(request, role, anchor=None):
        entry = seeds.exact_entry(request)
        exact = entry is not None
        if entry is None and anchor is not None:
            entry = seeds.exact_entry(anchor)
        if entry is None:
            raise ValueError(f"historical incumbent missing: {request}")
        config = seeds.data["configurations"][entry["config_id"]]
        if not seeds.usable(config, request):
            raise ValueError(
                f"historical incumbent is outside the current domain: {request}"
            )
        incumbent = from_config(config)
        tactics = bounded_tactics(seeds, request, incumbent)
        for p in seeds.parent_union(tactics):
            union[p["symbol"]] = p
        cases.append(
            dict(
                id=request.exact_key,
                request=asdict(request) | {"rows": list(request.rows)},
                role=role,
                incumbent=asdict(incumbent),
                incumbent_config_id=entry["config_id"],
                incumbent_evidence="EXACT_MEASURED" if exact else "TRANSFER_CONTROL",
                incumbent_source_key=entry["key"],
                candidates=[asdict(t) for t in tactics],
            )
        )

    for q in formats:
        if q not in range(10, 15):
            raise ValueError("qtype must be 10..14")
        for n, k in ((1024, 5120), (8192, 5120), (5120, 25600)):
            for m in (1, 4):
                add(Request("fq-dense", q, n, k, m), "DECODE")
            anchor = Request("sf-dense", q, n, k, 2048)
            add(anchor, "PREFILL")
            if n == 1024:
                add(Request("sf-dense", q, n, k, 3072), "NEW_M_BUCKET_CONTROL", anchor)
        for n, k, row_indices in ((512, 3072, (0, 11)), (3072, 512, (16, 17))):
            for route in ("fq-grouped", "sf-grouped"):
                for index in row_indices:
                    rows = tuple(seeds.data["row_vectors"][index])
                    expected = {
                        0: (8, 1, 8),
                        11: (16384, 239, 256),
                        16: (528, 129, 9),
                        17: (534, 129, 12),
                    }[index]
                    if (
                        len(rows) != 256
                        or (sum(rows), max(rows), sum(m > 0 for m in rows)) != expected
                    ):
                        raise ValueError("measured router identity/role changed")
                    role = {
                        0: "MOE_DECODE",
                        11: "MOE_PREFILL",
                        16: "ROUTER_BOUNDARY",
                        17: "CHANGED_ROUTER",
                    }[index]
                    add(Request(route, q, n, k, sum(rows), rows), role)
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("duplicate real-shape request")
    value = dict(
        schema="quactlize.kpack-warmup-real-plan.v1",
        policy_digest=seeds.data["policy_digest"],
        cases=cases,
        parents=list(union.values()),
        tuning=dict(
            max_candidates=15,
            budget_ms=100,
            warmups=2,
            repeats=5,
            samples=3,
            improve_pct=5,
        ),
        confirmation=dict(rounds=2, max_repeats=5, order="FORWARD_THEN_REVERSE"),
        scope="REPRESENTATIVE_REAL_SHAPES_BOUNDED_POOL_NOT_GLOBAL_OPTIMUM",
    )
    value["digest"] = digest(value)
    return value


def census(plan):
    cases = plan["cases"]
    return dict(
        contexts=len(cases),
        parents=len(plan["parents"]),
        max_candidates=max(len(c["candidates"]) for c in cases),
        candidate_contexts=sum(len(c["candidates"]) for c in cases),
        exact_incumbents=sum(
            c["incumbent_evidence"] == "EXACT_MEASURED" for c in cases
        ),
        transfer_controls=sum(
            c["incumbent_evidence"] == "TRANSFER_CONTROL" for c in cases
        ),
        route_contexts={
            r: sum(c["request"]["route"] == r for c in cases) for r in ROUTES
        },
    )
