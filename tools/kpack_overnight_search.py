#!/usr/bin/env python3
"""Measured, bounded search stages for the K-pack profiler (no kernel code)."""

from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import asdict
import math
import statistics
import time

import kpack_tuning_plan as tuning

FIELDS = ("tm", "tn", "tk", "wm", "wn", "stages", "ap", "dn", "persistent")


def family(r):
    p = r["problem"]
    return (r["qtype"], r["route"], p["n"], p["k"], p.get("experts", 1))


def work(r):
    p = r["problem"]
    return p["n"] * p["k"] * p.get("m", p.get("total_rows", 1))


def fix_workers(plan, count):
    if count < 1:
        raise ValueError("positive worker count required")
    result = copy.deepcopy(plan)
    groups = defaultdict(list)
    for r in result["requests"]:
        groups[family(r)].append(r)
    loads = [0.0] * count
    for group in sorted(
        groups.values(),
        key=lambda g: (-sum(work(r) * len(r["symbols"]) for r in g), g[0]["id"]),
    ):
        owner = min(range(count), key=lambda i: (loads[i], i))
        loads[owner] += sum(work(r) * len(r["symbols"]) for r in group)
        for r in group:
            r["worker_index"] = owner
    return result


def parent_ranking(cells):
    best = {}
    for c in cells:
        if c["status"] == "MEASURED":
            value = c["median_us"]
            if not math.isfinite(value) or value <= 0:
                raise ValueError("invalid measured time in search input")
            best[c["symbol"]] = min(value, best.get(c["symbol"], math.inf))
    return sorted(best, key=lambda s: (best[s], s)), best


def make_stage(base, selections, name, salt=0, details=None):
    """Retain exact workload/router/worker identity while selecting parents."""
    requests, union = [], {}
    if set(selections) - {r["id"] for r in base["requests"]}:
        raise ValueError("stage selected an unknown workload")
    for original in base["requests"]:
        symbols = set(selections.get(original["id"], []))
        if not symbols:
            continue
        legal = {
            c.symbol: c for c in tuning.candidates(original["qtype"], original["route"])
        }
        if any(
            s not in legal or not tuning.admissible(legal[s], original["problem"])
            for s in symbols
        ):
            raise ValueError("stage selected a missing/inadmissible generated parent")
        r = copy.deepcopy(original)
        r["symbols"] = sorted(symbols)
        r["reasons"] = {s: name for s in r["symbols"]}
        r["schedule_salt"] = salt
        r.pop("anchor_recall", None)
        requests.append(r)
        union.update((s, asdict(legal[s])) for s in symbols)
    result = {
        k: copy.deepcopy(v)
        for k, v in base.items()
        if k not in ("requests", "candidates", "denominator")
    }
    result.update(
        requests=requests,
        candidates=[union[s] for s in sorted(union)],
        night_stage=name,
        stage_details=details or {},
        denominator={
            "workloads": len({r["cell_key"] for r in requests}),
            "route_workloads": len(requests),
            "selected_parent_workloads": sum(len(r["symbols"]) for r in requests),
            "compile_parent_union": len(union),
        },
    )
    return result


def calibration_plan(base):
    # Every qtype/route: large work, large weight, and a small-M request. This
    # avoids extrapolating a cheap Q4 pilot to all formats and prefill shapes.
    groups = defaultdict(list)
    by_symbol = {c["symbol"]: c for c in base["candidates"]}
    for r in base["requests"]:
        if r["source_class"] == "real-inventory":
            groups[r["qtype"], r["route"]].append(r)
    selections = {}
    for rows in groups.values():
        heavy = max(rows, key=work)
        weight = max(
            rows,
            key=lambda r: r["problem"]["n"]
            * r["problem"]["k"]
            * r["problem"].get("experts", 1),
        )
        same = [r for r in rows if family(r) == family(heavy)]
        small = min(same, key=work)
        for r in (heavy, weight, small):
            ranked = sorted(
                r["symbols"],
                key=lambda s: (
                    tuning.estimates(
                        tuning.Candidate(**by_symbol[s]),
                        r["problem"],
                        base["compute_units_model"],
                    )[0],
                    s,
                ),
            )
            # Include a historical challenge and diverse, already admitted types.
            anchors = [s for s in ranked if r["reasons"][s].startswith("historical")]
            chosen = list(
                dict.fromkeys(
                    anchors[:1]
                    + ranked[:2]
                    + ranked[len(ranked) // 2 : len(ranked) // 2 + 1]
                )
            )
            selections[r["id"]] = chosen
    return make_stage(base, selections, "calibration")


def audit_ids(base, pool, fraction=0.15):
    groups = defaultdict(list)
    for r in base["requests"]:
        if parent_ranking(pool.get(r["id"], []))[0]:
            groups[family(r)].append(r)
    required, priority = set(), []
    for rows in groups.values():
        rows.sort(key=lambda r: (work(r), r["id"]))
        required.update((rows[0]["id"], rows[-1]["id"]))
        for a, b in zip(rows, rows[1:]):
            if (
                parent_ranking(pool[a["id"]])[0][0]
                != parent_ranking(pool[b["id"]])[0][0]
            ):
                priority.extend((a["id"], b["id"]))
        priority.extend(r["id"] for r in rows if r["source_class"] != "real-inventory")
    target = max(len(required), math.ceil(len(base["requests"]) * fraction))
    for rid in list(dict.fromkeys(priority)):
        if len(required) >= target:
            break
        required.add(rid)
    return required


def explore_plan(
    base, pool, compiled, *, audit=False, new_limit=256, deadline=math.inf
):
    selections, incumbents, omitted = {}, {}, 0
    new = set()
    wanted = audit_ids(base, pool) if audit else {r["id"] for r in base["requests"]}
    # Round-robin formats/routes so a global compile cap is not consumed by q10.
    buckets = defaultdict(list)
    for r in base["requests"]:
        if r["id"] in wanted:
            buckets[r["qtype"], r["route"]].append(r)
    rows = [
        r
        for i in range(max(map(len, buckets.values()), default=0))
        for key in sorted(buckets)
        for r in buckets[key][i : i + 1]
    ]
    for r in rows:
        if time.monotonic() >= deadline:
            break
        ranked, _ = parent_ranking(pool.get(r["id"], []))
        if not ranked:
            continue
        legal = [
            c
            for c in tuning.candidates(r["qtype"], r["route"])
            if tuning.admissible(c, r["problem"])
        ]
        by_name = {c.symbol: c for c in legal}
        seen = {c["symbol"] for c in pool[r["id"]]}
        seeds = [by_name[s] for s in ranked[:3]]
        if audit:
            proposed, _ = tuning.choose(
                tuple(legal),
                r["problem"],
                128,
                base["compute_units_model"],
                {ranked[0]},
            )
            preferred = [c for c in proposed if c.symbol not in seen]
        else:

            def distance(c):
                return min(
                    sum(getattr(c, f) != getattr(s, f) for f in FIELDS)
                    + 0.05
                    * sum(
                        abs(math.log2(getattr(c, f) / getattr(s, f)))
                        for f in FIELDS[:6]
                    )
                    for s in seeds
                )

            preferred = sorted(
                (c for c in legal if c.symbol not in seen),
                key=lambda c: (distance(c), c.symbol),
            )[:24]
        # Far challenges are deterministic but independent of analytic score.
        far = sorted(
            (c for c in legal if c.symbol not in seen),
            key=lambda c: tuning.digest([r["id"], "far", c.symbol]),
        )[:8]
        selected = {ranked[0]}
        for c in list(dict.fromkeys(preferred + far)):
            if c.symbol not in compiled and c.symbol not in new:
                if len(new) >= new_limit:
                    omitted += 1
                    continue
                new.add(c.symbol)
            selected.add(c.symbol)
        if len(selected) > 1:
            selections[r["id"]] = selected
            incumbents[r["id"]] = ranked[0]
    return make_stage(
        base,
        selections,
        "audit" if audit else "neighbors",
        details={
            "incumbents": incumbents,
            "new_parent_union": len(new),
            "compile_cap_omissions": omitted,
            "new_parent_limit": new_limit,
            "requested_workloads": len(wanted),
            "planning_deadline_reached": time.monotonic() >= deadline,
        },
    )


def audit_gains(stage, data, threshold=0.05):
    gains = {}
    for r in stage["requests"]:
        ranked, times = parent_ranking(data.get(r["id"], []))
        incumbent = stage["stage_details"]["incumbents"][r["id"]]
        if (
            ranked
            and incumbent in times
            and times[incumbent] / times[ranked[0]] - 1 > threshold
        ):
            gains.setdefault(family(r), set()).add(ranked[0])
    return gains


def propagation_plan(base, pool, gains):
    selections = {}
    for r in base["requests"]:
        ranked, _ = parent_ranking(pool.get(r["id"], []))
        if not ranked or family(r) not in gains:
            continue
        legal = {c.symbol: c for c in tuning.candidates(r["qtype"], r["route"])}
        extra = {
            s
            for s in gains[family(r)]
            if s in legal and tuning.admissible(legal[s], r["problem"])
        }
        if extra - {c["symbol"] for c in pool[r["id"]]}:
            selections[r["id"]] = extra | {ranked[0]}
    return make_stage(base, selections, "family-propagation")


def confirmation_plan(base, pool, initial, top=4):
    selected, capped = {}, 0
    for r in base["requests"]:
        ranked, times = parent_ranking(pool.get(r["id"], []))
        if not ranked:
            continue
        chosen = set(ranked[:top])
        near = [s for s in ranked[top:] if times[s] <= times[ranked[0]] * 1.05]
        chosen.update(near[:top])
        capped += max(0, len(near) - top)
        baseline, _ = parent_ranking(initial.get(r["id"], []))
        if baseline:
            chosen.add(baseline[0])
        selected[r["id"]] = chosen
    return make_stage(
        base, selected, "confirmation", details={"near_ties_capped": capped}
    )


def round_plan(plan, number):
    # Same parents, worker assignment and IDs; separate result directories and
    # independent deterministic row-order shuffles, not one 33-sample launch.
    result = copy.deepcopy(plan)
    for r in result["requests"]:
        r["schedule_salt"] = int(tuning.digest(["confirm-round", number])[:16], 16)
    result["confirm_round"] = number
    return result


def runtime_key(c):
    return (c["symbol"], c["algorithm"], c["split"], c["grid"])


def confirmed_rows(base, confirm, rounds, initial):
    selected = {r["id"]: r for r in confirm["requests"]}
    rows = []
    for r in base["requests"]:
        row = {
            "cell_key": r["cell_key"],
            "route": r["route"],
            "status": "UNCONFIRMED",
            "symbol": "",
            "algorithm": "",
            "split": "",
            "grid": "",
            "median_us": "",
            "round_medians_us": [],
            "max_round_regret_pct": None,
            "round_spread_pct": None,
            "confirmed_variants": 0,
            "requested_parents": len(selected.get(r["id"], {}).get("symbols", [])),
        }
        maps = []
        for data in rounds:
            cells = [c for c in data.get(r["id"], []) if c["status"] == "MEASURED"]
            if len({runtime_key(c) for c in cells}) != len(cells):
                raise ValueError("duplicate confirmation runtime variant")
            if any(
                c["symbol"] not in selected.get(r["id"], {}).get("symbols", [])
                for c in cells
            ):
                raise ValueError("foreign confirmation parent")
            if any(
                not math.isfinite(v) or v <= 0 for c in cells for v in c["samples_us"]
            ):
                raise ValueError("nonfinite/nonpositive confirmation sample")
            maps.append({runtime_key(c): c for c in cells})
        common = set.intersection(*(set(m) for m in maps)) if len(maps) == 3 else set()
        candidates = []
        for key in sorted(common):
            cells = [m[key] for m in maps]
            if any(len(c["samples_us"]) != 11 for c in cells):
                continue
            medians = [statistics.median(c["samples_us"]) for c in cells]
            samples = [x for c in cells for x in c["samples_us"]]
            candidates.append(
                {"key": key, "median": statistics.median(samples), "rounds": medians}
            )
        if candidates:
            best = min(candidates, key=lambda c: (c["median"], c["key"]))
            references = [min(c["rounds"][i] for c in candidates) for i in range(3)]
            regret = max(a / b - 1 for a, b in zip(best["rounds"], references)) * 100
            spread = (max(best["rounds"]) / min(best["rounds"]) - 1) * 100
            row.update(
                status=(
                    "CONFIRMED_SELECTED_SET"
                    if max(regret, spread) <= 5
                    else "NOISY_CONFIRMATION"
                ),
                symbol=best["key"][0],
                algorithm=best["key"][1],
                split=best["key"][2],
                grid=best["key"][3],
                median_us=best["median"],
                round_medians_us=best["rounds"],
                max_round_regret_pct=regret,
                round_spread_pct=spread,
                confirmed_variants=len(candidates),
            )
            confirmed_parents = {c["key"][0] for c in candidates}
            row["missing_confirmation_parents"] = sorted(
                set(selected[r["id"]]["symbols"]) - confirmed_parents
            )
            row["incomplete_runtime_variants"] = len(
                set.union(*(set(m) for m in maps)) - {c["key"] for c in candidates}
            )
            if (
                row["missing_confirmation_parents"]
                or row["incomplete_runtime_variants"]
            ):
                row["status"] = "PARTIAL_CONFIRMATION"
        elif initial.get(r["id"]):
            row["status"] = "SCREEN_ONLY"
        rows.append(row)
    return rows
