#!/usr/bin/env python3
"""Freeze bounded cache-miss/new-M challenges for the compile-free runner.

Reuses only the original confirmation parent's modules. Previous winners stay
in the same-run comparison; changing a proposed runtime after measuring it is
not a pass. This does not deploy a selector or initiate a device run.
"""

import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path

import kpack_overnight_search as search
import kpack_policy as policy
import kpack_tactic_evidence as evidence
import kpack_tactic_model as model
from plan_kpack_policy_validation import SCHEMA
from run_kpack_policy_validation import validate_suite


def target_config(parent, runtime, problem):
    # Freeze the resolved grid for THIS request for compatibility with the
    # existing verifier. Its source recipe is separately retained below.
    cell = dict(
        symbol=runtime["symbol"],
        algorithm=runtime["algorithm"],
        split=runtime["split"],
        grid=runtime["grid"],
    )
    return policy.runtime_config(parent, cell, problem)


def make_suite(base, confirmation, extra_plan, data, scorer):
    model.validate_model(scorer)
    if scorer["evidence_digest"] != policy.digest(data):
        raise ValueError("model was not fitted from this evidence")
    compiled = {p["symbol"]: p for p in confirmation["candidates"]}
    for symbol, p in scorer["parents"].items():
        if symbol not in compiled or any(
            p[k] != v for k, v in compiled[symbol].items()
        ):
            raise ValueError("model requires a new compiled parent")
    requests = {r["id"]: copy.deepcopy(r) for r in base["requests"]}
    requests.update({r["id"]: copy.deepcopy(r) for r in extra_plan["requests"]})
    observations = defaultdict(list)
    for o in data["observations"]:
        observations[model.cache_key(o["context"])].append(o)
    selections, roles, targets, recipes = {}, {}, {}, {}

    def add(request, ctx, role, challenge):
        rid = request["id"]
        if rid in selections:
            raise ValueError("duplicate validation request")
        request["problem"] = ctx["problem"]
        requests[rid] = request
        prediction = model.recommend(scorer, ctx, use_cache=False)
        if not prediction.get("candidates"):
            raise ValueError("validation query has no calibrated candidates")
        selections[rid] = set(challenge)
        roles[rid] = [role]
        frozen, source_recipes = {}, {}
        for candidate in prediction["candidates"]:
            symbol = candidate["parent"]
            selections[rid].add(symbol)
            for t in candidate["tactics"]:
                c = target_config(compiled[symbol], t, ctx["problem"])
                cid = policy.digest(c)
                frozen[cid] = dict(config_id=cid, config=c)
                source_recipes[cid] = t
        targets[rid] = [frozen[c] for c in sorted(frozen)]
        recipes[rid] = source_recipes
        if not selections[rid] <= set(compiled):
            raise ValueError("validation unexpectedly requests compilation")

    # All exact cache misses, including noise/conflicting cross-epoch choices.
    # Keep each old/new winner, rather than pruning the challenge denominator.
    for key, rows in sorted(observations.items()):
        if key in scorer["cache"]:
            continue
        ctx = rows[0]["context"]
        challenge = {
            min(
                o["cells"],
                key=lambda c: (c["cost"]["median_us"], model.runtime_key(c["tactic"])),
            )["tactic"]["symbol"]
            for o in rows
        }
        request = copy.deepcopy(requests[min(o["request_id"] for o in rows)])
        add(request, ctx, "tactic-cache-miss", challenge)

    # One genuinely unseen M inside each measured dense family's M range.
    # All old AND follow-up M values exclude candidates, not only training M.
    families = defaultdict(dict)
    for rows in observations.values():
        ctx = rows[0]["context"]
        if ctx["route"].endswith("dense"):
            families[model.anchor_family(ctx)][ctx["problem"]["m"]] = rows[0]
    for points in families.values():
        ms = sorted(points)
        gaps = [
            (math.log2(hi / lo), lo, hi) for lo, hi in zip(ms, ms[1:]) if hi - lo >= 2
        ]
        if not gaps:
            continue
        _, lo, hi = max(gaps)
        m = min(hi - 1, max(lo + 1, round(math.sqrt(lo * hi))))
        old = points[lo]
        p = dict(old["context"]["problem"], m=m)
        ctx = model.context(old["context"]["route"], p)
        r = copy.deepcopy(requests[old["request_id"]])
        r["workload_key"] = f"tactic_boundary_m{m}_n{p['n']}_k{p['k']}"
        r["cell_key"] = f"q{p['qtype']}/dense/{r['workload_key']}"
        r["id"] = policy.digest([r["cell_key"], r["route"]])
        r["source_class"] = "tactic-unmeasured-M"
        challenge = set()
        for edge in (lo, hi):
            c = min(
                points[edge]["cells"],
                key=lambda c: (c["cost"]["median_us"], model.runtime_key(c["tactic"])),
            )
            symbol = c["tactic"]["symbol"]
            if model.runtime_choices(scorer["parents"][symbol], ctx):
                challenge.add(symbol)
        add(r, ctx, "tactic-unmeasured-M", challenge)
    full = dict(
        base,
        requests=list(requests.values()),
        router_files=dict(base["router_files"], **extra_plan["router_files"]),
    )
    # Candidate pool is source-owned and includes the confirmation's adaptive
    # additions; make_stage may not synthesize new tuples for this suite.
    full["candidates"] = confirmation["candidates"]
    plan = search.fix_workers(
        search.make_stage(full, selections, "tactic-validation", salt=201), 8
    )
    suite = dict(
        schema=SCHEMA,
        policy_sha256=policy.digest(scorer),
        authority=data["authority"]["overnight"],
        plan=plan,
        targets=targets,
        roles=roles,
        limits=dict(
            rounds=3,
            iterations_per_round=11,
            correctness_repeats=1,
            shortlist_parents=5,
            runtimes_per_shortlisted_parent=3,
            new_M_per_dense_family=1,
        ),
        denominator=dict(
            plan["denominator"],
            new_parent_union=0,
            cache_misses=sum("tactic-cache-miss" in v for v in roles.values()),
            new_M_requests=sum("tactic-unmeasured-M" in v for v in roles.values()),
        ),
        model_digest=scorer["model_digest"],
        runtime_recipes=recipes,
        scope="FROZEN_TACTIC_SHORTLIST_AND_HISTORICAL_CHALLENGES_NOT_GLOBAL_OPTIMUM",
    )
    validate_suite(suite)
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    read = lambda p: json.loads(p.read_text())
    suite = make_suite(
        read(args.campaign / "base-plan.json"),
        read(args.campaign / "phases/confirmation-input/plan.json"),
        read(args.validation / "suite.json")["plan"],
        evidence.combine(*(read(p) for p in args.evidence)),
        read(args.model),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(suite, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(
        "KPACK_TACTIC_VALIDATION_PLAN "
        + json.dumps(suite["denominator"], sort_keys=True)
    )


if __name__ == "__main__":
    main()
