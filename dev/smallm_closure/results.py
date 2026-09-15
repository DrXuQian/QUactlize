"""Auditable closure over measured candidates, not an optimality claim."""
import json
import math
import os
from pathlib import Path
import statistics

from quactlize.runtime.tuning import digest


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def seal(data):
    return data | dict(receipt_sha256=digest(data))


def read(path, authority):
    try:
        r = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
    if r.get("authority") != authority or r.get("receipt_sha256") != digest({k: v for k, v in r.items() if k != "receipt_sha256"}):
        raise ValueError("stale/corrupt result receipt: " + str(path))
    return r


def finite(values, count):
    return len(values) == count and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values)


def shortlist(point, candidates, screen):
    required = set(point["mandatory"])
    measured = [r for r in screen.values() if r["status"] == "MEASURED"]
    selected = {c for c in required if c in screen and screen[c]["status"] == "MEASURED"}
    for family in ("simt", "tc", "q4"):
        rows = [r for r in measured if candidates[r["candidate"]]["kind"] == family]
        rows.sort(key=lambda r: statistics.median(r["samples_us"]))
        selected.update(r["candidate"] for r in rows[:2])
    return sorted(selected)


def adjudicate(point, candidates, screen, confirmation, protocol):
    issues = []
    missing = sorted(set(point["candidates"]) - screen.keys())
    if missing:
        issues.append(dict(reason="MISSING_SCREEN", candidates=missing))
    for cid, r in screen.items():
        if r["status"] == "MEASURED" and not finite(r["samples_us"], protocol["screen"]):
            issues.append(dict(reason="INVALID_SCREEN_SAMPLES", candidate=cid))
        elif r["status"] not in ("MEASURED", "STRUCTURAL"):
            issues.append(dict(reason="CANDIDATE_FAILURE", candidate=cid, status=r["status"]))
    wanted = sorted(set(shortlist(point, candidates, screen)) | set(confirmation))
    rows = []
    for cid in wanted:
        r = confirmation.get(cid)
        if not r or r["status"] != "MEASURED" or len(r.get("rounds", [])) != protocol["rounds"] or any(
                not finite(s, protocol["samples"]) for s in r.get("rounds", [])):
            issues.append(dict(reason="MISSING_OR_INVALID_CONFIRMATION", candidate=cid))
            continue
        values = r["rounds"]
        rounds = list(map(statistics.median, values))
        median = statistics.median(v for samples in values for v in samples)
        spread = (max(rounds) / min(rounds) - 1) * 100
        rows.append(dict(candidate=cid, median_us=median, round_medians_us=rounds, spread_pct=spread))
    families = {candidates[r["candidate"]]["kind"] for r in rows}
    if "tc" not in families or not families & {"simt", "q4"}:
        issues.append(dict(reason="NO_MATCHED_SIMT_TC_COMPARISON"))
    best = min(rows, key=lambda r: (r["median_us"], r["candidate"])) if rows else None
    if best:
        if best["spread_pct"] > protocol["threshold_pct"]:
            issues.append(dict(reason="WINNER_UNSTABLE", spread_pct=best["spread_pct"]))
        # A screen-only contender within five percent must not be discarded
        # silently. Record it for automatic additional confirmation.
        confirmed = {r["candidate"] for r in rows}
        competitive = [cid for cid, r in screen.items() if r["status"] == "MEASURED" and cid not in confirmed and
                       statistics.median(r["samples_us"]) <= best["median_us"] * 1.05]
        if competitive:
            issues.append(dict(reason="COMPETITIVE_UNCONFIRMED", candidates=sorted(competitive)))
    required_failed = [cid for cid in point["mandatory"] if cid not in screen or screen[cid]["status"] not in ("MEASURED", "STRUCTURAL")]
    if required_failed:
        issues.append(dict(reason="INCUMBENT_FAILED", candidates=required_failed))
    return dict(point=point["id"], status="MEASURED_POOL_CLOSED" if best and not issues else "OPEN",
                winner=best, confirmed=rows, issues=issues,
                structural=[cid for cid, r in screen.items() if r["status"] == "STRUCTURAL"],
                scope="BEST_MEASURED_POOL_NOT_GLOBAL_OPTIMUM_NOT_MODEL_ADMISSION")


def policy_review(rows):
    """A caller cannot choose by an oracle's GPU-only routing histogram."""
    groups = {}
    for row in rows:
        p = row["point_spec"]
        key = tuple(p[f] for f in ("q", "mode", "n", "k", "experts", "topk", "channels", "tokens", "compute"))
        groups.setdefault(key, []).append(row)
    reviews = []
    for key, peers in sorted(groups.items()):
        tables = [{r["candidate"]: r["median_us"] for r in p["confirmed"]} for p in peers]
        common = set.intersection(*(set(t) for t in tables))
        winner_union = {p["winner"]["candidate"] for p in peers if p["winner"]}
        missing = sorted(winner_union - common)
        candidates = []
        if all(p["winner"] for p in peers):
            for cid in common:
                regrets = [(t[cid] / p["winner"]["median_us"] - 1) * 100 for t, p in zip(tables, peers)]
                candidates.append(dict(candidate=cid, worst_regret_pct=max(regrets),
                                       mean_regret_pct=statistics.mean(regrets)))
        best = min(candidates, key=lambda r: (r["worst_regret_pct"], r["mean_regret_pct"], r["candidate"])) if candidates else None
        status = "MISSING_PROFILE_COMPARISON" if missing or not best else "WITHIN_5_PERCENT" if best["worst_regret_pct"] <= 5 else "ROUTER_SENSITIVE_PARETO"
        if any(p.get("status", "MEASURED_POOL_CLOSED") != "MEASURED_POOL_CLOSED" for p in peers):
            status = "MEASUREMENT_OPEN"
        reviews.append(dict(key=list(key), profiles=[p["point_spec"]["router"] for p in peers],
                            status=status, missing_cross_confirm=missing, minimax=best))
    return reviews


def tpot_estimate(inventory, rows, reviews):
    """Multiplicity-weighted projection estimates, never measured TPOT."""
    keys = ("q", "mode", "n", "k", "experts", "topk", "channels", "tokens", "compute")
    indexed = {}
    for r in rows:
        key = tuple(r["point_spec"][f] for f in keys)
        indexed.setdefault(key, []).append(r)
    chosen = {tuple(r["key"]): r for r in reviews}
    output = []
    names = sorted({m["model"] for m in inventory["matrices"]})
    for model in names:
        matrices = [m for m in inventory["matrices"] if m["model"] == model]
        for compute in ("f16", "bf16"):
            for pairing in (False, True):
                fused = {r["tensor"] for r in matrices if r.get("fused")}
                calls, missing = [], []
                for weight in matrices:
                    name = weight["tensor"]
                    if weight.get("fused") and not pairing:
                        continue
                    if pairing and not weight.get("fused") and any(name.replace(old, "ffn_gate_up") in fused for old in ("ffn_gate", "ffn_up")):
                        continue
                    request = weight | dict(tokens=1, compute=compute)
                    key = tuple(request[f] for f in keys)
                    policy = chosen.get(key)
                    if not policy or not policy["minimax"]:
                        missing.append(name)
                        continue
                    cid = policy["minimax"]["candidate"]
                    observations = [next((c["median_us"] for c in r["confirmed"] if c["candidate"] == cid), None) for r in indexed[key]]
                    if any(v is None for v in observations):
                        missing.append(name)
                        continue
                    calls.append(dict(tensor=name, candidate=cid, median_profile_us=statistics.median(observations),
                                      min_profile_us=min(observations), max_profile_us=max(observations),
                                      selection_status=policy["status"], fused=bool(weight.get("fused"))))
                output.append(dict(model=model, compute=compute, request_batch=1,
                    pairing_assumption="ALL_ELIGIBLE_GATE_UP_PAIRED" if pairing else "AS_STORED_IN_GGUF",
                    projection_sum_us=sum(c["median_profile_us"] for c in calls),
                    projection_profile_range_us=[sum(c[f] for c in calls) for f in ("min_profile_us", "max_profile_us")],
                    missing=missing, calls=calls, status="PARTIAL" if missing else "PROJECTION_ESTIMATE_ONLY",
                    scope="SUM_OF_ISOLATED_PROJECTIONS_NOT_MEASURED_TPOT",
                    cautions=["Indexed prepare/finish may be duplicated relative to a fused MoE chain; no guessed fusion discount is applied.",
                              "Attention, norm, activations/topk outside these calls, sampling, host scheduling and other unsupported operators are not included.",
                              "Weight pairing is an explicit scenario, not proof that the caller fused those weights.",
                              "Cold weight microbenchmarks do not reproduce the model's complete cache/stream state."]))
    return output
