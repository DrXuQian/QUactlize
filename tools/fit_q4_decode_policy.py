#!/usr/bin/env python3
"""Fit a bounded, SDK-free decode selector from confirmed Q4 measurements."""
import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.gemv_ppu import decode_sweep as sweep, run_decode_sweep as runner
from quactlize.runtime.compiler import sha

SCHEMA = "quactlize.q4-decode-policy.v1"
POLICY = ROOT / "policies/kpack_q4_decode_v1.json"


def capture(results):
    """Validate the frozen campaign before reducing it to policy inputs.

    Compare with the captured authority, not today's production selector.
    The original module receipts and raw result hashes remain in the input.
    """
    manifest = json.loads((sweep.BUNDLE / "manifest.json").read_text())
    authority = json.loads((results / "authority.json").read_text())
    complete = json.loads((results / "result.json").read_text())
    if complete["status"] != "PASS" or complete["passed"] != 372 or complete["failures"]:
        raise ValueError("only the complete 372-case Q4 campaign may fit this policy")
    for name, value in complete["files"].items():
        path = (results / name).resolve(strict=True)
        if path.parent != results.resolve() or sha(path) != value:
            raise ValueError("campaign file hash/path differs: " + name)
    cases = []
    for w in sweep.workloads():
        path = results / (w["id"] + ".json")
        p = json.loads(path.read_text())
        inventory = sweep.catalog(manifest, w)
        runner.validate_result(p, authority, w, inventory, runner.policy_key(manifest, w))
        medians = defaultdict(list)
        for r in p["confirmation"]:
            medians[r["key"]].append(statistics.median(r["samples_us"]))
        confirmed = {key: statistics.median(values) for key, values in medians.items()}
        cases.append(dict(workload=w, result_sha256=sha(path), confirmed=confirmed,
                          round_medians=dict(medians), best=p["best"],
                          current=p["current_policy"], current_us=p["current_policy_us"]))
    return dict(schema=SCHEMA, authority=authority, result_sha256=sha(results / "result.json"),
                manifest_sha256=sha(sweep.BUNDLE / "manifest.json"),
                modules=manifest["modules"], cases=cases)


def public_key(case):
    w = case["workload"]
    return w["operator"], w["n"], w["k"], w["tokens"]


def rank(cases, key):
    regrets = [c["confirmed"][key] / min(v["median_us"] for v in c["best"].values()) - 1
               for c in cases]
    return max(regrets), statistics.mean(regrets), key


def choose(cases, arm=None):
    # Only use recipes confirmed in EVERY indistinguishable public context.
    # No fixture name, expert IDs or future GPU result may be a host feature.
    common = set.intersection(*(set(c["confirmed"]) for c in cases))
    allowed = [key for key in common if (arm is None or key.startswith(arm + ":")) and
               all(c["confirmed"][key] <= c["current_us"] * 1.05 for c in cases)]
    if not allowed:
        raise ValueError("no confirmed non-regressing policy candidate")
    return min(allowed, key=lambda key: rank(cases, key))


def intervals(rows, cases, field):
    """Minimum contiguous M ranges; at most 5% slower than each local choice."""
    nrows = len(rows)
    best = [None] * (nrows + 1)
    best[0] = (0, 0.0, [])
    for end in range(1, nrows + 1):
        for begin in range(end):
            cohort = [c for r in rows[begin:end] for c in cases[r["key"]]]
            arm = rows[begin][field].split(":", 1)[0]
            if any(r[field].split(":", 1)[0] != arm for r in rows[begin:end]):
                continue
            common = set.intersection(*(set(c["confirmed"]) for c in cohort))
            eligible = []
            for key in common:
                if not key.startswith(arm + ":"):
                    continue
                if all(c["confirmed"][key] <= c["confirmed"][r[field]] * 1.05 and
                       c["confirmed"][key] <= c["current_us"] * 1.05
                       for r in rows[begin:end] for c in cases[r["key"]]):
                    eligible.append(key)
            if not eligible:
                continue
            key = min(eligible, key=lambda k: rank(cohort, k))
            cost, regret, previous = best[begin]
            candidate = (cost + 1, max(regret, rank(cohort, key)[0]),
                         previous + [dict(first=begin+1, last=end, recipe=key)])
            if best[end] is None or candidate[:2] < best[end][:2]:
                best[end] = candidate
    return best[-1][2]


def fit(evidence):
    if evidence.get("schema") != SCHEMA or len(evidence["cases"]) != 372:
        raise ValueError("decode evidence schema/denominator differs")
    groups = defaultdict(list)
    families = defaultdict(list)
    expected = {w["id"]: w for w in sweep.workloads()}
    seen = set()
    for case in evidence["cases"]:
        w = case["workload"]
        if w != expected.get(w["id"]) or w["id"] in seen:
            raise ValueError("missing, duplicate or changed public workload")
        seen.add(w["id"])
        if set(case["confirmed"]) != set(case["round_medians"]):
            raise ValueError("confirmation keys differ")
        for key, values in case["round_medians"].items():
            if (len(values) != 6 or not all(math.isfinite(x) and x > 0 for x in values) or
                    statistics.median(values) != case["confirmed"][key]):
                raise ValueError("confirmation samples/median differ")
        for arm in ("simt", "tc"):
            key = min((k for k in case["confirmed"] if k.startswith(arm + ":")),
                      key=lambda k: (case["confirmed"][k], k))
            if (case["best"][arm]["key"] != key or
                    case["best"][arm]["median_us"] != case["confirmed"][key]):
                raise ValueError("confirmed best differs")
        if case["confirmed"].get(case["current"]) != case["current_us"]:
            raise ValueError("previous production control missing/different")
        groups[public_key(case)].append(case)
    for key, cases in sorted(groups.items()):
        families[key[:3]].append(dict(key=key, auto=choose(cases), tc=choose(cases, "tc")))
    ranges = []
    for (op, n, k), rows in sorted(families.items()):
        if [r["key"][3] for r in rows] != list(range(1, 9)):
            raise ValueError("missing token in family")
        for field in ("auto", "tc"):
            for interval in intervals(rows, groups, field):
                ranges.append(dict(operator=op, n=n, k=k, role=field, **interval))
    results = []
    for c in evidence["cases"]:
        w = c["workload"]
        r = next(r for r in ranges if r["role"] == "auto" and
                 (r["operator"], r["n"], r["k"]) == (w["operator"], w["n"], w["k"]) and
                 r["first"] <= w["tokens"] <= r["last"])
        us = c["confirmed"][r["recipe"]]
        results.append(dict(case=w["id"], recipe=r["recipe"], median_us=us,
                            regret_pct=100*(us/min(v["median_us"] for v in c["best"].values())-1),
                            previous_delta_pct=100*(us/c["current_us"]-1)))
    return dict(schema=SCHEMA, qtype=12, ranges=ranges, replay=results,
                summary=dict(cases=len(results), ranges=len(ranges),
                    simt=sum(r["recipe"].startswith("simt:") for r in results),
                    max_previous_regression_pct=max(r["previous_delta_pct"] for r in results),
                    max_measured_regret_pct=max(r["regret_pct"] for r in results),
                    within_5pct=sum(r["regret_pct"] <= 5 for r in results)),
                selection="CONFIRMED_ONLY_PUBLIC_FEATURE_MINIMAX_THEN_5PCT_RANGE_MERGE",
                grouped_features="N,K,tokens; E256/top8; shared/slot-specific A pooled; no ID readback",
                scope="Q4_DECODE_F32_ENDPOINTS_NOT_MODEL_OR_FUSED_CHAIN_ADMISSION")


def header(policy, evidence):
    parents = {r["parent"]["symbol"]: r["parent"] for r in evidence["modules"]}
    keys = sorted({r["recipe"] for r in policy["ranges"]})
    text = ["// Generated by tools/fit_q4_decode_policy.py; confirmed cold F32 call timings.",
            "#pragma once", '#include "kpack_zw810_heuristic_v1.hpp"',
            "namespace quactlize::decode_policy {", "using Config = quactlize_kpack_heuristic_v1::Config;",
            "struct Simt { int reader, variant, warps, values, columns; };",
            "struct Choice { bool simt; Simt config; Config tc; };",
            "inline constexpr Choice kChoices[] = {"]
    for key in keys:
        if key.startswith("simt:"):
            parts = [int(v[1:]) for v in key[5:].split("-")]
            text.append("    {true,{" + ",".join(map(str, parts)) + "},{}},")
        else:
            _, symbol, split, b, mode = key.split(":")
            p = parents[symbol]
            values = [12, 0 if p["route"] == "fq-dense" else 2,
                      *[p[x] for x in ("tm", "tn", "tk", "wm", "wn", "stages", "ap", "dn", "persistent")],
                      int(split[1:]), int(mode[1:]), int(b[1:]), 0]
            fields = [json.dumps(key), json.dumps(symbol), '"DECODE_TC"', *map(str, values),
                      "UINT64_C(0x51344b5034540001)"]
            text.append("    {false,{}, {" + ",".join(fields) + "}},")
    text += ["};", "struct Range { int mode, n, k, first, last, role, choice; };",
             "inline constexpr Range kRanges[] = {"]
    for r in policy["ranges"]:
        values = [0 if r["operator"] == "dense" else 2, r["n"], r["k"], r["first"], r["last"],
                  0 if r["role"] == "auto" else 1, keys.index(r["recipe"])]
        text.append("    {" + ",".join(map(str, values)) + "},")
    text += ["};", "inline Choice const* select(int mode, int n, int k, int tokens, int role=0) {",
             "    for (auto const& r : kRanges)",
             "        if (r.mode==mode && r.n==n && r.k==k && r.role==role && tokens>=r.first && tokens<=r.last)",
             "            return &kChoices[r.choice];", "    return nullptr;", "}", "} // namespace quactlize::decode_policy", ""]
    return "\n".join(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, default=POLICY)
    a = p.parse_args()
    if a.results:
        data = capture(a.results)
        a.evidence.parent.mkdir(parents=True, exist_ok=True)
        a.evidence.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0))
    else:
        data = json.loads(gzip.decompress(a.evidence.read_bytes()))
    policy = fit(data)
    policy["evidence_sha256"] = sha(a.evidence)
    a.output.write_text(json.dumps(policy, indent=2) + "\n")
    a.output.with_suffix(".hpp").write_text(header(policy, data))
    print("Q4_DECODE_POLICY", json.dumps(policy["summary"]))


if __name__ == "__main__":
    main()
