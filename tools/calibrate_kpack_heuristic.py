#!/usr/bin/env python3
"""Freeze the real-module measurements into a single-choice host policy.

This consumes completed measurements, not a new sweep. Earlier observations
remain historical evidence; a recent correction does not acquire their old
cross-epoch bound. No averaging of absolute times across campaigns occurs.
"""

import argparse
import json
from pathlib import Path
import statistics
import tarfile

import kpack_heuristic as heuristic
import kpack_policy as common
import kpack_runtime_policy as runtime
from run_kpack_warmup_real import analyze_confirmation, source_identity
from quactlize.runtime.compiler import Compiler, sha
from quactlize.runtime.tuning import Request, Tactic, digest


def read_archive(path, base):
    names = {
        "plan.json",
        "results.json",
        "summary.json",
        "authority.json",
        "modules.json",
    }
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        if (
            len(members) != len(names)
            or {m.name for m in members} != names
            or any(not m.isfile() or m.size > 64 * 1024 * 1024 for m in members)
        ):
            raise ValueError("unexpected result archive members")
        parts = {m.name: json.load(archive.extractfile(m)) for m in members}
    plan, auth = parts["plan.json"], parts["authority.json"]
    results, modules = parts["results.json"], parts["modules.json"]
    summary = parts["summary.json"]
    if (
        plan["digest"] != digest({k: v for k, v in plan.items() if k != "digest"})
        or plan["policy_digest"] != base["policy_digest"]
        or auth["plan"] != plan["digest"]
        or auth["sources"] != source_identity()
        or summary["authority"] != digest(auth)
        or summary["expected"] != len(plan["cases"])
        or summary["completed"] != len(results)
        or len(results) != len(plan["cases"])
    ):
        raise ValueError("plan/source/authority/result denominator differs")
    module_map = {r["parent"]["symbol"]: r for r in modules}
    if (
        len(module_map) != len(modules)
        or len(modules) != len(plan["parents"])
        or any(
            module_map.get(p["symbol"], {}).get("parent") != p for p in plan["parents"]
        )
    ):
        raise ValueError("module/parent union differs")
    compiler = Compiler.__new__(Compiler)  # source() is pure; no SDK or compilation.
    for r in modules:
        expected = digest(
            dict(
                identity=auth["compiler"],
                parent=r["parent"],
                source=compiler.source(r["parent"], ""),
            )
        )
        if r["identity"] != auth["compiler"] or r["key"] != expected:
            raise ValueError("compiled module contract differs")
    cases = {c["id"]: c for c in plan["cases"]}
    if len(cases) != len(results) or len({r["case"]["id"] for r in results}) != len(
        results
    ):
        raise ValueError("duplicate/missing result context")
    for r in results:
        c, n = r["case"], r["numeric"]
        request = Request(**(c["request"] | {"rows": tuple(c["request"]["rows"])}))
        if (
            cases.get(c["id"]) != c
            or request.exact_key != c["id"]
            or r["digest"] != digest({k: v for k, v in r.items() if k != "digest"})
            or r["authority"] != digest(auth)
            or r["status"] != "PASS"
            or not r["cache_replay"]
            or not n["zero_low_rejected"]
            or not 0 <= n["max_error"] < 5e-3
            or not 5e-3 <= n["planted_error"] < float("inf")
            or r["warmup"]["status"] != "MEASURED_EXACT"
        ):
            raise ValueError("numerical/cache/context receipt differs")
        if (
            r["identity"]["device"] != "PPU-ZW810"
            or r["identity"]["compute_units"] != 72
            or any(r["identity"][k] != auth["compiler"][k] for k in ("sdk", "kernel"))
        ):
            raise ValueError("device/SDK/kernel binding differs")
        confirm = r["confirmation"]
        tactics = {Tactic(**t).key: t for t in c["candidates"]}
        if (
            confirm["rejected"]
            or confirm["tactics"] != tactics
            or set(confirm["samples_us"]) != set(tactics)
            or set(confirm["resources"]) != set(tactics)
            or n["checks"] < 2 * len(tactics) + 1
        ):
            raise ValueError("confirmed tactic/correctness denominator differs")
        replay = analyze_confirmation(
            confirm["samples_us"],
            Tactic(**r["warmup"]["tactic"]),
            Tactic(**c["incumbent"]),
        )
        if any(
            confirm.get(k) != v for k, v in replay.items() if k != "hinted_vs_best_pct"
        ):
            raise ValueError("confirmation summary differs from samples")
    return parts


def context(request):
    p = dict(
        qtype=request["qtype"],
        n=request["n"],
        k=request["k"],
        group_size=common.GROUP_SIZE[request["qtype"]],
    )
    rows = request["rows"]
    if request["route"].endswith("grouped"):
        p.update(experts=len(rows), total_rows=sum(rows), max_rows=max(rows))
    else:
        p["m"] = request["m"]
    return dict(route=request["route"], problem=p, rows=rows)


def calibrate(parts, base, archive_hash):
    parents = {c["symbol"]: c for c in base["configurations"].values()}
    configs, entries, changed = {}, [], []
    old = runtime.Selector(base)
    b = base["required_binding"]
    for r in parts["results.json"]:
        ctx = context(r["case"]["request"])
        p, route, rows = ctx["problem"], ctx["route"], ctx["rows"]
        before = old.select(
            route,
            p,
            rows if route.endswith("grouped") else None,
            mapping_id=common.mapping(p["qtype"]),
            kernel_source=b["kernel_source"],
            sdk_digest=b["sdk_digest"],
        )
        conf = r["confirmation"]
        timing = conf["samples_us"]
        medians = {k: statistics.median(v) for k, v in timing.items()}
        best = min(medians.values())
        good = [
            k
            for k, v in timing.items()
            if medians[k] <= best * 1.05 and max(v) <= min(v) * 1.05
        ]
        retained = []
        if "config" in before:
            c = before["config"]
            previous_grid = (
                before["grid"] if c["grid_mode"] in ("capacity", "balanced") else 0
            )
            retained = [
                k
                for k in good
                if (
                    conf["tactics"][k]["parent"],
                    conf["tactics"][k]["algorithm"],
                    conf["tactics"][k]["split"],
                    conf["resources"][k]["grid"],
                )
                == (c["symbol"], c["algorithm"], c["split"], previous_grid)
            ]
        key = min(retained or good or list(timing), key=lambda k: (medians[k], k))
        t, resources = conf["tactics"][key], conf["resources"][key]
        c = dict(
            parents[t["parent"]],
            **{k: t[k] for k in ("algorithm", "split", "grid_mode", "grid_b")},
        )
        c.update(
            occupancy=resources["occupancy"], shipping_smem=resources["shared_bytes"]
        )
        cid = common.digest(c)
        configs[cid] = c
        old_us = next(
            (
                medians[k]
                for k in timing
                if "config" in before
                and (
                    conf["tactics"][k]["parent"],
                    conf["tactics"][k]["algorithm"],
                    conf["tactics"][k]["split"],
                    conf["resources"][k]["grid"],
                )
                == (
                    before["config"]["symbol"],
                    before["config"]["algorithm"],
                    before["config"]["split"],
                    previous_grid,
                )
            ),
            None,
        )
        entry = dict(
            context=ctx,
            key=runtime.query_key(
                route, p, rows if route.endswith("grouped") else None
            ),
            config_id=cid,
            case_id=r["case"]["id"],
            result_digest=r["digest"],
            regret_pct=(medians[key] / best - 1) * 100,
            spread_pct=(max(timing[key]) / min(timing[key]) - 1) * 100,
            selected_us=medians[key],
            best_pool_us=best,
            selected_samples_us=timing[key],
            historical_us_in_same_run=old_us,
        )
        entries.append(entry)
        if old_us is not None and old_us > best * 1.05:
            changed.append(
                dict(
                    case_id=r["case"]["id"],
                    request=r["case"]["request"],
                    before_regret_pct=(old_us / best - 1) * 100,
                    after_regret_pct=entry["regret_pct"],
                )
            )
    auth = parts["authority.json"]
    model = dict(
        schema=heuristic.SCHEMA,
        base_policy_digest=base["policy_digest"],
        required_binding=dict(
            device="PPU-ZW810",
            compute_units=72,
            sdk=auth["compiler"]["sdk"],
            kernel=auth["compiler"]["kernel"],
        ),
        configurations={
            cid: c for cid, c in configs.items() if cid not in base["configurations"]
        },
        entries=sorted(entries, key=lambda e: (e["key"], e["context"]["rows"])),
        authority=dict(
            archive_sha256=archive_hash, receipt=auth, digest=common.digest(auth)
        ),
        scope="FP16_REAL_MODULE_CALIBRATION_NOT_GLOBAL_PERFORMANCE_OR_LOADER_ADMISSION",
    )
    model["model_digest"] = common.digest(model)
    selector = heuristic.Selector(model, base)
    report = dict(
        schema="quactlize.kpack-heuristic-report.v1",
        model_digest=model["model_digest"],
        recent_contexts=len(entries),
        exact_contexts=len(selector.entries),
        selected_parents=len({configs[e["config_id"]]["symbol"] for e in entries}),
        total_parent_closure=len(
            {
                selector.configs[e["config_id"]]["symbol"]
                for e in selector.entries.values()
            }
        ),
        recent_within_median_5pct=sum(e["regret_pct"] <= 5 for e in entries),
        recent_within_both_5pct=sum(
            e["regret_pct"] <= 5 and e["spread_pct"] <= 5 for e in entries
        ),
        max_recent_regret_pct=max(e["regret_pct"] for e in entries),
        corrected_historical_regressions=changed,
        source_verdicts=parts["summary.json"]["verdicts"],
        handwritten_shape_branches=0,
        fitted_coefficients=0,
        online_measurements=0,
        max_selected_parents_per_query=1,
        production_library_updated=False,
        calibration_scope="SAME_RUN_POSTHOC_CALIBRATION_NOT_INDEPENDENT_VALIDATION",
        historical_bound_transferred=False,
        unknown_prediction_admitted=False,
    )
    return model, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    base = json.loads(a.base.read_text())
    parts = read_archive(a.results, base)
    model, report = calibrate(parts, base, sha(a.results))
    from generate_kpack_heuristic_header import generate

    outputs = {a.output: model, a.output.with_suffix(".report.json"): report}
    header = a.output.with_suffix(".hpp")
    if any(path.exists() or path.is_symlink() for path in [*outputs, header]):
        p.error("output already exists; use a fresh path")
    for path, data in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n"
        )
    header.write_text(generate(model, base, a.base.with_suffix(".hpp").name))
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k != "corrected_historical_regressions"
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
