"""Verified sparse cost/geometry evidence for the host tactic ranker.

Original and follow-up timing epochs remain separate observations. Resource
metadata and exact routing rows supply features, never measured target labels.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import runpy
import statistics

import fit_kpack_tuner_policy as fit
import fq_grouped_multi_router as router
import gguf_internal_shape_inventory as inventory
import kpack_overnight_search as search
import kpack_policy as policy
import kpack_tactic_model as model
import run_kpack_policy_validation as validation
from run_kpack_tuner import kv
from export_kpack_tuner_compact import top_cells

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(None)
def legacy_rows(experts, topk, tokens):
    return tuple(inventory._routing_fixture(experts, topk, tokens)["rows_per_expert"])


def request_context(request, plan):
    rows = None
    if request["route"].endswith("grouped"):
        g = request["grouped"]
        if g["rows_file"] == "-":
            if g["profile"] != inventory.ROUTING_FIXTURE:
                raise ValueError("unknown generated router")
            rows = list(
                legacy_rows(int(g["experts"]), int(g["topk"]), int(g["tokens"]))
            )
        else:
            raw = plan["router_files"][g["rows_file"]]
            rows = [int(x) for x in raw.splitlines()]
            if policy.digest(rows) != g["rows_sha256"]:
                raise ValueError("router row file digest differs")
    return model.context(request["route"], request["problem"], rows)


def costs(plan, rounds, winners):
    definitions = {c["symbol"]: c for c in plan["candidates"]}
    decisions = {(r["cell_key"], r["route"]): r for r in winners}
    observations, configurations = [], {}
    for r in plan["requests"]:
        maps = [
            {
                search.runtime_key(c): c
                for c in data[r["id"]]
                if c["status"] == "MEASURED"
            }
            for data in rounds
        ]
        common = set.intersection(*(set(m) for m in maps))
        if not common:
            raise ValueError("request has no common measured runtime")
        references = [min(m[k]["median_us"] for k in common) for m in maps]
        values = {}
        for key in sorted(common):
            cells = [m[key] for m in maps]
            config = policy.runtime_config(definitions[key[0]], cells[0], r["problem"])
            cid = policy.digest(config)
            configurations[cid] = config
            times = [c["median_us"] for c in cells]
            values[cid] = dict(
                regret_pct=max(a / b - 1 for a, b in zip(times, references)) * 100,
                spread_pct=(max(times) / min(times) - 1) * 100,
                median_us=statistics.median(x for c in cells for x in c["samples_us"]),
                round_medians_us=times,
            )
        winner = decisions[r["cell_key"], r["route"]]
        observations.append(
            dict(
                id=r["id"],
                cell_key=r["cell_key"],
                route=r["route"],
                problem=r["problem"],
                status=winner["status"],
                costs=values,
                winner_median_us=winner["median_us"],
            )
        )
    return observations, configurations


def load_validation(root):
    campaign = fit.Campaign(root)
    suite = campaign.read("suite.json")
    validation.validate_suite(suite)
    final = campaign.read("results/summary.json")
    identity = suite["authority"]["campaign_identity"]
    for filename, expected in identity["orchestrator"]:
        if (
            hashlib.sha256((ROOT / "tools" / filename).read_bytes()).hexdigest()
            != expected
        ):
            raise ValueError("source parser identity differs")
    rounds, devices = [], None
    for number in (1, 2, 3):
        phase = f"confirm-{number}"
        prefix = "phases/" + phase
        plan = campaign.read(prefix + "/plan.json")
        epoch = campaign.read(prefix + "/run/epoch.json")
        bundle = campaign.read(prefix + "/bundle.json")
        if (
            plan != search.round_plan(suite["plan"], 100 + number)
            or epoch["plan_sha256"] != policy.digest(plan)
            or bundle["plan_sha256"] != policy.digest(plan)
            or epoch["bundle_sha256"] != policy.digest(bundle)
            or bundle["identity"]["source"] != identity["kernel_source"]
            or bundle["identity"]["sdk"] != identity["sdk"]
            or epoch["iterations"] != 11
            or epoch["correctness_repeats"] != 1
        ):
            raise ValueError("validation phase authority differs")
        if (
            len(epoch["device_identity"]) != 8
            or len({d["pci"] for d in epoch["device_identity"]}) != 8
            or any(
                d["name"] != "PPU-ZW810" or int(d["cu"]) != 72
                for d in epoch["device_identity"]
            )
        ):
            raise ValueError("invalid validation devices")
        if devices is None:
            devices = epoch["device_identity"]
        if epoch["device_identity"] != devices:
            raise ValueError("device identity changed across rounds")
        expected = [[] for _ in range(8)]
        for r in plan["requests"]:
            expected[r["worker_index"]].append(r["id"])
        if list(map(sorted, expected)) != list(map(sorted, epoch["assignment"])):
            raise ValueError("worker assignment differs")
        if {p.stem for p in (root / prefix / "run/results").glob("*.json")} != {
            r["id"] for r in plan["requests"]
        }:
            raise ValueError("validation result denominator differs")
        data = {}
        for r in plan["requests"]:
            record = campaign.read(prefix + "/run/results/" + r["id"] + ".json")
            if (
                record["request_sha256"] != policy.digest(r)
                or record["device"] != epoch["devices"][r["worker_index"]]
                or record["status"] != "MEASURED"
                or record.get("rejected")
                or record.get("infrastructure_failure")
            ):
                raise ValueError("validation request/status differs")
            top_cells(record["cells"], set(r["symbols"]), 11, 1, {})
            campaign.replay_record(record, r, phase)
            data[r["id"]] = record["cells"]
        rounds.append(data)
        print(
            f"KPACK_TACTIC_REPLAY phase={phase} requests={len(data)} logs={campaign.log_count}",
            flush=True,
        )
    if validation.assess(suite, rounds, set()) != {
        k: v for k, v in final.items() if k != "round_seconds"
    }:
        raise ValueError("validation summary differs from raw replay")
    observations, configurations = costs(suite["plan"], rounds, final["winner_rows"])
    return (
        campaign,
        suite["plan"],
        observations,
        configurations,
        dict(
            suite["authority"],
            validation_summary_sha256=policy.digest(final),
            validation_suite_sha256=policy.digest(suite),
        ),
    )


def resources(campaign, plan, parents, contexts):
    profiles = {s: dict(c, occupancy=0, shipping_smem=0) for s, c in parents.items()}
    mask_checks = 0
    for number in (1, 2, 3):
        phase = f"confirm-{number}"
        for r in plan["requests"]:
            record = campaign.read(f"phases/{phase}/run/results/{r['id']}.json")
            for receipt in record["logs"]:
                suffix = f"phases/{phase}/run/logs/"
                raw = (
                    campaign.root / suffix / receipt["path"].split(suffix, 1)[1]
                ).read_bytes()
                if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
                    raise ValueError("resource log changed after replay")
                seen = set()
                for line in raw.decode().splitlines():
                    if line.startswith("SF_CELL "):
                        row = json.loads(line[8:])
                        state = row["status"]
                    elif line.startswith(
                        ("FQ_TC_CELL ", "FQ_GROUPED_KPACK_CELL ", "SF_GROUPED_CELL ")
                    ):
                        row = kv(line)
                        state = row["state"]
                    elif line.startswith(
                        ("FQ_GROUPED_KPACK_SHARD ", "SF_GROUPED_SHARD ")
                    ):
                        header = kv(line)
                        if "rows_hash" in header and header[
                            "rows_hash"
                        ] != router.rows_fnv64(contexts[r["id"]]["rows"]):
                            raise ValueError(
                                "reconstructed router differs from device fixture"
                            )
                        continue
                    else:
                        continue
                    if state != "MEASURED":
                        continue
                    parent = profiles[row["symbol"]]
                    for name in ("occupancy", "shipping_smem"):
                        value = int(row.get(name, 0))
                        if value:
                            if parent[name] and parent[name] != value:
                                raise ValueError(f"same parent's {name} changed")
                            parent[name] = value
                    if (
                        "PERSISTENT" in row.get("algorithm", "")
                        and "NONPERSISTENT" not in row["algorithm"]
                    ):
                        key = (row["symbol"], int(row["grid"]))
                        if key in seen:
                            continue
                        seen.add(key)
                        q = model.tile_count(parent, contexts[r["id"]])
                        choices = {
                            g["grid"]: g
                            for g in model.grid_choices(q, 72, int(row["occupancy"]))
                        }
                        if key[1] not in choices or any(
                            int(row[n], 16) != choices[key[1]][n]
                            for n in ("capacity_b_mask", "balanced_b_mask")
                        ):
                            raise ValueError(
                                "actual expert/dense tile count disagrees with emitted grid masks"
                            )
                        mask_checks += 1
        print(
            f"KPACK_TACTIC_RESOURCES phase={phase} masks={mask_checks} PASS", flush=True
        )
    registry = inventory.load_format_registry()
    # Read the existing host traits without importing torch/backend loading.
    units = runpy.run_path(str(ROOT / "quactlize/formats.py"))["PACKED_UNITS"]
    for parent in profiles.values():
        parent.update(
            {k: registry[parent["qtype"]][k] for k in ("low_bits", "high_bits")}
        )
        parent["metadata_bytes_per_superblock"] = units[
            parent["qtype"]
        ].bytes_per_superblock
    return profiles, mask_checks


def load(root, label):
    if (root / "suite.json").exists():
        campaign, plan, observations, configurations, authority = load_validation(root)
    else:
        campaign = fit.Campaign(root)
        observations, configurations, authority = campaign.load()
        plan = campaign.read("phases/confirmation-input/plan.json")
    definitions = {c["symbol"]: c for c in plan["candidates"]}
    contexts = {r["id"]: request_context(r, plan) for r in plan["requests"]}
    profiles, mask_checks = resources(campaign, plan, definitions, contexts)
    result = []
    for row in observations:
        cells = []
        ctx = contexts[row["id"]]
        for cid, cost in row["costs"].items():
            c = configurations[cid]
            parent = profiles[c["symbol"]]
            raw_key = (
                c["symbol"],
                c["algorithm"],
                c["split"],
                policy.resolve_grid(c, row["problem"]),
            )
            choices = {
                model.runtime_key(t): t for t in model.runtime_choices(parent, ctx)
            }
            if raw_key not in choices:
                raise ValueError("generated runtime recipes lost a measured cell")
            cells.append(dict(tactic=choices[raw_key], cost=cost))
        result.append(
            dict(
                id=label + ":" + row["id"],
                request_id=row["id"],
                epoch=label,
                context=ctx,
                status=row["status"],
                cells=cells,
            )
        )
    return dict(
        schema="quactlize.kpack-tactic-evidence.v1",
        parents=profiles,
        observations=result,
        authority={
            label: dict(
                authority,
                raw_logs_replayed=campaign.log_count,
                grid_mask_checks=mask_checks,
                recipe_sources={
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [
                        ROOT / "quactlize/include/scalefirst_persistent_policy.hpp",
                        ROOT / "benchmarks/moe_router_fixture.hpp",
                        ROOT / "quactlize/formats.py",
                    ]
                },
            )
        },
    )


def combine(*datasets):
    output = dict(
        schema="quactlize.kpack-tactic-evidence.v1",
        parents={},
        observations=[],
        authority={},
    )
    for data in datasets:
        if data.get("schema") != output["schema"]:
            raise ValueError("evidence schema differs")
        if data.get("evidence_digest") != policy.digest(
            {k: v for k, v in data.items() if k != "evidence_digest"}
        ):
            raise ValueError("evidence digest differs; regenerate from original logs")
        for symbol, parent in data["parents"].items():
            if symbol in output["parents"] and output["parents"][symbol] != parent:
                raise ValueError("parent definition/resources differ across campaigns")
            output["parents"][symbol] = parent
        output["observations"].extend(data["observations"])
        if set(output["authority"]) & set(data["authority"]):
            raise ValueError("duplicate evidence epoch")
        output["authority"].update(data["authority"])
    if len({o["id"] for o in output["observations"]}) != len(output["observations"]):
        raise ValueError("duplicate observation")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--epoch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    value = load(args.root, args.epoch)
    value["evidence_digest"] = policy.digest(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
    print(
        f"KPACK_TACTIC_EVIDENCE parents={len(value['parents'])} "
        f"observations={len(value['observations'])} output={args.output}"
    )


if __name__ == "__main__":
    main()
