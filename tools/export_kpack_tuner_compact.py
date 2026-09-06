#!/usr/bin/env python3
"""Export a small JSON.gz for initial heuristic review; retain raw on the box.

Only metadata, final decisions, per-stage top parents and failure references
are exported. This is not a full measurement matrix or a raw-log replay proof.
"""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

PHASES = (
    "calibration",
    "screen",
    "neighbors",
    "audit",
    "family-propagation",
    "confirm-1",
    "confirm-2",
    "confirm-3",
)
REQUEST_FIELDS = (
    "id",
    "cell_key",
    "qtype",
    "route",
    "problem",
    "workload_key",
    "source_class",
    "grouped",
    "worker_index",
)


def encoded(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def runtime_key(cell):
    return tuple(cell.get(k) for k in ("symbol", "algorithm", "split", "grid"))


def top_cells(cells, symbols, iterations, top, final):
    measured, structural, seen = [], Counter(), set()
    for cell in cells:
        key = runtime_key(cell)
        if cell["symbol"] not in symbols or key in seen:
            raise ValueError("foreign or duplicate runtime cell")
        seen.add(key)
        if cell["status"] == "STRUCTURAL":
            structural[cell.get("reason", "STRUCTURAL")] += 1
            continue
        samples = cell.get("samples_us", [])
        if (
            cell["status"] != "MEASURED"
            or len(samples) != iterations
            or any(
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not math.isfinite(x)
                or x <= 0
                for x in samples
            )
        ):
            raise ValueError("invalid timing samples; cannot export as measured")
        if statistics.median(samples) != cell["median_us"]:
            raise ValueError("stored median differs from timing samples")
        measured.append(cell)
    measured.sort(key=lambda c: (c["median_us"], runtime_key(c)))
    chosen, parents = [], set()
    for cell in measured:
        if cell["symbol"] not in parents:
            chosen.append(cell)
            parents.add(cell["symbol"])
            if len(chosen) == top:
                break
    # Retain the final winner's EXACT split/grid even if it is not stage top-K.
    winner = next((c for c in measured if runtime_key(c) == runtime_key(final)), None)
    if winner is not None and winner not in chosen:
        chosen.append(winner)
    return {
        "measured_variants": len(measured),
        "structural_reasons": dict(structural),
        "omitted_measured_variants": len(measured) - len(chosen),
        "top": [
            {
                **{
                    k: c[k]
                    for k in ("symbol", "algorithm", "split", "grid", "median_us")
                },
                "sample_count": len(c["samples_us"]),
                "min_us": min(c["samples_us"]),
                "max_us": max(c["samples_us"]),
            }
            for c in chosen
        ],
    }


class Exporter:
    def __init__(self, source, top):
        self.source, self.top = source, top
        self.receipts = {}

    def read(self, relative, optional=False):
        path = self.source / relative
        if optional and not path.exists():
            return None
        raw = path.read_bytes()
        self.receipts[str(relative)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def export(self):
        base = self.read("base-plan.json")
        final = self.read("results/summary.json")
        if final.get("schema") != "quactlize.kpack-overnight.v1":
            raise ValueError("expected final overnight summary")
        by_final = {(r["cell_key"], r["route"]): r for r in final["rows"]}
        base_keys = {(r["cell_key"], r["route"]) for r in base["requests"]}
        if (
            len(by_final) != len(final["rows"])
            or set(by_final) != base_keys
            or len(base_keys) != len(base["requests"])
            or final["requests"] != len(base_keys)
            or final["confirmed_requests"]
            != sum(r["status"] == "CONFIRMED_SELECTED_SET" for r in final["rows"])
        ):
            raise ValueError("final/workload denominator differs")
        rows = {}
        for r in base["requests"]:
            if r["id"] in rows or r["id"] != digest([r["cell_key"], r["route"]]):
                raise ValueError("duplicate or noncanonical workload ID")
            rows[r["id"]] = {
                "request": {k: r[k] for k in REQUEST_FIELDS},
                "final": by_final[r["cell_key"], r["route"]],
                "stages": {},
            }
        definitions = {c["symbol"]: c for c in base["candidates"]}
        phases, failures = {}, []
        for name in PHASES:
            prefix = f"phases/{name}"
            plan = self.read(f"{prefix}/plan.json", optional=True)
            if plan is None:
                phases[name] = {"status": "NOT_PLANNED"}
                continue
            for c in plan["candidates"]:
                if c["symbol"] in definitions and definitions[c["symbol"]] != c:
                    raise ValueError("configuration definition changed between phases")
                definitions[c["symbol"]] = c
            epoch = self.read(f"{prefix}/run/epoch.json", optional=True)
            census = Counter()
            phases[name] = {
                "expected_requests": len(plan["requests"]),
                "compile_parents": len(plan["candidates"]),
                "iterations": epoch["iterations"] if epoch else None,
                "details": {
                    k: v
                    for k, v in plan.get("stage_details", {}).items()
                    if k != "incumbents"
                },
                "epoch": epoch,
            }
            if epoch:
                bundle = self.read(f"{prefix}/bundle.json")
                if epoch["plan_sha256"] != digest(plan) or epoch[
                    "bundle_sha256"
                ] != digest(bundle):
                    raise ValueError(f"{name}: plan/bundle receipt differs")
                # The raw epoch is hashed above; the large assignment is redundant
                # with exact request IDs and worker_index in this compact export.
                phases[name]["epoch"] = {
                    k: v for k, v in epoch.items() if k != "assignment"
                }
            phase_ids = set()
            for request in plan["requests"]:
                rid = request["id"]
                if (
                    rid not in rows
                    or rid in phase_ids
                    or any(
                        request[k] != rows[rid]["request"][k] for k in REQUEST_FIELDS
                    )
                ):
                    raise ValueError(f"{name}: workload identity differs")
                phase_ids.add(rid)
                relative = f"{prefix}/run/results/{rid}.json"
                record = self.read(relative, optional=True)
                status = record.get("status", "INCOMPLETE") if record else "MISSING"
                detail = {"status": status, "selected_parents": len(request["symbols"])}
                if record:
                    if (
                        not epoch
                        or record["request_sha256"] != digest(request)
                        or record["device"] != epoch["devices"][request["worker_index"]]
                    ):
                        raise ValueError(
                            f"{name}/{rid}: request/device receipt differs"
                        )
                    detail.update(
                        top_cells(
                            record["cells"],
                            set(request["symbols"]),
                            epoch["iterations"],
                            self.top,
                            rows[rid]["final"],
                        )
                    )
                    rejected = record.get("rejected", {})
                    detail["rejected_parents"] = len(rejected)
                    if rejected or record.get("infrastructure_failure"):
                        # Small references only; original raw bytes stay on box.
                        failures.append(
                            {
                                "phase": name,
                                "id": rid,
                                "rejected": rejected,
                                "infrastructure_failure": record.get(
                                    "infrastructure_failure"
                                ),
                            }
                        )
                rows[rid]["stages"][name] = detail
                census[status] += 1
            phases[name]["request_status_counts"] = dict(census)
            phases[name]["group_failures"] = self.read(
                f"{prefix}/run/failures.json", optional=True
            )
            print(
                f"KPACK_COMPACT_PHASE phase={name} requests={len(phase_ids)} states={dict(census)}",
                flush=True,
            )
        used = {
            row["final"]["symbol"]
            for row in rows.values()
            if row["final"].get("symbol")
        }
        used.update(
            c["symbol"]
            for row in rows.values()
            for stage in row["stages"].values()
            for c in stage.get("top", [])
        )
        if not used <= definitions.keys():
            raise ValueError("selected configuration definition missing")
        result = {
            "schema": "quactlize.kpack-compact-review.v1",
            "source_directory": str(self.source),
            "purpose": "INITIAL_HEURISTIC_REVIEW_NOT_FULL_COST_MATRIX",
            "raw_logs_included": False,
            "raw_logs_replayed": False,
            "source_files_modified": False,
            "top_parents_per_stage": self.top,
            "final_summary": {k: v for k, v in final.items() if k != "rows"},
            "final_status_counts": dict(
                Counter(r["final"]["status"] for r in rows.values())
            ),
            "campaign_identity": self.read("campaign-identity.json"),
            "device_identity": self.read("device-identity.json", optional=True),
            "budget_admission": self.read("budget-admission.json", optional=True),
            "workloads": list(rows.values()),
            "phases": phases,
            "configurations": {s: definitions[s] for s in sorted(used)},
            "router_files": base.get("router_files", {}),
            "failures": failures,
            "source_file_sha256": self.receipts,
        }
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--max-mib", type=float, default=10)
    args = parser.parse_args()
    if not 1 <= args.top <= 8 or not math.isfinite(args.max_mib) or args.max_mib <= 0:
        parser.error("top must be 1..8; max-mib must be finite and positive")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a new filename (no overwrite)")
    try:
        with (args.source / "night.lock").open("r") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            compact = Exporter(args.source.resolve(), args.top).export()
            payload = gzip.compress(encoded(compact), compresslevel=6, mtime=0)
            if len(payload) > args.max_mib * 1024**2:
                raise ValueError(
                    f"compact size {len(payload)/1024**2:.2f} MiB exceeds cap; retry with --top 1"
                )
            with args.output.open("xb") as stream:
                stream.write(payload)
    except BlockingIOError:
        parser.error("campaign still running; export after it stops")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    summary = compact["final_summary"]
    print(
        f"KPACK_COMPACT_DONE status={summary['status']} "
        f"confirmed={summary['confirmed_requests']}/{summary['requests']} "
        f"bytes={len(payload)} size_mib={len(payload)/1024**2:.2f} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
