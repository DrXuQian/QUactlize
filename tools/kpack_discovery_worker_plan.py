#!/usr/bin/env python3
"""Create and validate exactly-once K-pack discovery worker ledgers.

The execution atom is one prebuilt binary shard crossed with one canonical
workload.  A worker may enumerate any runtime variants owned by that atom
(including persistent grids), but the atom itself is never split between
workers.  Assignment is shard-affine for legacy composite bundles and
logical-partition-affine for distributed catalogs, so one fetched binary is
never replicated merely to execute another workload.

``work_item_id`` is the lowercase SHA-256 of the canonical compact JSON array
``[route, operator, qtype, shard_key, manifest_sha256,
parent_id_set_sha256, workload_key]``.  The parent-set digest uses the same
encoding over type-preserving, sorted integer/string parent IDs.

Typical use::

  python3 tools/kpack_discovery_worker_plan.py create \
      --bundle bundle.json --plan plan.json --workers 8 \
      --master master.json --assignment assignment.json

  # Reuse the same complete catalog and route plan, but execute Q4_K only.
  python3 tools/kpack_discovery_worker_plan.py create \
      --bundle catalog.json --plan route-plan.json --workers 8 --qtype 12 \
      --master q4/master.json --assignment q4/assignment.json \
      --selection-dir q4/selections

  python3 tools/kpack_discovery_worker_plan.py validate-assignment \
      --bundle bundle.json --plan plan.json \
      --master master.json --assignment assignment.json

  python3 tools/kpack_discovery_worker_plan.py validate-results \
      --bundle bundle.json --plan plan.json \
      --master master.json --assignment assignment.json \
      --device-homogeneity devices.json \
      --result worker-0.json --result worker-1.json

  python3 tools/kpack_discovery_worker_plan.py select-worker \
      --bundle bundle.json --plan plan.json \
      --master master.json --assignment assignment.json \
      --worker-id 0 --output worker-0.tsv

  python3 tools/kpack_discovery_worker_plan.py write-worker-result \
      --bundle bundle.json --plan plan.json \
      --master master.json --assignment assignment.json \
      --worker-id 0 --completed-ids worker-0.completed \
      --device-identity worker-0-device.json \
      --device-homogeneity devices.json --output worker-0.json

The device-homogeneity authority has this deliberately small schema::

  {"schema": "quactlize.kpack-discovery-device-homogeneity.v1",
   "workers": [
     {"worker_id": 0, "identity_sha256": "...",
      "homogeneity_key": "..."}]}

Every identity digest must be distinct and every homogeneity key identical.
Each worker result binds the exact bundle, workload plan, master ledger,
assignment, device-homogeneity authority, and that worker's device identity.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, NoReturn


MASTER_SCHEMA = "quactlize.kpack-discovery-work-master.v1"
MASTER_SCOPE_SCHEMA = "quactlize.kpack-discovery-work-scope.v1"
ASSIGNMENT_SCHEMA = "quactlize.kpack-discovery-worker-assignment.v1"
RESULT_SCHEMA = "quactlize.kpack-discovery-worker-result.v1"
DEVICE_SCHEMA = "quactlize.kpack-discovery-device-homogeneity.v1"
SELECTION_SCHEMA = "quactlize.kpack-discovery-worker-selection.v1"
COMPOSITE_BUNDLE_SCHEMA = "quactlize.kpack-discovery-composite-bundle.v1"
CATALOG_SCHEMA = "quactlize.kpack-discovery-distributed-catalog.v2"
ROUTES = {"scalefirst", "fully-quantized"}
OPERATORS = {"dense", "grouped"}
QTYPES = {10, 11, 12, 13, 14}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class PlanError(ValueError):
    pass


def fail(message: str) -> NoReturn:
    raise SystemExit(f"kpack discovery worker plan: {message}")


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PlanError(f"value is not canonical JSON: {exc}") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise PlanError(f"cannot hash {path}: {exc}") from exc


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlanError(f"{label} must be a JSON object")
    return value


def load_bundle_authority(path: Path) -> dict[str, Any]:
    document = load_json(path, "bundle")
    if document.get("schema") == CATALOG_SCHEMA:
        try:
            import kpack_discovery_build_partitions as partitions
            return partitions.validate_catalog(path)
        except (ImportError, OSError, ValueError) as exc:
            raise PlanError(
                f"distributed catalog live authority differs: {exc}") from exc
    if document.get("schema") != COMPOSITE_BUNDLE_SCHEMA:
        return document
    try:
        import compose_kpack_discovery_bundles as composite
        return composite.validate_composite(path)
    except (ImportError, OSError, ValueError) as exc:
        raise PlanError(
            f"composite bundle live authority differs: {exc}") from exc


def write_frozen_text(path: Path, encoded: str) -> None:
    if not encoded:
        raise PlanError(f"refusing to write empty output {path}")
    if path.exists():
        try:
            previous = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PlanError(f"cannot read existing output {path}: {exc}") from exc
        if previous != encoded:
            raise PlanError(f"refusing to replace stale output {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise PlanError(f"cannot write {path}: {exc}") from exc


def write_frozen_json(path: Path, value: dict[str, Any]) -> None:
    write_frozen_text(path, json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{label} must be a nonempty string")
    value = value.strip()
    if any(mark in value for mark in ("\n", "\r", "\0")):
        raise PlanError(f"{label} must be one line")
    return value


def _sha256(value: Any, label: str) -> str:
    value = _nonempty_string(value, label)
    if not SHA256_RE.fullmatch(value):
        raise PlanError(f"{label} must be one lowercase SHA-256")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanError(f"{label} must be an integer")
    return value


def _qtype_scope(values: Any, label: str = "qtype scope"
                 ) -> tuple[int, ...] | None:
    """Normalize an optional execution scope without weakening source authority.

    ``None`` preserves the historical full-product master byte-for-byte.  An
    explicit scope is sorted and recorded in the master, while the bundle and
    workload-plan hashes continue to bind the complete catalog and route plan.
    """
    if values is None:
        return None
    if not isinstance(values, (list, tuple, set, frozenset)) or not values:
        raise PlanError(f"{label} must be a nonempty qtype collection")
    qtypes = tuple(sorted(_integer(value, label) for value in values))
    if len(qtypes) != len(set(qtypes)):
        raise PlanError(f"{label} contains duplicate qtypes")
    unsupported = sorted(set(qtypes) - QTYPES)
    if unsupported:
        raise PlanError(f"{label} contains unsupported qtypes {unsupported}")
    return qtypes


def _scope_from_master(document: dict[str, Any]) -> tuple[int, ...] | None:
    scope = document.get("execution_scope")
    if scope is None:
        return None
    if not isinstance(scope, dict) or set(scope) != {"schema", "qtypes"} or \
            scope.get("schema") != MASTER_SCOPE_SCHEMA:
        raise PlanError("master execution scope schema differs")
    return _qtype_scope(scope.get("qtypes"), "master execution qtype scope")


def _workloads(plan: dict[str, Any]
               ) -> tuple[dict[tuple[int, str], list[str]], str]:
    """Return exact workload keys per (qtype, operator).

    The canonical route-optimal plan owns the complete product ``cells``
    denominator.  Real inventory, grouped router controls, and Q4 historical
    anchors are all measured work: controls prove that the public grouped
    selector features are sufficient, while anchors prove that a previously
    competitive workload/configuration was not lost.  The earlier
    dense/grouped schema remains supported and denotes a workload set shared
    by all five qtypes.
    """
    cells = plan.get("cells")
    if cells is not None:
        if not isinstance(cells, list) or not cells:
            raise PlanError("workload plan cells are empty/malformed")
        denominator = plan.get("workload_denominator")
        if not isinstance(denominator, dict):
            raise PlanError("cell workload plan has no workload_denominator")
        dense_real = _integer(
            denominator.get("dense_real"), "workload dense_real")
        grouped_real = _integer(
            denominator.get("grouped_real"), "workload grouped_real")
        grouped_controls = _integer(
            denominator.get("grouped_router_controls"),
            "workload grouped_router_controls")
        q4_anchors = _integer(
            denominator.get("q4_historical_anchor_only"),
            "workload q4_historical_anchor_only")
        expected_total = _integer(
            denominator.get("format_workload_cells"),
            "workload format_workload_cells")
        if min(dense_real, grouped_real, grouped_controls, q4_anchors) <= 0:
            raise PlanError("real workload denominator must be positive")
        expected = {
            (qtype, "dense"): dense_real + (q4_anchors if qtype == 12 else 0)
            for qtype in sorted(QTYPES)
        } | {
            (qtype, "grouped"): grouped_real + grouped_controls
            for qtype in sorted(QTYPES)
        }
        if sum(expected.values()) != expected_total:
            raise PlanError("format workload aggregate differs")
        result = {(qtype, operator): [] for qtype in sorted(QTYPES)
                  for operator in sorted(OPERATORS)}
        seen_cells: set[str] = set()
        observed_classes: dict[tuple[int, str], dict[str, int]] = {
            key: {} for key in result
        }
        for index, row in enumerate(cells):
            if not isinstance(row, dict):
                raise PlanError(f"workload cell {index} is not an object")
            cell_key = _nonempty_string(
                row.get("cell_key"), f"workload cell {index}.cell_key")
            if cell_key in seen_cells:
                raise PlanError("workload plan contains duplicate cell keys")
            seen_cells.add(cell_key)
            qtype = _integer(row.get("qtype"), f"workload cell {index}.qtype")
            operator = _nonempty_string(
                row.get("operator"), f"workload cell {index}.operator")
            workload_key = _nonempty_string(
                row.get("workload_key"),
                f"workload cell {index}.workload_key")
            source_class = _nonempty_string(
                row.get("source_class"),
                f"workload cell {index}.source_class")
            if qtype not in QTYPES or operator not in OPERATORS:
                raise PlanError(
                    f"workload cell {index} has unsupported qtype/operator")
            allowed = ({"real-inventory", "historical-anchor"}
                       if operator == "dense" else
                       {"real-inventory", "router-control"})
            if source_class not in allowed:
                raise PlanError(
                    f"workload cell {index} has inadmissible source_class "
                    f"{source_class!r} for {operator}")
            result[(qtype, operator)].append(workload_key)
            classes = observed_classes[(qtype, operator)]
            classes[source_class] = classes.get(source_class, 0) + 1
        for identity, keys in result.items():
            qtype, operator = identity
            if len(keys) != expected[identity] or len(keys) != len(set(keys)):
                raise PlanError(
                    f"q{qtype}/{operator} complete workload denominator differs: "
                    f"got {len(keys)} expected {expected[identity]}")
            wanted_classes = ({"real-inventory": dense_real,
                               **({"historical-anchor": q4_anchors}
                                  if qtype == 12 else {})}
                              if operator == "dense" else
                              {"real-inventory": grouped_real,
                               "router-control": grouped_controls})
            if observed_classes[identity] != wanted_classes:
                raise PlanError(
                    f"q{qtype}/{operator} workload source-class census differs")
            result[identity] = sorted(keys)
        return result, "CELLS_COMPLETE_PRODUCT_DENOMINATOR_V1"

    result: dict[tuple[int, str], list[str]] = {}
    for operator in sorted(OPERATORS):
        rows = plan.get(operator)
        if not isinstance(rows, list) or not rows:
            raise PlanError(f"workload plan {operator} rows are empty/malformed")
        keys: list[str] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise PlanError(f"{operator} workload {index} is not an object")
            keys.append(_nonempty_string(
                row.get("key"), f"{operator} workload {index}.key"))
        if len(keys) != len(set(keys)):
            raise PlanError(f"workload plan contains duplicate {operator} keys")
        for qtype in sorted(QTYPES):
            result[(qtype, operator)] = sorted(keys)
    return result, "SHARED_OPERATOR_ROWS_V1"


def _parent_id(value: Any, label: str) -> int | str:
    if isinstance(value, bool):
        raise PlanError(f"{label} must be an integer or nonempty string")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return _nonempty_string(value, label)
    raise PlanError(f"{label} must be an integer or nonempty string")


def _parent_sort_key(value: int | str) -> tuple[int, int | str]:
    return (0, value) if isinstance(value, int) else (1, value)


def _normalized_shard_rows(bundle: dict[str, Any]
                           ) -> list[tuple[str, dict[str, Any]]]:
    """Normalize SF's list and FQ's mapping to keyed shard records."""
    shards = bundle.get("shards")
    if isinstance(shards, dict) and shards:
        raw_rows = list(shards.items())
    elif isinstance(shards, list) and shards:
        raw_rows = [(None, row) for row in shards]
    else:
        raise PlanError("bundle shards are empty/malformed")

    normalized: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for index, (mapping_key, row) in enumerate(raw_rows):
        if not isinstance(row, dict):
            raise PlanError(f"bundle shard {index} is not an object")
        identities = [row[field] for field in ("shard_id", "shard_key")
                      if field in row]
        if not identities:
            raise PlanError(
                f"bundle shard {index} has no shard_id/shard_key")
        identity = _nonempty_string(identities[0], f"bundle shard {index} key")
        if any(_nonempty_string(value, f"bundle shard {index} key") != identity
               for value in identities[1:]):
            raise PlanError(
                f"bundle shard {index} has contradictory shard_id/shard_key")
        if mapping_key is not None and _nonempty_string(
                mapping_key, "bundle shard mapping key") != identity:
            raise PlanError(
                f"bundle shard mapping key differs from {identity}")
        if identity in seen:
            raise PlanError(f"bundle contains duplicate shard key {identity}")
        seen.add(identity)
        normalized.append((identity, row))
    return sorted(normalized, key=lambda item: item[0])


def _bundle_shards(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    shards = _normalized_shard_rows(bundle)
    parsed: list[dict[str, Any]] = []
    parent_owners: dict[tuple[str, str, int, type, int | str], str] = {}
    for shard_key, row in shards:
        route = _nonempty_string(row.get("route"), f"{shard_key}.route")
        operator = _nonempty_string(
            row.get("operator"), f"{shard_key}.operator")
        qtype = _integer(row.get("qtype"), f"{shard_key}.qtype")
        if route not in ROUTES:
            raise PlanError(f"{shard_key} has unsupported route {route!r}")
        if operator not in OPERATORS:
            raise PlanError(
                f"{shard_key} has unsupported operator {operator!r}")
        if qtype not in QTYPES:
            raise PlanError(f"{shard_key} has unsupported qtype {qtype}")
        manifest_sha = _sha256(
            row.get("manifest_sha256"), f"{shard_key}.manifest_sha256")
        raw_parents = row.get("parent_ids")
        if not isinstance(raw_parents, list) or not raw_parents:
            raise PlanError(f"{shard_key}.parent_ids are empty/malformed")
        parents = sorted(
            (_parent_id(parent, f"{shard_key}.parent_ids")
             for parent in raw_parents), key=_parent_sort_key)
        typed_parents = {(type(parent), parent) for parent in parents}
        if len(parents) != len(typed_parents):
            raise PlanError(f"{shard_key} contains duplicate parent IDs")
        for parent in parents:
            owner_key = (route, operator, qtype, type(parent), parent)
            previous = parent_owners.get(owner_key)
            if previous is not None:
                raise PlanError(
                    f"parent {parent!r} is duplicated across {previous} and "
                    f"{shard_key}")
            parent_owners[owner_key] = shard_key
        parent_set_sha = digest(parents)
        if (row.get("parent_id_set_sha256") is not None and
                _sha256(row.get("parent_id_set_sha256"),
                        f"{shard_key}.parent_id_set_sha256") != parent_set_sha):
            raise PlanError(f"{shard_key} parent-set digest differs")
        partition_id = row.get("partition_id")
        artifact_id = row.get("artifact_id")
        if partition_id is None or artifact_id is None:
            if partition_id is not None or artifact_id is not None:
                raise PlanError(
                    f"{shard_key} has an incomplete partition/artifact identity")
            partition_id = None
            artifact_id = None
        else:
            partition_id = _integer(partition_id, f"{shard_key}.partition_id")
            if partition_id < 0:
                raise PlanError(f"{shard_key}.partition_id must be nonnegative")
            artifact_id = _nonempty_string(
                artifact_id, f"{shard_key}.artifact_id")
        parsed.append({
            "shard_key": shard_key,
            "route": route,
            "operator": operator,
            "qtype": qtype,
            "manifest_sha256": manifest_sha,
            "parent_ids": parents,
            "parent_id_set_sha256": parent_set_sha,
            "partition_id": partition_id,
            "artifact_id": artifact_id,
        })
    return parsed


def make_master(bundle_path: Path, plan_path: Path,
                qtypes: Any = None) -> dict[str, Any]:
    bundle = load_bundle_authority(bundle_path)
    plan = load_json(plan_path, "workload plan")
    workloads, workload_schema = _workloads(plan)
    explicit_scope = _qtype_scope(qtypes)
    selected_qtypes = set(QTYPES if explicit_scope is None else explicit_scope)
    shards = [shard for shard in _bundle_shards(bundle)
              if shard["qtype"] in selected_qtypes]
    if not shards:
        raise PlanError("execution scope selects no binary shards")
    items: list[dict[str, Any]] = []
    for shard in shards:
        for workload_key in workloads[(shard["qtype"], shard["operator"])]:
            components = [
                shard["route"], shard["operator"], shard["qtype"],
                shard["shard_key"], shard["manifest_sha256"],
                shard["parent_id_set_sha256"], workload_key,
            ]
            items.append({
                "work_item_id": digest(components),
                "route": shard["route"],
                "operator": shard["operator"],
                "qtype": shard["qtype"],
                "shard_key": shard["shard_key"],
                "manifest_sha256": shard["manifest_sha256"],
                "parent_id_set_sha256": shard["parent_id_set_sha256"],
                "parent_count": len(shard["parent_ids"]),
                "workload_key": workload_key,
                "partition_id": shard["partition_id"],
                "artifact_id": shard["artifact_id"],
                "runtime_partition": "OWNED_BY_ITEM_NOT_SPLIT_ACROSS_WORKERS",
            })
    items.sort(key=lambda row: row["work_item_id"])
    ids = [row["work_item_id"] for row in items]
    if not items or len(ids) != len(set(ids)):
        raise PlanError("generated work-item IDs are empty or duplicated")
    by_operator = {
        operator: sum(row["operator"] == operator for row in items)
        for operator in sorted(OPERATORS)
    }
    workload_counts = {
        f"q{qtype}/{operator}": len(workloads[(qtype, operator)])
        for qtype in sorted(selected_qtypes) for operator in sorted(OPERATORS)
    }
    result = {
        "schema": MASTER_SCHEMA,
        "bundle_sha256": file_sha(bundle_path),
        "workload_plan_sha256": file_sha(plan_path),
        "denominator": {
            "binary_shards": len(shards),
            "workload_schema": workload_schema,
            "workload_keys": sum(workload_counts.values()),
            "workloads_by_qtype_operator": workload_counts,
            "work_items": len(items),
            "work_items_by_operator": by_operator,
            "assignment_groups": len({
                ("partition", row["partition_id"])
                if row["partition_id"] is not None else
                ("shard", row["shard_key"])
                for row in items}),
        },
        "work_items": items,
    }
    if explicit_scope is not None:
        result["execution_scope"] = {
            "schema": MASTER_SCOPE_SCHEMA,
            "qtypes": list(explicit_scope),
        }
    return result


def validate_master(document: dict[str, Any], bundle_path: Path,
                    plan_path: Path) -> dict[str, Any]:
    expected = make_master(
        bundle_path, plan_path, qtypes=_scope_from_master(document))
    if document != expected:
        raise PlanError("master ledger differs from bundle/workload-plan authority")
    return document


def make_assignment(master: dict[str, Any], master_sha256: str,
                    worker_count: int) -> dict[str, Any]:
    if worker_count <= 0:
        raise PlanError("worker count must be positive")
    items = master.get("work_items")
    if not isinstance(items, list):
        raise PlanError("master work items are malformed")
    groups: dict[tuple[str, int | str], list[dict[str, Any]]] = {}
    for item in items:
        partition_id = item.get("partition_id")
        if partition_id is not None:
            if isinstance(partition_id, bool) or not isinstance(partition_id, int):
                raise PlanError("work-item partition_id is malformed")
            group = ("partition", partition_id)
        else:
            group = ("shard", _nonempty_string(
                item.get("shard_key"), "work-item shard_key"))
        groups.setdefault(group, []).append(item)
    if worker_count > len(groups):
        raise PlanError("worker count exceeds the shard/partition denominator")

    # Longest-processing-time bin packing is deterministic and keeps every
    # logical partition (or legacy shard) whole.  The item count is a stable
    # proxy for device work because each item has the same measurement contract.
    ordered_groups = sorted(
        groups.items(), key=lambda pair: (-len(pair[1]), pair[0][0], pair[0][1]))
    bins = [{"worker_id": worker, "groups": [], "items": []}
            for worker in range(worker_count)]
    for group, group_items in ordered_groups:
        target = min(bins, key=lambda row: (
            len(row["items"]), len(row["groups"]), row["worker_id"]))
        target["groups"].append(group)
        target["items"].extend(group_items)
    workers = []
    for row in bins:
        worker_items = sorted(row["items"], key=lambda item: item["work_item_id"])
        workers.append({
            "worker_id": row["worker_id"],
            "partition_ids": sorted({item["partition_id"] for item in worker_items
                                     if item.get("partition_id") is not None}),
            "artifact_ids": sorted({item["artifact_id"] for item in worker_items
                                    if item.get("artifact_id") is not None}),
            "shard_keys": sorted({item["shard_key"] for item in worker_items}),
            "work_item_ids": [item["work_item_id"] for item in worker_items],
        })
    return {
        "schema": ASSIGNMENT_SCHEMA,
        "master_sha256": _sha256(master_sha256, "master_sha256"),
        "bundle_sha256": master["bundle_sha256"],
        "workload_plan_sha256": master["workload_plan_sha256"],
        "worker_count": worker_count,
        "assignment_policy": "PARTITION_OR_SHARD_AFFINE_GREEDY_LPT_V1",
        "workers": workers,
    }


def validate_assignment(document: dict[str, Any], master: dict[str, Any],
                        master_sha256: str) -> dict[str, Any]:
    if document.get("schema") != ASSIGNMENT_SCHEMA:
        raise PlanError("assignment schema differs")
    worker_count = _integer(document.get("worker_count"), "worker_count")
    expected = make_assignment(master, master_sha256, worker_count)
    if document != expected:
        # Name the two most dangerous cases before falling back to a generic
        # deterministic-assignment mismatch.
        workers = document.get("workers")
        if isinstance(workers, list):
            observed = []
            for row in workers:
                if isinstance(row, dict) and isinstance(
                        row.get("work_item_ids"), list):
                    observed.extend(row["work_item_ids"])
            expected_ids = [row["work_item_id"]
                            for row in master["work_items"]]
            if any(not isinstance(item, str) or
                   not SHA256_RE.fullmatch(item) for item in observed):
                raise PlanError("assignment work-item IDs are malformed")
            if len(observed) != len(set(observed)):
                raise PlanError("assignment work-item overlap/duplicate detected")
            if set(observed) != set(expected_ids):
                raise PlanError("assignment work-item gap/extra detected")
            owner: dict[str, int] = {}
            by_id = {row["work_item_id"]: row for row in master["work_items"]}
            for worker_row in workers:
                if not isinstance(worker_row, dict):
                    continue
                worker = worker_row.get("worker_id")
                for item_id in worker_row.get("work_item_ids", []):
                    item = by_id.get(item_id)
                    if item is None:
                        continue
                    shard = item["shard_key"]
                    if shard in owner and owner[shard] != worker:
                        raise PlanError("assignment scatters one shard across workers")
                    owner[shard] = worker
        raise PlanError("assignment differs from deterministic worker policy")
    return document


def _assigned_worker(assignment: dict[str, Any], worker_id: int
                     ) -> dict[str, Any]:
    worker_id = _integer(worker_id, "worker_id")
    worker_count = _integer(assignment.get("worker_count"), "worker_count")
    if worker_id < 0 or worker_id >= worker_count:
        raise PlanError(
            f"worker_id {worker_id} is outside [0,{worker_count})")
    rows = [row for row in assignment.get("workers", [])
            if isinstance(row, dict) and row.get("worker_id") == worker_id]
    if len(rows) != 1:
        raise PlanError(f"worker_id {worker_id} assignment row differs")
    return rows[0]


def make_worker_selection(
        master: dict[str, Any], assignment: dict[str, Any], worker_id: int,
        *, master_sha256: str, assignment_sha256: str) -> dict[str, Any]:
    assigned = _assigned_worker(assignment, worker_id)
    by_id = {row["work_item_id"]: row for row in master["work_items"]}
    ids = assigned.get("work_item_ids")
    if not isinstance(ids, list) or any(item not in by_id for item in ids):
        raise PlanError(f"worker_id {worker_id} references an unknown work item")
    items = [copy.deepcopy(by_id[item]) for item in ids]
    if [row["work_item_id"] for row in items] != ids:
        raise PlanError(f"worker_id {worker_id} selection order differs")
    return {
        "schema": SELECTION_SCHEMA,
        "worker_id": worker_id,
        "worker_count": assignment["worker_count"],
        "bundle_sha256": master["bundle_sha256"],
        "workload_plan_sha256": master["workload_plan_sha256"],
        "master_sha256": _sha256(master_sha256, "master_sha256"),
        "assignment_sha256": _sha256(
            assignment_sha256, "assignment_sha256"),
        "partition_ids": copy.deepcopy(assigned.get("partition_ids", [])),
        "artifact_ids": copy.deepcopy(assigned.get("artifact_ids", [])),
        "shard_keys": copy.deepcopy(assigned.get("shard_keys", [])),
        "work_items": items,
    }


def write_worker_selections(directory: Path, master: dict[str, Any],
                            assignment: dict[str, Any], *,
                            master_sha256: str,
                            assignment_sha256: str) -> None:
    """Materialize the complete deterministic selection set.

    The directory is resumable but fail-closed: existing expected files must
    be byte-identical and unrelated entries are rejected.  No selection is
    accepted as an input to this operation.
    """
    expected_names = {
        f"worker-{worker_id}.json"
        for worker_id in range(assignment["worker_count"])
    }
    if directory.exists():
        if directory.is_symlink() or not directory.is_dir():
            raise PlanError(f"selection output is not a directory: {directory}")
        observed_names = {entry.name for entry in directory.iterdir()}
        unexpected = sorted(observed_names - expected_names)
        if unexpected:
            raise PlanError(
                f"selection output contains unexpected entries {unexpected[:3]}")
    else:
        directory.mkdir(parents=True)
    for worker_id in range(assignment["worker_count"]):
        selection = make_worker_selection(
            master, assignment, worker_id,
            master_sha256=master_sha256,
            assignment_sha256=assignment_sha256)
        write_frozen_json(directory / f"worker-{worker_id}.json", selection)


def worker_selection_tsv(selection: dict[str, Any]) -> str:
    required = ("work_item_id", "route", "operator", "qtype",
                "shard_key", "workload_key")
    metadata = (
        f"# schema={selection['schema']}\n"
        f"# worker_id={selection['worker_id']}\n"
        f"# worker_count={selection['worker_count']}\n"
        f"# bundle_sha256={selection['bundle_sha256']}\n"
        f"# workload_plan_sha256={selection['workload_plan_sha256']}\n"
        f"# master_sha256={selection['master_sha256']}\n"
        f"# assignment_sha256={selection['assignment_sha256']}\n")
    lines = ["\t".join(required)]
    for index, row in enumerate(selection["work_items"]):
        values = [str(row[field]) for field in required]
        if any(any(mark in value for mark in ("\t", "\n", "\r", "\0"))
               for value in values):
            raise PlanError(
                f"worker selection row {index} cannot be represented as TSV")
        lines.append("\t".join(values))
    return metadata + "\n".join(lines) + "\n"


def validate_device_authority(document: dict[str, Any],
                              worker_count: int) -> dict[int, str]:
    if set(document) != {"schema", "workers"} or \
            document.get("schema") != DEVICE_SCHEMA:
        raise PlanError("device-homogeneity authority schema differs")
    workers = document.get("workers")
    if not isinstance(workers, list) or len(workers) != worker_count:
        raise PlanError("device-homogeneity worker denominator differs")
    identities: dict[int, str] = {}
    homogeneity: set[str] = set()
    for row in workers:
        if not isinstance(row, dict) or set(row) != {
                "worker_id", "identity_sha256", "homogeneity_key"}:
            raise PlanError("device-homogeneity worker row is malformed")
        worker = _integer(row["worker_id"], "device worker_id")
        if worker < 0 or worker >= worker_count or worker in identities:
            raise PlanError("device-homogeneity worker IDs differ")
        identities[worker] = _sha256(
            row["identity_sha256"], "device identity_sha256")
        homogeneity.add(_sha256(
            row["homogeneity_key"], "device homogeneity_key"))
    if set(identities) != set(range(worker_count)):
        raise PlanError("device-homogeneity worker IDs have a gap")
    if len(set(identities.values())) != worker_count:
        raise PlanError("device identity evidence was reused across workers")
    if len(homogeneity) != 1:
        raise PlanError("workers are not device-homogeneous")
    return identities


def validate_results(
        results: list[dict[str, Any]], assignment: dict[str, Any],
        *, master_sha256: str, assignment_sha256: str,
        device_authority_sha256: str,
        device_identities: dict[int, str]) -> None:
    worker_count = assignment["worker_count"]
    if len(results) != worker_count:
        raise PlanError("worker-result denominator differs")
    expected_by_worker = {
        row["worker_id"]: row["work_item_ids"]
        for row in assignment["workers"]
    }
    seen_workers: set[int] = set()
    seen_items: set[str] = set()
    for result in results:
        required = {
            "schema", "worker_id", "bundle_sha256",
            "workload_plan_sha256", "master_sha256",
            "assignment_sha256", "device_homogeneity_sha256",
            "device_identity_sha256", "completed_work_item_ids",
        }
        if set(result) != required or result.get("schema") != RESULT_SCHEMA:
            raise PlanError("worker result schema differs")
        worker = _integer(result["worker_id"], "result worker_id")
        if worker not in expected_by_worker or worker in seen_workers:
            raise PlanError("worker result is missing/duplicated/unassigned")
        seen_workers.add(worker)
        bindings = {
            "bundle_sha256": assignment["bundle_sha256"],
            "workload_plan_sha256": assignment["workload_plan_sha256"],
            "master_sha256": master_sha256,
            "assignment_sha256": assignment_sha256,
            "device_homogeneity_sha256": device_authority_sha256,
            "device_identity_sha256": device_identities[worker],
        }
        for key, expected in bindings.items():
            if _sha256(result.get(key), f"result {key}") != expected:
                raise PlanError(f"worker {worker} result has stale {key}")
        completed = result.get("completed_work_item_ids")
        if not isinstance(completed, list) or any(
                not isinstance(item, str) for item in completed):
            raise PlanError(f"worker {worker} completion list is malformed")
        if len(completed) != len(set(completed)):
            raise PlanError(f"worker {worker} completion list has duplicates")
        if completed != expected_by_worker[worker]:
            raise PlanError(
                f"worker {worker} completion IDs differ from assignment")
        overlap = seen_items.intersection(completed)
        if overlap:
            raise PlanError("worker results overlap")
        seen_items.update(completed)
    if seen_workers != set(range(worker_count)):
        raise PlanError("worker result set has a worker gap")
    master_items = {
        item for row in assignment["workers"] for item in row["work_item_ids"]}
    if seen_items != master_items:
        raise PlanError("worker result union differs from assignment")


def _load_authorities(bundle: Path, plan: Path, master_path: Path,
                      assignment_path: Path
                      ) -> tuple[dict[str, Any], dict[str, Any]]:
    master = load_json(master_path, "master ledger")
    validate_master(master, bundle, plan)
    assignment = load_json(assignment_path, "worker assignment")
    validate_assignment(assignment, master, file_sha(master_path))
    return master, assignment


def create_command(args: argparse.Namespace) -> int:
    qtypes = getattr(args, "qtype", None)
    master = make_master(args.bundle, args.plan, qtypes=qtypes)
    write_frozen_json(args.master, master)
    # Hash the exact persisted bytes, not an in-memory approximation.
    assignment = make_assignment(master, file_sha(args.master), args.workers)
    write_frozen_json(args.assignment, assignment)
    validate_assignment(assignment, master, file_sha(args.master))
    selection_dir = getattr(args, "selection_dir", None)
    if selection_dir is not None:
        write_worker_selections(
            selection_dir, master, assignment,
            master_sha256=file_sha(args.master),
            assignment_sha256=file_sha(args.assignment))
    scope = _scope_from_master(master)
    print(
        "KPACK_DISCOVERY_WORK_PLAN "
        f"shards={master['denominator']['binary_shards']} "
        f"items={master['denominator']['work_items']} "
        f"workers={args.workers} "
        f"qtypes={'ALL' if scope is None else ','.join(map(str, scope))} "
        f"selections={0 if selection_dir is None else args.workers} "
        "policy=PARTITION_OR_SHARD_AFFINE_GREEDY_LPT_V1")
    return 0


def validate_assignment_command(args: argparse.Namespace) -> int:
    master, assignment = _load_authorities(
        args.bundle, args.plan, args.master, args.assignment)
    print(
        "KPACK_DISCOVERY_ASSIGNMENT PASS "
        f"items={master['denominator']['work_items']} "
        f"workers={assignment['worker_count']} union=MASTER disjoint=1")
    return 0


def validate_results_command(args: argparse.Namespace) -> int:
    master, assignment = _load_authorities(
        args.bundle, args.plan, args.master, args.assignment)
    device = load_json(args.device_homogeneity, "device-homogeneity authority")
    identities = validate_device_authority(device, assignment["worker_count"])
    results = [load_json(path, "worker result") for path in args.result]
    validate_results(
        results, assignment,
        master_sha256=file_sha(args.master),
        assignment_sha256=file_sha(args.assignment),
        device_authority_sha256=file_sha(args.device_homogeneity),
        device_identities=identities)
    print(
        "KPACK_DISCOVERY_RESULTS PASS "
        f"items={master['denominator']['work_items']} "
        f"workers={assignment['worker_count']} exactly_once=1 "
        "device_homogeneous=1")
    return 0


def _expect_red(label: str, operation, needle: str) -> None:
    try:
        operation()
    except PlanError as exc:
        if needle not in str(exc):
            raise PlanError(f"{label}: wrong failure: {exc}") from exc
        return
    raise PlanError(f"{label}: negative stayed green")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="kpack-worker-plan-") as name:
        root = Path(name)
        bundle = root / "bundle.json"
        plan = root / "plan.json"
        master_path = root / "master.json"
        assignment_path = root / "assignment.json"
        bundle_doc = {
            "schema": "fixture-bundle",
            "shards": [
                {
                    "shard_id": "sf-q10-dense-000",
                    "route": "scalefirst", "operator": "dense",
                    "qtype": 10, "manifest_sha256": "1" * 64,
                    "parent_ids": [1, 0],
                },
                {
                    "shard_key": "fq-q12-dense-000",
                    "route": "fully-quantized", "operator": "dense",
                    "qtype": 12, "manifest_sha256": "2" * 64,
                    "parent_ids": ["fq-a"],
                },
                {
                    "shard_key": "fq-q14-grouped-000",
                    "route": "fully-quantized", "operator": "grouped",
                    "qtype": 14, "manifest_sha256": "3" * 64,
                    "parent_ids": ["fq-g-a", "fq-g-b"],
                },
            ],
        }
        plan_doc = {
            "schema": "fixture-plan",
            "dense": [{"key": "dense-0"}, {"key": "dense-1"}],
            "grouped": [{"key": "grouped-0"}],
        }
        write_frozen_json(bundle, bundle_doc)
        write_frozen_json(plan, plan_doc)
        master = make_master(bundle, plan)
        write_frozen_json(master_path, master)
        assignment = make_assignment(master, file_sha(master_path), 3)
        write_frozen_json(assignment_path, assignment)
        validate_assignment(assignment, master, file_sha(master_path))
        if master["denominator"]["work_items"] != 5:
            raise PlanError("self-test work-item denominator differs")
        expected_parent_hash = digest([0, 1])
        expected_item = digest([
            "scalefirst", "dense", 10, "sf-q10-dense-000", "1" * 64,
            expected_parent_hash, "dense-0",
        ])
        if expected_item not in {
                row["work_item_id"] for row in master["work_items"]}:
            raise PlanError("self-test work-item ID formula differs")
        assigned = [item for row in assignment["workers"]
                    for item in row["work_item_ids"]]
        if sorted(assigned) != sorted(
                row["work_item_id"] for row in master["work_items"]):
            raise PlanError("self-test assignment union differs")

        # The FQ bundle representation is a mapping.  It must normalize to
        # the exact same shard/work-item authority as SF's list form.
        mapping_bundle_doc = copy.deepcopy(bundle_doc)
        mapping_bundle_doc["shards"] = {
            row.get("shard_id", row.get("shard_key")): row
            for row in mapping_bundle_doc["shards"]
        }
        mapping_bundle = root / "mapping-bundle.json"
        write_frozen_json(mapping_bundle, mapping_bundle_doc)
        mapping_master = make_master(mapping_bundle, plan)
        if mapping_master["work_items"] != master["work_items"]:
            raise PlanError("list/mapping bundle normalization differs")

        # Exercise the canonical cells schema and its complete product
        # denominator.  Router controls and historical anchors are real work
        # items; omitting either would let a selector or an old winning region
        # escape the final sweep.
        cells = []
        for qtype in sorted(QTYPES):
            for operator, count in (("dense", 143), ("grouped", 52)):
                for ordinal in range(count):
                    key = f"{operator}-{ordinal}"
                    cells.append({
                        "cell_key": f"q{qtype}/{operator}/{key}",
                        "qtype": qtype,
                        "operator": operator,
                        "workload_key": key,
                        "source_class": "real-inventory",
                    })
            cells.append({
                "cell_key": f"q{qtype}/grouped/router-control",
                "qtype": qtype, "operator": "grouped",
                "workload_key": "router-control",
                "source_class": "router-control",
            })
        cells.append({
            "cell_key": "q12/dense/historical-anchor",
            "qtype": 12, "operator": "dense",
            "workload_key": "historical-anchor",
            "source_class": "historical-anchor",
        })
        cell_plan_doc = {
            "schema": "fixture-cell-plan",
            "workload_denominator": {"dense_real": 143,
                                     "grouped_real": 52,
                                     "grouped_router_controls": 1,
                                     "q4_historical_anchor_only": 1,
                                     "format_workload_cells": 981},
            "cells": cells,
        }
        cell_plan = root / "cell-plan.json"
        write_frozen_json(cell_plan, cell_plan_doc)
        formal, formal_schema = _workloads(cell_plan_doc)
        if formal_schema != "CELLS_COMPLETE_PRODUCT_DENOMINATOR_V1" or any(
                len(formal[(qtype, "dense")]) !=
                    (144 if qtype == 12 else 143) or
                len(formal[(qtype, "grouped")]) != 53
                for qtype in QTYPES):
            raise PlanError("self-test canonical cell denominator differs")
        if (formal[(12, "dense")].count("historical-anchor") != 1 or
                any(keys.count("router-control") != 1
                    for (qtype, operator), keys in formal.items()
                    if operator == "grouped")):
            raise PlanError("control/anchor workload left complete denominator")
        formal_master = make_master(bundle, cell_plan)
        if formal_master["denominator"]["work_items"] != 340:
            raise PlanError("self-test canonical cell work-product differs")

        # A qtype-scoped master is derived from, and remains bound to, the
        # complete catalog and workload plan.  Its item IDs are an exact
        # subset of the full master, so no payload or runner schema changes.
        scoped_bundle_doc = copy.deepcopy(bundle_doc)
        scoped_bundle_doc["shards"].append({
            "shard_key": "fq-q12-grouped-000",
            "route": "fully-quantized", "operator": "grouped",
            "qtype": 12, "manifest_sha256": "6" * 64,
            "parent_ids": ["fq-q12-grouped-a"],
        })
        scoped_bundle = root / "scoped-bundle.json"
        write_frozen_json(scoped_bundle, scoped_bundle_doc)
        full_for_scope = make_master(scoped_bundle, cell_plan)
        q4_master = make_master(scoped_bundle, cell_plan, qtypes=[12])
        if q4_master.get("execution_scope") != {
                "schema": MASTER_SCOPE_SCHEMA, "qtypes": [12]}:
            raise PlanError("self-test qtype scope authority differs")
        if (q4_master["denominator"]["work_items"] != 197 or
                q4_master["denominator"]["workloads_by_qtype_operator"] != {
                    "q12/dense": 144, "q12/grouped": 53} or
                {row["qtype"] for row in q4_master["work_items"]} != {12}):
            raise PlanError("self-test Q4 work-product differs")
        full_ids = {row["work_item_id"] for row in full_for_scope["work_items"]}
        if not {row["work_item_id"] for row in q4_master["work_items"]} < full_ids:
            raise PlanError("self-test scoped item IDs are not a strict subset")
        validate_master(q4_master, scoped_bundle, cell_plan)
        q4_master_path = root / "q4-master.json"
        q4_assignment_path = root / "q4-assignment.json"
        write_frozen_json(q4_master_path, q4_master)
        q4_assignment = make_assignment(
            q4_master, file_sha(q4_master_path), 1)
        write_frozen_json(q4_assignment_path, q4_assignment)
        q4_selections = root / "q4-selections"
        write_worker_selections(
            q4_selections, q4_master, q4_assignment,
            master_sha256=file_sha(q4_master_path),
            assignment_sha256=file_sha(q4_assignment_path))
        q4_selection = load_json(
            q4_selections / "worker-0.json", "Q4 worker selection")
        if {row["qtype"] for row in q4_selection["work_items"]} != {12}:
            raise PlanError("self-test Q4 selection escaped its scope")

        _expect_red(
            "empty qtype scope", lambda: make_master(
                scoped_bundle, cell_plan, qtypes=[]), "nonempty")
        _expect_red(
            "duplicate qtype scope", lambda: make_master(
                scoped_bundle, cell_plan, qtypes=[12, 12]), "duplicate")
        _expect_red(
            "unsupported qtype scope", lambda: make_master(
                scoped_bundle, cell_plan, qtypes=[9]), "unsupported")
        planted_scope = copy.deepcopy(q4_master)
        planted_scope["execution_scope"]["qtypes"] = [10]
        _expect_red(
            "planted qtype scope",
            lambda: validate_master(planted_scope, scoped_bundle, cell_plan),
            "differs")
        (q4_selections / "manual.json").write_text("{}\n", encoding="utf-8")
        _expect_red(
            "manual selection entry",
            lambda: write_worker_selections(
                q4_selections, q4_master, q4_assignment,
                master_sha256=file_sha(q4_master_path),
                assignment_sha256=file_sha(q4_assignment_path)),
            "unexpected entries")

        device_path = root / "devices.json"
        device_doc = {
            "schema": DEVICE_SCHEMA,
            "workers": [{
                "worker_id": worker,
                "identity_sha256": f"{worker + 4:x}" * 64,
                "homogeneity_key": "a" * 64,
            } for worker in range(3)],
        }
        write_frozen_json(device_path, device_doc)
        identities = validate_device_authority(device_doc, 3)
        results = [{
            "schema": RESULT_SCHEMA,
            "worker_id": row["worker_id"],
            "bundle_sha256": master["bundle_sha256"],
            "workload_plan_sha256": master["workload_plan_sha256"],
            "master_sha256": file_sha(master_path),
            "assignment_sha256": file_sha(assignment_path),
            "device_homogeneity_sha256": file_sha(device_path),
            "device_identity_sha256": identities[row["worker_id"]],
            "completed_work_item_ids": list(row["work_item_ids"]),
        } for row in assignment["workers"]]
        validate_results(
            results, assignment,
            master_sha256=file_sha(master_path),
            assignment_sha256=file_sha(assignment_path),
            device_authority_sha256=file_sha(device_path),
            device_identities=identities)

        gap = copy.deepcopy(assignment)
        gap["workers"][0]["work_item_ids"].pop()
        _expect_red(
            "assignment gap",
            lambda: validate_assignment(gap, master, file_sha(master_path)),
            "gap/extra")
        overlap = copy.deepcopy(assignment)
        overlap["workers"][1]["work_item_ids"].append(
            overlap["workers"][0]["work_item_ids"][0])
        _expect_red(
            "assignment overlap",
            lambda: validate_assignment(
                overlap, master, file_sha(master_path)),
            "overlap/duplicate")
        duplicate = copy.deepcopy(bundle_doc)
        duplicate["shards"].append({
            "shard_id": "sf-q10-dense-001",
            "route": "scalefirst", "operator": "dense", "qtype": 10,
            "manifest_sha256": "4" * 64, "parent_ids": [0],
        })
        duplicate_path = root / "duplicate-bundle.json"
        write_frozen_json(duplicate_path, duplicate)
        _expect_red(
            "duplicate parent",
            lambda: make_master(duplicate_path, plan),
            "duplicated across")

        stale_plan_doc = copy.deepcopy(plan_doc)
        stale_plan_doc["dense"][0]["key"] = "dense-stale"
        stale_plan = root / "stale-plan.json"
        write_frozen_json(stale_plan, stale_plan_doc)
        _expect_red(
            "stale plan",
            lambda: validate_master(master, bundle, stale_plan),
            "differs")
        stale_bundle_doc = copy.deepcopy(bundle_doc)
        next(row for row in stale_bundle_doc["shards"]
             if row.get("shard_key") == "fq-q12-dense-000")[
                 "manifest_sha256"] = "5" * 64
        stale_bundle = root / "stale-bundle.json"
        write_frozen_json(stale_bundle, stale_bundle_doc)
        _expect_red(
            "stale bundle",
            lambda: validate_master(master, stale_bundle, plan),
            "differs")
        stale_device = copy.deepcopy(results)
        stale_device[1]["device_homogeneity_sha256"] = "b" * 64
        _expect_red(
            "stale device binding",
            lambda: validate_results(
                stale_device, assignment,
                master_sha256=file_sha(master_path),
                assignment_sha256=file_sha(assignment_path),
                device_authority_sha256=file_sha(device_path),
                device_identities=identities),
            "stale device_homogeneity_sha256")
        heterogeneous = copy.deepcopy(device_doc)
        heterogeneous["workers"][2]["homogeneity_key"] = "c" * 64
        _expect_red(
            "heterogeneous devices",
            lambda: validate_device_authority(heterogeneous, 3),
            "not device-homogeneous")

        missing_control_doc = copy.deepcopy(cell_plan_doc)
        missing_control_doc["cells"] = [
            row for row in missing_control_doc["cells"]
            if row["cell_key"] != "q10/grouped/router-control"]
        _expect_red(
            "missing router control",
            lambda: _workloads(missing_control_doc),
            "complete workload denominator differs")
        missing_anchor_doc = copy.deepcopy(cell_plan_doc)
        missing_anchor_doc["cells"] = [
            row for row in missing_anchor_doc["cells"]
            if row["source_class"] != "historical-anchor"]
        _expect_red(
            "missing historical anchor",
            lambda: _workloads(missing_anchor_doc),
            "complete workload denominator differs")

    print(
        "[kpack-discovery-worker-plan:self-test] PASS "
        "master=5 workers=3 exactly-once list+mapping-bundles "
        "shared+cells-plans complete-controls+anchors Q4-scope=197; "
        "gap+overlap+duplicate+stale-plan+stale-bundle+stale-device+"
        "heterogeneous-device+missing-control+missing-anchor+scope-plants "
        "negatives=RED")


def self_test_command(_args: argparse.Namespace) -> int:
    self_test()
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create")
    create.add_argument("--bundle", type=Path, required=True)
    create.add_argument("--plan", type=Path, required=True)
    create.add_argument("--workers", type=int, required=True)
    create.add_argument("--master", type=Path, required=True)
    create.add_argument("--assignment", type=Path, required=True)
    create.add_argument(
        "--qtype", type=int, action="append", choices=sorted(QTYPES),
        help="limit execution to this qtype; repeat to select more than one")
    create.add_argument(
        "--selection-dir", type=Path,
        help="also write one deterministic JSON selection per worker")
    create.set_defaults(func=create_command)

    assignment = commands.add_parser("validate-assignment")
    assignment.add_argument("--bundle", type=Path, required=True)
    assignment.add_argument("--plan", type=Path, required=True)
    assignment.add_argument("--master", type=Path, required=True)
    assignment.add_argument("--assignment", type=Path, required=True)
    assignment.set_defaults(func=validate_assignment_command)

    results = commands.add_parser("validate-results")
    results.add_argument("--bundle", type=Path, required=True)
    results.add_argument("--plan", type=Path, required=True)
    results.add_argument("--master", type=Path, required=True)
    results.add_argument("--assignment", type=Path, required=True)
    results.add_argument("--device-homogeneity", type=Path, required=True)
    results.add_argument("--result", type=Path, action="append", required=True)
    results.set_defaults(func=validate_results_command)

    test = commands.add_parser("self-test")
    test.set_defaults(func=self_test_command)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        return args.func(args)
    except PlanError as exc:
        fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
