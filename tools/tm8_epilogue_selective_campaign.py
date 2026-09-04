#!/usr/bin/env python3
"""Materialize the exact build plan for the TM8 epilogue invalidation.

This is an adapter between ``plan_tm8_epilogue_fix_scope.py`` and the
existing distributed K-pack builders.  It does not invent parent ranges or
workloads: every selected binary range comes from the canonical 32-parent
shard authorities, and every runtime workload key comes from the validated
TM8 scope.  The resulting plan can therefore reuse the normal shard
generator, binary receipt, partition bundle, catalog, and worker machinery
without rebuilding or executing an unaffected shard.

Typical flow::

  python3 tools/plan_tm8_epilogue_fix_scope.py emit --out scope.json
  python3 tools/tm8_epilogue_selective_campaign.py emit \
      --scope scope.json --partitions 8 --output build-plan.json

Run ``build_kpack_discovery_partition_worker.sh`` once for each build worker,
all against that same immutable plan.  Once its 16 route/partition artifacts
are published, freeze the exact 8-device execution ledger::

  python3 tools/tm8_epilogue_selective_campaign.py finalize \
      --plan build-plan.json --publish-root /path/to/published \
      --workers 8 --output campaign

Use ``campaign/selections/worker-N.json`` together with the corresponding
``campaign/artifact-roots/worker-N.tsv`` as inputs to the existing prebuilt
worker.  Repeating ``--qtype 12`` while emitting the plan creates the exact
Q4-only 83-shard/5,296-atom proving subset.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, NoReturn


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kpack_discovery_build_partitions as partitions  # noqa: E402
import plan_tm8_epilogue_fix_scope as tm8_scope  # noqa: E402


PLAN_SCHEMA = "quactlize.tm8-epilogue-selective-build-partitions.v1"
SCOPE_RECORD_SCHEMA = "quactlize.tm8-epilogue-selective-scope-record.v1"
ASSIGNMENT = "SELECTIVE_PAIR_LOCAL_ROUND_ROBIN_V1"
DEFAULT_PARTITIONS = 8
QTYPES = (10, 11, 12, 13, 14)
ROUTES = ("scalefirst", "fully-quantized")
OPERATORS = ("dense", "grouped")
_VALID_SCOPE_DIGESTS: set[tuple[str, str]] = set()
_VALID_PLAN_DIGESTS: set[tuple[str, str]] = set()


class CampaignError(ValueError):
    """The selective build/run campaign differs from live authority."""


def _live_authority_fingerprint() -> str:
    records = [
        tm8_scope._git_oid(tm8_scope.ROOT),
        tm8_scope._git_oid(tm8_scope.ROOT / "third_party/actlize"),
    ]
    records.extend(
        [relative, tm8_scope._sha256(tm8_scope.ROOT / relative)]
        for relative in tm8_scope.AUTHORITY_PATHS)
    return partitions.digest(records)


def fail(message: str) -> NoReturn:
    raise SystemExit(f"tm8 epilogue selective campaign: {message}")


def normalize_qtypes(values: Any) -> tuple[int, ...]:
    if values is None:
        return QTYPES
    if not isinstance(values, (list, tuple, set, frozenset)) or not values:
        raise CampaignError("qtypes must be one nonempty collection")
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in values):
        raise CampaignError("qtypes must contain integers")
    result = tuple(sorted(values))
    if len(result) != len(set(result)):
        raise CampaignError("qtypes contain a duplicate")
    unsupported = sorted(set(result) - set(QTYPES))
    if unsupported:
        raise CampaignError(f"unsupported qtypes {unsupported}")
    return result


def scope_record(scope: dict[str, Any], qtypes: Any = None) -> dict[str, Any]:
    selected_qtypes = normalize_qtypes(qtypes)
    rows = [row for row in scope["rebuild_shards"]
            if row["qtype"] in selected_qtypes]
    return {
        "schema": SCOPE_RECORD_SCHEMA,
        "source_schema": tm8_scope.SCHEMA,
        "canonical_sha256": partitions.digest(scope),
        "qtypes": list(selected_qtypes),
        "affected_shards": len(rows),
        "recompiled_parents": sum(row["compiled_parents"] for row in rows),
        "runtime_candidate_work_items": sum(
            row["runtime_candidate_work_items"] for row in rows),
        "shard_keys_sha256": partitions.digest(sorted(
            f"{row['route']}:{row['shard_key']}" for row in rows)),
    }


def _scope_document(value: dict[str, Any]) -> dict[str, Any]:
    identity = partitions.digest(value)
    cache_key = (identity, _live_authority_fingerprint())
    if cache_key in _VALID_SCOPE_DIGESTS:
        return value
    try:
        tm8_scope.validate_plan(value)
    except (KeyError, TypeError, tm8_scope.ScopeError) as error:
        raise CampaignError(f"TM8 scope differs: {error}") from error
    _VALID_SCOPE_DIGESTS.add(cache_key)
    return value


def read_scope(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read TM8 scope {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError("TM8 scope must be a JSON object")
    return _scope_document(value)


def make_plan(scope: dict[str, Any], partition_count: int,
              qtypes: Any = None) -> dict[str, Any]:
    scope = _scope_document(scope)
    selected_qtypes = normalize_qtypes(qtypes)
    if (isinstance(partition_count, bool) or
            not 1 <= partition_count <= partitions.MAX_PARTITIONS):
        raise CampaignError(
            f"partition count must be in [1,{partitions.MAX_PARTITIONS}]")

    # Start from the normal native rows so parent IDs, symbols, layouts, and
    # range boundaries retain exactly the same representation as a full build.
    native = partitions.make_plan(partition_count)["shards"]
    native_by_key = {row["shard_key"]: row for row in native}
    if len(native_by_key) != len(native):
        raise CampaignError("native shard authority contains a duplicate")

    selected: list[dict[str, Any]] = []
    for scoped in scope["rebuild_shards"]:
        if scoped["qtype"] not in selected_qtypes:
            continue
        key = f"{scoped['route']}:{scoped['shard_key']}"
        row = native_by_key.get(key)
        if row is None:
            raise CampaignError(f"scope shard {key} is absent from native authority")
        expected = {
            "route": scoped["route"],
            "qtype": scoped["qtype"],
            "operator": scoped["operator"],
            "native_shard_key": scoped["shard_key"],
            "parent_begin": scoped["parent_begin"],
            "parent_end": scoped["parent_end"],
            "parent_count": scoped["compiled_parents"],
        }
        if any(row[field] != value for field, value in expected.items()):
            raise CampaignError(f"scope/native range differs for {key}")
        workload_set = scoped["runtime_workload_set"]
        workload = scope["workload_sets"].get(workload_set)
        if (not isinstance(workload, dict) or
                workload.get("qtype") != scoped["qtype"] or
                workload.get("operator") != scoped["operator"] or
                workload.get("count") != scoped["runtime_candidate_work_items"] or
                not isinstance(workload.get("workload_keys"), list) or
                len(workload["workload_keys"]) != workload["count"] or
                len(set(workload["workload_keys"])) != workload["count"]):
            raise CampaignError(f"runtime workload set differs for {key}")
        copied = copy.deepcopy(row)
        copied.update({
            "runtime_workload_set": workload_set,
            "runtime_workload_keys": list(workload["workload_keys"]),
            "affected_parent_count": scoped["affected_parent_count"],
            "affected_parent_ids_sha256": partitions.digest(
                scoped["affected_parent_ids"]),
        })
        selected.append(copied)

    selected.sort(key=lambda row: (
        ROUTES.index(row["route"]), row["qtype"],
        OPERATORS.index(row["operator"]), row["parent_begin"]))
    if not selected:
        raise CampaignError("selective scope contains no shards")
    keys = [row["shard_key"] for row in selected]
    if len(keys) != len(set(keys)):
        raise CampaignError("selective scope contains a duplicate shard")

    # Stripe each route/qtype/operator bucket independently.  This retains
    # deterministic affinity and prevents a large bucket from starving a
    # format/operator on one logical partition.
    ordinals: dict[tuple[str, int, str], int] = {}
    for row in selected:
        bucket = (row["route"], row["qtype"], row["operator"])
        ordinal = ordinals.get(bucket, 0)
        row["partition_id"] = ordinal % partition_count
        row["bucket_ordinal"] = ordinal
        ordinals[bucket] = ordinal + 1
    missing_route_partitions = [
        (route, partition_id) for route in ROUTES
        for partition_id in range(partition_count)
        if not any(row["route"] == route and
                   row["partition_id"] == partition_id for row in selected)]
    if missing_route_partitions:
        raise CampaignError(
            "partition count leaves an empty route artifact: "
            f"{missing_route_partitions}")

    by_route = {
        route: sum(row["route"] == route for row in selected)
        for route in ROUTES
    }
    parents_by_route = {
        route: sum(row["parent_count"] for row in selected
                   if row["route"] == route)
        for route in ROUTES
    }
    runtime_by_route = {
        route: sum(len(row["runtime_workload_keys"]) for row in selected
                   if row["route"] == route)
        for route in ROUTES
    }
    plan_partitions = []
    for partition_id in range(partition_count):
        rows = [row for row in selected
                if row["partition_id"] == partition_id]
        plan_partitions.append({
            "partition_id": partition_id,
            "shards": len(rows),
            "parents": sum(row["parent_count"] for row in rows),
            "runtime_candidate_work_items": sum(
                len(row["runtime_workload_keys"]) for row in rows),
            "shards_by_route": {
                route: sum(row["route"] == route for row in rows)
                for route in ROUTES},
            "parents_by_route": {
                route: sum(row["parent_count"] for row in rows
                           if row["route"] == route)
                for route in ROUTES},
            "shard_keys_sha256": partitions.digest(sorted(
                row["shard_key"] for row in rows)),
        })

    record = scope_record(scope, selected_qtypes)
    denominator = {
        "routes": list(ROUTES),
        "qtypes": list(selected_qtypes),
        "operators": list(OPERATORS),
        "shards": len(selected),
        "parents": sum(row["parent_count"] for row in selected),
        "affected_parents": sum(
            row["affected_parent_count"] for row in selected),
        "runtime_candidate_work_items": sum(
            len(row["runtime_workload_keys"]) for row in selected),
        "shards_by_route": by_route,
        "parents_by_route": parents_by_route,
        "runtime_candidate_work_items_by_route": runtime_by_route,
    }
    if (denominator["shards"] != record["affected_shards"] or
            denominator["parents"] != record["recompiled_parents"] or
            denominator["runtime_candidate_work_items"] !=
            record["runtime_candidate_work_items"]):
        raise CampaignError("selective plan denominator differs from TM8 scope")
    return {
        "schema": PLAN_SCHEMA,
        "selective_scope": record,
        "partition_count": partition_count,
        "assignment": ASSIGNMENT,
        "native_max_parents_per_binary": tm8_scope.PARENTS_PER_BINARY,
        "denominator": denominator,
        "partitions": plan_partitions,
        "shards": selected,
    }


def validate_plan(document: dict[str, Any],
                  scope: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(document, dict) or document.get("schema") != PLAN_SCHEMA:
        raise CampaignError("selective build-plan schema differs")
    identity = partitions.digest(document)
    cache_key = (identity, _live_authority_fingerprint())
    if scope is None and cache_key in _VALID_PLAN_DIGESTS:
        return document
    record = document.get("selective_scope")
    if not isinstance(record, dict):
        raise CampaignError("selective build plan has no scope record")
    qtypes = normalize_qtypes(record.get("qtypes"))
    live_scope = tm8_scope.make_plan() if scope is None else _scope_document(scope)
    expected = make_plan(live_scope, document.get("partition_count"), qtypes)
    if document != expected:
        raise CampaignError("selective build plan differs from live authority")
    if scope is None:
        _VALID_PLAN_DIGESTS.add(cache_key)
    return document


def _validate_against(document: dict[str, Any],
                      expected: dict[str, Any]) -> None:
    if document != expected:
        raise CampaignError("selective build plan differs from live authority")


def _encoded_plan_sha(plan: dict[str, Any]) -> str:
    encoded = (json.dumps(
        plan, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def _synthetic_catalog(plan: dict[str, Any], scope: dict[str, Any]
                       ) -> dict[str, Any]:
    """Create metadata-only bytes for local schema/worker self-tests."""
    plan_sha = _encoded_plan_sha(plan)
    source = scope["source"]
    source_sha = source["repository_git_commit"]
    source_tree = subprocess.check_output(
        ["git", "-C", str(tm8_scope.ROOT), "rev-parse", "HEAD^{tree}"],
        text=True).strip()
    actlize = source["actlize_git_commit"]
    sdk_file = {
        "path": "synthetic", "size": 1, "sha256": "1" * 64,
        "symlink_target": None,
    }
    common = {
        "source_sha": source_sha,
        "source_tree": source_tree,
        "submodules": [{
            "path": "third_party/actlize", "gitlink": actlize,
            "current": actlize,
        }],
        "sdk": {
            "receipt": sdk_file, "compiler": sdk_file,
            "inspector": sdk_file, "runtime_libraries": [sdk_file],
        },
    }
    artifact = lambda route, partition_id: (
        f"kpack-discovery/{source_sha}/{plan_sha[:16]}/{route}/"
        f"p{partition_id:02d}-of-{plan['partition_count']:02d}")
    shard_rows = []
    partition_rows = []
    for route in ROUTES:
        for partition_id in range(plan["partition_count"]):
            rows = [row for row in plan["shards"]
                    if row["route"] == route and
                    row["partition_id"] == partition_id]
            artifact_id = artifact(route, partition_id)
            partition_rows.append({
                "route": route, "partition_id": partition_id,
                "artifact_id": artifact_id,
                "partition_manifest": {
                    "path": "partition-bundle.json", "size": 1,
                    "sha256": partitions.digest([artifact_id, "partition"]),
                },
                "shards": len(rows),
                "parents": sum(row["parent_count"] for row in rows),
                "shard_keys_sha256": partitions.digest(sorted(
                    row["shard_key"] for row in rows)),
            })
            for row in rows:
                key = row["shard_key"]
                manifest_sha = partitions.digest([key, "manifest"])
                file_record = lambda label: {
                    "path": f"payloads/{key}/{label}", "size": 1,
                    "sha256": partitions.digest([key, label]),
                }
                shard_rows.append({
                    **row,
                    "artifact_id": artifact_id,
                    "mapping_id": partitions.sf_analyzer.MAPPING[row["layout"]],
                    "manifest_sha256": manifest_sha,
                    "files": {
                        "manifest": {
                            **file_record("manifest.json"),
                            "sha256": manifest_sha,
                        },
                        "binary": file_record("binary"),
                        "binary_receipt": file_record("binary-receipt.json"),
                    },
                    "device_arch": "PPU 0010",
                    "inspector_output_sha256": partitions.digest(
                        [key, "inspector"]),
                })
    return {
        "schema": partitions.CATALOG_SCHEMA,
        **common,
        "partition_plan_sha256": plan_sha,
        "partition_count": plan["partition_count"],
        "partition_assignment": plan["assignment"],
        "payload_residency": "PER_WORKER_PARTITION_FETCH_AND_VERIFY",
        "denominator": partitions._expected_catalog_denominator(plan),
        "partitions": partition_rows,
        "shards": sorted(shard_rows, key=lambda row: row["shard_key"]),
        "selective_scope": plan["selective_scope"],
    }


def finalize(plan_path: Path, publish_root: Path, workers: int,
             output: Path) -> dict[str, Any]:
    """Merge built partitions and freeze the exact worker execution ledger."""
    if isinstance(workers, bool) or workers <= 0:
        raise CampaignError("worker count must be positive")
    plan = partitions.read_plan(plan_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise CampaignError("finalize requires one TM8 selective build plan")
    if workers > plan["partition_count"]:
        raise CampaignError("worker count exceeds selective partitions")
    if publish_root.is_symlink():
        raise CampaignError("publish root may not be a symlink")
    try:
        publish_root = publish_root.resolve(strict=True)
    except OSError as error:
        raise CampaignError(f"cannot resolve publish root: {error}") from error
    if not publish_root.is_dir():
        raise CampaignError("publish root must be a directory")
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise CampaignError("existing output is not a regular directory")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve(strict=True)

    source_sha = subprocess.check_output(
        ["git", "-C", str(tm8_scope.ROOT), "rev-parse", "HEAD"],
        text=True).strip()
    manifests = [
        publish_root / source_sha / route / f"p{partition_id:02d}" /
        "partition-bundle.json"
        for route in ROUTES
        for partition_id in range(plan["partition_count"])
    ]
    missing = [str(path) for path in manifests
               if path.is_symlink() or not path.is_file()]
    if missing:
        raise CampaignError(
            f"published partition manifest union is incomplete: {missing}")

    catalog_path = output / "catalog.json"
    catalog = partitions.make_catalog(plan_path, manifests)
    partitions.write_frozen(catalog_path, catalog)

    import kpack_discovery_worker_plan as worker_plan
    import plan_fq_kpack_route_optimal as workload_plan
    workload_path = output / "workload-plan.json"
    workload_document = workload_plan.materialize()
    try:
        workload_plan.validate_plan(workload_document)
    except (AssertionError, KeyError, workload_plan.PlanError) as error:
        raise CampaignError(f"canonical workload plan differs: {error}") from error
    partitions.write_frozen(workload_path, workload_document)

    master_path = output / "master.json"
    assignment_path = output / "assignment.json"
    selection_dir = output / "selections"
    master = worker_plan.make_master(catalog_path, workload_path)
    worker_plan.write_frozen_json(master_path, master)
    assignment = worker_plan.make_assignment(
        master, worker_plan.file_sha(master_path), workers)
    worker_plan.write_frozen_json(assignment_path, assignment)
    worker_plan.validate_assignment(
        assignment, master, worker_plan.file_sha(master_path))
    worker_plan.write_worker_selections(
        selection_dir, master, assignment,
        master_sha256=worker_plan.file_sha(master_path),
        assignment_sha256=worker_plan.file_sha(assignment_path))

    partitions_by_artifact = {
        row["artifact_id"]: row for row in catalog["partitions"]}
    artifact_root_dir = output / "artifact-roots"
    expected_artifact_files = {
        f"worker-{worker_id}.tsv" for worker_id in range(workers)}
    if artifact_root_dir.exists():
        if artifact_root_dir.is_symlink() or not artifact_root_dir.is_dir():
            raise CampaignError("artifact-root output is not a regular directory")
        unexpected = {
            path.name for path in artifact_root_dir.iterdir()
        } - expected_artifact_files
        if unexpected:
            raise CampaignError(
                f"artifact-root output contains unexpected files {sorted(unexpected)}")
    else:
        artifact_root_dir.mkdir(parents=True)
    for worker in assignment["workers"]:
        worker_id = worker["worker_id"]
        selection = worker_plan.make_worker_selection(
            master, assignment, worker_id,
            master_sha256=worker_plan.file_sha(master_path),
            assignment_sha256=worker_plan.file_sha(assignment_path))
        lines = []
        for artifact_id in selection["artifact_ids"]:
            record = partitions_by_artifact.get(artifact_id)
            if record is None:
                raise CampaignError("worker selected a foreign artifact")
            root = (publish_root / source_sha / record["route"] /
                    f"p{record['partition_id']:02d}")
            lines.append(f"{artifact_id}\t{root}\n")
        worker_plan.write_frozen_text(
            artifact_root_dir / f"worker-{worker_id}.tsv", "".join(lines))

    result = {
        "schema": "quactlize.tm8-epilogue-selective-finalization.v1",
        "source_sha": source_sha,
        "selective_scope": plan["selective_scope"],
        "partition_plan_sha256": partitions.file_sha(plan_path),
        "catalog_sha256": partitions.file_sha(catalog_path),
        "workload_plan_sha256": partitions.file_sha(workload_path),
        "master_sha256": partitions.file_sha(master_path),
        "assignment_sha256": partitions.file_sha(assignment_path),
        "workers": workers,
        "binary_shards": master["denominator"]["binary_shards"],
        "work_items": master["denominator"]["work_items"],
    }
    partitions.write_frozen(output / "finalization.json", result)
    return result


def bind_devices(campaign: Path) -> dict[str, Any]:
    """Create the worker homogeneity authority from eight live probe files."""
    if campaign.is_symlink():
        raise CampaignError("campaign directory may not be a symlink")
    try:
        campaign = campaign.resolve(strict=True)
    except OSError as error:
        raise CampaignError(f"cannot resolve campaign directory: {error}") from error
    if not campaign.is_dir():
        raise CampaignError("campaign path is not a directory")

    import box_identity_schema
    import kpack_discovery_worker_plan as worker_plan
    master_path = campaign / "master.json"
    assignment_path = campaign / "assignment.json"
    catalog_path = campaign / "catalog.json"
    workload_path = campaign / "workload-plan.json"
    master = worker_plan.load_json(master_path, "master")
    worker_plan.validate_master(master, catalog_path, workload_path)
    assignment = worker_plan.load_json(assignment_path, "assignment")
    worker_plan.validate_assignment(
        assignment, master, worker_plan.file_sha(master_path))

    rows = []
    homogeneous_keys = set()
    for worker_id in range(assignment["worker_count"]):
        path = campaign / f"worker-{worker_id}-device.json"
        if path.is_symlink() or not path.is_file():
            raise CampaignError(f"worker {worker_id} device identity is missing")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            values, _sources = box_identity_schema.values_and_sources(document)
        except (OSError, json.JSONDecodeError,
                box_identity_schema.IdentityProbeError) as error:
            raise CampaignError(
                f"worker {worker_id} device identity differs: {error}") from error
        device = document["device_probe"]
        candidates = device.get("candidates")
        if (device.get("status") != "measured" or
                not isinstance(candidates, list) or len(candidates) != 1):
            raise CampaignError(
                f"worker {worker_id} lacks one measured device")
        candidate = candidates[0]
        homogeneous = {
            "device_model": values["device_model"],
            "driver_version": values["driver_version"],
            "sdk_compiler_identity": values["sdk_compiler_identity"],
            "compute_capability": candidate["compute_capability"],
            "compute_units": candidate["compute_units"],
        }
        key = worker_plan.digest(homogeneous)
        homogeneous_keys.add(key)
        rows.append({
            "worker_id": worker_id,
            "identity_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "homogeneity_key": key,
        })
    if len(homogeneous_keys) != 1:
        raise CampaignError("worker devices are not homogeneous")
    result = {"schema": worker_plan.DEVICE_SCHEMA, "workers": rows}
    worker_plan.validate_device_authority(result, assignment["worker_count"])
    worker_plan.write_frozen_json(campaign / "device-homogeneity.json", result)
    return result


def self_test() -> None:
    scope = tm8_scope.make_plan()
    full = make_plan(scope, DEFAULT_PARTITIONS)
    q4 = make_plan(scope, DEFAULT_PARTITIONS, [12])
    wanted = (
        (full, 227, 7264, 13144),
        (q4, 83, 2656, 5296),
    )
    for plan, shards, parents, items in wanted:
        denominator = plan["denominator"]
        if (denominator["shards"], denominator["parents"],
                denominator["runtime_candidate_work_items"]) != (
                    shards, parents, items):
            raise CampaignError("selective denominator drifted")
        validate_plan(plan, scope)
        if any(not row["runtime_workload_keys"] or
               len(row["runtime_workload_keys"]) !=
               len(set(row["runtime_workload_keys"]))
               for row in plan["shards"]):
            raise CampaignError("runtime allowlist is empty or duplicated")
    plants = []
    missing = copy.deepcopy(q4)
    missing["shards"].pop()
    plants.append(missing)
    workload = copy.deepcopy(q4)
    workload["shards"][0]["runtime_workload_keys"].pop()
    plants.append(workload)
    foreign = copy.deepcopy(q4)
    foreign["shards"][0]["shard_key"] += "-foreign"
    plants.append(foreign)
    wrong_scope = copy.deepcopy(q4)
    wrong_scope["selective_scope"]["canonical_sha256"] = "0" * 64
    plants.append(wrong_scope)
    for planted in plants:
        try:
            _validate_against(planted, q4)
        except CampaignError:
            continue
        raise CampaignError("selective negative plant stayed green")

    catalog = _synthetic_catalog(full, scope)
    partitions.validate_catalog_document(catalog)
    planted_catalog = copy.deepcopy(catalog)
    planted_catalog["shards"][0]["runtime_workload_keys"].pop()
    try:
        partitions.validate_catalog_document(planted_catalog)
    except partitions.PartitionError:
        pass
    else:
        raise CampaignError("catalog workload allowlist plant stayed green")
    with tempfile.TemporaryDirectory(prefix="tm8-selective-self-test-") as name:
        root = Path(name)
        catalog_path = root / "catalog.json"
        workload_path = root / "workloads.json"
        catalog_path.write_text(json.dumps(
            catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        import kpack_discovery_worker_plan as worker_plan
        import plan_fq_kpack_route_optimal as workload_plan
        workload_path.write_text(json.dumps(
            workload_plan.materialize(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        master = worker_plan.make_master(catalog_path, workload_path)
        if (master["denominator"]["binary_shards"] != 227 or
                master["denominator"]["work_items"] != 13144 or
                len(master["work_items"]) != 13144):
            raise CampaignError("selective worker master denominator differs")
        q4_master = worker_plan.make_master(
            catalog_path, workload_path, qtypes=[12])
        if (q4_master["denominator"]["binary_shards"] != 83 or
                q4_master["denominator"]["work_items"] != 5296 or
                len(q4_master["work_items"]) != 5296):
            raise CampaignError("selective Q4 worker master denominator differs")
        rogue = copy.deepcopy(catalog)
        rogue["schema"] = "unvalidated-selective-bundle"
        rogue_path = root / "rogue.json"
        rogue_path.write_text(json.dumps(
            rogue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            worker_plan.make_master(rogue_path, workload_path)
        except worker_plan.PlanError:
            pass
        else:
            raise CampaignError("unvalidated workload allowlist stayed green")
    print(
        "[tm8-epilogue-selective-campaign:self-test] PASS "
        "all=227-shards/7264-parents/13144-atoms "
        "q4=83-shards/2656-parents/5296-atoms partitions=8 "
        "master=13144/q4-5296-exact "
        "negatives=missing+workload+foreign+scope+catalog+rogue-RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    emit = commands.add_parser("emit")
    emit.add_argument("--scope", type=Path, required=True)
    emit.add_argument("--partitions", type=int, default=DEFAULT_PARTITIONS)
    emit.add_argument("--qtype", type=int, action="append", choices=QTYPES)
    emit.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--scope", type=Path, required=True)
    validate.add_argument("--plan", type=Path, required=True)
    finish = commands.add_parser("finalize")
    finish.add_argument("--plan", type=Path, required=True)
    finish.add_argument("--publish-root", type=Path, required=True)
    finish.add_argument("--workers", type=int, required=True)
    finish.add_argument("--output", type=Path, required=True)
    bind = commands.add_parser("bind-devices")
    bind.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
            return 0
        if args.command == "finalize":
            result = finalize(
                args.plan, args.publish_root, args.workers, args.output)
            print(
                "[tm8-epilogue-selective-campaign] FINALIZED "
                f"shards={result['binary_shards']} "
                f"atoms={result['work_items']} workers={result['workers']} "
                f"output={args.output}")
            return 0
        if args.command == "bind-devices":
            result = bind_devices(args.campaign)
            print(
                "[tm8-epilogue-selective-campaign] DEVICES "
                f"workers={len(result['workers'])} homogeneous=1 "
                f"output={args.campaign / 'device-homogeneity.json'}")
            return 0
        scope = read_scope(args.scope)
        if args.command == "emit":
            value = make_plan(scope, args.partitions, args.qtype)
            partitions.write_frozen(args.output, value)
            print(
                "[tm8-epilogue-selective-campaign] PLAN "
                f"qtypes={','.join(map(str, value['selective_scope']['qtypes']))} "
                f"partitions={value['partition_count']} "
                f"shards={value['denominator']['shards']} "
                f"parents={value['denominator']['parents']} "
                f"atoms={value['denominator']['runtime_candidate_work_items']} "
                f"output={args.output}")
            return 0
        value = partitions.load_json(args.plan, "selective build plan")
        validate_plan(value, scope)
        print(
            "[tm8-epilogue-selective-campaign] PASS "
            f"shards={value['denominator']['shards']} "
            f"atoms={value['denominator']['runtime_candidate_work_items']} "
            f"plan={args.plan}")
        return 0
    except (CampaignError, OSError, subprocess.SubprocessError,
            ValueError) as error:
        print(f"[tm8-epilogue-selective-campaign] FAIL: {error}",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
