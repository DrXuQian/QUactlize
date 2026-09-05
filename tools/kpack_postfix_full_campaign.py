#!/usr/bin/env python3
"""Freeze and validate one source-clean, five-format K-pack campaign.

This module is the authority adapter for the post-fix exhaustive run.  It
accepts only the complete native build plan; selective catalogs and overlays
are deliberately outside its schema.  Every Q2_K/Q3_K/Q4_K/Q5_K/Q6_K shard
for ScaleFirst and FullyQuantized, dense and grouped, is rebuilt under one
source/submodule/SDK authority before the canonical workload cross product is
assigned to workers.

The device workers and result aggregator remain the existing generic tools.
This module only freezes their inputs, proves the exact denominator, binds the
published partition roots, and records homogeneous device identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import box_identity_schema  # noqa: E402
import kpack_discovery_build_partitions as partitions  # noqa: E402
import kpack_discovery_worker_plan as worker_plan  # noqa: E402
import kpack_global_build_preflight as global_preflight  # noqa: E402
import materialize_kpack_discovery_workloads as workloads  # noqa: E402
import plan_fq_kpack_route_optimal as route_plan  # noqa: E402


SCHEMA = "quactlize.kpack-postfix-full-finalization.v1"
DEFAULT_PARTITIONS = 32
DEFAULT_RUNTIME_WORKERS = 8
QTYPES = (10, 11, 12, 13, 14)
ROUTES = ("scalefirst", "fully-quantized")
OPERATORS = ("dense", "grouped")

# These constants are intentionally exact.  Adding or removing a static
# tactic, shard, real workload, router control, or Q4 historical anchor makes
# the campaign stop here until its denominator is reviewed and updated.
EXPECTED = {
    "binary_shards": 2211,
    "parents": 70483,
    "shards_by_route": {
        "scalefirst": 892,
        "fully-quantized": 1319,
    },
    "parents_by_route": {
        "scalefirst": 28402,
        "fully-quantized": 42081,
    },
    "workload_cells": 1381,
    "workload_cells_by_operator": {"dense": 1001, "grouped": 380},
    "work_items": 339196,
    "work_items_by_route": {
        "scalefirst": 153372,
        "fully-quantized": 185824,
    },
    "work_items_by_qtype": {
        "10": 55556,
        "11": 19018,
        "12": 203018,
        "13": 17990,
        "14": 43614,
    },
}


class CampaignError(ValueError):
    """The campaign is incomplete, stale, selective, or contradictory."""


def _git(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *arguments], text=True).strip()
    except subprocess.CalledProcessError as error:
        raise CampaignError(f"git authority query failed: {arguments}") from error


def _plain_directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise CampaignError(f"{label} may not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CampaignError(f"cannot resolve {label}: {error}") from error
    if not resolved.is_dir():
        raise CampaignError(f"{label} is not a directory")
    return resolved


def _actlize_sha(catalog: dict[str, Any]) -> str:
    rows = [row.get("current") for row in catalog.get("submodules", [])
            if isinstance(row, dict) and
            row.get("path") == "third_party/actlize"]
    if len(rows) != 1 or not isinstance(rows[0], str) or not rows[0]:
        raise CampaignError("catalog has no unique actlize authority")
    return rows[0]


def _verify_live_sdk(catalog: dict[str, Any]) -> Path:
    raw_root = os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME")
    if not raw_root:
        raise CampaignError("PPU_SDK is required to bind the campaign SDK")
    try:
        sdk = Path(raw_root).resolve(strict=True)
    except OSError as error:
        raise CampaignError(f"cannot resolve PPU SDK: {error}") from error
    if not sdk.is_dir():
        raise CampaignError("PPU SDK is not a directory")
    records = [catalog["sdk"][field]
               for field in ("receipt", "compiler", "inspector")]
    records.extend(catalog["sdk"]["runtime_libraries"])
    seen = set()
    for record in records:
        relative = record["path"]
        if relative in seen:
            raise CampaignError(f"catalog SDK path is duplicated: {relative}")
        seen.add(relative)
        path = sdk / relative
        expected_target = record["symlink_target"]
        if ((expected_target is None and path.is_symlink()) or
                (expected_target is not None and
                 (not path.is_symlink() or os.readlink(path) != expected_target)) or
                not path.is_file() or path.stat().st_size != record["size"] or
                partitions.file_sha(path) != record["sha256"]):
            raise CampaignError(f"live SDK file differs: {relative}")
    for relative in (catalog["sdk"]["compiler"]["path"],
                     catalog["sdk"]["inspector"]["path"]):
        if not os.access(sdk / relative, os.X_OK):
            raise CampaignError(f"live SDK tool is not executable: {relative}")
    return sdk


def require_clean_source() -> dict[str, Any]:
    try:
        authority = global_preflight.live_authority(ROOT)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        raise CampaignError(f"source authority is not clean: {error}") from error
    repository = authority.get("repository", {})
    if (repository.get("head") != _git("rev-parse", "HEAD") or
            repository.get("tree") != _git("rev-parse", "HEAD^{tree}") or
            repository.get("clean_relevant_worktree") is not True):
        raise CampaignError("source preflight identity differs")
    return authority


def _full_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != partitions.PLAN_SCHEMA:
        raise CampaignError(
            "post-fix campaign requires the complete native build plan; "
            "selective/overlay plans are forbidden")
    try:
        partitions.validate_plan(plan)
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignError(f"complete build plan differs: {error}") from error
    return plan


def derive_denominator(build: dict[str, Any], workload: dict[str, Any]
                       ) -> dict[str, Any]:
    """Derive the runtime cross product without trusting summary constants."""
    _full_plan(build)
    try:
        route_plan.validate_plan(workload)
    except (AssertionError, KeyError, TypeError, route_plan.PlanError) as error:
        raise CampaignError(f"canonical workload plan differs: {error}") from error

    workload_counts: dict[tuple[int, str], int] = {}
    for cell in workload["cells"]:
        key = (int(cell["qtype"]), str(cell["operator"]))
        workload_counts[key] = workload_counts.get(key, 0) + 1
    expected_workload_keys = {
        (qtype, operator) for qtype in QTYPES for operator in OPERATORS
    }
    if set(workload_counts) != expected_workload_keys:
        raise CampaignError("workload qtype/operator product is incomplete")

    shard_counts: dict[tuple[str, int, str], int] = {}
    for shard in build["shards"]:
        key = (str(shard["route"]), int(shard["qtype"]),
               str(shard["operator"]))
        shard_counts[key] = shard_counts.get(key, 0) + 1
    expected_shard_keys = {
        (route, qtype, operator)
        for route in ROUTES for qtype in QTYPES for operator in OPERATORS
    }
    if set(shard_counts) != expected_shard_keys:
        raise CampaignError("build qtype/route/operator product is incomplete")

    item_rows = []
    for (route, qtype, operator), shard_count in sorted(shard_counts.items()):
        workload_count = workload_counts[(qtype, operator)]
        item_rows.append({
            "route": route,
            "qtype": qtype,
            "operator": operator,
            "binary_shards": shard_count,
            "workload_keys": workload_count,
            "work_items": shard_count * workload_count,
        })

    denominator = build["denominator"]
    return {
        "binary_shards": int(denominator["shards"]),
        "parents": int(denominator["parents"]),
        "shards_by_route": dict(denominator["shards_by_route"]),
        "parents_by_route": dict(denominator["parents_by_route"]),
        "workload_cells": len(workload["cells"]),
        "workload_cells_by_operator": {
            operator: sum(
                count for (qtype, observed), count in workload_counts.items()
                if observed == operator)
            for operator in OPERATORS
        },
        "work_items": sum(row["work_items"] for row in item_rows),
        "work_items_by_route": {
            route: sum(row["work_items"] for row in item_rows
                       if row["route"] == route)
            for route in ROUTES
        },
        "work_items_by_qtype": {
            str(qtype): sum(row["work_items"] for row in item_rows
                            if row["qtype"] == qtype)
            for qtype in QTYPES
        },
        "cross_product": item_rows,
    }


def live_denominator(partition_count: int = DEFAULT_PARTITIONS
                     ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    build = partitions.make_plan(partition_count)
    workload = route_plan.materialize()
    observed = derive_denominator(build, workload)
    comparable = {key: observed[key] for key in EXPECTED}
    if comparable != EXPECTED:
        raise CampaignError(
            "post-fix full denominator drifted: "
            f"observed={comparable!r} expected={EXPECTED!r}")
    return build, workload, observed


def emit_plan(output: Path, partition_count: int) -> dict[str, Any]:
    build, _workload, denominator = live_denominator(partition_count)
    partitions.write_frozen(output, build)
    return denominator


def _published_manifests(plan: dict[str, Any], publish_root: Path,
                         source_sha: str) -> list[Path]:
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
            "published partition manifest union is incomplete: "
            + ", ".join(missing[:8]))
    for manifest in manifests:
        try:
            partitions.verify_partition(manifest, manifest.parent)
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise CampaignError(
                f"published partition payload differs: {manifest}: {error}") \
                from error
    return manifests


def _write_artifact_roots(output: Path, publish_root: Path,
                          source_sha: str, catalog: dict[str, Any],
                          master: dict[str, Any], assignment: dict[str, Any],
                          master_path: Path, assignment_path: Path) -> None:
    records = {row["artifact_id"]: row for row in catalog["partitions"]}
    root = output / "artifact-roots"
    expected = {
        f"worker-{worker}.tsv" for worker in range(assignment["worker_count"])
    }
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise CampaignError("artifact-root output is not a regular directory")
        unexpected = {path.name for path in root.iterdir()} - expected
        if unexpected:
            raise CampaignError(
                f"artifact-root output has unexpected files {sorted(unexpected)}")
    else:
        root.mkdir(parents=True)

    for worker in assignment["workers"]:
        worker_id = worker["worker_id"]
        selection = worker_plan.make_worker_selection(
            master, assignment, worker_id,
            master_sha256=worker_plan.file_sha(master_path),
            assignment_sha256=worker_plan.file_sha(assignment_path))
        lines = []
        for artifact_id in selection["artifact_ids"]:
            record = records.get(artifact_id)
            if record is None:
                raise CampaignError("worker selected a foreign artifact")
            artifact_root = (
                publish_root / source_sha / record["route"] /
                f"p{record['partition_id']:02d}")
            if artifact_root.is_symlink() or not artifact_root.is_dir():
                raise CampaignError(
                    f"worker artifact root is missing: {artifact_root}")
            lines.append(f"{artifact_id}\t{artifact_root}\n")
        worker_plan.write_frozen_text(
            root / f"worker-{worker_id}.tsv", "".join(lines))


def finalize(plan_path: Path, publish_root: Path, workers: int,
             output: Path) -> dict[str, Any]:
    require_clean_source()
    if isinstance(workers, bool) or workers <= 0:
        raise CampaignError("runtime worker count must be positive")
    plan = _full_plan(partitions.read_plan(plan_path))
    _build, workload_document, denominator = live_denominator(
        plan["partition_count"])
    if workers > plan["partition_count"]:
        raise CampaignError("runtime workers exceed build partitions")
    publish_root = _plain_directory(publish_root, "publish root")
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise CampaignError("existing campaign output is not a regular directory")
    output.mkdir(parents=True, exist_ok=True)
    output = _plain_directory(output, "campaign output")

    source_sha = _git("rev-parse", "HEAD")
    source_tree = _git("rev-parse", "HEAD^{tree}")
    manifests = _published_manifests(plan, publish_root, source_sha)
    frozen_plan_path = output / "build-plan.json"
    partitions.write_frozen(frozen_plan_path, plan)
    catalog_path = output / "catalog.json"
    catalog = partitions.make_catalog(plan_path, manifests)
    if (catalog.get("selective_scope") is not None or
            catalog.get("source_sha") != source_sha or
            catalog.get("source_tree") != source_tree):
        raise CampaignError("catalog is selective or differs from live source authority")
    _verify_live_sdk(catalog)
    partitions.write_frozen(catalog_path, catalog)

    workload_path = output / "workload-plan.json"
    partitions.write_frozen(workload_path, workload_document)
    workload_root = output / "workloads"
    workload_index = workloads.materialize(workload_path, workload_root)

    master_path = output / "master.json"
    assignment_path = output / "assignment.json"
    master = worker_plan.make_master(catalog_path, workload_path)
    if (master.get("execution_scope") is not None or
            master["denominator"]["binary_shards"] != EXPECTED["binary_shards"] or
            master["denominator"]["work_items"] != EXPECTED["work_items"]):
        raise CampaignError("master is scoped or its full denominator differs")
    worker_plan.write_frozen_json(master_path, master)
    assignment = worker_plan.make_assignment(
        master, worker_plan.file_sha(master_path), workers)
    worker_plan.write_frozen_json(assignment_path, assignment)
    worker_plan.validate_assignment(
        assignment, master, worker_plan.file_sha(master_path))
    worker_plan.write_worker_selections(
        output / "selections", master, assignment,
        master_sha256=worker_plan.file_sha(master_path),
        assignment_sha256=worker_plan.file_sha(assignment_path))
    _write_artifact_roots(
        output, publish_root, source_sha, catalog, master, assignment,
        master_path, assignment_path)

    result = {
        "schema": SCHEMA,
        "source_sha": source_sha,
        "source_tree": source_tree,
        "actlize_sha": _actlize_sha(catalog),
        "sdk_authority_sha256": partitions.digest(catalog["sdk"]),
        "partition_plan_sha256": partitions.file_sha(frozen_plan_path),
        "catalog_sha256": partitions.file_sha(catalog_path),
        "workload_plan_sha256": partitions.file_sha(workload_path),
        "workload_index_sha256": partitions.file_sha(
            workload_root / "index.json"),
        "master_sha256": partitions.file_sha(master_path),
        "assignment_sha256": partitions.file_sha(assignment_path),
        "partition_artifacts": len(catalog["partitions"]),
        "runtime_workers": workers,
        "denominator": denominator,
        "workload_index": workload_index,
        "old_bundle_overlay": False,
    }
    partitions.write_frozen(output / "finalization.json", result)
    validate_campaign(output, require_devices=False)
    return result


def validate_campaign(campaign: Path, *, require_devices: bool) -> dict[str, Any]:
    require_clean_source()
    campaign = _plain_directory(campaign, "campaign")
    final = partitions.load_json(campaign / "finalization.json", "finalization")
    if final.get("schema") != SCHEMA or final.get("old_bundle_overlay") is not False:
        raise CampaignError("campaign finalization schema/overlay policy differs")

    plan_path = campaign / "build-plan.json"
    if plan_path.is_symlink() or not plan_path.is_file():
        raise CampaignError("campaign build plan is missing")
    plan = _full_plan(partitions.read_plan(plan_path))
    _build, _workload, denominator = live_denominator(plan["partition_count"])
    if final.get("denominator") != denominator:
        raise CampaignError("campaign full denominator differs")

    catalog_path = campaign / "catalog.json"
    workload_path = campaign / "workload-plan.json"
    master_path = campaign / "master.json"
    assignment_path = campaign / "assignment.json"
    catalog = partitions.validate_catalog(catalog_path)
    if catalog.get("selective_scope") is not None:
        raise CampaignError("selective catalog is forbidden")
    _verify_live_sdk(catalog)
    workload_index = workloads.validate(workload_path, campaign / "workloads")
    master = worker_plan.load_json(master_path, "master")
    worker_plan.validate_master(master, catalog_path, workload_path)
    if master.get("execution_scope") is not None:
        raise CampaignError("qtype-scoped master is forbidden")
    assignment = worker_plan.load_json(assignment_path, "assignment")
    worker_plan.validate_assignment(
        assignment, master, worker_plan.file_sha(master_path))
    hashes = {
        "partition_plan_sha256": partitions.file_sha(plan_path),
        "catalog_sha256": partitions.file_sha(catalog_path),
        "workload_plan_sha256": partitions.file_sha(workload_path),
        "workload_index_sha256": partitions.file_sha(
            campaign / "workloads" / "index.json"),
        "master_sha256": partitions.file_sha(master_path),
        "assignment_sha256": partitions.file_sha(assignment_path),
    }
    if any(final.get(name) != value for name, value in hashes.items()) or \
            final.get("source_sha") != catalog["source_sha"] or \
            final.get("source_tree") != catalog["source_tree"] or \
            final.get("source_sha") != _git("rev-parse", "HEAD") or \
            final.get("source_tree") != _git("rev-parse", "HEAD^{tree}") or \
            final.get("actlize_sha") != _actlize_sha(catalog) or \
            final.get("sdk_authority_sha256") != partitions.digest(catalog["sdk"]) or \
            final.get("workload_index") != workload_index or \
            final.get("runtime_workers") != assignment["worker_count"] or \
            final.get("partition_artifacts") != len(catalog["partitions"]):
        raise CampaignError("campaign frozen authority hash differs")
    if (master["denominator"]["binary_shards"] != EXPECTED["binary_shards"] or
            master["denominator"]["work_items"] != EXPECTED["work_items"]):
        raise CampaignError("campaign master denominator differs")

    catalog_artifacts = {row["artifact_id"] for row in catalog["partitions"]}
    observed_artifacts = set()
    for worker_id in range(assignment["worker_count"]):
        roots_path = campaign / "artifact-roots" / f"worker-{worker_id}.tsv"
        if roots_path.is_symlink() or not roots_path.is_file():
            raise CampaignError(f"worker {worker_id} artifact roots are missing")
        for line_number, line in enumerate(
                roots_path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split("\t")
            if len(fields) != 2 or not fields[0] or not fields[1]:
                raise CampaignError(
                    f"worker {worker_id} artifact root row {line_number} differs")
            artifact_id, root_value = fields
            if artifact_id in observed_artifacts:
                raise CampaignError("artifact root is duplicated across workers")
            observed_artifacts.add(artifact_id)
            try:
                partitions.verify_catalog_artifact(
                    catalog, artifact_id, Path(root_value))
            except (OSError, KeyError, TypeError, ValueError) as error:
                raise CampaignError(
                    f"artifact {artifact_id} failed live verification: {error}") \
                    from error
    if observed_artifacts != catalog_artifacts:
        raise CampaignError("artifact root union has a gap or foreign entry")

    if require_devices:
        device_path = campaign / "device-homogeneity.json"
        device = worker_plan.load_json(device_path, "device homogeneity")
        worker_plan.validate_device_authority(device, assignment["worker_count"])
        observed = _device_rows(campaign, assignment)
        if device != {"schema": worker_plan.DEVICE_SCHEMA,
                      "workers": observed}:
            raise CampaignError("frozen device homogeneity differs from live probes")
    return final


def _device_rows(campaign: Path, assignment: dict[str, Any]
                 ) -> list[dict[str, Any]]:
    rows = []
    homogeneity_keys = set()
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
        candidates = document["device_probe"].get("candidates")
        if (document["device_probe"].get("status") != "measured" or
                not isinstance(candidates, list) or len(candidates) != 1):
            raise CampaignError(f"worker {worker_id} lacks one measured device")
        device = candidates[0]
        homogeneous = {
            "device_model": values["device_model"],
            "driver_version": values["driver_version"],
            "sdk_compiler_identity": values["sdk_compiler_identity"],
            "compute_capability": device["compute_capability"],
            "compute_units": device["compute_units"],
        }
        key = worker_plan.digest(homogeneous)
        homogeneity_keys.add(key)
        rows.append({
            "worker_id": worker_id,
            "identity_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "homogeneity_key": key,
        })
    if len(homogeneity_keys) != 1:
        raise CampaignError("runtime worker devices are not homogeneous")
    return rows


def bind_devices(campaign: Path) -> dict[str, Any]:
    campaign = _plain_directory(campaign, "campaign")
    final = validate_campaign(campaign, require_devices=False)
    assignment = worker_plan.load_json(
        campaign / "assignment.json", "assignment")
    rows = _device_rows(campaign, assignment)
    result = {"schema": worker_plan.DEVICE_SCHEMA, "workers": rows}
    worker_plan.validate_device_authority(result, assignment["worker_count"])
    worker_plan.write_frozen_json(
        campaign / "device-homogeneity.json", result)
    validate_campaign(campaign, require_devices=True)
    if final["runtime_workers"] != len(rows):
        raise CampaignError("bound device denominator differs")
    return result


def self_test() -> None:
    build, workload, observed = live_denominator(DEFAULT_PARTITIONS)
    if (len(observed["cross_product"]) != 20 or
            observed["work_items"] != EXPECTED["work_items"]):
        raise AssertionError("full campaign denominator differs")

    missing_shard = json.loads(json.dumps(build))
    missing_shard["shards"].pop()
    try:
        derive_denominator(missing_shard, workload)
    except CampaignError:
        pass
    else:
        raise AssertionError("missing-shard negative stayed green")

    selective = json.loads(json.dumps(build))
    selective["schema"] = "quactlize.tm8-epilogue-selective-build-plan.v1"
    try:
        derive_denominator(selective, workload)
    except CampaignError:
        pass
    else:
        raise AssertionError("selective-plan negative stayed green")

    missing_workload = json.loads(json.dumps(workload))
    missing_workload["cells"].pop()
    try:
        derive_denominator(build, missing_workload)
    except CampaignError:
        pass
    else:
        raise AssertionError("missing-workload negative stayed green")

    with tempfile.TemporaryDirectory(prefix="kpack-postfix-full-") as name:
        path = Path(name) / "build-plan.json"
        emit_plan(path, DEFAULT_PARTITIONS)
        if path.read_bytes() != (
                json.dumps(build, indent=2, sort_keys=True,
                           allow_nan=False) + "\n").encode():
            raise AssertionError("frozen full build plan differs")
    print(
        "[kpack-postfix-full:self-test] PASS "
        "formats=5 routes=2 operators=2 shards=2211 parents=70483 "
        "workloads=1381 atoms=339196 q4=203018 "
        "partitions=32 selective+missing-shard+missing-workload negatives=RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    commands.add_parser("denominator")
    commands.add_parser("check-source")
    emit = commands.add_parser("emit-plan")
    emit.add_argument("--partitions", type=int, default=DEFAULT_PARTITIONS)
    emit.add_argument("--output", type=Path, required=True)
    finish = commands.add_parser("finalize")
    finish.add_argument("--plan", type=Path, required=True)
    finish.add_argument("--publish-root", type=Path, required=True)
    finish.add_argument("--workers", type=int, default=DEFAULT_RUNTIME_WORKERS)
    finish.add_argument("--output", type=Path, required=True)
    bind = commands.add_parser("bind-devices")
    bind.add_argument("--campaign", type=Path, required=True)
    check = commands.add_parser("validate")
    check.add_argument("--campaign", type=Path, required=True)
    check.add_argument("--require-devices", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "denominator":
            _build, _workload, denominator = live_denominator()
            print(json.dumps(denominator, indent=2, sort_keys=True))
        elif args.command == "check-source":
            authority = require_clean_source()
            print(
                "KPACK_POSTFIX_FULL_SOURCE "
                f"sha={authority['repository']['head']} "
                f"tree={authority['repository']['tree']} clean=1")
        elif args.command == "emit-plan":
            denominator = emit_plan(args.output, args.partitions)
            print(
                "KPACK_POSTFIX_FULL_PLAN "
                f"shards={denominator['binary_shards']} "
                f"parents={denominator['parents']} "
                f"atoms={denominator['work_items']} output={args.output}")
        elif args.command == "finalize":
            result = finalize(
                args.plan, args.publish_root, args.workers, args.output)
            print(
                "KPACK_POSTFIX_FULL_FINALIZED "
                f"partitions={result['partition_artifacts']} "
                f"shards={result['denominator']['binary_shards']} "
                f"atoms={result['denominator']['work_items']} "
                f"workers={result['runtime_workers']} output={args.output}")
        elif args.command == "bind-devices":
            result = bind_devices(args.campaign)
            print(
                "KPACK_POSTFIX_FULL_DEVICES "
                f"workers={len(result['workers'])} homogeneous=1")
        else:
            result = validate_campaign(
                args.campaign, require_devices=args.require_devices)
            print(
                "KPACK_POSTFIX_FULL_VALID "
                f"source={result['source_sha']} "
                f"shards={result['denominator']['binary_shards']} "
                f"atoms={result['denominator']['work_items']} "
                f"devices={int(args.require_devices)}")
        return 0
    except (CampaignError, OSError, subprocess.SubprocessError,
            KeyError, TypeError, ValueError) as error:
        print(f"[kpack-postfix-full] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
