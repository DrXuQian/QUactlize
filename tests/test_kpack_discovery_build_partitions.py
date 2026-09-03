from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import re
import subprocess
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kpack_discovery_build_partitions as partitions  # noqa: E402
import kpack_discovery_worker_plan as worker_plan  # noqa: E402


def _write(path: pathlib.Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _file(seed: str) -> dict:
    return {"path": f"payloads/{seed}", "size": 1,
            "sha256": hashlib.sha256(seed.encode()).hexdigest()}


def _partition_document(plan: dict, plan_sha: str, route: str,
                        partition_id: int) -> dict:
    def sdk_file(path: str, marker: str) -> dict:
        return {"path": path, "size": 1, "sha256": marker * 64,
                "symlink_target": None}
    rows = []
    selected = partitions.selection(plan, partition_id, route)
    for index, source in enumerate(selected):
        rows.append({**source, "files": {
            "manifest": _file(f"{route}-{partition_id}-{index}-manifest"),
            "binary": _file(f"{route}-{partition_id}-{index}-binary"),
            "binary_receipt": _file(
                f"{route}-{partition_id}-{index}-receipt")},
            "device_arch": "PPU ppu0010",
            "inspector_output_sha256": "9" * 64})
    return {
        "schema": partitions.PARTITION_SCHEMA,
        "artifact_id": (
            f"kpack-discovery/{'1' * 40}/{plan_sha[:16]}/{route}/"
            f"p{partition_id:02d}-of-{plan['partition_count']:02d}"),
        "route": route, "partition_id": partition_id,
        "partition_count": plan["partition_count"],
        "source_sha": "1" * 40, "source_tree": "2" * 40,
        "submodules": [{"path": "third_party/actlize",
                        "gitlink": "3" * 40, "current": "3" * 40}],
        "sdk": {
            "receipt": sdk_file("VERSION.txt", "4"),
            "compiler": sdk_file("bin/hgcc", "5"),
            "inspector": sdk_file("bin/hgobjdump", "6"),
            "runtime_libraries": [sdk_file("lib/libhggcrt.so", "7")]},
        "partition_plan": {"path": "inputs/build-partition-plan.json",
                           "size": 1, "sha256": plan_sha},
        "build_input_authority": _file(f"{route}-{partition_id}-authority"),
        "runtime_identity_probe": {
            "binary": _file(f"{route}-{partition_id}-probe"),
            "receipt": _file(f"{route}-{partition_id}-probe-receipt")},
        "payload_validation": "RECORDED_FROM_LOCAL_BYTES",
        "denominator": {
            "shards": len(rows),
            "parents": sum(row["parent_count"] for row in rows),
            "shard_keys_sha256": partitions.digest(sorted(
                row["shard_key"] for row in rows))},
        "shards": rows,
    }


def test_partition_planner_self_test() -> None:
    partitions.self_test()


def test_metadata_merge_needs_exact_two_route_union(tmp_path: pathlib.Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan = partitions.make_plan(2)
    _write(plan_path, plan)
    plan_sha = partitions.file_sha(plan_path)
    manifests = []
    for route in partitions.ROUTES:
        for partition_id in range(2):
            document = _partition_document(
                plan, plan_sha, route, partition_id)
            path = tmp_path / f"{route}-{partition_id}.json"
            _write(path, document)
            manifests.append(path)
    catalog = partitions.make_catalog(plan_path, manifests)
    assert str(tmp_path) not in json.dumps(catalog)
    assert catalog["denominator"]["parents"] == 70618
    assert catalog["denominator"]["shards"] == 2216
    assert catalog["payload_residency"] == \
        "PER_WORKER_PARTITION_FETCH_AND_VERIFY"
    assert len(catalog["shards"]) == 2216
    assert all(row["artifact_id"] and row["parent_ids"] and
               row["parent_id_set_sha256"] and row["manifest_sha256"] ==
               row["files"]["manifest"]["sha256"]
               for row in catalog["shards"])
    catalog_path = tmp_path / "catalog.json"
    _write(catalog_path, catalog)
    workload_path = tmp_path / "workloads.json"
    _write(workload_path, {
        "schema": "fixture-workloads",
        "dense": [{"key": "dense"}],
        "grouped": [{"key": "grouped"}],
    })
    master = worker_plan.make_master(catalog_path, workload_path)
    assert master["denominator"]["binary_shards"] == 2216
    assignment = worker_plan.make_assignment(master, "a" * 64, 2)
    assert all(len(row["partition_ids"]) == 1
               for row in assignment["workers"])
    selected = worker_plan.make_worker_selection(
        master, assignment, 0, master_sha256="a" * 64,
        assignment_sha256="b" * 64)
    assert len(selected["partition_ids"]) == 1
    assert len(selected["artifact_ids"]) == 2
    assert not any(str(tmp_path) in value for value in selected["artifact_ids"])

    # Legacy device-bearing rows deliberately have no payload_kind field.
    fq_document = json.loads(
        (tmp_path / "fully-quantized-0.json").read_text())
    partitions.validate_partition_document(fq_document, plan)
    assert all("payload_kind" not in row for row in fq_document["shards"])

    fq_dense = next(row for row in fq_document["shards"]
                    if row["operator"] == "dense")
    missing_proof = copy.deepcopy(fq_document)
    planted = next(row for row in missing_proof["shards"]
                   if row["shard_key"] == fq_dense["shard_key"])
    planted["payload_kind"] = partitions.NO_DEVICE_KERNEL_STRUCTURAL
    planted["device_arch"] = "NO_DEVICE_KERNEL"
    with pytest.raises(partitions.PartitionError, match="file set differs"):
        partitions.validate_partition_document(missing_proof, plan)

    malformed_hash = copy.deepcopy(missing_proof)
    planted = next(row for row in malformed_hash["shards"]
                   if row["shard_key"] == fq_dense["shard_key"])
    planted["files"]["structural_proof"] = _file("structural-proof")
    planted["files"]["structural_proof"]["sha256"] = "not-a-sha256"
    with pytest.raises(partitions.PartitionError, match="lowercase SHA-256"):
        partitions.validate_partition_document(malformed_hash, plan)

    wrong_scope = copy.deepcopy(fq_document)
    planted = next(row for row in wrong_scope["shards"]
                   if row["operator"] == "grouped")
    planted["payload_kind"] = partitions.NO_DEVICE_KERNEL_STRUCTURAL
    planted["device_arch"] = "NO_DEVICE_KERNEL"
    with pytest.raises(
            partitions.PartitionError, match="outside FQ dense"):
        partitions.validate_partition_document(wrong_scope, plan)

    with pytest.raises(partitions.PartitionError):
        partitions.make_catalog(plan_path, manifests[:-1])
    with pytest.raises(partitions.PartitionError):
        partitions.make_catalog(plan_path, manifests + [manifests[0]])

    for field in ("source_sha", "sdk"):
        planted = json.loads(manifests[0].read_text())
        if field == "source_sha":
            planted[field] = "8" * 40
            planted["artifact_id"] = (
                f"kpack-discovery/{planted[field]}/{plan_sha[:16]}/"
                f"{planted['route']}/p{planted['partition_id']:02d}-of-02")
        else:
            planted[field]["compiler"]["sha256"] = "8" * 64
        planted_path = tmp_path / f"cross-{field}.json"
        _write(planted_path, planted)
        replaced = [planted_path, *manifests[1:]]
        with pytest.raises(partitions.PartitionError, match=f"{field} differs"):
            partitions.make_catalog(plan_path, replaced)

    stale = json.loads(manifests[0].read_text())
    stale["shards"][0]["parent_begin"] += 1
    _write(manifests[0], stale)
    with pytest.raises(partitions.PartitionError):
        partitions.make_catalog(plan_path, manifests)


def test_builders_expose_only_explicit_partition_contract() -> None:
    for name, route in (
            ("build_scalefirst_kpack_discovery_bundle.sh", "scalefirst"),
            ("build_fully_quantized_kpack_discovery_bundle.sh",
             "fully-quantized")):
        text = (TOOLS / name).read_text()
        assert "KPACK_BUILD_PARTITION_PLAN" in text
        assert "KPACK_BUILD_PARTITION_ID" in text
        assert "partition builds require an explicit unique OUT" in text
        assert f"--root \"$out\" --route {route}" in text
        assert "distributed partition builds require PILOT=0" in text


def test_fully_quantized_scratch_setup_is_nounset_safe(
        tmp_path: pathlib.Path) -> None:
    source = (TOOLS / "build_fully_quantized_kpack_discovery_bundle.sh").read_text()
    match = re.search(r"(?ms)^ensure_owned_scratch\(\) \{.*?^\}", source)
    assert match is not None
    bundle = tmp_path / "bundle"
    result = subprocess.run(
        ["bash", "-c",
         "set -uo pipefail\n" + match.group(0) +
         '\nensure_owned_scratch "$1" 0\n', "scratch-test", str(bundle)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False)
    assert result.returncode == 0, result.stdout
    assert (bundle / "scratch" / ".fq-kpack-owned-scratch").read_text() == \
        "quactlize-fq-kpack-owned-v2\n"
