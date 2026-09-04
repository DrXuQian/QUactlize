from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import aggregate_kpack_discovery_results as aggregate
import compose_kpack_discovery_bundles as compose
import kpack_discovery_build_partitions as partitions
import kpack_discovery_worker_plan as workers
import run_kpack_discovery_worker as worker_runner


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _file(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(),
            "sha256": aggregate.file_sha(path)}


def _samples(count: int, base: float) -> str:
    return "[" + ",".join(f"{base + index / 100:.9f}"
                            for index in range(count)) + "]"


def _sf_dense(candidates: list[dict], count: int, repeats: int,
              seed: int) -> str:
    lines = [
        "SF_SHARD qtype=10 artifact_tile_k=0 bchunk=0 "
        f"typed_rows={len(candidates)} weight_layout=2 "
        "weight_mapping_id=0x514b504b54000001 "
        f"selected_rows={len(candidates)} algorithm_mask=0x3 device=0 cu=72 "
        f"iterations={count} correctness_repeats={repeats} "
        f"schedule_seed=0x{seed:016x}"]
    records = 0
    for ordinal, candidate in enumerate(candidates):
        for algorithm, policy, grid in (
                ("NONPERSISTENT", "ordinary", 1),
                ("PERSISTENT", "capacity", 72)):
            for sample in range(count):
                lines.append("SF_CELL " + json.dumps({
                    "shape": "1x16x128", "qtype": 10,
                    "symbol": candidate["symbol"],
                    "a_provider": candidate["manifest_row"].get(
                        "a_provider", 0),
                    "algorithm": algorithm,
                    "metric_scope": "FULL_OUTPUT", "policy": policy,
                    "grid": grid, "occupancy": 1, "split": 1,
                    "capacity_b_mask": "0x1",
                    "balanced_b_mask": "0x0", "status": "MEASURED",
                    "reason": "MEASURED",
                    "sample": sample, "sample_us": 1 + ordinal / 10 + sample / 100,
                    "raw_bad": 0}))
                records += 1
    lines.append(
        "SF_COMPLETE status=COMPLETE shape=1x16x128 "
        f"typed_rows={len(candidates)} runtime_cells={2 * len(candidates)} "
        f"measured_cells={2 * len(candidates)} records={records} iterations={count} "
        "fixture=ORDER-INDEPENDENT+FP16-EXACT fixture_mode=ordinary roundtrip=PASS")
    return "\n".join(lines) + "\n"


def _fq_dense(candidates: list[dict], count: int, repeats: int,
              seed: int) -> str:
    lines = [
        "FQ_SHARD q=10 A=0 bchunk=0 shape=1x16x128 weight_layout=2 "
        "weight_mapping_id=0x514b504b54000001 weight_delivery_n=16 "
        f"typed_rows={len(candidates)} selected_rows={len(candidates)} "
        "only_split=0 bc_mode=skip bc_batch=native-grid-y-m-lt8 "
        f"split_timing=ordered-close iterations={count} "
        f"correctness_repeats={repeats} schedule_seed=0x{seed:016x}"]
    for ordinal, candidate in enumerate(candidates):
        for split in (1, 2, 4, 8):
            scope = ("FULL_OUTPUT" if split == 1 else
                     "PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS")
            lines.append(
                "FQ_TC_CELL q=10 A=0 bchunk=0 shape=1x16x128 "
                f"symbol={candidate['symbol']} tm=8 tn=16 tk=128 wm=8 wn=16 "
                f"stages=2 provider=standard-aiu S={split} scope={scope} "
                "resolved_delivery_n=16 provider_capacity_rows=0 scalezero_fused=0 "
                f"state=MEASURED us={1 + ordinal / 10:.9f} raw_bad=0 "
                f"reducer_untimed={int(split > 1)} failure_step=NONE failure_repeat=-1 "
                "first_bad=0 first_want=0x0000 first_got=0x0000 "
                f"shipping_smem=1 split_smem=1 partial_bytes=1 samples={_samples(count, 1 + ordinal / 10)}")
    lines.append(
        "FQ_SHAPE_DONE q=10 A=0 bchunk=0 shape=1x16x128 weight_layout=2 "
        "weight_mapping_id=0x514b504b54000001 weight_delivery_n=16 "
        f"typed_rows={len(candidates)} selected_rows={len(candidates)} "
        f"only_split=0 bc_mode=skip iterations={count} status=PASS")
    return "\n".join(lines) + "\n"


def _grouped(route: str, candidates: list[dict], count: int, repeats: int,
             warmups: int, seed: int) -> str:
    sf = route == "scalefirst"
    head = "SF_GROUPED_SHARD" if sf else "FQ_GROUPED_KPACK_SHARD"
    cell = "SF_GROUPED_CELL" if sf else "FQ_GROUPED_KPACK_CELL"
    done = "SF_GROUPED_COMPLETE" if sf else "FQ_GROUPED_KPACK_COMPLETE"
    type_field = "typed_rows" if sf else "type_rows"
    lines = [
        f"{head} q=10 layout=2 mapping_id=0x514b504b54000001 "
        f"{type_field}={len(candidates)} selected_rows={len(candidates)} "
        "router=fixture tokens=1 topk=8 experts=256 total_rows=8 max_rows=1 "
        "active=8 empty=248 workload=grouped router_profile=test "
        f"rows_hash=0x1 iterations={count} warmups={warmups} "
        f"correctness_repeats={repeats} schedule_seed=0x{seed:016x} "
        "roundtrip=PASS"]
    for ordinal, candidate in enumerate(candidates):
        if sf:
            algorithms = (("GROUPED_NONPERSISTENT", "ordinary", 0),
                          ("GROUPED_PERSISTENT", "capacity", 72))
        else:
            algorithm = candidate["manifest_row"]["algorithm"]
            algorithms = ((algorithm,
                           "ordinary" if algorithm.endswith("NONPERSISTENT") else "capacity",
                           0 if algorithm.endswith("NONPERSISTENT") else 72),)
        for algorithm, policy, grid in algorithms:
            lines.append(
                f"{cell} q=10 layout=2 symbol={candidate['symbol']} "
                "config=8x16x128_w8x16_s2 "
                f"algorithm={algorithm} policy={policy} grid={grid} occupancy=1 "
                "capacity_b_mask=0x1 balanced_b_mask=0x0 state=MEASURED "
                "raw_bad=0 first_bad=0 want=0x0000 got=0x0000 failure_repeat=-1 "
                f"median_us={1 + ordinal / 10:.9f} min_us=1 max_us=2 "
                f"execution_ordinal={ordinal} samples={_samples(count, 1 + ordinal / 10)}")
    runtime = len(candidates) * 2 if sf else len(candidates)
    tail = ("correctness=RAW_FP16 timing=AFTER_CORRECTNESS "
            "splitk=STRUCTURAL_UNAVAILABLE" if sf else
            "correctness=RAW_FP16 timing=AFTER_CORRECTNESS top_n=NONE")
    lines.append(f"{done} q=10 status=PASS rows={len(candidates)} cells={runtime} "
                 f"measured={runtime} structural=0 {tail}")
    return "\n".join(lines) + "\n"


def _argv(item: dict, binary: Path, count: int, repeats: int, seed: int,
          warmups: int, symbol_file: Path | None) -> list[str]:
    values = [str(binary)]
    if item["operator"] == "dense":
        values += ["--shape=1x16x128", f"--iterations={count}",
                   f"--correctness-repeats={repeats}"]
        values += (["--algorithm=full-output"]
                   if item["route"] == "scalefirst" else ["--bc-mode=skip"])
        values.append(f"--schedule-seed={seed}")
    else:
        values += ["--experts=256", "--n=512", "--k=2048",
                   "--workload-key=grouped", "--router-profile=test",
                   f"--iterations={count}", f"--warmups={warmups}",
                   f"--correctness-repeats={repeats}",
                   f"--schedule-seed={seed}"]
    if symbol_file is not None:
        values.append(f"--symbol-file={symbol_file}")
    if item["operator"] == "grouped":
        values += ["--tokens=1", "--topk=8"]
    return values


def _atom(item_id: str, phase: str, round_index: int, order: str,
          worker: int, item: dict, seed: int, argv: list[str]) -> str:
    warmups = "3" if item["operator"] == "grouped" else "NONE"
    return (
        "KPACK_DISCOVERY_ATOM "
        f"grouped_warmups={warmups} phase={phase} round={round_index} "
        f"order={order} schedule_seed=0x{seed:016x} worker={worker} "
        f"work_item_id={item_id} argv_sha256={aggregate.digest(argv)}\n")


def _make_fixture(root: Path, worker_count: int = 2) -> dict:
    sf_bundle, fq_bundle, bundle_path = compose._fixture(root)
    document = compose.compose_document(
        output=bundle_path, scalefirst_bundle=sf_bundle,
        fully_quantized_bundle=fq_bundle)
    compose.write_composite(bundle_path, document)
    plan_path = root / "plan.json"
    _write_json(plan_path, {
        "dense": [{"key": "dense", "m": 1, "n": 16, "k": 128}],
        "grouped": [{"key": "grouped", "tokens": 1, "topk": 8,
                     "experts": 256, "n": 512, "k": 2048,
                     "total_rows": 8, "max_rows": 1, "profile": "test"}]})
    workload_index = root / "workload-index.json"
    _write_json(workload_index, {
        "schema": "quactlize.kpack-discovery-workloads-test.v1",
        "plan_file_sha256": aggregate.file_sha(plan_path)})
    master_path, assignment_path = root / "master.json", root / "assignment.json"
    master = workers.make_master(bundle_path, plan_path)
    workers.write_frozen_json(master_path, master)
    assignment = workers.make_assignment(
        master, aggregate.file_sha(master_path), worker_count)
    workers.write_frozen_json(assignment_path, assignment)
    device_path = root / "devices.json"
    device = {"schema": workers.DEVICE_SCHEMA, "workers": [
        {"worker_id": worker, "identity_sha256": str(worker + 1) * 64,
         "homogeneity_key": "a" * 64} for worker in range(worker_count)]}
    _write_json(device_path, device)

    shards = {row["shard_key"]: row for row in document["shards"]}
    items = {row["work_item_id"]: row for row in master["work_items"]}
    evidences = []
    for worker_row in assignment["workers"]:
        worker = worker_row["worker_id"]
        worker_root = root / f"worker-{worker}"
        worker_root.mkdir()
        completion_path = worker_root / "completion.json"
        completion = {
            "schema": workers.RESULT_SCHEMA, "worker_id": worker,
            "bundle_sha256": assignment["bundle_sha256"],
            "workload_plan_sha256": assignment["workload_plan_sha256"],
            "master_sha256": aggregate.file_sha(master_path),
            "assignment_sha256": aggregate.file_sha(assignment_path),
            "device_homogeneity_sha256": aggregate.file_sha(device_path),
            "device_identity_sha256": device["workers"][worker]["identity_sha256"],
            "completed_work_item_ids": worker_row["work_item_ids"],
        }
        _write_json(completion_path, completion)
        assigned_routes = sorted({items[item]["route"] for item in worker_row["work_item_ids"]})
        selection = workers.make_worker_selection(
            master, assignment, worker,
            master_sha256=aggregate.file_sha(master_path),
            assignment_sha256=aggregate.file_sha(assignment_path))
        runtime_linkage = [["libhggc.so.13", "/runtime/libhggc.so.13",
                            "/runtime/libhggc.so.13.0", 1, "f" * 64]]
        execution_path = worker_root / "execution-authority.json"
        _write_json(execution_path, {
            "schema": worker_runner.EXECUTION_SCHEMA,
            "worker_id": worker, "worker_count": worker_count,
            "bundle_sha256": aggregate.file_sha(bundle_path),
            "workload_plan_sha256": aggregate.file_sha(plan_path),
            "master_sha256": aggregate.file_sha(master_path),
            "assignment_sha256": aggregate.file_sha(assignment_path),
            "selection_sha256": aggregate._pretty_json_sha(selection),
            "device_identity_sha256": device["workers"][worker]["identity_sha256"],
            "device_homogeneity_sha256": aggregate.file_sha(device_path),
            "visible_device_ordinal": str(worker),
            "workload_index_sha256": aggregate.file_sha(workload_index),
            "executor_sha256": "e" * 64,
            "work_items": len(worker_row["work_item_ids"]),
            "validated_shards": len({items[item]["shard_key"]
                                      for item in worker_row["work_item_ids"]}),
            "validated_shard_keys_sha256": aggregate.digest(sorted({
                items[item]["shard_key"] for item in worker_row["work_item_ids"]})),
            "validated_partition_artifacts": [],
            "candidate_policy":
                "SCREEN_ALL_COMPILED_CONFIRM_ALL_ADMISSIBLE_NO_TIMING_PRUNE",
            "measurement": {
                "screen_iterations": 5, "confirm_iterations": 11,
                "confirm_rounds": 3, "correctness_repeats": 17,
                "grouped_warmups": 3,
                "schedule_seed": worker_runner.schedule_seed_contract(),
                "outer_round_order": "ASSIGNMENT_REVERSE_THEN_HASHED_V1",
                "inner_candidate_order":
                    "ROUND_SEED_VARIED_ALL_ROUTES_OPERATORS",
            },
            "runtime_linkage": runtime_linkage,
        })
        authorities = []
        authority_sha = {}
        for route in assigned_routes:
            path = worker_root / f"{route}.authority.json"
            _write_json(path, {
                "schema":
                    "quactlize.kpack-discovery-worker-route-result-authority.v2",
                "route": route, "worker_id": worker,
                "execution_authority_sha256": aggregate.file_sha(execution_path),
                "candidate_policy": "ALL_ADMISSIBLE_NO_TIMING_PRUNE",
                "work_item_ids": [item for item in worker_row["work_item_ids"]
                                  if items[item]["route"] == route],
                "screen_iterations": 5, "confirm_iterations": 11,
                "confirm_rounds": 3, "correctness_repeats": 17,
                "grouped_warmups": 3,
                "schedule_seed": worker_runner.schedule_seed_contract(),
            })
            authorities.append({"route": route, **_file(path, worker_root)})
            authority_sha[route] = aggregate.file_sha(path)

        item_docs = []
        for item_id in worker_row["work_item_ids"]:
            item = items[item_id]
            shard = shards[item["shard_key"]]
            manifest_path = root / shard["files"]["manifest"]["path"]
            manifest = json.loads(manifest_path.read_text())
            candidates = aggregate.manifest_candidates(
                item["route"], item["operator"], manifest, shard)
            binary_path = (root / shard["files"]["binary"]["path"]).resolve()
            receipt_path = (root / shard["files"]["binary_receipt"]["path"]).resolve()
            if item["route"] == "scalefirst" and item["operator"] == "dense":
                make_log = lambda count, repeats, seed: _sf_dense(
                    candidates, count, repeats, seed)
            elif item["operator"] == "dense":
                make_log = lambda count, repeats, seed: _fq_dense(
                    candidates, count, repeats, seed)
            else:
                make_log = lambda count, repeats, seed: _grouped(
                    item["route"], candidates, count, repeats, 3, seed)
            prefix = item_id[:12]
            screen_path = worker_root / f"{prefix}.screen.log"
            retention = None
            symbols_path = None
            if item["route"] == "scalefirst":
                symbols_path = worker_root / f"{prefix}.symbols"
                symbols_path.write_text("".join(row["symbol"] + "\n" for row in candidates))
            screen_seed = worker_runner.schedule_seed(item_id, "screen", 0)
            screen_argv = _argv(
                item, binary_path, 5, 17, screen_seed, 3, None)
            screen_path.write_text(
                _atom(item_id, "screen", 0, "SCREEN", worker, item,
                      screen_seed, screen_argv) + make_log(5, 17, screen_seed))
            if item["route"] == "scalefirst":
                assert symbols_path is not None
                sidecar_path = worker_root / f"{prefix}.retention.json"
                _write_json(sidecar_path, {
                    "operator": item["operator"], "qtype": item["qtype"],
                    "manifest_sha256": aggregate.file_sha(manifest_path),
                    "screen_sha256": aggregate.file_sha(screen_path),
                    "retained_symbols": [row["symbol"] for row in candidates],
                    "retained_count": len(candidates),
                    "timing_rank_used_for_elimination": False})
                retention = {"symbols": _file(symbols_path, worker_root),
                             "sidecar": _file(sidecar_path, worker_root)}
            manifest_snapshot = worker_root / f"{prefix}.manifest.json"
            manifest_snapshot.write_bytes(manifest_path.read_bytes())
            execution_inputs = {
                "artifact_id": None, "shard_key": item["shard_key"],
                "binary": {"executed_path": str(binary_path),
                           "size": binary_path.stat().st_size,
                           "sha256": aggregate.file_sha(binary_path)},
                "manifest": {"size": manifest_path.stat().st_size,
                             "sha256": aggregate.file_sha(manifest_path),
                             "snapshot": _file(manifest_snapshot, worker_root)},
                "binary_receipt": {"size": receipt_path.stat().st_size,
                                   "sha256": aggregate.file_sha(receipt_path)},
                "rows_file": None,
                "retention_symbols_executed_path": (
                    str(symbols_path) if symbols_path is not None else None),
            }
            confirms = []
            for number, order in ((1, "FORWARD"), (2, "REVERSE"),
                                  (3, "HASHED")):
                path = worker_root / f"{prefix}.confirm-{number}.log"
                seed = worker_runner.schedule_seed(item_id, "confirm", number)
                argv = _argv(item, binary_path, 11, 17, seed, 3,
                             symbols_path)
                path.write_text(
                    _atom(item_id, "confirm", number, order, worker, item,
                          seed, argv) + make_log(11, 17, seed))
                confirms.append({"round": number, "order": order,
                                 "schedule_seed":
                                     worker_runner.schedule_seed_hex(seed),
                                 "argv": argv, "argv_sha256": aggregate.digest(argv),
                                 "log": _file(path, worker_root)})
            item_docs.append({
                "work_item_id": item_id,
                "result_authority_sha256": authority_sha[item["route"]],
                "execution_inputs": execution_inputs,
                "screen": {"schedule_seed":
                               worker_runner.schedule_seed_hex(screen_seed),
                           "argv": screen_argv,
                           "argv_sha256": aggregate.digest(screen_argv),
                           "log": _file(screen_path, worker_root)},
                "retention": retention, "confirm": confirms})
        evidence = {
            "schema": aggregate.EVIDENCE_SCHEMA, "worker_id": worker,
            "worker_count": worker_count,
            "bundle_sha256": aggregate.file_sha(bundle_path),
            "workload_plan_sha256": aggregate.file_sha(plan_path),
            "workload_index_sha256": aggregate.file_sha(workload_index),
            "master_sha256": aggregate.file_sha(master_path),
            "assignment_sha256": aggregate.file_sha(assignment_path),
            "device_homogeneity_sha256": aggregate.file_sha(device_path),
            "device_identity_sha256": device["workers"][worker]["identity_sha256"],
            "execution_authority": _file(execution_path, worker_root),
            "completion_result": _file(completion_path, worker_root),
            "result_authorities": authorities,
            "run_contract": {
                "schedule_seed": worker_runner.schedule_seed_contract(),
                "grouped_warmups": 3,
                "screen": {"timing_samples_per_runtime": 5,
                           "correctness_repeats": 17},
                "confirm": {"timing_samples_per_runtime": 11,
                            "correctness_repeats": 17,
                            "rounds": [{"round": 1, "order": "FORWARD"},
                                       {"round": 2, "order": "REVERSE"},
                                       {"round": 3, "order": "HASHED"}]},
                "all_admissible_candidates": True, "top_n": None},
            "work_items": item_docs,
        }
        evidence_path = worker_root / "evidence.json"
        _write_json(evidence_path, evidence)
        evidences.append(evidence_path)
    return {"bundle": bundle_path, "plan": plan_path,
            "workload_index": workload_index, "master": master_path,
            "assignment": assignment_path, "device": device_path,
            "evidence": evidences}


def _real_catalog(root: Path, composite: dict) -> dict:
    """Build a schema-valid full catalog while retaining four fixture shards."""
    plan_path = root / "partition-plan.json"
    plan = partitions.make_plan(1)
    _write_json(plan_path, plan)
    plan_sha = partitions.file_sha(plan_path)
    selected = {row["shard_key"]: row for row in composite["shards"]}

    def record(path: str, marker: str, selected_record: dict | None = None) -> dict:
        if selected_record is not None:
            source = root / selected_record["path"]
            return {"path": path, "size": source.stat().st_size,
                    "sha256": aggregate.file_sha(source)}
        return {"path": path, "size": 1,
                "sha256": aggregate.digest(marker)}

    def sdk_record(value: dict) -> dict:
        return {"path": value["path"], "size": value["size"],
                "sha256": value["sha256"], "symlink_target": None}

    manifests = []
    for route in partitions.ROUTES:
        rows = []
        for ordinal, source in enumerate(partitions.selection(plan, 0, route)):
            chosen = selected.get(source["shard_key"])
            if chosen is not None:
                assert chosen["parent_ids"] == source["parent_ids"]
            stem = f"payloads/{route}/{ordinal:04d}"
            rows.append({
                **source,
                "files": {
                    field: record(
                        f"{stem}/{field}", f"{route}-{ordinal}-{field}",
                        None if chosen is None else chosen["files"][field])
                    for field in ("manifest", "binary", "binary_receipt")
                },
                "device_arch": "PPU ppu0010",
                "inspector_output_sha256": "9" * 64,
            })
        artifact_id = (
            f"kpack-discovery/{composite['source_sha']}/{plan_sha[:16]}/"
            f"{route}/p00-of-01")
        document = {
            "schema": partitions.PARTITION_SCHEMA,
            "artifact_id": artifact_id, "route": route,
            "partition_id": 0, "partition_count": 1,
            "source_sha": composite["source_sha"],
            "source_tree": composite["source_tree"],
            "submodules": composite["submodules"],
            "sdk": {
                "receipt": sdk_record(composite["sdk"]["receipt"]),
                "compiler": sdk_record(composite["sdk"]["compiler"]),
                "inspector": sdk_record(composite["sdk"]["inspector"]),
                "runtime_libraries": [sdk_record(row) for row in
                                      composite["sdk"]["runtime_libraries"]],
            },
            "partition_plan": {"path": "inputs/build-partition-plan.json",
                               "size": plan_path.stat().st_size,
                               "sha256": plan_sha},
            "build_input_authority": record(
                f"inputs/{route}-build-authority.json", f"{route}-authority"),
            "runtime_identity_probe": {
                "binary": record(f"probes/{route}", f"{route}-probe"),
                "receipt": record(
                    f"probes/{route}.json", f"{route}-probe-receipt"),
            },
            "payload_validation": "RECORDED_FROM_LOCAL_BYTES",
            "denominator": {
                "shards": len(rows),
                "parents": sum(row["parent_count"] for row in rows),
                "shard_keys_sha256": partitions.digest(sorted(
                    row["shard_key"] for row in rows)),
            },
            "shards": rows,
        }
        path = root / f"partition-{route}.json"
        _write_json(path, document)
        manifests.append(path)
    catalog = partitions.make_catalog(plan_path, manifests)
    partitions.validate_catalog_document(catalog)
    return catalog


def _rebind_fixture_to_catalog(paths: dict, catalog: dict) -> None:
    """Rebind the one-worker fixture to selected rows of a validated catalog."""
    _write_json(paths["bundle"], catalog)
    catalog_sha = aggregate.file_sha(paths["bundle"])
    by_shard = {row["shard_key"]: row for row in catalog["shards"]}
    master = json.loads(paths["master"].read_text())
    master["bundle_sha256"] = catalog_sha
    for item in master["work_items"]:
        shard = by_shard[item["shard_key"]]
        item["partition_id"] = shard["partition_id"]
        item["artifact_id"] = shard["artifact_id"]
    _write_json(paths["master"], master)
    master_sha = aggregate.file_sha(paths["master"])
    assignment = workers.make_assignment(master, master_sha, 1)
    _write_json(paths["assignment"], assignment)
    assignment_sha = aggregate.file_sha(paths["assignment"])

    evidence_path = paths["evidence"][0]
    evidence = json.loads(evidence_path.read_text())
    items = {row["work_item_id"]: row for row in evidence["work_items"]}
    evidence["work_items"] = [items[item_id]
                              for item_id in assignment["workers"][0]["work_item_ids"]]
    evidence["bundle_sha256"] = catalog_sha
    evidence["master_sha256"] = master_sha
    evidence["assignment_sha256"] = assignment_sha
    for row in evidence["work_items"]:
        item = next(item for item in master["work_items"]
                    if item["work_item_id"] == row["work_item_id"])
        row["execution_inputs"]["artifact_id"] = item["artifact_id"]

    completion_path = evidence_path.parent / evidence["completion_result"]["path"]
    completion = json.loads(completion_path.read_text())
    completion["bundle_sha256"] = catalog_sha
    completion["master_sha256"] = master_sha
    completion["assignment_sha256"] = assignment_sha
    completion["completed_work_item_ids"] = assignment["workers"][0]["work_item_ids"]
    _write_json(completion_path, completion)
    evidence["completion_result"]["sha256"] = aggregate.file_sha(completion_path)

    execution_path = evidence_path.parent / evidence["execution_authority"]["path"]
    execution = json.loads(execution_path.read_text())
    execution["bundle_sha256"] = catalog_sha
    execution["master_sha256"] = master_sha
    execution["assignment_sha256"] = assignment_sha
    selection = workers.make_worker_selection(
        master, assignment, 0, master_sha256=master_sha,
        assignment_sha256=assignment_sha)
    execution["selection_sha256"] = aggregate._pretty_json_sha(selection)
    shard_keys = sorted({item["shard_key"] for item in master["work_items"]})
    execution["validated_shards"] = len(shard_keys)
    execution["validated_shard_keys_sha256"] = aggregate.digest(shard_keys)
    partitions_by_id = {row["artifact_id"]: row for row in catalog["partitions"]}
    execution["validated_partition_artifacts"] = [{
        "artifact_id": artifact_id,
        "partition_manifest_sha256":
            partitions_by_id[artifact_id]["partition_manifest"]["sha256"],
    } for artifact_id in sorted(assignment["workers"][0]["artifact_ids"])]
    _write_json(execution_path, execution)
    execution_sha = aggregate.file_sha(execution_path)
    evidence["execution_authority"]["sha256"] = execution_sha

    authority_sha = {}
    for record in evidence["result_authorities"]:
        path = evidence_path.parent / record["path"]
        document = json.loads(path.read_text())
        document["execution_authority_sha256"] = execution_sha
        document["work_item_ids"] = [
            item_id for item_id in assignment["workers"][0]["work_item_ids"]
            if next(item for item in master["work_items"]
                    if item["work_item_id"] == item_id)["route"] == record["route"]]
        _write_json(path, document)
        record["sha256"] = aggregate.file_sha(path)
        authority_sha[record["route"]] = record["sha256"]
    for row in evidence["work_items"]:
        item = next(item for item in master["work_items"]
                    if item["work_item_id"] == row["work_item_id"])
        row["result_authority_sha256"] = authority_sha[item["route"]]
    _write_json(evidence_path, evidence)


def _aggregate(paths: dict, output: Path) -> dict:
    return aggregate.aggregate(
        bundle_path=paths["bundle"], plan_path=paths["plan"],
        workload_index_path=paths["workload_index"],
        master_path=paths["master"], assignment_path=paths["assignment"],
        device_path=paths["device"], evidence_paths=paths["evidence"],
        output_dir=output)


def test_self_test() -> None:
    aggregate.self_test()


def test_execution_inputs_preserve_legacy_device_schema_and_bind_proof_hash(
        tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"dense_tc_parents": []})
    manifest_sha = aggregate.file_sha(manifest)
    binary_sha, receipt_sha = "a" * 64, "b" * 64
    shard = {
        "shard_key": "fully-quantized:fixture", "artifact_id": None,
        "files": {
            "binary": {"size": 1, "sha256": binary_sha},
            "manifest": {"size": manifest.stat().st_size,
                         "sha256": manifest_sha},
            "binary_receipt": {"size": 1, "sha256": receipt_sha},
        },
    }
    legacy = {
        "artifact_id": None, "shard_key": shard["shard_key"],
        "binary": {"executed_path": str(tmp_path / "kernel"),
                   "size": 1, "sha256": binary_sha},
        "manifest": {"size": manifest.stat().st_size,
                     "sha256": manifest_sha,
                     "snapshot": _file(manifest, tmp_path)},
        "binary_receipt": {"size": 1, "sha256": receipt_sha},
        "rows_file": None, "retention_symbols_executed_path": None,
    }
    normalized = aggregate.validate_execution_inputs(
        legacy, tmp_path, shard=shard,
        workload={"source_class": "real-inventory"}, label="legacy")
    assert normalized["payload_kind"] == worker_runner.DEVICE_KERNEL
    assert normalized["structural_proof"] is None

    proof = tmp_path / "structural-proof.json"
    _write_json(proof, {})
    structural_shard = {
        **shard, "payload_kind": worker_runner.NO_DEVICE_KERNEL_STRUCTURAL,
        "route": "fully-quantized", "operator": "dense", "qtype": 10,
        "parent_begin": 0, "parent_end": 1, "parent_count": 1,
        "authority_count": 1, "parent_ids": ["parent"],
        "files": {
            **shard["files"],
            "structural_proof": {
                "size": proof.stat().st_size,
                "sha256": aggregate.file_sha(proof)},
        },
    }
    missing = {
        **legacy,
        "payload_kind": worker_runner.NO_DEVICE_KERNEL_STRUCTURAL,
        "structural_proof": None,
    }
    with pytest.raises(aggregate.AggregateError,
                       match="structural proof metadata differs"):
        aggregate.validate_execution_inputs(
            missing, tmp_path, shard=structural_shard,
            workload={"source_class": "real-inventory"}, label="structural")

    wrong_hash = {
        **missing,
        "structural_proof": {
            "size": proof.stat().st_size, "sha256": "0" * 64,
            "snapshot": _file(proof, tmp_path)},
    }
    with pytest.raises(aggregate.AggregateError,
                       match="structural proof snapshot differs"):
        aggregate.validate_execution_inputs(
            wrong_hash, tmp_path, shard=structural_shard,
            workload={"source_class": "real-inventory"}, label="structural")


def test_complete_authority_and_runtime_census(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path / "fixture")
    result = _aggregate(paths, tmp_path / "output")
    assert result["verdict"] == "COMPLETE_RAW_BIT_CLEAN_CENSUS"
    assert result["denominator"] == {
        **result["denominator"],
        "work_items": 4,
        "static_candidate_instances": 128,
        "runtime_instances": 288,
        "admissible_runtime_instances": 288,
        "structural_runtime_instances": 0,
        "confirm_log_references": 12,
        "timing_samples": 288 * 33,
    }
    checked = aggregate.validate_output(tmp_path / "output")
    assert checked["census"]["records"] == 288
    rows = [json.loads(line) for line in
            (tmp_path / "output/census.jsonl").read_text().splitlines()]
    assert len(rows) == 288
    assert all(row["raw_bad"] == 0 and row["timing"]["sample_count"] == 33
               for row in rows)
    assert all(len(row["timing"]["confirm_runs"]) == 3 and
               sum(len(run["samples"]) for run in
                   row["timing"]["confirm_runs"]) == 33 and
               all(run["provenance_sha256"] ==
                   aggregate.digest(row["timing"]["provenance"])
                   for run in row["timing"]["confirm_runs"])
               for row in rows)
    worker_records = result["authorities"]["worker_evidence"]
    evidence_paths = [row["evidence"]["path"] for row in worker_records]
    assert len(evidence_paths) == len(set(evidence_paths)) == 2
    assert all(f"worker-{row['worker_id']:04d}-" in row["evidence"]["path"]
               for row in worker_records)
    assert all((tmp_path / "output" / row["evidence"]["path"]).is_file()
               for row in worker_records)


def test_stale_log_hash_fails_closed(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path / "fixture")
    evidence_path = paths["evidence"][0]
    evidence = json.loads(evidence_path.read_text())
    log = evidence_path.parent / evidence["work_items"][0]["screen"]["log"]["path"]
    log.write_text(log.read_text() + "tamper\n")
    with pytest.raises(aggregate.AggregateError, match="SHA-256 differs"):
        _aggregate(paths, tmp_path / "output")


def test_raw_bit_mismatch_fails_even_with_rehashed_receipt(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path / "fixture")
    target = None
    for evidence_path in paths["evidence"]:
        evidence = json.loads(evidence_path.read_text())
        for item in evidence["work_items"]:
            log = evidence_path.parent / item["screen"]["log"]["path"]
            if "FQ_TC_CELL " in log.read_text():
                target = evidence_path, evidence, item, log
                break
        if target:
            break
    assert target is not None
    evidence_path, evidence, item, log = target
    body = log.read_text().replace("state=MEASURED us=", "state=MEASURED us=", 1)
    body = body.replace("raw_bad=0", "raw_bad=1", 1)
    log.write_text(body)
    item["screen"]["log"]["sha256"] = aggregate.file_sha(log)
    _write_json(evidence_path, evidence)
    with pytest.raises(aggregate.AggregateError, match="raw-bit mismatch"):
        _aggregate(paths, tmp_path / "output")


def test_scalefirst_packed_a_shape_rejection_is_structural() -> None:
    reason = "INADMISSIBLE_PACKED_A_PROVIDER_CAPACITY"
    candidate = {
        "symbol": "packed-a",
        "manifest_row": {"a_provider": 1},
    }
    row = {
        "symbol": "packed-a", "a_provider": 1,
        "status": "INADMISSIBLE", "reason": reason,
    }
    aggregate.validate_sf_dense_provider_states(
        [row], [candidate], 3, "packed-a")
    assert aggregate.check_state(
        reason, 0, [], 11, "packed-a",
        aggregate.SF_DENSE_STRUCTURAL) == \
        "STRUCTURAL_UNAVAILABLE"
    with pytest.raises(aggregate.AggregateError,
                       match="non-admissible runtime state"):
        aggregate.check_state(reason, 0, [], 11, "non-SF")
    with pytest.raises(aggregate.AggregateError,
                       match="structural runtime carries timing samples"):
        aggregate.check_state(
            reason, 0, [1.0], 11, "packed-a",
            aggregate.SF_DENSE_STRUCTURAL)


@pytest.mark.parametrize(
    ("provider", "m", "message"),
    ((0, 3, "belongs only to AP1"),
     (1, 1, "invalid for M=1")),
)
def test_scalefirst_packed_a_shape_rejection_checks_owner(
        provider: int, m: int, message: str) -> None:
    reason = "INADMISSIBLE_PACKED_A_PROVIDER_CAPACITY"
    candidate = {
        "symbol": "packed-a",
        "manifest_row": {"a_provider": provider},
    }
    row = {
        "symbol": "packed-a", "a_provider": provider,
        "status": "INADMISSIBLE", "reason": reason,
    }
    with pytest.raises(aggregate.AggregateError, match=message):
        aggregate.validate_sf_dense_provider_states(
            [row], [candidate], m, "packed-a")


def test_scalefirst_runtime_provider_must_match_manifest() -> None:
    candidate = {
        "symbol": "packed-a",
        "manifest_row": {"a_provider": 1},
    }
    row = {
        "symbol": "packed-a", "a_provider": 0,
        "status": "MEASURED", "reason": "NONE",
    }
    with pytest.raises(aggregate.AggregateError,
                       match="runtime/manifest A provider differs"):
        aggregate.validate_sf_dense_provider_states(
            [row], [candidate], 1, "packed-a")


def test_top_n_and_missing_round_fail_before_logs(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path / "fixture")
    evidence_path = paths["evidence"][0]
    original = json.loads(evidence_path.read_text())
    planted = json.loads(json.dumps(original)); planted["run_contract"]["top_n"] = 4
    _write_json(evidence_path, planted)
    with pytest.raises(aggregate.AggregateError, match="ranking"):
        _aggregate(paths, tmp_path / "top-n")
    _write_json(evidence_path, original)
    planted = json.loads(json.dumps(original))
    planted["work_items"][0]["confirm"].pop()
    _write_json(evidence_path, planted)
    with pytest.raises(aggregate.AggregateError, match="confirmation run denominator"):
        _aggregate(paths, tmp_path / "round-gap")


def test_candidate_runtime_gap_and_stale_master_fail_closed(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path / "fixture")
    evidence_path = paths["evidence"][0]
    evidence = json.loads(evidence_path.read_text())
    evidence["master_sha256"] = "f" * 64
    _write_json(evidence_path, evidence)
    with pytest.raises(aggregate.AggregateError, match="stale master_sha256"):
        _aggregate(paths, tmp_path / "stale-master")

    paths = _make_fixture(tmp_path / "fixture-gap")
    for evidence_path in paths["evidence"]:
        evidence = json.loads(evidence_path.read_text())
        for item in evidence["work_items"]:
            log = evidence_path.parent / item["screen"]["log"]["path"]
            lines = log.read_text().splitlines()
            cells = [index for index, line in enumerate(lines)
                     if line.startswith("FQ_TC_CELL ")]
            if cells:
                lines.pop(cells[0])
                log.write_text("\n".join(lines) + "\n")
                item["screen"]["log"]["sha256"] = aggregate.file_sha(log)
                _write_json(evidence_path, evidence)
                with pytest.raises(aggregate.AggregateError,
                                   match="FQ split denominator differs"):
                    _aggregate(paths, tmp_path / "runtime-gap")
                return
    raise AssertionError("fixture contains no FQ dense screen log")


def _item_with_log_prefix(paths: dict, prefix: str,
                          phase: str = "screen") -> tuple[Path, dict, dict, Path]:
    for evidence_path in paths["evidence"]:
        evidence = json.loads(evidence_path.read_text())
        for item in evidence["work_items"]:
            run = item["screen"] if phase == "screen" else item["confirm"][0]
            log = evidence_path.parent / run["log"]["path"]
            if any(line.startswith(prefix) for line in log.read_text().splitlines()):
                return evidence_path, evidence, item, log
    raise AssertionError(f"fixture has no {prefix} log")


def test_wrong_schedule_seed_is_red_even_when_receipts_are_rehashed(
        tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path / "fixture")
    evidence_path, evidence, item, log = _item_with_log_prefix(paths, "SF_SHARD ")
    run = item["screen"]
    old = int(run["schedule_seed"], 0)
    wrong = old ^ 1
    run["schedule_seed"] = worker_runner.schedule_seed_hex(wrong)
    run["argv"] = [f"--schedule-seed={wrong}" if value.startswith(
        "--schedule-seed=") else value for value in run["argv"]]
    run["argv_sha256"] = aggregate.digest(run["argv"])
    lines = log.read_text().splitlines()
    for index, line in enumerate(lines):
        if index == 0:
            line = re.sub(r"schedule_seed=0x[0-9a-f]+",
                          f"schedule_seed=0x{wrong:016x}", line)
            line = re.sub(r"argv_sha256=[0-9a-f]{64}",
                          f"argv_sha256={run['argv_sha256']}", line)
        elif line.startswith("SF_SHARD "):
            line = re.sub(r"schedule_seed=0x[0-9a-f]+",
                          f"schedule_seed=0x{wrong:016x}", line)
        lines[index] = line
    log.write_text("\n".join(lines) + "\n")
    run["log"]["sha256"] = aggregate.file_sha(log)
    _write_json(evidence_path, evidence)
    with pytest.raises(aggregate.AggregateError, match="schedule seed differs"):
        _aggregate(paths, tmp_path / "output")


def test_wrong_grouped_warmup_is_red_with_consistent_argv_atom_and_header(
        tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path / "fixture")
    evidence_path, evidence, item, log = _item_with_log_prefix(
        paths, "FQ_GROUPED_KPACK_SHARD ")
    run = item["screen"]
    run["argv"] = ["--warmups=4" if value.startswith("--warmups=") else value
                   for value in run["argv"]]
    run["argv_sha256"] = aggregate.digest(run["argv"])
    lines = log.read_text().splitlines()
    lines[0] = lines[0].replace("grouped_warmups=3", "grouped_warmups=4")
    lines[0] = re.sub(r"argv_sha256=[0-9a-f]{64}",
                      f"argv_sha256={run['argv_sha256']}", lines[0])
    lines = [line.replace("warmups=3", "warmups=4")
             if line.startswith("FQ_GROUPED_KPACK_SHARD ") else line
             for line in lines]
    log.write_text("\n".join(lines) + "\n")
    run["log"]["sha256"] = aggregate.file_sha(log)
    _write_json(evidence_path, evidence)
    with pytest.raises(aggregate.AggregateError, match="exact argv options differ"):
        _aggregate(paths, tmp_path / "output")


def test_reordered_argv_is_red_even_with_rehashed_atom(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path / "fixture")
    evidence_path, evidence, item, log = _item_with_log_prefix(paths, "FQ_SHARD ")
    run = item["screen"]
    run["argv"][1], run["argv"][2] = run["argv"][2], run["argv"][1]
    run["argv_sha256"] = aggregate.digest(run["argv"])
    lines = log.read_text().splitlines()
    lines[0] = re.sub(r"argv_sha256=[0-9a-f]{64}",
                      f"argv_sha256={run['argv_sha256']}", lines[0])
    log.write_text("\n".join(lines) + "\n")
    run["log"]["sha256"] = aggregate.file_sha(log)
    _write_json(evidence_path, evidence)
    with pytest.raises(aggregate.AggregateError, match="exact argv options differ"):
        _aggregate(paths, tmp_path / "output")


@pytest.mark.parametrize("prefix", [
    "SF_SHARD ", "FQ_SHARD ", "SF_GROUPED_SHARD ",
    "FQ_GROUPED_KPACK_SHARD "])
def test_each_native_header_schedule_seed_is_independently_authoritative(
        tmp_path: Path, prefix: str) -> None:
    paths = _make_fixture(tmp_path / "fixture")
    evidence_path, evidence, item, log = _item_with_log_prefix(paths, prefix)
    lines = log.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = re.sub(r"schedule_seed=0x[0-9a-f]+",
                                  "schedule_seed=0x0000000000000000", line)
            break
    log.write_text("\n".join(lines) + "\n")
    item["screen"]["log"]["sha256"] = aggregate.file_sha(log)
    _write_json(evidence_path, evidence)
    with pytest.raises(aggregate.AggregateError,
                       match="invocation.*differ|shard identity differ"):
        _aggregate(paths, tmp_path / "output")


@pytest.mark.parametrize("extra", [-1, 1])
def test_confirm_sample_underflow_and_overflow_are_red(
        tmp_path: Path, extra: int) -> None:
    paths = _make_fixture(tmp_path / "fixture")
    evidence_path, evidence, item, log = _item_with_log_prefix(
        paths, "FQ_SHARD ", phase="confirm")
    lines = log.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("FQ_TC_CELL "):
            continue
        match = re.search(r"samples=\[([^]]+)\]", line)
        assert match is not None
        samples = match.group(1).split(",")
        samples = samples[:-1] if extra < 0 else samples + ["9.999000000"]
        lines[index] = line[:match.start(1)] + ",".join(samples) + line[match.end(1):]
        break
    log.write_text("\n".join(lines) + "\n")
    item["confirm"][0]["log"]["sha256"] = aggregate.file_sha(log)
    _write_json(evidence_path, evidence)
    with pytest.raises(aggregate.AggregateError, match="timing denominator differs"):
        _aggregate(paths, tmp_path / "output")


def test_worker_outputs_relocate_and_artifact_payloads_can_be_removed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _make_fixture(tmp_path / "fixture")
    bundle = workers.load_bundle_authority(paths["bundle"])
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    moved = []
    for evidence_path in paths["evidence"]:
        destination = relocated / evidence_path.parent.name
        shutil.move(str(evidence_path.parent), destination)
        moved.append(destination / evidence_path.name)
    paths["evidence"] = moved
    for shard in bundle["shards"]:
        for record in shard["files"].values():
            candidate = paths["bundle"].parent / record["path"]
            candidate.unlink(missing_ok=True)
    monkeypatch.setattr(aggregate.worker_plan, "load_bundle_authority",
                        lambda _path: bundle)
    monkeypatch.setattr(aggregate.worker_plan, "validate_master",
                        lambda *_args, **_kwargs: None)
    result = _aggregate(paths, tmp_path / "output")
    assert result["verdict"] == "COMPLETE_RAW_BIT_CLEAN_CENSUS"


def test_validated_distributed_catalog_survives_artifact_removal_and_relocation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Aggregation consumes portable evidence, not the worker's live payloads.

    The catalog is validated by the real distributed-catalog loader.  The
    compact execution fixture represents a selected four-shard subset of that
    full catalog, so only the exhaustive-master denominator check is replaced
    here; all catalog, evidence, file-record, and cross-authority validation is
    production code.
    """
    paths = _make_fixture(tmp_path / "fixture", worker_count=1)
    composite = workers.load_bundle_authority(paths["bundle"])
    catalog = _real_catalog(paths["bundle"].parent, composite)
    _rebind_fixture_to_catalog(paths, catalog)

    loaded = workers.load_bundle_authority(paths["bundle"])
    assert loaded["schema"] == partitions.CATALOG_SCHEMA
    assert loaded == catalog

    original_payloads = []
    for shard in composite["shards"]:
        for record in shard["files"].values():
            original_payloads.append(paths["bundle"].parent / record["path"])

    relocated = tmp_path / "relocated" / "worker-result"
    relocated.parent.mkdir()
    evidence_path = paths["evidence"][0]
    shutil.move(str(evidence_path.parent), relocated)
    paths["evidence"] = [relocated / evidence_path.name]
    for path in original_payloads:
        path.unlink(missing_ok=True)

    # The tiny fixture intentionally contains only four selected work items;
    # a real master is exhaustive.  Do not bypass catalog validation or any
    # evidence/input validation merely to accommodate that reduced denominator.
    monkeypatch.setattr(aggregate.worker_plan, "validate_master",
                        lambda *_args, **_kwargs: None)
    result = _aggregate(paths, tmp_path / "output")
    assert result["verdict"] == "COMPLETE_RAW_BIT_CLEAN_CENSUS"
    assert result["authorities"]["bundle_schema"] == partitions.CATALOG_SCHEMA
    aggregate.validate_output(tmp_path / "output")

    # Even if the copied metadata-only catalog and its summary file record are
    # rehashed together, changing a selected binary hash must disagree with
    # the worker's execution authority.
    summary_path = tmp_path / "output/summary.json"
    summary = json.loads(summary_path.read_text())
    bundle_record = summary["authorities"]["bundle"]
    catalog_path = tmp_path / "output" / bundle_record["path"]
    catalog_snapshot = json.loads(catalog_path.read_text())
    selected_key = result["authorities"]["worker_evidence"][0]["shards"][0][
        "shard_key"]
    selected_shard = next(row for row in catalog_snapshot["shards"]
                          if row["shard_key"] == selected_key)
    selected_shard["files"]["binary"]["sha256"] = "0" * 64
    _write_json(catalog_path, catalog_snapshot)
    bundle_record["sha256"] = aggregate.file_sha(catalog_path)
    _write_json(summary_path, summary)
    with pytest.raises(
            aggregate.AggregateError,
            match="evidence bundle_sha256 differs|shard/bundle authority differs"):
        aggregate.validate_output(tmp_path / "output")


@pytest.mark.parametrize("target", ["evidence", "bundle"])
def test_binary_hash_authority_mutation_is_red(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    paths = _make_fixture(tmp_path / "fixture")
    if target == "evidence":
        evidence_path = paths["evidence"][0]
        evidence = json.loads(evidence_path.read_text())
        evidence["work_items"][0]["execution_inputs"]["binary"]["sha256"] = "0" * 64
        _write_json(evidence_path, evidence)
    else:
        bundle = workers.load_bundle_authority(paths["bundle"])
        bundle["shards"][0]["files"]["binary"]["sha256"] = "0" * 64
        monkeypatch.setattr(aggregate.worker_plan, "load_bundle_authority",
                            lambda _path: bundle)
        monkeypatch.setattr(aggregate.worker_plan, "validate_master",
                            lambda *_args, **_kwargs: None)
    with pytest.raises(aggregate.AggregateError, match="binary catalog hash/size differs"):
        _aggregate(paths, tmp_path / "output")


@pytest.mark.parametrize("extra", [-1, 1])
def test_validate_output_rejects_rehashed_raw_sample_underflow_or_overflow(
        tmp_path: Path, extra: int) -> None:
    paths = _make_fixture(tmp_path / "fixture")
    output = tmp_path / "output"
    _aggregate(paths, output)
    census = output / "census.jsonl"
    rows = [json.loads(line) for line in census.read_text().splitlines()]
    samples = rows[0]["timing"]["confirm_runs"][0]["samples"]
    if extra < 0:
        samples.pop()
    else:
        samples.append({"sample_index": len(samples), "us": 9.99})
    census.write_bytes(b"".join(aggregate.canonical(row) + b"\n" for row in rows))
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["census"]["sha256"] = aggregate.file_sha(census)
    _write_json(summary_path, summary)
    with pytest.raises(aggregate.AggregateError,
                       match="raw sample denominator differs"):
        aggregate.validate_output(output)


def test_validate_output_cross_checks_summary_source_and_worker_authority(
        tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path / "fixture")
    output = tmp_path / "output"
    _aggregate(paths, output)
    summary_path = output / "summary.json"
    original = json.loads(summary_path.read_text())
    planted = json.loads(json.dumps(original))
    planted["authorities"]["source_sha"] = "0" * 40
    _write_json(summary_path, planted)
    with pytest.raises(aggregate.AggregateError,
                       match="bundle/source authority differs"):
        aggregate.validate_output(output)

    _write_json(summary_path, original)
    census = output / "census.jsonl"
    rows = [json.loads(line) for line in census.read_text().splitlines()]
    rows[0]["authority"]["worker_evidence_sha256"] = "0" * 64
    rows[0]["timing"]["provenance"]["worker_evidence_sha256"] = "0" * 64
    provenance_sha = aggregate.digest(rows[0]["timing"]["provenance"])
    rows[0]["timing"]["screen"]["provenance_sha256"] = provenance_sha
    for run in rows[0]["timing"]["confirm_runs"]:
        run["provenance_sha256"] = provenance_sha
    census.write_bytes(b"".join(aggregate.canonical(row) + b"\n" for row in rows))
    planted = json.loads(json.dumps(original))
    planted["census"]["sha256"] = aggregate.file_sha(census)
    _write_json(summary_path, planted)
    with pytest.raises(aggregate.AggregateError,
                       match="census/summary measurement authority differs"):
        aggregate.validate_output(output)


@pytest.mark.parametrize("target", ["evidence", "bundle"])
def test_validate_output_rejects_rehashed_binary_authority_mutation(
        tmp_path: Path, target: str) -> None:
    paths = _make_fixture(tmp_path / "fixture")
    output = tmp_path / "output"
    _aggregate(paths, output)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    if target == "evidence":
        record = summary["authorities"]["worker_evidence"][0]["evidence"]
        evidence_path = output / record["path"]
        evidence = json.loads(evidence_path.read_text())
        evidence["work_items"][0]["execution_inputs"]["binary"][
            "sha256"] = "0" * 64
        _write_json(evidence_path, evidence)
        record["sha256"] = aggregate.file_sha(evidence_path)
    else:
        record = summary["authorities"]["bundle"]
        bundle_path = output / record["path"]
        bundle = json.loads(bundle_path.read_text())
        bundle["shards"][0]["files"]["binary"]["sha256"] = "0" * 64
        _write_json(bundle_path, bundle)
        record["sha256"] = aggregate.file_sha(bundle_path)
    _write_json(summary_path, summary)
    with pytest.raises(
            aggregate.AggregateError,
            match=("authority path differs|evidence bundle_sha256 differs|"
                   "shard/bundle authority differs")):
        aggregate.validate_output(output)
