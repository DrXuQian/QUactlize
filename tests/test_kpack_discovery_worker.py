from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kpack_discovery_worker_plan as worker_plan  # noqa: E402
import run_kpack_discovery_worker as worker  # noqa: E402
import aggregate_kpack_discovery_results as aggregate  # noqa: E402
import kpack_discovery_build_partitions as partitions  # noqa: E402


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atom_metadata(item_id: str, phase: str, round_index: int, order: str,
                   seed: int, *, grouped_warmups: int | None = None) -> dict:
    return {
        "work_item_id": item_id,
        "phase": phase,
        "round": round_index,
        "order": order,
        "worker": 0,
        "schedule_seed": worker.schedule_seed_hex(seed),
        "grouped_warmups": ("NONE" if grouped_warmups is None else
                            grouped_warmups),
    }


def _write_fake_binary(path: Path, counter: Path, *, route: str,
                       operator: str, qtype: int, parents: int) -> None:
    """Write a fake that echoes the controls received through its real argv."""
    prelude = (
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import sys\n"
        f"p=Path({str(counter)!r}); "
        "p.write_text(str((int(p.read_text()) if p.exists() else 0)+1))\n"
        "def option(name):\n"
        "    prefix = '--' + name + '='\n"
        "    values = [arg[len(prefix):] for arg in sys.argv[1:] "
        "if arg.startswith(prefix)]\n"
        "    if len(values) != 1:\n"
        "        raise SystemExit(91)\n"
        "    return values[0]\n"
        "iterations = int(option('iterations'))\n"
        "correctness = int(option('correctness-repeats'))\n"
        "seed = int(option('schedule-seed'), 0)\n"
    )
    if operator == "grouped":
        prelude += "warmups = int(option('warmups'))\n"
    if (route, operator) == ("scalefirst", "dense"):
        body = (
            f"print(f'SF_SHARD qtype={qtype} typed_rows={parents} "
            f"selected_rows={parents} iterations={{iterations}} "
            "correctness_repeats={correctness} "
            "schedule_seed=0x{seed:016x}')\n"
            "print('SF_ATTEMPT symbol=s0')\n"
            "print('SF_ATTEMPT symbol=s1')\n"
            f"print('SF_COMPLETE status=COMPLETE shape=8x256x512 "
            f"typed_rows={parents}')\n"
        )
    elif (route, operator) == ("fully-quantized", "dense"):
        body = (
            f"print(f'FQ_SHARD q={qtype} typed_rows={parents} "
            f"selected_rows={parents} iterations={{iterations}} "
            "correctness_repeats={correctness} "
            "schedule_seed=0x{seed:016x}')\n"
            f"print('FQ_SHAPE_DONE q={qtype} shape=8x256x512 "
            f"typed_rows={parents} selected_rows={parents} status=PASS')\n"
        )
    elif (route, operator) == ("scalefirst", "grouped"):
        body = (
            f"print(f'SF_GROUPED_SHARD q={qtype} selected_rows={parents} "
            "total_rows=8 max_rows=1 workload=grouped router_profile=uniform "
            "iterations={iterations} warmups={warmups} "
            "correctness_repeats={correctness} "
            "schedule_seed=0x{seed:016x}')\n"
            "print('SF_GROUPED_CELL symbol=s0')\n"
            "print('SF_GROUPED_CELL symbol=s1')\n"
            f"print('SF_GROUPED_COMPLETE status=PASS rows={parents}')\n"
        )
    elif (route, operator) == ("fully-quantized", "grouped"):
        body = (
            f"print(f'FQ_GROUPED_KPACK_SHARD q={qtype} "
            f"selected_rows={parents} total_rows=8 max_rows=1 "
            "workload=grouped router_profile=uniform "
            "iterations={iterations} warmups={warmups} "
            "correctness_repeats={correctness} "
            "schedule_seed=0x{seed:016x}')\n"
            f"print('FQ_GROUPED_KPACK_COMPLETE status=PASS rows={parents}')\n"
        )
    else:  # pragma: no cover - the test helper owns exactly four binaries.
        raise AssertionError((route, operator))
    path.write_text(prelude + body, encoding="utf-8")
    path.chmod(0o755)


def test_worker_executor_self_test() -> None:
    worker.self_test()


def test_selection_is_rederived_from_live_bundle_plan_and_assignment(
        tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.json"
    plan = tmp_path / "plan.json"
    master_path = tmp_path / "master.json"
    assignment_path = tmp_path / "assignment.json"
    selection_path = tmp_path / "selection.json"
    _write_json(bundle, {
        "schema": "fixture-bundle",
        "shards": [{
            "shard_key": "fq-q10-dense-0", "route": "fully-quantized",
            "operator": "dense", "qtype": 10,
            "manifest_sha256": "1" * 64, "parent_ids": ["p0", "p1"],
        }],
    })
    _write_json(plan, {
        "schema": "fixture-plan", "dense": [{"key": "shape-a"}],
        "grouped": [{"key": "router-a"}],
    })
    master = worker_plan.make_master(bundle, plan)
    _write_json(master_path, master)
    assignment = worker_plan.make_assignment(
        master, worker_plan.file_sha(master_path), 1)
    _write_json(assignment_path, assignment)
    expected = worker_plan.make_worker_selection(
        master, assignment, 0,
        master_sha256=worker_plan.file_sha(master_path),
        assignment_sha256=worker_plan.file_sha(assignment_path))
    _write_json(selection_path, expected)

    _master, _assignment, observed = worker._selection(
        bundle, plan, master_path, assignment_path, selection_path, 0)
    assert observed == expected

    stale = copy.deepcopy(expected)
    stale["work_items"][0]["workload_key"] = "shape-stale"
    _write_json(selection_path, stale)
    with pytest.raises(worker.ExecutionError, match="selection differs"):
        worker._selection(
            bundle, plan, master_path, assignment_path, selection_path, 0)


def test_assignment_is_partition_affine_and_balanced_for_32_by_8(
        tmp_path: Path) -> None:
    bundle = tmp_path / "catalog.json"
    workload_plan = tmp_path / "workloads.json"
    shards = []
    for partition_id in range(32):
        for route in ("scalefirst", "fully-quantized"):
            key = f"{route}:q10-dense-p{partition_id:02d}"
            shards.append({
                "shard_key": key, "route": route, "operator": "dense",
                "qtype": 10, "manifest_sha256": f"{partition_id + 1:064x}",
                "parent_ids": [f"{route}-{partition_id}"],
                "partition_id": partition_id,
                "artifact_id": f"artifact/{route}/{partition_id}",
            })
    _write_json(bundle, {"schema": "fixture-bundle", "shards": shards})
    _write_json(workload_plan, {
        "schema": "fixture-plan",
        "dense": [{"key": f"shape-{index}"} for index in range(5)],
        "grouped": [{"key": "router"}],
    })
    master = worker_plan.make_master(bundle, workload_plan)
    assignment = worker_plan.make_assignment(master, "a" * 64, 8)
    assert assignment["assignment_policy"] == \
        "PARTITION_OR_SHARD_AFFINE_GREEDY_LPT_V1"
    assert [len(row["partition_ids"]) for row in assignment["workers"]] == [4] * 8
    owners = {}
    items = {row["work_item_id"]: row for row in master["work_items"]}
    for row in assignment["workers"]:
        for item_id in row["work_item_ids"]:
            partition_id = items[item_id]["partition_id"]
            assert owners.setdefault(partition_id, row["worker_id"]) == row["worker_id"]
    assert set(owners) == set(range(32))

    # Preserve the exact union while scattering one shard's workload.
    planted = copy.deepcopy(assignment)
    source = planted["workers"][0]
    target = planted["workers"][1]
    source_by_shard = {}
    for item_id in source["work_item_ids"]:
        source_by_shard.setdefault(items[item_id]["shard_key"], []).append(item_id)
    split = next(ids for ids in source_by_shard.values() if len(ids) > 1)
    moved = split[0]
    replacement = next(item_id for item_id in target["work_item_ids"]
                       if items[item_id]["shard_key"] != items[moved]["shard_key"])
    source["work_item_ids"].remove(moved)
    target["work_item_ids"].remove(replacement)
    source["work_item_ids"].append(replacement)
    target["work_item_ids"].append(moved)
    with pytest.raises(worker_plan.PlanError, match="scatters one shard"):
        worker_plan.validate_assignment(planted, master, "a" * 64)


def _catalog_fixture(tmp_path: Path) -> tuple[dict, dict, Path, Path]:
    root = tmp_path / "artifact"
    root.mkdir()
    binary = root / "kernel"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "compiled_parents": [{"parent_id": 0, "symbol": "candidate"}],
    }) + "\n")
    receipt = root / "receipt.json"
    receipt.write_text("{}\n")

    def record(path: Path) -> dict:
        return {"path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": worker.file_sha(path)}

    base = {
        "shard_key": "scalefirst:sf-q10-dense-000",
        "native_shard_key": "sf-q10-dense-000",
        "route": "scalefirst", "operator": "dense", "qtype": 10,
        "layout": 2, "parent_begin": 0, "parent_end": 1,
        "parent_count": 1, "authority_count": 1, "parent_ids": [0],
        "parent_ids_sha256": partitions._parent_digest([0]),
        "parent_id_set_sha256": partitions._parent_set_digest([0]),
        "parent_symbols_sha256": partitions.digest(["candidate"]),
        "partition_id": 0, "bucket_ordinal": 0,
        "files": {"manifest": record(manifest), "binary": record(binary),
                  "binary_receipt": record(receipt)},
        "device_arch": "PPU ppu0010", "inspector_output_sha256": "9" * 64,
    }
    source = {
        "source_sha": "1" * 40, "source_tree": "2" * 40,
        "submodules": [], "sdk": {"identity": "sdk-a"},
    }
    artifact_id = "artifact/scalefirst/0"
    partition = {
        **source, "artifact_id": artifact_id, "route": "scalefirst",
        "partition_id": 0, "partition_count": 1, "shards": [base],
    }
    partition_path = root / "partition-bundle.json"
    _write_json(partition_path, partition)
    catalog_row = {
        **base, "artifact_id": artifact_id,
        "mapping_id": partitions.sf_analyzer.MAPPING[2],
        "manifest_sha256": base["files"]["manifest"]["sha256"],
    }
    catalog = {
        "schema": partitions.CATALOG_SCHEMA, **source,
        "partition_count": 1,
        "partitions": [{"artifact_id": artifact_id,
                        "route": "scalefirst", "partition_id": 0,
                        "partition_manifest": record(partition_path)}],
        "shards": [catalog_row],
    }
    selection = {"artifact_ids": [artifact_id],
                 "shard_keys": [catalog_row["shard_key"]]}
    return catalog, selection, root, partition_path


def test_catalog_to_worker_live_artifact_closure_and_negative_plants(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog, selection, root, partition_path = _catalog_fixture(tmp_path)
    monkeypatch.setattr(partitions, "validate_catalog_document", lambda value: value)
    monkeypatch.setattr(
        partitions, "verify_partition",
        lambda path, _root: json.loads(path.read_text()))
    argument = [f"{selection['artifact_ids'][0]}={root}"]
    contexts = worker._catalog_artifacts(catalog, selection, argument)
    resolved = worker.resolve_catalog_shard(
        catalog, contexts, selection["shard_keys"][0])
    assert resolved.parent_count == 1
    assert resolved.binary.name == "kernel"

    with pytest.raises(worker.ExecutionError, match="artifact-root set differs"):
        worker._catalog_artifacts(catalog, selection, [])

    original = partition_path.read_bytes()
    partition_path.write_bytes(original + b" ")
    with pytest.raises(worker.ExecutionError, match="manifest differs"):
        worker._catalog_artifacts(catalog, selection, argument)
    partition_path.write_bytes(original)

    for field, planted in (("source_sha", "3" * 40),
                           ("sdk", {"identity": "sdk-b"})):
        document = json.loads(original)
        document[field] = planted
        _write_json(partition_path, document)
        catalog["partitions"][0]["partition_manifest"] = {
            "path": "partition-bundle.json", "size": partition_path.stat().st_size,
            "sha256": worker.file_sha(partition_path)}
        with pytest.raises(worker.ExecutionError, match=f"{field} differs"):
            worker._catalog_artifacts(catalog, selection, argument)
    partition_path.write_bytes(original)
    catalog["partitions"][0]["partition_manifest"] = {
        "path": "partition-bundle.json", "size": partition_path.stat().st_size,
        "sha256": worker.file_sha(partition_path)}
    contexts = worker._catalog_artifacts(catalog, selection, argument)
    (root / "kernel").write_text("#!/bin/sh\nexit 7\n")
    with pytest.raises(worker.ExecutionError, match="binary bytes differ"):
        worker.resolve_catalog_shard(
            catalog, contexts, selection["shard_keys"][0])


def test_atomic_log_reuses_only_a_structurally_complete_full_parent_run(
        tmp_path: Path) -> None:
    count = tmp_path / "count"
    binary = tmp_path / "kernel.py"
    _write_fake_binary(
        binary, count, route="scalefirst", operator="dense", qtype=10,
        parents=4)
    manifest, receipt = tmp_path / "manifest", tmp_path / "receipt"
    manifest.write_text("manifest\n")
    receipt.write_text("receipt\n")
    shard = worker.ResolvedShard(
        "sf", "scalefirst", "dense", 10, 4,
        binary, manifest, receipt)
    workload = worker.Workload("shape", "dense", {
        "workload_key": "shape", "source_class": "real-inventory",
        "m": "8", "n": "256", "k": "512"})
    log = tmp_path / "results/screen/item.log"
    seed = worker.schedule_seed("a" * 64, "screen", 0)
    command = worker.command_for(
        shard, workload, iterations=5, correctness_repeats=8,
        warmups=3, schedule_seed=seed)
    metadata = _atom_metadata(
        "a" * 64, "screen", 0, "SCREEN", seed)
    worker.run_atomic_log(
        log, command, shard, workload, metadata)
    assert count.read_text() == "1"
    assert log.is_file()

    # A valid final log is reused, not rerun.
    worker.run_atomic_log(log, command, shard, workload, metadata)
    assert count.read_text() == "1"

    with pytest.raises(worker.ExecutionError, match="metadata/argv"):
        worker.run_atomic_log(
            log, command, shard, workload,
            _atom_metadata("a" * 64, "confirm", 1, "FORWARD", 7))
    with pytest.raises(worker.ExecutionError, match="metadata/argv"):
        worker.run_atomic_log(
            log, command + ["--changed-argv"], shard, workload, metadata)
    assert count.read_text() == "1"

    log.write_text(log.read_text().replace("selected_rows=4", "selected_rows=3"))
    with pytest.raises(worker.ExecutionError, match="header authority differs"):
        worker.run_atomic_log(log, command, shard, workload, metadata)
    assert count.read_text() == "1"


def test_seed_warmup_and_dense_no_warmup_headers_fail_closed(
        tmp_path: Path) -> None:
    auxiliary = tmp_path / "authority"
    auxiliary.write_text("authority\n")

    dense_binary = tmp_path / "dense.py"
    _write_fake_binary(
        dense_binary, tmp_path / "dense.count", route="scalefirst",
        operator="dense", qtype=10, parents=2)
    dense = worker.ResolvedShard(
        "sf-d", "scalefirst", "dense", 10, 2,
        dense_binary, auxiliary, auxiliary, ("s0", "s1"))
    dense_workload = worker.Workload("dense", "dense", {
        "workload_key": "dense", "source_class": "real-inventory",
        "m": "8", "n": "256", "k": "512"})
    item_id = "1" * 64
    seed = worker.schedule_seed(item_id, "screen", 0)
    dense_command = worker.command_for(
        dense, dense_workload, iterations=5, correctness_repeats=8,
        warmups=3, schedule_seed=seed)
    dense_metadata = _atom_metadata(
        item_id, "screen", 0, "SCREEN", seed)
    dense_log = tmp_path / "results/screen/dense.log"
    worker.run_atomic_log(
        dense_log, dense_command, dense, dense_workload, dense_metadata)
    lines = dense_log.read_text().splitlines()
    wrong_seed = seed ^ 1
    lines = [
        line.replace(worker.schedule_seed_hex(seed),
                     worker.schedule_seed_hex(wrong_seed))
        if line.startswith("SF_SHARD ") else line
        for line in lines]
    with pytest.raises(worker.ExecutionError, match="dense shard schedule seed differs"):
        worker.validate_atom_log(
            "\n".join(lines) + "\n", dense_command, dense, dense_workload,
            dense_metadata)

    with pytest.raises(worker.ExecutionError,
                       match="dense atom may not use grouped warmups"):
        worker.run_atomic_log(
            tmp_path / "results/screen/dense-warmup.log",
            dense_command + ["--warmups=3"], dense, dense_workload,
            dense_metadata)

    grouped_binary = tmp_path / "grouped.py"
    _write_fake_binary(
        grouped_binary, tmp_path / "grouped.count", route="scalefirst",
        operator="grouped", qtype=10, parents=2)
    grouped = worker.ResolvedShard(
        "sf-g", "scalefirst", "grouped", 10, 2,
        grouped_binary, auxiliary, auxiliary, ("s0", "s1"))
    grouped_workload = worker.Workload("grouped", "grouped", {
        "workload_key": "grouped", "source_class": "real-inventory",
        "tokens": "4", "topk": "2", "experts": "16",
        "n": "256", "k": "512", "profile": "uniform",
        "rows_file": "-", "total_rows": "8", "max_rows": "1",
        "rows_sha256": "-"})
    grouped_id = "2" * 64
    grouped_seed = worker.schedule_seed(grouped_id, "screen", 0)
    grouped_command = worker.command_for(
        grouped, grouped_workload, iterations=5, correctness_repeats=8,
        warmups=3, schedule_seed=grouped_seed)
    grouped_metadata = _atom_metadata(
        grouped_id, "screen", 0, "SCREEN", grouped_seed,
        grouped_warmups=3)
    grouped_log = tmp_path / "results/screen/grouped.log"
    worker.run_atomic_log(
        grouped_log, grouped_command, grouped, grouped_workload,
        grouped_metadata)
    lines = grouped_log.read_text().splitlines()
    lines = [line.replace("warmups=3", "warmups=4")
             if line.startswith("SF_GROUPED_SHARD ") else line
             for line in lines]
    with pytest.raises(worker.ExecutionError, match="grouped shard warmups differs"):
        worker.validate_atom_log(
            "\n".join(lines) + "\n", grouped_command, grouped,
            grouped_workload, grouped_metadata)


def test_loaded_runtime_symlink_records_alias_and_hashes_target(
        tmp_path: Path) -> None:
    target = tmp_path / "libhggc.so.13.0"
    target.write_bytes(b"runtime-identity")
    alias = tmp_path / "libhggc.so.13"
    alias.symlink_to(target.name)
    worker._loaded_library_record.cache_clear()
    reported, resolved, size, sha = worker._loaded_library_record(str(alias))
    assert reported == str(alias)
    assert resolved == str(target)
    assert size == len(b"runtime-identity")
    assert sha == hashlib.sha256(b"runtime-identity").hexdigest()


def _runtime_link(
        soname: str, *, reported: str | None = None,
        resolved: str | None = None, size: int = 4096,
        sha: str = "a" * 64) -> tuple[str, str, str, int, str]:
    stem = soname.replace("/", "_")
    return (
        soname,
        reported or f"/sdk/alias/{stem}",
        resolved or f"/sdk/runtime/{stem}",
        size,
        sha,
    )


@pytest.mark.parametrize("payload_kind", ["equal", "strict-subset"])
def test_payload_runtime_linkage_accepts_the_probe_closure_or_its_exact_subset(
        payload_kind: str) -> None:
    wrapper = _runtime_link("libhggc_wrapper.so", sha="1" * 64)
    runtime = _runtime_link("libhggcrt.13.0.so", sha="2" * 64)
    compiler = _runtime_link("libhggc.so", sha="3" * 64)
    probe = tuple(sorted((wrapper, runtime, compiler)))
    payload = probe if payload_kind == "equal" else (wrapper,)

    worker._validate_payload_linkage(payload, probe, "fixture-shard")


@pytest.mark.parametrize(("payload", "failure"), [
    ((), "payload runtime linkage is empty"),
    (tuple(sorted((
        _runtime_link("libhggc_wrapper.so", sha="1" * 64),
        _runtime_link("libhggc_foreign.so", sha="4" * 64),
    ))), "payload runtime linkage differs"),
    ((_runtime_link("libhggc_wrapper.so", sha="5" * 64),),
     "payload runtime linkage differs"),
    ((_runtime_link("libhggcrt.13.0.so", sha="2" * 64),),
     "payload runtime linkage lacks libhggc_wrapper.so"),
], ids=("empty", "foreign-record", "same-soname-changed-hash",
        "missing-wrapper-anchor"))
def test_payload_runtime_linkage_rejects_empty_foreign_changed_or_unanchored(
        payload: tuple[tuple[str, str, str, int, str], ...],
        failure: str) -> None:
    probe = tuple(sorted((
        _runtime_link("libhggc_wrapper.so", sha="1" * 64),
        _runtime_link("libhggcrt.13.0.so", sha="2" * 64),
        _runtime_link("libhggc.so", sha="3" * 64),
    )))

    with pytest.raises(worker.ExecutionError, match=failure):
        worker._validate_payload_linkage(payload, probe, "fixture-shard")


def test_runtime_linkage_rejects_duplicate_sonames(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "payload"
    binary.write_bytes(b"ELF fixture")
    output = (
        "libhggc_wrapper.so => /sdk/one/libhggc_wrapper.so (0x1)\n"
        "libhggc_wrapper.so => /sdk/two/libhggc_wrapper.so (0x2)\n"
    )
    monkeypatch.setattr(
        worker.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=output))
    monkeypatch.setattr(
        worker, "_loaded_library_record",
        lambda path: (path, path, 4096, hashlib.sha256(path.encode()).hexdigest()))
    worker._linkage.cache_clear()

    with pytest.raises(worker.ExecutionError, match="duplicate.*SONAME"):
        worker._linkage(binary)


def test_distinct_prebuilt_probes_produce_byte_identical_device_identity(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wire = (
        "#!/bin/sh\n"
        "printf '%s\\n' 'QZ_HGGC_DEVICE_PROBE_V1' "
        "'count\t1' "
        "'device\t0\t5050552d5a57383130\t0\t0\t72\t0000:08:00.0' "
        "'pci_method\thggcDeviceGetPCIBusId' "
        "'driver\t13000\thggcDriverGetVersion'\n")
    records = {}
    for route in ("scalefirst", "fully-quantized"):
        path = tmp_path / f"{route}-probe"
        path.write_text(wire)
        path.chmod(0o755)
        records[route] = {
            "runtime_probe": {"binary": {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }}}
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}\n")
    for name in ("PPU_SDK", "PPU_HOME", "PPU_SDK_SITE_DEFAULT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("QUACTLIZE_BOX_SDK_COMPILER_IDENTITY", "sdk-fixture")
    payload = worker._live_device_bytes(
        bundle, {"component_bundles": records})
    identity = json.loads(payload)
    assert identity["identity"]["device_model"]["value"] == "PPU-ZW810"
    assert identity["identity"]["pci_identity"]["value"] == "0000:08:00.0"
    # Probe binary paths are intentionally not part of the identity schema.
    assert "scalefirst-probe" not in payload.decode()
    assert "fully-quantized-probe" not in payload.decode()


def test_empty_structural_confirmation_is_an_explicit_non_device_marker(
        tmp_path: Path) -> None:
    screen = tmp_path / "screen.log"
    symbols = tmp_path / "symbols"
    marker = tmp_path / "confirm.log"
    screen.write_text("screen authority\n")
    symbols.write_bytes(b"")
    item_id = "a" * 64
    seed = worker.schedule_seed(item_id, "confirm", 2)
    worker.write_empty_confirm(
        marker, item_id, 2, "REVERSE", screen, symbols, seed)
    record = {"path": marker.name, "sha256": worker.file_sha(marker)}
    parsed = aggregate.validate_empty_confirm(
        {"round": 2, "order": "REVERSE", "empty_structural": True,
         "schedule_seed": worker.schedule_seed_hex(seed), "log": record},
        tmp_path,
        item={"work_item_id": item_id},
        round_row={"round": 2, "order": "REVERSE"},
        screen_sha256=worker.file_sha(screen),
        retention_sha256=worker.file_sha(symbols))
    assert parsed["empty_structural"] is True
    assert parsed["schedule_seed"] == seed
    assert marker.read_text().count("\n") == 1
    assert "KPACK_DISCOVERY_ATOM" not in marker.read_text()


def test_all_structural_scalefirst_screen_materializes_empty_retention(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    (out / "inputs").mkdir(parents=True)
    manifest_path = tmp_path / "manifest.json"
    screen = tmp_path / "screen.log"
    manifest_path.write_text("{}\n")
    screen.write_text("screen\n")
    auxiliary = tmp_path / "aux"
    auxiliary.write_text("x")
    shard = worker.ResolvedShard(
        "sf", "scalefirst", "dense", 10, 2,
        auxiliary, manifest_path, auxiliary, ("s0", "s1"))
    manifest = {
        "typed_rows": [{"symbol": "s0"}, {"symbol": "s1"}],
        "compiled_parents": [
            {"parent_id": 4, "symbol": "s0"},
            {"parent_id": 5, "symbol": "s1"}],
        "parent_range": {"begin": 4, "end": 6, "count": 2,
                         "authority_count": 8},
        "denominator": {"typed_rows": 2},
    }
    records = {
        symbol: [{"status": "INADMISSIBLE"}] for symbol in ("s0", "s1")}
    monkeypatch.setattr(
        worker.sf_analyzer, "retain",
        lambda *_args: (_ for _ in ()).throw(ValueError("no clean measurement")))
    monkeypatch.setattr(worker.sf_analyzer, "validate_manifest",
                        lambda *_args: manifest)
    monkeypatch.setattr(worker.sf_analyzer, "load_candidates",
                        lambda *_args: [])
    monkeypatch.setattr(worker.sf_analyzer, "product_census",
                        lambda *_args: records)
    retention = worker.materialize_sf_retention(
        out, {"work_item_id": "b" * 64}, shard, screen)
    assert retention.symbols == ()
    assert retention.symbols_path.read_bytes() == b""
    sidecar = json.loads(retention.sidecar_path.read_text())
    assert sidecar["retained_count"] == 0
    assert sidecar["structural_unavailable_symbols"] == ["s0", "s1"]
    assert sidecar["timing_rank_used_for_elimination"] is False


def test_fq_structural_log_rejects_measured_state_and_split_gap(
        tmp_path: Path) -> None:
    authority = tmp_path / "authority"
    authority.write_text("authority\n")
    symbol = "fqk_tc_q10_fixture"
    shard = worker.ResolvedShard(
        "fq", "fully-quantized", "dense", 10, 1,
        authority, authority, authority, (symbol,),
        payload_kind=worker.NO_DEVICE_KERNEL_STRUCTURAL,
        structural_proof=authority)
    workload = worker.Workload("dense", "dense", {
        "workload_key": "dense", "source_class": "real-inventory",
        "m": "1", "n": "16", "k": "128"})
    cells = "".join(
        "FQ_TC_CELL shape=1x16x128 "
        f"symbol={symbol} S={split} "
        "state=SHIPPING_SHARED_STORAGE raw_bad=0\n"
        for split in (1, 2, 4, 8))
    valid = (
        "FQ_SHARD q=10 typed_rows=1 selected_rows=1\n" + cells +
        "FQ_SHAPE_DONE q=10 shape=1x16x128 typed_rows=1 "
        "selected_rows=1 status=PASS\n")
    worker.validate_log(valid, shard, workload)

    measured = valid.replace(
        "state=SHIPPING_SHARED_STORAGE", "state=MEASURED", 1)
    with pytest.raises(worker.ExecutionError,
                       match="proved structural state differs"):
        worker.validate_log(measured, shard, workload)

    split_gap = valid.replace("S=8 ", "S=4 ", 1)
    with pytest.raises(worker.ExecutionError,
                       match="proved structural state differs"):
        worker.validate_log(split_gap, shard, workload)


def test_exact_rows_are_passed_by_path_and_bound_by_fnv(tmp_path: Path) -> None:
    rows = [0] * 256
    rows[0], rows[19], rows[255] = 2, 3, 1
    rows_path = tmp_path / "rows.txt"
    rows_path.write_text("".join(f"{value}\n" for value in rows))
    rows_hash = worker._rows_fnv64(rows_path, 256)
    binary = tmp_path / "grouped"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    auxiliary = tmp_path / "aux"
    auxiliary.write_text("x")
    shard = worker.ResolvedShard(
        "fq", "fully-quantized", "grouped", 11, 7,
        binary, auxiliary, auxiliary)
    workload = worker.Workload("permutation-a", "grouped", {
        "workload_key": "permutation-a", "source_class": "router-control",
        "tokens": "0", "topk": "0", "experts": "256", "n": "512",
        "k": "3072", "profile": "permutation-a",
        "rows_file": "router-rows/permutation-a.txt", "total_rows": "6",
        "max_rows": "3", "rows_sha256": hashlib.sha256(b"x").hexdigest(),
    }, rows_path, rows_hash)
    command = worker.command_for(
        shard, workload, iterations=11, correctness_repeats=256, warmups=3,
        schedule_seed=19)
    assert f"--rows-file={rows_path}" in command
    assert not any(value.startswith("--tokens=") for value in command)
    assert not any(value.startswith("--topk=") for value in command)
    assert not any(value.startswith("--symbol") for value in command)
    assert command.count("--schedule-seed=19") == 1
    text = (
        "FQ_GROUPED_KPACK_SHARD q=11 selected_rows=7 router=exact-rows-v1 "
        "total_rows=6 max_rows=3 workload=permutation-a "
        f"router_profile=permutation-a rows_hash={rows_hash}\n"
        "FQ_GROUPED_KPACK_COMPLETE q=11 status=PASS rows=7\n")
    worker.validate_log(text, shard, workload)
    with pytest.raises(worker.ExecutionError, match="header authority differs"):
        worker.validate_log(text.replace(rows_hash, "0x0000000000000000"),
                            shard, workload)


def test_no_builder_or_timing_prune_is_present() -> None:
    source = (TOOLS / "run_kpack_discovery_worker.py").read_text()
    assert "build.sh" not in source
    assert "--top-n" not in source
    assert "SCREEN_ALL_COMPILED_CONFIRM_ALL_ADMISSIBLE_NO_TIMING_PRUNE" in source
    assert "timing_rank_used_for_elimination" in source
    assert 'default=3' in source


def test_worker_runs_only_selected_four_route_operator_atoms_and_resumes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    authorities = {}
    for name in ("bundle", "plan", "master", "assignment", "selection",
                 "device", "homogeneity"):
        path = tmp_path / f"{name}.json"
        path.write_text(f"{{\"name\":\"{name}\"}}\n")
        authorities[name] = path

    workloads = {
        (10, "dense"): {
            "dense": worker.Workload("dense", "dense", {
                "workload_key": "dense", "source_class": "real-inventory",
                "m": "8", "n": "256", "k": "512"})},
        (10, "grouped"): {
            "grouped": worker.Workload("grouped", "grouped", {
                "workload_key": "grouped", "source_class": "real-inventory",
                "tokens": "4", "topk": "2", "experts": "16",
                "n": "256", "k": "512", "profile": "uniform",
                "rows_file": "-", "total_rows": "8", "max_rows": "1",
                "rows_sha256": "-"})},
    }
    cases = [
        ("a" * 64, "sf-d", "scalefirst", "dense"),
        ("b" * 64, "fq-d", "fully-quantized", "dense"),
        ("c" * 64, "sf-g", "scalefirst", "grouped"),
        ("d" * 64, "fq-g", "fully-quantized", "grouped"),
    ]
    resolved = {}
    selected_items = []
    counters = {}
    for item_id, shard_key, route, operator in cases:
        binary = tmp_path / f"{shard_key}.py"
        counter = tmp_path / f"{shard_key}.count"
        _write_fake_binary(
            binary, counter, route=route, operator=operator, qtype=10,
            parents=2)
        auxiliary = tmp_path / f"{shard_key}.aux"
        auxiliary.write_text("authority\n")
        resolved[shard_key] = worker.ResolvedShard(
            shard_key, route, operator, 10, 2,
            binary, auxiliary, auxiliary,
            ("s0", "s1") if route == "scalefirst" else ())
        counters[shard_key] = counter
        selected_items.append({
            "work_item_id": item_id, "route": route,
            "operator": operator, "qtype": 10,
            "shard_key": shard_key, "manifest_sha256": "1" * 64,
            "parent_id_set_sha256": "2" * 64, "parent_count": 2,
            "workload_key": "dense" if operator == "dense" else "grouped",
            "runtime_partition": "OWNED_BY_ITEM_NOT_SPLIT_ACROSS_WORKERS",
        })
    selection = {"schema": worker_plan.SELECTION_SCHEMA, "worker_id": 0,
                 "worker_count": 1, "work_items": selected_items}
    master = {"bundle_sha256": worker.file_sha(authorities["bundle"]),
              "workload_plan_sha256": worker.file_sha(authorities["plan"])}
    assignment = {"worker_count": 1,
                  "workers": [{"worker_id": 0,
                               "work_item_ids": [row["work_item_id"]
                                                 for row in selected_items]}]}
    composite_doc = {"shards": [
        {"shard_key": key} for key in [*resolved, "unassigned-shard"]]}
    resolve_calls = []

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(worker.composite, "validate_composite",
                        lambda _path: composite_doc)
    monkeypatch.setattr(
        worker, "_selection",
        lambda *_args, **_kwargs: (master, assignment, selection))
    monkeypatch.setattr(worker, "_live_device_bytes", lambda *_args: b"device\n")
    monkeypatch.setattr(worker, "_device_binding",
                        lambda *_args: ({}, "e" * 64))

    def materialize(_plan, output):
        output.mkdir(parents=True, exist_ok=True)
        (output / "index.json").write_text("{}\n")

    monkeypatch.setattr(worker.workload_authority, "materialize", materialize)
    monkeypatch.setattr(worker.workload_authority, "validate",
                        lambda *_args: {})
    monkeypatch.setattr(worker, "read_workloads",
                        lambda _root, qtype, operator: workloads[(qtype, operator)])

    def resolve(_bundle, _document, key):
        resolve_calls.append(key)
        return resolved[key]

    monkeypatch.setattr(worker, "resolve_native_shard", resolve)

    def retain(out, item, _shard, _screen):
        directory = out / "retention"
        directory.mkdir(exist_ok=True)
        symbols = directory / f"{item['work_item_id']}.symbols"
        sidecar = directory / f"{item['work_item_id']}.json"
        if not symbols.exists():
            symbols.write_text("s0\ns1\n")
            sidecar.write_text("{}\n")
        return worker.Retention(("s0", "s1"), symbols, sidecar)

    monkeypatch.setattr(worker, "materialize_sf_retention", retain)
    probes = [tmp_path / "probe-a", tmp_path / "probe-b"]
    monkeypatch.setattr(worker, "_probe_binaries", lambda *_args: probes)
    payload_linkage = (
        ("libhggc_wrapper.so", "/alias/wrapper", "/target/wrapper", 1,
         "d" * 64),)
    linkage = tuple(sorted((
        *payload_linkage,
        ("libhggcrt.13.0.so", "/alias/runtime", "/target/runtime", 2,
         "e" * 64),
        ("libhggc.so", "/alias/compiler", "/target/compiler", 3,
         "f" * 64),
    )))
    monkeypatch.setattr(
        worker, "_linkage",
        lambda binary: linkage if binary in probes else payload_linkage)

    args = SimpleNamespace(
        bundle=authorities["bundle"], plan=authorities["plan"],
        master=authorities["master"], assignment=authorities["assignment"],
        selection=authorities["selection"], device_identity=authorities["device"],
        device_homogeneity=authorities["homogeneity"],
        output=tmp_path / "output", worker_id=0, phase="all", resume=False,
        screen_iterations=5, confirm_iterations=11, confirm_rounds=3,
        correctness_repeats=8, warmups=3)
    assert worker.run_worker(args) == 0
    assert resolve_calls == [row[1] for row in cases]
    assert all(path.read_text() == "4" for path in counters.values())
    result = json.loads((args.output / "worker-result.json").read_text())
    assert result["completed_work_item_ids"] == assignment["workers"][0]["work_item_ids"]
    evidence = json.loads((args.output / "worker-evidence.json").read_text())
    assert evidence["schema"] == aggregate.EVIDENCE_SCHEMA
    aggregate.validate_run_contract(evidence["run_contract"])
    assert evidence["run_contract"]["schedule_seed"] == \
        worker.schedule_seed_contract()
    assert evidence["run_contract"]["grouped_warmups"] == 3
    assert evidence["run_contract"]["confirm"]["rounds"] == [
        {"round": 1, "order": "FORWARD"},
        {"round": 2, "order": "REVERSE"},
        {"round": 3, "order": "HASHED"},
    ]
    execution_record = evidence["execution_authority"]
    execution_path = args.output / execution_record["path"]
    assert worker.file_sha(execution_path) == execution_record["sha256"]
    execution = json.loads(execution_path.read_text())
    assert execution["schema"] == worker.EXECUTION_SCHEMA
    assert execution["worker_id"] == 0
    assert execution["device_identity_sha256"] == "e" * 64
    assert execution["measurement"] == {
        "screen_iterations": 5,
        "confirm_iterations": 11,
        "confirm_rounds": 3,
        "correctness_repeats": 8,
        "grouped_warmups": 3,
        "schedule_seed": worker.schedule_seed_contract(),
        "outer_round_order": "ASSIGNMENT_REVERSE_THEN_HASHED_V1",
        "inner_candidate_order": "ROUND_SEED_VARIED_ALL_ROUTES_OPERATORS",
    }
    assert execution["runtime_linkage"] == [list(row) for row in linkage]
    assert [row["work_item_id"] for row in evidence["work_items"]] == \
        assignment["workers"][0]["work_item_ids"]
    assert all(len(row["confirm"]) == 3 for row in evidence["work_items"])
    route_by_id = {item["work_item_id"]: item["route"]
                   for item in selected_items}
    selected_by_id = {item["work_item_id"]: item for item in selected_items}
    for row in evidence["work_items"]:
        item_id = row["work_item_id"]
        selected = selected_by_id[item_id]
        shard = resolved[selected["shard_key"]]
        inputs = row["execution_inputs"]
        assert inputs["artifact_id"] is None
        assert inputs["shard_key"] == shard.key
        assert inputs["binary"] == {
            "executed_path": str(shard.binary),
            "size": shard.binary.stat().st_size,
            "sha256": worker.file_sha(shard.binary),
        }
        assert inputs["binary_receipt"] == {
            "size": shard.receipt.stat().st_size,
            "sha256": worker.file_sha(shard.receipt),
        }
        manifest = inputs["manifest"]
        snapshot = args.output / manifest["snapshot"]["path"]
        assert manifest["size"] == shard.manifest.stat().st_size
        assert manifest["sha256"] == worker.file_sha(shard.manifest)
        assert worker.file_sha(snapshot) == manifest["snapshot"]["sha256"]
        assert inputs["rows_file"] is None
        if selected["route"] == "scalefirst":
            retained_path = args.output / row["retention"]["symbols"]["path"]
            assert inputs["retention_symbols_executed_path"] == str(
                retained_path.resolve())
        else:
            assert inputs["retention_symbols_executed_path"] is None
        if selected["operator"] == "grouped":
            assert "--warmups=3" in row["screen"]["argv"]
        else:
            assert not any(arg.startswith("--warmups=")
                           for arg in row["screen"]["argv"])
        assert row["screen"]["schedule_seed"] == worker.schedule_seed_hex(
            worker.schedule_seed(item_id, "screen", 0))
        for round_index, confirm in enumerate(row["confirm"], 1):
            assert confirm["schedule_seed"] == worker.schedule_seed_hex(
                worker.schedule_seed(item_id, "confirm", round_index))
        for phase, round_index, order, log in [
                ("screen", 0, "SCREEN",
                 args.output / row["screen"]["log"]["path"]),
                *(("confirm", index, worker._round_label(index),
                   args.output / confirm["log"]["path"])
                  for index, confirm in enumerate(row["confirm"], 1))]:
            atom = worker._one_line(log.read_text(), "KPACK_DISCOVERY_ATOM ")
            seed = worker.schedule_seed(item_id, phase, round_index)
            assert atom["work_item_id"] == item_id
            assert atom["phase"] == phase
            assert atom["round"] == str(round_index)
            assert atom["order"] == order
            assert atom["worker"] == "0"
            assert atom["schedule_seed"] == worker.schedule_seed_hex(seed)
            assert atom["grouped_warmups"] == (
                "3" if selected["operator"] == "grouped" else "NONE")
    assert all(row["retention"] is not None
               for row in evidence["work_items"]
               if route_by_id[row["work_item_id"]] == "scalefirst")
    assert all(any(arg.startswith("--symbol-file=") for arg in confirm["argv"])
               for row in evidence["work_items"] if row["retention"] is not None
               for confirm in row["confirm"])
    assert all(row["retention"] is None
               for row in evidence["work_items"]
               if route_by_id[row["work_item_id"]] == "fully-quantized")
    assert len(list((args.output / "results/screen").glob("*.log"))) == 4
    assert sum(len(list((args.output / f"results/confirm-r{round_}").glob("*.log")))
               for round_ in (1, 2, 3)) == 12

    # The second run validates and reuses every immutable log and marker.
    args.resume = True
    assert worker.run_worker(args) == 0
    assert all(path.read_text() == "4" for path in counters.values())
