#!/usr/bin/env python3
"""Freeze and seal an adaptive K-pack execution epoch.

The adaptive campaign deliberately separates immutable build authority from
measurement authority.  ``prepare`` imports only metadata from one finalized
full campaign, verifies every referenced partition artifact from its live
bytes, and creates a new execution epoch.  Historical result directories are
never copied or accepted as inputs.

The build commit and the executor commit may differ.  This is safe only while
the frozen catalog still validates against the live contract and every
partition payload matches its recorded size and SHA-256.  Both identities and
the exact executor file hashes are recorded in ``epoch.json``.

``seal-screen`` is the eight-worker barrier.  It binds every screen log, the
exact assignment, the current device identities, and the fixed quick-screen
measurement contract.  Confirmation is intentionally outside this first
adapter; a sealed stage ends at ``SCREEN_COMPLETE_CONFIRM_PENDING`` so a later
global retention adapter cannot accidentally fall back to confirming the full
catalog.
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
from datetime import datetime, timezone
from typing import Any, Iterable, NoReturn


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import box_identity_schema  # noqa: E402
import kpack_discovery_build_partitions as partitions  # noqa: E402
import kpack_discovery_worker_plan as worker_plan  # noqa: E402
import kpack_postfix_full_campaign as full_campaign  # noqa: E402
import materialize_kpack_discovery_workloads as workload_materializer  # noqa: E402


EPOCH_SCHEMA = "quactlize.kpack-postfix-adaptive-epoch.v1"
SCREEN_BARRIER_SCHEMA = "quactlize.kpack-postfix-adaptive-screen-barrier.v1"
STAGE_SCHEMA = "quactlize.kpack-postfix-adaptive-stage.v1"
WORKERS = 8
SCREEN_ITERATIONS = 2
SCREEN_WARMUPS = 1
CORRECTNESS_REPEATS = 1
CONFIRM_ITERATIONS = 11
CONFIRM_ROUNDS = 3
CONFIRM_WARMUPS = 3
PENDING_STAGE = "SCREEN_COMPLETE_CONFIRM_PENDING"
SHA256 = set("0123456789abcdef")


class AdaptiveError(ValueError):
    """A reused build, execution epoch, or phase barrier is unsafe."""


def fail(message: str) -> NoReturn:
    raise SystemExit(f"kpack postfix adaptive campaign: {message}")


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as error:
        raise AdaptiveError(f"value is not canonical JSON: {error}") from error


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    try:
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                value.update(block)
        return value.hexdigest()
    except OSError as error:
        raise AdaptiveError(f"cannot hash {path}: {error}") from error


def _json(path: Path, label: str) -> dict[str, Any]:
    _plain_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdaptiveError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise AdaptiveError(f"{label} is not an object")
    return value


def _plain_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise AdaptiveError(f"{label} must be one regular non-symlink file: {path}")
    return path


def _plain_directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise AdaptiveError(f"{label} may not be a symlink")
    try:
        result = path.resolve(strict=True)
    except OSError as error:
        raise AdaptiveError(f"cannot resolve {label}: {error}") from error
    if not result.is_dir():
        raise AdaptiveError(f"{label} is not a directory")
    return result


def _sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in SHA256 for character in value)):
        raise AdaptiveError(f"{label} is not one lowercase SHA-256")
    return value


def _git(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *arguments], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise AdaptiveError(f"cannot read executor Git authority: {error}") from error


def _write_frozen_bytes(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise AdaptiveError(f"frozen output may not be a symlink: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise AdaptiveError(f"existing frozen output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_frozen_json(path: Path, value: dict[str, Any]) -> None:
    _write_frozen_bytes(path, (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8"))


def _copy_frozen(source: Path, target: Path, label: str) -> None:
    _plain_file(source, label)
    _write_frozen_bytes(target, source.read_bytes())


def _disjoint(first: Path, second: Path) -> None:
    first = first.resolve(strict=True)
    second = second.resolve(strict=True)
    try:
        first.relative_to(second)
    except ValueError:
        pass
    else:
        raise AdaptiveError("adaptive output is inside the reused campaign")
    try:
        second.relative_to(first)
    except ValueError:
        pass
    else:
        raise AdaptiveError("reused campaign is inside the adaptive output")


def _selection(source: Path, master: dict[str, Any],
               assignment: dict[str, Any], worker: int) -> dict[str, Any]:
    path = source / "selections" / f"worker-{worker}.json"
    observed = _json(path, f"worker {worker} selection")
    expected = worker_plan.make_worker_selection(
        master, assignment, worker,
        master_sha256=file_sha(source / "master.json"),
        assignment_sha256=file_sha(source / "assignment.json"))
    if observed != expected:
        raise AdaptiveError(f"worker {worker} selection differs")
    return observed


def _artifact_roots(source: Path, catalog: dict[str, Any],
                    selections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = {row["artifact_id"] for row in catalog["partitions"]}
    observed: dict[str, Path] = {}
    for worker, selection in enumerate(selections):
        path = source / "artifact-roots" / f"worker-{worker}.tsv"
        _plain_file(path, f"worker {worker} artifact roots")
        rows: list[tuple[str, Path]] = []
        for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split("\t")
            if len(fields) != 2 or not fields[0] or not fields[1]:
                raise AdaptiveError(
                    f"worker {worker} artifact row {line_number} is malformed")
            artifact_id, raw_root = fields
            root = Path(raw_root)
            if not root.is_absolute():
                raise AdaptiveError("artifact root must be absolute")
            rows.append((artifact_id, root))
        if [artifact_id for artifact_id, _root in rows] != \
                selection["artifact_ids"]:
            raise AdaptiveError(f"worker {worker} artifact-root order differs")
        for artifact_id, root in rows:
            if artifact_id in observed:
                raise AdaptiveError("artifact root is duplicated across workers")
            observed[artifact_id] = root
    if set(observed) != expected:
        raise AdaptiveError("artifact-root union has a gap or foreign entry")

    # This is intentionally a full live-byte walk.  A build/source commit
    # mismatch is admitted only because every manifest, binary, receipt, and
    # structural proof is checked against the catalog before the epoch exists.
    records = []
    for artifact_id, root in sorted(observed.items()):
        try:
            partitions.verify_catalog_artifact(catalog, artifact_id, root)
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise AdaptiveError(
                f"artifact {artifact_id} failed live hash validation: {error}") \
                from error
        manifest = root / "partition-bundle.json"
        records.append({
            "artifact_id": artifact_id,
            "root": str(root.resolve(strict=True)),
            "partition_manifest_sha256": file_sha(manifest),
        })
    return records


def _executor_authority() -> dict[str, Any]:
    names = (
        "tools/run_kpack_postfix_adaptive_campaign_box.sh",
        "tools/kpack_postfix_adaptive_campaign.py",
        "tools/run_kpack_discovery_worker.py",
        "tools/plan_kpack_screen_retention.py",
    )
    files = []
    for name in names:
        path = ROOT / name
        _plain_file(path, f"executor file {name}")
        files.append({"path": name, "size": path.stat().st_size,
                      "sha256": file_sha(path)})
    return {
        "source_sha": _git("rev-parse", "HEAD"),
        "source_tree": _git("rev-parse", "HEAD^{tree}"),
        "files": files,
    }


def _validate_reused_campaign(source: Path
                              ) -> tuple[dict[str, Any], dict[str, Any],
                                         dict[str, Any], dict[str, Any],
                                         list[dict[str, Any]],
                                         list[dict[str, Any]]]:
    final = _json(source / "finalization.json", "full finalization")
    if (final.get("schema") != full_campaign.SCHEMA or
            final.get("old_bundle_overlay") is not False):
        raise AdaptiveError("reuse source is not one finalized full campaign")
    plan = partitions.read_plan(source / "build-plan.json")
    catalog = partitions.validate_catalog(source / "catalog.json")
    try:
        workload_index = workload_materializer.validate(
            source / "workload-plan.json", source / "workloads")
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise AdaptiveError(f"reused workload authority differs: {error}") from error
    master = worker_plan.load_json(source / "master.json", "master")
    worker_plan.validate_master(
        master, source / "catalog.json", source / "workload-plan.json")
    assignment = worker_plan.load_json(source / "assignment.json", "assignment")
    worker_plan.validate_assignment(
        assignment, master, file_sha(source / "master.json"))
    if assignment.get("worker_count") != WORKERS:
        raise AdaptiveError(f"reused campaign must have exactly {WORKERS} workers")

    expected_hashes = {
        "partition_plan_sha256": file_sha(source / "build-plan.json"),
        "catalog_sha256": file_sha(source / "catalog.json"),
        "workload_plan_sha256": file_sha(source / "workload-plan.json"),
        "workload_index_sha256": file_sha(source / "workloads/index.json"),
        "master_sha256": file_sha(source / "master.json"),
        "assignment_sha256": file_sha(source / "assignment.json"),
    }
    if (any(final.get(name) != value for name, value in expected_hashes.items()) or
            final.get("source_sha") != catalog.get("source_sha") or
            final.get("source_tree") != catalog.get("source_tree") or
            final.get("runtime_workers") != WORKERS or
            final.get("partition_artifacts") != len(catalog["partitions"]) or
            final.get("denominator") != full_campaign.derive_denominator(
                plan, worker_plan.load_json(
                    source / "workload-plan.json", "workload plan")) or
            final.get("workload_index") != workload_index):
        raise AdaptiveError("reused finalization hash/denominator differs")
    comparable = {
        key: final["denominator"].get(key) for key in full_campaign.EXPECTED
    }
    if comparable != full_campaign.EXPECTED:
        raise AdaptiveError("reused campaign is not the complete denominator")
    try:
        full_campaign._verify_live_sdk(catalog)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise AdaptiveError(f"live SDK differs from reused build: {error}") from error

    selections = [_selection(source, master, assignment, worker)
                  for worker in range(WORKERS)]
    artifacts = _artifact_roots(source, catalog, selections)
    return final, catalog, master, assignment, selections, artifacts


def prepare(source: Path, output: Path) -> dict[str, Any]:
    source = _plain_directory(source, "reused campaign")
    output = _plain_directory(output, "adaptive input root")
    _disjoint(source, output)
    final, catalog, master, assignment, selections, artifacts = \
        _validate_reused_campaign(source)

    for name in ("finalization.json", "build-plan.json", "catalog.json",
                 "workload-plan.json", "master.json", "assignment.json"):
        _copy_frozen(source / name, output / name, f"reused {name}")
    for worker in range(WORKERS):
        _copy_frozen(
            source / "selections" / f"worker-{worker}.json",
            output / "selections" / f"worker-{worker}.json",
            f"worker {worker} selection")
        _copy_frozen(
            source / "artifact-roots" / f"worker-{worker}.tsv",
            output / "artifact-roots" / f"worker-{worker}.tsv",
            f"worker {worker} artifact roots")

    epoch_path = output / "epoch.json"
    created = (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
               if not epoch_path.exists()
               else _json(epoch_path, "adaptive epoch").get("created_utc"))
    if not isinstance(created, str) or not created:
        raise AdaptiveError("adaptive epoch creation time differs")
    inputs = {
        name: file_sha(output / name) for name in (
            "finalization.json", "build-plan.json", "catalog.json",
            "workload-plan.json", "master.json", "assignment.json")
    }
    inputs["selections_sha256"] = digest([
        file_sha(output / "selections" / f"worker-{worker}.json")
        for worker in range(WORKERS)])
    inputs["artifact_roots_sha256"] = digest([
        file_sha(output / "artifact-roots" / f"worker-{worker}.tsv")
        for worker in range(WORKERS)])
    epoch = {
        "schema": EPOCH_SCHEMA,
        "created_utc": created,
        "reuse_policy": "HASH_VALIDATED_BUILD_ONLY_NO_HISTORICAL_TIMINGS",
        "reused_campaign": str(source),
        "build": {
            "source_sha": catalog["source_sha"],
            "source_tree": catalog["source_tree"],
            "actlize_sha": final["actlize_sha"],
            "sdk_authority_sha256": final["sdk_authority_sha256"],
            "partition_artifacts": len(artifacts),
            "artifact_records_sha256": digest(artifacts),
            "artifacts": artifacts,
        },
        "executor": _executor_authority(),
        "inputs": inputs,
        "denominator": final["denominator"],
        "workers": WORKERS,
        "measurement": {
            "screen_iterations": SCREEN_ITERATIONS,
            "screen_warmups": SCREEN_WARMUPS,
            "correctness_repeats": CORRECTNESS_REPEATS,
            "confirm_iterations": CONFIRM_ITERATIONS,
            "confirm_rounds": CONFIRM_ROUNDS,
            "confirm_warmups": CONFIRM_WARMUPS,
        },
        "historical_result_files_imported": 0,
    }
    if epoch["build"]["source_sha"] == "":
        raise AdaptiveError("build source is empty")
    _write_frozen_json(epoch_path, epoch)
    return epoch


def _device_rows(probes: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    homogeneity = set()
    for worker, path in enumerate(probes):
        document = _json(path, f"worker {worker} device probe")
        try:
            values, _sources = box_identity_schema.values_and_sources(document)
        except box_identity_schema.IdentityProbeError as error:
            raise AdaptiveError(f"worker {worker} device probe differs: {error}") \
                from error
        candidates = document.get("device_probe", {}).get("candidates")
        if (document.get("device_probe", {}).get("status") != "measured" or
                not isinstance(candidates, list) or len(candidates) != 1):
            raise AdaptiveError(f"worker {worker} lacks exactly one measured device")
        device = candidates[0]
        common = {
            "device_model": values["device_model"],
            "driver_version": values["driver_version"],
            "sdk_compiler_identity": values["sdk_compiler_identity"],
            "compute_capability": device["compute_capability"],
            "compute_units": device["compute_units"],
        }
        key = digest(common)
        homogeneity.add(key)
        rows.append({
            "worker_id": worker,
            "identity_sha256": file_sha(path),
            "homogeneity_key": key,
        })
    if len(rows) != WORKERS or len(homogeneity) != 1:
        raise AdaptiveError("eight runtime devices are absent or heterogeneous")
    return rows


def bind_devices(output: Path, probes: list[Path]) -> dict[str, Any]:
    output = _plain_directory(output, "adaptive input root")
    if len(probes) != WORKERS:
        raise AdaptiveError(f"device probe denominator must be {WORKERS}")
    document = {"schema": worker_plan.DEVICE_SCHEMA,
                "workers": _device_rows(probes)}
    worker_plan.validate_device_authority(document, WORKERS)
    _write_frozen_json(output / "device-homogeneity.json", document)
    return document


def _ids(path: Path, label: str) -> list[str]:
    _plain_file(path, label)
    try:
        values = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise AdaptiveError(f"cannot read {label}: {error}") from error
    if (not values or len(values) != len(set(values)) or
            any(len(value) != 64 or any(c not in SHA256 for c in value)
                for value in values)):
        raise AdaptiveError(f"{label} ID set differs")
    return values


def _directory_names(path: Path, label: str) -> set[str]:
    if path.is_symlink() or not path.is_dir():
        raise AdaptiveError(f"{label} is not a regular directory")
    return {entry.name for entry in path.iterdir()}


def _screen_digest(screen: Path, expected: list[str], worker: int) -> str:
    names = _directory_names(screen, f"worker {worker} screen directory")
    expected_names = {f"{item}.log" for item in expected}
    if names != expected_names:
        raise AdaptiveError(f"worker {worker} screen log union differs")
    records = []
    for item in expected:
        path = _plain_file(screen / f"{item}.log", "screen log")
        records.append([item, path.stat().st_size, file_sha(path)])
    return digest(records)


def seal_screen(inputs: Path, results: Path) -> dict[str, Any]:
    inputs = _plain_directory(inputs, "adaptive input root")
    results = _plain_directory(results, "adaptive result root")
    epoch = _json(inputs / "epoch.json", "adaptive epoch")
    if (epoch.get("schema") != EPOCH_SCHEMA or epoch.get("workers") != WORKERS or
            epoch.get("measurement") != {
                "screen_iterations": SCREEN_ITERATIONS,
                "screen_warmups": SCREEN_WARMUPS,
                "correctness_repeats": CORRECTNESS_REPEATS,
                "confirm_iterations": CONFIRM_ITERATIONS,
                "confirm_rounds": CONFIRM_ROUNDS,
                "confirm_warmups": CONFIRM_WARMUPS,
            } or epoch.get("historical_result_files_imported") != 0):
        raise AdaptiveError("adaptive epoch contract differs")
    device = _json(inputs / "device-homogeneity.json", "device homogeneity")
    identities = worker_plan.validate_device_authority(device, WORKERS)
    assignment = worker_plan.load_json(inputs / "assignment.json", "assignment")
    master = worker_plan.load_json(inputs / "master.json", "master")
    worker_plan.validate_assignment(
        assignment, master, file_sha(inputs / "master.json"))

    worker_rows = []
    all_ids: set[str] = set()
    expected_total = 0
    for worker in range(WORKERS):
        selection = _json(
            inputs / "selections" / f"worker-{worker}.json",
            f"worker {worker} selection")
        expected = [row["work_item_id"] for row in selection["work_items"]]
        assigned = assignment["workers"][worker]["work_item_ids"]
        if expected != assigned:
            raise AdaptiveError(f"worker {worker} selection/assignment differs")
        root = _plain_directory(results / f"worker-{worker}",
                                f"worker {worker} result root")
        completed = _ids(root / "screen-completed.ids",
                         f"worker {worker} screen completion")
        if completed != expected:
            raise AdaptiveError(f"worker {worker} screen completion differs")
        overlap = all_ids.intersection(completed)
        if overlap:
            raise AdaptiveError("screen completion overlaps workers")
        all_ids.update(completed)
        expected_total += len(expected)
        if _directory_names(root / "completion", "completion directory"):
            raise AdaptiveError("screen epoch contains confirmation completions")
        forbidden = ("worker-result.json", "worker-evidence.json", "completed.ids")
        if any((root / name).exists() or (root / name).is_symlink()
               for name in forbidden):
            raise AdaptiveError("screen epoch contains final worker authority")
        if any(path.is_dir() and path.name.startswith("confirm-r")
               for path in (root / "results").iterdir()):
            raise AdaptiveError("screen epoch contains confirmation timing files")
        execution = _json(root / "inputs/execution-authority.json",
                          f"worker {worker} execution authority")
        bindings = {
            "worker_id": worker,
            "worker_count": WORKERS,
            "bundle_sha256": file_sha(inputs / "catalog.json"),
            "workload_plan_sha256": file_sha(inputs / "workload-plan.json"),
            "master_sha256": file_sha(inputs / "master.json"),
            "assignment_sha256": file_sha(inputs / "assignment.json"),
            "selection_sha256": file_sha(
                inputs / "selections" / f"worker-{worker}.json"),
            "device_identity_sha256": identities[worker],
            "device_homogeneity_sha256": file_sha(
                inputs / "device-homogeneity.json"),
        }
        if any(execution.get(name) != value for name, value in bindings.items()):
            raise AdaptiveError(f"worker {worker} execution authority differs")
        measurement = execution.get("measurement", {})
        wanted = {
            "screen_iterations": SCREEN_ITERATIONS,
            "confirm_iterations": CONFIRM_ITERATIONS,
            "confirm_rounds": CONFIRM_ROUNDS,
            "correctness_repeats": CORRECTNESS_REPEATS,
            "grouped_warmups": {
                "screen": SCREEN_WARMUPS,
                "confirm": CONFIRM_WARMUPS,
            },
        }
        if any(measurement.get(name) != value for name, value in wanted.items()):
            raise AdaptiveError(f"worker {worker} measurement contract differs")
        worker_rows.append({
            "worker_id": worker,
            "work_items": len(expected),
            "work_item_ids_sha256": digest(expected),
            "screen_completed_sha256": file_sha(root / "screen-completed.ids"),
            "screen_logs_sha256": _screen_digest(
                root / "results/screen", expected, worker),
            "execution_authority_sha256": file_sha(
                root / "inputs/execution-authority.json"),
            "device_identity_sha256": identities[worker],
        })
    if expected_total != master["denominator"]["work_items"] or \
            len(all_ids) != expected_total:
        raise AdaptiveError("screen barrier denominator differs from master")
    barrier = {
        "schema": SCREEN_BARRIER_SCHEMA,
        "epoch_sha256": file_sha(inputs / "epoch.json"),
        "catalog_sha256": file_sha(inputs / "catalog.json"),
        "workload_plan_sha256": file_sha(inputs / "workload-plan.json"),
        "master_sha256": file_sha(inputs / "master.json"),
        "assignment_sha256": file_sha(inputs / "assignment.json"),
        "device_homogeneity_sha256": file_sha(
            inputs / "device-homogeneity.json"),
        "workers": worker_rows,
        "denominator": {"workers": WORKERS, "work_items": expected_total,
                        "screen_logs": expected_total},
        "measurement": {
            "screen_iterations": SCREEN_ITERATIONS,
            "screen_warmups": SCREEN_WARMUPS,
            "correctness_repeats": CORRECTNESS_REPEATS,
        },
        "historical_result_files_imported": 0,
    }
    _write_frozen_json(results / "screen-barrier.json", barrier)
    stage = {
        "schema": STAGE_SCHEMA,
        "state": PENDING_STAGE,
        "screen_barrier_sha256": file_sha(results / "screen-barrier.json"),
        "next": {
            "planner": "plan_kpack_screen_retention.py",
            "policy": "GLOBAL_NO_TOP_N_CONSERVATIVE_ENVELOPE",
            "confirm_iterations": CONFIRM_ITERATIONS,
            "confirm_rounds": CONFIRM_ROUNDS,
            "confirm_warmups": CONFIRM_WARMUPS,
            "fallback_to_full_confirm": False,
        },
    }
    _write_frozen_json(results / "stage.json", stage)
    return barrier


def self_test() -> None:
    if (WORKERS != 8 or SCREEN_ITERATIONS != 2 or SCREEN_WARMUPS != 1 or
            CORRECTNESS_REPEATS != 1 or CONFIRM_ITERATIONS != 11 or
            CONFIRM_ROUNDS != 3 or CONFIRM_WARMUPS != 3):
        raise AssertionError("adaptive measurement constants differ")
    with tempfile.TemporaryDirectory(prefix="kpack-adaptive-self-test-") as name:
        root = Path(name)
        first, second = root / "first", root / "second"
        first.mkdir()
        second.mkdir()
        _disjoint(first, second)
        try:
            _disjoint(first, first)
        except AdaptiveError:
            pass
        else:
            raise AssertionError("path-alias negative stayed green")
        frozen = root / "frozen"
        _write_frozen_bytes(frozen, b"one\n")
        _write_frozen_bytes(frozen, b"one\n")
        try:
            _write_frozen_bytes(frozen, b"two\n")
        except AdaptiveError:
            pass
        else:
            raise AssertionError("stale frozen output negative stayed green")
    print("[kpack-postfix-adaptive:self-test] PASS build/executor split "
          "fresh timing epoch, eight-worker screen barrier, frozen resume, "
          "2-sample screen and 3x11 shortlisted-confirm contract; "
          "path-alias+stale-output negatives=RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--reuse-campaign", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    devices = commands.add_parser("bind-devices")
    devices.add_argument("--output", type=Path, required=True)
    devices.add_argument("--probe", type=Path, action="append", required=True)
    seal = commands.add_parser("seal-screen")
    seal.add_argument("--inputs", type=Path, required=True)
    seal.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "prepare":
            value = prepare(args.reuse_campaign, args.output)
            print("KPACK_ADAPTIVE_PREPARED "
                  f"build_source={value['build']['source_sha']} "
                  f"executor_source={value['executor']['source_sha']} "
                  f"artifacts={value['build']['partition_artifacts']} "
                  f"work_items={value['denominator']['work_items']} "
                  f"output={args.output}")
        elif args.command == "bind-devices":
            value = bind_devices(args.output, args.probe)
            print(f"KPACK_ADAPTIVE_DEVICES workers={len(value['workers'])} "
                  "homogeneous=1")
        else:
            value = seal_screen(args.inputs, args.results)
            print("KPACK_ADAPTIVE_SCREEN_BARRIER "
                  f"workers={value['denominator']['workers']} "
                  f"work_items={value['denominator']['work_items']} "
                  f"state={PENDING_STAGE} output={args.results}")
        return 0
    except (AdaptiveError, OSError, subprocess.SubprocessError,
            KeyError, TypeError, ValueError) as error:
        print(f"[kpack-postfix-adaptive] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
