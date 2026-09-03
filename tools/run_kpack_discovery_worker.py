#!/usr/bin/env python3
"""Execute one exactly-assigned canonical K-pack discovery worker.

This runner is deliberately prebuilt-only.  It never invokes a compiler or a
bundle builder.  Its execution atom is the work-master's indivisible
``binary shard x canonical workload`` pair, and every compiled parent in the
shard is measured in both screen and confirmation.  There is no top-N or
timing-based retention step.

Typical distributed workflow::

  python3 tools/run_kpack_discovery_worker.py select \
      --bundle bundle.json --plan plan.json --master master.json \
      --assignment assignment.json --worker-id 0 --output worker-0.json

  CUDA_VISIBLE_DEVICES=0 python3 tools/run_kpack_discovery_worker.py probe-device \
      --bundle bundle.json --output worker-0-device.json

  CUDA_VISIBLE_DEVICES=0 python3 tools/run_kpack_discovery_worker.py run \
      --bundle bundle.json --plan plan.json --master master.json \
      --assignment assignment.json --selection worker-0.json --worker-id 0 \
      --device-identity worker-0-device.json \
      --device-homogeneity devices.json --output worker-0-results \
      --phase all

For a distributed catalog, replace the monolithic bundle path with the
catalog and pass only this worker's assigned roots, once per artifact::

  --artifact-root 'kpack-discovery/.../scalefirst/p00-of-32=/fetch/sf-p00' \
  --artifact-root 'kpack-discovery/.../fully-quantized/p00-of-32=/fetch/fq-p00'

``probe-device`` executes both component bundles' immutable runtime probes and
requires byte-identical identity documents.  ``run`` repeats that live check,
validates every authority and native shard path, then writes successful logs
and completion records with atomic renames.  An interrupted run is resumed by
passing ``--resume``; stale successful files are rejected rather than
overwritten.  Evidence snapshots the assigned candidate manifests and exact
router-row files, while recording (not copying) binary/receipt hashes and the
executed path strings.  Aggregation therefore does not require the partition
artifact roots to remain mounted after execution.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable, NoReturn


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import compose_kpack_discovery_bundles as composite  # noqa: E402
import kpack_discovery_worker_plan as worker_plan  # noqa: E402
import materialize_kpack_discovery_workloads as workload_authority  # noqa: E402
import probe_box_identity  # noqa: E402
import analyze_scalefirst_kpack_discovery as sf_analyzer  # noqa: E402
import fully_quantized_kpack_bundle_index as fq_index  # noqa: E402
import fq_dense_structural_proof as fq_structural  # noqa: E402
import kpack_discovery_build_partitions as partitions  # noqa: E402


EXECUTION_SCHEMA = "quactlize.kpack-discovery-worker-execution.v2"
EVIDENCE_SCHEMA = "quactlize.kpack-discovery-worker-evidence.v2"
COMPLETION_SCHEMA = "quactlize.kpack-discovery-atom-completion.v1"
DEVICE_KERNEL = "DEVICE_KERNEL"
NO_DEVICE_KERNEL_STRUCTURAL = "NO_DEVICE_KERNEL_STRUCTURAL"
SCHEDULE_SEED_SCHEMA = "quactlize.kpack-discovery-schedule-seed.v1"
SAFE_TOKEN = re.compile(r"[^\s\0]+\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
VISIBLE_DEVICE = re.compile(r"[0-9]+\Z")


class ExecutionError(RuntimeError):
    """The worker cannot make a trustworthy progress claim."""


def schedule_seed_contract() -> dict[str, str]:
    return {
        "schema": SCHEDULE_SEED_SCHEMA,
        "message_encoding": "ASCII(work_item_id:phase:round_index)",
        "hash": "SHA-256",
        "projection": "digest_hex[0:16]_as_unsigned_u64",
        "screen_coordinate": "phase=screen,round_index=0",
        "confirm_coordinate": "phase=confirm,round_index=1..3",
    }


def schedule_seed(item_id: str, phase: str, round_index: int) -> int:
    if not SHA256.fullmatch(item_id):
        raise ExecutionError("schedule seed work-item ID is malformed")
    if ((phase, round_index) != ("screen", 0) and
            not (phase == "confirm" and 1 <= round_index <= 3)):
        raise ExecutionError("schedule seed coordinate is malformed")
    message = f"{item_id}:{phase}:{round_index}".encode("ascii")
    return int(hashlib.sha256(message).hexdigest()[:16], 16)


def schedule_seed_hex(value: int) -> str:
    return f"0x{value:016x}"


def fail(message: str) -> NoReturn:
    raise SystemExit(f"kpack discovery worker: {message}")


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ExecutionError(f"value is not canonical JSON: {error}") from error


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ExecutionError(f"cannot hash {path}: {error}") from error


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExecutionError(f"{label} must be a JSON object")
    return value


def atomic_bytes(path: Path, payload: bytes, *, frozen: bool = False,
                 allow_empty: bool = False) -> None:
    if not payload and not allow_empty:
        raise ExecutionError(f"refusing to publish empty file {path}")
    if path.is_symlink():
        raise ExecutionError(f"output may not be a symlink: {path}")
    if path.exists():
        if frozen and path.is_file() and path.read_bytes() == payload:
            return
        raise ExecutionError(f"refusing to replace existing output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise ExecutionError(f"atomic staging path already exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        raise ExecutionError(f"cannot publish {path}: {error}") from error


def atomic_json(path: Path, value: dict[str, Any], *, frozen: bool = False) -> None:
    atomic_bytes(
        path, json.dumps(value, indent=2, sort_keys=True,
                         allow_nan=False).encode("utf-8") + b"\n",
        frozen=frozen)


def _visible_device(environ: dict[str, str]) -> str:
    value = environ.get("CUDA_VISIBLE_DEVICES", "")
    if not VISIBLE_DEVICE.fullmatch(value):
        raise ExecutionError(
            "CUDA_VISIBLE_DEVICES must name exactly one numeric ordinal")
    return value


def _regular_file(path: Path, label: str, *, executable: bool = False) -> Path:
    if path.is_symlink():
        raise ExecutionError(f"{label} may not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ExecutionError(f"cannot resolve {label} {path}: {error}") from error
    if not resolved.is_file():
        raise ExecutionError(f"{label} is not a regular file: {path}")
    if executable and not os.access(resolved, os.X_OK):
        raise ExecutionError(f"{label} is not executable: {path}")
    return resolved


def _relative(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ExecutionError(f"{label} is not a normalized relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
            part in ("", ".", "..") for part in path.parts):
        raise ExecutionError(f"{label} is not a normalized relative path")
    return path


def _under(root: Path, relative: Any, label: str,
           *, executable: bool = False) -> Path:
    rel = _relative(relative, label)
    path = _regular_file(root.joinpath(*rel.parts), label,
                         executable=executable)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ExecutionError(f"{label} escapes its authority root") from error
    return path


def _selection(
        bundle: Path, plan: Path, master_path: Path, assignment_path: Path,
        selection_path: Path | None, worker_id: int
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    # validate_master calls the composite validator through load_bundle_authority.
    master = load_json(master_path, "work master")
    try:
        worker_plan.validate_master(master, bundle, plan)
        assignment = load_json(assignment_path, "worker assignment")
        worker_plan.validate_assignment(
            assignment, master, worker_plan.file_sha(master_path))
        expected = worker_plan.make_worker_selection(
            master, assignment, worker_id,
            master_sha256=worker_plan.file_sha(master_path),
            assignment_sha256=worker_plan.file_sha(assignment_path))
    except worker_plan.PlanError as error:
        raise ExecutionError(str(error)) from error
    if selection_path is not None:
        selected = load_json(selection_path, "worker selection")
        if selected != expected:
            raise ExecutionError(
                "worker selection differs from live master/assignment authority")
    return master, assignment, expected


def _bundle_document(path: Path) -> dict[str, Any]:
    raw = load_json(path, "bundle")
    try:
        if raw.get("schema") == partitions.CATALOG_SCHEMA:
            return worker_plan.load_bundle_authority(path)
        return composite.validate_composite(path)
    except (OSError, ValueError, worker_plan.PlanError) as error:
        raise ExecutionError(f"bundle/catalog authority differs: {error}") from error


def select_worker(args: argparse.Namespace) -> int:
    _master, _assignment, selection = _selection(
        args.bundle, args.plan, args.master, args.assignment, None,
        args.worker_id)
    atomic_json(args.output, selection, frozen=True)
    print("KPACK_DISCOVERY_WORKER_SELECTION PASS "
          f"worker={args.worker_id} items={len(selection['work_items'])} "
          f"partitions={len(selection['partition_ids'])} "
          f"artifacts={len(selection['artifact_ids'])} "
          f"output={args.output}")
    return 0


@dataclass(frozen=True)
class ArtifactContext:
    artifact_id: str
    root: Path
    partition: dict[str, Any]


def _artifact_root_values(values: list[str] | None) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for index, value in enumerate(values or []):
        if not isinstance(value, str) or "=" not in value:
            raise ExecutionError(
                f"artifact root {index} must be ARTIFACT_ID=/absolute/path")
        artifact_id, raw_path = value.split("=", 1)
        if (not artifact_id or not SAFE_TOKEN.fullmatch(artifact_id) or
                not raw_path or not Path(raw_path).is_absolute()):
            raise ExecutionError(
                f"artifact root {index} must be ARTIFACT_ID=/absolute/path")
        if artifact_id in result:
            raise ExecutionError(f"duplicate artifact root {artifact_id}")
        result[artifact_id] = Path(raw_path)
    return result


def _catalog_artifacts(document: dict[str, Any], selection: dict[str, Any],
                       values: list[str] | None
                       ) -> dict[str, ArtifactContext]:
    expected = selection.get("artifact_ids")
    if not isinstance(expected, list) or any(
            not isinstance(value, str) for value in expected):
        raise ExecutionError("catalog worker selection artifact IDs are malformed")
    supplied = _artifact_root_values(values)
    if set(supplied) != set(expected):
        raise ExecutionError(
            "worker artifact-root set differs from assigned partition artifacts")
    contexts: dict[str, ArtifactContext] = {}
    seen_roots: set[Path] = set()
    for artifact_id in expected:
        root_arg = supplied[artifact_id]
        if root_arg.is_symlink():
            raise ExecutionError(f"artifact root may not be a symlink: {root_arg}")
        try:
            root = root_arg.resolve(strict=True)
        except OSError as error:
            raise ExecutionError(
                f"assigned artifact root is missing: {root_arg}: {error}") from error
        if not root.is_dir() or root in seen_roots:
            raise ExecutionError("assigned artifact roots are not distinct directories")
        seen_roots.add(root)
        try:
            partition = partitions.verify_catalog_artifact(
                document, artifact_id, root)
        except (OSError, ValueError, partitions.PartitionError) as error:
            raise ExecutionError(
                f"assigned artifact {artifact_id} failed live verification: "
                f"{error}") from error
        contexts[artifact_id] = ArtifactContext(
            artifact_id, root, partition)
    selected_shards = set(selection.get("shard_keys", []))
    catalog_shards = {
        row["shard_key"] for row in document["shards"]
        if row["artifact_id"] in contexts}
    if selected_shards != catalog_shards:
        raise ExecutionError(
            "assigned artifact roots do not contain the exact selected shard union")
    return contexts


def _probe_binaries(bundle_path: Path, document: dict[str, Any],
                    artifacts: dict[str, ArtifactContext] | None = None
                    ) -> list[Path]:
    if document.get("schema") == partitions.CATALOG_SCHEMA:
        if not artifacts:
            raise ExecutionError("distributed catalog requires assigned artifacts")
        paths = []
        for artifact_id, context in sorted(artifacts.items()):
            record = context.partition["runtime_identity_probe"]["binary"]
            path = _under(context.root, record["path"],
                          f"{artifact_id} runtime probe", executable=True)
            if (path.stat().st_size != record["size"] or
                    file_sha(path) != record["sha256"]):
                raise ExecutionError(f"{artifact_id} runtime probe bytes differ")
            paths.append(path)
        return paths
    root = bundle_path.parent.resolve(strict=True)
    paths: list[Path] = []
    for route in sorted(worker_plan.ROUTES):
        component = document.get("component_bundles", {}).get(route)
        if not isinstance(component, dict):
            raise ExecutionError(f"composite has no {route} component")
        probe = component.get("runtime_probe")
        if not isinstance(probe, dict) or not isinstance(
                probe.get("binary"), dict):
            raise ExecutionError(f"{route} runtime probe record is malformed")
        record = probe["binary"]
        path = _under(root, record.get("path"), f"{route} runtime probe",
                      executable=True)
        if file_sha(path) != record.get("sha256"):
            raise ExecutionError(f"{route} runtime probe hash differs")
        paths.append(path)
    return paths


def _live_device_bytes(bundle_path: Path,
                       document: dict[str, Any],
                       artifacts: dict[str, ArtifactContext] | None = None) -> bytes:
    _visible_device(dict(os.environ))
    observed: list[bytes] = []
    for probe in _probe_binaries(bundle_path, document, artifacts):
        with tempfile.TemporaryDirectory(prefix="kpack-worker-device-") as name:
            output = Path(name) / "identity.json"
            try:
                subprocess.run(
                    [sys.executable, str(TOOLS / "probe_box_identity.py"),
                     "resolve", "--output", str(output),
                     "--runtime-probe-binary", str(probe)],
                    check=True, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, env=dict(os.environ))
                payload = output.read_bytes()
                parsed = json.loads(payload)
                probe_box_identity._validate_document(parsed)
            except (OSError, json.JSONDecodeError,
                    probe_box_identity.ProbeError,
                    subprocess.CalledProcessError) as error:
                stderr = ""
                if isinstance(error, subprocess.CalledProcessError):
                    stderr_value = error.stderr
                    if isinstance(stderr_value, bytes):
                        stderr = stderr_value.decode("utf-8", "replace")
                    elif stderr_value is not None:
                        stderr = str(stderr_value)
                raise ExecutionError(
                    f"prebuilt device identity probe failed: {error}; "
                    f"stderr={stderr[-1200:]}") from error
            observed.append(payload)
    expected = (len(artifacts) if artifacts is not None else 2)
    if len(observed) != expected or len(set(observed)) != 1:
        raise ExecutionError(
            "assigned runtime probes disagree on device identity")
    return observed[0]


def probe_device(args: argparse.Namespace) -> int:
    document = _bundle_document(args.bundle)
    artifacts = None
    if document.get("schema") == partitions.CATALOG_SCHEMA:
        required = (args.plan, args.master, args.assignment,
                    args.selection, args.worker_id)
        if any(value is None for value in required):
            raise ExecutionError(
                "catalog probe-device requires plan/master/assignment/selection/worker-id")
        _master, _assignment, selection = _selection(
            args.bundle, args.plan, args.master, args.assignment,
            args.selection, args.worker_id)
        artifacts = _catalog_artifacts(
            document, selection, args.artifact_root)
    elif args.artifact_root:
        raise ExecutionError("monolithic bundle may not use --artifact-root")
    payload = _live_device_bytes(args.bundle, document, artifacts)
    atomic_bytes(args.output, payload, frozen=True)
    print("KPACK_DISCOVERY_DEVICE PASS "
          f"visible_ordinal={_visible_device(dict(os.environ))} "
          f"sha256={hashlib.sha256(payload).hexdigest()} output={args.output}")
    return 0


def _native_rows(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = document.get("shards")
    if isinstance(rows, dict):
        return rows
    if not isinstance(rows, list):
        raise ExecutionError("native component shard rows are malformed")
    result: dict[str, dict[str, Any]] = {}
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ExecutionError(f"native shard {ordinal} is malformed")
        key = row.get("shard_id", row.get("shard_key"))
        if not isinstance(key, str) or not key or key in result:
            raise ExecutionError("native component shard identity differs")
        result[key] = row
    return result


@dataclass(frozen=True)
class ResolvedShard:
    key: str
    route: str
    operator: str
    qtype: int
    parent_count: int
    binary: Path
    manifest: Path
    receipt: Path
    symbols: tuple[str, ...] = ()
    artifact_id: str | None = None
    payload_kind: str = DEVICE_KERNEL
    structural_proof: Path | None = None


def _payload_kind(row: dict[str, Any], shard_key: str) -> str:
    """Normalize the legacy device payload and the proved structural case."""
    kind = row.get("payload_kind", DEVICE_KERNEL)
    if kind not in (DEVICE_KERNEL, NO_DEVICE_KERNEL_STRUCTURAL):
        raise ExecutionError(f"{shard_key}: payload kind differs")
    if kind == NO_DEVICE_KERNEL_STRUCTURAL and not (
            row.get("route") == "fully-quantized" and
            row.get("operator") == "dense"):
        raise ExecutionError(
            f"{shard_key}: structural no-kernel payload is outside FQ dense")
    return kind


def _payload_file_fields(kind: str) -> tuple[str, ...]:
    fields = ("manifest", "binary", "binary_receipt")
    return (fields + ("structural_proof",)
            if kind == NO_DEVICE_KERNEL_STRUCTURAL else fields)


def _validate_structural_payload(
        row: dict[str, Any], files: dict[str, Path],
        symbols: tuple[str, ...], shard_key: str) -> None:
    if _payload_kind(row, shard_key) != NO_DEVICE_KERNEL_STRUCTURAL:
        if "structural_proof" in files:
            raise ExecutionError(
                f"{shard_key}: device payload has a structural proof")
        return
    try:
        receipt = load_json(files["binary_receipt"],
                            f"{shard_key} structural receipt")
        proof = load_json(files["structural_proof"],
                          f"{shard_key} structural proof")
        native = {
            "shard_key": row.get("native_shard_key", shard_key),
            "qtype": row["qtype"], "operator": row["operator"],
            "route": row["route"],
            "parent_begin": row["parent_begin"],
            "parent_end": row["parent_end"],
            "parent_count": row["parent_count"],
            "authority_count": row["authority_count"],
            "parent_ids": row["parent_ids"],
        }
        fq_index.validate_receipt(
            receipt, native, file_sha(files["manifest"]),
            file_sha(files["binary"]))
        fq_structural.validate_structural_proof(
            proof, native, file_sha(files["manifest"]),
            file_sha(files["binary"]), receipt)
        proof_symbols = tuple(record.get("symbol") for record in proof["rows"])
        if proof_symbols != symbols:
            raise ExecutionError(
                f"{shard_key}: structural proof/manifest symbols differ")
    except (KeyError, TypeError, ValueError) as error:
        raise ExecutionError(
            f"{shard_key}: structural proof validation failed: {error}") from error


def resolve_native_shard(bundle_path: Path, composite_doc: dict[str, Any],
                         shard_key: str) -> ResolvedShard:
    root = bundle_path.parent.resolve(strict=True)
    matches = [row for row in composite_doc.get("shards", [])
               if isinstance(row, dict) and row.get("shard_key") == shard_key]
    if len(matches) != 1:
        raise ExecutionError(f"composite shard {shard_key!r} is not unique")
    row = matches[0]
    route = row.get("route")
    component = composite_doc.get("component_bundles", {}).get(route)
    if not isinstance(component, dict):
        raise ExecutionError(f"{shard_key}: component authority is absent")
    component_root_rel = _relative(component.get("root"),
                                   f"{route} component root")
    component_root = root.joinpath(*component_root_rel.parts).resolve(strict=True)
    try:
        component_root.relative_to(root)
    except ValueError as error:
        raise ExecutionError(f"{shard_key}: component root escapes bundle") from error
    native_bundle = _under(root, component.get("bundle"),
                           f"{route} native bundle")
    if native_bundle.parent != component_root:
        raise ExecutionError(f"{shard_key}: native bundle/root binding differs")
    native_doc = load_json(native_bundle, f"{route} native bundle")
    native_key = row.get("native_shard_key")
    native = _native_rows(native_doc).get(native_key)
    if not isinstance(native, dict):
        raise ExecutionError(f"{shard_key}: native shard row is absent")
    for field in ("qtype", "operator", "parent_begin", "parent_end",
                  "parent_ids"):
        if native.get(field) != row.get(field):
            raise ExecutionError(f"{shard_key}: native {field} differs")
    kind = _payload_kind(row, shard_key)
    files: dict[str, Path] = {}
    for field in _payload_file_fields(kind):
        record = row.get("files", {}).get(field)
        if not isinstance(record, dict):
            raise ExecutionError(f"{shard_key}: composite {field} is malformed")
        native_path = _under(
            component_root, native.get(field), f"{shard_key} native {field}",
            executable=field == "binary")
        composite_path = _under(
            root, record.get("path"), f"{shard_key} composite {field}",
            executable=field == "binary")
        if native_path != composite_path or file_sha(native_path) != record.get("sha256"):
            raise ExecutionError(f"{shard_key}: {field} path/hash binding differs")
        files[field] = native_path
    manifest_doc = load_json(files["manifest"], f"{shard_key} manifest")
    manifest_field = ("compiled_parents" if route == "scalefirst" else
                      "dense_tc_parents" if row["operator"] == "dense" else
                      "grouped_parents")
    parents = manifest_doc.get(manifest_field)
    if not isinstance(parents, list):
        raise ExecutionError(f"{shard_key}: manifest parent rows are malformed")
    symbols = tuple(parent.get("symbol") if isinstance(parent, dict) else None
                    for parent in parents)
    if (len(symbols) != len(row["parent_ids"]) or
            len(set(symbols)) != len(symbols) or
            any(not isinstance(symbol, str) or not SAFE_TOKEN.fullmatch(symbol)
                for symbol in symbols)):
        raise ExecutionError(f"{shard_key}: manifest symbol census differs")
    if (route == "scalefirst" and
            native.get("parent_symbols") != list(symbols)):
        raise ExecutionError(f"{shard_key}: native parent symbols differ")
    _validate_structural_payload(row, files, symbols, shard_key)
    return ResolvedShard(
        key=shard_key, route=str(route), operator=str(row["operator"]),
        qtype=int(row["qtype"]), parent_count=len(row["parent_ids"]),
        binary=files["binary"], manifest=files["manifest"],
        receipt=files["binary_receipt"], symbols=symbols,
        artifact_id=None, payload_kind=kind,
        structural_proof=files.get("structural_proof"))


def resolve_catalog_shard(catalog: dict[str, Any],
                          artifacts: dict[str, ArtifactContext],
                          shard_key: str) -> ResolvedShard:
    matches = [row for row in catalog.get("shards", [])
               if isinstance(row, dict) and row.get("shard_key") == shard_key]
    if len(matches) != 1:
        raise ExecutionError(f"catalog shard {shard_key!r} is not unique")
    row = matches[0]
    artifact_id = row.get("artifact_id")
    context = artifacts.get(artifact_id)
    if context is None:
        raise ExecutionError(f"{shard_key}: assigned artifact is unavailable")
    partition_rows = [candidate for candidate in context.partition["shards"]
                      if candidate.get("shard_key") == shard_key]
    if len(partition_rows) != 1:
        raise ExecutionError(f"{shard_key}: partition shard is not unique")
    try:
        expected = partitions._catalog_shard(context.partition,
                                             partition_rows[0])
    except (KeyError, ValueError, partitions.PartitionError) as error:
        raise ExecutionError(
            f"{shard_key}: partition shard metadata is malformed: {error}") from error
    if expected != row:
        raise ExecutionError(f"{shard_key}: catalog/partition metadata differs")
    kind = _payload_kind(row, shard_key)
    files: dict[str, Path] = {}
    for field in _payload_file_fields(kind):
        record = row["files"][field]
        path = _under(context.root, record["path"],
                      f"{shard_key} {field}", executable=field == "binary")
        if (path.stat().st_size != record["size"] or
                file_sha(path) != record["sha256"]):
            raise ExecutionError(f"{shard_key}: {field} bytes differ")
        files[field] = path
    manifest_doc = load_json(files["manifest"], f"{shard_key} manifest")
    route = row["route"]
    manifest_field = ("compiled_parents" if route == "scalefirst" else
                      "dense_tc_parents" if row["operator"] == "dense" else
                      "grouped_parents")
    parents = manifest_doc.get(manifest_field)
    if not isinstance(parents, list):
        raise ExecutionError(f"{shard_key}: manifest parent rows are malformed")
    symbols = tuple(parent.get("symbol") if isinstance(parent, dict) else None
                    for parent in parents)
    if (len(symbols) != len(row["parent_ids"]) or
            len(set(symbols)) != len(symbols) or
            any(not isinstance(symbol, str) or not SAFE_TOKEN.fullmatch(symbol)
                for symbol in symbols)):
        raise ExecutionError(f"{shard_key}: manifest symbol census differs")
    _validate_structural_payload(row, files, symbols, shard_key)
    return ResolvedShard(
        key=shard_key, route=route, operator=row["operator"],
        qtype=row["qtype"], parent_count=len(row["parent_ids"]),
        binary=files["binary"], manifest=files["manifest"],
        receipt=files["binary_receipt"], symbols=symbols,
        artifact_id=artifact_id, payload_kind=kind,
        structural_proof=files.get("structural_proof"))


@dataclass(frozen=True)
class Workload:
    key: str
    operator: str
    values: dict[str, str]
    rows_path: Path | None = None
    rows_fnv64: str | None = None


def _positive(row: dict[str, str], field: str) -> int:
    value = row.get(field, "")
    if not value.isdecimal() or int(value) <= 0:
        raise ExecutionError(f"workload {row.get('workload_key')} has invalid {field}")
    return int(value)


def _rows_fnv64(path: Path, experts: int) -> str:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
        rows = [int(line) for line in lines]
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ExecutionError(f"cannot parse exact grouped rows {path}: {error}") from error
    if len(rows) != experts or any(value < 0 or value >= (1 << 31) for value in rows):
        raise ExecutionError("exact grouped rows differ from expert denominator")
    value = 14695981039346656037
    for row in rows:
        for byte in row.to_bytes(4, "little", signed=False):
            value ^= byte
            value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"0x{value:016x}"


def read_workloads(root: Path, qtype: int, operator: str) -> dict[str, Workload]:
    path = root / f"q{qtype}.{operator}.tsv"
    expected = (workload_authority.DENSE_COLUMNS if operator == "dense" else
                workload_authority.GROUPED_COLUMNS)
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if tuple(reader.fieldnames or ()) != expected:
                raise ExecutionError(f"{path.name} columns differ")
            rows = list(reader)
    except OSError as error:
        raise ExecutionError(f"cannot read workloads {path}: {error}") from error
    try:
        index = json.loads((root / "index.json").read_text(encoding="utf-8"))
        indexed_files = {record["path"]: record["sha256"]
                         for record in index["files"]}
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ExecutionError(f"cannot read workload file index: {error}") from error
    result: dict[str, Workload] = {}
    for row in rows:
        key = row.get("workload_key", "")
        if not SAFE_TOKEN.fullmatch(key) or key in result:
            raise ExecutionError(f"{path.name} has an unsafe/duplicate workload key")
        rows_path: Path | None = None
        rows_hash: str | None = None
        if operator == "dense":
            for field in ("m", "n", "k"):
                _positive(row, field)
        else:
            experts = _positive(row, "experts")
            for field in ("n", "k", "total_rows", "max_rows"):
                _positive(row, field)
            if not SAFE_TOKEN.fullmatch(row.get("profile", "")):
                raise ExecutionError(f"{key}: router profile is not one token")
            if row.get("rows_file") == "-":
                tokens, topk = _positive(row, "tokens"), _positive(row, "topk")
                if tokens * topk != int(row["total_rows"]):
                    raise ExecutionError(f"{key}: token/top-k total differs")
            else:
                rel = _relative(row.get("rows_file"), f"{key}.rows_file")
                rows_path = _under(root, rel.as_posix(), f"{key} exact rows")
                declared = indexed_files.get(rel.as_posix())
                if (not isinstance(declared, str) or
                        not SHA256.fullmatch(declared) or
                        file_sha(rows_path) != declared):
                    raise ExecutionError(f"{key}: exact rows file hash differs")
                rows_hash = _rows_fnv64(rows_path, experts)
        result[key] = Workload(key, operator, dict(row), rows_path, rows_hash)
    return result


def command_for(shard: ResolvedShard, workload: Workload, *, iterations: int,
                correctness_repeats: int, warmups: int,
                schedule_seed: int | None = None,
                symbol_file: Path | None = None) -> list[str]:
    row = workload.values
    command = [str(shard.binary)]
    if shard.operator == "dense":
        command += [f"--shape={row['m']}x{row['n']}x{row['k']}",
                    f"--iterations={iterations}",
                    f"--correctness-repeats={correctness_repeats}"]
        if shard.route == "scalefirst":
            command += ["--algorithm=full-output"]
        else:
            command += ["--bc-mode=skip"]
        if schedule_seed is not None:
            command += [f"--schedule-seed={schedule_seed}"]
        if symbol_file is not None:
            if shard.route != "scalefirst":
                raise ExecutionError("FullyQuantized may not use SF retention")
            command += [f"--symbol-file={symbol_file}"]
        return command
    command += [f"--experts={row['experts']}", f"--n={row['n']}",
                f"--k={row['k']}", f"--workload-key={workload.key}",
                f"--router-profile={row['profile']}",
                f"--iterations={iterations}", f"--warmups={warmups}",
                f"--correctness-repeats={correctness_repeats}"]
    if schedule_seed is not None:
        command += [f"--schedule-seed={schedule_seed}"]
    if symbol_file is not None:
        if shard.route != "scalefirst":
            raise ExecutionError("FullyQuantized may not use SF retention")
        command += [f"--symbol-file={symbol_file}"]
    if workload.rows_path is None:
        command += [f"--tokens={row['tokens']}", f"--topk={row['topk']}"]
    else:
        command += [f"--rows-file={workload.rows_path}"]
    return command


def _kv(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            if key in result:
                raise ExecutionError(f"duplicate log field {key}")
            result[key] = value
    return result


def _one_line(text: str, prefix: str) -> dict[str, str]:
    rows = [_kv(line) for line in text.splitlines() if line.startswith(prefix)]
    if len(rows) != 1:
        raise ExecutionError(f"log expected one {prefix.strip()} row, got {len(rows)}")
    return rows[0]


def _header_u64(row: dict[str, str], field: str, label: str) -> int:
    value = row.get(field)
    try:
        parsed = int(value, 0) if value is not None else -1
    except ValueError as error:
        raise ExecutionError(f"{label} {field} is malformed") from error
    if parsed < 0 or parsed >= 1 << 64:
        raise ExecutionError(f"{label} {field} is malformed")
    return parsed


def validate_log(text: str, shard: ResolvedShard, workload: Workload,
                 selected_symbols: tuple[str, ...] | None = None,
                 *, expected_schedule_seed: int | None = None,
                 expected_grouped_warmups: int | None = None,
                 expected_iterations: int | None = None,
                 expected_correctness_repeats: int | None = None) -> None:
    row = workload.values
    selected = shard.symbols if selected_symbols is None else selected_symbols
    selected_count = len(selected) if selected else shard.parent_count
    if shard.operator == "dense":
        shape = f"{row['m']}x{row['n']}x{row['k']}"
        if shard.route == "scalefirst":
            header = _one_line(text, "SF_SHARD ")
            done = _one_line(text, "SF_COMPLETE ")
            wanted_header = {"qtype": str(shard.qtype),
                             "typed_rows": str(shard.parent_count),
                             "selected_rows": str(selected_count)}
            wanted_done = {"status": "COMPLETE", "shape": shape,
                           "typed_rows": str(selected_count)}
        else:
            header = _one_line(text, "FQ_SHARD ")
            done = _one_line(text, "FQ_SHAPE_DONE ")
            wanted_header = {"q": str(shard.qtype),
                             "typed_rows": str(shard.parent_count),
                             "selected_rows": str(shard.parent_count)}
            wanted_done = {"q": str(shard.qtype), "shape": shape,
                           "typed_rows": str(shard.parent_count),
                           "selected_rows": str(shard.parent_count),
                           "status": "PASS"}
        if any(header.get(key) != value for key, value in wanted_header.items()):
            raise ExecutionError("dense shard header authority differs")
        if "warmups" in header:
            raise ExecutionError("dense shard header may not claim grouped warmups")
        for field, expected in (("iterations", expected_iterations),
                                ("correctness_repeats",
                                 expected_correctness_repeats)):
            if expected is not None and header.get(field) != str(expected):
                raise ExecutionError(f"dense shard {field} differs")
        if (expected_schedule_seed is not None and
                _header_u64(header, "schedule_seed", "dense shard") !=
                expected_schedule_seed):
            raise ExecutionError("dense shard schedule seed differs")
        if any(done.get(key) != value for key, value in wanted_done.items()):
            raise ExecutionError("dense completion authority differs")
        if selected:
            if shard.route == "scalefirst":
                attempts = [_kv(line).get("symbol") for line in text.splitlines()
                            if line.startswith("SF_ATTEMPT ")]
                if (len(attempts) != selected_count or
                        set(attempts) != set(selected)):
                    raise ExecutionError("dense attempt symbol census differs")
            else:
                cells = [_kv(line) for line in text.splitlines()
                         if line.startswith("FQ_TC_CELL ") and
                         _kv(line).get("shape") == shape]
                observed = {cell.get("symbol") for cell in cells}
                if observed != set(selected):
                    raise ExecutionError("dense cell symbol census differs")
                if shard.payload_kind == NO_DEVICE_KERNEL_STRUCTURAL:
                    if (shard.structural_proof is None or
                            len(cells) != len(selected) * 4):
                        raise ExecutionError(
                            "structural no-kernel cell denominator differs")
                    by_symbol: dict[str, list[dict[str, str]]] = {}
                    for cell in cells:
                        symbol = cell.get("symbol", "")
                        by_symbol.setdefault(symbol, []).append(cell)
                    for symbol in selected:
                        rows = by_symbol.get(symbol, [])
                        if ({row.get("S") for row in rows} !=
                                {"1", "2", "4", "8"} or
                                any(row.get("state") !=
                                    "SHIPPING_SHARED_STORAGE" or
                                    row.get("raw_bad") != "0"
                                    for row in rows)):
                            raise ExecutionError(
                                f"{symbol}: proved structural state differs")
                elif shard.structural_proof is not None:
                    raise ExecutionError(
                        "device-kernel payload carries a structural proof")
        return
    if shard.route == "scalefirst":
        header = _one_line(text, "SF_GROUPED_SHARD ")
        done = _one_line(text, "SF_GROUPED_COMPLETE ")
    else:
        header = _one_line(text, "FQ_GROUPED_KPACK_SHARD ")
        done = _one_line(text, "FQ_GROUPED_KPACK_COMPLETE ")
    wanted = {
        "q": str(shard.qtype), "selected_rows": str(selected_count),
        "workload": workload.key, "router_profile": row["profile"],
        "total_rows": row["total_rows"], "max_rows": row["max_rows"],
    }
    if workload.rows_fnv64 is not None:
        wanted["rows_hash"] = workload.rows_fnv64
        wanted["router"] = "exact-rows-v1"
    if any(header.get(key) != value for key, value in wanted.items()):
        raise ExecutionError("grouped fixture/shard header authority differs")
    if expected_grouped_warmups is not None:
        for field, expected in (("warmups", expected_grouped_warmups),
                                ("iterations", expected_iterations),
                                ("correctness_repeats",
                                 expected_correctness_repeats)):
            if expected is not None and header.get(field) != str(expected):
                raise ExecutionError(f"grouped shard {field} differs")
    if (expected_schedule_seed is not None and
            _header_u64(header, "schedule_seed", "grouped shard") !=
            expected_schedule_seed):
        raise ExecutionError("grouped shard schedule seed differs")
    if (done.get("status") != "PASS" or
            done.get("rows") != str(selected_count)):
        raise ExecutionError("grouped completion authority differs")
    if selected:
        prefix = ("SF_GROUPED_CELL " if shard.route == "scalefirst" else
                  "FQ_GROUPED_KPACK_CELL ")
        observed = {_kv(line).get("symbol") for line in text.splitlines()
                    if line.startswith(prefix)}
        if observed != set(selected):
            raise ExecutionError("grouped cell symbol census differs")


def validate_atom_log(text: str, command: list[str], shard: ResolvedShard,
                      workload: Workload, metadata: dict[str, Any],
                      selected_symbols: tuple[str, ...] | None = None) -> None:
    atom = _one_line(text, "KPACK_DISCOVERY_ATOM ")
    expected = {key: str(value) for key, value in metadata.items()}
    expected["argv_sha256"] = digest(command)
    if atom != expected:
        raise ExecutionError("atom log metadata/argv authority differs")
    seeds = [value[len("--schedule-seed="):] for value in command
             if value.startswith("--schedule-seed=")]
    try:
        seed = int(seeds[0], 0) if len(seeds) == 1 else -1
    except ValueError as error:
        raise ExecutionError("atom schedule-seed argv is malformed") from error
    if seed < 0 or seed >= 1 << 64:
        raise ExecutionError("atom schedule-seed argv differs")
    try:
        expected_seed = schedule_seed(
            str(metadata["work_item_id"]), str(metadata["phase"]),
            int(metadata["round"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ExecutionError("atom schedule-seed coordinate is malformed") from error
    if (seed != expected_seed or
            metadata.get("schedule_seed") != schedule_seed_hex(expected_seed)):
        raise ExecutionError("atom deterministic schedule seed differs")
    warmup_values = [value[len("--warmups="):] for value in command
                     if value.startswith("--warmups=")]
    grouped_warmups = None
    if shard.operator == "grouped":
        if (len(warmup_values) != 1 or not warmup_values[0].isdecimal() or
                int(warmup_values[0]) <= 0):
            raise ExecutionError("grouped atom warmups argv differs")
        grouped_warmups = int(warmup_values[0])
    elif warmup_values:
        raise ExecutionError("dense atom may not use grouped warmups")
    controls: dict[str, int] = {}
    for option in ("iterations", "correctness-repeats"):
        values = [value[len(f"--{option}="):] for value in command
                  if value.startswith(f"--{option}=")]
        if (len(values) != 1 or not values[0].isdecimal() or
                int(values[0]) <= 0):
            raise ExecutionError(f"atom {option} argv differs")
        controls[option] = int(values[0])
    validate_log(
        text, shard, workload, selected_symbols,
        expected_schedule_seed=seed,
        expected_grouped_warmups=grouped_warmups,
        expected_iterations=controls["iterations"],
        expected_correctness_repeats=controls["correctness-repeats"])


def _failure_path(target: Path) -> Path:
    directory = target.parents[1] / "failures"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{target.parent.name}.{target.name}.failed.{os.getpid()}"


def run_atomic_log(target: Path, command: list[str], shard: ResolvedShard,
                   workload: Workload, metadata: dict[str, Any],
                   selected_symbols: tuple[str, ...] | None = None) -> None:
    if target.is_symlink():
        raise ExecutionError(f"log may not be a symlink: {target}")
    if target.exists():
        if not target.is_file() or target.stat().st_size == 0:
            raise ExecutionError(f"resumed log is empty/nonregular: {target}")
        try:
            text = target.read_text(encoding="utf-8")
            validate_atom_log(
                text, command, shard, workload, metadata, selected_symbols)
        except (OSError, UnicodeDecodeError) as error:
            raise ExecutionError(f"cannot validate resumed log {target}: {error}") from error
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.current.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise ExecutionError(f"log staging path already exists: {temporary}")
    header = ("KPACK_DISCOVERY_ATOM " + " ".join(
        f"{key}={value}" for key, value in sorted(metadata.items())) +
        f" argv_sha256={digest(command)}\n").encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            stream.write(header)
            completed = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ), check=False)
            stream.flush()
            os.fsync(stream.fileno())
        payload = temporary.read_bytes()
        text = payload.decode("utf-8")
        if completed.returncode != 0:
            failed = _failure_path(target)
            os.replace(temporary, failed)
            raise ExecutionError(
                f"prebuilt atom failed rc={completed.returncode}; log={failed}")
        try:
            validate_atom_log(
                text, command, shard, workload, metadata, selected_symbols)
        except ExecutionError:
            failed = _failure_path(target)
            os.replace(temporary, failed)
            raise
        os.replace(temporary, target)
    except ExecutionError:
        raise
    except (OSError, UnicodeDecodeError) as error:
        raise ExecutionError(f"cannot execute/publish atom log {target}: {error}") from error


@lru_cache(maxsize=None)
def _loaded_library_record(path_text: str) -> tuple[str, str, int, str]:
    try:
        library = Path(path_text).resolve(strict=True)
    except OSError as error:
        raise ExecutionError(
            f"cannot resolve loaded runtime library {path_text}: {error}") from error
    if not library.is_file():
        raise ExecutionError(f"loaded runtime library is not a file: {path_text}")
    return path_text, str(library), library.stat().st_size, file_sha(library)


@lru_cache(maxsize=None)
def _linkage(binary: Path) -> tuple[tuple[str, str, str, int, str], ...]:
    try:
        completed = subprocess.run(
            ["ldd", str(binary)], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
    except OSError as error:
        raise ExecutionError(f"cannot inspect runtime linkage for {binary}: {error}") from error
    if completed.returncode:
        raise ExecutionError(f"ldd failed for {binary}: {completed.stdout[-1000:]}")
    result = []
    for line in completed.stdout.splitlines():
        match = re.match(r"\s*(libhggc\S*)\s+=>\s+(\S+)", line)
        if not match:
            continue
        reported, resolved, size, sha = _loaded_library_record(match.group(2))
        result.append((match.group(1), reported, resolved, size, sha))
    if not result:
        raise ExecutionError(f"cannot identify loaded libhggc runtime for {binary}")
    return tuple(sorted(result))


def _round_order(items: list[dict[str, Any]], round_index: int
                 ) -> list[dict[str, Any]]:
    if round_index == 1:
        return list(items)
    if round_index == 2:
        return list(reversed(items))
    ordered = sorted(items, key=lambda row: digest(
        [round_index, row["work_item_id"]]))
    return ordered


def _round_label(round_index: int) -> str:
    return {1: "FORWARD", 2: "REVERSE", 3: "HASHED"}[round_index]


@dataclass(frozen=True)
class Retention:
    symbols: tuple[str, ...]
    symbols_path: Path
    sidecar_path: Path


def materialize_sf_retention(out: Path, item: dict[str, Any],
                             shard: ResolvedShard, screen_log: Path
                             ) -> Retention:
    if shard.route != "scalefirst" or not shard.symbols:
        raise ExecutionError("ScaleFirst retention requires manifest symbols")
    directory = out / "retention"
    directory.mkdir(parents=True, exist_ok=True)
    symbols_path = directory / f"{item['work_item_id']}.symbols"
    sidecar_path = directory / f"{item['work_item_id']}.json"
    with tempfile.TemporaryDirectory(
            prefix="kpack-retention-", dir=out / "inputs") as name:
        temporary = Path(name)
        generated_symbols = temporary / "symbols"
        generated_sidecar = temporary / "sidecar.json"
        try:
            sf_analyzer.retain(
                shard.operator, shard.qtype, screen_log, shard.manifest,
                generated_symbols, generated_sidecar)
            symbols_payload = generated_symbols.read_bytes()
            sidecar_payload = generated_sidecar.read_bytes()
        except (OSError, ValueError, KeyError) as error:
            # A parent-range shard can be wholly structural for one workload.
            # The native selector rejects an empty symbol file, so retain an
            # explicit empty authority here and publish synthetic, hash-bound
            # confirmation markers later instead of pretending to launch it.
            try:
                manifest = sf_analyzer.validate_manifest(
                    shard.operator, shard.qtype, shard.manifest)
                candidates = sf_analyzer.load_candidates(
                    shard.operator, screen_log)
                records = sf_analyzer.product_census(
                    shard.operator, screen_log)
                authority_symbols = {
                    row["symbol"] for row in manifest["typed_rows"]}
                if candidates or set(records) != authority_symbols:
                    raise ExecutionError("screen is not an all-structural shard")
                for symbol in authority_symbols:
                    cells = records[symbol]
                    explicit = (
                        all(row.get("status") == "INADMISSIBLE" for row in cells)
                        if shard.operator == "dense" else
                        all(row.get("state") in {
                            "INADMISSIBLE_SHARED_STORAGE",
                            "INADMISSIBLE_OCCUPANCY"} for row in cells))
                    if not cells or not explicit:
                        raise ExecutionError(
                            f"{symbol} is not explicitly structural-unavailable")
                parent_range = manifest["parent_range"]
                typed = (manifest["denominator"]["typed_rows"]
                         if shard.operator == "dense" else
                         manifest["denominator"]["compiled_rows"])
                anchor_status = "NOT_APPLICABLE"
                if shard.operator == "dense" and shard.qtype == 12:
                    anchor = sf_analyzer.Q4_HISTORICAL_GEOMETRY_ANCHOR
                    anchor_status = ("STRUCTURAL_UNAVAILABLE"
                                     if anchor in authority_symbols else
                                     "OUTSIDE_THIS_PARENT_RANGE")
                sidecar = {
                    "schema": "quactlize.scalefirst_kpack_retention.v1",
                    "operator": shard.operator, "qtype": shard.qtype,
                    "layout": sf_analyzer.LAYOUT[shard.qtype],
                    "mapping_id": sf_analyzer.MAPPING[
                        sf_analyzer.LAYOUT[shard.qtype]],
                    "parent_range": parent_range,
                    "parent_ids": [row["parent_id"]
                                   for row in manifest["compiled_parents"]],
                    "manifest_sha256": file_sha(shard.manifest),
                    "screen_sha256": file_sha(screen_log),
                    "authority_typed_symbols": typed,
                    "retained_symbols": [], "retained_count": 0,
                    "structural_unavailable_symbols": sorted(authority_symbols),
                    "structural_unavailable_count": len(authority_symbols),
                    "elimination_rule": (
                        "NO_RAW_BIT_CLEAN_PRODUCT_FULL_OUTPUT_MEASUREMENT"),
                    "timing_rank_used_for_elimination": False,
                    "heuristic_algorithm_denominator": [
                        "NONPERSISTENT", "PERSISTENT"],
                    "split_k_policy": "EXCLUDED_DIAGNOSTIC_ONLY",
                    "q4_historical_geometry_anchor_status": anchor_status,
                    "q4_historical_geometry_translation": {
                        "source_xplane_symbol":
                            sf_analyzer.Q4_HISTORICAL_SOURCE_SYMBOL,
                        "canonical_kpack_candidate":
                            sf_analyzer.Q4_HISTORICAL_GEOMETRY_ANCHOR,
                        "authority": "CANDIDATE_ONLY_NOT_WINNER",
                    },
                }
                symbols_payload = b""
                sidecar_payload = (json.dumps(
                    sidecar, indent=2, sort_keys=True) + "\n").encode("utf-8")
            except (OSError, ValueError, KeyError, ExecutionError) as fallback:
                raise ExecutionError(
                    f"{item['work_item_id']}: ScaleFirst retention failed: "
                    f"{error}; all-structural proof failed: {fallback}") from error
    atomic_bytes(symbols_path, symbols_payload, frozen=True, allow_empty=True)
    atomic_bytes(sidecar_path, sidecar_payload, frozen=True)
    try:
        symbols = tuple(symbols_path.read_text(encoding="utf-8").splitlines())
        sidecar = load_json(sidecar_path, "ScaleFirst retention sidecar")
    except (OSError, UnicodeDecodeError) as error:
        raise ExecutionError(f"cannot read ScaleFirst retention: {error}") from error
    if (len(symbols) != len(set(symbols)) or
            not set(symbols).issubset(shard.symbols) or
            sidecar.get("retained_symbols") != list(symbols) or
            sidecar.get("screen_sha256") != file_sha(screen_log) or
            sidecar.get("manifest_sha256") != file_sha(shard.manifest) or
            sidecar.get("timing_rank_used_for_elimination") is not False):
        raise ExecutionError(
            f"{item['work_item_id']}: ScaleFirst retention authority differs")
    return Retention(symbols, symbols_path, sidecar_path)


def write_empty_confirm(path: Path, item_id: str, round_index: int,
                        order: str, screen_log: Path,
                        retention_symbols: Path, seed: int) -> None:
    payload = (
        "KPACK_DISCOVERY_EMPTY_CONFIRM "
        f"work_item_id={item_id} round={round_index} order={order} "
        f"schedule_seed={schedule_seed_hex(seed)} "
        f"screen_sha256={file_sha(screen_log)} "
        f"retention_symbols_sha256={file_sha(retention_symbols)}\n"
    ).encode("ascii")
    atomic_bytes(path, payload, frozen=True)


def _prepare_output(path: Path, resume: bool) -> Path:
    if path.is_symlink():
        raise ExecutionError("worker output may not be a symlink")
    if path.exists():
        if not resume or not path.is_dir():
            raise ExecutionError("existing worker output requires --resume")
    else:
        if resume:
            raise ExecutionError("--resume requires an existing worker output")
        path.mkdir(parents=True)
    for name in ("inputs", "results/screen", "completion"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path.resolve(strict=True)


def _device_binding(identity_path: Path, homogeneity_path: Path,
                    assignment: dict[str, Any], worker_id: int,
                    live_payload: bytes) -> tuple[dict[str, Any], str]:
    identity = load_json(identity_path, "device identity")
    try:
        probe_box_identity._validate_document(identity)
    except probe_box_identity.ProbeError as error:
        raise ExecutionError(f"device identity is malformed: {error}") from error
    persisted = identity_path.read_bytes()
    if persisted != live_payload:
        raise ExecutionError("persisted device identity differs from live probes")
    authority = load_json(homogeneity_path, "device homogeneity authority")
    try:
        identities = worker_plan.validate_device_authority(
            authority, assignment["worker_count"])
    except worker_plan.PlanError as error:
        raise ExecutionError(str(error)) from error
    identity_sha = hashlib.sha256(persisted).hexdigest()
    if identities.get(worker_id) != identity_sha:
        raise ExecutionError("worker device identity is not its homogeneity entry")
    return authority, identity_sha


def _execution_authority(
        args: argparse.Namespace, selection: dict[str, Any], identity_sha: str,
        linkage: tuple[tuple[str, str, str, int, str], ...], workload_index: Path,
        visible: str, shard_keys: Iterable[str],
        artifacts: dict[str, ArtifactContext] | None = None) -> dict[str, Any]:
    validated_shards = sorted(shard_keys)
    return {
        "schema": EXECUTION_SCHEMA,
        "worker_id": args.worker_id,
        "worker_count": selection["worker_count"],
        "bundle_sha256": file_sha(args.bundle),
        "workload_plan_sha256": file_sha(args.plan),
        "master_sha256": file_sha(args.master),
        "assignment_sha256": file_sha(args.assignment),
        "selection_sha256": file_sha(args.selection),
        "device_identity_sha256": identity_sha,
        "device_homogeneity_sha256": file_sha(args.device_homogeneity),
        "visible_device_ordinal": visible,
        "workload_index_sha256": file_sha(workload_index),
        "executor_sha256": file_sha(Path(__file__)),
        "work_items": len(selection["work_items"]),
        "validated_shards": len(validated_shards),
        "validated_shard_keys_sha256": digest(validated_shards),
        "validated_partition_artifacts": ([] if artifacts is None else [{
            "artifact_id": artifact_id,
            "partition_manifest_sha256": file_sha(
                context.root / "partition-bundle.json"),
        } for artifact_id, context in sorted(artifacts.items())]),
        "candidate_policy": (
            "SCREEN_ALL_COMPILED_CONFIRM_ALL_ADMISSIBLE_NO_TIMING_PRUNE"),
        "measurement": {
            "screen_iterations": args.screen_iterations,
            "confirm_iterations": args.confirm_iterations,
            "confirm_rounds": args.confirm_rounds,
            "correctness_repeats": args.correctness_repeats,
            "grouped_warmups": args.warmups,
            "schedule_seed": schedule_seed_contract(),
            "outer_round_order": "ASSIGNMENT_REVERSE_THEN_HASHED_V1",
            "inner_candidate_order": "ROUND_SEED_VARIED_ALL_ROUTES_OPERATORS",
        },
        "runtime_linkage": [list(row) for row in linkage],
    }


def _completion_document(
        item: dict[str, Any], out: Path, args: argparse.Namespace,
        authority_sha: str, retention: Retention | None,
        shard: ResolvedShard) -> dict[str, Any]:
    item_id = item["work_item_id"]
    execution_kind = (
        "STRUCTURAL_CENSUS_NO_DEVICE_KERNEL"
        if shard.payload_kind == NO_DEVICE_KERNEL_STRUCTURAL
        else "DEVICE_EXECUTION")
    logs = [{"phase": "screen", "round": 0,
             "evidence_kind": execution_kind,
             "path": f"results/screen/{item_id}.log",
             "sha256": file_sha(out / f"results/screen/{item_id}.log")}]
    for round_index in range(1, args.confirm_rounds + 1):
        relative = f"results/confirm-r{round_index}/{item_id}.log"
        logs.append({"phase": "confirm", "round": round_index,
                     "evidence_kind": (
                         "EMPTY_STRUCTURAL_MARKER" if retention is not None and
                         not retention.symbols else execution_kind),
                     "path": relative, "sha256": file_sha(out / relative)})
    return {
        "schema": COMPLETION_SCHEMA,
        "work_item_id": item_id,
        "worker_id": args.worker_id,
        "execution_authority_sha256": authority_sha,
        "candidate_policy": (
            "SCREEN_ALL_COMPILED_CONFIRM_ALL_ADMISSIBLE_NO_TIMING_PRUNE"),
        "retention": (None if retention is None else {
            "symbols_sha256": file_sha(retention.symbols_path),
            "sidecar_sha256": file_sha(retention.sidecar_path),
            "retained_count": len(retention.symbols),
            "rule": "STRUCTURAL_UNAVAILABLE_ONLY_NO_TIMING_RANK",
        }),
        "logs": logs,
    }


def _relative_record(out: Path, path: Path) -> dict[str, str]:
    try:
        relative = path.resolve(strict=True).relative_to(out)
    except (OSError, ValueError) as error:
        raise ExecutionError(f"evidence file escapes worker output: {path}") from error
    if path.is_symlink() or not path.is_file():
        raise ExecutionError(f"evidence path is not one regular file: {path}")
    return {"path": relative.as_posix(), "sha256": file_sha(path)}


def _route_result_authorities(
        out: Path, items: list[dict[str, Any]], args: argparse.Namespace,
        execution_authority_sha: str) -> dict[str, Path]:
    routes = sorted({item["route"] for item in items})
    result: dict[str, Path] = {}
    for route in routes:
        path = out / f"inputs/{route}-result-authority.json"
        document = {
            "schema": "quactlize.kpack-discovery-worker-route-result-authority.v2",
            "route": route,
            "worker_id": args.worker_id,
            "execution_authority_sha256": execution_authority_sha,
            "candidate_policy": "ALL_ADMISSIBLE_NO_TIMING_PRUNE",
            "work_item_ids": [item["work_item_id"] for item in items
                              if item["route"] == route],
            "screen_iterations": args.screen_iterations,
            "confirm_iterations": args.confirm_iterations,
            "confirm_rounds": args.confirm_rounds,
            "correctness_repeats": args.correctness_repeats,
            "grouped_warmups": args.warmups,
            "schedule_seed": schedule_seed_contract(),
        }
        atomic_json(path, document, frozen=True)
        result[route] = path
    return result


def _execution_inputs(out: Path, shard: ResolvedShard, workload: Workload,
                      retention: Retention | None) -> dict[str, Any]:
    """Freeze portable metadata for paths that existed on the execution host."""
    manifest_snapshot = (
        out / "inputs/shard-manifests" / f"{digest(shard.key)[:24]}.json")
    try:
        manifest_payload = shard.manifest.read_bytes()
    except OSError as error:
        raise ExecutionError(
            f"{shard.key}: cannot snapshot candidate manifest: {error}") from error
    atomic_bytes(manifest_snapshot, manifest_payload, frozen=True)
    if file_sha(manifest_snapshot) != file_sha(shard.manifest):
        raise ExecutionError(f"{shard.key}: manifest snapshot differs")

    structural_proof = None
    if shard.payload_kind == NO_DEVICE_KERNEL_STRUCTURAL:
        if shard.structural_proof is None:
            raise ExecutionError(f"{shard.key}: structural proof is absent")
        proof_snapshot = (
            out / "inputs/structural-proofs" / f"{digest(shard.key)[:24]}.json")
        try:
            proof_payload = shard.structural_proof.read_bytes()
        except OSError as error:
            raise ExecutionError(
                f"{shard.key}: cannot snapshot structural proof: {error}") from error
        atomic_bytes(proof_snapshot, proof_payload, frozen=True)
        if file_sha(proof_snapshot) != file_sha(shard.structural_proof):
            raise ExecutionError(f"{shard.key}: structural proof snapshot differs")
        structural_proof = {
            "size": shard.structural_proof.stat().st_size,
            "sha256": file_sha(shard.structural_proof),
            "snapshot": _relative_record(out, proof_snapshot),
        }
    elif shard.structural_proof is not None:
        raise ExecutionError(
            f"{shard.key}: device-kernel payload has a structural proof")

    rows_file = None
    if workload.rows_path is not None:
        rows_file = {
            "executed_path": str(workload.rows_path),
            "file": _relative_record(out, workload.rows_path),
        }
    result = {
        "artifact_id": shard.artifact_id,
        "shard_key": shard.key,
        "binary": {
            "executed_path": str(shard.binary),
            "size": shard.binary.stat().st_size,
            "sha256": file_sha(shard.binary),
        },
        "manifest": {
            "size": shard.manifest.stat().st_size,
            "sha256": file_sha(shard.manifest),
            "snapshot": _relative_record(out, manifest_snapshot),
        },
        "binary_receipt": {
            "size": shard.receipt.stat().st_size,
            "sha256": file_sha(shard.receipt),
        },
        "rows_file": rows_file,
        "retention_symbols_executed_path": (
            None if retention is None else str(retention.symbols_path)),
    }
    if shard.payload_kind == NO_DEVICE_KERNEL_STRUCTURAL:
        result["payload_kind"] = shard.payload_kind
        result["structural_proof"] = structural_proof
    return result


def _worker_evidence(
        out: Path, args: argparse.Namespace, selection: dict[str, Any],
        atoms: dict[str, tuple[ResolvedShard, Workload]],
        retentions: dict[str, Retention], route_authorities: dict[str, Path],
        identity_sha: str, execution_authority_path: Path) -> dict[str, Any]:
    rounds = [{"round": index, "order": _round_label(index)}
              for index in range(1, args.confirm_rounds + 1)]
    item_rows = []
    for item in selection["work_items"]:
        item_id = item["work_item_id"]
        shard, workload = atoms[item_id]
        screen_seed = schedule_seed(item_id, "screen", 0)
        screen_argv = command_for(
            shard, workload, iterations=args.screen_iterations,
            correctness_repeats=args.correctness_repeats,
            warmups=args.warmups,
            schedule_seed=screen_seed)
        retention = retentions.get(item_id)
        confirm = []
        for row in rounds:
            confirm_log = out / f"results/confirm-r{row['round']}/{item_id}.log"
            if retention is not None and not retention.symbols:
                confirm_seed = schedule_seed(
                    item_id, "confirm", row["round"])
                confirm.append({
                    **row, "empty_structural": True,
                    "schedule_seed": schedule_seed_hex(confirm_seed),
                    "log": _relative_record(out, confirm_log),
                })
                continue
            confirm_seed = schedule_seed(item_id, "confirm", row["round"])
            argv = command_for(
                shard, workload, iterations=args.confirm_iterations,
                correctness_repeats=args.correctness_repeats,
                warmups=args.warmups,
                schedule_seed=confirm_seed,
                symbol_file=(retention.symbols_path if retention else None))
            confirm.append({
                **row, "schedule_seed": schedule_seed_hex(confirm_seed),
                "argv": argv, "argv_sha256": digest(argv),
                "log": _relative_record(out, confirm_log),
            })
        authority = route_authorities[item["route"]]
        item_rows.append({
            "work_item_id": item_id,
            "result_authority_sha256": file_sha(authority),
            "execution_inputs": _execution_inputs(
                out, shard, workload, retention),
            "screen": {
                "schedule_seed": schedule_seed_hex(screen_seed),
                "argv": screen_argv, "argv_sha256": digest(screen_argv),
                "log": _relative_record(
                    out, out / f"results/screen/{item_id}.log"),
            },
            "retention": (None if retention is None else {
                "symbols": _relative_record(out, retention.symbols_path),
                "sidecar": _relative_record(out, retention.sidecar_path),
            }),
            "confirm": confirm,
        })
    return {
        "schema": EVIDENCE_SCHEMA,
        "worker_id": args.worker_id,
        "worker_count": selection["worker_count"],
        "bundle_sha256": file_sha(args.bundle),
        "workload_plan_sha256": file_sha(args.plan),
        "workload_index_sha256": file_sha(out / "inputs/workloads/index.json"),
        "master_sha256": file_sha(args.master),
        "assignment_sha256": file_sha(args.assignment),
        "device_homogeneity_sha256": file_sha(args.device_homogeneity),
        "device_identity_sha256": identity_sha,
        "execution_authority": _relative_record(
            out, execution_authority_path),
        "completion_result": _relative_record(out, out / "worker-result.json"),
        "result_authorities": [
            {"route": route, **_relative_record(out, path)}
            for route, path in sorted(route_authorities.items())],
        "run_contract": {
            "schedule_seed": schedule_seed_contract(),
            "grouped_warmups": args.warmups,
            "screen": {
                "timing_samples_per_runtime": args.screen_iterations,
                "correctness_repeats": args.correctness_repeats,
            },
            "confirm": {
                "timing_samples_per_runtime": args.confirm_iterations,
                "correctness_repeats": args.correctness_repeats,
                "rounds": rounds,
            },
            "all_admissible_candidates": True,
            "top_n": None,
        },
        "work_items": item_rows,
    }


def _write_id_file(path: Path, ids: Iterable[str]) -> None:
    payload = "".join(f"{item}\n" for item in ids).encode("ascii")
    atomic_bytes(path, payload, frozen=True)


def run_worker(args: argparse.Namespace) -> int:
    if args.confirm_iterations != 11 or args.confirm_rounds != 3:
        raise ExecutionError(
            "final worker contract requires exactly three 11-sample confirm rounds")
    visible = _visible_device(dict(os.environ))
    bundle_doc = _bundle_document(args.bundle)
    master, assignment, selection = _selection(
        args.bundle, args.plan, args.master, args.assignment, args.selection,
        args.worker_id)
    artifacts = None
    if bundle_doc.get("schema") == partitions.CATALOG_SCHEMA:
        artifacts = _catalog_artifacts(
            bundle_doc, selection, getattr(args, "artifact_root", []))
    elif getattr(args, "artifact_root", []):
        raise ExecutionError("monolithic bundle may not use --artifact-root")
    live_identity = _live_device_bytes(args.bundle, bundle_doc, artifacts)
    _homogeneity, identity_sha = _device_binding(
        args.device_identity, args.device_homogeneity, assignment,
        args.worker_id, live_identity)

    out = _prepare_output(args.output, args.resume)
    workloads_root = out / "inputs/workloads"
    try:
        workload_authority.materialize(args.plan, workloads_root)
        workload_authority.validate(args.plan, workloads_root)
    except (OSError, KeyError, ValueError,
            workload_authority.WorkloadError) as error:
        raise ExecutionError(f"canonical workload materialization failed: {error}") from error

    shard_rows = {row["shard_key"]: row for row in bundle_doc["shards"]}
    shards: dict[str, ResolvedShard] = {}
    workload_tables: dict[tuple[int, str], dict[str, Workload]] = {}
    atoms: dict[str, tuple[ResolvedShard, Workload]] = {}
    for item in selection["work_items"]:
        key = item["shard_key"]
        if key not in shard_rows:
            raise ExecutionError(f"selected shard {key} is absent")
        if key not in shards:
            shards[key] = (resolve_catalog_shard(bundle_doc, artifacts, key)
                           if artifacts is not None else
                           resolve_native_shard(args.bundle, bundle_doc, key))
        shard = shards[key]
        if (shard.route != item["route"] or shard.operator != item["operator"] or
                shard.qtype != item["qtype"] or
                shard.parent_count != item["parent_count"]):
            raise ExecutionError(f"{item['work_item_id']}: shard selection differs")
        table_key = (shard.qtype, shard.operator)
        if table_key not in workload_tables:
            workload_tables[table_key] = read_workloads(
                workloads_root, *table_key)
        workload = workload_tables[table_key].get(item["workload_key"])
        if workload is None:
            raise ExecutionError(
                f"{item['work_item_id']}: canonical workload is absent")
        atoms[item["work_item_id"]] = (shard, workload)

    probes = _probe_binaries(args.bundle, bundle_doc, artifacts)
    probe_linkage = _linkage(probes[0])
    if any(_linkage(probe) != probe_linkage for probe in probes[1:]):
        raise ExecutionError("component identity probes load different runtimes")
    for shard in shards.values():
        if _linkage(shard.binary) != probe_linkage:
            raise ExecutionError(f"{shard.key}: payload runtime linkage differs")

    authority = _execution_authority(
        args, selection, identity_sha, probe_linkage,
        workloads_root / "index.json", visible, shards, artifacts)
    authority_path = out / "inputs/execution-authority.json"
    atomic_json(authority_path, authority, frozen=True)
    authority_sha = file_sha(authority_path)

    items = selection["work_items"]
    if args.phase in ("screen", "all"):
        for item in items:
            shard, workload = atoms[item["work_item_id"]]
            seed = schedule_seed(item["work_item_id"], "screen", 0)
            command = command_for(
                shard, workload, iterations=args.screen_iterations,
                correctness_repeats=args.correctness_repeats,
                warmups=args.warmups, schedule_seed=seed)
            run_atomic_log(
                out / f"results/screen/{item['work_item_id']}.log",
                command, shard, workload,
                {"work_item_id": item["work_item_id"], "phase": "screen",
                 "round": 0, "order": "SCREEN", "worker": args.worker_id,
                 "schedule_seed": schedule_seed_hex(seed),
                 "grouped_warmups": (args.warmups if shard.operator == "grouped"
                                     else "NONE")})
        _write_id_file(out / "screen-completed.ids",
                       (item["work_item_id"] for item in items))
    else:
        for item in items:
            path = out / f"results/screen/{item['work_item_id']}.log"
            if not path.is_file():
                raise ExecutionError(
                    "confirm phase requires every assigned screen log")
            shard, workload = atoms[item["work_item_id"]]
            command = command_for(
                shard, workload, iterations=args.screen_iterations,
                correctness_repeats=args.correctness_repeats,
                warmups=args.warmups,
                schedule_seed=schedule_seed(
                    item["work_item_id"], "screen", 0))
            validate_atom_log(
                path.read_text(encoding="utf-8"), command, shard, workload,
                {"work_item_id": item["work_item_id"], "phase": "screen",
                 "round": 0, "order": "SCREEN", "worker": args.worker_id,
                 "schedule_seed": schedule_seed_hex(schedule_seed(
                     item["work_item_id"], "screen", 0)),
                 "grouped_warmups": (args.warmups if shard.operator == "grouped"
                                     else "NONE")})

    retentions: dict[str, Retention] = {}
    for item in items:
        shard, _workload = atoms[item["work_item_id"]]
        if shard.route == "scalefirst":
            retentions[item["work_item_id"]] = materialize_sf_retention(
                out, item, shard,
                out / f"results/screen/{item['work_item_id']}.log")

    if args.phase == "screen":
        print("KPACK_DISCOVERY_WORKER_SCREEN_COMPLETE "
              f"worker={args.worker_id} items={len(items)} top_n=NONE "
              f"output={out}")
        return 0

    for round_index in range(1, args.confirm_rounds + 1):
        directory = out / f"results/confirm-r{round_index}"
        directory.mkdir(parents=True, exist_ok=True)
        for item in _round_order(items, round_index):
            shard, workload = atoms[item["work_item_id"]]
            retention = retentions.get(item["work_item_id"])
            if retention is not None and not retention.symbols:
                write_empty_confirm(
                    directory / f"{item['work_item_id']}.log",
                    item["work_item_id"], round_index,
                    _round_label(round_index),
                    out / f"results/screen/{item['work_item_id']}.log",
                    retention.symbols_path,
                    schedule_seed(item["work_item_id"], "confirm", round_index))
                continue
            seed = schedule_seed(
                item["work_item_id"], "confirm", round_index)
            command = command_for(
                shard, workload, iterations=args.confirm_iterations,
                correctness_repeats=args.correctness_repeats,
                warmups=args.warmups, schedule_seed=seed,
                symbol_file=(retention.symbols_path
                             if retention is not None else None))
            run_atomic_log(
                directory / f"{item['work_item_id']}.log", command,
                shard, workload,
                {"work_item_id": item["work_item_id"], "phase": "confirm",
                 "round": round_index, "order": _round_label(round_index),
                 "worker": args.worker_id,
                 "schedule_seed": schedule_seed_hex(seed),
                 "grouped_warmups": (args.warmups if shard.operator == "grouped"
                                     else "NONE")},
                retention.symbols if retention is not None else None)

    expected_markers: set[str] = set()
    for item in items:
        marker = _completion_document(
            item, out, args, authority_sha,
            retentions.get(item["work_item_id"]),
            atoms[item["work_item_id"]][0])
        path = out / f"completion/{item['work_item_id']}.json"
        atomic_json(path, marker, frozen=True)
        expected_markers.add(path.name)
    observed_markers = {
        path.name for path in (out / "completion").iterdir()
        if path.is_file() and re.fullmatch(r"[0-9a-f]{64}\.json", path.name)}
    if observed_markers != expected_markers:
        raise ExecutionError("completion marker census differs from assignment")

    ids = [item["work_item_id"] for item in items]
    _write_id_file(out / "completed.ids", ids)
    result = {
        "schema": worker_plan.RESULT_SCHEMA,
        "worker_id": args.worker_id,
        "bundle_sha256": master["bundle_sha256"],
        "workload_plan_sha256": master["workload_plan_sha256"],
        "master_sha256": file_sha(args.master),
        "assignment_sha256": file_sha(args.assignment),
        "device_homogeneity_sha256": file_sha(args.device_homogeneity),
        "device_identity_sha256": identity_sha,
        "completed_work_item_ids": ids,
    }
    expected_ids = next(row["work_item_ids"] for row in assignment["workers"]
                        if row["worker_id"] == args.worker_id)
    if ids != expected_ids:
        raise ExecutionError("completed IDs differ from worker assignment")
    atomic_json(out / "worker-result.json", result, frozen=True)
    route_authorities = _route_result_authorities(
        out, items, args, authority_sha)
    evidence = _worker_evidence(
        out, args, selection, atoms, retentions, route_authorities,
        identity_sha, authority_path)
    atomic_json(out / "worker-evidence.json", evidence, frozen=True)
    print("KPACK_DISCOVERY_WORKER_COMPLETE "
          f"worker={args.worker_id} items={len(items)} rounds={args.confirm_rounds} "
          f"exactly_once=1 top_n=NONE output={out}")
    return 0


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="kpack-worker-executor-") as name:
        root = Path(name)
        binary = root / "kernel"
        binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        binary.chmod(0o755)
        manifest = root / "manifest.json"
        receipt = root / "receipt.json"
        manifest.write_text("{}\n", encoding="utf-8")
        receipt.write_text("{}\n", encoding="utf-8")
        dense = Workload("dense", "dense", {
            "workload_key": "dense", "source_class": "real-inventory",
            "m": "8", "n": "256", "k": "512"})
        grouped = Workload("grouped", "grouped", {
            "workload_key": "grouped", "source_class": "real-inventory",
            "tokens": "4", "topk": "2", "experts": "16", "n": "256",
            "k": "512", "profile": "uniform", "rows_file": "-",
            "total_rows": "8", "max_rows": "1", "rows_sha256": "-"})
        cases = (
            (ResolvedShard("sf-d", "scalefirst", "dense", 10, 3,
                           binary, manifest, receipt), dense,
             "SF_SHARD qtype=10 typed_rows=3 selected_rows=3\n"
             "SF_COMPLETE status=COMPLETE shape=8x256x512 typed_rows=3\n"),
            (ResolvedShard("fq-d", "fully-quantized", "dense", 10, 3,
                           binary, manifest, receipt), dense,
             "FQ_SHARD q=10 typed_rows=3 selected_rows=3\n"
             "FQ_SHAPE_DONE q=10 shape=8x256x512 typed_rows=3 "
             "selected_rows=3 status=PASS\n"),
            (ResolvedShard("sf-g", "scalefirst", "grouped", 10, 3,
                           binary, manifest, receipt), grouped,
             "SF_GROUPED_SHARD q=10 selected_rows=3 total_rows=8 max_rows=1 "
             "workload=grouped router_profile=uniform\n"
             "SF_GROUPED_COMPLETE status=PASS rows=3\n"),
            (ResolvedShard("fq-g", "fully-quantized", "grouped", 10, 3,
                           binary, manifest, receipt), grouped,
             "FQ_GROUPED_KPACK_SHARD q=10 selected_rows=3 total_rows=8 "
             "max_rows=1 workload=grouped router_profile=uniform\n"
             "FQ_GROUPED_KPACK_COMPLETE status=PASS rows=3\n"),
        )
        for shard, workload, log in cases:
            validate_log(log, shard, workload)
            command = command_for(
                shard, workload, iterations=5, correctness_repeats=7,
                warmups=3, schedule_seed=11)
            if (str(binary) != command[0] or
                    "--symbol-file" in " ".join(command) or
                    command.count("--schedule-seed=11") != 1):
                raise ExecutionError("self-test command pruned compiled parents")
        bad = cases[0][2].replace("selected_rows=3", "selected_rows=2")
        try:
            validate_log(bad, cases[0][0], dense)
        except ExecutionError:
            pass
        else:
            raise ExecutionError("stale selected-row negative stayed green")
        ids = [{"work_item_id": f"{index:064x}"} for index in range(4)]
        if (_round_order(ids, 1) != ids or _round_order(ids, 2) != ids[::-1] or
                set(row["work_item_id"] for row in _round_order(ids, 3)) !=
                set(row["work_item_id"] for row in ids)):
            raise ExecutionError("confirmation round schedule differs")
    print("[kpack-discovery-worker:self-test] PASS four route/operator CLIs "
          "full-parent screen+confirm validation, atomic/fail-closed model, "
          "no-top-N and three-round ordering; stale-row negative=RED")


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    select = commands.add_parser("select")
    for child in (select,):
        child.add_argument("--bundle", type=Path, required=True)
        child.add_argument("--plan", type=Path, required=True)
        child.add_argument("--master", type=Path, required=True)
        child.add_argument("--assignment", type=Path, required=True)
        child.add_argument("--worker-id", type=int, required=True)
    select.add_argument("--output", type=Path, required=True)
    probe = commands.add_parser("probe-device")
    probe.add_argument("--bundle", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--plan", type=Path)
    probe.add_argument("--master", type=Path)
    probe.add_argument("--assignment", type=Path)
    probe.add_argument("--selection", type=Path)
    probe.add_argument("--worker-id", type=int)
    probe.add_argument("--artifact-root", action="append", default=[])
    run = commands.add_parser("run")
    for argument in ("bundle", "plan", "master", "assignment", "selection",
                     "device-identity", "device-homogeneity", "output"):
        run.add_argument(f"--{argument}", type=Path, required=True)
    run.add_argument("--worker-id", type=int, required=True)
    run.add_argument("--artifact-root", action="append", default=[])
    run.add_argument("--phase", choices=("screen", "confirm", "all"),
                     default="all")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--screen-iterations", type=positive, default=5)
    run.add_argument("--confirm-iterations", type=positive, choices=(11,),
                     default=11)
    run.add_argument("--confirm-rounds", type=positive, choices=(3,),
                     default=3)
    run.add_argument("--correctness-repeats", type=positive, default=256)
    run.add_argument("--warmups", type=positive, default=3)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "self-test":
            self_test()
            return 0
        if args.command == "select":
            return select_worker(args)
        if args.command == "probe-device":
            return probe_device(args)
        return run_worker(args)
    except (ExecutionError, OSError, ValueError) as error:
        fail(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
