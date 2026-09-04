#!/usr/bin/env python3
"""Partition the exhaustive K-pack build without weakening its denominator.

The native ScaleFirst and FullyQuantized planners define individual binary
shards.  This module assigns those already-canonical shards to independent
build partitions.  It does not split or regenerate a shard, and it validates
the live planners whenever a plan is read.

A partition builder records a self-contained ``partition-bundle.json`` after
hashing every generated manifest, executable, receipt, and identity probe.
The metadata-only ``merge`` command then proves that independently published
partition bundles form the exact two-route union and freezes every shard's
parent and file identity into a distributed catalog.  It intentionally does
not copy, open, or require all executable payloads to be resident on one
machine.  Each assigned payload is revalidated after a worker fetches it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Iterable, NoReturn


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import analyze_scalefirst_kpack_discovery as sf_analyzer  # noqa: E402
import fully_quantized_kpack_bundle_index as fq_index  # noqa: E402
import fq_dense_structural_proof as fq_structural  # noqa: E402
import gen_fully_quantized_grouped_kpack_units as fq_grouped  # noqa: E402
import gen_fully_quantized_kpack_discovery_units as fq_dense  # noqa: E402
import scalefirst_kpack_binary_shards as sf_index  # noqa: E402


PLAN_SCHEMA = "quactlize.kpack-discovery-build-partitions.v1"
PARTITION_SCHEMA = "quactlize.kpack-discovery-partition-bundle.v1"
CATALOG_SCHEMA = "quactlize.kpack-discovery-distributed-catalog.v2"
ROUTES = ("scalefirst", "fully-quantized")
OPERATORS = ("dense", "grouped")
QTYPES = (10, 11, 12, 13, 14)
MAX_PARTITIONS = 32
DEVICE_KERNEL = "DEVICE_KERNEL"
NO_DEVICE_KERNEL_STRUCTURAL = "NO_DEVICE_KERNEL_STRUCTURAL"
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
OID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_PLAN_CACHE: dict[int, bytes] = {}


class PartitionError(ValueError):
    """A build partition cannot be proven against live authority."""


def fail(message: str) -> NoReturn:
    raise SystemExit(f"kpack discovery build partitions: {message}")


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PartitionError(f"value is not canonical JSON: {exc}") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    try:
        result = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(chunk)
        return result.hexdigest()
    except OSError as exc:
        raise PartitionError(f"cannot hash {path}: {exc}") from exc


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PartitionError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PartitionError(f"{label} must be a JSON object")
    return value


def write_frozen(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True,
                         allow_nan=False) + "\n"
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise PartitionError(f"existing output is not a regular file: {path}")
        if path.read_text(encoding="utf-8") != encoded:
            raise PartitionError(f"refusing to replace stale output {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PartitionError(f"{label} must be an integer")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or \
            any(mark in value for mark in ("\0", "\n", "\r")):
        raise PartitionError(f"{label} must be one nonempty line")
    return value


def _sha(value: Any, label: str) -> str:
    value = _string(value, label)
    if not SHA_RE.fullmatch(value):
        raise PartitionError(f"{label} must be one lowercase SHA-256")
    return value


def _oid(value: Any, label: str) -> str:
    value = _string(value, label)
    if not OID_RE.fullmatch(value):
        raise PartitionError(f"{label} must be one Git object ID")
    return value


def _relative(value: Any, label: str) -> PurePosixPath:
    raw = _string(value, label)
    if "\\" in raw:
        raise PartitionError(f"{label} must use POSIX separators")
    path = PurePosixPath(raw)
    if path.is_absolute() or raw != path.as_posix() or any(
            part in ("", ".", "..") for part in path.parts):
        raise PartitionError(f"{label} must be a normalized relative path")
    return path


def _inside(root: Path, value: Any, label: str, *, executable: bool = False
            ) -> Path:
    relative = _relative(value, label)
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise PartitionError(f"{label} may not be a symlink")
    try:
        path = candidate.resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as exc:
        raise PartitionError(f"{label} escapes or is missing: {exc}") from exc
    if not path.is_file():
        raise PartitionError(f"{label} is not a regular file")
    if executable and not os.access(path, os.X_OK):
        raise PartitionError(f"{label} is not executable")
    return path


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    try:
        relative = path.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise PartitionError(f"payload path escapes partition root: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise PartitionError(f"payload must be one regular non-symlink: {path}")
    return {"path": relative.as_posix(), "size": path.stat().st_size,
            "sha256": file_sha(path)}


def _parent_digest(values: list[int | str]) -> str:
    return digest([[type(value).__name__, value] for value in values])


def _parent_set_digest(values: list[int | str]) -> str:
    """Match the worker master's canonical, type-preserving parent set."""
    ordered = sorted(values, key=lambda value: (
        0, value) if isinstance(value, int) else (1, value))
    return digest(ordered)


def live_shards(partition_count: int) -> list[dict[str, Any]]:
    if isinstance(partition_count, bool) or not 1 <= partition_count <= MAX_PARTITIONS:
        raise PartitionError(f"partition count must be in [1,{MAX_PARTITIONS}]")
    rows: list[dict[str, Any]] = []
    sf_plan = sf_index.make_plan("full", 32)
    for native in sf_plan["shards"]:
        qtype = int(native["qtype"])
        operator = str(native["operator"])
        begin, end = int(native["parent_begin"]), int(native["parent_end"])
        symbols = sf_index.authority_symbols(operator, qtype)
        parents: list[int | str] = list(range(begin, end))
        rows.append({
            "shard_key": f"scalefirst:{native['shard_id']}",
            "native_shard_key": native["shard_id"],
            "route": "scalefirst", "qtype": qtype, "operator": operator,
            "layout": int(native["layout"]), "parent_begin": begin,
            "parent_end": end, "parent_count": end - begin,
            "authority_count": int(native["authority_parents"]),
            "parent_ids": parents,
            "parent_ids_sha256": _parent_digest(parents),
            "parent_id_set_sha256": _parent_set_digest(parents),
            "parent_symbols_sha256": digest(symbols[begin:end]),
        })
    for native in fq_index.plan(False, 32):
        parents = list(native["parent_ids"])
        rows.append({
            "shard_key": f"fully-quantized:{native['shard_key']}",
            "native_shard_key": native["shard_key"],
            "route": "fully-quantized", "qtype": int(native["qtype"]),
            "operator": str(native["operator"]),
            "layout": int(native["layout"]),
            "parent_begin": int(native["parent_begin"]),
            "parent_end": int(native["parent_end"]),
            "parent_count": int(native["parent_count"]),
            "authority_count": int(native["authority_count"]),
            "parent_ids": parents,
            "parent_ids_sha256": _parent_digest(parents),
            "parent_id_set_sha256": _parent_set_digest(parents),
            "parent_symbols_sha256": None,
        })

    # Each qtype/operator bucket is striped independently.  This prevents a
    # partition from accidentally becoming only Q4 or only grouped work and
    # keeps the parent count balanced while preserving every native range.
    bucket_ordinals: dict[tuple[str, int, str], int] = {}
    for row in rows:
        bucket = (row["route"], row["qtype"], row["operator"])
        ordinal = bucket_ordinals.get(bucket, 0)
        row["partition_id"] = ordinal % partition_count
        row["bucket_ordinal"] = ordinal
        bucket_ordinals[bucket] = ordinal + 1
    return rows


def make_plan(partition_count: int) -> dict[str, Any]:
    result = make_plan_unvalidated(partition_count)
    validate_plan(result)
    return result


def validate_plan(document: dict[str, Any]) -> None:
    if document.get("schema") != PLAN_SCHEMA:
        # The TM8 epilogue repair uses the same native shard/receipt/catalog
        # machinery with a strict subset plan.  Import lazily because that
        # adapter in turn uses this module's canonical native rows.
        try:
            import tm8_epilogue_selective_campaign as selective
        except ImportError as exc:
            raise PartitionError("partition plan schema differs") from exc
        if document.get("schema") != selective.PLAN_SCHEMA:
            raise PartitionError("partition plan schema differs")
        try:
            selective.validate_plan(document)
        except (KeyError, TypeError, ValueError) as exc:
            raise PartitionError(f"selective partition plan differs: {exc}") from exc
        return
    count = _integer(document.get("partition_count"), "partition_count")
    if not 1 <= count <= MAX_PARTITIONS:
        raise PartitionError("partition_count is out of range")
    expected = make_plan_unvalidated(count)
    if document != expected:
        raise PartitionError("partition plan differs from live shard authority")


def make_plan_unvalidated(partition_count: int) -> dict[str, Any]:
    # Avoid recursion from validate_plan while retaining one construction path.
    cached = _PLAN_CACHE.get(partition_count)
    if cached is not None:
        return json.loads(cached)
    rows = live_shards(partition_count)
    by_route = {route: sum(row["route"] == route for row in rows)
                for route in ROUTES}
    parents_by_route = {
        route: sum(row["parent_count"] for row in rows if row["route"] == route)
        for route in ROUTES}
    partitions = []
    for partition_id in range(partition_count):
        selected = [row for row in rows if row["partition_id"] == partition_id]
        partitions.append({
            "partition_id": partition_id,
            "shards": len(selected),
            "parents": sum(row["parent_count"] for row in selected),
            "shards_by_route": {
                route: sum(row["route"] == route for row in selected)
                for route in ROUTES},
            "parents_by_route": {
                route: sum(row["parent_count"] for row in selected
                           if row["route"] == route)
                for route in ROUTES},
            "shard_keys_sha256": digest(
                sorted(row["shard_key"] for row in selected)),
        })
    result = {
        "schema": PLAN_SCHEMA,
        "partition_count": partition_count,
        "assignment": "PAIR_LOCAL_ROUND_ROBIN_V1",
        "native_max_parents_per_binary": 32,
        "denominator": {
            "routes": list(ROUTES), "qtypes": list(QTYPES),
            "operators": list(OPERATORS), "shards": len(rows),
            "parents": sum(row["parent_count"] for row in rows),
            "shards_by_route": by_route,
            "parents_by_route": parents_by_route,
        },
        "partitions": partitions,
        "shards": rows,
    }
    _PLAN_CACHE[partition_count] = canonical(result)
    return result


def read_plan(path: Path) -> dict[str, Any]:
    document = load_json(path, "partition plan")
    validate_plan(document)
    return document


def selection(document: dict[str, Any], partition_id: int,
              route: str) -> list[dict[str, Any]]:
    validate_plan(document)
    count = document["partition_count"]
    if not 0 <= partition_id < count:
        raise PartitionError(f"partition id must be in [0,{count})")
    if route not in ROUTES:
        raise PartitionError(f"unsupported route {route!r}")
    rows = [row for row in document["shards"]
            if row["partition_id"] == partition_id and row["route"] == route]
    if not rows:
        raise PartitionError(f"partition {partition_id} has no {route} shards")
    return rows


def selection_tsv(document: dict[str, Any], partition_id: int,
                  route: str) -> str:
    lines = []
    for row in selection(document, partition_id, route):
        if route == "scalefirst":
            fields = (row["native_shard_key"], row["qtype"], row["operator"],
                      row["layout"], row["parent_begin"], row["parent_end"])
        else:
            fields = (row["native_shard_key"], row["qtype"], row["operator"],
                      row["parent_begin"], row["parent_end"],
                      row["parent_count"], row["authority_count"])
        lines.append("\t".join(map(str, fields)))
    return "\n".join(lines) + "\n"


def freeze_selection(plan_path: Path, frozen_plan: Path, output: Path,
                     partition_id: int, route: str) -> None:
    document = read_plan(plan_path)
    plan_bytes = plan_path.read_bytes()
    if frozen_plan.exists():
        if frozen_plan.is_symlink() or frozen_plan.read_bytes() != plan_bytes:
            raise PartitionError("resumed frozen partition plan differs")
    else:
        frozen_plan.parent.mkdir(parents=True, exist_ok=True)
        temporary = frozen_plan.with_name(
            f".{frozen_plan.name}.current.{os.getpid()}")
        temporary.write_bytes(plan_bytes)
        os.replace(temporary, frozen_plan)
    encoded = selection_tsv(document, partition_id, route)
    if output.exists():
        if output.is_symlink() or output.read_text(encoding="utf-8") != encoded:
            raise PartitionError("resumed build partition selection differs")
    else:
        temporary = output.with_name(f".{output.name}.current.{os.getpid()}")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, output)


def authority_partition_record(plan_path: Path, document: dict[str, Any],
                               partition_id: int, route: str) -> dict[str, Any]:
    rows = selection(document, partition_id, route)
    keys = [row["shard_key"] for row in rows]
    return {
        "schema": document["schema"],
        "plan_sha256": file_sha(plan_path),
        "partition_id": partition_id,
        "partition_count": document["partition_count"],
        "route": route,
        "selected_shards": len(rows),
        "selected_parents": sum(row["parent_count"] for row in rows),
        "selected_shard_keys_sha256": digest(sorted(keys)),
    }


def _check_source(value: dict[str, Any], label: str) -> None:
    _oid(value.get("source_sha"), f"{label}.source_sha")
    _oid(value.get("source_tree"), f"{label}.source_tree")
    submodules = value.get("submodules")
    if not isinstance(submodules, list):
        raise PartitionError(f"{label}.submodules must be an array")
    seen = set()
    for index, row in enumerate(submodules):
        if not isinstance(row, dict) or set(row) != {
                "path", "gitlink", "current"}:
            raise PartitionError(f"{label}.submodules[{index}] fields differ")
        path = _relative(row["path"], f"{label}.submodules[{index}].path")
        if path.as_posix() in seen:
            raise PartitionError(f"{label}.submodules contains a duplicate")
        seen.add(path.as_posix())
        if (_oid(row["gitlink"], f"{label}.submodules[{index}].gitlink") !=
                _oid(row["current"], f"{label}.submodules[{index}].current")):
            raise PartitionError(f"{label}.submodules[{index}] is not clean")
    sdk = value.get("sdk")
    if not isinstance(sdk, dict) or set(sdk) != {
            "receipt", "compiler", "inspector", "runtime_libraries"}:
        raise PartitionError(f"{label}.sdk must be an object")
    def sdk_file(record: Any, file_label: str) -> None:
        if not isinstance(record, dict) or set(record) != {
                "path", "size", "sha256", "symlink_target"}:
            raise PartitionError(f"{file_label} fields differ")
        _relative(record["path"], f"{file_label}.path")
        if _integer(record["size"], f"{file_label}.size") < 0:
            raise PartitionError(f"{file_label}.size must be nonnegative")
        _sha(record["sha256"], f"{file_label}.sha256")
        if record["symlink_target"] is not None:
            _string(record["symlink_target"], f"{file_label}.symlink_target")
    for field in ("receipt", "compiler", "inspector"):
        sdk_file(sdk[field], f"{label}.sdk.{field}")
    runtime = sdk["runtime_libraries"]
    if not isinstance(runtime, list) or not runtime:
        raise PartitionError(f"{label}.sdk.runtime_libraries must be nonempty")
    runtime_paths = set()
    for index, record in enumerate(runtime):
        sdk_file(record, f"{label}.sdk.runtime_libraries[{index}]")
        if record["path"] in runtime_paths:
            raise PartitionError(f"{label}.sdk runtime path is duplicate")
        runtime_paths.add(record["path"])


def _inspector_output(sdk: Path, binary: Path) -> tuple[str, str]:
    inspector = sdk / "bin/hgobjdump"
    if not inspector.is_file() or not os.access(inspector, os.X_OK):
        raise PartitionError("build SDK inspector is unavailable")
    try:
        output = subprocess.check_output(
            [str(inspector), "-lelf", str(binary)], text=True,
            stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PartitionError(f"cannot inspect {binary}: {exc}") from exc
    return output, hashlib.sha256(output.encode()).hexdigest()


def _inspector_arch(sdk: Path, binary: Path) -> tuple[str, str, str]:
    output, output_sha = _inspector_output(sdk, binary)
    match = re.search(r"ELF FILE \d+ \((PPU [^)]+)\)", output)
    if not match:
        raise PartitionError(f"{binary} contains no PPU image")
    arch = match.group(1)
    return (arch, output_sha,
            hashlib.sha256(arch.encode()).hexdigest())


def _record_scalefirst_shard(root: Path, sdk: Path, build_sha: str,
                             row: dict[str, Any]) -> dict[str, Any]:
    key = row["native_shard_key"]
    manifest = root / "generated" / key / "manifest.json"
    binary_name = ("test_scalefirst_internal_sweep"
                   if row["operator"] == "dense" else
                   "test_scalefirst_grouped_kpack_discovery")
    binary = root / "payloads" / key / binary_name
    receipt = root / "payloads" / key / "binary-receipt.json"
    for path, label, executable in ((manifest, "manifest", False),
                                    (binary, "binary", True),
                                    (receipt, "receipt", False)):
        _inside(root, path.relative_to(root).as_posix(),
                f"{key} {label}", executable=executable)
    parsed = sf_analyzer.validate_manifest(
        row["operator"], row["qtype"], manifest)
    begin, end = row["parent_begin"], row["parent_end"]
    compiled = parsed["compiled_parents"]
    ids = [entry["parent_id"] for entry in compiled]
    symbols = [entry["symbol"] for entry in compiled]
    expected_symbols = sf_index.authority_symbols(
        row["operator"], row["qtype"])[begin:end]
    if (parsed.get("parent_range") != {
            "begin": begin, "end": end, "count": end - begin,
            "authority_count": row["authority_count"]} or
            ids != list(range(begin, end)) or symbols != expected_symbols or
            _parent_digest(ids) != row["parent_ids_sha256"] or
            digest(symbols) != row["parent_symbols_sha256"]):
        raise PartitionError(f"{key} manifest parent authority differs")
    receipt_doc = load_json(receipt, f"{key} receipt")
    manifest_sha, binary_sha = file_sha(manifest), file_sha(binary)
    expected_receipt = {
        "schema": "quactlize.scalefirst_kpack_binary_receipt.v1",
        "build_input_authority_sha256": build_sha,
        "manifest_sha256": manifest_sha, "binary_sha256": binary_sha,
    }
    if receipt_doc != expected_receipt:
        raise PartitionError(f"{key} binary receipt differs")
    arch, inspector_sha, _arch_sha = _inspector_arch(sdk, binary)
    return {**row, "files": {
        "manifest": _file_record(root, manifest),
        "binary": _file_record(root, binary),
        "binary_receipt": _file_record(root, receipt)},
        "device_arch": arch, "inspector_output_sha256": inspector_sha}


def _record_fq_shard(root: Path, sdk: Path, build: dict[str, Any],
                     build_sha: str, row: dict[str, Any]) -> dict[str, Any]:
    key = row["native_shard_key"]
    manifest = root / "generated" / key / "manifest.json"
    binary_name = ("test_fully_quantized_internal_sweep"
                   if row["operator"] == "dense" else
                   "test_fully_quantized_grouped_kpack_discovery")
    binary = root / "payloads" / key / binary_name
    receipt = root / "payloads" / key / "binary-receipt.json"
    for path, label, executable in ((manifest, "manifest", False),
                                    (binary, "binary", True),
                                    (receipt, "receipt", False)):
        _inside(root, path.relative_to(root).as_posix(),
                f"{key} {label}", executable=executable)
    parsed = load_json(manifest, f"{key} manifest")
    validator = fq_dense.validate_manifest if row["operator"] == "dense" \
        else fq_grouped.validate_manifest
    validator(parsed)
    field = "dense_tc_parents" if row["operator"] == "dense" \
        else "grouped_parents"
    compiled = parsed[field]
    ids = [entry["static_candidate_id"] for entry in compiled]
    expected_ids = fq_index.authority_parent_ids(
        row["qtype"], row["operator"])[row["parent_begin"]:row["parent_end"]]
    if (parsed.get("parent_range") != {
            "begin": row["parent_begin"], "end": row["parent_end"],
            "count": row["parent_count"],
            "authority_count": row["authority_count"]} or
            ids != expected_ids or
            [entry["parent_ordinal"] for entry in compiled] !=
            list(range(row["parent_begin"], row["parent_end"])) or
            _parent_digest(ids) != row["parent_ids_sha256"]):
        raise PartitionError(f"{key} manifest parent authority differs")
    manifest_sha, binary_sha = file_sha(manifest), file_sha(binary)
    receipt_doc = load_json(receipt, f"{key} receipt")
    native = {
        "shard_key": key, "qtype": row["qtype"],
        "operator": row["operator"], "route": "fully-quantized",
        "layout": row["layout"], "parent_begin": row["parent_begin"],
        "parent_end": row["parent_end"], "parent_count": row["parent_count"],
        "authority_count": row["authority_count"], "parent_ids": ids,
        "typed_rows": len(compiled),
    }
    fq_index.validate_receipt(receipt_doc, native, manifest_sha, binary_sha)
    payload_kind = fq_index.receipt_kind(receipt_doc)
    if payload_kind not in (DEVICE_KERNEL, NO_DEVICE_KERNEL_STRUCTURAL):
        raise PartitionError(f"{key} payload kind differs")
    expected_manifest_path = manifest.relative_to(root).as_posix()
    expected_binary_path = binary.relative_to(root).as_posix()
    if (receipt_doc.get("build_input_authority_sha256") != build_sha or
            receipt_doc.get("source_sha") != build["source_sha"] or
            receipt_doc.get("source_tree") != build["source_tree"] or
            receipt_doc.get("submodules") != build["submodules"] or
            receipt_doc.get("sdk_compiler_sha256") !=
            build["sdk"]["compiler"]["sha256"] or
            receipt_doc.get("sdk_inspector_sha256") !=
            build["sdk"]["inspector"]["sha256"] or
            receipt_doc.get("manifest") != expected_manifest_path or
            receipt_doc.get("binary") != expected_binary_path):
        raise PartitionError(f"{key} source/SDK/path receipt chain differs")
    files = {
        "manifest": _file_record(root, manifest),
        "binary": _file_record(root, binary),
        "binary_receipt": _file_record(root, receipt),
    }
    if payload_kind == DEVICE_KERNEL:
        arch, _full_inspector_sha, inspector_sha = _inspector_arch(sdk, binary)
        if (receipt_doc.get("device_arch") != arch or
                receipt_doc.get("inspector_output_sha256") != inspector_sha):
            raise PartitionError(f"{key} device image receipt differs")
        return {**row, "files": files, "device_arch": arch,
                "inspector_output_sha256": inspector_sha}

    if row["operator"] != "dense":
        raise PartitionError(
            f"{key} structural no-kernel payload is outside FQ dense")
    proof = root / "payloads" / key / "structural-proof.json"
    _inside(root, proof.relative_to(root).as_posix(),
            f"{key} structural proof")
    proof_doc = load_json(proof, f"{key} structural proof")
    proof_sha = file_sha(proof)
    if (receipt_doc.get("structural_proof") !=
            proof.relative_to(root).as_posix() or
            receipt_doc.get("structural_proof_sha256") != proof_sha):
        raise PartitionError(f"{key} structural proof receipt chain differs")
    try:
        fq_structural.validate_structural_proof(
            proof_doc, native, manifest_sha, binary_sha, receipt_doc)
    except (KeyError, TypeError, ValueError) as exc:
        raise PartitionError(f"{key} structural proof differs: {exc}") from exc
    inspector_output, inspector_sha = _inspector_output(sdk, binary)
    if re.search(r"ELF FILE \d+ \(PPU [^)]+\)", inspector_output):
        raise PartitionError(f"{key} structural payload unexpectedly has a PPU image")
    if (receipt_doc.get("device_arch") != "NO_DEVICE_KERNEL" or
            receipt_doc.get("inspector_output_sha256") != inspector_sha):
        raise PartitionError(f"{key} structural image receipt differs")
    files["structural_proof"] = _file_record(root, proof)
    return {**row, "payload_kind": payload_kind, "files": files,
            "device_arch": "NO_DEVICE_KERNEL",
            "inspector_output_sha256": inspector_sha}


def record_partition(root: Path, route: str) -> dict[str, Any]:
    if root.is_symlink():
        raise PartitionError("partition root may not be a symlink")
    root = root.resolve(strict=True)
    if route not in ROUTES:
        raise PartitionError(f"unsupported route {route!r}")
    plan_path = root / "inputs" / "build-partition-plan.json"
    if route == "scalefirst":
        authority_path = root / "build-input-authority.json"
        selection_path = root / "selected-shards.tsv"
        probe = root / "payloads/support/box_identity_probe"
        probe_receipt = root / "payloads/support/identity-probe-receipt.json"
        build_schema = "quactlize.scalefirst_kpack_build_input.v1"
    else:
        authority_path = root / "inputs/build-input-authority.json"
        selection_path = root / "inputs/selected-shards.tsv"
        probe = root / "payloads/box_identity_probe"
        probe_receipt = root / "payloads/box_identity_probe.receipt.json"
        build_schema = "quactlize.fully_quantized_kpack_build_input.v2"
    plan = read_plan(plan_path)
    build = load_json(authority_path, "build input authority")
    if build.get("schema") != build_schema:
        raise PartitionError("build input authority schema differs")
    _check_source(build, "build input authority")
    partition = build.get("configuration", {}).get("distributed_partition")
    if not isinstance(partition, dict):
        raise PartitionError("build input authority lacks distributed partition")
    partition_id = _integer(partition.get("partition_id"), "partition_id")
    expected_partition = authority_partition_record(
        plan_path, plan, partition_id, route)
    if partition != expected_partition:
        raise PartitionError("build input distributed partition authority differs")
    expected_selection = selection_tsv(plan, partition_id, route)
    if (not selection_path.is_file() or selection_path.is_symlink() or
            selection_path.read_text(encoding="utf-8") != expected_selection):
        raise PartitionError("built shard selection differs")
    build_sha = file_sha(authority_path)
    sdk_path = Path(build["sdk"]["compiler"]["path"])
    # SDK paths are relative to the SDK root in both native authorities.  The
    # builder freezes the actual root separately in its process; recover it
    # from the resolved hgcc path supplied by the record command environment.
    sdk_env = os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME")
    if not sdk_env:
        raise PartitionError("set PPU_SDK to record a partition")
    sdk = Path(sdk_env).resolve(strict=True)
    compiler = _inside(sdk, sdk_path.as_posix(), "SDK compiler", executable=True)
    if file_sha(compiler) != build["sdk"]["compiler"]["sha256"]:
        raise PartitionError("recording SDK compiler differs from build authority")
    probe_record = _file_record(root, probe)
    receipt_record = _file_record(root, probe_receipt)
    probe_doc = load_json(probe_receipt, "identity probe receipt")
    expected_probe_schema = (
        "quactlize.scalefirst_kpack_identity_probe_receipt.v1"
        if route == "scalefirst" else
        "quactlize.fq_kpack_identity_probe_receipt.v1")
    if (probe_doc.get("schema") != expected_probe_schema or
            probe_doc.get("build_input_authority_sha256") != build_sha or
            probe_doc.get("binary_sha256") != probe_record["sha256"]):
        raise PartitionError("identity probe receipt chain differs")
    rows = selection(plan, partition_id, route)
    recorded = [(_record_scalefirst_shard(root, sdk, build_sha, row)
                 if route == "scalefirst" else
                 _record_fq_shard(root, sdk, build, build_sha, row))
                for row in rows]
    artifact_id = (
        f"kpack-discovery/{build['source_sha']}/{file_sha(plan_path)[:16]}/"
        f"{route}/p{partition_id:02d}-of-{plan['partition_count']:02d}")
    result = {
        "schema": PARTITION_SCHEMA,
        "artifact_id": artifact_id,
        "route": route, "partition_id": partition_id,
        "partition_count": plan["partition_count"],
        "source_sha": build["source_sha"],
        "source_tree": build["source_tree"],
        "submodules": build["submodules"], "sdk": build["sdk"],
        "partition_plan": _file_record(root, plan_path),
        "build_input_authority": _file_record(root, authority_path),
        "runtime_identity_probe": {
            "binary": probe_record, "receipt": receipt_record},
        "payload_validation": "RECORDED_FROM_LOCAL_BYTES",
        "denominator": {
            "shards": len(recorded),
            "parents": sum(row["parent_count"] for row in recorded),
            "shard_keys_sha256": digest(sorted(
                row["shard_key"] for row in recorded)),
        },
        "shards": recorded,
    }
    validate_partition_document(result, plan)
    write_frozen(root / "partition-bundle.json", result)
    return result


def _validate_file_record(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"path", "size", "sha256"}:
        raise PartitionError(f"{label} file record fields differ")
    _relative(value["path"], f"{label}.path")
    if _integer(value["size"], f"{label}.size") < 0:
        raise PartitionError(f"{label}.size must be nonnegative")
    _sha(value["sha256"], f"{label}.sha256")


def _partition_payload_kind(item: dict[str, Any], key: str) -> str:
    kind = item.get("payload_kind", DEVICE_KERNEL)
    if kind not in (DEVICE_KERNEL, NO_DEVICE_KERNEL_STRUCTURAL):
        raise PartitionError(f"{key} payload kind differs")
    if kind == NO_DEVICE_KERNEL_STRUCTURAL and not (
            item.get("route") == "fully-quantized" and
            item.get("operator") == "dense"):
        raise PartitionError(
            f"{key} structural no-kernel payload is outside FQ dense")
    return kind


def _partition_file_fields(kind: str) -> set[str]:
    fields = {"manifest", "binary", "binary_receipt"}
    if kind == NO_DEVICE_KERNEL_STRUCTURAL:
        fields.add("structural_proof")
    return fields


def validate_partition_document(document: dict[str, Any],
                                plan: dict[str, Any]) -> None:
    validate_plan(plan)
    expected_top = {
        "schema", "artifact_id", "route", "partition_id",
        "partition_count", "source_sha", "source_tree", "submodules", "sdk",
        "partition_plan", "build_input_authority", "runtime_identity_probe",
        "payload_validation", "denominator", "shards"}
    if set(document) != expected_top:
        raise PartitionError("partition bundle fields differ")
    if document.get("schema") != PARTITION_SCHEMA:
        raise PartitionError("partition bundle schema differs")
    route = _string(document.get("route"), "partition route")
    partition_id = _integer(document.get("partition_id"), "partition id")
    if document.get("partition_count") != plan["partition_count"]:
        raise PartitionError("partition count differs from plan")
    rows = selection(plan, partition_id, route)
    _check_source(document, "partition bundle")
    expected_artifact = (
        f"kpack-discovery/{document['source_sha']}/"
        f"{document.get('partition_plan', {}).get('sha256', '')[:16]}/"
        f"{route}/p{partition_id:02d}-of-{plan['partition_count']:02d}")
    if document.get("artifact_id") != expected_artifact:
        raise PartitionError("partition artifact identity differs")
    for field in ("partition_plan", "build_input_authority"):
        _validate_file_record(document.get(field), field)
    probe = document.get("runtime_identity_probe")
    if not isinstance(probe, dict) or set(probe) != {"binary", "receipt"}:
        raise PartitionError("partition runtime probe differs")
    _validate_file_record(probe["binary"], "runtime probe binary")
    _validate_file_record(probe["receipt"], "runtime probe receipt")
    if document.get("payload_validation") != "RECORDED_FROM_LOCAL_BYTES":
        raise PartitionError("partition payload validation state differs")
    actual = document.get("shards")
    if not isinstance(actual, list) or len(actual) != len(rows):
        raise PartitionError("partition shard count differs")
    expected_by_key = {row["shard_key"]: row for row in rows}
    seen = set()
    file_paths = {
        document["partition_plan"]["path"],
        document["build_input_authority"]["path"],
        probe["binary"]["path"], probe["receipt"]["path"]}
    if len(file_paths) != 4:
        raise PartitionError("partition authority files alias each other")
    for item in actual:
        if not isinstance(item, dict):
            raise PartitionError("partition shard is malformed")
        key = item.get("shard_key")
        if key in seen or key not in expected_by_key:
            raise PartitionError("partition shard union is duplicate or foreign")
        seen.add(key)
        expected = expected_by_key[key]
        kind = _partition_payload_kind(item, key)
        extra = {"files", "device_arch", "inspector_output_sha256"}
        if kind == NO_DEVICE_KERNEL_STRUCTURAL:
            extra.add("payload_kind")
        if set(item) != set(expected) | extra:
            raise PartitionError(f"{key} shard fields differ")
        for field, value in expected.items():
            if item.get(field) != value:
                raise PartitionError(f"{key} field {field} differs")
        files = item.get("files")
        if not isinstance(files, dict) or set(files) != _partition_file_fields(kind):
            raise PartitionError(f"{key} file set differs")
        for label, value in files.items():
            _validate_file_record(value, f"{key}.{label}")
            if value["path"] in file_paths:
                raise PartitionError(f"{key}.{label} aliases another file")
            file_paths.add(value["path"])
        arch = _string(item.get("device_arch"), f"{key}.device_arch")
        if ((kind == NO_DEVICE_KERNEL_STRUCTURAL) !=
                (arch == "NO_DEVICE_KERNEL")):
            raise PartitionError(f"{key} payload kind/device arch differs")
        _sha(item.get("inspector_output_sha256"),
             f"{key}.inspector_output_sha256")
    if seen != set(expected_by_key):
        raise PartitionError("partition shard union has a gap")
    den = document.get("denominator")
    expected_den = {
        "shards": len(rows),
        "parents": sum(row["parent_count"] for row in rows),
        "shard_keys_sha256": digest(sorted(row["shard_key"] for row in rows)),
    }
    if den != expected_den:
        raise PartitionError("partition denominator differs")


def verify_partition(path: Path, root: Path) -> dict[str, Any]:
    if root.is_symlink():
        raise PartitionError("partition root may not be a symlink")
    root = root.resolve(strict=True)
    manifest = path.resolve(strict=True)
    if manifest != root / "partition-bundle.json":
        raise PartitionError("partition manifest must be ROOT/partition-bundle.json")
    document = load_json(manifest, "partition bundle")
    plan_record = document.get("partition_plan")
    _validate_file_record(plan_record, "partition plan")
    plan_path = _inside(root, plan_record["path"], "partition plan")
    if file_sha(plan_path) != plan_record["sha256"]:
        raise PartitionError("partition plan hash differs")
    plan = read_plan(plan_path)
    validate_partition_document(document, plan)
    records: list[tuple[str, dict[str, Any], bool]] = [
        ("build input authority", document["build_input_authority"], False),
        ("runtime probe binary",
         document["runtime_identity_probe"]["binary"], True),
        ("runtime probe receipt",
         document["runtime_identity_probe"]["receipt"], False),
    ]
    for row in document["shards"]:
        for label, record in row["files"].items():
            records.append((f"{row['shard_key']} {label}", record,
                            label == "binary"))
    for label, record, executable in records:
        path_value = _inside(root, record["path"], label,
                             executable=executable)
        if path_value.stat().st_size != record["size"] or \
                file_sha(path_value) != record["sha256"]:
            raise PartitionError(f"{label} bytes differ")
    for row in document["shards"]:
        if _partition_payload_kind(row, row["shard_key"]) != \
                NO_DEVICE_KERNEL_STRUCTURAL:
            continue
        files = row["files"]
        receipt_path = _inside(
            root, files["binary_receipt"]["path"],
            f"{row['shard_key']} structural receipt")
        proof_path = _inside(
            root, files["structural_proof"]["path"],
            f"{row['shard_key']} structural proof")
        receipt_doc = load_json(receipt_path, "structural receipt")
        proof_doc = load_json(proof_path, "structural proof")
        native = {
            "shard_key": row["native_shard_key"],
            "qtype": row["qtype"], "operator": row["operator"],
            "route": row["route"], "layout": row["layout"],
            "parent_begin": row["parent_begin"],
            "parent_end": row["parent_end"],
            "parent_count": row["parent_count"],
            "authority_count": row["authority_count"],
            "parent_ids": row["parent_ids"],
            "typed_rows": row["parent_count"],
        }
        try:
            fq_index.validate_receipt(
                receipt_doc, native, files["manifest"]["sha256"],
                files["binary"]["sha256"])
            fq_structural.validate_structural_proof(
                proof_doc, native, files["manifest"]["sha256"],
                files["binary"]["sha256"], receipt_doc)
        except (KeyError, TypeError, ValueError) as exc:
            raise PartitionError(
                f"{row['shard_key']} fetched structural proof differs: {exc}") \
                from exc
        sdk_env = os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME")
        if not sdk_env:
            raise PartitionError(
                "set PPU_SDK to verify a structural partition payload")
        binary_path = _inside(
            root, files["binary"]["path"],
            f"{row['shard_key']} structural binary", executable=True)
        inspector_output, inspector_sha = _inspector_output(
            Path(sdk_env).resolve(strict=True), binary_path)
        if (re.search(r"ELF FILE \d+ \(PPU [^)]+\)", inspector_output) or
                inspector_sha != row["inspector_output_sha256"] or
                receipt_doc.get("inspector_output_sha256") != inspector_sha):
            raise PartitionError(
                f"{row['shard_key']} fetched structural image differs")
    return document


def _manifest_entry(path: Path, plan: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise PartitionError(f"partition bundle is not a regular file: {path}")
    document = load_json(path, "partition bundle")
    validate_partition_document(document, plan)
    return document, file_sha(path)


def _catalog_shard(document: dict[str, Any], row: dict[str, Any]
                   ) -> dict[str, Any]:
    """Freeze every shard fact needed by a metadata-only master."""
    layout = _integer(row.get("layout"), f"{row.get('shard_key')} layout")
    try:
        mapping_id = sf_analyzer.MAPPING[layout]
    except KeyError as exc:
        raise PartitionError(f"unsupported canonical layout {layout}") from exc
    manifest = row.get("files", {}).get("manifest")
    _validate_file_record(manifest, f"{row.get('shard_key')} manifest")
    return {
        **row,
        "artifact_id": document["artifact_id"],
        "mapping_id": mapping_id,
        "manifest_sha256": manifest["sha256"],
    }


def _expected_catalog_denominator(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        **plan["denominator"],
        "logical_partitions": plan["partition_count"],
        "partition_artifacts": plan["partition_count"] * len(ROUTES),
    }


def validate_catalog_document(document: dict[str, Any]) -> dict[str, Any]:
    """Validate a distributed catalog without requiring payload residency."""
    expected_top = {
        "schema", "source_sha", "source_tree", "submodules", "sdk",
        "partition_plan_sha256", "partition_count", "partition_assignment",
        "payload_residency", "denominator", "partitions", "shards",
    }
    selective_record = document.get("selective_scope")
    if selective_record is not None:
        expected_top.add("selective_scope")
    if set(document) != expected_top or document.get("schema") != CATALOG_SCHEMA:
        raise PartitionError("distributed catalog schema/fields differ")
    count = _integer(document.get("partition_count"), "catalog partition_count")
    if selective_record is None:
        plan = make_plan(count)
    else:
        try:
            import tm8_epilogue_selective_campaign as selective
            if not isinstance(selective_record, dict):
                raise PartitionError("catalog selective scope record is malformed")
            live_scope = selective.tm8_scope.make_plan()
            plan = selective.make_plan(
                live_scope, count,
                selective_record.get("qtypes"))
            if plan["selective_scope"] != selective_record:
                raise PartitionError("catalog selective scope differs")
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, PartitionError):
                raise
            raise PartitionError(f"catalog selective scope differs: {exc}") from exc
    _check_source(document, "distributed catalog")
    if selective_record is not None:
        source = live_scope["source"]
        actlize = [row for row in document["submodules"]
                   if row["path"] == "third_party/actlize"]
        if (document["source_sha"] != source["repository_git_commit"] or
                len(actlize) != 1 or
                actlize[0]["gitlink"] != source["actlize_git_commit"]):
            raise PartitionError("catalog source/TM8 scope identity differs")
    plan_sha = _sha(document.get("partition_plan_sha256"),
                    "catalog partition plan SHA-256")
    if selective_record is not None:
        encoded_plan = (json.dumps(
            plan, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        if hashlib.sha256(encoded_plan).hexdigest() != plan_sha:
            raise PartitionError("catalog selective partition plan hash differs")
    if document.get("partition_assignment") != plan["assignment"]:
        raise PartitionError("catalog partition assignment differs")
    if document.get("payload_residency") != "PER_WORKER_PARTITION_FETCH_AND_VERIFY":
        raise PartitionError("catalog payload residency policy differs")
    if document.get("denominator") != _expected_catalog_denominator(plan):
        raise PartitionError("catalog denominator differs from live authority")

    raw_partitions = document.get("partitions")
    if not isinstance(raw_partitions, list):
        raise PartitionError("catalog partitions are malformed")
    expected_partition_keys = {
        (route, partition_id) for route in ROUTES
        for partition_id in range(count)}
    partitions_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    artifact_ids: set[str] = set()
    for index, record in enumerate(raw_partitions):
        if not isinstance(record, dict) or set(record) != {
                "route", "partition_id", "artifact_id",
                "partition_manifest", "shards", "parents",
                "shard_keys_sha256"}:
            raise PartitionError(f"catalog partition {index} fields differ")
        route = _string(record.get("route"), f"catalog partition {index}.route")
        partition_id = _integer(record.get("partition_id"),
                                f"catalog partition {index}.partition_id")
        key = (route, partition_id)
        if key in partitions_by_key:
            raise PartitionError("catalog contains a duplicate partition")
        artifact_id = _string(record.get("artifact_id"),
                              f"catalog partition {index}.artifact_id")
        expected_artifact = (
            f"kpack-discovery/{document['source_sha']}/{plan_sha[:16]}/"
            f"{route}/p{partition_id:02d}-of-{count:02d}")
        if artifact_id != expected_artifact or artifact_id in artifact_ids:
            raise PartitionError("catalog partition artifact identity differs")
        artifact_ids.add(artifact_id)
        manifest = record.get("partition_manifest")
        _validate_file_record(manifest, "catalog partition manifest")
        if manifest["path"] != "partition-bundle.json":
            raise PartitionError("catalog partition manifest path differs")
        partitions_by_key[key] = record
    if set(partitions_by_key) != expected_partition_keys:
        raise PartitionError("catalog partition union has a gap/extra")

    raw_shards = document.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise PartitionError("catalog shards are empty/malformed")
    live_by_key = {row["shard_key"]: row for row in plan["shards"]}
    observed_by_key: dict[str, dict[str, Any]] = {}
    expected_extra = {
        "artifact_id", "mapping_id", "manifest_sha256", "files",
        "device_arch", "inspector_output_sha256"}
    artifact_shards: dict[str, list[dict[str, Any]]] = {
        artifact_id: [] for artifact_id in artifact_ids}
    for row in raw_shards:
        if not isinstance(row, dict):
            raise PartitionError("catalog shard is malformed")
        key = row.get("shard_key")
        if key in observed_by_key or key not in live_by_key:
            raise PartitionError("catalog shard union is duplicate or foreign")
        expected = live_by_key[key]
        kind = _partition_payload_kind(row, str(key))
        row_extra = set(expected_extra)
        if kind == NO_DEVICE_KERNEL_STRUCTURAL:
            row_extra.add("payload_kind")
        if set(row) != set(expected) | row_extra:
            raise PartitionError(f"catalog shard {key} fields differ")
        for field, value in expected.items():
            if row.get(field) != value:
                raise PartitionError(f"catalog shard {key} field {field} differs")
        if (row.get("parent_id_set_sha256") !=
                _parent_set_digest(row["parent_ids"])):
            raise PartitionError(f"catalog shard {key} parent-set digest differs")
        expected_artifact = partitions_by_key[
            (row["route"], row["partition_id"])]["artifact_id"]
        if row.get("artifact_id") != expected_artifact:
            raise PartitionError(f"catalog shard {key} artifact differs")
        expected_mapping = sf_analyzer.MAPPING.get(row["layout"])
        if row.get("mapping_id") != expected_mapping:
            raise PartitionError(f"catalog shard {key} mapping differs")
        files = row.get("files")
        if not isinstance(files, dict) or set(files) != _partition_file_fields(kind):
            raise PartitionError(f"catalog shard {key} files differ")
        paths: set[str] = set()
        for label, value in files.items():
            _validate_file_record(value, f"catalog shard {key}.{label}")
            if value["path"] in paths:
                raise PartitionError(f"catalog shard {key} files alias")
            paths.add(value["path"])
        if row.get("manifest_sha256") != files["manifest"]["sha256"]:
            raise PartitionError(f"catalog shard {key} manifest hash differs")
        arch = _string(row.get("device_arch"),
                       f"catalog shard {key}.device_arch")
        if ((kind == NO_DEVICE_KERNEL_STRUCTURAL) !=
                (arch == "NO_DEVICE_KERNEL")):
            raise PartitionError(
                f"catalog shard {key} payload kind/device arch differs")
        _sha(row.get("inspector_output_sha256"),
             f"catalog shard {key}.inspector_output_sha256")
        observed_by_key[key] = row
        artifact_shards[expected_artifact].append(row)
    if set(observed_by_key) != set(live_by_key):
        raise PartitionError("catalog shard union has a gap")

    for record in partitions_by_key.values():
        rows = artifact_shards[record["artifact_id"]]
        expected = {
            "shards": len(rows),
            "parents": sum(row["parent_count"] for row in rows),
            "shard_keys_sha256": digest(sorted(
                row["shard_key"] for row in rows)),
        }
        if any(record[field] != value for field, value in expected.items()):
            raise PartitionError("catalog partition/shard denominator differs")
    return document


def validate_catalog(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PartitionError("distributed catalog must be a regular file")
    document = load_json(path, "distributed catalog")
    return validate_catalog_document(document)


def verify_catalog_artifact(catalog: dict[str, Any], artifact_id: str,
                            root: Path) -> dict[str, Any]:
    """Live-verify one fetched artifact against its metadata-only catalog."""
    validate_catalog_document(catalog)
    matches = [row for row in catalog["partitions"]
               if row["artifact_id"] == artifact_id]
    if len(matches) != 1:
        raise PartitionError(f"catalog artifact {artifact_id!r} is not unique")
    record = matches[0]
    if root.is_symlink():
        raise PartitionError("catalog artifact root may not be a symlink")
    root = root.resolve(strict=True)
    manifest = root / "partition-bundle.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise PartitionError(f"artifact {artifact_id} partition manifest is missing")
    manifest_record = record["partition_manifest"]
    if (manifest.stat().st_size != manifest_record["size"] or
            file_sha(manifest) != manifest_record["sha256"]):
        raise PartitionError(f"artifact {artifact_id} partition manifest differs")
    document = verify_partition(manifest, root)
    for field in ("artifact_id", "route", "partition_id", "partition_count",
                  "source_sha", "source_tree", "submodules", "sdk"):
        expected = (catalog["partition_count"] if field == "partition_count"
                    else catalog.get(field, record.get(field)))
        if document.get(field) != expected:
            raise PartitionError(f"artifact {artifact_id} {field} differs")
    expected_shards = sorted(
        (row for row in catalog["shards"]
         if row["artifact_id"] == artifact_id),
        key=lambda row: row["shard_key"])
    observed_shards = sorted(
        (_catalog_shard(document, row) for row in document["shards"]),
        key=lambda row: row["shard_key"])
    if observed_shards != expected_shards:
        raise PartitionError(f"artifact {artifact_id} shard metadata differs")
    return document


def make_catalog(plan_path: Path,
                 manifest_paths: Iterable[Path]) -> dict[str, Any]:
    plan = read_plan(plan_path)
    manifest_paths = list(manifest_paths)
    entries = [(*_manifest_entry(path, plan), path) for path in manifest_paths]
    expected = {(route, partition_id) for route in ROUTES
                for partition_id in range(plan["partition_count"])}
    actual: dict[tuple[str, int], tuple[dict[str, Any], str, Path]] = {}
    for document, manifest_sha, path in entries:
        key = (document["route"], document["partition_id"])
        if key in actual:
            raise PartitionError(f"duplicate partition bundle {key}")
        actual[key] = (document, manifest_sha, path)
    if set(actual) != expected:
        missing = sorted(expected - set(actual))
        foreign = sorted(set(actual) - expected)
        raise PartitionError(
            f"partition bundle union differs missing={missing} foreign={foreign}")
    first = next(iter(actual.values()))[0]
    identity_fields = ("source_sha", "source_tree", "submodules", "sdk")
    for key, (document, _manifest_sha, _path) in actual.items():
        for field in identity_fields:
            if document[field] != first[field]:
                raise PartitionError(f"partition {key} {field} differs")
        if document["partition_plan"]["sha256"] != file_sha(plan_path):
            raise PartitionError(f"partition {key} plan hash differs")
    all_shards = [row for document, _sha_value, _path in actual.values()
                  for row in document["shards"]]
    keys = [row["shard_key"] for row in all_shards]
    expected_keys = [row["shard_key"] for row in plan["shards"]]
    if len(keys) != len(set(keys)) or set(keys) != set(expected_keys):
        raise PartitionError("merged partition shard union has gap/overlap")
    partition_rows = []
    catalog_shards = []
    for key in sorted(actual):
        document, manifest_sha, manifest_path = actual[key]
        partition_rows.append({
            "route": key[0], "partition_id": key[1],
            "artifact_id": document["artifact_id"],
            "partition_manifest": {
                "path": "partition-bundle.json",
                "size": manifest_path.stat().st_size,
                "sha256": manifest_sha,
            },
            "shards": document["denominator"]["shards"],
            "parents": document["denominator"]["parents"],
            "shard_keys_sha256": document["denominator"]["shard_keys_sha256"],
        })
        catalog_shards.extend(
            _catalog_shard(document, row) for row in document["shards"])
    result = {
        "schema": CATALOG_SCHEMA,
        "source_sha": first["source_sha"],
        "source_tree": first["source_tree"],
        "submodules": first["submodules"], "sdk": first["sdk"],
        "partition_plan_sha256": file_sha(plan_path),
        "partition_count": plan["partition_count"],
        "partition_assignment": plan["assignment"],
        "payload_residency": "PER_WORKER_PARTITION_FETCH_AND_VERIFY",
        "denominator": _expected_catalog_denominator(plan),
        "partitions": partition_rows,
        "shards": sorted(catalog_shards, key=lambda row: row["shard_key"]),
    }
    if plan.get("schema") != PLAN_SCHEMA:
        result["selective_scope"] = plan["selective_scope"]
    return validate_catalog_document(result)


def self_test() -> None:
    plan = make_plan(16)
    if (plan["denominator"]["shards"] != 2211 or
            plan["denominator"]["parents"] != 70483 or
            plan["denominator"]["shards_by_route"] != {
                "scalefirst": 892, "fully-quantized": 1319} or
            plan["denominator"]["parents_by_route"] != {
                "scalefirst": 28402, "fully-quantized": 42081}):
        raise PartitionError("live exhaustive denominator differs")
    shard_counts = [row["shards"] for row in plan["partitions"]]
    parent_counts = [row["parents"] for row in plan["partitions"]]
    if max(shard_counts) - min(shard_counts) > 20 or \
            max(parent_counts) - min(parent_counts) > 640:
        raise PartitionError("partition balance differs")
    sf_rows = selection(plan, 0, "scalefirst")
    fq_rows = selection(plan, 0, "fully-quantized")
    if len(selection_tsv(plan, 0, "scalefirst").splitlines()) != len(sf_rows) or \
            len(selection_tsv(plan, 0, "fully-quantized").splitlines()) != len(fq_rows):
        raise PartitionError("selection TSV denominator differs")
    # Four authority mutations must remain RED.
    plants = []
    missing = json.loads(json.dumps(plan))
    missing["shards"].pop()
    plants.append(missing)
    duplicate = json.loads(json.dumps(plan))
    duplicate["shards"][1] = duplicate["shards"][0]
    plants.append(duplicate)
    wrong_owner = json.loads(json.dumps(plan))
    wrong_owner["shards"][0]["partition_id"] ^= 1
    plants.append(wrong_owner)
    wrong_parent = json.loads(json.dumps(plan))
    wrong_parent["shards"][0]["parent_ids_sha256"] = "0" * 64
    plants.append(wrong_parent)
    for planted in plants:
        try:
            validate_plan(planted)
        except PartitionError:
            continue
        raise PartitionError("partition authority negative stayed green")
    print("[kpack-build-partitions:self-test] PASS "
          "parents=70483 shards=2211 sf=28402/892 fq=42081/1319 "
          "partitions=16 exact-disjoint balance=BOUNDED negatives=4-RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    emit = sub.add_parser("emit")
    emit.add_argument("--partitions", type=int, required=True)
    emit.add_argument("--output", type=Path, required=True)
    select = sub.add_parser("select")
    select.add_argument("--plan", type=Path, required=True)
    select.add_argument("--partition", type=int, required=True)
    select.add_argument("--route", choices=ROUTES, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--freeze-plan", type=Path, required=True)
    record = sub.add_parser("record")
    record.add_argument("--root", type=Path, required=True)
    record.add_argument("--route", choices=ROUTES, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--plan", type=Path, required=True)
    merge.add_argument("--partition-manifest", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    validate_catalog_parser = sub.add_parser("validate-catalog")
    validate_catalog_parser.add_argument("--catalog", type=Path, required=True)
    sub.add_parser("self-test")
    args = parser.parse_args()
    try:
        if args.command == "emit":
            write_frozen(args.output, make_plan(args.partitions))
            result = read_plan(args.output)
            print(f"[kpack-build-partitions] PLAN partitions={result['partition_count']} "
                  f"shards={result['denominator']['shards']} "
                  f"parents={result['denominator']['parents']} output={args.output}")
        elif args.command == "select":
            freeze_selection(args.plan, args.freeze_plan, args.output,
                             args.partition, args.route)
            result = read_plan(args.freeze_plan)
            rows = selection(result, args.partition, args.route)
            print(f"[kpack-build-partitions] SELECT route={args.route} "
                  f"partition={args.partition}/{result['partition_count']} "
                  f"shards={len(rows)} parents={sum(r['parent_count'] for r in rows)}")
        elif args.command == "record":
            result = record_partition(args.root, args.route)
            print(f"[kpack-build-partitions] RECORDED artifact={result['artifact_id']} "
                  f"shards={result['denominator']['shards']} "
                  f"parents={result['denominator']['parents']}")
        elif args.command == "verify":
            result = verify_partition(args.manifest, args.root)
            print(f"[kpack-build-partitions] VERIFIED artifact={result['artifact_id']} "
                  f"payloads=LOCAL-HASH-CLEAN")
        elif args.command == "merge":
            result = make_catalog(args.plan, args.partition_manifest)
            write_frozen(args.output, result)
            print(f"[kpack-build-partitions] MERGED "
                  f"partitions={len(result['partitions'])} "
                  f"shards={result['denominator']['shards']} "
                  f"parents={result['denominator']['parents']} "
                  f"payload_residency={result['payload_residency']}")
        elif args.command == "validate-catalog":
            result = validate_catalog(args.catalog)
            print(f"[kpack-build-partitions] CATALOG_PASS "
                  f"partitions={result['partition_count']} "
                  f"artifacts={result['denominator']['partition_artifacts']} "
                  f"shards={result['denominator']['shards']} "
                  f"parents={result['denominator']['parents']}")
        else:
            self_test()
        return 0
    except (KeyError, OSError, PartitionError, subprocess.SubprocessError,
            ValueError) as exc:
        print(f"[kpack-build-partitions] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
