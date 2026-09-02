#!/usr/bin/env python3
"""Compose ScaleFirst and FullyQuantized K-pack discovery bundles.

The component payloads stay in their original directories.  The composite
bundle contains paths relative to the composite root, so it can be handed to
``kpack_discovery_worker_plan.py`` without copying large executables.

Place the output in a common ancestor of both component roots::

  python3 tools/compose_kpack_discovery_bundles.py compose \
      --scalefirst-bundle /root/autodl-tmp/sf/bundle.json \
      --fully-quantized-bundle /root/autodl-tmp/fq/bundle.json \
      --output /root/autodl-tmp/bundle.json

``--scalefirst-root`` and ``--fully-quantized-root`` may explicitly name the
directory against which each source bundle's paths are resolved.  A relative
root is a route prefix below the composite root; an absolute root names the
same directory directly.  Each root must be a strict, non-overlapping child of
the composite root and each input must be that root's ``bundle.json``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from typing import Any, NoReturn

import analyze_scalefirst_kpack_discovery as sf_analyzer
import fully_quantized_kpack_bundle_index as fq_index
import gen_fully_quantized_grouped_kpack_units as fq_grouped_generator
import gen_fully_quantized_kpack_discovery_units as fq_dense_generator
import gen_scalefirst_grouped_kpack_units as sf_grouped_generator
import gen_scalefirst_internal_units as sf_dense_generator
import scalefirst_kpack_binary_shards as sf_shard_plan


SCHEMA = "quactlize.kpack-discovery-composite-bundle.v1"
SF_SCHEMA = "quactlize.scalefirst_kpack_prebuilt_bundle.v2"
FQ_SCHEMA = "quactlize.fully_quantized_kpack_prebuilt_bundle.v2"
SF_RECEIPT_SCHEMA = "quactlize.scalefirst_kpack_binary_receipt.v1"
FQ_RECEIPT_SCHEMA = "quactlize.fully_quantized_kpack_binary_receipt.v2"
SF_BUILD_SCHEMA = "quactlize.scalefirst_kpack_build_input.v1"
FQ_BUILD_SCHEMA = "quactlize.fully_quantized_kpack_build_input.v2"
SF_PROBE_RECEIPT_SCHEMA = (
    "quactlize.scalefirst_kpack_identity_probe_receipt.v1")
FQ_PROBE_RECEIPT_SCHEMA = "quactlize.fq_kpack_identity_probe_receipt.v1"
ROUTES = ("fully-quantized", "scalefirst")
OPERATORS = {"dense", "grouped"}
QTYPES = {10, 11, 12, 13, 14}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_OID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


class BundleError(ValueError):
    """An input cannot safely participate in a composite bundle."""


def fail(message: str) -> NoReturn:
    raise SystemExit(f"kpack discovery bundle compose: {message}")


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BundleError(f"value is not canonical JSON: {exc}") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        raise BundleError(f"cannot hash {path}: {exc}") from exc


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleError(f"{label} must be a JSON object")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BundleError(f"{label} must be a nonempty string")
    if value != value.strip():
        raise BundleError(f"{label} may not have surrounding whitespace")
    if any(mark in value for mark in ("\0", "\n", "\r")):
        raise BundleError(f"{label} must be one line")
    return value


def _sha256(value: Any, label: str) -> str:
    value = _string(value, label)
    if not SHA256_RE.fullmatch(value):
        raise BundleError(f"{label} must be one lowercase SHA-256")
    return value


def _git_oid(value: Any, label: str) -> str:
    value = _string(value, label)
    if not GIT_OID_RE.fullmatch(value):
        raise BundleError(f"{label} must be one lowercase Git object ID")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BundleError(f"{label} must be an integer")
    return value


def _relative_path(value: Any, label: str) -> PurePosixPath:
    raw = _string(value, label)
    if "\\" in raw:
        raise BundleError(f"{label} must use POSIX separators")
    path = PurePosixPath(raw)
    if path.is_absolute() or raw != path.as_posix() or any(
            part in ("", ".", "..") for part in path.parts):
        raise BundleError(f"{label} must be a normalized relative path")
    return path


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise BundleError(f"{label} may not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise BundleError(f"cannot resolve {label} {path}: {exc}") from exc
    if not resolved.is_file():
        raise BundleError(f"{label} is not a regular file: {path}")
    return resolved


def _directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise BundleError(f"{label} may not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise BundleError(f"cannot resolve {label} {path}: {exc}") from exc
    if not resolved.is_dir():
        raise BundleError(f"{label} is not a directory: {path}")
    return resolved


def _within(path: Path, root: Path, label: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise BundleError(f"{label} escapes {root}: {path}") from exc
    return path


def _component_file(root: Path, value: Any, label: str) -> Path:
    relative = _relative_path(value, label)
    candidate = root.joinpath(*relative.parts)
    resolved = _regular_file(candidate, label)
    return _within(resolved, root, label)


def _composite_relative(path: Path, composite_root: Path, label: str) -> str:
    return _within(path, composite_root, label).relative_to(
        composite_root).as_posix()


def _checked_file_record(root: Path, composite_root: Path, path_value: Any,
                         sha_value: Any, label: str) -> dict[str, str]:
    path = _component_file(root, path_value, label)
    declared = _sha256(sha_value, f"{label} SHA-256")
    observed = _file_sha(path)
    if observed != declared:
        raise BundleError(
            f"{label} SHA-256 differs: declared={declared} observed={observed}")
    return {
        "path": _composite_relative(path, composite_root, label),
        "sha256": observed,
    }


def _validate_sdk_file(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise BundleError(f"{label} must be an object")
    if set(value) != {"path", "size", "sha256", "symlink_target"}:
        raise BundleError(f"{label} fields differ")
    _relative_path(value.get("path"), f"{label}.path")
    _sha256(value.get("sha256"), f"{label}.sha256")
    size = _integer(value.get("size"), f"{label}.size")
    if size < 0:
        raise BundleError(f"{label}.size must be nonnegative")
    target = value.get("symlink_target")
    if target is not None:
        _string(target, f"{label}.symlink_target")


def _validate_sdk(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BundleError(f"{label} must be an object")
    for field in ("receipt", "compiler", "inspector"):
        _validate_sdk_file(value.get(field), f"{label}.{field}")
    runtime = value.get("runtime_libraries")
    if not isinstance(runtime, list) or not runtime:
        raise BundleError(f"{label}.runtime_libraries must be nonempty")
    runtime_paths: set[str] = set()
    for index, entry in enumerate(runtime):
        _validate_sdk_file(entry, f"{label}.runtime_libraries[{index}]")
        path = entry["path"]
        if path in runtime_paths:
            raise BundleError(f"{label} contains duplicate runtime path {path}")
        runtime_paths.add(path)
    _canonical(value)
    return copy.deepcopy(value)


def _validate_submodules(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise BundleError(f"{label} must be an array")
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise BundleError(f"{label}[{index}] must be an object")
        if set(row) != {"path", "gitlink", "current"}:
            raise BundleError(f"{label}[{index}] fields differ")
        path = _string(row.get("path"), f"{label}[{index}].path")
        _relative_path(path, f"{label}[{index}].path")
        if path in seen:
            raise BundleError(f"{label} contains duplicate path {path}")
        seen.add(path)
        gitlink = _git_oid(
            row.get("gitlink"), f"{label}[{index}].gitlink")
        current = _git_oid(
            row.get("current"), f"{label}[{index}].current")
        if gitlink != current:
            raise BundleError(f"{label}[{index}] gitlink/current differ")
        result.append(copy.deepcopy(row))
    _canonical(value)
    return result


def _bundle_identity(document: dict[str, Any], route: str) -> dict[str, Any]:
    if route == "scalefirst":
        if document.get("schema") != SF_SCHEMA:
            raise BundleError("ScaleFirst bundle schema differs")
        if document.get("route") != route:
            raise BundleError("ScaleFirst bundle route differs")
        scope = document.get("scope")
        if scope not in ("full", "pilot"):
            raise BundleError("ScaleFirst bundle scope differs")
        repository = document.get("repository")
        if not isinstance(repository, dict):
            raise BundleError("ScaleFirst repository authority is malformed")
        tree = repository.get("tree")
        submodules = repository.get("submodules")
        mode = scope.upper()
    else:
        if document.get("schema") != FQ_SCHEMA:
            raise BundleError("FullyQuantized bundle schema differs")
        if "route" in document and document["route"] != route:
            raise BundleError("FullyQuantized bundle route differs")
        mode = document.get("mode")
        if mode not in ("FULL", "PILOT"):
            raise BundleError("FullyQuantized bundle mode differs")
        tree = document.get("source_tree")
        submodules = document.get("submodules")
        try:
            fq_index.validate_index(document)
        except (KeyError, TypeError, ValueError) as exc:
            raise BundleError(
                f"FullyQuantized native bundle authority differs: {exc}") from exc
    return {
        "mode": mode,
        "source_sha": _git_oid(document.get("source_sha"),
                               f"{route} source_sha"),
        "source_tree": _git_oid(tree, f"{route} source tree"),
        "submodules": _validate_submodules(
            submodules, f"{route} submodules"),
        "sdk": _validate_sdk(document.get("sdk"), f"{route} sdk"),
    }


def _normalize_shards(document: dict[str, Any], route: str
                      ) -> list[tuple[str, dict[str, Any]]]:
    raw = document.get("shards")
    if isinstance(raw, dict) and raw:
        rows = list(raw.items())
    elif isinstance(raw, list) and raw:
        rows = [(None, row) for row in raw]
    else:
        raise BundleError(f"{route} shards are empty or malformed")
    result: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for index, (mapping_key, row) in enumerate(rows):
        if not isinstance(row, dict):
            raise BundleError(f"{route} shard {index} must be an object")
        identities = [row[field] for field in ("shard_id", "shard_key")
                      if field in row]
        if not identities:
            raise BundleError(f"{route} shard {index} has no identity")
        identity = _string(identities[0], f"{route} shard {index} identity")
        if any(_string(value, f"{route} shard {index} identity") != identity
               for value in identities[1:]):
            raise BundleError(f"{route} shard {index} identities disagree")
        if mapping_key is not None and _string(
                mapping_key, f"{route} shard mapping key") != identity:
            raise BundleError(f"{route} shard mapping key differs from row")
        if identity in seen:
            raise BundleError(f"{route} contains duplicate shard key {identity}")
        seen.add(identity)
        result.append((identity, row))
    return sorted(result, key=lambda item: item[0])


def _parent_id(value: Any, label: str) -> int | str:
    if isinstance(value, bool):
        raise BundleError(f"{label} must be an integer or string")
    if isinstance(value, int):
        return value
    return _string(value, label)


def _build_authority(document: dict[str, Any], route: str, root: Path,
                     composite_root: Path, source: dict[str, Any]
                     ) -> tuple[dict[str, str], dict[str, Any]]:
    if route == "scalefirst":
        value = document.get("build_input_authority")
        if not isinstance(value, dict):
            raise BundleError("ScaleFirst build authority is malformed")
        record = _checked_file_record(
            root, composite_root, value.get("path"), value.get("sha256"),
            "ScaleFirst build authority")
        schema = SF_BUILD_SCHEMA
    else:
        record = _checked_file_record(
            root, composite_root, document.get("build_input_authority"),
            document.get("build_input_authority_sha256"),
            "FullyQuantized build authority")
        schema = FQ_BUILD_SCHEMA
    authority = _load_json(
        composite_root / record["path"], f"{route} build authority")
    if authority.get("schema") != schema:
        raise BundleError(f"{route} build authority schema differs")
    for field in ("source_sha", "source_tree", "submodules", "sdk"):
        if authority.get(field) != source[field]:
            raise BundleError(f"{route} build authority {field} differs")
    if not isinstance(authority.get("configuration"), dict):
        raise BundleError(f"{route} build authority configuration is malformed")
    return record, authority


def _probe_record(document: dict[str, Any], route: str, root: Path,
                  composite_root: Path,
                  build_authority_sha: str) -> dict[str, Any]:
    if route == "scalefirst":
        raw = document.get("runtime_probe")
        binary_field, binary_sha_field = "binary", "binary_sha256"
        expected_schema = SF_PROBE_RECEIPT_SCHEMA
    else:
        raw = document.get("runtime_identity_probe")
        binary_field, binary_sha_field = "path", "sha256"
        expected_schema = FQ_PROBE_RECEIPT_SCHEMA
    if not isinstance(raw, dict):
        raise BundleError(f"{route} runtime probe is malformed")
    binary = _checked_file_record(
        root, composite_root, raw.get(binary_field),
        raw.get(binary_sha_field), f"{route} runtime probe binary")
    receipt = _checked_file_record(
        root, composite_root, raw.get("receipt"),
        raw.get("receipt_sha256"), f"{route} runtime probe receipt")
    receipt_doc = _load_json(
        composite_root / receipt["path"], f"{route} runtime probe receipt")
    if receipt_doc.get("schema") != expected_schema:
        raise BundleError(f"{route} runtime probe receipt schema differs")
    if (receipt_doc.get("build_input_authority_sha256") !=
            build_authority_sha or
            receipt_doc.get("binary_sha256") != binary["sha256"]):
        raise BundleError(f"{route} runtime probe receipt chain differs")
    result: dict[str, Any] = {"binary": binary, "receipt": receipt}
    for field in ("host_machine",):
        if field in raw:
            result[field] = copy.deepcopy(raw[field])
    return result


def _optional_shard_plan(document: dict[str, Any], route: str, root: Path,
                         composite_root: Path,
                         build_authority: dict[str, Any]
                         ) -> tuple[dict[str, str] | None,
                                    dict[str, dict[str, Any]] | None]:
    if route != "scalefirst":
        configuration = build_authority["configuration"]
        if (configuration.get("mode") != document.get("mode") or
                configuration.get("max_parents_per_binary") !=
                document.get("max_parents_per_binary") or
                configuration.get("scratch_policy") !=
                "ONE_PARENT_RANGE_THEN_COMPACT_PAYLOAD"):
            raise BundleError(
                "FullyQuantized build configuration differs from bundle")
        return None, None
    raw = document.get("shard_plan")
    if not isinstance(raw, dict):
        raise BundleError("ScaleFirst shard plan is malformed")
    record = _checked_file_record(
        root, composite_root, raw.get("path"), raw.get("sha256"),
        "ScaleFirst shard plan")
    plan = _load_json(
        composite_root / record["path"], "ScaleFirst shard plan")
    try:
        sf_shard_plan.validate_plan(plan)
    except (KeyError, TypeError, ValueError) as exc:
        raise BundleError(
            f"ScaleFirst native shard plan differs: {exc}") from exc
    if (document.get("scope") != plan.get("scope") or
            document.get("parents_per_binary") !=
            plan.get("parents_per_binary") or
            raw.get("pairs") != plan.get("pairs")):
        raise BundleError("ScaleFirst bundle/shard-plan authority differs")
    configuration = build_authority["configuration"]
    if (configuration.get("scope") != plan["scope"] or
            configuration.get("parents_per_binary") !=
            plan["parents_per_binary"] or
            configuration.get("shard_plan_sha256") != record["sha256"] or
            configuration.get("scratch_policy") !=
            "ONE_SHARD_THEN_COMPACT_PAYLOAD"):
        raise BundleError("ScaleFirst build/shard-plan authority differs")
    return record, {row["shard_id"]: row for row in plan["shards"]}


def _validate_binary_receipt(route: str, receipt: dict[str, Any],
                             row: dict[str, Any], manifest_sha: str,
                             binary_sha: str, build_authority_sha: str,
                             source: dict[str, Any]) -> None:
    schema = SF_RECEIPT_SCHEMA if route == "scalefirst" else FQ_RECEIPT_SCHEMA
    if receipt.get("schema") != schema:
        raise BundleError(f"{route} binary receipt schema differs")
    if (receipt.get("manifest_sha256") != manifest_sha or
            receipt.get("binary_sha256") != binary_sha or
            receipt.get("build_input_authority_sha256") !=
            build_authority_sha):
        raise BundleError(f"{route} binary receipt payload chain differs")
    if route == "fully-quantized":
        for field in ("shard_key", "qtype", "operator", "route",
                      "parent_begin", "parent_end", "parent_count",
                      "authority_count", "parent_ids"):
            if receipt.get(field) != row.get(field):
                raise BundleError(
                    f"FullyQuantized binary receipt stale field {field}")
        if (receipt.get("source_sha") != source["source_sha"] or
                receipt.get("source_tree") != source["source_tree"] or
                receipt.get("submodules") != source["submodules"] or
                receipt.get("sdk_compiler_sha256") !=
                source["sdk"]["compiler"]["sha256"] or
                receipt.get("sdk_inspector_sha256") !=
                source["sdk"]["inspector"]["sha256"]):
            raise BundleError(
                "FullyQuantized binary receipt source/SDK authority differs")
        if (receipt.get("manifest") != row.get("manifest") or
                receipt.get("binary") != row.get("binary") or
                receipt.get("device_arch") != row.get("device_arch") or
                receipt.get("inspector_output_sha256") !=
                row.get("inspector_output_sha256")):
            raise BundleError(
                "FullyQuantized binary receipt path/image authority differs")


def _component(document: dict[str, Any], route: str, root: Path,
               bundle_path: Path, composite_root: Path,
               source: dict[str, Any]) -> tuple[dict[str, Any],
                                                list[dict[str, Any]]]:
    bundle_sha = _file_sha(bundle_path)
    build, build_document = _build_authority(
        document, route, root, composite_root, source)
    probe = _probe_record(
        document, route, root, composite_root, build["sha256"])
    shard_plan, sf_planned = _optional_shard_plan(
        document, route, root, composite_root, build_document)
    component_root = _composite_relative(root, composite_root,
                                         f"{route} root")
    rewritten: list[dict[str, Any]] = []
    coverage: dict[tuple[int, str], list[tuple[int, int, int]]] = {}
    native_shards = _normalize_shards(document, route)
    if sf_planned is not None and {key for key, _row in native_shards} != set(
            sf_planned):
        raise BundleError("ScaleFirst bundle shard union differs from plan")
    width_field = ("parents_per_binary" if route == "scalefirst" else
                   "max_parents_per_binary")
    width = _integer(document.get(width_field), f"{route} {width_field}")
    if not 1 <= width <= 32:
        raise BundleError(f"{route} parent width must be in [1,32]")
    for shard_key, raw in native_shards:
        if raw.get("route") != route:
            raise BundleError(f"{shard_key} route differs from {route}")
        qtype = _integer(raw.get("qtype"), f"{shard_key}.qtype")
        operator = _string(raw.get("operator"), f"{shard_key}.operator")
        if qtype not in QTYPES or operator not in OPERATORS:
            raise BundleError(f"{shard_key} qtype/operator is unsupported")
        expected_layout = sf_analyzer.LAYOUT[qtype]
        if raw.get("layout") != expected_layout:
            raise BundleError(f"{shard_key} canonical layout differs")
        expected_mapping = sf_analyzer.MAPPING[expected_layout]
        begin = _integer(raw.get("parent_begin"), f"{shard_key}.parent_begin")
        end = _integer(raw.get("parent_end"), f"{shard_key}.parent_end")
        authority_field = ("authority_parents" if route == "scalefirst"
                           else "authority_count")
        authority = _integer(
            raw.get(authority_field), f"{shard_key}.{authority_field}")
        if begin < 0 or end <= begin or end > authority or end - begin > width:
            raise BundleError(f"{shard_key} parent range is invalid")
        parents_raw = raw.get("parent_ids")
        if not isinstance(parents_raw, list) or not parents_raw:
            raise BundleError(f"{shard_key}.parent_ids are empty/malformed")
        parents = [_parent_id(value, f"{shard_key}.parent_ids")
                   for value in parents_raw]
        if len(parents) != end - begin or len({(type(x), x) for x in parents}) != len(parents):
            raise BundleError(f"{shard_key} parent identity count differs")
        if route == "scalefirst" and parents != list(range(begin, end)):
            raise BundleError(f"{shard_key} ScaleFirst parent IDs differ")
        if route == "fully-quantized":
            count = _integer(raw.get("parent_count"),
                             f"{shard_key}.parent_count")
            if count != end - begin:
                raise BundleError(f"{shard_key} parent_count differs")
        else:
            planned = sf_planned[shard_key] if sf_planned is not None else None
            if planned is None or any(raw.get(field) != planned[field] for field in (
                    "shard_id", "qtype", "operator", "layout",
                    "parent_begin", "parent_end", "authority_parents")) or \
                    planned["compiled_parents"] != end - begin:
                raise BundleError(f"{shard_key} differs from ScaleFirst plan")
            if raw.get("mapping_id") != expected_mapping:
                raise BundleError(f"{shard_key} ScaleFirst mapping differs")
            if raw.get("parent_symbols") != sf_shard_plan.authority_symbols(
                    operator, qtype)[begin:end]:
                raise BundleError(
                    f"{shard_key} ScaleFirst parent symbols differ")
        coverage.setdefault((qtype, operator), []).append(
            (begin, end, authority))

        manifest = _checked_file_record(
            root, composite_root, raw.get("manifest"),
            raw.get("manifest_sha256"), f"{shard_key} manifest")
        binary = _checked_file_record(
            root, composite_root, raw.get("binary"),
            raw.get("binary_sha256"), f"{shard_key} binary")
        receipt = _checked_file_record(
            root, composite_root, raw.get("binary_receipt"),
            raw.get("binary_receipt_sha256"),
            f"{shard_key} binary receipt")
        receipt_doc = _load_json(
            composite_root / receipt["path"], f"{shard_key} binary receipt")
        manifest_path = composite_root / manifest["path"]
        try:
            if route == "scalefirst":
                parsed_manifest = sf_analyzer.validate_manifest(
                    operator, qtype, manifest_path)
                compiled = parsed_manifest["compiled_parents"]
                if ([row["parent_id"] for row in compiled] != parents or
                        [row["symbol"] for row in compiled] !=
                        raw.get("parent_symbols") or
                        parsed_manifest["parent_range"] != {
                            "begin": begin, "end": end,
                            "count": end - begin,
                            "authority_count": authority}):
                    raise BundleError(
                        f"{shard_key} ScaleFirst manifest parent authority differs")
            else:
                parsed_manifest = _load_json(
                    manifest_path, f"{shard_key} manifest")
                validator = (fq_dense_generator.validate_manifest
                             if operator == "dense" else
                             fq_grouped_generator.validate_manifest)
                validator(parsed_manifest)
                field = ("dense_tc_parents" if operator == "dense" else
                         "grouped_parents")
                compiled = parsed_manifest[field]
                if ([row["static_candidate_id"] for row in compiled] != parents or
                        [row["parent_ordinal"] for row in compiled] !=
                        list(range(begin, end)) or
                        parsed_manifest["parent_range"] != {
                            "begin": begin, "end": end,
                            "count": end - begin,
                            "authority_count": authority}):
                    raise BundleError(
                        f"{shard_key} FullyQuantized manifest parent authority differs")
        except BundleError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise BundleError(
                f"{shard_key} native manifest authority differs: {exc}") from exc
        _validate_binary_receipt(
            route, receipt_doc, raw, manifest["sha256"], binary["sha256"],
            build["sha256"], source)
        row = {
            "shard_key": f"{route}:{shard_key}",
            "native_shard_key": shard_key,
            "component": route,
            "route": route,
            "qtype": qtype,
            "operator": operator,
            "layout": expected_layout,
            "mapping_id": expected_mapping,
            "parent_begin": begin,
            "parent_end": end,
            "authority_count": authority,
            "parent_ids": copy.deepcopy(parents),
            "manifest_sha256": manifest["sha256"],
            "files": {
                "manifest": manifest,
                "binary": binary,
                "binary_receipt": receipt,
            },
        }
        rewritten.append(row)

    for identity, ranges in coverage.items():
        ranges.sort()
        authority_values = {row[2] for row in ranges}
        if len(authority_values) != 1 or ranges[0][0] != 0:
            raise BundleError(
                f"{route} q{identity[0]}/{identity[1]} range authority differs")
        for previous, current in zip(ranges, ranges[1:]):
            if previous[1] != current[0]:
                raise BundleError(
                    f"{route} q{identity[0]}/{identity[1]} has a range gap/overlap")
        if source["mode"] == "FULL" and ranges[-1][1] != ranges[-1][2]:
            raise BundleError(
                f"{route} q{identity[0]}/{identity[1]} full range has a tail gap")
        if source["mode"] == "PILOT" and len(ranges) != 1:
            raise BundleError(
                f"{route} q{identity[0]}/{identity[1]} pilot has multiple ranges")
    expected_pairs = ({(qtype, operator) for qtype in QTYPES
                       for operator in OPERATORS}
                      if source["mode"] == "FULL" else
                      {(10, operator) for operator in OPERATORS})
    if set(coverage) != expected_pairs:
        raise BundleError(f"{route} {source['mode']} pair denominator differs")

    keys = [row["shard_key"] for row in rewritten]
    component: dict[str, Any] = {
        "schema": document["schema"],
        "root": component_root,
        "bundle": _composite_relative(
            bundle_path, composite_root, f"{route} bundle"),
        "bundle_sha256": bundle_sha,
        "build_input_authority": build,
        "runtime_probe": probe,
        "shard_count": len(rewritten),
        "shard_keys_sha256": _digest(sorted(keys)),
    }
    if shard_plan is not None:
        component["shard_plan"] = shard_plan
    return component, rewritten


def _resolve_inputs(output: Path, scalefirst_bundle: Path,
                    fully_quantized_bundle: Path,
                    scalefirst_root: Path | None,
                    fully_quantized_root: Path | None
                    ) -> tuple[Path, Path, Path, Path, Path]:
    if output.name != "bundle.json":
        raise BundleError("composite output must be named bundle.json")
    if not output.parent.exists():
        raise BundleError("composite output parent must already exist")
    composite_root = _directory(output.parent, "composite root")
    if output.exists() and (output.is_symlink() or not output.is_file()):
        raise BundleError("existing composite output is not one regular file")
    sf_bundle = _regular_file(scalefirst_bundle, "ScaleFirst bundle")
    fq_bundle = _regular_file(fully_quantized_bundle, "FullyQuantized bundle")
    sf_root_arg = (sf_bundle.parent if scalefirst_root is None else
                   (scalefirst_root if scalefirst_root.is_absolute() else
                    composite_root / scalefirst_root))
    fq_root_arg = (fq_bundle.parent if fully_quantized_root is None else
                   (fully_quantized_root if fully_quantized_root.is_absolute()
                    else composite_root / fully_quantized_root))
    sf_root = _directory(sf_root_arg, "ScaleFirst root")
    fq_root = _directory(fq_root_arg, "FullyQuantized root")
    if sf_bundle.parent != sf_root or sf_bundle.name != "bundle.json":
        raise BundleError("ScaleFirst input must be its root's bundle.json")
    if fq_bundle.parent != fq_root or fq_bundle.name != "bundle.json":
        raise BundleError("FullyQuantized input must be its root's bundle.json")
    for route, root in (("scalefirst", sf_root),
                        ("fully-quantized", fq_root)):
        _within(root, composite_root, f"{route} root")
        if root == composite_root:
            raise BundleError(f"{route} root must be a strict composite child")
    if sf_root == fq_root or sf_root in fq_root.parents or fq_root in sf_root.parents:
        raise BundleError("component roots must be distinct and non-overlapping")
    return composite_root, sf_bundle, fq_bundle, sf_root, fq_root


def compose_document(*, output: Path, scalefirst_bundle: Path,
                     fully_quantized_bundle: Path,
                     scalefirst_root: Path | None = None,
                     fully_quantized_root: Path | None = None
                     ) -> dict[str, Any]:
    (composite_root, sf_path, fq_path, sf_root,
     fq_root) = _resolve_inputs(
         output, scalefirst_bundle, fully_quantized_bundle,
         scalefirst_root, fully_quantized_root)
    sf = _load_json(sf_path, "ScaleFirst bundle")
    fq = _load_json(fq_path, "FullyQuantized bundle")
    sf_identity = _bundle_identity(sf, "scalefirst")
    fq_identity = _bundle_identity(fq, "fully-quantized")
    for field in ("mode", "source_sha", "source_tree", "submodules", "sdk"):
        if sf_identity[field] != fq_identity[field]:
            raise BundleError(f"component {field} authorities differ")

    sf_component, sf_shards = _component(
        sf, "scalefirst", sf_root, sf_path, composite_root, sf_identity)
    fq_component, fq_shards = _component(
        fq, "fully-quantized", fq_root, fq_path, composite_root, fq_identity)
    shards = sf_shards + fq_shards
    keys: set[str] = set()
    parents: set[tuple[str, str, int, type, int | str]] = set()
    for row in shards:
        key = row["shard_key"]
        if key in keys:
            raise BundleError(f"component bundles collide on shard key {key}")
        keys.add(key)
        for parent in row["parent_ids"]:
            identity = (row["route"], row["operator"], row["qtype"],
                        type(parent), parent)
            if identity in parents:
                raise BundleError(
                    f"component bundles contain duplicate parent identity {identity}")
            parents.add(identity)
    shards.sort(key=lambda row: (
        row["route"], row["qtype"], row["operator"],
        row["shard_key"]))
    by_route = {
        route: sum(row["route"] == route for row in shards)
        for route in ROUTES
    }
    parents_by_route = {
        route: sum(len(row["parent_ids"]) for row in shards
                   if row["route"] == route)
        for route in ROUTES
    }
    return {
        "schema": SCHEMA,
        "mode": sf_identity["mode"],
        "source_sha": sf_identity["source_sha"],
        "source_tree": sf_identity["source_tree"],
        "submodules": sf_identity["submodules"],
        "sdk": sf_identity["sdk"],
        "component_bundles": {
            "scalefirst": sf_component,
            "fully-quantized": fq_component,
        },
        "path_contract": {
            "composite_files": "shards[].files.*.path",
            "composite_files_base": "COMPOSITE_ROOT",
            "native_rows": "LOOKUP_COMPONENT_BUNDLE_BY_NATIVE_SHARD_KEY",
            "native_paths_base": "COMPONENT_ROOT",
            "receipts_are_immutable": True,
        },
        "denominator": {
            "routes": len(ROUTES),
            "shards": len(shards),
            "shards_by_route": by_route,
            "parents": sum(parents_by_route.values()),
            "parents_by_route": parents_by_route,
        },
        "shards": shards,
    }


def _encoded(document: dict[str, Any]) -> str:
    return json.dumps(
        document, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_composite(path: Path, document: dict[str, Any]) -> None:
    encoded = _encoded(document)
    if path.exists():
        try:
            previous = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise BundleError(f"cannot read existing output {path}: {exc}") from exc
        if previous != encoded:
            raise BundleError(f"refusing to replace stale output {path}")
        return
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise BundleError(f"cannot write composite output {path}: {exc}") from exc


def validate_composite(path: Path) -> dict[str, Any]:
    path = _regular_file(path, "composite bundle")
    document = _load_json(path, "composite bundle")
    if document.get("schema") != SCHEMA:
        raise BundleError("composite bundle schema differs")
    components = document.get("component_bundles")
    if not isinstance(components, dict) or set(components) != set(ROUTES):
        raise BundleError("composite component bundle set differs")
    root = path.parent
    inputs: dict[str, tuple[Path, Path]] = {}
    for route in ROUTES:
        record = components[route]
        if not isinstance(record, dict):
            raise BundleError(f"composite {route} component is malformed")
        component_root_rel = _relative_path(
            record.get("root"), f"composite {route} root")
        component_root = _directory(
            root.joinpath(*component_root_rel.parts), f"composite {route} root")
        _within(component_root, root, f"composite {route} root")
        bundle = _component_file(
            root, record.get("bundle"), f"composite {route} bundle")
        if bundle.parent != component_root:
            raise BundleError(f"composite {route} bundle/root differ")
        declared = _sha256(
            record.get("bundle_sha256"), f"composite {route} bundle SHA-256")
        if _file_sha(bundle) != declared:
            raise BundleError(f"composite {route} original bundle is stale")
        inputs[route] = (bundle, component_root)
    expected = compose_document(
        output=path,
        scalefirst_bundle=inputs["scalefirst"][0],
        fully_quantized_bundle=inputs["fully-quantized"][0],
        scalefirst_root=inputs["scalefirst"][1],
        fully_quantized_root=inputs["fully-quantized"][1])
    if document != expected:
        raise BundleError("composite bundle differs from live component authorities")
    return document


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    sf_root = root / "scalefirst"
    fq_root = root / "fully-quantized"
    sf_root.mkdir(parents=True)
    fq_root.mkdir(parents=True)
    output = root / "bundle.json"
    source_sha = "1" * 40
    source_tree = "2" * 40
    submodules = [{"path": "third_party/actlize", "gitlink": "3" * 40,
                   "current": "3" * 40}]

    def sdk_file(path: str, marker: str) -> dict[str, Any]:
        return {"path": path, "size": 1, "sha256": marker * 64,
                "symlink_target": None}

    sdk = {
        "receipt": sdk_file("VERSION.txt", "4"),
        "compiler": sdk_file("bin/hgcc", "5"),
        "inspector": sdk_file("bin/hgobjdump", "6"),
        "runtime_libraries": [sdk_file("lib/libhggcrt.so", "7")],
    }

    def write(path: Path, data: bytes) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return _file_sha(path)

    def write_json(path: Path, value: dict[str, Any]) -> str:
        return write(path, _encoded(value).encode("utf-8"))

    sf_plan = sf_shard_plan.make_plan("pilot", 32)
    sf_plan_rel = "shard-plan.json"
    sf_plan_sha = write_json(sf_root / sf_plan_rel, sf_plan)
    sf_build_rel = "build-input-authority.json"
    fq_build_rel = "inputs/build-input-authority.json"
    sf_build_sha = write_json(sf_root / sf_build_rel, {
        "schema": SF_BUILD_SCHEMA,
        "source_sha": source_sha,
        "source_tree": source_tree,
        "submodules": submodules,
        "sdk": sdk,
        "configuration": {
            "scope": "pilot",
            "parents_per_binary": 32,
            "shard_plan_sha256": sf_plan_sha,
            "scratch_policy": "ONE_SHARD_THEN_COMPACT_PAYLOAD",
        },
    })
    fq_build_sha = write_json(fq_root / fq_build_rel, {
        "schema": FQ_BUILD_SCHEMA,
        "source_sha": source_sha,
        "source_tree": source_tree,
        "submodules": submodules,
        "sdk": sdk,
        "configuration": {
            "mode": "PILOT",
            "max_parents_per_binary": 32,
            "scratch_policy": "ONE_PARENT_RANGE_THEN_COMPACT_PAYLOAD",
        },
    })

    def probe(route_root: Path, route: str, build_sha: str
              ) -> tuple[dict[str, Any], str]:
        binary_rel = "payloads/support/box_identity_probe"
        receipt_rel = "payloads/support/identity-probe-receipt.json"
        binary_sha = write(route_root / binary_rel, b"probe\n")
        schema = (SF_PROBE_RECEIPT_SCHEMA if route == "scalefirst"
                  else FQ_PROBE_RECEIPT_SCHEMA)
        receipt_sha = write_json(route_root / receipt_rel, {
            "schema": schema,
            "build_input_authority_sha256": build_sha,
            "source_sha256": "8" * 64,
            "binary_sha256": binary_sha,
        })
        if route == "scalefirst":
            return ({"binary": binary_rel, "binary_sha256": binary_sha,
                     "receipt": receipt_rel,
                     "receipt_sha256": receipt_sha,
                     "host_machine": "Advanced Micro Devices X86-64"},
                    binary_sha)
        return ({"path": binary_rel, "sha256": binary_sha,
                 "receipt": receipt_rel, "receipt_sha256": receipt_sha},
                binary_sha)

    sf_probe, _ = probe(sf_root, "scalefirst", sf_build_sha)
    fq_probe, _ = probe(fq_root, "fully-quantized", fq_build_sha)
    sf_shards: list[dict[str, Any]] = []
    for planned in sf_plan["shards"]:
        operator = planned["operator"]
        sf_key = planned["shard_id"]
        generated = sf_root / "generated" / sf_key
        if operator == "dense":
            sf_dense_generator.generate(
                10, 0, 0, generated, 4, False, None,
                sf_analyzer.LAYOUT[10], planned["parent_begin"],
                planned["compiled_parents"])
        else:
            sf_grouped_generator.generate(
                10, generated, 4, False, None,
                planned["parent_begin"], planned["compiled_parents"])
        manifest_rel = f"generated/{sf_key}/manifest.json"
        manifest_sha = _file_sha(sf_root / manifest_rel)
        binary_rel = f"payloads/{sf_key}/kernel"
        binary_sha = write(sf_root / binary_rel, b"binary\n")
        receipt_rel = f"payloads/{sf_key}/binary-receipt.json"
        sf_receipt_sha = write_json(sf_root / receipt_rel, {
            "schema": SF_RECEIPT_SCHEMA,
            "build_input_authority_sha256": sf_build_sha,
            "manifest_sha256": manifest_sha,
            "binary_sha256": binary_sha,
        })
        parsed = sf_analyzer.validate_manifest(
            operator, 10, sf_root / manifest_rel)
        sf_shards.append({
            "shard_id": sf_key, "route": "scalefirst", "qtype": 10,
            "operator": operator, "layout": sf_analyzer.LAYOUT[10],
            "mapping_id": sf_analyzer.MAPPING[sf_analyzer.LAYOUT[10]],
            "parent_begin": planned["parent_begin"],
            "parent_end": planned["parent_end"],
            "parent_ids": [row["parent_id"]
                           for row in parsed["compiled_parents"]],
            "parent_symbols": [row["symbol"]
                               for row in parsed["compiled_parents"]],
            "authority_parents": planned["authority_parents"],
            "manifest": manifest_rel, "manifest_sha256": manifest_sha,
            "binary": binary_rel, "binary_sha256": binary_sha,
            "binary_receipt": receipt_rel,
            "binary_receipt_sha256": sf_receipt_sha,
        })

    fq_shards: dict[str, dict[str, Any]] = {}
    for planned in fq_index.plan(True, 32):
        operator = planned["operator"]
        fq_key = planned["shard_key"]
        generated = fq_root / "generated" / fq_key
        if operator == "dense":
            manifest_doc = fq_dense_generator.generate(
                10, generated, 4, parent_begin=planned["parent_begin"],
                parent_count=planned["parent_count"])
            rows = manifest_doc["dense_tc_parents"]
        else:
            manifest_doc = fq_grouped_generator.generate(
                10, generated, 4, parent_begin=planned["parent_begin"],
                parent_count=planned["parent_count"])
            rows = manifest_doc["grouped_parents"]
        manifest_rel = f"generated/{fq_key}/manifest.json"
        manifest_sha = _file_sha(fq_root / manifest_rel)
        binary_rel = f"payloads/{fq_key}/kernel"
        binary_sha = write(fq_root / binary_rel, b"binary\n")
        receipt_rel = f"payloads/{fq_key}/binary-receipt.json"
        fq_row = {**planned, "typed_rows": len(rows),
            "manifest": manifest_rel, "manifest_sha256": manifest_sha,
            "binary": binary_rel, "binary_sha256": binary_sha,
            "binary_receipt": receipt_rel,
            "device_arch": "PPU ppu0010",
            "inspector_output_sha256": "9" * 64,
        }
        fq_receipt_sha = write_json(fq_root / receipt_rel, {
            "schema": FQ_RECEIPT_SCHEMA,
            **{field: fq_row[field] for field in (
                "shard_key", "qtype", "operator", "route", "parent_begin",
                "parent_end", "parent_count", "authority_count",
                "parent_ids")},
            "build_input_authority_sha256": fq_build_sha,
            "source_sha": source_sha, "source_tree": source_tree,
            "submodules": submodules,
            "sdk_compiler_sha256": sdk["compiler"]["sha256"],
            "sdk_inspector_sha256": sdk["inspector"]["sha256"],
            "manifest": manifest_rel,
            "manifest_sha256": manifest_sha,
            "binary": binary_rel,
            "binary_sha256": binary_sha,
            "device_arch": "PPU ppu0010",
            "inspector_output_sha256": "9" * 64,
        })
        fq_row["binary_receipt_sha256"] = fq_receipt_sha
        fq_shards[fq_key] = fq_row

    sf_bundle = {
        "schema": SF_SCHEMA, "scope": "pilot", "route": "scalefirst",
        "parents_per_binary": 32, "source_sha": source_sha,
        "repository": {"tree": source_tree, "tracked_dirty_ignored": [],
                       "submodules": submodules},
        "sdk": sdk,
        "build_input_authority": {"path": sf_build_rel,
                                  "sha256": sf_build_sha},
        "shard_plan": {"path": sf_plan_rel, "sha256": sf_plan_sha,
                       "pairs": sf_plan["pairs"]},
        "runtime_probe": sf_probe,
        "shards": sf_shards,
    }
    fq_bundle = {
        "schema": FQ_SCHEMA, "mode": "PILOT", "max_parents_per_binary": 32,
        "source_sha": source_sha, "source_tree": source_tree,
        "submodules": submodules, "sdk": sdk,
        "build_input_authority": fq_build_rel,
        "build_input_authority_sha256": fq_build_sha,
        "runtime_identity_probe": fq_probe, "shards": fq_shards,
    }
    write_json(sf_root / "bundle.json", sf_bundle)
    write_json(fq_root / "bundle.json", fq_bundle)
    return sf_root / "bundle.json", fq_root / "bundle.json", output


def _expect_red(label: str, action: Any) -> None:
    try:
        action()
    except BundleError:
        return
    raise BundleError(f"{label} negative stayed green")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="kpack-compose-") as temporary:
        root = Path(temporary)
        template = root / "template"
        _fixture(template)

        def clone(name: str) -> tuple[Path, Path, Path]:
            destination = root / name
            shutil.copytree(template, destination)
            return (destination / "scalefirst/bundle.json",
                    destination / "fully-quantized/bundle.json",
                    destination / "bundle.json")

        sf, fq, output = clone("positive")
        positive = output.parent
        workload_plan = positive / "workload-plan.json"
        workload_plan.write_text(_encoded({
            "dense": [{"key": "dense-control"}],
            "grouped": [{"key": "grouped-control"}],
        }), encoding="utf-8")
        before = {path.relative_to(positive) for path in positive.rglob("*")
                  if path.is_file()}
        document = compose_document(
            output=output, scalefirst_bundle=sf,
            fully_quantized_bundle=fq,
            scalefirst_root=Path("scalefirst"),
            fully_quantized_root=Path("fully-quantized"))
        write_composite(output, document)
        validate_composite(output)
        after = {path.relative_to(positive) for path in positive.rglob("*")
                 if path.is_file()}
        if after != before | {Path("bundle.json")}:
            raise BundleError("compose copied or created an undeclared payload")
        tools = Path(__file__).resolve().parent
        if str(tools) not in sys.path:
            sys.path.insert(0, str(tools))
        import kpack_discovery_worker_plan as worker
        if len(worker._normalized_shard_rows(document)) != 4 or \
                len(worker._bundle_shards(document)) != 4:
            raise BundleError("composite is not worker-plan compatible")
        master = worker.make_master(output, workload_plan)
        if master["denominator"]["work_items"] != 4:
            raise BundleError("worker master did not consume the shard union")
        fq_native = _load_json(fq, "fixture")
        fq_row = next(row for row in document["shards"]
                      if row["route"] == "fully-quantized")
        native_row = fq_native["shards"][fq_row["native_shard_key"]]
        if (fq_row["shard_key"] !=
                f"fully-quantized:{fq_row['native_shard_key']}" or
                fq_row["files"]["manifest"]["path"] ==
                native_row["manifest"] or
                not (positive / fq_row["files"]["manifest"]["path"]).is_file()):
            raise BundleError("native/composite shard path contract differs")

        output.write_text("{}\n", encoding="utf-8")
        _expect_red("stale output", lambda: write_composite(output, document))

        sf, fq, output = clone("missing")
        sf_doc = _load_json(sf, "fixture")
        (sf.parent / sf_doc["shards"][0]["binary"]).unlink()
        _expect_red("missing payload", lambda: compose_document(
            output=output, scalefirst_bundle=sf,
            fully_quantized_bundle=fq))

        sf, fq, output = clone("escape")
        (output.parent / "outside").write_bytes(b"outside\n")
        sf_doc = _load_json(sf, "fixture")
        sf_doc["shards"][0]["manifest"] = "../outside"
        sf.write_text(_encoded(sf_doc), encoding="utf-8")
        _expect_red("escaping payload", lambda: compose_document(
            output=output, scalefirst_bundle=sf,
            fully_quantized_bundle=fq))

        sf, fq, output = clone("source")
        fq_doc = _load_json(fq, "fixture")
        fq_doc["source_sha"] = "a" * 40
        fq.write_text(_encoded(fq_doc), encoding="utf-8")
        _expect_red("source authority", lambda: compose_document(
            output=output, scalefirst_bundle=sf,
            fully_quantized_bundle=fq))

        sf, fq, output = clone("sdk")
        fq_doc = _load_json(fq, "fixture")
        fq_doc["sdk"]["compiler"]["sha256"] = "a" * 64
        fq.write_text(_encoded(fq_doc), encoding="utf-8")
        _expect_red("SDK authority", lambda: compose_document(
            output=output, scalefirst_bundle=sf,
            fully_quantized_bundle=fq))

        sf, fq, output = clone("route")
        fq_doc = _load_json(fq, "fixture")
        first = next(iter(fq_doc["shards"].values()))
        first["route"] = "scalefirst"
        fq.write_text(_encoded(fq_doc), encoding="utf-8")
        _expect_red("route", lambda: compose_document(
            output=output, scalefirst_bundle=sf,
            fully_quantized_bundle=fq))

        sf, fq, output = clone("collision")
        sf_doc = _load_json(sf, "fixture")
        sf_doc["shards"][1]["shard_id"] = sf_doc["shards"][0]["shard_id"]
        sf.write_text(_encoded(sf_doc), encoding="utf-8")
        _expect_red("shard collision", lambda: compose_document(
            output=output, scalefirst_bundle=sf,
            fully_quantized_bundle=fq))

        sf, fq, output = clone("parent")
        fq_doc = _load_json(fq, "fixture")
        first = next(iter(fq_doc["shards"].values()))
        first["parent_ids"][1] = first["parent_ids"][0]
        fq.write_text(_encoded(fq_doc), encoding="utf-8")
        _expect_red("parent collision", lambda: compose_document(
            output=output, scalefirst_bundle=sf,
            fully_quantized_bundle=fq))

        sf, fq, output = clone("oversized")
        fq_doc = _load_json(fq, "fixture")
        first = next(iter(fq_doc["shards"].values()))
        first["parent_end"] = 40
        first["parent_count"] = 40
        first["parent_ids"] = [f"planted-{index}" for index in range(40)]
        first["typed_rows"] = 40
        fq.write_text(_encoded(fq_doc), encoding="utf-8")
        _expect_red("oversized native shard", lambda: compose_document(
            output=output, scalefirst_bundle=sf,
            fully_quantized_bundle=fq))

        sf, fq, output = clone("native-symbol")
        sf_doc = _load_json(sf, "fixture")
        row = sf_doc["shards"][0]
        row["parent_symbols"][0] = "planted_non_authority_symbol"
        manifest_path = sf.parent / row["manifest"]
        manifest = _load_json(manifest_path, "fixture manifest")
        for field in ("compiled_parents", "typed_rows"):
            manifest[field][0]["symbol"] = "planted_non_authority_symbol"
            manifest[field][0]["static_candidate_id"] = \
                "planted_non_authority_symbol"
        manifest_path.write_text(_encoded(manifest), encoding="utf-8")
        manifest_sha = _file_sha(manifest_path)
        row["manifest_sha256"] = manifest_sha
        receipt_path = sf.parent / row["binary_receipt"]
        receipt = _load_json(receipt_path, "fixture binary receipt")
        receipt["manifest_sha256"] = manifest_sha
        receipt_path.write_text(_encoded(receipt), encoding="utf-8")
        row["binary_receipt_sha256"] = _file_sha(receipt_path)
        sf.write_text(_encoded(sf_doc), encoding="utf-8")
        _expect_red("ScaleFirst native symbol", lambda: compose_document(
            output=output, scalefirst_bundle=sf,
            fully_quantized_bundle=fq))

        sf, fq, output = clone("internal-authority")
        sf_doc = _load_json(sf, "fixture")
        authority_path = sf.parent / sf_doc["build_input_authority"]["path"]
        authority = _load_json(authority_path, "fixture build authority")
        authority["source_sha"] = "a" * 40
        authority_path.write_text(_encoded(authority), encoding="utf-8")
        authority_sha = _file_sha(authority_path)
        sf_doc["build_input_authority"]["sha256"] = authority_sha
        probe_receipt_path = sf.parent / sf_doc["runtime_probe"]["receipt"]
        probe_receipt = _load_json(probe_receipt_path, "fixture probe receipt")
        probe_receipt["build_input_authority_sha256"] = authority_sha
        probe_receipt_path.write_text(_encoded(probe_receipt), encoding="utf-8")
        sf_doc["runtime_probe"]["receipt_sha256"] = _file_sha(
            probe_receipt_path)
        for row in sf_doc["shards"]:
            receipt_path = sf.parent / row["binary_receipt"]
            receipt = _load_json(receipt_path, "fixture binary receipt")
            receipt["build_input_authority_sha256"] = authority_sha
            receipt_path.write_text(_encoded(receipt), encoding="utf-8")
            row["binary_receipt_sha256"] = _file_sha(receipt_path)
        sf.write_text(_encoded(sf_doc), encoding="utf-8")
        _expect_red("internal source authority", lambda: compose_document(
            output=output, scalefirst_bundle=sf,
            fully_quantized_bundle=fq))

        sf, fq, output = clone("child-mutation")
        document = compose_document(
            output=output, scalefirst_bundle=sf,
            fully_quantized_bundle=fq)
        write_composite(output, document)
        plan = output.parent / "workload-plan.json"
        plan.write_text(_encoded({
            "dense": [{"key": "dense-control"}],
            "grouped": [{"key": "grouped-control"}],
        }), encoding="utf-8")
        master = worker.make_master(output, plan)
        sf_doc = _load_json(sf, "fixture")
        binary = sf.parent / sf_doc["shards"][0]["binary"]
        binary.write_bytes(b"mutated\n")
        for label, action in (
                ("make_master", lambda: worker.make_master(output, plan)),
                ("validate_master", lambda: worker.validate_master(
                    master, output, plan))):
            try:
                action()
            except worker.PlanError:
                pass
            else:
                raise BundleError(
                    f"worker {label} accepted a mutated composite child")

    print("[kpack-discovery-compose:self-test] PASS native-authorities routes=2 "
          "shards=4 "
          "relative-no-copy worker-list-compatible "
          "stale+missing+escape+source+sdk+route+shard+parent+width+"
          "native-symbol+internal-source+child-mutation=RED")


def _compose_command(args: argparse.Namespace) -> int:
    document = compose_document(
        output=args.output,
        scalefirst_bundle=args.scalefirst_bundle,
        fully_quantized_bundle=args.fully_quantized_bundle,
        scalefirst_root=args.scalefirst_root,
        fully_quantized_root=args.fully_quantized_root)
    write_composite(args.output, document)
    print(f"[kpack-discovery-compose] PASS mode={document['mode']} "
          f"shards={document['denominator']['shards']} output={args.output}")
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    document = validate_composite(args.bundle)
    print(f"[kpack-discovery-compose] VALID mode={document['mode']} "
          f"shards={document['denominator']['shards']} bundle={args.bundle}")
    return 0


def _self_test_command(_args: argparse.Namespace) -> int:
    self_test()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compose = commands.add_parser("compose")
    compose.add_argument("--scalefirst-bundle", type=Path, required=True)
    compose.add_argument("--fully-quantized-bundle", type=Path, required=True)
    compose.add_argument("--scalefirst-root", type=Path)
    compose.add_argument("--fully-quantized-root", type=Path)
    compose.add_argument("--output", type=Path, required=True)
    compose.set_defaults(func=_compose_command)
    validate = commands.add_parser("validate")
    validate.add_argument("--bundle", type=Path, required=True)
    validate.set_defaults(func=_validate_command)
    test = commands.add_parser("self-test")
    test.set_defaults(func=_self_test_command)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        return int(args.func(args))
    except BundleError as exc:
        fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
