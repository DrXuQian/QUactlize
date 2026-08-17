#!/usr/bin/env python3
"""Cross-check the frozen catalog, resolved GGUF set, and shape inventory.

Each producer validates its own JSON schema, but the top-level sweep needs a
stronger statement: all three files describe *exactly the same model set*.
This checker is deliberately independent of device measurement and does not
rehash multi-gigabyte GGUF payloads.  The resolver/inventory own those byte
checks; this seam binds their already-recorded hashes to the frozen catalog.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import gguf_internal_shape_inventory as inventory  # noqa: E402
import resolve_internal_sweep_models as resolver  # noqa: E402


class AuthorityError(ValueError):
    pass


def _load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthorityError(f"cannot read {path}: {exc}") from exc


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_sha256(value: Any) -> str:
    return hashlib.sha256(resolver.canonical_json(value)).hexdigest()


def validate(catalog_path: pathlib.Path, resolved_path: pathlib.Path,
             spec_path: pathlib.Path) -> dict[str, Any]:
    catalog = resolver.load_catalog(catalog_path)
    resolved = inventory.load_resolved(resolved_path)
    spec = _load_json(spec_path)

    catalog_sha = _sha256(catalog_path)
    if resolved.get("catalog_sha256") != catalog_sha:
        raise AuthorityError(
            "resolved model set was not produced from the frozen catalog: "
            f"{resolved.get('catalog_sha256')} != {catalog_sha}")
    for field in ("shape_directory", "workload_axes"):
        if resolved.get(field) != catalog.get(field):
            raise AuthorityError(f"resolved {field} differs from frozen catalog")

    catalog_models = catalog["models"]
    resolved_models = resolved["models"]
    expected_ids = [row["model_id"] for row in catalog_models]
    actual_ids = [row["model_id"] for row in resolved_models]
    if actual_ids != expected_ids:
        raise AuthorityError(
            f"resolved model membership/order differs: got={actual_ids} "
            f"expected={expected_ids}")
    policies = catalog["tp_policies"]
    for declared, actual in zip(catalog_models, resolved_models):
        for field, value in declared.items():
            if actual.get(field) != value:
                raise AuthorityError(
                    f"{declared['model_id']}: resolved {field} differs from catalog")
        expected_policy = policies[declared["tp_policy"]]
        if actual.get("tp_policy_definition") != expected_policy:
            raise AuthorityError(
                f"{declared['model_id']}: resolved TP policy definition drifted")

    resolved_identity = {
        "catalog_sha256": catalog_sha,
        "models": [{
            "model_id": row["model_id"],
            "fileset_sha256": row["fileset_sha256"],
            "tp_world_size": row["tp_world_size"],
            "tp_policy": row["tp_policy"],
        } for row in resolved_models],
    }
    resolved_set_sha = _identity_sha256(resolved_identity)
    if resolved.get("resolved_set_sha256") != resolved_set_sha:
        raise AuthorityError("resolved_set_sha256 does not describe resolved members")

    if not isinstance(spec, dict) or spec.get("schema") != inventory.SCHEMA:
        raise AuthorityError(f"inventory expected schema {inventory.SCHEMA}")
    if spec.get("status") != "COMPLETE":
        raise AuthorityError("inventory is not COMPLETE")
    if spec.get("resolved_models_schema") != inventory.RESOLVED_SCHEMA:
        raise AuthorityError("inventory resolved-model schema drifted")
    if spec.get("resolved_set_sha256") != resolved_set_sha:
        raise AuthorityError("inventory belongs to a different resolved model set")
    if spec.get("shape_directory_contract") != catalog["shape_directory"]:
        raise AuthorityError("inventory shape-directory contract differs from catalog")
    if spec.get("workload_axes") != catalog["workload_axes"]:
        raise AuthorityError("inventory workload axes differ from catalog")

    filesets = {row["model_id"]: row["fileset_sha256"] for row in resolved_models}
    provenance = spec.get("provenance")
    if not isinstance(provenance, dict):
        raise AuthorityError("inventory provenance is missing")
    if provenance.get("gguf_hashes") != filesets:
        raise AuthorityError("inventory GGUF member map differs from resolved set")
    if provenance.get("gguf_set_sha256") != _identity_sha256(filesets):
        raise AuthorityError("inventory gguf_set_sha256 does not describe its members")
    if provenance.get("shape_directory") != catalog["shape_directory"]:
        raise AuthorityError("inventory provenance shape-directory contract drifted")

    inventory_models = spec.get("models")
    if not isinstance(inventory_models, list):
        raise AuthorityError("inventory models list is missing")
    inventory_ids = [row.get("model_id") for row in inventory_models
                     if isinstance(row, dict)]
    if inventory_ids != expected_ids:
        raise AuthorityError(
            f"inventory model membership/order differs: got={inventory_ids} "
            f"expected={expected_ids}")
    for row in inventory_models:
        model_id = row["model_id"]
        if row.get("fileset_sha256") != filesets[model_id]:
            raise AuthorityError(f"{model_id}: inventory fileset hash drifted")
        resolved_model = resolved_models[expected_ids.index(model_id)]
        for field in ("tp_world_size", "tp_policy", "problem_kinds"):
            if row.get(field) != resolved_model.get(field):
                raise AuthorityError(
                    f"{model_id}: inventory model {field} differs from resolved authority")
        for field in ("observed_tensor_count", "logical_tensor_count",
                      "matrix_tensor_count"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AuthorityError(f"{model_id}: invalid inventory {field}={value!r}")
        if row["observed_tensor_count"] == 0:
            raise AuthorityError(f"{model_id}: inventory observed no GGUF tensors")

    collections_by_name: dict[str, list[dict[str, Any]]] = {}
    for collection in ("cells", "sweep_shapes", "tensors"):
        rows = spec.get(collection)
        if not isinstance(rows, list):
            raise AuthorityError(f"inventory {collection} is missing")
        malformed = [index for index, row in enumerate(rows)
                     if not isinstance(row, dict)]
        if malformed:
            raise AuthorityError(
                f"inventory {collection} has non-object rows: {malformed[:8]}")
        unknown = sorted({row.get("model_id") for row in rows
                          if row.get("model_id") not in filesets}, key=str)
        if unknown:
            raise AuthorityError(f"inventory {collection} names unknown models: {unknown}")
        collections_by_name[collection] = rows

    cells = collections_by_name["cells"]
    shapes = collections_by_name["sweep_shapes"]
    tensors = collections_by_name["tensors"]
    count_fields = {
        "model_count": len(inventory_models),
        "tensor_count": len(tensors),
        "expanded_cell_count": len(cells),
        "deduplicated_shape_count": len(shapes),
        "physical_tensor_count": sum(row["observed_tensor_count"]
                                     for row in inventory_models),
        "matrix_tensor_count": sum(bool(row.get("matmul_tensor")) for row in tensors),
        "unclassified_tensor_count": sum(row.get("role") == "UNCLASSIFIED"
                                         for row in tensors),
    }
    for field, expected in count_fields.items():
        if spec.get(field) != expected:
            raise AuthorityError(
                f"inventory {field} does not describe rows: {spec.get(field)!r} != {expected}")

    tensor_counts = collections.Counter(row["model_id"] for row in tensors)
    for model in inventory_models:
        if tensor_counts[model["model_id"]] != model["logical_tensor_count"]:
            raise AuthorityError(
                f"{model['model_id']}: logical tensor count differs from tensor rows")

    for collection, rows in (("cells", cells), ("sweep_shapes", shapes),
                             ("tensors", tensors)):
        for index, row in enumerate(rows):
            model_id = row["model_id"]
            expected_world = resolved_models[expected_ids.index(model_id)]["tp_world_size"]
            world = (row.get("tp_world") if collection == "sweep_shapes"
                     else (row.get("tp") or {}).get("world_size")
                     if collection in {"cells", "tensors"} else None)
            if world != expected_world:
                raise AuthorityError(
                    f"inventory {collection}[{index}] TP world differs for {model_id}: "
                    f"{world!r} != {expected_world}")
            if collection in {"cells", "tensors"} and \
                    row.get("fileset_sha256") != filesets[model_id]:
                raise AuthorityError(
                    f"inventory {collection}[{index}] fileset differs for {model_id}")

    cell_ids = [row.get("cell_id") for row in cells]
    shape_ids = [row.get("shape_id") for row in shapes]
    if any(not isinstance(value, str) or not value for value in cell_ids) or \
            len(cell_ids) != len(set(cell_ids)):
        raise AuthorityError("inventory expanded cell IDs are missing or duplicated")
    if any(not isinstance(value, str) or not value for value in shape_ids) or \
            len(shape_ids) != len(set(shape_ids)):
        raise AuthorityError("inventory shape IDs are missing or duplicated")
    if set(shape_ids) != {row.get("dedup_key") for row in cells}:
        raise AuthorityError("inventory deduplicated shapes do not cover expanded cells")

    return {
        "catalog_sha256": catalog_sha,
        "resolved_set_sha256": resolved_set_sha,
        "model_ids": expected_ids,
        "gguf_hashes": filesets,
        "inventory_sha256": _sha256(spec_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=pathlib.Path, required=True)
    parser.add_argument("--resolved", type=pathlib.Path, required=True)
    parser.add_argument("--inventory", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    try:
        result = validate(args.catalog.resolve(), args.resolved.resolve(),
                          args.inventory.resolve())
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(encoded)
        print("[internal-sweep-authority] PASS "
              f"models={len(result['model_ids'])} "
              f"resolved={result['resolved_set_sha256']}")
        return 0
    except (AuthorityError, inventory.InventoryError, resolver.ResolveError, OSError) as exc:
        print(f"[internal-sweep-authority] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
