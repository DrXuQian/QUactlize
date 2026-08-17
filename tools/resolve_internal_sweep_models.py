#!/usr/bin/env python3
"""Resolve a data-only model catalog to an immutable GGUF file set.

The checked-in catalog owns model identities, workload axes, and deployment
semantics.  The actual GGUF headers own tensor names, qtypes, and dimensions.
This resolver is the narrow seam between them: operators bind each model ID to
one file or directory, and it writes the exact ordered file list consumed by
``gguf_internal_shape_inventory.py``.  The inventory never performs a glob, so
a resumed sweep cannot silently gain or lose a shard.

Examples::

  python3 tools/resolve_internal_sweep_models.py self-test

  python3 tools/resolve_internal_sweep_models.py resolve \
    --bind qwen3.5-35b-a3b-q4_k_m=/models/Qwen3.5-35B-A3B-Q4_K_M \
    --bind qwen3-32b-q4_k_m=/models/Qwen3-32B-Q4_K_M \
    --bind qwen3.5-122b-a10b-q4_k_m-tp2=/models/Qwen3.5-122B-A10B-Q4_K_M \
    --output /workspace/quactlize-internal-sweep/resolved-models.json

The catalog's ``binding_env`` names are an equivalent convenience.  Explicit
``--bind`` values win, but conflicting duplicate bindings fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "benchmarks" / "internal_sweep_models.json"
CATALOG_SCHEMA = "quactlize.internal_sweep.model_catalog.v1"
RESOLVED_SCHEMA = "quactlize.internal_sweep.resolved_models.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SPLIT_RE = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$", re.I)
MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class ResolveError(ValueError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_catalog(doc: Any, source: str) -> dict[str, Any]:
    if not isinstance(doc, dict) or doc.get("schema") != CATALOG_SCHEMA:
        raise ResolveError(f"{source}: expected schema {CATALOG_SCHEMA}")
    models = doc.get("models")
    policies = doc.get("tp_policies")
    if not isinstance(models, list) or not models:
        raise ResolveError(f"{source}: models must be a nonempty list")
    if not isinstance(policies, dict) or not policies:
        raise ResolveError(f"{source}: tp_policies must be a nonempty object")
    seen_ids: set[str] = set()
    seen_env: set[str] = set()
    for ordinal, model in enumerate(models):
        if not isinstance(model, dict):
            raise ResolveError(f"{source}: models[{ordinal}] is not an object")
        model_id = model.get("model_id")
        binding_env = model.get("binding_env")
        if not isinstance(model_id, str) or not MODEL_ID_RE.fullmatch(model_id):
            raise ResolveError(f"{source}: invalid model_id {model_id!r}")
        if model_id in seen_ids:
            raise ResolveError(f"{source}: duplicate model_id {model_id}")
        if not isinstance(binding_env, str) or not binding_env or binding_env in seen_env:
            raise ResolveError(f"{source}: missing or duplicate binding_env {binding_env!r}")
        seen_ids.add(model_id)
        seen_env.add(binding_env)
        tp_world = model.get("tp_world_size")
        if isinstance(tp_world, bool) or not isinstance(tp_world, int) or tp_world < 1:
            raise ResolveError(f"{source}: {model_id} has invalid tp_world_size={tp_world!r}")
        policy = model.get("tp_policy")
        if policy not in policies:
            raise ResolveError(f"{source}: {model_id} names unknown tp_policy={policy!r}")
        kinds = model.get("problem_kinds")
        if not isinstance(kinds, list) or not kinds or any(
                kind not in {"dense", "grouped"} for kind in kinds):
            raise ResolveError(f"{source}: {model_id} has invalid problem_kinds={kinds!r}")
    shape_directory = doc.get("shape_directory")
    if not isinstance(shape_directory, dict) or set(shape_directory) != {"dense", "grouped"}:
        raise ResolveError(f"{source}: shape_directory must define dense and grouped")
    workload_axes = doc.get("workload_axes")
    if not isinstance(workload_axes, dict) or set(workload_axes) != {"dense", "grouped"}:
        raise ResolveError(f"{source}: workload_axes must define dense and grouped")
    dense_axes = workload_axes["dense"]
    grouped_axes = workload_axes["grouped"]
    for owner, field in ((dense_axes, "decode_m"), (dense_axes, "prefill_m"),
                         (grouped_axes, "decode_tokens"),
                         (grouped_axes, "prefill_tokens")):
        values = owner.get(field) if isinstance(owner, dict) else None
        if (not isinstance(values, list) or not values or
                any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in values) or len(values) != len(set(values))):
            raise ResolveError(f"{source}: invalid workload axis {field}={values!r}")
    if grouped_axes.get("expert_count_source") != "gguf:{architecture}.expert_count":
        raise ResolveError(f"{source}: grouped expert_count_source is not GGUF-owned")
    if grouped_axes.get("top_k_source") != "gguf:{architecture}.expert_used_count":
        raise ResolveError(f"{source}: grouped top_k_source is not GGUF-owned")
    profiles = grouped_axes.get("ragged_profiles")
    if (not isinstance(profiles, list) or not profiles or
            any(not isinstance(value, str) or not MODEL_ID_RE.fullmatch(value)
                for value in profiles) or len(profiles) != len(set(profiles))):
        raise ResolveError(f"{source}: invalid versioned ragged_profiles={profiles!r}")
    return doc


def load_catalog(path: pathlib.Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ResolveError(f"cannot read catalog {path}: {exc}") from exc
    return validate_catalog(doc, str(path))


def parse_bindings(values: Iterable[str]) -> dict[str, pathlib.Path]:
    result: dict[str, pathlib.Path] = {}
    for raw in values:
        model_id, separator, path = raw.partition("=")
        if not separator or not model_id or not path:
            raise ResolveError(f"binding must be model_id=/absolute/path, got {raw!r}")
        if model_id in result:
            raise ResolveError(f"duplicate explicit binding for {model_id}")
        candidate = pathlib.Path(path)
        if not candidate.is_absolute():
            raise ResolveError(f"binding for {model_id} is not absolute: {path}")
        result[model_id] = candidate
    return result


def split_group(paths: Iterable[pathlib.Path]) -> list[pathlib.Path]:
    """Validate one plain GGUF or one complete standard split filename set."""
    resolved = sorted({path.resolve() for path in paths}, key=lambda item: item.name)
    if not resolved:
        raise ResolveError("binding resolved to no .gguf files")
    for path in resolved:
        if not path.is_file():
            raise ResolveError(f"GGUF path is not a regular file: {path}")
        if path.suffix.lower() != ".gguf":
            raise ResolveError(f"bound file does not end in .gguf: {path}")
    matches = [SPLIT_RE.fullmatch(path.name) for path in resolved]
    if not any(matches):
        if len(resolved) != 1:
            raise ResolveError(
                "directory contains multiple unsplit GGUF files; bind the exact file or one split directory")
        return resolved
    if not all(matches):
        raise ResolveError("binding mixes split and unsplit GGUF filenames")
    groups = {(match.group("prefix"), int(match.group("count"))) for match in matches if match}
    if len(groups) != 1:
        raise ResolveError(f"binding contains multiple GGUF split groups: {sorted(groups)}")
    prefix, count = next(iter(groups))
    indices = [int(match.group("index")) for match in matches if match]
    # GGUF filenames are one-based while split.no inside the header is
    # zero-based.  Header agreement is checked by the inventory.
    expected = list(range(1, count + 1))
    if indices != expected:
        raise ResolveError(
            f"incomplete or duplicated split set {prefix}: got={indices} expected={expected}")
    if len(resolved) != count:
        raise ResolveError(f"split count {count} disagrees with files={len(resolved)}")
    return resolved


def resolve_binding(path: pathlib.Path) -> list[pathlib.Path]:
    if path.is_file():
        match = SPLIT_RE.fullmatch(path.name)
        if match:
            siblings = []
            for sibling in path.parent.iterdir():
                sibling_match = SPLIT_RE.fullmatch(sibling.name)
                if (sibling.is_file() and sibling_match and
                        sibling_match.group("prefix") == match.group("prefix") and
                        sibling_match.group("count") == match.group("count")):
                    siblings.append(sibling)
            return split_group(siblings)
        return split_group([path])
    if path.is_dir():
        return split_group(
            child for child in path.iterdir()
            if child.is_file() and child.suffix.lower() == ".gguf")
    raise ResolveError(f"binding path does not exist: {path}")


def selected_models(catalog: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    by_id = {model["model_id"]: model for model in catalog["models"]}
    if not names:
        return list(catalog["models"])
    requested: list[str] = []
    for value in names:
        requested.extend(token for token in value.split(",") if token)
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ResolveError(f"unknown model IDs: {unknown}; available={sorted(by_id)}")
    if len(requested) != len(set(requested)):
        raise ResolveError(f"duplicate selected model IDs: {requested}")
    return [by_id[model_id] for model_id in requested]


def resolve(catalog_path: pathlib.Path, output: pathlib.Path,
            binding_values: list[str], names: list[str]) -> dict[str, Any]:
    catalog = load_catalog(catalog_path)
    explicit = parse_bindings(binding_values)
    models = selected_models(catalog, names)
    selected_ids = {model["model_id"] for model in models}
    extra = sorted(set(explicit) - selected_ids)
    if extra:
        raise ResolveError(f"bindings supplied for unselected/unknown models: {extra}")

    resolved_models = []
    for model in models:
        model_id = model["model_id"]
        env_name = model["binding_env"]
        explicit_path = explicit.get(model_id)
        env_value = os.environ.get(env_name)
        if explicit_path is not None and env_value:
            env_path = pathlib.Path(env_value)
            if env_path != explicit_path:
                raise ResolveError(
                    f"{model_id}: --bind={explicit_path} contradicts {env_name}={env_path}")
        binding = explicit_path or (pathlib.Path(env_value) if env_value else None)
        if binding is None:
            raise ResolveError(
                f"{model_id}: no binding; pass --bind {model_id}=/absolute/path "
                f"or set {env_name}")
        if not binding.is_absolute():
            raise ResolveError(f"{model_id}: {env_name} path is not absolute: {binding}")
        files = resolve_binding(binding)
        file_rows = [
            {"path": str(path), "size": path.stat().st_size,
             "sha256": sha256_file(path)} for path in files
        ]
        fileset = hashlib.sha256(canonical_json(
            [{"size": row["size"], "sha256": row["sha256"]} for row in file_rows]
        )).hexdigest()
        resolved_models.append({
            **model,
            "binding_source": "--bind" if explicit_path is not None else f"env:{env_name}",
            "files": file_rows,
            "fileset_sha256": fileset,
            "tp_policy_definition": catalog["tp_policies"][model["tp_policy"]],
        })

    result = {
        "schema": RESOLVED_SCHEMA,
        "catalog": str(catalog_path.resolve()),
        "catalog_sha256": sha256_file(catalog_path),
        "shape_directory": catalog["shape_directory"],
        "workload_axes": catalog["workload_axes"],
        "models": resolved_models,
    }
    result["resolved_set_sha256"] = hashlib.sha256(canonical_json({
        "catalog_sha256": result["catalog_sha256"],
        "models": [{"model_id": model["model_id"],
                    "fileset_sha256": model["fileset_sha256"],
                    "tp_world_size": model["tp_world_size"],
                    "tp_policy": model["tp_policy"]}
                   for model in resolved_models],
    })).hexdigest()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def self_test() -> None:
    catalog = load_catalog(DEFAULT_CATALOG)
    assert [model["model_id"] for model in catalog["models"]] == [
        "qwen3.5-35b-a3b-q4_k_m",
        "qwen3-32b-q4_k_m",
        "qwen3.5-122b-a10b-q4_k_m-tp2",
    ]
    assert catalog["models"][2]["tp_world_size"] == 2
    assert catalog["tp_policies"]["qwen-tensor-parallel-v1"][
        "role_partition_axis"]["attn_o"] == "k"
    assert catalog["tp_policies"]["qwen-tensor-parallel-v1"][
        "role_partition_axis"]["moe_expert_up"] == "n"
    assert catalog["workload_axes"]["grouped"]["top_k_source"] == (
        "gguf:{architecture}.expert_used_count")
    assert catalog["workload_axes"]["grouped"]["ragged_profiles"] == [
        "token-topk-hot16x4-wor-sm64-s44-v1"]

    planted_old_semantics = json.loads(json.dumps(catalog))
    grouped = planted_old_semantics["workload_axes"]["grouped"]
    grouped["active_expert_source"] = grouped.pop("top_k_source")
    try:
        validate_catalog(planted_old_semantics, "planted-active-means-top-k")
    except ResolveError as exc:
        assert "top_k_source" in str(exc)
    else:
        raise AssertionError("expert_used_count was accepted as active-expert authority")

    # Split grouping is tested without creating files: its filename arithmetic
    # is factored here and header-level split metadata has an independent
    # negative in gguf_internal_shape_inventory.py.
    def filename_contract(names: list[str]) -> tuple[str, int, list[int]]:
        matches = [SPLIT_RE.fullmatch(name) for name in names]
        if not matches or not all(matches):
            raise ResolveError("not one split filename family")
        groups = {(m.group("prefix"), int(m.group("count"))) for m in matches if m}
        if len(groups) != 1:
            raise ResolveError("multiple split filename families")
        prefix, count = next(iter(groups))
        indices = sorted(int(m.group("index")) for m in matches if m)
        if indices != list(range(1, count + 1)):
            raise ResolveError("incomplete split filename family")
        return prefix, count, indices

    assert filename_contract([
        "model-00001-of-00003.gguf", "model-00002-of-00003.gguf",
        "model-00003-of-00003.gguf"])[1:] == (3, [1, 2, 3])
    for bad in (
        ["model-00001-of-00003.gguf", "model-00003-of-00003.gguf"],
        ["a-00001-of-00001.gguf", "b-00001-of-00001.gguf"],
    ):
        try:
            filename_contract(bad)
        except ResolveError:
            pass
        else:
            raise AssertionError(f"bad split set did not fail: {bad}")

    duplicate = json.loads(json.dumps(catalog))
    duplicate["models"].append(dict(duplicate["models"][0]))
    try:
        validate_catalog(duplicate, "planted-duplicate")
    except ResolveError as exc:
        assert "duplicate model_id" in str(exc)
    else:
        raise AssertionError("duplicate model ID did not fail")
    assert catalog["shape_directory"]["dense"] == "m{m}_n{n}_k{k}_g{group_size}"
    print("[internal-model-resolver:self-test] PASS: 3-model authority, TP role axes, "
          "split completeness/family negatives, duplicate-ID red, and stable shape folders")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    resolve_parser = sub.add_parser("resolve")
    resolve_parser.add_argument("--catalog", type=pathlib.Path, default=DEFAULT_CATALOG)
    resolve_parser.add_argument("--bind", action="append", default=[],
                                help="model_id=/absolute/file-or-directory")
    resolve_parser.add_argument("--model", action="append", default=[],
                                help="model ID or comma-separated model IDs; default all")
    resolve_parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
            return 0
        result = resolve(args.catalog, args.output, args.bind, args.model)
        print(f"[internal-model-resolver] PASS models={len(result['models'])} "
              f"set_sha256={result['resolved_set_sha256']} output={args.output.resolve()}")
        for model in result["models"]:
            print(f"  {model['model_id']}: tp={model['tp_world_size']} "
                  f"files={len(model['files'])} fileset={model['fileset_sha256']}")
        return 0
    except ResolveError as exc:
        print(f"[internal-model-resolver] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
