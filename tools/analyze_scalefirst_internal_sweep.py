#!/usr/bin/env python3
"""Close the all-format ScaleFirst internal-sweep denominator.

The finite tactic graph comes from ``scalefirst_internal_matrix`` and the
generated shard manifests.  The device may expand only the persistent grid
axis, because that axis depends on the exact compiled kernel's
``maximum_active_blocks()``.  Missing, duplicate, extra, malformed, or failed
runtime records make the component INCOMPLETE; they are never pruning.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import pathlib
import re
import stat
import statistics
import sys
from typing import Any

import analyze_fully_quantized_internal_sweep as inventory
import gen_scalefirst_internal_units as generator
import scalefirst_internal_matrix as matrix


SCHEMA = "quactlize.scalefirst_internal_sweep.v3"
PLAN_SCHEMA = "quactlize.scalefirst_internal_plan.v1"
RUN_CONTRACT_SCHEMA = "quactlize.scalefirst_internal_sweep.run_contract.v1"
RUN_COMMIT_SCHEMA = "quactlize.scalefirst_internal_sweep.run_commit.v1"
GENERATED_SHARD_V2 = "quactlize.scalefirst.generated_shard.v2"
GENERATED_SHARD_V3 = "quactlize.scalefirst.generated_shard.v3"
VALID_QTYPES = {fmt.qtype for fmt in matrix.FORMATS}
FULL = "FULL_OUTPUT"
PRODUCER = "PRODUCER_ONLY_REDUCER_EXCLUDED"
CELL_PREFIX = "SF_CELL "
COMPLETE_PREFIX = "SF_COMPLETE "
SHARD_PREFIX = "SF_SHARD "
SHARD_RE = re.compile(r"^q(\d+)-a(\d+)-bc([01])$")
ATTEMPT_RE = re.compile(r"[A-Za-z0-9._:-]+")
EXPECTED_SOURCE_HASHES = {
    "benchmarks/scalefirst_internal_sweep_bench.hpp",
    "benchmarks/scalefirst_internal_sweep_unit.inc",
    "benchmarks/test_scalefirst_internal_sweep.cu",
    "quactlize/csrc/scalefirst_internal_sweep.cmake.in",
    "quactlize/csrc/CMakeLists.txt.in",
    "quactlize/csrc/device/ppu_dense_layout.cu",
    "quactlize/include/dense_splitk_multiformat_ppu.cuh",
    "quactlize/include/dense_splitk_parallel_ppu.cuh",
    "quactlize/include/ppu_format_config.inc",
    "quactlize/include/ppu_group_schedule.hpp",
    "quactlize/include/ppu_dense_shipping_policy.hpp",
    "quactlize/include/ppu_tactic_space.hpp",
    "quactlize/include/scalefirst_persistent_policy.hpp",
    "tests/helper.h",
    "tools/analyze_fully_quantized_internal_sweep.py",
    "tools/analyze_scalefirst_internal_sweep.py",
    "tools/emit_scalefirst_internal_superset.cpp",
    "tools/gen_scalefirst_internal_units.py",
    "tools/probe_box_identity.py",
    "tools/box_identity_schema.py",
    "tools/box_identity_probe.cpp",
    "tools/run_scalefirst_internal_sweep_box.sh",
    "tools/scalefirst_internal_matrix.py",
    "ci/check_scalefirst_internal_runner_contract.py",
    "build.sh",
    "tree/quactlize/include",
    "tree/third_party/actlize/include",
    "tree/third_party/actlize/tools/util/include",
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def atomic_json(path: pathlib.Path, value: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    text = (json.dumps(value, indent=2 if pretty else None, sort_keys=True,
                       ensure_ascii=False,
                       separators=None if pretty else (",", ":")) + "\n")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def require_sha256(value: Any, field: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"{field} must be a lowercase sha256")
    return text


def validate_bind_binary(binary: pathlib.Path, hashes_path: pathlib.Path,
                         shard: str, evidence: bool) -> str:
    """Validate one exact binary and bind it only for a fresh shard.

    A resumed shard must preserve the ordinary, non-symlink executable whose
    digest was published before its run commit.  Missing evidence is never an
    authorization to rebuild.
    """
    try:
        mode = binary.lstat().st_mode
    except FileNotFoundError as error:
        raise ValueError(f"{shard}: resume binary is missing") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode) or \
            not os.access(binary, os.X_OK):
        raise ValueError(
            f"{shard}: binary must be a regular non-symlink executable")
    document = json.loads(hashes_path.read_text())
    if not isinstance(document, dict):
        raise ValueError("binary-hashes authority is not an object")
    digest = sha256(binary)
    previous = document.get(shard)
    if evidence and previous is None:
        raise ValueError(f"{shard}: binary/run evidence lost binary authority")
    if previous is not None and previous != digest:
        raise ValueError(f"{shard}: binary hash changed")
    if not evidence and previous is not None:
        raise ValueError(f"{shard}: fresh shard already has binary authority")
    if previous is None:
        document[shard] = digest
        atomic_json(hashes_path, document)
    return digest


def runtime_authority(raw_root: pathlib.Path, expected_shards: set[str],
                      run_contract_path: pathlib.Path,
                      generated_hashes: dict[str, str],
                      binary_hashes: dict[str, str],
                      raw_hashes: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Validate every measured shard as one committed evidence transaction."""
    observed = {
        path.parent.name
        for pattern in ("q*-a*-bc*/run.log", "q*-a*-bc*/run.rc",
                        "q*-a*-bc*/run.commit.json")
        for path in raw_root.glob(pattern)
    }
    if observed != expected_shards:
        raise ValueError(
            "runtime evidence shard set differs from typed graph: "
            f"missing={sorted(expected_shards-observed)} "
            f"extra={sorted(observed-expected_shards)}")
    contract_sha = sha256(run_contract_path)
    result: dict[str, dict[str, Any]] = {}
    for shard in sorted(expected_shards):
        directory = raw_root / shard
        log = directory / "run.log"
        rc_path = directory / "run.rc"
        commit_path = directory / "run.commit.json"
        if not all(path.is_file() for path in (log, rc_path, commit_path)) or \
                log.stat().st_size == 0:
            raise ValueError(f"runtime shard {shard} lacks log/rc/commit authority")
        rc_text = rc_path.read_text().strip()
        if not rc_text.isdigit() or not 0 <= int(rc_text) <= 255:
            raise ValueError(f"runtime shard {shard} has malformed run.rc")
        commit = json.loads(commit_path.read_text())
        expected = {
            "schema": RUN_COMMIT_SCHEMA,
            "rc": int(rc_text),
            "run_log_sha256": sha256(log),
            "run_rc_sha256": sha256(rc_path),
            "run_contract_sha256": contract_sha,
            "generated_source_sha256": generated_hashes.get(shard),
            "binary_sha256": binary_hashes.get(shard),
        }
        if commit != expected or any(value is None for value in expected.values()):
            raise ValueError(f"runtime shard {shard} run evidence authority changed")
        if raw_hashes.get(shard) != expected["run_log_sha256"]:
            raise ValueError(f"runtime shard {shard} raw-log index differs from commit")
        result[shard] = {
            "rc": int(rc_text),
            "run_log_sha256": expected["run_log_sha256"],
            "run_rc_sha256": expected["run_rc_sha256"],
            "run_commit_sha256": sha256(commit_path),
        }
    return result


def parse_kv(line: str, prefix: str) -> dict[str, str]:
    if not line.startswith(prefix):
        raise ValueError(f"record lacks prefix {prefix!r}")
    fields: dict[str, str] = {}
    for token in line[len(prefix):].strip().split():
        if "=" not in token:
            raise ValueError(f"non key=value token {token!r}")
        key, value = token.split("=", 1)
        if key in fields:
            raise ValueError(f"duplicate field {key!r}")
        fields[key] = value
    return fields


def parse_shape(text: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"(\d+)x(\d+)x(\d+)", text)
    if not match:
        raise ValueError(f"invalid shape {text!r}")
    shape = tuple(map(int, match.groups()))
    if min(shape) <= 0:
        raise ValueError(f"non-positive shape {shape}")
    return shape


def materialize_plan(spec_path: pathlib.Path, output: pathlib.Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text())
    plan = inventory._materialize_spec(spec, spec_path, sha256(spec_path))
    plan["schema"] = PLAN_SCHEMA
    atomic_json(output, plan)
    return plan


def load_plan(path: pathlib.Path) -> dict[str, Any]:
    plan = json.loads(path.read_text())
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"{path}: expected {PLAN_SCHEMA}")
    if not isinstance(plan.get("cells"), list) or not plan["cells"]:
        raise ValueError("materialized ScaleFirst plan has no cells")
    if not isinstance(plan.get("provenance"), dict):
        raise ValueError("materialized ScaleFirst plan lacks provenance")
    return plan


def validate_resolved_models(plan: dict[str, Any], path: pathlib.Path) -> None:
    inventory._validate_resolved_document(plan, json.loads(path.read_text()), str(path))


def plan_dim(cell: dict[str, Any], name: str) -> int:
    return int(cell[name] if name in cell else cell[name.upper()])


def workload_metadata(cell: dict[str, Any], registry_group_size: int | None = None
                      ) -> dict[str, Any]:
    return inventory.workload_metadata(cell, registry_group_size)


def primary_tensor(cell: dict[str, Any]) -> str:
    return inventory.primary_tensor(cell)


def is_grouped(cell: dict[str, Any]) -> bool:
    return inventory.is_grouped(cell)


def fold_for(bits: int, artifact: int) -> int:
    if bits == 0:
        return 1
    bytes_per_run = bits * artifact // 8
    if bits * artifact % 8 or bytes_per_run <= 0:
        raise ValueError(f"non-byte plane qbits={bits} A={artifact}")
    return 1 if bytes_per_run >= 32 else 32 // bytes_per_run


def layout_for(fmt: matrix.Format, artifact: int) -> dict[str, Any]:
    low = fold_for(fmt.low_bits, artifact)
    high = fold_for(fmt.high_bits, artifact)
    return {
        "name": f"xplane-q{fmt.qtype}-a{artifact}-f{low}x{high}-scalefirst-fp16",
        "artifact_tile_k": artifact,
        "fold_n": {"low": low, "high": high},
        "metadata": "FP16_SCALE_ZERO_PLANES",
    }


def tactic_name(row: dict[str, Any]) -> str:
    return (f"{row['tile_m']}x{row['tile_n']}x{row['tactic_tile_k']}_"
            f"w{row['warp_m']}x{row['warp_n']}_s{row['stages']}_bc{row['bchunk']}")


def cell_id(identity: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(identity).encode()).hexdigest()


def algorithm_descriptors() -> list[tuple[str, int, str, str, int]]:
    return [
        ("non-persistent", 1, FULL, "non-persistent", 0),
        ("persistent", 1, FULL, "capacity+balanced", 0),
        ("scale-first-splitk", 2, PRODUCER, "fixed-split-k", 0),
        ("scale-first-splitk", 4, PRODUCER, "fixed-split-k", 0),
        ("scale-first-splitk", 8, PRODUCER, "fixed-split-k", 0),
    ]


def base_cell(plan_cell: dict[str, Any], fmt: matrix.Format, artifact: int,
              row: dict[str, Any], algorithm: str, split: int, scope: str,
              policy: str, grid: int) -> dict[str, Any]:
    shape = {name: plan_dim(plan_cell, name) for name in ("m", "n", "k")}
    shape["l"] = int(plan_cell.get("l", plan_cell.get("L", 1)))
    layout = layout_for(fmt, artifact)
    config = tactic_name(row)
    workload = workload_metadata(plan_cell, fmt.group_size)
    identity = {
        "component": "scale_first", **workload, "qtype": fmt.qtype,
        "shape": shape, "layout": layout, "algorithm": algorithm,
        "S": split, "config": config, "policy": policy, "grid": grid,
    }
    return {
        "cell_id": cell_id(identity), "tensor": primary_tensor(plan_cell),
        "qtype": fmt.qtype, "format": fmt.name, "shape": shape,
        "layout": layout, "FoldN": [layout["fold_n"]["low"],
                                      layout["fold_n"]["high"]],
        "ArtifactTileK": artifact, "algorithm": algorithm, "S": split,
        "metric_scope": scope, "config": config, "policy": policy,
        "grid": grid, "workspace_bytes": 0, "route": "",
        **workload,
    }


def unsupported_cells(plan_cell: dict[str, Any], reason: str,
                      fmt: matrix.Format | None = None) -> list[dict[str, Any]]:
    qtype = int(plan_cell["qtype"])
    shape = {name: plan_dim(plan_cell, name) for name in ("m", "n", "k")}
    shape["l"] = int(plan_cell.get("l", plan_cell.get("L", 1)))
    workload = workload_metadata(plan_cell, fmt.group_size if fmt else None)
    tensor = primary_tensor(plan_cell)
    result = []
    for algorithm, split, scope, policy, grid in algorithm_descriptors():
        identity = {"component": "scale_first", **workload, "qtype": qtype,
                    "shape": shape, "algorithm": algorithm, "S": split,
                    "config": f"unsupported-scalefirst-qtype-{qtype}",
                    "policy": policy, "grid": grid}
        result.append({
            "cell_id": cell_id(identity), "tensor": tensor,
            "qtype": qtype, "format": plan_cell.get("format", f"qtype-{qtype}"),
            "shape": shape, "algorithm": algorithm, "S": split,
            "metric_scope": scope, "config": identity["config"],
            "policy": policy, "grid": grid, "status": "UNSUPPORTED",
            "reason": reason, "workspace_bytes": 0, "route": "", **workload,
        })
    return result


def generated_shards(plan_cells: list[dict[str, Any]]) -> set[str]:
    active = {int(cell["qtype"]) for cell in plan_cells
              if cell.get("inventory_status") == "SUPPORTED" and
              not is_grouped(cell) and int(cell["qtype"]) in VALID_QTYPES}
    return {f"q{qtype}-a{artifact}-bc{bchunk}"
            for qtype in active for artifact in matrix.ARTIFACT_TILE_K
            for bchunk in (0, 1)}


def _compact_rows_sha256(rows: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_generated_manifest(
        doc: dict[str, Any], shard: str, *,
        expected_raw_rows: int = matrix.RAW_ROWS_PER_PAIR // 2
        ) -> dict[str, Any]:
    """Accept historical v2 or an exact full-authority Xplane v3 shard.

    V3 also represents compact parent-range K-pack discovery shards. Those
    are not interchangeable with the exhaustive historical internal sweep:
    its shard key has no layout/range coordinate and its result layout is
    Xplane. Therefore this compatibility path admits only v3 authority-full.
    """
    match = SHARD_RE.fullmatch(shard)
    if match is None:
        raise ValueError(f"{shard}: invalid generated shard key")
    qtype, artifact, bchunk = map(int, match.groups())
    schema = doc.get("schema")
    if schema not in (GENERATED_SHARD_V2, GENERATED_SHARD_V3):
        raise ValueError(f"{shard}: generated manifest schema mismatch")
    identity = doc.get("identity")
    expected_identity = {
        "qtype": qtype,
        "format": matrix.format_for(qtype).name,
        "artifact_tile_k": artifact,
        "bchunk": bchunk,
    }
    if identity != expected_identity:
        raise ValueError(
            f"{shard}: generated manifest identity/layout mismatch")
    typed = doc.get("typed_rows")
    rejected = doc.get("non_typed_rows")
    if not isinstance(typed, list) or not isinstance(rejected, list):
        raise ValueError(f"{shard}: generated rows are not arrays")
    if not all(isinstance(row, dict) for row in typed + rejected):
        raise ValueError(f"{shard}: generated row is not an object")
    denominator = doc.get("denominator") or {}
    if (denominator.get("raw_rows") != expected_raw_rows or
            denominator.get("typed_rows") != len(typed) or
            denominator.get("non_typed_rows") != len(rejected) or
            len(typed) + len(rejected) != expected_raw_rows):
        raise ValueError(f"{shard}: generated denominator drift")
    symbols = [row.get("symbol") for row in typed + rejected]
    if None in symbols or len(symbols) != len(set(symbols)):
        raise ValueError(f"{shard}: generated symbol duplicate/missing")

    if schema == GENERATED_SHARD_V3:
        if denominator.get("authority_typed_rows") != len(typed):
            raise ValueError(f"{shard}: v3 typed authority differs")
        expected_range = {
            "begin": 0, "end": len(typed), "count": len(typed),
            "authority_count": len(typed),
        }
        if doc.get("parent_range") != expected_range:
            raise ValueError(
                f"{shard}: v3 is not one full-authority parent range")
        expected_selection = {
            "mode": "authority-full", "begin": 0, "end": len(typed),
            "authority_typed_rows": len(typed),
            "compiled_rows": len(typed),
        }
        if doc.get("selection") != expected_selection:
            raise ValueError(f"{shard}: v3 selection authority differs")
        expected_parents = [
            {"parent_id": index, "symbol": row["symbol"]}
            for index, row in enumerate(typed)
        ]
        if doc.get("compiled_parents") != expected_parents or any(
                row.get("parent_id") != index
                for index, row in enumerate(typed)):
            raise ValueError(f"{shard}: v3 parent identity differs")
        expected_rejected = {
            "count": len(rejected),
            "sha256": _compact_rows_sha256(rejected),
            "encoding": "JSON_SORT_KEYS_COMPACT_V1",
        }
        if doc.get("non_typed_authority") != expected_rejected:
            raise ValueError(f"{shard}: v3 non-typed authority differs")
    return doc


def load_manifests(generated_root: pathlib.Path, expected: set[str]
                   ) -> dict[str, dict[str, Any]]:
    observed = {path.parent.name for path in generated_root.glob("*/manifest.json")}
    if observed != expected:
        raise ValueError("generated manifest set differs from plan graph: "
                         f"missing={sorted(expected-observed)} "
                         f"extra={sorted(observed-expected)}")
    manifests = {}
    for shard in sorted(expected):
        doc = json.loads((generated_root / shard / "manifest.json").read_text())
        manifests[shard] = _validate_generated_manifest(doc, shard)
    return manifests


def _same_except_sample(records: list[dict[str, Any]]) -> bool:
    ignored = {"sample", "sample_us", "MFU_pct", "distinct_MBU_model_pct"}
    first = {key: value for key, value in records[0].items() if key not in ignored}
    return all({key: value for key, value in record.items() if key not in ignored} == first
               for record in records[1:])


def _validated_runtime_cell(records: list[dict[str, Any]], iterations: int
                           ) -> dict[str, Any]:
    if not records or not _same_except_sample(records):
        raise ValueError("runtime samples disagree outside sample/metric fields")
    first = dict(records[0])
    status = first.get("status")
    if status == "MEASURED":
        if len(records) != iterations:
            raise ValueError(f"measured sample denominator {len(records)}/{iterations}")
        if sorted(int(row["sample"]) for row in records) != list(range(iterations)):
            raise ValueError("measured sample ordinals are not exact")
        samples = [float(row["sample_us"]) for row in records]
        if any(not math.isfinite(value) or value <= 0 for value in samples):
            raise ValueError("measured sample is non-positive/non-finite")
        if any(int(row.get("raw_bad", -1)) != 0 for row in records):
            raise ValueError("measured runtime row has raw mismatch")
        if len({row.get("fingerprint") for row in records}) != 1:
            raise ValueError("timed outputs are not bit-stable")
        first["raw_samples_us"] = samples
        first["median_us"] = statistics.median(samples)
    elif status == "INADMISSIBLE":
        if len(records) != 1 or int(first.get("sample", -1)) != 0 or \
                float(first.get("sample_us", -1)) != 0:
            raise ValueError("inadmissible runtime coordinate has timing samples")
        if not str(first.get("reason", "")).startswith("INADMISSIBLE_"):
            raise ValueError("inadmissible runtime coordinate lacks named reason")
    else:
        raise ValueError(f"unknown runtime terminal status {status!r}")
    return first


def _persistent_grid_space(q: int, cu: int, occupancy: int
                           ) -> list[tuple[int, int, int, str]]:
    if q <= 0 or cu <= 0 or occupancy <= 0 or occupancy > 63:
        return []
    grids: dict[int, list[int]] = {}
    ceil_div = lambda value, divisor: (value + divisor - 1) // divisor
    for b in range(1, occupancy + 1):
        wave = cu * b
        capacity = min(q, wave)
        balanced = ceil_div(q, ceil_div(q, wave))
        grids.setdefault(capacity, [0, 0])[0] |= 1 << b
        grids.setdefault(balanced, [0, 0])[1] |= 1 << b
    result = []
    for grid, (capacity_mask, balanced_mask) in sorted(grids.items()):
        policy = ("capacity+balanced" if capacity_mask and balanced_mask else
                  "capacity" if capacity_mask else "balanced")
        result.append((grid, capacity_mask, balanced_mask, policy))
    return result


def _validate_row_algorithms(cells: list[dict[str, Any]], *,
                             shape: tuple[int, int, int], cu: int,
                             tile_m: int, tile_n: int) -> None:
    counts = collections.Counter(cell["algorithm"] for cell in cells)
    if counts.get("NONPERSISTENT") != 1 or \
            counts.get("SPLITK_S2_PRODUCER") != 1 or \
            counts.get("SPLITK_S4_PRODUCER") != 1 or \
            counts.get("SPLITK_S8_PRODUCER") != 1 or \
            counts.get("PERSISTENT", 0) < 1 or set(counts) != {
                "NONPERSISTENT", "PERSISTENT", "SPLITK_S2_PRODUCER",
                "SPLITK_S4_PRODUCER", "SPLITK_S8_PRODUCER"}:
        raise ValueError(f"runtime algorithm denominator is not NP+P+S2/S4/S8: {counts}")
    persistent = [cell for cell in cells if cell["algorithm"] == "PERSISTENT"]
    occupancies = {int(cell.get("occupancy", -1)) for cell in persistent}
    if len(occupancies) != 1:
        raise ValueError(f"persistent occupancy is not unique: {sorted(occupancies)}")
    occupancy = occupancies.pop()
    if occupancy < 0 or occupancy > 63:
        raise ValueError(f"persistent occupancy is outside [0,63]: {occupancy}")
    q = ((shape[0] + tile_m - 1) // tile_m) * \
        ((shape[1] + tile_n - 1) // tile_n)
    expected = _persistent_grid_space(q, cu, occupancy)
    if not expected:
        expected = [(0, 0, 0, "capacity+balanced")]
    observed = []
    for cell in persistent:
        try:
            capacity_mask = int(str(cell.get("capacity_b_mask", "")), 0)
            balanced_mask = int(str(cell.get("balanced_b_mask", "")), 0)
        except ValueError as error:
            raise ValueError("persistent b-mask is malformed") from error
        observed.append((int(cell["grid"]), capacity_mask, balanced_mask,
                         str(cell["policy"]).lower()))
    if sorted(observed) != sorted(expected):
        raise ValueError(
            "persistent exact grid/mask denominator differs from "
            f"Q={q}/CU={cu}/occupancy={occupancy}: "
            f"missing={sorted(set(expected)-set(observed))} "
            f"extra={sorted(set(observed)-set(expected))}")


def read_runtime(raw_root: pathlib.Path,
                 manifests: dict[str, dict[str, Any]],
                 expected_shapes_by_q: dict[int, set[tuple[int, int, int]]],
                 iterations: int
                 ) -> tuple[dict[tuple[Any, ...], list[dict[str, Any]]],
                            list[dict[str, Any]], set[str]]:
    typed_shards = {name for name, doc in manifests.items()
                    if int(doc["denominator"]["typed_rows"]) > 0}
    observed = {path.parent.name for path in raw_root.glob("*/run.log")}
    if observed != typed_shards:
        raise ValueError("runtime shard set differs from typed graph: "
                         f"missing={sorted(typed_shards-observed)} "
                         f"extra={sorted(observed-typed_shards)}")
    runtime: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    for shard in sorted(typed_shards):
        match = SHARD_RE.fullmatch(shard)
        assert match is not None
        qtype, artifact, bchunk = map(int, match.groups())
        expected_shapes = expected_shapes_by_q[qtype]
        expected_symbols = {row["symbol"] for row in manifests[shard]["typed_rows"]}
        log = raw_root / shard / "run.log"
        rc_path = raw_root / shard / "run.rc"
        try:
            rc = int(rc_path.read_text().strip())
        except (OSError, ValueError):
            failures.append({"shard": shard, "reason": "missing/invalid run.rc"})
            continue
        if rc != 0:
            failures.append({"shard": shard, "reason": f"runner rc={rc}"})
        text = log.read_text(errors="replace")
        shard_lines = [line for line in text.splitlines()
                       if line.startswith(SHARD_PREFIX)]
        if len(shard_lines) != 1:
            failures.append({"shard": shard, "reason": "SF_SHARD count is not one"})
            continue
        try:
            header = parse_kv(shard_lines[0], SHARD_PREFIX)
            if (int(header["qtype"]), int(header["artifact_tile_k"]),
                    int(header["bchunk"]), int(header["typed_rows"])) != (
                        qtype, artifact, bchunk, len(expected_symbols)):
                raise ValueError("SF_SHARD identity/typed denominator mismatch")
            if int(header["iterations"]) != iterations:
                raise ValueError("SF_SHARD iteration denominator mismatch")
            header_cu = int(header["cu"])
            if header_cu <= 0:
                raise ValueError("SF_SHARD CU must be positive")
        except (KeyError, TypeError, ValueError) as error:
            failures.append({"shard": shard, "reason": str(error)})
            continue
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
        raw_count: collections.Counter[tuple[int, int, int]] = collections.Counter()
        for lineno, line in enumerate(text.splitlines(), 1):
            if not line.startswith(CELL_PREFIX):
                continue
            try:
                row = json.loads(line[len(CELL_PREFIX):])
                shape = parse_shape(str(row["shape"]))
                if shape not in expected_shapes:
                    raise ValueError(f"unexpected runtime shape {shape}")
                if (int(row["qtype"]), int(row["artifact_tile_k"]),
                        int(row["bchunk"])) != (qtype, artifact, bchunk):
                    raise ValueError("runtime parent identity mismatch")
                symbol = str(row["symbol"])
                if symbol not in expected_symbols:
                    raise ValueError(f"runtime symbol absent from manifest: {symbol}")
                key = (shape, symbol, str(row["algorithm"]),
                       str(row["metric_scope"]), str(row["policy"]),
                       int(row["split"]), int(row["grid"]))
                groups[key].append(row)
                raw_count[shape] += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                failures.append({"shard": shard, "line": lineno,
                                 "reason": str(error)})
        validated: dict[tuple[Any, ...], dict[str, Any]] = {}
        for key, records in groups.items():
            try:
                validated[key] = _validated_runtime_cell(records, iterations)
            except (KeyError, TypeError, ValueError) as error:
                failures.append({"shard": shard, "key": repr(key),
                                 "reason": str(error)})
        completes: dict[tuple[int, int, int], dict[str, str]] = {}
        for lineno, line in enumerate(text.splitlines(), 1):
            if not line.startswith(COMPLETE_PREFIX):
                continue
            try:
                row = parse_kv(line, COMPLETE_PREFIX)
                shape = parse_shape(row["shape"])
                if shape in completes:
                    raise ValueError(f"duplicate completion for {shape}")
                completes[shape] = row
            except (KeyError, ValueError) as error:
                failures.append({"shard": shard, "line": lineno,
                                 "reason": str(error)})
        if set(completes) != expected_shapes:
            failures.append({"shard": shard,
                             "reason": "SF_COMPLETE shape set mismatch",
                             "missing": sorted(expected_shapes-set(completes)),
                             "extra": sorted(set(completes)-expected_shapes)})
        for shape in expected_shapes:
            row_cells = [value for key, value in validated.items() if key[0] == shape]
            symbols = {str(value["symbol"]) for value in row_cells}
            if symbols != expected_symbols:
                failures.append({"shard": shard, "shape": shape,
                                 "reason": "typed symbol set mismatch",
                                 "missing": sorted(expected_symbols-symbols),
                                 "extra": sorted(symbols-expected_symbols)})
            by_symbol: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
            for cell in row_cells:
                by_symbol[str(cell["symbol"])].append(cell)
            manifest_by_symbol = {
                str(row["symbol"]): row for row in manifests[shard]["typed_rows"]}
            for symbol, cells in by_symbol.items():
                try:
                    manifest_row = manifest_by_symbol[symbol]
                    _validate_row_algorithms(
                        cells, shape=shape, cu=header_cu,
                        tile_m=int(manifest_row["tile_m"]),
                        tile_n=int(manifest_row["tile_n"]))
                except ValueError as error:
                    failures.append({"shard": shard, "shape": shape,
                                     "symbol": symbol, "reason": str(error)})
                runtime[(qtype, artifact, bchunk, shape, symbol)] = cells
            if shape in completes:
                completion = completes[shape]
                measured = sum(cell["status"] == "MEASURED" for cell in row_cells)
                records = raw_count[shape]
                expected_tuple = ("COMPLETE", len(expected_symbols), len(row_cells),
                                  measured, records, iterations,
                                  "ORDER-INDEPENDENT+FP16-EXACT", "PASS", "PASS")
                observed_tuple = (
                    completion.get("status"), int(completion.get("typed_rows", -1)),
                    int(completion.get("runtime_cells", -1)),
                    int(completion.get("measured_cells", -1)),
                    int(completion.get("records", -1)),
                    int(completion.get("iterations", -1)),
                    completion.get("fixture"), completion.get("roundtrip"),
                    completion.get("high_plane_coverage"))
                if observed_tuple != expected_tuple:
                    failures.append({"shard": shard, "shape": shape,
                                     "reason": "SF_COMPLETE denominator mismatch",
                                     "observed": observed_tuple,
                                     "expected": expected_tuple})
    return runtime, failures, typed_shards


def metric(shape: tuple[int, int, int], fmt: matrix.Format, split: int,
           median_us: float, partial_bytes: int, peak_tflops: float,
           hbm_gbs: float) -> tuple[float, float, str]:
    m, n, k = shape
    mfu = (2.0 * m * n * k) / median_us / (peak_tflops * 1e6) * 100.0
    weights = n * k * (fmt.low_bits + fmt.high_bits) / 8.0
    metadata = n * (k // fmt.group_size) * fmt.metadata_planes * 2.0
    if split == 1:
        bytes_total = weights + metadata + 2.0 * m * k + 2.0 * m * n
        kind = "DISTINCT_WEIGHT_SCALE_ZERO_PLUS_A_PLUS_FP16_D"
    else:
        bytes_total = weights + metadata + 2.0 * m * k + partial_bytes
        kind = "PRODUCER_DISTINCT_PLUS_FP32_PARTIAL_WRITE_MODEL"
    mbu = bytes_total / median_us / (hbm_gbs * 1000.0) * 100.0
    return mfu, mbu, kind


def terminal_from_runtime(plan_cell: dict[str, Any], fmt: matrix.Format,
                          artifact: int, manifest_row: dict[str, Any],
                          runtime: dict[str, Any], peak_tflops: float,
                          hbm_gbs: float) -> dict[str, Any]:
    raw_algorithm = str(runtime["algorithm"])
    if raw_algorithm == "NONPERSISTENT":
        algorithm, split, scope, policy = "non-persistent", 1, FULL, "non-persistent"
    elif raw_algorithm == "PERSISTENT":
        algorithm, split, scope, policy = "persistent", 1, FULL, str(runtime["policy"])
    else:
        match = re.fullmatch(r"SPLITK_S(2|4|8)_PRODUCER", raw_algorithm)
        if not match:
            raise ValueError(f"unknown ScaleFirst runtime algorithm {raw_algorithm}")
        split = int(match.group(1))
        algorithm, scope, policy = "scale-first-splitk", PRODUCER, "fixed-split-k"
    expected_scope = FULL if split == 1 else "PRODUCER_ONLY_NOT_PRODUCT_E2E"
    if runtime["metric_scope"] != expected_scope:
        raise ValueError(f"runtime scope {runtime['metric_scope']} contradicts {raw_algorithm}")
    grid = int(runtime["grid"])
    cell = base_cell(plan_cell, fmt, artifact, manifest_row, algorithm, split,
                     scope, policy, grid)
    status = str(runtime["status"])
    cell.update(status=status, reason=str(runtime.get("reason", "")),
                workspace_bytes=int(runtime.get("partial_bytes", 0)),
                occupancy=int(runtime.get("occupancy", 0)),
                capacity_b_mask=str(runtime.get("capacity_b_mask", "0x0")),
                balanced_b_mask=str(runtime.get("balanced_b_mask", "0x0")),
                shipping_smem=int(runtime.get("shipping_smem", 0)),
                persistent_smem=int(runtime.get("persistent_smem", 0)),
                split_smem=int(runtime.get("split_smem", 0)))
    if status == "MEASURED":
        samples = [float(value) for value in runtime["raw_samples_us"]]
        median = float(runtime["median_us"])
        shape = tuple(plan_dim(plan_cell, name) for name in ("m", "n", "k"))
        partial = int(runtime.get("partial_bytes", 0))
        mfu, mbu, kind = metric(shape, fmt, split, median, partial,
                                peak_tflops, hbm_gbs)
        if split > 1 and int(runtime.get("reducer_correctness_untimed", 0)) != 1:
            raise ValueError("measured Split-K producer lacks untimed reducer closure")
        cell.update(raw_samples_us=samples, median_us=median,
                    MFU_pct=mfu, distinct_MBU_model_pct=mbu,
                    mbu_kind=kind, correctness="RAW-FP16/PASS")
    return cell


def terminal_from_static(plan_cell: dict[str, Any], fmt: matrix.Format,
                         artifact: int, row: dict[str, Any]
                         ) -> list[dict[str, Any]]:
    raw_status = str(row["status"])
    if raw_status == "STATIC_REJECT":
        status = "INADMISSIBLE"
    elif raw_status == "UNSUPPORTED":
        status = "UNSUPPORTED"
    else:
        raise ValueError(f"non-runtime row has invalid status {raw_status!r}")
    result = []
    for algorithm, split, scope, policy, grid in algorithm_descriptors():
        cell = base_cell(plan_cell, fmt, artifact, row, algorithm, split,
                         scope, policy, grid)
        cell.update(status=status, reason=str(row["reason"]))
        result.append(cell)
    return result


def analyze(plan_path: pathlib.Path, generated_root: pathlib.Path,
            raw_root: pathlib.Path, output: pathlib.Path,
            identity_path: pathlib.Path, source_hashes_path: pathlib.Path,
            binary_hashes_path: pathlib.Path, raw_hashes_path: pathlib.Path,
            run_contract_path: pathlib.Path, attempt_id: str,
            peak_tflops: float, hbm_gbs: float) -> int:
    if not ATTEMPT_RE.fullmatch(attempt_id):
        raise ValueError("orchestration attempt ID is empty or malformed")
    plan = load_plan(plan_path)
    run_contract = json.loads(run_contract_path.read_text())
    if run_contract.get("schema") != RUN_CONTRACT_SCHEMA:
        raise ValueError("run-contract schema mismatch")
    iterations = int(run_contract.get("iterations", 0))
    if iterations <= 0:
        raise ValueError("run-contract iterations must be positive")
    if run_contract.get("plan_sha256") != sha256(plan_path):
        raise ValueError("run contract does not bind analyzed plan")
    if (float(run_contract.get("peak_tflops", 0)) != peak_tflops or
            float(run_contract.get("hbm_gbs", 0)) != hbm_gbs):
        raise ValueError("metric denominators differ from run contract")
    plan_cells = plan["cells"]
    expected_generated = generated_shards(plan_cells)
    manifests = load_manifests(generated_root, expected_generated)
    expected_shapes_by_q: dict[int, set[tuple[int, int, int]]] = collections.defaultdict(set)
    for plan_cell in plan_cells:
        qtype = int(plan_cell["qtype"])
        if (plan_cell.get("inventory_status") == "SUPPORTED" and
                not is_grouped(plan_cell) and qtype in VALID_QTYPES):
            expected_shapes_by_q[qtype].add(tuple(
                plan_dim(plan_cell, name) for name in ("m", "n", "k")))
    runtime, failures, typed_shards = read_runtime(
        raw_root, manifests, expected_shapes_by_q, iterations)
    consumed: set[tuple[Any, ...]] = set()
    missing: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    expected = 0
    formats = {fmt.qtype: fmt for fmt in matrix.FORMATS}
    for plan_cell in plan_cells:
        qtype = int(plan_cell["qtype"])
        fmt = formats.get(qtype)
        if plan_cell.get("inventory_status") != "SUPPORTED":
            terminal = unsupported_cells(
                plan_cell, "INVENTORY_" + str(plan_cell.get("inventory_reason")), fmt)
            cells.extend(terminal); expected += len(terminal); continue
        if is_grouped(plan_cell):
            terminal = unsupported_cells(
                plan_cell, "NO_GROUPED_SCALEFIRST_SWEEP_KERNEL", fmt)
            cells.extend(terminal); expected += len(terminal); continue
        if fmt is None:
            terminal = unsupported_cells(
                plan_cell, "QTYPE_NOT_IN_SCALEFIRST_REGISTRY")
            cells.extend(terminal); expected += len(terminal); continue
        shape = tuple(plan_dim(plan_cell, name) for name in ("m", "n", "k"))
        for artifact in matrix.ARTIFACT_TILE_K:
            for bchunk in (0, 1):
                shard = f"q{qtype}-a{artifact}-bc{bchunk}"
                manifest = manifests[shard]
                for row in manifest["non_typed_rows"]:
                    terminal = terminal_from_static(plan_cell, fmt, artifact, row)
                    cells.extend(terminal); expected += len(terminal)
                for row in manifest["typed_rows"]:
                    key = (qtype, artifact, bchunk, shape, row["symbol"])
                    records = runtime.get(key)
                    if records is None:
                        # Persistent expansion is device-owned, but NP and the
                        # three fixed Split-K coordinates make absence visible.
                        missing.append({"runtime_key": repr(key),
                                        "reason": "COMPILED_ADMISSION_RECORD_MISSING"})
                        continue
                    consumed.add(key)
                    for record in records:
                        try:
                            cells.append(terminal_from_runtime(
                                plan_cell, fmt, artifact, row, record,
                                peak_tflops, hbm_gbs))
                            expected += 1
                        except (KeyError, TypeError, ValueError) as error:
                            failures.append({"runtime_key": repr(key),
                                             "reason": str(error)})
    extras = sorted(set(runtime) - consumed, key=repr)
    if extras:
        failures.append({"reason": "unconsumed runtime rows",
                         "examples": list(map(repr, extras[:8]))})
    ids = [cell["cell_id"] for cell in cells]
    duplicates = [key for key, count in collections.Counter(ids).items() if count != 1]
    if duplicates:
        failures.append({"reason": "duplicate cell IDs", "examples": duplicates[:8]})
    if len(cells) != expected:
        failures.append({"reason": f"denominator mismatch {len(cells)}/{expected}"})
    nonterminal = [cell for cell in cells if cell["status"] not in
                   {"MEASURED", "INADMISSIBLE", "UNSUPPORTED", "BUILD_REJECT"}]

    source_document = json.loads(source_hashes_path.read_text())
    root_sha = str(source_document.get("root_sha", ""))
    actlize_sha = str(source_document.get("actlize_sha", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", root_sha) or not re.fullmatch(
            r"[0-9a-f]{40}", actlize_sha):
        raise ValueError("source authority root/actlize SHA invalid")
    source_hashes = source_document.get("source_hashes")
    if not isinstance(source_hashes, dict) or set(source_hashes) != EXPECTED_SOURCE_HASHES:
        raise ValueError("source-hash authority set mismatch: "
                         f"missing={sorted(EXPECTED_SOURCE_HASHES-set(source_hashes or {}))} "
                         f"extra={sorted(set(source_hashes or {})-EXPECTED_SOURCE_HASHES)}")
    for name, digest in source_hashes.items():
        require_sha256(digest, f"source_hashes.{name}")
    generated_hashes = source_document.get("generated_shards")
    if not isinstance(generated_hashes, dict) or set(generated_hashes) != expected_generated:
        raise ValueError("generated source-hash shard set mismatch")
    for name, digest in generated_hashes.items():
        require_sha256(digest, f"generated_shards.{name}")
    published_hashes = dict(source_hashes)
    published_hashes.update({f"generated/{name}": digest
                             for name, digest in generated_hashes.items()})
    binary_hashes = json.loads(binary_hashes_path.read_text())
    if not isinstance(binary_hashes, dict) or set(binary_hashes) != typed_shards:
        raise ValueError("binary shard set differs from typed runtime graph")
    raw_hashes = json.loads(raw_hashes_path.read_text())
    if not isinstance(raw_hashes, dict) or set(raw_hashes) != typed_shards:
        raise ValueError("raw-log hash set differs from typed runtime graph")
    for mapping_name, mapping in (("binary", binary_hashes), ("raw", raw_hashes)):
        for name, digest in mapping.items():
            require_sha256(digest, f"{mapping_name}_hashes.{name}")
            path = ((raw_root / name / "run.log") if mapping_name == "raw" else None)
            if path is not None and sha256(path) != digest:
                raise ValueError(f"raw log changed after authority binding: {name}")
    runtime_hashes = runtime_authority(
        raw_root, typed_shards, run_contract_path, generated_hashes,
        binary_hashes, raw_hashes)
    identity = json.loads(identity_path.read_text())
    if run_contract.get("identity_sha256") != sha256(identity_path):
        raise ValueError("run contract does not bind device identity")
    plan_provenance = plan["provenance"]
    status = "COMPLETE" if not missing and not failures and not nonterminal else "INCOMPLETE"
    status_counts = collections.Counter(cell["status"] for cell in cells)
    summary = {
        "schema": SCHEMA, "component": "scale_first", "status": status,
        "expected_cells": expected, "status_counts": dict(status_counts),
        "missing": missing, "failures": failures,
        "measurement_contract": {
            "full_output": "ordinary non-persistent plus all exact persistent capacity/balanced grids",
            "producer_only": "fixed Split-K S2/S4/S8 excludes reducer timing and requires untimed deterministic raw-fp16 closure",
            "cross_scope_ranking": "forbidden",
        },
        "provenance": {
            "root_sha": root_sha, "actlize_sha": actlize_sha,
            "device": identity, "source_hashes": published_hashes,
            "generated_source_hashes": generated_hashes,
            "binary_hashes": binary_hashes, "raw_log_hashes": raw_hashes,
            "runtime_hashes": runtime_hashes,
            "run_contract": run_contract,
            "orchestration_attempt_id": attempt_id,
            "shape_manifest_sha256": require_sha256(
                plan_provenance.get("shape_manifest_sha256"),
                "plan.provenance.shape_manifest_sha256"),
            "gguf_hashes": plan_provenance.get("gguf_hashes"),
            "gguf_set_sha256": require_sha256(
                plan_provenance.get("gguf_set_sha256"),
                "plan.provenance.gguf_set_sha256"),
            "shape_directory": plan_provenance.get("shape_directory"),
        },
        "cells": cells,
    }
    atomic_json(output, summary, pretty=False)
    print(f"[scalefirst-internal] status={status} denominator={expected} "
          f"measured={status_counts.get('MEASURED',0)} "
          f"inadmissible={status_counts.get('INADMISSIBLE',0)} "
          f"unsupported={status_counts.get('UNSUPPORTED',0)} "
          f"missing={len(missing)} failures={len(failures)} summary={output}")
    return 0 if status == "COMPLETE" else 1


def _manifest_schema_self_test() -> None:
    identity = {
        "qtype": 12, "format": matrix.format_for(12).name,
        "artifact_tile_k": 64, "bchunk": 0,
    }
    typed_v2 = [{"symbol": "typed"}]
    rejected = [{"symbol": "rejected"}]
    v2 = {
        "schema": GENERATED_SHARD_V2, "identity": identity,
        "denominator": {"raw_rows": 2, "typed_rows": 1,
                        "non_typed_rows": 1},
        "typed_rows": typed_v2, "non_typed_rows": rejected,
    }
    _validate_generated_manifest(v2, "q12-a64-bc0", expected_raw_rows=2)
    typed_v3 = [{"symbol": "typed", "parent_id": 0}]
    v3 = {
        "schema": GENERATED_SHARD_V3, "identity": identity,
        "denominator": {"raw_rows": 2, "typed_rows": 1,
                        "authority_typed_rows": 1, "non_typed_rows": 1},
        "parent_range": {"begin": 0, "end": 1, "count": 1,
                         "authority_count": 1},
        "selection": {"mode": "authority-full", "begin": 0, "end": 1,
                      "authority_typed_rows": 1, "compiled_rows": 1},
        "compiled_parents": [{"parent_id": 0, "symbol": "typed"}],
        "typed_rows": typed_v3, "non_typed_rows": rejected,
        "non_typed_authority": {
            "count": 1, "sha256": _compact_rows_sha256(rejected),
            "encoding": "JSON_SORT_KEYS_COMPACT_V1",
        },
    }
    _validate_generated_manifest(v3, "q12-a64-bc0", expected_raw_rows=2)
    plants = []
    broken = dict(v3, schema="quactlize.scalefirst.generated_shard.v4")
    plants.append(broken)
    broken = dict(v3, parent_range={"begin": 0, "end": 1, "count": 1,
                                    "authority_count": 2})
    plants.append(broken)
    broken = dict(v3, non_typed_authority={
        "count": 1, "sha256": "0" * 64,
        "encoding": "JSON_SORT_KEYS_COMPACT_V1"})
    plants.append(broken)
    for broken in plants:
        try:
            _validate_generated_manifest(
                broken, "q12-a64-bc0", expected_raw_rows=2)
        except ValueError:
            pass
        else:
            raise AssertionError("generated manifest schema plant stayed green")


def self_test() -> int:
    _manifest_schema_self_test()
    # One typed row with a deduplicated capacity+balanced persistent cell.
    common = {
        "shape": "2048x4096x4096", "qtype": 12, "artifact_tile_k": 32,
        "bchunk": 0, "symbol": "row", "config": "8x128x256_w8x16_s3_bc0",
        "status": "MEASURED", "reason": "MEASURED", "sample": 0,
        "sample_us": 2.0, "MFU_pct": 1.0, "distinct_MBU_model_pct": 2.0,
        "raw_bad": 0, "fingerprint": "0x1", "partial_bytes": 0,
        "shipping_smem": 1, "persistent_smem": 1, "split_smem": 1,
        "occupancy": 8, "capacity_b_mask": "0x1", "balanced_b_mask": "0x1",
        "reducer_correctness_untimed": 0,
    }
    records = []
    def add(algorithm: str, scope: str, policy: str, split: int, grid: int,
            reducer: int = 0) -> None:
        records.append(dict(common, algorithm=algorithm, metric_scope=scope,
                            policy=policy, split=split, grid=grid,
                            reducer_correctness_untimed=reducer))
    add("NONPERSISTENT", FULL, "ordinary", 1, 32)
    for grid, capacity_mask, balanced_mask, policy in \
            _persistent_grid_space(2048, 72, 8):
        row = dict(common, algorithm="PERSISTENT", metric_scope=FULL,
                   policy=policy, split=1, grid=grid,
                   capacity_b_mask=hex(capacity_mask),
                   balanced_b_mask=hex(balanced_mask))
        records.append(row)
    for split in (2, 4, 8):
        add(f"SPLITK_S{split}_PRODUCER", "PRODUCER_ONLY_NOT_PRODUCT_E2E",
            "fixed-split-k", split, 32 * split, 1)
    validated = [_validated_runtime_cell([row], 1) for row in records]
    _validate_row_algorithms(validated, shape=(2048, 4096, 4096), cu=72,
                             tile_m=64, tile_n=64)
    # A compiled TM8 type is decode-only.  Prefill retains the exact five
    # algorithm coordinates as named inadmissible terminals rather than
    # launching the physical-A specialization or dropping denominator rows.
    m8_records = []
    for algorithm, scope, policy, split in (
            ("NONPERSISTENT", FULL, "ordinary", 1),
            ("PERSISTENT", FULL, "capacity+balanced", 1),
            ("SPLITK_S2_PRODUCER", "PRODUCER_ONLY_NOT_PRODUCT_E2E",
             "fixed-split-k", 2),
            ("SPLITK_S4_PRODUCER", "PRODUCER_ONLY_NOT_PRODUCT_E2E",
             "fixed-split-k", 4),
            ("SPLITK_S8_PRODUCER", "PRODUCER_ONLY_NOT_PRODUCT_E2E",
             "fixed-split-k", 8)):
        m8_records.append(dict(
            common, algorithm=algorithm, metric_scope=scope, policy=policy,
            split=split, grid=0, occupancy=0, capacity_b_mask="0x0",
            balanced_b_mask="0x0", status="INADMISSIBLE",
            reason="INADMISSIBLE_M8_DECODE_ONLY", sample_us=0.0))
    m8_validated = [_validated_runtime_cell([row], 1) for row in m8_records]
    _validate_row_algorithms(m8_validated, shape=(2048, 4096, 4096), cu=72,
                             tile_m=8, tile_n=128)
    planted_m8_reason = dict(m8_records[0], reason="MEASURED")
    try:
        _validated_runtime_cell([planted_m8_reason], 1)
    except ValueError:
        pass
    else:
        raise AssertionError("unnamed TM8 prefill exclusion stayed green")
    # Negative 1: missing S4 must be red even when arithmetic is otherwise closed.
    try:
        _validate_row_algorithms(
            [row for row in validated if row["algorithm"] != "SPLITK_S4_PRODUCER"],
            shape=(2048, 4096, 4096), cu=72, tile_m=64, tile_n=64)
    except ValueError:
        pass
    else:
        raise AssertionError("missing S4 was accepted")
    # Negative 2: one-bit permutation/fixture corruption is a raw mismatch.
    planted = dict(common, raw_bad=1)
    try:
        _validated_runtime_cell([planted], 1)
    except ValueError:
        pass
    else:
        raise AssertionError("raw mismatch was accepted")
    # Negative 3: an extra algorithm cannot enlarge the denominator.
    extra = dict(validated[0], algorithm="SPLITK_S16_PRODUCER")
    try:
        _validate_row_algorithms(validated + [extra],
                                 shape=(2048, 4096, 4096), cu=72,
                                 tile_m=64, tile_n=64)
    except ValueError:
        pass
    else:
        raise AssertionError("extra algorithm was accepted")
    # Negative 4/5: a missing occupancy coordinate or wrong grid must remain
    # red even if SF_COMPLETE is forged to match the shortened record count.
    persistent_indexes = [i for i, row in enumerate(validated)
                          if row["algorithm"] == "PERSISTENT"]
    planted_drop = [row for i, row in enumerate(validated)
                    if i != persistent_indexes[-1]]
    try:
        _validate_row_algorithms(planted_drop, shape=(2048, 4096, 4096),
                                 cu=72, tile_m=64, tile_n=64)
    except ValueError:
        pass
    else:
        raise AssertionError("missing persistent grid stayed green")
    planted_grid = [dict(row) for row in validated]
    planted_grid[persistent_indexes[0]]["grid"] += 1
    try:
        _validate_row_algorithms(planted_grid, shape=(2048, 4096, 4096),
                                 cu=72, tile_m=64, tile_n=64)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong persistent grid stayed green")
    print("[scalefirst-internal:self-test] PASS exact NP/P(all-b)/S2/S4/S8; "
          "TM8-prefill=5xINADMISSIBLE_M8_DECODE_ONLY; "
          "generated-manifest=v2-history+v3-full-authority; "
          "negative=unnamed-TM8+missing-S4+raw-bit+extra-algorithm+"
          "drop-P-grid+wrong-grid")
    return 0


def list_plan(path: pathlib.Path) -> int:
    plan = load_plan(path)
    for cell in plan["cells"]:
        print("\t".join((str(cell["shape_id"]), str(cell["qtype"]),
                         str(plan_dim(cell, "m")), str(plan_dim(cell, "n")),
                         str(plan_dim(cell, "k")), str(cell["model_id"]),
                         str(cell["problem_route"]),
                         str(cell.get("inventory_status", "UNSUPPORTED")))))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--materialize-plan", type=pathlib.Path)
    parser.add_argument("--materialized-output", type=pathlib.Path)
    parser.add_argument("--validate-plan", type=pathlib.Path)
    parser.add_argument("--gguf-set", type=pathlib.Path)
    parser.add_argument("--list-plan", type=pathlib.Path)
    parser.add_argument("--plan", type=pathlib.Path)
    parser.add_argument("--generated-root", type=pathlib.Path)
    parser.add_argument("--raw-root", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--identity", type=pathlib.Path)
    parser.add_argument("--source-hashes", type=pathlib.Path)
    parser.add_argument("--binary-hashes", type=pathlib.Path)
    parser.add_argument("--raw-log-hashes", type=pathlib.Path)
    parser.add_argument("--run-contract", type=pathlib.Path)
    parser.add_argument("--validate-bind-binary", type=pathlib.Path)
    parser.add_argument("--binary-shard")
    parser.add_argument("--binary-evidence", choices=("0", "1"))
    parser.add_argument("--attempt-id")
    parser.add_argument("--peak-tflops", type=float, default=500.0)
    parser.add_argument("--hbm-gbs", type=float, default=2766.0)
    args = parser.parse_args()
    try:
        if args.self_test:
            return self_test()
        if args.materialize_plan:
            if args.materialized_output is None:
                parser.error("--materialize-plan requires --materialized-output")
            materialize_plan(args.materialize_plan, args.materialized_output)
            return 0
        if args.validate_plan:
            if args.gguf_set is None:
                parser.error("--validate-plan requires --gguf-set")
            validate_resolved_models(load_plan(args.validate_plan), args.gguf_set)
            return 0
        if args.list_plan:
            return list_plan(args.list_plan)
        if args.validate_bind_binary:
            if args.binary_hashes is None or args.binary_shard is None or \
                    args.binary_evidence is None:
                parser.error("--validate-bind-binary requires --binary-hashes, "
                             "--binary-shard, and --binary-evidence")
            validate_bind_binary(args.validate_bind_binary, args.binary_hashes,
                                 args.binary_shard,
                                 args.binary_evidence == "1")
            return 0
        required = (args.plan, args.generated_root, args.raw_root, args.output,
                    args.identity, args.source_hashes, args.binary_hashes,
                    args.raw_log_hashes, args.run_contract, args.attempt_id)
        if any(value is None for value in required):
            parser.error("analysis requires plan/generated/raw/output/identity/hash/contract/attempt arguments")
        return analyze(args.plan, args.generated_root, args.raw_root, args.output,
                       args.identity, args.source_hashes, args.binary_hashes,
                       args.raw_log_hashes, args.run_contract, args.attempt_id,
                       args.peak_tflops, args.hbm_gbs)
    except (AssertionError, KeyError, OSError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        print(f"[scalefirst-internal] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
