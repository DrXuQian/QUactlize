#!/usr/bin/env python3
"""Validate and summarize the exact FullyQuantized Split-K reducer census.

The analyzer treats the three raw device logs as evidence, not as a loose
benchmark stream.  It recomputes each seeded case order, requires every case
and every raw timing sample exactly once, and rejects any missing or nonzero
raw-bit correctness result before emitting the (M,N,S) lookup table.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import plan_fq_splitk_reducer_lookup as reducer_plan
import box_identity_schema
import kpack_discovery_worker_plan as worker_plan


SCHEMA = "quactlize.fq-splitk-reducer-lookup-result.v1"
AUTHORITY_SCHEMA = "quactlize.fq-splitk-reducer-result-authority.v1"
EXECUTION_SCHEMA = "quactlize.fq-splitk-reducer-execution-authority.v1"
BEGIN_PREFIX = "FQ_REDUCER_LOOKUP_RUN "
SAMPLE_PREFIX = "FQ_REDUCER_LOOKUP_SAMPLE "
CASE_PREFIX = "FQ_REDUCER_LOOKUP_CASE "
DONE_PREFIX = "FQ_REDUCER_LOOKUP_DONE "
HEX64 = re.compile(r"[0-9a-f]{64}")
UINT64_MASK = (1 << 64) - 1


class AnalyzeError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")).hexdigest()


def read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise AnalyzeError(f"required plain JSON file is absent: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnalyzeError(f"cannot read {path}: {error}") from error


def parse_fields(line: str, prefix: str) -> dict[str, str]:
    if not line.startswith(prefix):
        raise AnalyzeError(f"record prefix differs: {line[:80]!r}")
    result: dict[str, str] = {}
    for token in line[len(prefix):].split():
        if "=" not in token:
            raise AnalyzeError(f"record token has no '=': {token!r}")
        key, value = token.split("=", 1)
        if not key or not value or key in result:
            raise AnalyzeError(f"record token is empty or duplicated: {token!r}")
        result[key] = value
    return result


def require_keys(row: dict[str, str], expected: set[str], label: str) -> None:
    if set(row) != expected:
        missing = sorted(expected - set(row))
        extra = sorted(set(row) - expected)
        raise AnalyzeError(f"{label} fields differ: missing={missing} extra={extra}")


def integer(value: str, label: str, *, minimum: int = 0) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as error:
        raise AnalyzeError(f"{label} is not an integer: {value!r}") from error
    if parsed < minimum:
        raise AnalyzeError(f"{label} is below {minimum}: {parsed}")
    return parsed


def timing(value: str, label: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise AnalyzeError(f"{label} is not a decimal: {value!r}") from error
    if not parsed.is_finite() or parsed <= 0:
        raise AnalyzeError(f"{label} is not positive and finite: {value!r}")
    return parsed


def splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & UINT64_MASK
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return state, (value ^ (value >> 31)) & UINT64_MASK


def seeded_order(count: int, seed: int) -> list[int]:
    order = list(range(count))
    state = seed & UINT64_MASK
    for remaining in range(count, 1, -1):
        state, value = splitmix64(state)
        selected = value % remaining
        order[remaining - 1], order[selected] = order[selected], order[remaining - 1]
    return order


def order_hash(order: list[int]) -> int:
    value = 1469598103934665603
    for ordinal in order:
        for byte in int(ordinal).to_bytes(4, "little", signed=False):
            value ^= byte
            value = (value * 1099511628211) & UINT64_MASK
    return value


BEGIN_KEYS = {
    "schema", "plan_sha256", "total_cases", "case_begin", "case_end",
    "selected_cases", "round", "warmups", "samples", "schedule_seed",
    "order_hash", "partial_dtype", "output_dtype", "reducer", "fixture",
    "plant_output_fault", "status",
}
SAMPLE_KEYS = {
    "ordinal", "execution_ordinal", "case_id", "round", "sample", "M",
    "N", "S", "implementation", "us",
}
CASE_KEYS = {
    "ordinal", "execution_ordinal", "case_id", "M", "N", "S",
    "partial_dtype", "output_dtype", "implementation", "workspace_bytes",
    "output_bytes", "grid_ctas", "block_threads", "round", "warmups",
    "samples", "raw_bad", "first_bad", "median_us", "min_us", "max_us",
    "status",
}
DONE_KEYS = {
    "plan_sha256", "selected_cases", "measured", "failures", "round",
    "warmups", "samples", "schedule_seed", "order_hash", "status",
}


def expected_rounds(plan: dict[str, Any]) -> list[tuple[int, int]]:
    rows = plan["measurement"]["rounds"]
    expected = [(index + 1, reducer_plan.ROUND_SEEDS[index])
                for index in range(len(reducer_plan.ROUND_SEEDS))]
    parsed = [(int(row["round"]), int(row["schedule_seed"], 0)) for row in rows]
    if parsed != expected or len(parsed) != 3:
        raise AnalyzeError("plan three-round seed contract differs")
    return parsed


def runtime_linkage(binary: Path) -> list[list[Any]]:
    if binary.is_symlink() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise AnalyzeError("reducer binary is absent, symlinked, or not executable")
    result = subprocess.run(
        ["ldd", str(binary)], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise AnalyzeError(f"ldd failed for reducer binary: {result.stdout[-1000:]}")
    rows = []
    for line in result.stdout.splitlines():
        match = re.match(r"\s*(libhggc\S*)\s+=>\s+(\S+)", line)
        if not match:
            continue
        reported = match.group(2)
        try:
            resolved = Path(reported).resolve(strict=True)
        except OSError as error:
            raise AnalyzeError(f"cannot resolve runtime library {reported}: {error}") from error
        if not resolved.is_file():
            raise AnalyzeError(f"runtime library is not a file: {resolved}")
        rows.append([match.group(1), reported, str(resolved),
                     resolved.stat().st_size, sha256_file(resolved)])
    rows.sort()
    if not rows or len({row[0] for row in rows}) != len(rows):
        raise AnalyzeError("reducer libhggc runtime linkage is empty or duplicated")
    return rows


def write_execution_authority(
        manifest_path: Path, plan_path: Path, identity_path: Path,
        homogeneity_path: Path, binary: Path, worker_id: int,
        visible_device: str, output: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    plan = read_json(plan_path)
    reducer_plan.validate_plan(plan)
    identity = read_json(identity_path)
    try:
        box_identity_schema.validate(identity)
    except box_identity_schema.IdentityProbeError as error:
        raise AnalyzeError(f"device identity is malformed: {error}") from error
    probe = identity["device_probe"]
    if probe["device_count"] != 1 or probe["status"] not in {
            "measured", "properties-unavailable"}:
        raise AnalyzeError("reducer requires one uniquely selected device")
    homogeneity = read_json(homogeneity_path)
    workers = homogeneity.get("workers") if isinstance(homogeneity, dict) else None
    if not isinstance(workers, list):
        raise AnalyzeError("device-homogeneity worker list is missing")
    try:
        identities = worker_plan.validate_device_authority(
            homogeneity, len(workers))
    except worker_plan.PlanError as error:
        raise AnalyzeError(f"device homogeneity differs: {error}") from error
    identity_sha = sha256_file(identity_path)
    if worker_id not in identities or identities[worker_id] != identity_sha:
        raise AnalyzeError("reducer device identity is not its homogeneity worker entry")
    worker_row = next(row for row in workers if row["worker_id"] == worker_id)
    if not visible_device.isdecimal():
        raise AnalyzeError("visible device ordinal must be one decimal integer")
    if manifest.get("schema") != "quactlize.fq-splitk-reducer-prebuilt.v1":
        raise AnalyzeError("reducer bundle manifest schema differs")
    artifact = manifest.get("artifacts", {}).get("binary", {})
    if artifact.get("sha256") != sha256_file(binary) or \
            artifact.get("size") != binary.stat().st_size:
        raise AnalyzeError("reducer binary differs from bundle manifest")
    if manifest.get("build", {}).get("plan_sha256") != reducer_plan.digest(plan):
        raise AnalyzeError("reducer plan differs from bundle manifest")
    linkage = runtime_linkage(binary)
    value = {
        "schema": EXECUTION_SCHEMA,
        "source_commit": manifest["source"]["commit"],
        "source_tree": manifest["source"]["tree"],
        "sdk_authority_sha256": canonical_sha(manifest["sdk"]),
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "binary_sha256": artifact["sha256"],
        "build_authority_sha256": manifest["build"]["build_authority_sha256"],
        "plan_file_sha256": sha256_file(plan_path),
        "plan_sha256": reducer_plan.digest(plan),
        "worker_id": worker_id,
        "visible_device_ordinal": visible_device,
        "device_identity_sha256": identity_sha,
        "device_homogeneity_sha256": sha256_file(homogeneity_path),
        "device_homogeneity_key": worker_row["homogeneity_key"],
        "runtime_linkage": linkage,
        "runtime_linkage_sha256": worker_plan.digest(linkage),
        "measurement": {
            "rounds": plan["measurement"]["rounds"],
            "warmups": reducer_plan.DEFAULT_WARMUPS,
            "samples": reducer_plan.DEFAULT_SAMPLES,
            "case_order": plan["measurement"]["case_order"],
            "raw_bit_correctness": "EVERY_OUTPUT_ELEMENT_BEFORE_AND_AFTER_TIMING",
            "top_n": None,
            "point_estimate_pruning": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new(output, json.dumps(value, indent=2, sort_keys=True,
                                  allow_nan=False) + "\n")
    return value


def validate_execution_authority(
        value: Any, manifest: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "source_commit", "source_tree", "sdk_authority_sha256",
        "bundle_manifest_sha256", "binary_sha256", "build_authority_sha256",
        "plan_file_sha256", "plan_sha256", "worker_id",
        "visible_device_ordinal", "device_identity_sha256",
        "device_homogeneity_sha256", "device_homogeneity_key",
        "runtime_linkage", "runtime_linkage_sha256", "measurement",
    }
    if not isinstance(value, dict) or set(value) != required or \
            value.get("schema") != EXECUTION_SCHEMA:
        raise AnalyzeError("execution authority schema differs")
    for key in ("sdk_authority_sha256", "bundle_manifest_sha256", "binary_sha256",
                "build_authority_sha256", "plan_file_sha256", "plan_sha256",
                "device_identity_sha256", "device_homogeneity_sha256",
                "device_homogeneity_key", "runtime_linkage_sha256"):
        if not isinstance(value.get(key), str) or not HEX64.fullmatch(value[key]):
            raise AnalyzeError(f"execution authority {key} is malformed")
    if value["source_commit"] != manifest["source"]["commit"] or \
            value["source_tree"] != manifest["source"]["tree"] or \
            value["sdk_authority_sha256"] != canonical_sha(manifest["sdk"]) or \
            value["binary_sha256"] != manifest["artifacts"]["binary"]["sha256"] or \
            value["build_authority_sha256"] != manifest["build"]["build_authority_sha256"] or \
            value["plan_sha256"] != reducer_plan.digest(plan) or \
            value["runtime_linkage_sha256"] != worker_plan.digest(value["runtime_linkage"]):
        raise AnalyzeError("execution authority source/SDK/binary/plan binding differs")
    if value["measurement"] != {
            "rounds": plan["measurement"]["rounds"],
            "warmups": 3, "samples": 11,
            "case_order": plan["measurement"]["case_order"],
            "raw_bit_correctness": "EVERY_OUTPUT_ELEMENT_BEFORE_AND_AFTER_TIMING",
            "top_n": None, "point_estimate_pruning": False}:
        raise AnalyzeError("execution authority measurement contract differs")
    return value


def _case_identity(record: dict[str, str], expected: dict[str, Any],
                   round_number: int, execution: int) -> None:
    checks = {
        "ordinal": expected["ordinal"],
        "execution_ordinal": execution,
        "M": expected["m"],
        "N": expected["n"],
        "S": expected["split"],
        "round": round_number,
    }
    for key, want in checks.items():
        if integer(record[key], key) != want:
            raise AnalyzeError(
                f"case {expected['case_id']} {key} differs: {record[key]} != {want}")
    if record["case_id"] != expected["case_id"] or \
            record["implementation"] != expected["expected_implementation"]:
        raise AnalyzeError(f"case identity differs for {expected['case_id']}")


def parse_round(path: Path, plan: dict[str, Any], round_number: int,
                seed: int) -> dict[int, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise AnalyzeError(f"round log is absent or symlinked: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if any(line.startswith("FQ_REDUCER_LOOKUP_ERROR ") for line in lines):
        raise AnalyzeError(f"round {round_number} contains a reducer error")
    begins = [parse_fields(line, BEGIN_PREFIX) for line in lines
              if line.startswith(BEGIN_PREFIX)]
    dones = [parse_fields(line, DONE_PREFIX) for line in lines
             if line.startswith(DONE_PREFIX)]
    if len(begins) != 1 or len(dones) != 1:
        raise AnalyzeError(
            f"round {round_number} requires one BEGIN and one DONE marker")
    begin, done = begins[0], dones[0]
    require_keys(begin, BEGIN_KEYS, "BEGIN")
    require_keys(done, DONE_KEYS, "DONE")

    plan_sha = reducer_plan.digest(plan)
    count = len(plan["cases"])
    order = seeded_order(count, seed)
    hash_value = order_hash(order)
    common = {
        "plan_sha256": plan_sha,
        "selected_cases": str(count),
        "round": str(round_number),
        "warmups": str(reducer_plan.DEFAULT_WARMUPS),
        "samples": str(reducer_plan.DEFAULT_SAMPLES),
        "schedule_seed": f"0x{seed:016x}",
        "order_hash": f"0x{hash_value:016x}",
    }
    for key, want in common.items():
        if begin[key] != want or done[key] != want:
            raise AnalyzeError(f"round {round_number} {key} differs")
    begin_exact = {
        "schema": reducer_plan.SCHEMA,
        "total_cases": str(count),
        "case_begin": "0",
        "case_end": str(count),
        "partial_dtype": "fp32",
        "output_dtype": "fp16",
        "reducer": "M1FastReductionE2",
        "fixture": "period31-plus-rounding257-v1",
        "plant_output_fault": "0",
        "status": "BEGIN",
    }
    done_exact = {
        "measured": str(count), "failures": "0", "status": "PASS",
    }
    if any(begin[key] != want for key, want in begin_exact.items()) or \
            any(done[key] != want for key, want in done_exact.items()):
        raise AnalyzeError(f"round {round_number} completion contract differs")

    samples: dict[int, dict[int, Decimal]] = {}
    cases: dict[int, dict[str, str]] = {}
    sample_execution: dict[int, int] = {}
    for line in lines:
        if line.startswith(SAMPLE_PREFIX):
            row = parse_fields(line, SAMPLE_PREFIX)
            require_keys(row, SAMPLE_KEYS, "SAMPLE")
            ordinal = integer(row["ordinal"], "sample ordinal")
            execution = integer(row["execution_ordinal"], "sample execution")
            sample = integer(row["sample"], "sample")
            if execution >= count or order[execution] != ordinal:
                raise AnalyzeError("sample execution order differs from fixed seed")
            expected = plan["cases"][ordinal]
            _case_identity(row, expected, round_number, execution)
            if sample >= reducer_plan.DEFAULT_SAMPLES:
                raise AnalyzeError("sample index is outside [0,11)")
            if ordinal in sample_execution and sample_execution[ordinal] != execution:
                raise AnalyzeError("one case has multiple execution ordinals")
            sample_execution[ordinal] = execution
            case_samples = samples.setdefault(ordinal, {})
            if sample in case_samples:
                raise AnalyzeError("duplicate raw timing sample")
            case_samples[sample] = timing(row["us"], "sample us")
        elif line.startswith(CASE_PREFIX):
            row = parse_fields(line, CASE_PREFIX)
            require_keys(row, CASE_KEYS, "CASE")
            ordinal = integer(row["ordinal"], "case ordinal")
            execution = integer(row["execution_ordinal"], "case execution")
            if ordinal >= count or execution >= count or order[execution] != ordinal:
                raise AnalyzeError("case execution order differs from fixed seed")
            _case_identity(row, plan["cases"][ordinal], round_number, execution)
            if ordinal in cases:
                raise AnalyzeError("duplicate case completion record")
            cases[ordinal] = row

    if set(samples) != set(range(count)) or set(cases) != set(range(count)):
        raise AnalyzeError(f"round {round_number} case/sample denominator is incomplete")
    result: dict[int, dict[str, Any]] = {}
    for ordinal, expected in enumerate(plan["cases"]):
        row = cases[ordinal]
        by_index = samples[ordinal]
        if set(by_index) != set(range(reducer_plan.DEFAULT_SAMPLES)):
            raise AnalyzeError(f"case {expected['case_id']} lacks 11 exact samples")
        if row["partial_dtype"] != "fp32" or row["output_dtype"] != "fp16" or \
                integer(row["workspace_bytes"], "workspace bytes") != expected["workspace_bytes"] or \
                integer(row["output_bytes"], "output bytes") != expected["output_bytes"] or \
                integer(row["warmups"], "warmups") != reducer_plan.DEFAULT_WARMUPS or \
                integer(row["samples"], "samples") != reducer_plan.DEFAULT_SAMPLES or \
                integer(row["raw_bad"], "raw bad") != 0 or \
                integer(row["first_bad"], "first bad") != 0xFFFFFFFF or \
                row["status"] != "PASS":
            raise AnalyzeError(f"case {expected['case_id']} failed correctness/resource closure")
        integer(row["grid_ctas"], "grid CTAs", minimum=1)
        integer(row["block_threads"], "block threads", minimum=1)
        values = [by_index[index] for index in range(reducer_plan.DEFAULT_SAMPLES)]
        ordered = sorted(values)
        reported = {
            "median": timing(row["median_us"], "median us"),
            "min": timing(row["min_us"], "min us"),
            "max": timing(row["max_us"], "max us"),
        }
        if reported != {
                "median": ordered[len(ordered) // 2],
                "min": ordered[0], "max": ordered[-1]}:
            raise AnalyzeError(f"case {expected['case_id']} sample statistics differ")
        result[ordinal] = {
            "samples": values,
            "median": reported["median"],
            "grid_ctas": integer(row["grid_ctas"], "grid CTAs", minimum=1),
            "block_threads": integer(row["block_threads"], "block threads", minimum=1),
        }
    return result


def write_new(path: Path, content: str) -> None:
    if path.exists() or path.is_symlink():
        raise AnalyzeError(f"refusing to overwrite output: {path}")
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def analyze(plan_path: Path, manifest_path: Path, execution_authority_path: Path,
            runs: Path, output: Path) -> dict[str, Any]:
    plan = read_json(plan_path)
    reducer_plan.validate_plan(plan)
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "quactlize.fq-splitk-reducer-prebuilt.v1" or \
            manifest.get("build", {}).get("plan_sha256") != reducer_plan.digest(plan):
        raise AnalyzeError("bundle manifest is not bound to the reducer plan")
    execution = validate_execution_authority(
        read_json(execution_authority_path), manifest, plan)
    if execution["bundle_manifest_sha256"] != sha256_file(manifest_path) or \
            execution["plan_file_sha256"] != sha256_file(plan_path):
        raise AnalyzeError("execution authority file hashes differ")
    per_round = []
    for round_number, seed in expected_rounds(plan):
        path = runs / f"round-{round_number}.log"
        per_round.append((path, parse_round(path, plan, round_number, seed)))

    rows = []
    for ordinal, expected in enumerate(plan["cases"]):
        raw = [sample for _path, round_result in per_round
               for sample in round_result[ordinal]["samples"]]
        ordered = sorted(raw)
        round_medians = [round_result[ordinal]["median"]
                         for _path, round_result in per_round]
        grid = {round_result[ordinal]["grid_ctas"]
                for _path, round_result in per_round}
        block = {round_result[ordinal]["block_threads"]
                 for _path, round_result in per_round}
        if len(grid) != 1 or len(block) != 1:
            raise AnalyzeError(f"launch shape drifted for {expected['case_id']}")
        rows.append({
            "ordinal": ordinal,
            "case_id": expected["case_id"],
            "m": expected["m"], "n": expected["n"], "split": expected["split"],
            "partial_dtype": "fp32", "output_dtype": "fp16",
            "implementation": expected["expected_implementation"],
            "workspace_bytes": expected["workspace_bytes"],
            "output_bytes": expected["output_bytes"],
            "grid_ctas": next(iter(grid)), "block_threads": next(iter(block)),
            "round_medians_us": [format(value, "f") for value in round_medians],
            "samples_us": [format(value, "f") for value in raw],
            "sample_count": len(raw),
            "median_us": format(ordered[len(ordered) // 2], "f"),
            "min_us": format(ordered[0], "f"),
            "max_us": format(ordered[-1], "f"),
            "raw_bit_correctness": "PASS",
        })
    if len(rows) != 1035 or any(row["sample_count"] != 33 for row in rows):
        raise AnalyzeError("final reducer lookup denominator differs")
    output.mkdir(parents=True, exist_ok=False)

    summary = {
        "schema": SCHEMA,
        "verdict": "EXACT_REDUCER_LOOKUP_MEASURED",
        "plan_sha256": reducer_plan.digest(plan),
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "authorities": {
            "execution_authority_sha256": sha256_file(execution_authority_path),
            "source_commit": execution["source_commit"],
            "source_tree": execution["source_tree"],
            "sdk_authority_sha256": execution["sdk_authority_sha256"],
            "binary_sha256": execution["binary_sha256"],
            "build_authority_sha256": execution["build_authority_sha256"],
            "device_identity_sha256": execution["device_identity_sha256"],
            "device_homogeneity_sha256": execution["device_homogeneity_sha256"],
            "device_homogeneity_key": execution["device_homogeneity_key"],
            "runtime_linkage_sha256": execution["runtime_linkage_sha256"],
        },
        "measurement": {
            "rounds": plan["measurement"]["rounds"],
            "warmups_per_round": reducer_plan.DEFAULT_WARMUPS,
            "samples_per_round": reducer_plan.DEFAULT_SAMPLES,
            "samples_per_case": 33,
            "case_order": plan["measurement"]["case_order"],
            "top_n": None,
            "point_estimate_pruning": False,
            "raw_bit_correctness": "EVERY_OUTPUT_ELEMENT_BEFORE_AND_AFTER_TIMING",
        },
        "denominator": {"cases": 1035, "unique_m_n": 345, "splits": [2, 4, 8]},
        "rows": rows,
    }
    summary_path = output / "summary.json"
    write_new(summary_path, json.dumps(
        summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    columns = [
        "ordinal", "case_id", "M", "N", "S", "implementation",
        "grid_ctas", "block_threads", "workspace_bytes", "output_bytes",
        "sample_count", "median_us", "min_us", "max_us",
        "round1_median_us", "round2_median_us", "round3_median_us",
        "raw_bit_correctness",
    ]
    tsv = ["\t".join(columns)]
    for row in rows:
        tsv.append("\t".join(map(str, (
            row["ordinal"], row["case_id"], row["m"], row["n"], row["split"],
            row["implementation"], row["grid_ctas"], row["block_threads"],
            row["workspace_bytes"], row["output_bytes"], row["sample_count"],
            row["median_us"], row["min_us"], row["max_us"],
            *row["round_medians_us"], row["raw_bit_correctness"],
        ))))
    summary_tsv = output / "summary.tsv"
    write_new(summary_tsv, "\n".join(tsv) + "\n")
    authority = {
        "schema": AUTHORITY_SCHEMA,
        "execution_authority": {
            "path": str(execution_authority_path.name),
            "sha256": sha256_file(execution_authority_path)},
        "plan": {"path": str(plan_path.name),
                 "file_sha256": sha256_file(plan_path),
                 "canonical_sha256": reducer_plan.digest(plan)},
        "bundle_manifest": {
            "path": str(manifest_path.name), "sha256": sha256_file(manifest_path)},
        "binary": {
            "sha256": execution["binary_sha256"],
            "build_authority_sha256": execution["build_authority_sha256"],
            "sdk_authority_sha256": execution["sdk_authority_sha256"]},
        "device": {
            "worker_id": execution["worker_id"],
            "device_identity_sha256": execution["device_identity_sha256"],
            "device_homogeneity_sha256": execution["device_homogeneity_sha256"],
            "device_homogeneity_key": execution["device_homogeneity_key"],
            "runtime_linkage_sha256": execution["runtime_linkage_sha256"]},
        "runs": [{"round": index + 1, "path": path.name, "sha256": sha256_file(path)}
                 for index, (path, _parsed) in enumerate(per_round)],
        "outputs": {
            "summary.json": sha256_file(summary_path),
            "summary.tsv": sha256_file(summary_tsv),
        },
        "analyzer": {
            "path": "tools/analyze_fq_splitk_reducer_lookup.py",
            "sha256": sha256_file(Path(__file__)),
        },
    }
    write_new(output / "result-authority.json", json.dumps(
        authority, indent=2, sort_keys=True, allow_nan=False) + "\n")
    write_new(output / "verdict.log", (
        "FQ_SPLITK_REDUCER_LOOKUP_VERDICT "
        "verdict=EXACT_REDUCER_LOOKUP_MEASURED cases=1035 unique_mn=345 "
        "splits=S2/S4/S8 rounds=3 samples_per_case=33 raw_bit=PASS "
        "top_n=NONE point_pruning=0\n"))
    return summary


def self_test() -> None:
    plan = reducer_plan.materialize()
    reducer_plan.validate_plan(plan, plan)
    hashes = []
    for round_number, seed in expected_rounds(plan):
        order = seeded_order(len(plan["cases"]), seed)
        if sorted(order) != list(range(1035)):
            raise AssertionError("seeded order is not a permutation")
        hashes.append(order_hash(order))
    if len(set(hashes)) != 3:
        raise AssertionError("three fixed rounds do not have distinct orders")
    broken = seeded_order(1035, reducer_plan.ROUND_SEEDS[0])
    broken[0] = broken[1]
    if len(set(broken)) == 1035:
        raise AssertionError("order negative stayed green")
    print(
        "[fq-splitk-reducer-analysis:self-test] PASS cases=1035 rounds=3 "
        "samples=33-per-case fixed-orders=3 raw-bit=FAIL-CLOSED top_n=NONE")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    execution = commands.add_parser("write-execution-authority")
    execution.add_argument("--manifest", type=Path, required=True)
    execution.add_argument("--plan", type=Path, required=True)
    execution.add_argument("--device-identity", type=Path, required=True)
    execution.add_argument("--device-homogeneity", type=Path, required=True)
    execution.add_argument("--binary", type=Path, required=True)
    execution.add_argument("--worker-id", type=int, required=True)
    execution.add_argument("--visible-device", required=True)
    execution.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("analyze")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--execution-authority", type=Path, required=True)
    command.add_argument("--runs", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "write-execution-authority":
            write_execution_authority(
                args.manifest, args.plan, args.device_identity,
                args.device_homogeneity, args.binary, args.worker_id,
                args.visible_device, args.output)
        else:
            analyze(args.plan, args.manifest, args.execution_authority,
                    args.runs, args.output)
        return 0
    except (AnalyzeError, reducer_plan.PlanError, OSError, KeyError,
            TypeError, ValueError) as error:
        print(f"[fq-splitk-reducer-analysis] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
