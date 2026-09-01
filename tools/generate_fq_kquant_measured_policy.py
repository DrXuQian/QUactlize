#!/usr/bin/env python3
"""Generate the dense-only exact measured K-pack host policy.

The fitted JSON is not trusted as an authority by itself. Every dense interval
is joined back to the all-config measurement summary and admitted only when it
covers exact measured points, names a candidate present at every point, and
has at most three percent measured regret. Grouped evidence and source
heuristic fallback rules are validated as denominator authority but are never
emitted. A runtime exact-key miss therefore retains the compiled default.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "quactlize/include/ppu_kquant_measured_policy_data.inc"

SUMMARY_SCHEMA = "quactlize.fq-kquant-kpack-perf-result.v3"
HEURISTIC_SCHEMA = "quactlize.fq-kquant-config-heuristic.v1"
AUTHORITY_SCHEMA = "quactlize.fq-kquant-prebuilt-result-authority.v2"
OUTPUT_SCHEMA = "quactlize.ppu-kquant-dense-exact-policy.v1"
RUNTIME_EXACT_MISS_POLICY = "compiled-default"
SOURCE_COMMIT = "2b513637fc3d315077b14ab81784ff1fb21e1bb7"
EVIDENCE_GRADE = "unverified-sdk"
REGRET_THRESHOLD_PCT = 3.0
REGRET_THRESHOLD = REGRET_THRESHOLD_PCT / 100.0
CPP_INT32_MIN = -(1 << 31)
CPP_INT32_MAX = (1 << 31) - 1

EXPECTED_BOARD_KEYS = (
    (10, "dense"), (10, "grouped"),
    (11, "dense"), (11, "grouped"),
    (12, "grouped"),
    (13, "dense"), (13, "grouped"),
    (14, "dense"), (14, "grouped"),
)
EXPECTED_FORMATS = {
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
}
EXPECTED_DENSE_FAMILIES = 44
EXPECTED_GROUPED_FAMILIES = 20
EXPECTED_DENSE_INTERVALS = 110
EXPECTED_GROUPED_INTERVALS = 38
EXPECTED_DYNAMIC_VALUES = 13
EXPECTED_GROUPED_SIGNATURES = 13
EXPECTED_AUTHORITY_FILES = 42
EXPECTED_PREFLIGHT_STATUS = "MISMATCH_ALLOWED"
EXPECTED_PREFLIGHT_MISMATCH_COUNT = 2
EXPECTED_GROUPED_EXPERTS = 256
EXPECTED_GROUPED_TOPK = 8
EXPECTED_GROUPED_ROUTER = "token-topk-hot16x4-wor-sm64-s44-v1"

# Values are the ConfigId enumerator spellings in the corresponding shipping
# inventory.  Generated policy data intentionally contains tokens, not tactic
# strings.
DENSE_CONFIG_TOKENS = {
    "64x64:32x32:s3": "Default",
    "32x32:16x16:s3": "SmallSquare",
    "16x128:16x32:s3": "ShortWide",
    "8x128:8x32:s2": "ShortWideM8S2",
    "8x128:8x32:s3": "ShortWideM8S3",
    "8x128:8x32:s4": "ShortWideM8S4",
    "8x128:8x32:s6": "ShortWideM8S6",
    "8x128:8x32:s8": "ShortWideM8S8",
    "8x128:8x32:s12": "ShortWideM8S12",
    "32x128:32x32:s3": "MidWide",
    "128x64:64x32:s2": "Tall",
}
GROUPED_CONFIG_TOKENS = {
    "16x128:16x16:s2": "Default",
    "32x32:16x16:s3": "SmallSquare",
    "16x128:16x32:s3": "ShortWide",
    "32x128:32x32:s3": "MidWide",
    "128x64:64x32:s2": "Tall",
}

_DENSE_CANDIDATES = frozenset(DENSE_CONFIG_TOKENS) - {
    "8x128:8x32:s12",
}
_DENSE_Q5_CANDIDATES = _DENSE_CANDIDATES - {
    "8x128:8x32:s8",
}
_GROUPED_CANDIDATES = frozenset(GROUPED_CONFIG_TOKENS)
EXPECTED_CANDIDATE_CONFIGS = {
    (10, "dense"): _DENSE_CANDIDATES,
    (10, "grouped"): _GROUPED_CANDIDATES,
    (11, "dense"): _DENSE_CANDIDATES,
    (11, "grouped"): _GROUPED_CANDIDATES,
    (12, "grouped"): _GROUPED_CANDIDATES,
    (13, "dense"): _DENSE_Q5_CANDIDATES,
    (13, "grouped"): _GROUPED_CANDIDATES,
    (14, "dense"): _DENSE_CANDIDATES,
    (14, "grouped"): _GROUPED_CANDIDATES,
}

HEURISTIC_TOP_KEYS = {
    "boards", "max_fallback_leaves", "min_fallback_leaf_families",
    "min_fallback_leaf_rows", "regret_threshold_pct", "schema",
    "selection_order", "source_iterations", "source_rounds",
    "source_schema", "source_summary_sha256",
}
HEURISTIC_BOARD_KEYS = {
    "fallback", "family_count", "family_tables", "format",
    "measured_rows", "operator", "qtype",
}
FAMILY_KEYS = {"K", "N", "dynamic_axis", "intervals"}
INTERVAL_KEYS = {
    "config", "dynamic_max", "dynamic_min", "max_regret",
    "mean_regret", "measured_values",
}
FALLBACK_KEYS = {"leaves", "max_regret", "mean_regret", "within_threshold"}
FALLBACK_LEAF_KEYS = {
    "config", "family_count", "max_regret", "mean_regret",
    "predicates", "row_count",
}
PREDICATE_KEYS = {"feature", "op", "value"}
CANDIDATE_KEYS = {"config", "max_us", "median_us", "min_us", "samples"}
DENSE_SHAPE_KEYS = {"k", "key", "m", "n", "sources"}
GROUPED_SHAPE_KEYS = {
    "active", "experts", "k", "key", "max_rows", "n", "router",
    "sources", "tokens", "topk", "total_rows", "zero",
}
AUTHORITY_KEYS = {
    "bundle_manifest_sha256", "controls", "evidence_grade", "files",
    "runtime_preflight", "schema", "source_commit",
}
AUTHORITY_CONTROL_KEYS = {
    "all_configs", "heuristic_max_leaves", "heuristic_min_leaf_families",
    "heuristic_min_leaf_rows", "iterations", "profile", "rounds",
    "threshold_pct", "warmups",
}
AUTHORITY_FILE_KEYS = {"path", "sha256", "size"}
AUTHORITY_PREFLIGHT_KEYS = {
    "path", "sdk_identity_status", "sdk_mismatch_count", "sha256",
}


class PolicyError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PolicyError(message)


def exact_keys(value: Any, keys: set[str], what: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{what} must be an object")
    present = set(value)
    require(present == keys,
            f"{what} fields differ: missing={sorted(keys - present)} "
            f"extra={sorted(present - keys)}")
    return value


def required_keys(value: Any, keys: set[str], what: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{what} must be an object")
    missing = keys - set(value)
    require(not missing, f"{what} is missing fields {sorted(missing)}")
    return value


def exact_int(value: Any, what: str, *, positive: bool = False) -> int:
    require(type(value) is int, f"{what} must be an integer")
    require(not positive or value > 0, f"{what} must be positive")
    return value


def cpp_int32(value: Any, what: str, *, positive: bool = False,
              nonnegative: bool = False) -> int:
    result = exact_int(value, what, positive=positive)
    require(CPP_INT32_MIN <= result <= CPP_INT32_MAX,
            f"{what} must fit the C++ int32 boundary")
    require(not nonnegative or result >= 0,
            f"{what} must be nonnegative")
    return result


def finite_number(value: Any, what: str, *, positive: bool = False) -> float:
    require(type(value) in (int, float), f"{what} must be a number")
    result = float(value)
    require(math.isfinite(result), f"{what} must be finite")
    require(not positive or result > 0.0, f"{what} must be positive")
    return result


def close(lhs: float, rhs: float) -> bool:
    return math.isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-12)


def nonempty_string(value: Any, what: str) -> str:
    require(isinstance(value, str) and value != "", f"{what} must be a nonempty string")
    return value


def object_list(value: Any, what: str, *, nonempty: bool = True) -> list[dict[str, Any]]:
    require(isinstance(value, list), f"{what} must be an array")
    require(not nonempty or value, f"{what} must not be empty")
    require(all(isinstance(row, dict) for row in value),
            f"{what} entries must be objects")
    return value


def integer_list(value: Any, what: str) -> list[int]:
    require(isinstance(value, list) and value, f"{what} must be a nonempty array")
    require(all(type(item) is int and item > 0 for item in value),
            f"{what} entries must be positive integers")
    require(value == sorted(set(value)), f"{what} must be sorted and unique")
    return value


def load_json(path: Path, what: str) -> tuple[dict[str, Any], str, int]:
    raw = path.read_bytes()

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise PolicyError(f"{what} repeats JSON key {key!r}")
            result[key] = item
        return result

    def reject_constant(token: str) -> Any:
        raise PolicyError(f"{what} contains non-JSON numeric token {token}")

    try:
        value = json.loads(raw, object_pairs_hook=unique_object,
                           parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PolicyError(f"{what} is not valid UTF-8 JSON: {error}") from error
    require(isinstance(value, dict), f"{what} root must be an object")
    return value, hashlib.sha256(raw).hexdigest(), len(raw)


def config_tokens(operator: str) -> dict[str, str]:
    if operator == "dense":
        return DENSE_CONFIG_TOKENS
    if operator == "grouped":
        return GROUPED_CONFIG_TOKENS
    raise PolicyError(f"unsupported operator {operator!r}")


def sha256_string(value: Any, what: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value),
            f"{what} must be a lowercase SHA-256")
    return value


def validate_evidence_files(
        records: dict[str, tuple[str, int]], evidence_roots: tuple[Path, ...],
        *, required: bool) -> None:
    if not evidence_roots:
        require(not required,
                "production authority requires at least one --evidence-root")
        return

    roots: list[Path] = []
    for index, root in enumerate(evidence_roots):
        what = f"evidence root[{index}]"
        require(root.exists() and root.is_dir() and not root.is_symlink(),
                f"{what} must be an existing non-symlink directory: {root}")
        resolved = root.resolve(strict=True)
        require(resolved not in roots, f"duplicate evidence root: {resolved}")
        roots.append(resolved)

    for relative, (expected_sha256, expected_size) in records.items():
        parts = PurePosixPath(relative).parts
        matches: list[Path] = []
        for root in roots:
            candidate = root.joinpath(*parts)
            if not candidate.exists() and not candidate.is_symlink():
                continue
            cursor = root
            for part in parts:
                cursor = cursor / part
                require(not cursor.is_symlink(),
                        f"authority evidence path must not traverse a symlink: "
                        f"{candidate}")
            require(candidate.is_file(),
                    f"authority evidence path is not a regular file: {candidate}")
            resolved_candidate = candidate.resolve(strict=True)
            require(resolved_candidate.is_relative_to(root),
                    f"authority evidence escapes its root: {candidate}")
            matches.append(resolved_candidate)
        require(len(matches) == 1,
                f"authority evidence {relative!r} must exist in exactly one "
                f"evidence root; found {len(matches)}")
        raw = matches[0].read_bytes()
        require(len(raw) == expected_size,
                f"authority evidence {relative!r} size differs: "
                f"expected={expected_size} actual={len(raw)}")
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        require(actual_sha256 == expected_sha256,
                f"authority evidence {relative!r} SHA-256 differs: "
                f"expected={expected_sha256} actual={actual_sha256}")


def validate_authority(value: dict[str, Any], summary: dict[str, Any],
                       summary_sha256: str, summary_size: int,
                       heuristic: dict[str, Any], heuristic_sha256: str,
                       heuristic_size: int, *, production_census: bool,
                       evidence_roots: tuple[Path, ...]) -> None:
    authority = exact_keys(value, AUTHORITY_KEYS, "result authority")
    require(authority["schema"] == AUTHORITY_SCHEMA,
            f"result authority.schema must be {AUTHORITY_SCHEMA}")
    require(authority["source_commit"] == SOURCE_COMMIT,
            f"result authority.source_commit must be {SOURCE_COMMIT}")
    require(authority["evidence_grade"] == EVIDENCE_GRADE,
            f"result authority.evidence_grade must be {EVIDENCE_GRADE}")
    bundle_sha = sha256_string(authority["bundle_manifest_sha256"],
                               "result authority.bundle_manifest_sha256")

    controls = exact_keys(authority["controls"], AUTHORITY_CONTROL_KEYS,
                          "result authority.controls")
    require(exact_int(controls["all_configs"],
                      "result authority.controls.all_configs") == 1,
            "result authority controls must bind all-config measurement")
    require(exact_int(controls["iterations"],
                      "result authority.controls.iterations", positive=True)
            == summary["iterations"],
            "result authority iterations differ from summary")
    require(exact_int(controls["rounds"],
                      "result authority.controls.rounds", positive=True)
            == summary["rounds"],
            "result authority rounds differ from summary")
    require(controls["profile"] == summary["profile"],
            "result authority profile differs from summary")
    require(close(finite_number(controls["threshold_pct"],
                                "result authority.controls.threshold_pct",
                                positive=True), REGRET_THRESHOLD_PCT),
            "result authority threshold differs from measured policy")
    require(exact_int(controls["heuristic_max_leaves"],
                      "result authority.controls.heuristic_max_leaves", positive=True)
            == heuristic["max_fallback_leaves"],
            "result authority max leaves differ from config heuristic")
    require(exact_int(controls["heuristic_min_leaf_rows"],
                      "result authority.controls.heuristic_min_leaf_rows", positive=True)
            == heuristic["min_fallback_leaf_rows"],
            "result authority min leaf rows differ from config heuristic")
    require(exact_int(controls["heuristic_min_leaf_families"],
                      "result authority.controls.heuristic_min_leaf_families", positive=True)
            == heuristic["min_fallback_leaf_families"],
            "result authority min leaf families differ from config heuristic")
    require(exact_int(controls["warmups"],
                      "result authority.controls.warmups", positive=True) == 3,
            "result authority warmups must be 3")

    files = object_list(authority["files"], "result authority.files")
    if production_census:
        require(len(files) == EXPECTED_AUTHORITY_FILES,
                f"result authority must contain {EXPECTED_AUTHORITY_FILES} file records")
    records: dict[str, tuple[str, int]] = {}
    for index, file_value in enumerate(files):
        what = f"result authority.files[{index}]"
        row = exact_keys(file_value, AUTHORITY_FILE_KEYS, what)
        path = nonempty_string(row["path"], f"{what}.path")
        parsed = PurePosixPath(path)
        require(not parsed.is_absolute() and ".." not in parsed.parts and
                str(parsed) == path,
                f"{what}.path must be a canonical relative path")
        require(path not in records, f"result authority repeats file {path!r}")
        records[path] = (
            sha256_string(row["sha256"], f"{what}.sha256"),
            exact_int(row["size"], f"{what}.size", positive=True),
        )
    require(records.get("results/summary.json") == (summary_sha256, summary_size),
            "result authority summary file SHA/size differs from input bytes")
    require(records.get("results/config-heuristic.json") ==
            (heuristic_sha256, heuristic_size),
            "result authority config heuristic SHA/size differs from input bytes")
    require(records.get("inputs/bundle-manifest.json", (None, None))[0] == bundle_sha,
            "result authority bundle manifest SHA differs from its file record")
    validate_evidence_files(records, evidence_roots,
                            required=production_census)

    preflight = exact_keys(authority["runtime_preflight"], AUTHORITY_PREFLIGHT_KEYS,
                           "result authority.runtime_preflight")
    preflight_path = nonempty_string(preflight["path"],
                                     "result authority.runtime_preflight.path")
    preflight_sha = sha256_string(preflight["sha256"],
                                  "result authority.runtime_preflight.sha256")
    require(preflight["sdk_identity_status"] == EXPECTED_PREFLIGHT_STATUS,
            "result authority runtime preflight sdk_identity_status must be "
            f"{EXPECTED_PREFLIGHT_STATUS}")
    mismatch_count = exact_int(preflight["sdk_mismatch_count"],
                               "result authority.runtime_preflight.sdk_mismatch_count")
    require(mismatch_count == EXPECTED_PREFLIGHT_MISMATCH_COUNT,
            "result authority runtime preflight sdk_mismatch_count must be "
            f"{EXPECTED_PREFLIGHT_MISMATCH_COUNT}")
    require(records.get(preflight_path, (None, None))[0] == preflight_sha,
            "result authority runtime preflight SHA differs from its file record")


def validate_candidate(value: Any, what: str, sample_count: int) -> tuple[str, float]:
    candidate = exact_keys(value, CANDIDATE_KEYS, what)
    name = nonempty_string(candidate["config"], f"{what}.config")
    samples_value = candidate["samples"]
    require(isinstance(samples_value, list) and len(samples_value) == sample_count,
            f"{what}.samples must contain exactly {sample_count} values")
    samples = [finite_number(sample, f"{what}.samples[{index}]", positive=True)
               for index, sample in enumerate(samples_value)]
    minimum = finite_number(candidate["min_us"], f"{what}.min_us", positive=True)
    median = finite_number(candidate["median_us"], f"{what}.median_us", positive=True)
    maximum = finite_number(candidate["max_us"], f"{what}.max_us", positive=True)
    require(close(minimum, min(samples)), f"{what}.min_us does not match samples")
    require(close(median, statistics.median(samples)),
            f"{what}.median_us does not match samples")
    require(close(maximum, max(samples)), f"{what}.max_us does not match samples")
    require(minimum <= median <= maximum, f"{what} timing order is invalid")
    return name, median


def validate_sources(value: Any, what: str) -> None:
    require(isinstance(value, list) and value, f"{what} must be a nonempty array")
    require(all(isinstance(item, str) and item for item in value),
            f"{what} entries must be nonempty strings")
    require(len(value) == len(set(value)), f"{what} entries must be unique")


def validate_summary(value: dict[str, Any]) -> tuple[
        dict[tuple[int, str, int, int, int], dict[str, float]],
        dict[int, tuple[int, int]], dict[tuple[int, str], int]]:
    # Comparison-only evidence fields in this versioned summary are ignored;
    # every field read below belongs to the K-pack measurement authority.
    required_keys(value, {
        "all_configs", "boards", "dense_shape_count", "grouped_shape_count",
        "iterations", "profile", "rounds", "rows", "schema", "threshold_pct",
    }, "summary")
    require(value["schema"] == SUMMARY_SCHEMA,
            f"summary.schema must be {SUMMARY_SCHEMA}")
    require(value["all_configs"] is True,
            "summary.all_configs must be true")
    require(value["profile"] == "heuristic", "summary.profile must be heuristic")
    rounds = exact_int(value["rounds"], "summary.rounds", positive=True)
    iterations = exact_int(value["iterations"], "summary.iterations", positive=True)
    threshold = finite_number(value["threshold_pct"], "summary.threshold_pct", positive=True)
    require(close(threshold, REGRET_THRESHOLD_PCT),
            f"summary.threshold_pct must be {REGRET_THRESHOLD_PCT:g}")
    dense_shape_count = exact_int(value["dense_shape_count"],
                                  "summary.dense_shape_count", positive=True)
    grouped_shape_count = exact_int(value["grouped_shape_count"],
                                    "summary.grouped_shape_count", positive=True)

    cells: dict[tuple[int, str, int, int, int], dict[str, float]] = {}
    grouped_signatures: dict[int, tuple[int, int]] = {}
    grouped_signature_tokens: dict[tuple[int, int], int] = {}
    shape_keys: dict[str, set[str]] = {"dense": set(), "grouped": set()}
    board_counts: dict[tuple[int, str], int] = {}
    sample_count = rounds * iterations
    rows = object_list(value["rows"], "summary.rows")
    for row_index, row_value in enumerate(rows):
        what = f"summary.rows[{row_index}]"
        row = required_keys(row_value, {
            "candidates", "format", "key", "operator", "qtype", "shape",
        }, what)
        qtype = cpp_int32(row["qtype"], f"{what}.qtype", positive=True)
        require(qtype in EXPECTED_FORMATS, f"{what}.qtype is not a K-quant type")
        fmt = nonempty_string(row["format"], f"{what}.format")
        require(fmt == EXPECTED_FORMATS[qtype],
                f"{what}.format does not match qtype {qtype}")
        operator = nonempty_string(row["operator"], f"{what}.operator")
        tokens = config_tokens(operator)
        row_key = nonempty_string(row["key"], f"{what}.key")
        shape = exact_keys(row["shape"],
                           DENSE_SHAPE_KEYS if operator == "dense" else GROUPED_SHAPE_KEYS,
                           f"{what}.shape")
        require(shape["key"] == row_key, f"{what}.shape.key differs from row key")
        n = cpp_int32(shape["n"], f"{what}.shape.n", positive=True)
        k = cpp_int32(shape["k"], f"{what}.shape.k", positive=True)
        validate_sources(shape["sources"], f"{what}.shape.sources")
        if operator == "dense":
            dynamic = cpp_int32(shape["m"], f"{what}.shape.m", positive=True)
        else:
            dynamic = cpp_int32(
                shape["tokens"], f"{what}.shape.tokens", positive=True)
            total_rows = cpp_int32(
                shape["total_rows"], f"{what}.shape.total_rows", positive=True)
            max_rows = cpp_int32(
                shape["max_rows"], f"{what}.shape.max_rows", positive=True)
            experts = cpp_int32(
                shape["experts"], f"{what}.shape.experts", positive=True)
            topk = cpp_int32(
                shape["topk"], f"{what}.shape.topk", positive=True)
            active = cpp_int32(
                shape["active"], f"{what}.shape.active", positive=True)
            zero = cpp_int32(
                shape["zero"], f"{what}.shape.zero", nonnegative=True)
            router = nonempty_string(shape["router"], f"{what}.shape.router")
            require(experts == EXPECTED_GROUPED_EXPERTS,
                    f"{what}.shape.experts must be {EXPECTED_GROUPED_EXPERTS}")
            require(topk == EXPECTED_GROUPED_TOPK,
                    f"{what}.shape.topk must be {EXPECTED_GROUPED_TOPK}")
            require(router == EXPECTED_GROUPED_ROUTER,
                    f"{what}.shape.router must be {EXPECTED_GROUPED_ROUTER}")
            require(topk <= experts and active <= experts and zero >= 0,
                    f"{what}.shape grouped counts are invalid")
            require(active + zero == experts,
                    f"{what}.shape active+zero must equal experts")
            signature = (total_rows, max_rows)
            previous = grouped_signatures.setdefault(dynamic, signature)
            require(previous == signature,
                    f"grouped token {dynamic} has conflicting exact signatures")
            previous_token = grouped_signature_tokens.setdefault(signature, dynamic)
            require(previous_token == dynamic,
                    "grouped exact signatures must map one-to-one to tokens")
            require(total_rows == dynamic * topk,
                    f"{what}.shape total_rows must equal tokens*topk")
            require(max_rows <= total_rows,
                    f"{what}.shape max_rows exceeds total_rows")

        candidates = required_keys(row["candidates"], {"kpack"}, f"{what}.candidates")
        arm = object_list(candidates["kpack"], f"{what}.candidates.kpack")
        times: dict[str, float] = {}
        for candidate_index, candidate in enumerate(arm):
            name, median = validate_candidate(
                candidate, f"{what}.candidates.kpack[{candidate_index}]", sample_count)
            require(name in tokens,
                    f"{what} candidate {name!r} has no shipping ConfigId token")
            require(name not in times, f"{what} repeats candidate {name!r}")
            times[name] = median
        expected_candidates = EXPECTED_CANDIDATE_CONFIGS[(qtype, operator)]
        actual_candidates = set(times)
        require(actual_candidates == expected_candidates,
                f"{what} candidate census differs: "
                f"missing={sorted(expected_candidates - actual_candidates)} "
                f"extra={sorted(actual_candidates - expected_candidates)}")

        identity = (qtype, operator, n, k, dynamic)
        require(identity not in cells, f"duplicate measured cell {identity}")
        cells[identity] = times
        shape_keys[operator].add(row_key)
        board_key = (qtype, operator)
        board_counts[board_key] = board_counts.get(board_key, 0) + 1

    require(len(shape_keys["dense"]) == dense_shape_count,
            "summary.dense_shape_count differs from exact row keys")
    require(len(shape_keys["grouped"]) == grouped_shape_count,
            "summary.grouped_shape_count differs from exact row keys")
    require(tuple(sorted(board_counts)) == EXPECTED_BOARD_KEYS,
            "summary board denominator differs from the K-quant measured policy")

    boards = object_list(value["boards"], "summary.boards")
    seen_boards: list[tuple[int, str]] = []
    for index, board in enumerate(boards):
        what = f"summary.boards[{index}]"
        required_keys(board, {"format", "operator", "qtype", "shapes"}, what)
        qtype = cpp_int32(board["qtype"], f"{what}.qtype", positive=True)
        operator = nonempty_string(board["operator"], f"{what}.operator")
        key = (qtype, operator)
        require(key in board_counts, f"{what} has no measured rows")
        require(board["format"] == EXPECTED_FORMATS[qtype],
                f"{what}.format does not match qtype")
        require(exact_int(board["shapes"], f"{what}.shapes", positive=True) == board_counts[key],
                f"{what}.shapes differs from measured rows")
        seen_boards.append(key)
    require(tuple(seen_boards) == EXPECTED_BOARD_KEYS,
            "summary.boards must be complete, unique, and deterministically ordered")
    return cells, grouped_signatures, board_counts


def validate_fallback(value: Any, operator: str, what: str) -> None:
    fallback = exact_keys(value, FALLBACK_KEYS, what)
    finite_number(fallback["max_regret"], f"{what}.max_regret")
    finite_number(fallback["mean_regret"], f"{what}.mean_regret")
    require(type(fallback["within_threshold"]) is bool,
            f"{what}.within_threshold must be boolean")
    leaves = object_list(fallback["leaves"], f"{what}.leaves")
    tokens = config_tokens(operator)
    for index, leaf_value in enumerate(leaves):
        leaf_what = f"{what}.leaves[{index}]"
        leaf = exact_keys(leaf_value, FALLBACK_LEAF_KEYS, leaf_what)
        require(nonempty_string(leaf["config"], f"{leaf_what}.config") in tokens,
                f"{leaf_what}.config has no shipping ConfigId token")
        exact_int(leaf["family_count"], f"{leaf_what}.family_count", positive=True)
        exact_int(leaf["row_count"], f"{leaf_what}.row_count", positive=True)
        finite_number(leaf["max_regret"], f"{leaf_what}.max_regret")
        finite_number(leaf["mean_regret"], f"{leaf_what}.mean_regret")
        predicates = object_list(leaf["predicates"], f"{leaf_what}.predicates", nonempty=False)
        for predicate_index, predicate_value in enumerate(predicates):
            pred_what = f"{leaf_what}.predicates[{predicate_index}]"
            predicate = exact_keys(predicate_value, PREDICATE_KEYS, pred_what)
            nonempty_string(predicate["feature"], f"{pred_what}.feature")
            require(predicate["op"] in ("le", "gt"),
                    f"{pred_what}.op must be le or gt")
            exact_int(predicate["value"], f"{pred_what}.value", positive=True)


def validate_heuristic(value: dict[str, Any], summary: dict[str, Any],
                       summary_sha256: str,
                       cells: dict[tuple[int, str, int, int, int], dict[str, float]],
                       board_counts: dict[tuple[int, str], int]) -> tuple[
                           list[tuple[int, int, int, int, int]],
                           list[tuple[int, int, str]],
                           list[tuple[int, int, int, int, int]],
                           list[tuple[int, int, str]]]:
    exact_keys(value, HEURISTIC_TOP_KEYS, "config heuristic")
    require(value["schema"] == HEURISTIC_SCHEMA,
            f"config heuristic.schema must be {HEURISTIC_SCHEMA}")
    require(value["source_schema"] == SUMMARY_SCHEMA,
            f"config heuristic.source_schema must be {SUMMARY_SCHEMA}")
    require(value["source_summary_sha256"] == summary_sha256,
            "config heuristic source_summary_sha256 does not match summary bytes")
    require(value["source_rounds"] == summary["rounds"],
            "config heuristic source_rounds does not match summary")
    require(value["source_iterations"] == summary["iterations"],
            "config heuristic source_iterations does not match summary")
    threshold = finite_number(value["regret_threshold_pct"],
                              "config heuristic.regret_threshold_pct", positive=True)
    require(close(threshold, REGRET_THRESHOLD_PCT),
            f"config heuristic regret threshold must be {REGRET_THRESHOLD_PCT:g} percent")
    exact_int(value["max_fallback_leaves"],
              "config heuristic.max_fallback_leaves", positive=True)
    exact_int(value["min_fallback_leaf_rows"],
              "config heuristic.min_fallback_leaf_rows", positive=True)
    exact_int(value["min_fallback_leaf_families"],
              "config heuristic.min_fallback_leaf_families", positive=True)
    require(value["selection_order"] == [
        "exact-family-measured-interval",
        "bounded-regret-fallback-rules",
        "compiled-default-after-runtime-validity-failure",
    ], "config heuristic.selection_order differs from schema v1")

    source_families: dict[tuple[int, str, int, int], dict[int, dict[str, float]]] = {}
    for (qtype, operator, n, k, dynamic), times in cells.items():
        source_families.setdefault((qtype, operator, n, k), {})[dynamic] = times

    dense_families: list[tuple[int, int, int, int, int]] = []
    dense_intervals: list[tuple[int, int, str]] = []
    grouped_families: list[tuple[int, int, int, int, int]] = []
    grouped_intervals: list[tuple[int, int, str]] = []
    seen_families: set[tuple[int, str, int, int]] = set()
    seen_boards: list[tuple[int, str]] = []

    boards = object_list(value["boards"], "config heuristic.boards")
    for board_index, board_value in enumerate(boards):
        what = f"config heuristic.boards[{board_index}]"
        board = exact_keys(board_value, HEURISTIC_BOARD_KEYS, what)
        qtype = cpp_int32(board["qtype"], f"{what}.qtype", positive=True)
        operator = nonempty_string(board["operator"], f"{what}.operator")
        config_tokens(operator)
        board_key = (qtype, operator)
        require(board_key in board_counts, f"{what} has no summary board")
        require(board["format"] == EXPECTED_FORMATS[qtype],
                f"{what}.format does not match qtype")
        require(exact_int(board["measured_rows"], f"{what}.measured_rows", positive=True)
                == board_counts[board_key],
                f"{what}.measured_rows differs from summary")
        validate_fallback(board["fallback"], operator, f"{what}.fallback")
        tables = object_list(board["family_tables"], f"{what}.family_tables")
        require(exact_int(board["family_count"], f"{what}.family_count", positive=True)
                == len(tables), f"{what}.family_count differs from family_tables")
        expected_family_keys = sorted(
            key for key in source_families if key[:2] == board_key)
        actual_family_keys: list[tuple[int, str, int, int]] = []

        for family_index, family_value in enumerate(tables):
            family_what = f"{what}.family_tables[{family_index}]"
            family = exact_keys(family_value, FAMILY_KEYS, family_what)
            n = cpp_int32(family["N"], f"{family_what}.N", positive=True)
            k = cpp_int32(family["K"], f"{family_what}.K", positive=True)
            family_key = (qtype, operator, n, k)
            require(family_key in source_families,
                    f"{family_what} has no summary family")
            require(family_key not in seen_families,
                    f"duplicate heuristic family {family_key}")
            seen_families.add(family_key)
            actual_family_keys.append(family_key)
            expected_axis = "m" if operator == "dense" else "tokens"
            require(family["dynamic_axis"] == expected_axis,
                    f"{family_what}.dynamic_axis must be {expected_axis}")
            source_dynamic = source_families[family_key]
            expected_values = sorted(source_dynamic)
            intervals = object_list(family["intervals"], f"{family_what}.intervals")
            first = len(dense_intervals if operator == "dense" else grouped_intervals)
            covered: list[int] = []
            emitted: list[tuple[int, int, str]] = []
            for interval_index, interval_value in enumerate(intervals):
                interval_what = f"{family_what}.intervals[{interval_index}]"
                interval = exact_keys(interval_value, INTERVAL_KEYS, interval_what)
                measured_values = integer_list(
                    interval["measured_values"],
                    f"{interval_what}.measured_values")
                measured_values = [
                    cpp_int32(dynamic, f"{interval_what}.measured_values[{index}]",
                              positive=True)
                    for index, dynamic in enumerate(measured_values)
                ]
                dynamic_min = cpp_int32(
                    interval["dynamic_min"], f"{interval_what}.dynamic_min",
                    positive=True)
                dynamic_max = cpp_int32(
                    interval["dynamic_max"], f"{interval_what}.dynamic_max",
                    positive=True)
                require(dynamic_min == measured_values[0] and
                        dynamic_max == measured_values[-1],
                        f"{interval_what} bounds differ from measured_values")
                require(not covered or covered[-1] < measured_values[0],
                        f"{interval_what} overlaps or is out of order")
                name = nonempty_string(interval["config"], f"{interval_what}.config")
                tokens = config_tokens(operator)
                require(name in tokens,
                        f"{interval_what}.config has no shipping ConfigId token")
                regrets: list[float] = []
                for dynamic in measured_values:
                    require(dynamic in source_dynamic,
                            f"{interval_what} includes unmeasured dynamic value {dynamic}")
                    times = source_dynamic[dynamic]
                    require(name in times,
                            f"{interval_what}.config {name!r} is absent from candidate set "
                            f"at dynamic value {dynamic}")
                    regrets.append(times[name] / min(times.values()) - 1.0)
                max_regret = max(regrets)
                mean_regret = statistics.fmean(regrets)
                reported_max = finite_number(interval["max_regret"],
                                             f"{interval_what}.max_regret")
                reported_mean = finite_number(interval["mean_regret"],
                                              f"{interval_what}.mean_regret")
                require(close(reported_max, max_regret),
                        f"{interval_what}.max_regret differs from K-pack candidates")
                require(close(reported_mean, mean_regret),
                        f"{interval_what}.mean_regret differs from K-pack candidates")
                require(max_regret <= REGRET_THRESHOLD + 1e-12,
                        f"{interval_what} exceeds {REGRET_THRESHOLD_PCT:g}% regret")
                covered.extend(measured_values)
                emitted.append((dynamic_min, dynamic_max, tokens[name]))
            require(covered == expected_values,
                    f"{family_what} intervals do not exactly cover summary values: "
                    f"expected={expected_values} got={covered}")
            target_families = dense_families if operator == "dense" else grouped_families
            target_intervals = dense_intervals if operator == "dense" else grouped_intervals
            target_families.append((qtype, n, k, first, len(emitted)))
            target_intervals.extend(emitted)

        require(actual_family_keys == expected_family_keys,
                f"{what}.family_tables must exactly cover and order summary families")
        seen_boards.append(board_key)

    require(tuple(seen_boards) == EXPECTED_BOARD_KEYS,
            "config heuristic boards must be complete, unique, and deterministically ordered")
    require(seen_families == set(source_families),
            "config heuristic does not cover every summary family")
    return dense_families, dense_intervals, grouped_families, grouped_intervals


def validate_domains(
        cells: dict[tuple[int, str, int, int, int], dict[str, float]],
        grouped_signatures: dict[int, tuple[int, int]]) -> list[int]:
    families: dict[tuple[int, str, int, int], set[int]] = {}
    for qtype, operator, n, k, dynamic in cells:
        families.setdefault((qtype, operator, n, k), set()).add(dynamic)
    domains = {tuple(sorted(values)) for values in families.values()}
    require(len(domains) == 1,
            "every measured family must have the same exact dynamic domain")
    dynamic_values = list(next(iter(domains)))
    require(all(value > 0 and value & (value - 1) == 0 for value in dynamic_values),
            "exact dynamic domain must contain only powers of two")
    require(set(grouped_signatures) == set(dynamic_values),
            "grouped exact signatures must cover the complete dynamic domain")
    return dynamic_values


def validate_dense_cpp_output(
        dense_families: list[tuple[int, int, int, int, int]],
        dense_intervals: list[tuple[int, int, str]],
        dynamic_values: list[int]) -> None:
    for index, (qtype, n, k, first, count) in enumerate(dense_families):
        cpp_int32(qtype, f"dense output family[{index}].qtype", positive=True)
        cpp_int32(n, f"dense output family[{index}].n", positive=True)
        cpp_int32(k, f"dense output family[{index}].k", positive=True)
        cpp_int32(first, f"dense output family[{index}].first_interval",
                  nonnegative=True)
        cpp_int32(count, f"dense output family[{index}].interval_count",
                  positive=True)
    for index, (minimum, maximum, _token) in enumerate(dense_intervals):
        cpp_int32(minimum, f"dense output interval[{index}].dynamic_min",
                  positive=True)
        cpp_int32(maximum, f"dense output interval[{index}].dynamic_max",
                  positive=True)
    for index, dynamic in enumerate(dynamic_values):
        cpp_int32(dynamic, f"dense output dynamic[{index}]", positive=True)
    cpp_int32(len(dense_families), "dense output family count",
              nonnegative=True)
    cpp_int32(len(dense_intervals), "dense output interval count",
              nonnegative=True)
    cpp_int32(len(dynamic_values), "dense output dynamic count",
              nonnegative=True)


def macro(name: str, rows: list[str]) -> list[str]:
    require(bool(rows), f"cannot emit empty macro {name}")
    result = [f"#define {name}(X) \\"]
    for index, row in enumerate(rows):
        result.append(f"  X({row})" + (" \\" if index + 1 != len(rows) else ""))
    return result


def render(summary_sha256: str, heuristic_sha256: str, authority_sha256: str,
           dense_families: list[tuple[int, int, int, int, int]],
           dense_intervals: list[tuple[int, int, str]],
           dynamic_values: list[int]) -> str:
    lines = [
        "// GENERATED by tools/generate_fq_kquant_measured_policy.py; do not edit.",
        "// Dense-only exact measured K-pack policy. Lookup is valid only after",
        "// the complete M/N/K/qtype key has matched an exact measured point.",
        "// A runtime exact-key miss retains the compiled default. Grouped",
        "// measurements and source heuristic fallback rules are evidence-validated",
        "// inputs, but neither is emitted into this runtime policy.",
        f"// Source commit: {SOURCE_COMMIT}",
        f"// Source summary SHA-256: {summary_sha256}",
        f"// Source config heuristic SHA-256: {heuristic_sha256}",
        f"// Source result authority SHA-256: {authority_sha256}",
        f"// Evidence grade: {EVIDENCE_GRADE}",
        "#pragma once",
        "",
        f'#define QUACTLIZE_PPU_KQUANT_MEASURED_POLICY_SCHEMA "{OUTPUT_SCHEMA}"',
        f'#define QUACTLIZE_PPU_KQUANT_MEASURED_EXACT_MISS_POLICY "{RUNTIME_EXACT_MISS_POLICY}"',
        f'#define QUACTLIZE_PPU_KQUANT_MEASURED_SOURCE_COMMIT "{SOURCE_COMMIT}"',
        f'#define QUACTLIZE_PPU_KQUANT_MEASURED_SUMMARY_SHA256 "{summary_sha256}"',
        f'#define QUACTLIZE_PPU_KQUANT_MEASURED_CONFIG_SHA256 "{heuristic_sha256}"',
        f'#define QUACTLIZE_PPU_KQUANT_MEASURED_AUTHORITY_SHA256 "{authority_sha256}"',
        f'#define QUACTLIZE_PPU_KQUANT_MEASURED_EVIDENCE_GRADE "{EVIDENCE_GRADE}"',
        "#define QUACTLIZE_PPU_KQUANT_MEASURED_REGRET_THRESHOLD_BPS 300",
        f"#define QUACTLIZE_PPU_KQUANT_MEASURED_DYNAMIC_VALUE_COUNT {len(dynamic_values)}",
        f"#define QUACTLIZE_PPU_KQUANT_DENSE_MEASURED_FAMILY_COUNT {len(dense_families)}",
        f"#define QUACTLIZE_PPU_KQUANT_DENSE_MEASURED_INTERVAL_COUNT {len(dense_intervals)}",
        "",
        "// X(dynamic). Membership in this list is mandatory before interval lookup.",
    ]
    lines.extend(macro("QUACTLIZE_PPU_KQUANT_MEASURED_DYNAMIC_VALUES",
                       [str(value) for value in dynamic_values]))
    lines.extend([
        "",
        "// X(qtype, n, k, first_interval, interval_count).",
    ])
    lines.extend(macro("QUACTLIZE_PPU_KQUANT_DENSE_MEASURED_FAMILIES", [
        f"{qtype}, {n}, {k}, {first}, {count}"
        for qtype, n, k, first, count in dense_families
    ]))
    lines.extend([
        "",
        "// X(dynamic_min, dynamic_max, dense_ConfigId_token).",
    ])
    lines.extend(macro("QUACTLIZE_PPU_KQUANT_DENSE_MEASURED_INTERVALS", [
        f"{minimum}, {maximum}, {token}"
        for minimum, maximum, token in dense_intervals
    ]))
    return "\n".join(lines) + "\n"


def generate(summary: dict[str, Any], summary_sha256: str, summary_size: int,
             heuristic: dict[str, Any], heuristic_sha256: str, heuristic_size: int,
             authority: dict[str, Any], authority_sha256: str,
             *, production_census: bool,
             evidence_roots: tuple[Path, ...] = ()) -> str:
    validate_authority(authority, summary, summary_sha256, summary_size,
                       heuristic, heuristic_sha256, heuristic_size,
                       production_census=production_census,
                       evidence_roots=evidence_roots)
    cells, grouped_signatures, board_counts = validate_summary(summary)
    dense_families, dense_intervals, grouped_families, grouped_intervals = \
        validate_heuristic(heuristic, summary, summary_sha256, cells, board_counts)
    dynamic_values = validate_domains(cells, grouped_signatures)
    validate_dense_cpp_output(dense_families, dense_intervals, dynamic_values)
    if production_census:
        require(len(dense_families) == EXPECTED_DENSE_FAMILIES,
                f"dense family census must be {EXPECTED_DENSE_FAMILIES}")
        require(len(grouped_families) == EXPECTED_GROUPED_FAMILIES,
                f"grouped family census must be {EXPECTED_GROUPED_FAMILIES}")
        require(len(dense_intervals) == EXPECTED_DENSE_INTERVALS,
                f"dense interval census must be {EXPECTED_DENSE_INTERVALS}")
        require(len(grouped_intervals) == EXPECTED_GROUPED_INTERVALS,
                f"grouped interval census must be {EXPECTED_GROUPED_INTERVALS}")
        require(len(dynamic_values) == EXPECTED_DYNAMIC_VALUES,
                f"dynamic-value census must be {EXPECTED_DYNAMIC_VALUES}")
        require(len(grouped_signatures) == EXPECTED_GROUPED_SIGNATURES,
                f"grouped signature census must be {EXPECTED_GROUPED_SIGNATURES}")
    return render(summary_sha256, heuristic_sha256, authority_sha256,
                  dense_families, dense_intervals, dynamic_values)


def synthetic_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    formats = {10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K"}
    board_keys = list(EXPECTED_BOARD_KEYS)
    rows: list[dict[str, Any]] = []
    for qtype, operator in board_keys:
        for dynamic in (1, 2):
            if operator == "dense":
                shape = {
                    "k": 32, "key": f"dense_m{dynamic}_n16_k32_q{qtype}",
                    "m": dynamic, "n": 16, "sources": ["self-test"],
                }
                selected_name = "64x64:32x32:s3"
            else:
                shape = {
                    "active": 2, "experts": EXPECTED_GROUPED_EXPERTS, "k": 32,
                    "key": f"grouped_t{dynamic}_n16_k32_q{qtype}",
                    "max_rows": dynamic, "n": 16,
                    "router": EXPECTED_GROUPED_ROUTER,
                    "sources": ["self-test"], "tokens": dynamic,
                    "topk": EXPECTED_GROUPED_TOPK,
                    "total_rows": dynamic * EXPECTED_GROUPED_TOPK,
                    "zero": EXPECTED_GROUPED_EXPERTS - 2,
                }
                selected_name = "16x128:16x16:s2"
            configs = tuple(
                (name, 10.0 if name == selected_name else 11.0 + index)
                for index, name in enumerate(sorted(
                    EXPECTED_CANDIDATE_CONFIGS[(qtype, operator)])))
            candidates = [{
                "config": name, "min_us": timing, "median_us": timing,
                "max_us": timing, "samples": [timing],
            } for name, timing in configs]
            rows.append({
                "candidates": {"kpack": candidates},
                "format": formats[qtype], "key": shape["key"],
                "operator": operator, "qtype": qtype, "shape": shape,
            })
    summary: dict[str, Any] = {
        "all_configs": True,
        "boards": [{
            "format": formats[qtype], "operator": operator, "qtype": qtype,
            "shapes": 2, "unresolved": 0,
        } for qtype, operator in board_keys],
        "dense_shape_count": 8,
        "grouped_shape_count": 10,
        "iterations": 1,
        "profile": "heuristic",
        "rounds": 1,
        "rows": rows,
        "schema": SUMMARY_SCHEMA,
        "threshold_pct": REGRET_THRESHOLD_PCT,
    }
    summary_raw = (json.dumps(summary, sort_keys=True) + "\n").encode()
    summary_sha = hashlib.sha256(summary_raw).hexdigest()
    boards: list[dict[str, Any]] = []
    for qtype, operator in board_keys:
        selected = ("64x64:32x32:s3" if operator == "dense"
                    else "16x128:16x16:s2")
        fallback = ("128x64:64x32:s2" if operator == "dense"
                    else "32x32:16x16:s3")
        boards.append({
            "fallback": {
                "leaves": [{
                    "config": fallback, "family_count": 1,
                    "max_regret": 0.0, "mean_regret": 0.0,
                    "predicates": [], "row_count": 2,
                }],
                "max_regret": 0.0, "mean_regret": 0.0,
                "within_threshold": True,
            },
            "family_count": 1,
            "family_tables": [{
                "K": 32, "N": 16,
                "dynamic_axis": "m" if operator == "dense" else "tokens",
                "intervals": [{
                    "config": selected, "dynamic_max": 2, "dynamic_min": 1,
                    "max_regret": 0.0, "mean_regret": 0.0,
                    "measured_values": [1, 2],
                }],
            }],
            "format": formats[qtype], "measured_rows": 2,
            "operator": operator, "qtype": qtype,
        })
    heuristic = {
        "boards": boards,
        "max_fallback_leaves": 1,
        "min_fallback_leaf_families": 1,
        "min_fallback_leaf_rows": 1,
        "regret_threshold_pct": REGRET_THRESHOLD_PCT,
        "schema": HEURISTIC_SCHEMA,
        "selection_order": [
            "exact-family-measured-interval",
            "bounded-regret-fallback-rules",
            "compiled-default-after-runtime-validity-failure",
        ],
        "source_iterations": 1,
        "source_rounds": 1,
        "source_schema": SUMMARY_SCHEMA,
        "source_summary_sha256": summary_sha,
    }
    heuristic_raw = (json.dumps(heuristic, sort_keys=True) + "\n").encode()
    heuristic_sha = hashlib.sha256(heuristic_raw).hexdigest()
    authority = {
        "bundle_manifest_sha256": "b" * 64,
        "controls": {
            "all_configs": 1,
            "heuristic_max_leaves": 1,
            "heuristic_min_leaf_families": 1,
            "heuristic_min_leaf_rows": 1,
            "iterations": 1,
            "profile": "heuristic",
            "rounds": 1,
            "threshold_pct": REGRET_THRESHOLD_PCT,
            "warmups": 3,
        },
        "evidence_grade": EVIDENCE_GRADE,
        "files": [
            {"path": "inputs/bundle-manifest.json", "sha256": "b" * 64, "size": 1},
            {"path": "inputs/runtime-preflight.json", "sha256": "c" * 64, "size": 1},
            {"path": "results/config-heuristic.json", "sha256": heuristic_sha,
             "size": len(heuristic_raw)},
            {"path": "results/summary.json", "sha256": summary_sha,
             "size": len(summary_raw)},
        ],
        "runtime_preflight": {
            "path": "inputs/runtime-preflight.json",
            "sdk_identity_status": EXPECTED_PREFLIGHT_STATUS,
            "sdk_mismatch_count": EXPECTED_PREFLIGHT_MISMATCH_COUNT,
            "sha256": "c" * 64,
        },
        "schema": AUTHORITY_SCHEMA,
        "source_commit": SOURCE_COMMIT,
    }
    return summary, heuristic, authority


def rebound_authority(authority: dict[str, Any], summary_raw: bytes,
                      heuristic_raw: bytes) -> dict[str, Any]:
    result = copy.deepcopy(authority)
    bindings = {
        "results/summary.json": summary_raw,
        "results/config-heuristic.json": heuristic_raw,
    }
    for row in result["files"]:
        raw = bindings.get(row["path"])
        if raw is not None:
            row["sha256"] = hashlib.sha256(raw).hexdigest()
            row["size"] = len(raw)
    return result


def expect_rejected(label: str, summary: dict[str, Any], heuristic: dict[str, Any],
                    authority: dict[str, Any], expected: str) -> None:
    summary_raw = (json.dumps(summary, sort_keys=True) + "\n").encode()
    summary_sha = hashlib.sha256(summary_raw).hexdigest()
    heuristic_raw = (json.dumps(heuristic, sort_keys=True) + "\n").encode()
    heuristic_sha = hashlib.sha256(heuristic_raw).hexdigest()
    bound_authority = rebound_authority(authority, summary_raw, heuristic_raw)
    try:
        generate(summary, summary_sha, len(summary_raw),
                 heuristic, heuristic_sha, len(heuristic_raw),
                 bound_authority, "0" * 64, production_census=False)
    except PolicyError as error:
        require(expected in str(error),
                f"negative {label!r} failed for unexpected reason: {error}")
    else:
        raise AssertionError(f"negative {label!r} was accepted")


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        require(path.is_file() and not path.is_symlink(),
                f"output must be a regular non-symlink file: {path}")
        mode = path.stat().st_mode & 0o777
    else:
        mode = 0o644

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n",
                prefix=f".{path.name}.", suffix=".tmp",
                dir=path.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def self_test() -> None:
    summary, heuristic, authority = synthetic_inputs()
    summary_raw = (json.dumps(summary, sort_keys=True) + "\n").encode()
    summary_sha = hashlib.sha256(summary_raw).hexdigest()
    heuristic_raw = (json.dumps(heuristic, sort_keys=True) + "\n").encode()
    heuristic_sha = hashlib.sha256(heuristic_raw).hexdigest()
    first = generate(summary, summary_sha, len(summary_raw),
                     heuristic, heuristic_sha, len(heuristic_raw),
                     authority, "1" * 64,
                     production_census=False)
    second = generate(copy.deepcopy(summary), summary_sha, len(summary_raw),
                      copy.deepcopy(heuristic), heuristic_sha, len(heuristic_raw),
                      copy.deepcopy(authority), "1" * 64,
                      production_census=False)
    require(first == second, "positive self-test is not deterministic")
    require(f'QUACTLIZE_PPU_KQUANT_MEASURED_POLICY_SCHEMA "{OUTPUT_SCHEMA}"'
            in first and
            f'QUACTLIZE_PPU_KQUANT_MEASURED_EXACT_MISS_POLICY '
            f'"{RUNTIME_EXACT_MISS_POLICY}"' in first and
            "X(10, 16, 32, 0, 1)" in first and
            "X(1, 2, Default)" in first and
            "GROUPED_MEASURED" not in first and
            "Tall" not in first,
            "positive self-test emitted an unexpected interface")

    def grouped_shape(document: dict[str, Any], tokens: int = 1) -> dict[str, Any]:
        return next(
            row["shape"] for row in document["rows"]
            if row["qtype"] == 10 and row["operator"] == "grouped" and
            row["shape"]["tokens"] == tokens)

    broken_authority = copy.deepcopy(authority)
    broken_authority["schema"] = "wrong"
    expect_rejected("authority schema", summary, heuristic, broken_authority,
                    "result authority.schema")

    broken_authority = copy.deepcopy(authority)
    broken_authority["runtime_preflight"]["sdk_identity_status"] = "PASS"
    expect_rejected("preflight status", summary, heuristic, broken_authority,
                    "sdk_identity_status must be MISMATCH_ALLOWED")

    broken_authority = copy.deepcopy(authority)
    broken_authority["runtime_preflight"]["sdk_mismatch_count"] = 1
    expect_rejected("preflight mismatch count", summary, heuristic,
                    broken_authority, "sdk_mismatch_count must be 2")

    broken_summary = copy.deepcopy(summary)
    broken_summary["schema"] = "wrong"
    broken_heuristic = copy.deepcopy(heuristic)
    broken_heuristic["source_summary_sha256"] = hashlib.sha256(
        (json.dumps(broken_summary, sort_keys=True) + "\n").encode()).hexdigest()
    expect_rejected("summary schema", broken_summary, broken_heuristic, authority,
                    "summary.schema")

    broken_summary = copy.deepcopy(summary)
    broken_summary["all_configs"] = False
    broken_heuristic = copy.deepcopy(heuristic)
    broken_heuristic["source_summary_sha256"] = hashlib.sha256(
        (json.dumps(broken_summary, sort_keys=True) + "\n").encode()).hexdigest()
    expect_rejected("all configs", broken_summary, broken_heuristic, authority,
                    "all_configs")

    broken_summary = copy.deepcopy(summary)
    candidates = broken_summary["rows"][0]["candidates"]["kpack"]
    nonwinner = next(
        index for index, candidate in enumerate(candidates)
        if candidate["config"] != "64x64:32x32:s3")
    del candidates[nonwinner]
    expect_rejected("candidate census", broken_summary, heuristic, authority,
                    "candidate census differs")

    broken_summary = copy.deepcopy(summary)
    broken_summary["rows"][0]["shape"]["m"] = CPP_INT32_MAX + 1
    expect_rejected("summary C++ int32", broken_summary, heuristic, authority,
                    "must fit the C++ int32 boundary")

    broken_summary = copy.deepcopy(summary)
    grouped_shape(broken_summary)["experts"] = EXPECTED_GROUPED_EXPERTS - 1
    expect_rejected("grouped experts", broken_summary, heuristic, authority,
                    "shape.experts must be 256")

    broken_summary = copy.deepcopy(summary)
    grouped_shape(broken_summary)["topk"] = EXPECTED_GROUPED_TOPK - 1
    expect_rejected("grouped topk", broken_summary, heuristic, authority,
                    "shape.topk must be 8")

    broken_summary = copy.deepcopy(summary)
    grouped_shape(broken_summary)["router"] = "different-router"
    expect_rejected("grouped router", broken_summary, heuristic, authority,
                    "shape.router must be token-topk-hot16x4-wor-sm64-s44-v1")

    broken_summary = copy.deepcopy(summary)
    collision = grouped_shape(broken_summary, tokens=2)
    collision["total_rows"] = EXPECTED_GROUPED_TOPK
    collision["max_rows"] = 1
    expect_rejected("grouped signature injection", broken_summary, heuristic,
                    authority, "signatures must map one-to-one to tokens")

    broken_heuristic = copy.deepcopy(heuristic)
    broken_heuristic["source_summary_sha256"] = "f" * 64
    expect_rejected("source sha", summary, broken_heuristic, authority,
                    "source_summary_sha256")

    broken_heuristic = copy.deepcopy(heuristic)
    broken_heuristic["boards"][0]["family_tables"][0]["intervals"][0][
        "dynamic_max"] = CPP_INT32_MAX + 1
    expect_rejected("heuristic C++ int32", summary, broken_heuristic, authority,
                    "must fit the C++ int32 boundary")

    broken_heuristic = copy.deepcopy(heuristic)
    interval = broken_heuristic["boards"][0]["family_tables"][0]["intervals"][0]
    interval["dynamic_max"] = 1
    interval["measured_values"] = [1]
    expect_rejected("coverage hole", summary, broken_heuristic, authority,
                    "exactly cover")

    broken_heuristic = copy.deepcopy(heuristic)
    broken_heuristic["boards"][0]["family_tables"][0]["intervals"][0]["config"] = \
        "8x128:8x32:s12"
    expect_rejected("candidate absence", summary, broken_heuristic, authority,
                    "absent from candidate set")

    broken_heuristic = copy.deepcopy(heuristic)
    interval = broken_heuristic["boards"][0]["family_tables"][0]["intervals"][0]
    interval["config"] = "128x64:64x32:s2"
    interval["max_regret"] = 0.1
    interval["mean_regret"] = 0.1
    expect_rejected("regret bound", summary, broken_heuristic, authority,
                    "exceeds 3% regret")

    with tempfile.TemporaryDirectory(prefix="qz-kquant-evidence-") as temporary:
        root = Path(temporary) / "root"
        payload_path = root / "nested" / "payload.bin"
        payload_path.parent.mkdir(parents=True)
        payload = b"authority"
        payload_path.write_bytes(payload)
        records = {
            "nested/payload.bin": (
                hashlib.sha256(payload).hexdigest(), len(payload)),
        }
        try:
            validate_evidence_files(records, (), required=True)
        except PolicyError as error:
            require("requires at least one --evidence-root" in str(error),
                    f"missing-evidence-root negative failed unexpectedly: {error}")
        else:
            raise AssertionError("production authority without evidence root was accepted")
        validate_evidence_files(records, (root,), required=True)
        payload_path.write_bytes(b"AUTHORITY")
        try:
            validate_evidence_files(records, (root,), required=True)
        except PolicyError as error:
            require("SHA-256 differs" in str(error),
                    f"evidence-bytes negative failed unexpectedly: {error}")
        else:
            raise AssertionError("modified authority evidence bytes were accepted")

    with tempfile.TemporaryDirectory(prefix="qz-kquant-atomic-") as temporary:
        output = Path(temporary) / "policy.inc"
        output.write_text("stable\n")
        original_fsync = os.fsync

        def fail_before_replace(_descriptor: int) -> None:
            raise OSError("planted pre-replace fsync failure")

        os.fsync = fail_before_replace
        try:
            try:
                atomic_write_text(output, "truncated\n")
            except OSError as error:
                require("planted pre-replace" in str(error),
                        f"atomic-write negative failed unexpectedly: {error}")
            else:
                raise AssertionError("planted pre-replace failure was accepted")
        finally:
            os.fsync = original_fsync
        require(output.read_text() == "stable\n",
                "pre-replace failure changed the existing output")
        require(not list(output.parent.glob(f".{output.name}.*.tmp")),
                "pre-replace failure left a temporary output")
        atomic_write_text(output, "replacement\n")
        require(output.read_text() == "replacement\n",
                "atomic output replacement did not publish complete bytes")

    with tempfile.TemporaryDirectory(prefix="qz-kquant-policy-") as temporary:
        duplicate = Path(temporary) / "duplicate.json"
        duplicate.write_text('{"schema":"first","schema":"second"}\n')
        try:
            load_json(duplicate, "duplicate self-test")
        except PolicyError as error:
            require("repeats JSON key" in str(error),
                    f"duplicate-key negative failed unexpectedly: {error}")
        else:
            raise AssertionError("duplicate JSON key was accepted")
    print("[fq-kquant-measured-policy:self-test] PASS deterministic generation; "
          "dense-only/fallback/preflight/int32/grouped/census/evidence/atomic/"
          "schema/SHA/coverage/regret/JSON negatives RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("generate", "self-test"),
                        default="generate")
    parser.add_argument("--self-test", action="store_true",
                        help="run the built-in positive and negative tests")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--config-heuristic", "--heuristic", dest="heuristic", type=Path)
    parser.add_argument("--authority", type=Path)
    parser.add_argument(
        "--evidence-root", action="append", type=Path, default=[],
        help="repeat for each root needed to resolve every authority.files path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true",
                        help="verify byte equality without writing")
    args = parser.parse_args()
    try:
        if args.self_test or args.command == "self-test":
            require(not args.check, "--check cannot be combined with self-test")
            self_test()
            return 0
        require(args.summary is not None, "--summary is required")
        require(args.heuristic is not None, "--config-heuristic is required")
        require(args.authority is not None, "--authority is required")
        summary, summary_sha, summary_size = load_json(args.summary, "summary")
        heuristic, heuristic_sha, heuristic_size = load_json(
            args.heuristic, "config heuristic")
        authority, authority_sha, _authority_size = load_json(
            args.authority, "result authority")
        wanted = generate(summary, summary_sha, summary_size,
                          heuristic, heuristic_sha, heuristic_size,
                          authority, authority_sha,
                          production_census=True,
                          evidence_roots=tuple(args.evidence_root))
        if args.check:
            current = args.output.read_text() if args.output.is_file() else ""
            if current != wanted:
                print(f"[fq-kquant-measured-policy] FAIL stale or missing {args.output}",
                      file=os.sys.stderr)
                return 1
            print("[fq-kquant-measured-policy] PASS authority-complete "
                  "dense-only policy: 44 families and 110 intervals; "
                  "grouped evidence validated but not emitted")
            return 0
        atomic_write_text(args.output, wanted)
        print(f"wrote {args.output} (authority-complete dense-only policy: "
              f"{EXPECTED_DENSE_FAMILIES} families, "
              f"{EXPECTED_DENSE_INTERVALS} intervals; grouped evidence "
              "validated but not emitted)")
        return 0
    except (AssertionError, OSError, PolicyError) as error:
        print(f"[fq-kquant-measured-policy] FAIL: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
