#!/usr/bin/env python3
"""Adjudicate the two frozen box runs from their preregistered policy.

The policy is deliberately not a Python constant.  It is the single JSON block
inside dev/fold_derivation/BOX_RUN_PREREGISTRATION.md.  Publication mode reads
that block, verifies its generated mirror and the hash of every other byte in
the document, and then applies it to either a dense observation JSON or the
GEMV raw+manifest pair.

This file does not build or launch either kernel.  It is safe while the GEMV
and Marlin implementation paths are frozen for the queued box measurements.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib.util
import json
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "dev" / "fold_derivation" / "BOX_RUN_PREREGISTRATION.md"
POLICY_BEGIN = "<!-- BOX_RUN_POLICY_V1_BEGIN -->"
POLICY_END = "<!-- BOX_RUN_POLICY_V1_END -->"
MIRROR_BEGIN = "<!-- BOX_RUN_POLICY_MIRROR_V1_BEGIN -->"
MIRROR_END = "<!-- BOX_RUN_POLICY_MIRROR_V1_END -->"
POLICY_SCHEMA = "quactlize-box-run-policy-v1"
DENSE_SCHEMA = "quactlize-dense-box-observation-v1"
PUBLICATION_CODE_PATHS = (
    "tools/adjudicate_box_runs.py",
    "benchmarks/sweep_gemv_perf.py",
)
_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedPolicy:
    value: dict[str, Any]
    policy_sha256: str
    document_sha256: str
    source: str


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise PolicyError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _region(text: str, begin: str, end: str) -> tuple[str, int, int]:
    if text.count(begin) != 1 or text.count(end) != 1:
        raise PolicyError(f"expected exactly one {begin!r}/{end!r} region")
    start = text.index(begin)
    content_start = start + len(begin)
    stop = text.index(end, content_start)
    if stop < content_start:
        raise PolicyError(f"malformed {begin!r}/{end!r} region")
    return text[content_start:stop].strip(), start, stop + len(end)


def _expect_keys(obj: Any, keys: Iterable[str], where: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise PolicyError(f"{where} must be an object")
    expected = set(keys)
    got = set(obj)
    if got != expected:
        raise PolicyError(
            f"{where} keys differ: missing={sorted(expected - got)} extra={sorted(got - expected)}")
    return obj


def _decimal_string(value: Any, where: str) -> Decimal:
    if not isinstance(value, str) or not _DECIMAL.fullmatch(value):
        raise PolicyError(f"{where} must be a canonical non-negative decimal string")
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise PolicyError(f"{where} is not a finite decimal") from exc


def _integer(value: Any, where: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PolicyError(f"{where} must be an integer")
    if value < (1 if positive else 0):
        raise PolicyError(f"{where} is outside its non-negative domain")
    return value


def _semantic_policy(policy: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(policy)
    out.pop("prose_sha256", None)
    return out


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def policy_digest(policy: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(_semantic_policy(policy)).encode()).hexdigest()


def _reason_text(census: dict[str, Any]) -> str:
    return ",".join(f"{key}:{census['prune_reasons'][key]}"
                    for key in sorted(census["prune_reasons"])) or "none"


def render_policy_mirror(policy: dict[str, Any]) -> str:
    """Canonical human-visible mirror.  It is evidence, never an input."""
    dense = policy["dense"]
    gemv = policy["gemv"]
    classic = Decimal(dense["classic_anchor_us"])
    historical = Decimal(dense["historical_anchor_us"])
    fraction = Decimal(dense["converged_recovered_fraction"])
    gap = historical - classic
    boundary = historical - fraction * gap
    cells = ",".join(
        f"WK{cell['warp_k']}/B{cell['blocks_per_cu']}:{cell['role']}"
        for cell in dense["cells"])
    lines = [
        f"policy_sha256={policy_digest(policy)}",
        ("dense anchors_us=" + dense["classic_anchor_us"] + "/" +
         dense["historical_anchor_us"] + " recovered_fraction=" +
         dense["converged_recovered_fraction"] + f" derived_gap_us={gap} "
         f"derived_boundary_us={boundary} samples={dense['sample_count']}"),
        ("dense problem=" + _canonical(dense["problem"]) +
         " decomposition=" + _canonical(dense["decomposition"]) +
         " invocation=" + _canonical(dense["invocation"])),
        (f"dense primary=WK{dense['primary_cell']['warp_k']}/"
         f"B{dense['primary_cell']['blocks_per_cu']} cells={cells}"),
        ("dense prerequisites=" + ",".join(dense["required_prerequisites"]) +
         f" wk1_byte_map={dense['wk1_admission']['byte_map_diff']}/"
         f"{dense['wk1_admission']['byte_map_total']}"),
        ("gemv minimum_claimable_us=" + gemv["minimum_claimable_us"] +
         f" timer_normalization_us={gemv['timer_normalization_us']} "
         f"samples={gemv['sample_count']} "
         f" publication_partial_space={str(gemv['publication_partial_space']).lower()} "
         f"resolution_rule={_canonical(gemv['resolution_rule'])} "
         f"incumbent_rules={_canonical(gemv['incumbent_rules'])}"),
    ]
    for name in ("base_census", "full_manifest", "smoke_manifest"):
        census = gemv[name]
        jobs = f" jobs={census['jobs']}" if "jobs" in census else ""
        lines.append(
            f"gemv {name}{jobs} total={census['total']} legal={census['legal']} "
            f"pruned={census['pruned']} reasons={_reason_text(census)}")
    return "\n".join(lines)


def _validate_census(value: Any, where: str, *, jobs: bool) -> dict[str, Any]:
    keys = ["total", "legal", "pruned", "prune_reasons"] + (["jobs"] if jobs else [])
    out = _expect_keys(value, keys, where)
    total = _integer(out["total"], f"{where}.total")
    legal = _integer(out["legal"], f"{where}.legal")
    pruned = _integer(out["pruned"], f"{where}.pruned")
    if jobs:
        _integer(out["jobs"], f"{where}.jobs", positive=True)
    reasons = out["prune_reasons"]
    if not isinstance(reasons, dict) or any(
            not isinstance(k, str) or not k or isinstance(v, bool) or
            not isinstance(v, int) or v < 0 for k, v in reasons.items()):
        raise PolicyError(f"{where}.prune_reasons must map names to non-negative integers")
    if total != legal + pruned:
        raise PolicyError(f"{where}: total != legal + pruned")
    if sum(reasons.values()) != pruned:
        raise PolicyError(f"{where}: prune histogram does not sum to pruned")
    return out


def validate_policy(policy: Any) -> dict[str, Any]:
    top = _expect_keys(policy, ["schema", "prose_sha256", "dense", "gemv"], "policy")
    if top["schema"] != POLICY_SCHEMA:
        raise PolicyError(f"policy schema must be {POLICY_SCHEMA!r}")
    if not isinstance(top["prose_sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", top["prose_sha256"]):
        raise PolicyError("policy.prose_sha256 must be a lowercase SHA-256")

    dense = _expect_keys(
        top["dense"],
        ["classic_anchor_us", "historical_anchor_us", "converged_recovered_fraction",
         "sample_count", "problem", "decomposition", "invocation", "primary_cell",
         "required_prerequisites", "wk1_admission", "cells"],
        "policy.dense")
    classic = _decimal_string(dense["classic_anchor_us"], "policy.dense.classic_anchor_us")
    historical = _decimal_string(
        dense["historical_anchor_us"], "policy.dense.historical_anchor_us")
    fraction = _decimal_string(
        dense["converged_recovered_fraction"],
        "policy.dense.converged_recovered_fraction")
    if not (Decimal(0) < classic < historical):
        raise PolicyError("dense anchors must satisfy 0 < classic < historical")
    if not (Decimal(0) < fraction <= Decimal(1)):
        raise PolicyError("dense recovered fraction must be in (0,1]")
    _integer(dense["sample_count"], "policy.dense.sample_count", positive=True)
    problem = _expect_keys(
        dense["problem"], ["m", "n", "k", "l", "group_size"],
        "policy.dense.problem")
    for axis in problem:
        _integer(problem[axis], f"policy.dense.problem.{axis}", positive=True)
    decomposition = _expect_keys(
        dense["decomposition"], ["output_tiles", "k_tiles"],
        "policy.dense.decomposition")
    for axis in decomposition:
        _integer(decomposition[axis], f"policy.dense.decomposition.{axis}", positive=True)
    invocation = _expect_keys(
        dense["invocation"], ["flags", "options"], "policy.dense.invocation")
    flags = invocation["flags"]
    if (not isinstance(flags, list) or not flags or
            any(not isinstance(flag, str) or not flag.startswith("--") for flag in flags) or
            len(flags) != len(set(flags))):
        raise PolicyError("dense invocation flags must be unique --options")
    invocation_options = _expect_keys(
        invocation["options"], ["mode", "alpha", "beta"],
        "policy.dense.invocation.options")
    _integer(invocation_options["mode"], "policy.dense.invocation.options.mode")
    for axis in ("alpha", "beta"):
        _decimal_string(invocation_options[axis], f"policy.dense.invocation.options.{axis}")
    primary = _expect_keys(dense["primary_cell"], ["warp_k", "blocks_per_cu"],
                           "policy.dense.primary_cell")
    primary_key = (_integer(primary["warp_k"], "primary warp_k", positive=True),
                   _integer(primary["blocks_per_cu"], "primary blocks_per_cu", positive=True))
    prereqs = dense["required_prerequisites"]
    if (not isinstance(prereqs, list) or not prereqs or
            any(not isinstance(x, str) or not x for x in prereqs) or
            len(set(prereqs)) != len(prereqs)):
        raise PolicyError("dense prerequisites must be unique non-empty strings")
    wk1 = _expect_keys(dense["wk1_admission"], ["byte_map_total", "byte_map_diff"],
                       "policy.dense.wk1_admission")
    _integer(wk1["byte_map_total"], "wk1 byte_map_total", positive=True)
    _integer(wk1["byte_map_diff"], "wk1 byte_map_diff")
    cells = dense["cells"]
    if not isinstance(cells, list) or not cells:
        raise PolicyError("dense cells must be a non-empty array")
    seen: set[tuple[int, int]] = set()
    primary_roles = 0
    for i, item in enumerate(cells):
        cell = _expect_keys(item, ["warp_k", "blocks_per_cu", "role"], f"dense.cells[{i}]")
        key = (_integer(cell["warp_k"], f"dense.cells[{i}].warp_k", positive=True),
               _integer(cell["blocks_per_cu"], f"dense.cells[{i}].blocks_per_cu", positive=True))
        if key in seen:
            raise PolicyError(f"duplicate dense cell {key}")
        seen.add(key)
        if not isinstance(cell["role"], str) or not cell["role"]:
            raise PolicyError(f"dense.cells[{i}].role must be a non-empty string")
        if cell["role"] == "primary":
            primary_roles += 1
            if key != primary_key:
                raise PolicyError("dense primary role disagrees with primary_cell")
    if primary_roles != 1 or primary_key not in seen:
        raise PolicyError("dense policy must contain exactly one declared primary cell")

    gemv = _expect_keys(
        top["gemv"],
        ["minimum_claimable_us", "timer_normalization_us", "sample_count",
         "publication_partial_space", "resolution_rule", "incumbent_rules",
         "base_census", "full_manifest", "smoke_manifest"], "policy.gemv")
    if _decimal_string(gemv["minimum_claimable_us"], "policy.gemv.minimum_claimable_us") <= 0:
        raise PolicyError("GEMV minimum claimable time must be positive")
    if _decimal_string(gemv["timer_normalization_us"],
                       "policy.gemv.timer_normalization_us") <= 0:
        raise PolicyError("GEMV timer normalization must be positive")
    _integer(gemv["sample_count"], "policy.gemv.sample_count", positive=True)
    if not isinstance(gemv["publication_partial_space"], bool):
        raise PolicyError("GEMV publication_partial_space must be boolean")
    resolution = _expect_keys(
        gemv["resolution_rule"],
        ["require_runner_up", "require_disjoint_bands", "max_unresolved_quanta"],
        "policy.gemv.resolution_rule")
    for key in ("require_runner_up", "require_disjoint_bands"):
        if not isinstance(resolution[key], bool):
            raise PolicyError(f"GEMV resolution_rule.{key} must be boolean")
    if _decimal_string(resolution["max_unresolved_quanta"],
                       "policy.gemv.resolution_rule.max_unresolved_quanta") < 0:
        raise PolicyError("GEMV max_unresolved_quanta must be non-negative")
    incumbent_rules = _expect_keys(
        gemv["incumbent_rules"], ["geometry_cta_m", "format_axes"],
        "policy.gemv.incumbent_rules")
    geometry_cta_m = incumbent_rules["geometry_cta_m"]
    if not isinstance(geometry_cta_m, dict) or not geometry_cta_m or any(
            not isinstance(k, str) or not k or isinstance(v, bool) or
            not isinstance(v, int) or v <= 0 for k, v in geometry_cta_m.items()):
        raise PolicyError("GEMV geometry_cta_m must map geometry IDs to positive integers")
    format_axes = incumbent_rules["format_axes"]
    if not isinstance(format_axes, dict) or not format_axes:
        raise PolicyError("GEMV format_axes must be a non-empty object")
    for fmt, axes in format_axes.items():
        if not isinstance(fmt, str) or not fmt:
            raise PolicyError("GEMV incumbent format names must be non-empty")
        axes = _expect_keys(
            axes, ["layout", "step_k", "threads", "cta_n", "chunk"],
            f"policy.gemv.incumbent_rules.format_axes.{fmt}")
        if not isinstance(axes["layout"], str) or not axes["layout"]:
            raise PolicyError(f"GEMV incumbent {fmt} layout must be non-empty")
        for axis in ("step_k", "threads", "cta_n", "chunk"):
            _integer(axes[axis], f"GEMV incumbent {fmt}.{axis}", positive=True)
    _validate_census(gemv["base_census"], "policy.gemv.base_census", jobs=False)
    _validate_census(gemv["full_manifest"], "policy.gemv.full_manifest", jobs=True)
    _validate_census(gemv["smoke_manifest"], "policy.gemv.smoke_manifest", jobs=True)
    return top


def load_policy(path: pathlib.Path | str = DEFAULT_POLICY) -> LoadedPolicy:
    source = pathlib.Path(path)
    return _load_policy_text(source.read_text(), str(source))


def _load_policy_text(text: str, source: str) -> LoadedPolicy:
    block, block_start, block_stop = _region(text, POLICY_BEGIN, POLICY_END)
    mirror, _, _ = _region(text, MIRROR_BEGIN, MIRROR_END)
    try:
        policy = json.loads(block, object_pairs_hook=_pairs_without_duplicates)
    except (json.JSONDecodeError, PolicyError) as exc:
        raise PolicyError(f"invalid policy JSON: {exc}") from exc
    policy = validate_policy(policy)
    expected_mirror = render_policy_mirror(policy)
    if mirror != expected_mirror:
        raise PolicyError("policy mirror differs from the JSON authority")
    prose = text[:block_start] + text[block_stop:]
    prose_sha = hashlib.sha256(prose.encode()).hexdigest()
    if prose_sha != policy["prose_sha256"]:
        raise PolicyError(
            f"preregistration prose drift: got {prose_sha}, policy records {policy['prose_sha256']}")
    return LoadedPolicy(
        value=policy,
        policy_sha256=policy_digest(policy),
        document_sha256=hashlib.sha256(text.encode()).hexdigest(),
        source=source,
    )


def load_policy_from_git(root_sha: str, repo: pathlib.Path | str = ROOT) -> LoadedPolicy:
    if not re.fullmatch(r"[0-9a-f]{40,64}", root_sha):
        raise PolicyError(f"invalid result root SHA {root_sha!r}")
    rel = DEFAULT_POLICY.relative_to(ROOT).as_posix()
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{root_sha}:{rel}"],
        text=True, capture_output=True)
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()
        raise PolicyError(
            f"cannot read preregistration from result root SHA {root_sha}: "
            f"{detail[-1] if detail else 'git show failed'}")
    return _load_policy_text(proc.stdout, f"git:{root_sha}:{rel}")


def publication_code_errors(root_sha: str, repo: pathlib.Path | str = ROOT,
                            current_root: pathlib.Path | str = ROOT) -> list[str]:
    """Refuse to reinterpret an old bundle with changed decision code.

    Result data may be committed after the measured SHA, so HEAD equality is
    unnecessarily strict.  Exact file equality pins the reader and raw GEMV
    analyser while allowing a later data-only commit to consume the bundle.
    """
    errors: list[str] = []
    for relative in PUBLICATION_CODE_PATHS:
        proc = subprocess.run(
            ["git", "-C", str(repo), "show", f"{root_sha}:{relative}"],
            capture_output=True)
        if proc.returncode != 0:
            errors.append(f"cannot read {relative} from result SHA")
            continue
        try:
            current = (pathlib.Path(current_root) / relative).read_bytes()
        except OSError as exc:
            errors.append(f"cannot read current {relative}: {exc}")
            continue
        if proc.stdout != current:
            errors.append(
                f"current {relative} differs from result SHA; adjudicate from a clean worktree "
                f"at {root_sha}")
    return errors


def _decimal_input(value: Any, where: str) -> Decimal:
    try:
        return _decimal_string(value, where)
    except PolicyError as exc:
        raise ValueError(str(exc)) from exc


def _decimal_output(value: Decimal) -> str:
    return format(value, "f")


def _policy_identity(loaded: LoadedPolicy) -> dict[str, str]:
    return {
        "source": loaded.source,
        "policy_sha256": loaded.policy_sha256,
        "document_sha256": loaded.document_sha256,
    }


def adjudicate_dense(loaded: LoadedPolicy, observation: Any) -> dict[str, Any]:
    policy = loaded.value["dense"]
    reasons: list[str] = []
    cells_out: list[dict[str, Any]] = []
    if not isinstance(observation, dict) or observation.get("schema") != DENSE_SCHEMA:
        return {"schema": "quactlize-box-adjudication-v1", "kind": "dense",
                "verdict": "VOID", "reasons": ["wrong or missing dense observation schema"],
                "policy": _policy_identity(loaded), "cells": []}

    if observation.get("identity_valid") is not True:
        reasons.append("identity/provenance prerequisites did not close")
    wk_obs = observation.get("wk1_admission")
    wk_policy = policy["wk1_admission"]
    wk1_ok = isinstance(wk_obs, dict)
    if wk1_ok:
        wk1_ok = (wk_obs.get("structural_identity") is True and
                  wk_obs.get("byte_map_total") == wk_policy["byte_map_total"] and
                  wk_obs.get("byte_map_diff") == wk_policy["byte_map_diff"])
        if wk_obs.get("device_control_present") is True:
            wk1_ok = wk1_ok and wk_obs.get("device_raw_bitdiff") == 0
    if not wk1_ok:
        reasons.append("WK1 structural/byte/device admission did not close")

    # WK1 is an admission condition, not one more timing cell.  Once it fails,
    # interpreting the candidate timings would attach a category to an experiment
    # whose historical control changed.  Preserve only the raw identities/statuses.
    if not wk1_ok or observation.get("identity_valid") is not True:
        raw = observation.get("cells")
        if not isinstance(raw, list):
            raw = []
            reasons.append("dense observation cells must be an array")
        for item in raw:
            if isinstance(item, dict):
                cells_out.append({
                    "warp_k": item.get("warp_k"),
                    "blocks_per_cu": item.get("blocks_per_cu"),
                    "status": item.get("status"),
                    "verdict": "NOT_ADJUDICATED",
                })
        return {
            "schema": "quactlize-box-adjudication-v1", "kind": "dense",
            "verdict": "VOID", "boundary_unresolved": False,
            "reasons": reasons, "policy": _policy_identity(loaded),
            "cells": cells_out,
        }

    registered = {(c["warp_k"], c["blocks_per_cu"]): c for c in policy["cells"]}
    primary_key = (policy["primary_cell"]["warp_k"],
                   policy["primary_cell"]["blocks_per_cu"])
    observed_keys: set[tuple[int, int]] = set()
    primary_output: dict[str, Any] | None = None
    raw_cells = observation.get("cells")
    if not isinstance(raw_cells, list):
        raw_cells = []
        reasons.append("dense observation cells must be an array")

    classic = Decimal(policy["classic_anchor_us"])
    historical = Decimal(policy["historical_anchor_us"])
    fraction = Decimal(policy["converged_recovered_fraction"])
    gap = historical - classic
    boundary = historical - fraction * gap

    for index, raw in enumerate(raw_cells):
        if not isinstance(raw, dict):
            reasons.append(f"cell[{index}] is not an object")
            cells_out.append({"verdict": "UNREGISTERED", "reason": "cell is not an object"})
            continue
        wk = raw.get("warp_k")
        bpc = raw.get("blocks_per_cu")
        if (isinstance(wk, bool) or not isinstance(wk, int) or wk <= 0 or
                isinstance(bpc, bool) or not isinstance(bpc, int) or bpc <= 0):
            reasons.append(f"cell[{index}] has a non-positive/non-integer identity")
            cells_out.append({
                "warp_k": wk, "blocks_per_cu": bpc,
                "verdict": "UNREGISTERED", "reason": "invalid cell identity",
            })
            continue
        key = (wk, bpc)
        base = {"warp_k": wk, "blocks_per_cu": bpc}
        if key in observed_keys:
            reasons.append(f"duplicate dense cell WK{wk}/B{bpc}")
            cells_out.append(dict(base, verdict="UNREGISTERED", reason="duplicate cell"))
            continue
        observed_keys.add(key)
        declared = registered.get(key)
        if declared is None:
            reasons.append(f"unregistered dense cell WK{wk}/B{bpc}")
            cells_out.append(dict(base, verdict="UNREGISTERED", reason="not in preregistration"))
            continue
        role = declared["role"]
        status = raw.get("status")
        out = dict(base, role=role, status=status)
        cell_errors: list[str] = []
        if role == "compile_negative":
            if status != "PREREGISTERED_COMPILE_NEGATIVE_NOT_RERUN":
                cell_errors.append("WK2 must remain the preregistered compile-negative control")
            out["verdict"] = (
                "PREREGISTERED_COMPILE_NEGATIVE_NOT_RERUN" if not cell_errors else "VOID")
        elif wk == 1 and role == "shipping_default_control":
            if status != "ADMISSION_CONTROL":
                cell_errors.append("WK1/B1 must be the executable admission control")
            out["verdict"] = "CONTROL" if not cell_errors else "VOID"
        elif wk == 1:
            if status != "NOT_IN_QUEUED_RUN":
                cell_errors.append("WK1 scheduler diagnostic is outside this queued run")
            out["verdict"] = "NOT_IN_QUEUED_RUN" if not cell_errors else "VOID"
        elif status == "NOT_RUN":
            if key == primary_key:
                cell_errors.append("primary cell was not run")
                out["verdict"] = "VOID"
            else:
                out["verdict"] = "NOT_RUN"
        elif status != "RUN":
            cell_errors.append("timing cell status must be RUN or NOT_RUN")
            out["verdict"] = "VOID"
        else:
            if raw.get("sample_count") != policy["sample_count"]:
                cell_errors.append("sample count differs from preregistration")
            prereqs = raw.get("prerequisites")
            if not isinstance(prereqs, dict):
                cell_errors.append("prerequisites are missing")
            else:
                missing = [name for name in policy["required_prerequisites"]
                           if prereqs.get(name) is not True]
                if missing:
                    cell_errors.append("failed prerequisites: " + ",".join(missing))
            try:
                median = _decimal_input(raw.get("median_us"), "median_us")
                low = _decimal_input(raw.get("min_us"), "min_us")
                high = _decimal_input(raw.get("max_us"), "max_us")
                if not (low <= median <= high):
                    cell_errors.append("timing band does not contain median")
            except ValueError as exc:
                cell_errors.append(str(exc))
                median = low = high = Decimal(0)
            if cell_errors:
                out["verdict"] = "VOID"
            else:
                out.update({
                    "median_us": _decimal_output(median),
                    "band_us": [_decimal_output(low), _decimal_output(high)],
                    "sample_count": raw.get("sample_count"),
                })
                if isinstance(raw.get("decomposition"), dict):
                    out["decomposition"] = raw["decomposition"]
                if key == primary_key:
                    recovered = (historical - median) / gap
                    if recovered >= fraction:
                        out["verdict"] = "CONVERGED_OR_BETTER"
                    elif recovered > 0:
                        out["verdict"] = "PARTIAL"
                    else:
                        out["verdict"] = "NO_RECOVERY_OR_WORSE"
                    crossed = [name for name, value in
                               (("converged_partial", boundary),
                                ("partial_no_recovery", historical))
                               if low < value < high]
                    out.update({
                        "recovered_gap_fraction": _decimal_output(recovered),
                        "boundary_unresolved": bool(crossed),
                        "crossed_boundaries": crossed,
                    })
                elif role == "shipping_default_control":
                    out["verdict"] = "CONTROL"
                else:
                    out["verdict"] = "DIAGNOSTIC"
        if cell_errors:
            out["reasons"] = cell_errors
            reasons.extend(f"WK{wk}/B{bpc}: {reason}" for reason in cell_errors)
        cells_out.append(out)
        if key == primary_key:
            primary_output = out

    missing_registered = sorted(set(registered) - observed_keys)
    if missing_registered:
        reasons.append("registered dense cells are absent: " + ",".join(
            f"WK{wk}/B{bpc}" for wk, bpc in missing_registered))
    if primary_output is None:
        reasons.append("primary dense cell is absent")
    if reasons or primary_output is None or primary_output.get("verdict") == "VOID":
        verdict = "VOID"
    else:
        verdict = primary_output["verdict"]
    return {
        "schema": "quactlize-box-adjudication-v1",
        "kind": "dense",
        "verdict": verdict,
        "boundary_unresolved": bool(primary_output and
                                    primary_output.get("boundary_unresolved")),
        "reasons": reasons,
        "policy": _policy_identity(loaded),
        "cells": cells_out,
    }


def _load_sweep_module():
    path = ROOT / "benchmarks" / "sweep_gemv_perf.py"
    spec = importlib.util.spec_from_file_location("quactlize_sweep_gemv_perf", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expected_gemv_groups(manifest: dict[str, Any]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for job in manifest.get("jobs", []):
        for item in job.get("expected", []):
            out.add((job.get("shape_id"), item.get("format")))
    return out


def _manifest_census_errors(manifest: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("partial_space") != policy["publication_partial_space"]:
        errors.append("manifest partial_space differs from publication policy")
    wanted = policy["full_manifest"]
    counts = manifest.get("counts", {})
    for key in ("total", "legal", "pruned", "prune_reasons"):
        if counts.get(key) != wanted[key]:
            errors.append(f"full manifest {key} differs from preregistration")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != wanted["jobs"]:
        errors.append("full manifest job count differs from preregistration")
    return errors


def _manifest_group_configs(manifest: dict[str, Any]) -> dict[tuple[str, str], set[str]]:
    out: dict[tuple[str, str], set[str]] = {}
    for job in manifest.get("jobs", []):
        shape_id = job.get("shape_id")
        for item in job.get("expected", []):
            key = (shape_id, item.get("format"))
            out.setdefault(key, set()).add(item.get("config_id"))
    return out


def _registered_incumbent(gemv_policy: dict[str, Any], shape_id: str, fmt: str,
                          shape: Any) -> tuple[str | None, str | None]:
    """Return the preregistered shipping route, or an intentional-unknown reason.

    The rules are the policy's source authority.  Derivation here merely renders
    their axes in the config-id syntax emitted by gemv_perf_manifest.hpp.
    """
    if not isinstance(shape, dict):
        return None, "shape identity is absent or malformed"
    if shape.get("semantic") != "shipping":
        return None, f"semantic={shape.get('semantic')} has no registered shipping route"
    axes = gemv_policy["incumbent_rules"]["format_axes"].get(fmt)
    if axes is None:
        return None, f"format={fmt} is intentionally outside registered shipping routes"
    geometry = shape_id.split("/", 1)[0]
    cta_m = gemv_policy["incumbent_rules"]["geometry_cta_m"].get(geometry)
    if cta_m is None:
        raise PolicyError(f"shipping geometry {geometry!r} has no registered cta_m")
    route = shape.get("route")
    if route not in ("dense", "grouped"):
        raise PolicyError(f"shipping shape {shape_id!r} has invalid route {route!r}")
    return (f"{fmt}/{axes['layout']}/s{axes['step_k']}/t{axes['threads']}/"
            f"{route}/m{cta_m}/n{axes['cta_n']}/c{axes['chunk']}"), None


def _raw_protocol_errors(data: Any, gemv_policy: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    samples = gemv_policy["sample_count"]
    if len(data.runs) != 1:
        errors.append(
            f"raw data must contain exactly one run identity, found {len(data.runs)}")
    attempts_per_candidate = collections.Counter(
        aid.candidate for aid in data.attempts)
    for candidate, count in sorted(attempts_per_candidate.items(), key=lambda item: repr(item[0])):
        if count != 1:
            errors.append(
                "raw candidate has more than one attempt: "
                f"{candidate.shape_id}/{candidate.fmt}/{candidate.config_id} count={count}")
    for aid, attempt in data.attempts.items():
        if attempt.get("expected_samples") != samples:
            errors.append(
                f"raw attempt {aid!r} expected_samples={attempt.get('expected_samples')} "
                f"differs from preregistered {samples}")
        if aid not in data.exclusions and len(data.samples.get(aid, {})) != samples:
            errors.append(
                f"raw attempt {aid!r} has {len(data.samples.get(aid, {}))}/{samples} samples")
    return errors


def _marlin_dense_decomposition(q: int, k_tiles: int, real_cu: int,
                                blocks_per_cu: int) -> dict[str, int]:
    """Independent integer oracle for Marlin's flat-(q,k) stripe geometry."""
    total = q * k_tiles
    grid = max(q, real_cu * blocks_per_cu)
    stripe = (total + grid - 1) // grid
    active = (total + stripe - 1) // stripe
    peers = [
        (((tile + 1) * k_tiles - 1) // stripe) -
        ((tile * k_tiles) // stripe) + 1
        for tile in range(q)
    ]
    return {
        "grid_ctas": grid,
        "stripe_iters": stripe,
        "active_ctas": active,
        "idle_ctas": grid - active,
        "handoffs": sum(peer - 1 for peer in peers),
        "max_peers": max(peers),
    }


def _candidate_evidence(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "config_id": candidate.get("config_id"),
        "median_us": candidate.get("median_us"),
        "raw_samples": candidate.get("raw_samples"),
        "raw_band_us": candidate.get("band_us"),
    }


def _registered_gemv_resolution(
        group: dict[str, Any], policy: dict[str, Any]
) -> tuple[str, list[str], dict[str, Any]]:
    """Apply the JSON-registered resolution rule to analyser-produced facts.

    The analyser owns raw decoding and timer-lattice inference.  It does not own
    the decision boundary: runner-up, disjoint-band, and quantum requirements
    all come from the sealed preregistration and are checked independently here.
    """
    rule = policy["resolution_rule"]
    reasons: list[str] = []
    facts: dict[str, Any] = {
        "normalized_gap_us": None,
        "quantum_us": None,
        "max_unresolved_quanta": rule["max_unresolved_quanta"],
        "unresolved_limit_us": None,
    }
    runner = group.get("runner_up")
    if rule["require_runner_up"] and runner is None:
        reasons.append("NO_RUNNER_UP")
    quantum = group.get("quantum")
    if not isinstance(quantum, dict) or quantum.get("status") != "KNOWN":
        reasons.append("QUANTUM_UNKNOWN")
    overlap = group.get("leader_runner_bands_overlap")
    if rule["require_disjoint_bands"] and overlap is not False:
        reasons.append("BAND_OVERLAP" if overlap is True else "BAND_RELATION_UNKNOWN")
    gap = group.get("leader_runner_gap_us")
    q_value = quantum.get("us") if isinstance(quantum, dict) else None
    if runner is not None and gap is not None and q_value is not None:
        try:
            normalized = Decimal(str(gap)).quantize(
                Decimal(policy["timer_normalization_us"]), rounding=ROUND_HALF_EVEN)
            q_decimal = Decimal(str(q_value))
            limit = q_decimal * Decimal(rule["max_unresolved_quanta"])
            facts.update({
                "normalized_gap_us": _decimal_output(normalized),
                "quantum_us": _decimal_output(q_decimal),
                "unresolved_limit_us": _decimal_output(limit),
            })
            if normalized <= limit:
                reasons.append(
                    "WITHIN_ONE_QUANTUM" if rule["max_unresolved_quanta"] == "1"
                    else "WITHIN_REGISTERED_QUANTUM_LIMIT")
        except InvalidOperation:
            reasons.append("GAP_OR_QUANTUM_MALFORMED")
    elif runner is not None and "QUANTUM_UNKNOWN" not in reasons:
        reasons.append("GAP_OR_QUANTUM_MISSING")
    return ("UNRESOLVED" if reasons else "RESOLVED"), reasons, facts


def adjudicate_gemv(loaded: LoadedPolicy, manifest_path: pathlib.Path | str,
                    raw_paths: Iterable[pathlib.Path | str]) -> dict[str, Any]:
    sweep = _load_sweep_module()
    reasons: list[str] = []
    gemv_policy = loaded.value["gemv"]
    try:
        manifest = json.loads(pathlib.Path(manifest_path).read_text())
        sweep.validate_manifest(manifest)
        reasons.extend(_manifest_census_errors(manifest, gemv_policy))
        data = sweep.load_raw([pathlib.Path(p) for p in raw_paths])
        if Decimal(str(sweep.TIMER_NORMALIZATION_US)) != Decimal(
                gemv_policy["timer_normalization_us"]):
            reasons.append("analyser timer normalization differs from preregistration")
        reasons.extend(_raw_protocol_errors(data, gemv_policy))
        analysed = sweep.analyse_data(
            data,
            min_quantum_us=Decimal(gemv_policy["minimum_claimable_us"]),
            manifest=manifest)
        if not analysed.get("complete"):
            reasons.extend("analysis incomplete: " + text for text in analysed.get("complaints", []))
    except Exception as exc:  # contract failures become an auditable VOID, never a traceback-as-verdict
        reasons.append(f"raw/manifest analysis failed: {exc}")
        manifest = {}
        analysed = {"groups": [], "complete": False}

    analysed_groups = {
        (group["shape_id"], group["format"]): group
        for group in analysed.get("groups", [])
    }
    all_groups = sorted(_expected_gemv_groups(manifest) | set(analysed_groups))
    manifest_configs = _manifest_group_configs(manifest)
    shapes = {job.get("shape_id"): job.get("shape") for job in manifest.get("jobs", [])}
    groups_out: list[dict[str, Any]] = []
    for key in all_groups:
        shape_id, fmt = key
        group = analysed_groups.get(key)
        try:
            incumbent, incumbent_unknown = _registered_incumbent(
                gemv_policy, shape_id, fmt, shapes.get(shape_id))
        except PolicyError as exc:
            reasons.append(str(exc))
            incumbent, incumbent_unknown = None, str(exc)
        if incumbent is not None and incumbent not in manifest_configs.get(key, set()):
            reasons.append(
                f"registered shipping route {incumbent!r} is absent from {shape_id}/{fmt} manifest")
        if group is None:
            groups_out.append({
                "shape_id": shape_id, "format": fmt,
                "measurement_verdict": "UNRESOLVED",
                "measurement_reasons": ["NO_RANKABLE_CANDIDATE"],
                "leader": None, "runner_up": None,
                "leader_runner_gap_us": None, "leader_runner_relative_gap": None,
                "leader_runner_bands_overlap": None,
                "resolution_floor": {
                    "status": "UNKNOWN", "us": None,
                    "minimum_claimable_us": gemv_policy["minimum_claimable_us"]},
                "incumbent": incumbent,
                "incumbent_unknown_reason": incumbent_unknown,
                "routing_verdict": "INCUMBENT_UNKNOWN" if incumbent is None else "UNRESOLVED",
            })
            continue
        candidates = {candidate["config_id"]: candidate for candidate in group["candidates"]}
        leader = candidates.get(group["leader"])
        runner = candidates.get(group["runner_up"])
        ordered = sorted(
            group["candidates"], key=lambda item: (Decimal(str(item["median_us"])),
                                                   item["config_id"]))
        expected_leader = ordered[0]["config_id"] if ordered else None
        expected_runner = ordered[1]["config_id"] if len(ordered) > 1 else None
        if group.get("leader") != expected_leader or group.get("runner_up") != expected_runner:
            reasons.append(f"analyser leader/runner ordering differs from raw medians for {shape_id}/{fmt}")
        registered_measurement, registered_reasons, registered_facts = _registered_gemv_resolution(
            group, gemv_policy)
        # sweep_gemv_perf owns raw decoding, ranking facts, and timer-lattice
        # inference.  Its convenience verdict/reasons are intentionally not an
        # adjudication input: requiring agreement would make those hard-coded
        # labels a second decision authority beside the sealed JSON policy.
        gap = group.get("leader_runner_gap_us")
        relative = None
        if leader is not None and gap is not None:
            leader_median = Decimal(str(leader["median_us"]))
            if leader_median > 0:
                relative = _decimal_output(Decimal(str(gap)) / leader_median)
        if incumbent is None:
            routing = "INCUMBENT_UNKNOWN"
        elif registered_measurement != "RESOLVED":
            routing = "UNRESOLVED"
        elif group["leader"] == incumbent:
            routing = "SHIPPING_ROUTE_REMAINS_LEADER"
        else:
            routing = "CHANGED_FROM_SHIPPING_ROUTE"
        groups_out.append({
            "shape_id": shape_id,
            "format": fmt,
            "measurement_verdict": registered_measurement,
            "measurement_reasons": registered_reasons,
            "leader": group["leader"],
            "runner_up": group["runner_up"],
            "leader_evidence": _candidate_evidence(leader),
            "runner_up_evidence": _candidate_evidence(runner),
            "leader_runner_gap_us": gap,
            "leader_runner_relative_gap": relative,
            "leader_runner_bands_overlap": group["leader_runner_bands_overlap"],
            "resolution_floor": group["quantum"],
            "registered_resolution": registered_facts,
            "incumbent": incumbent,
            "incumbent_unknown_reason": incumbent_unknown,
            "routing_verdict": routing,
        })

    verdict = "VOID" if reasons else "ADJUDICATED"
    return {
        "schema": "quactlize-box-adjudication-v1",
        "kind": "gemv",
        "verdict": verdict,
        "reasons": reasons,
        "policy": _policy_identity(loaded),
        "analysis_complete": bool(analysed.get("complete")) and not reasons,
        "pruning_accounting": {
            "base_census": gemv_policy["base_census"],
            "full_manifest": manifest.get("counts"),
        },
        "groups": groups_out,
    }


_PROVENANCE_FIELDS = (
    "schema", "root_sha", "root_status", "submodule_status", "actlize_sha", "binary_sha256",
    "device_model", "pci_identity", "driver_version", "sdk_compiler_identity",
    "groups", "run_identity_sha256", "argv", "commands", "runner_exit_status",
    "protocol_sample_count",
)


def _read_provenance(bundle: pathlib.Path) -> tuple[dict[str, Any], list[str]]:
    path = bundle / "provenance.json"
    try:
        value = json.loads(path.read_text(), object_pairs_hook=_pairs_without_duplicates)
    except (OSError, json.JSONDecodeError, PolicyError) as exc:
        return {}, [f"missing/unreadable provenance.json: {exc}"]
    if not isinstance(value, dict):
        return {}, ["provenance.json root is not an object"]
    errors = []
    extra = set(value) - set(_PROVENANCE_FIELDS)
    missing = set(_PROVENANCE_FIELDS) - set(value)
    if extra or missing:
        errors.append(f"provenance keys differ: missing={sorted(missing)} extra={sorted(extra)}")
    for name in _PROVENANCE_FIELDS:
        item = value.get(name)
        if name in ("runner_exit_status", "protocol_sample_count"):
            good = isinstance(item, int) and not isinstance(item, bool)
        elif name == "argv":
            good = isinstance(item, list) and item and all(isinstance(x, str) for x in item)
        elif name == "commands":
            good = isinstance(item, list) and bool(item)
        else:
            good = isinstance(item, str) and bool(item.strip())
        if not good:
            errors.append(f"provenance field {name} is missing or malformed")
    if isinstance(value.get("binary_sha256"), str) and not re.fullmatch(
            r"[0-9a-f]{64}", value["binary_sha256"]):
        errors.append("provenance binary_sha256 is not a lowercase SHA-256")
    if isinstance(value.get("root_sha"), str) and not re.fullmatch(
            r"[0-9a-f]{40,64}", value["root_sha"]):
        errors.append("provenance root_sha is not a full Git object ID")
    if value.get("runner_exit_status") != 0:
        errors.append("runner_exit_status is nonzero")
    if value.get("schema") != "quactlize-box-run-provenance-v1":
        errors.append("provenance schema differs")
    if (isinstance(value.get("protocol_sample_count"), int) and
            value["protocol_sample_count"] <= 0):
        errors.append("protocol_sample_count is not positive")
    commands = value.get("commands")
    if isinstance(commands, list):
        for index, item in enumerate(commands):
            if (not isinstance(item, dict) or set(item) != {"role", "argv", "exit_status"} or
                    not isinstance(item.get("role"), str) or not item["role"] or
                    not isinstance(item.get("argv"), list) or not item["argv"] or
                    not all(isinstance(word, str) for word in item["argv"]) or
                    isinstance(item.get("exit_status"), bool) or
                    not isinstance(item.get("exit_status"), int)):
                errors.append(f"provenance commands[{index}] is malformed")
    if value.get("root_status") != "clean":
        errors.append("provenance root_status is not exactly 'clean'")
    submodules = value.get("submodule_status")
    if isinstance(submodules, str):
        dirty = [line for line in submodules.splitlines()
                 if line and line[0] in "+-U"]
        if dirty:
            errors.append("submodule_status records a dirty/uninitialized/conflicted checkout")
        actlize = value.get("actlize_sha")
        act_lines = [line for line in submodules.splitlines()
                     if "third_party/actlize" in line]
        if (isinstance(actlize, str) and
                (len(act_lines) != 1 or act_lines[0].lstrip().split(" ", 1)[0] != actlize)):
            errors.append("actlize_sha differs from recursive submodule status")
    return value, errors


_RUN_IDENTITY_FIELDS = {
    "schema", "root_sha", "submodule_status", "actlize_sha", "binary_sha256",
    "device_model", "pci_identity", "driver_version", "sdk_compiler_identity",
    "protocol_sample_count", "groups", "identity_sha256",
}


def _crosscheck_run_identity(bundle: pathlib.Path, provenance: dict[str, Any],
                             expected_groups: str) -> list[str]:
    path = bundle / "run-identity.json"
    try:
        identity = json.loads(path.read_text(), object_pairs_hook=_pairs_without_duplicates)
    except (OSError, json.JSONDecodeError, PolicyError) as exc:
        return [f"missing/unreadable run-identity.json: {exc}"]
    if not isinstance(identity, dict) or set(identity) != _RUN_IDENTITY_FIELDS:
        return ["run-identity.json does not have the exact v1 schema"]
    errors: list[str] = []
    if identity.get("schema") != "quactlize-box-run-identity-v1":
        errors.append("run-identity.json schema differs")
    for field in (
            "root_sha", "submodule_status", "actlize_sha", "binary_sha256",
            "device_model", "pci_identity", "driver_version", "sdk_compiler_identity",
            "groups", "identity_sha256"):
        value = identity.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"run-identity field {field} is missing or malformed")
    sample_count = identity.get("protocol_sample_count")
    if (not isinstance(sample_count, int) or isinstance(sample_count, bool) or
            sample_count <= 0):
        errors.append("run-identity protocol_sample_count is missing or malformed")
    if (isinstance(identity.get("root_sha"), str) and
            not re.fullmatch(r"[0-9a-f]{40,64}", identity["root_sha"])):
        errors.append("run-identity root_sha is not a full Git object ID")
    if (isinstance(identity.get("actlize_sha"), str) and
            not re.fullmatch(r"[0-9a-f]{40,64}", identity["actlize_sha"])):
        errors.append("run-identity actlize_sha is not a full Git object ID")
    if (isinstance(identity.get("binary_sha256"), str) and
            not re.fullmatch(r"[0-9a-f]{64}", identity["binary_sha256"])):
        errors.append("run-identity binary_sha256 is not a lowercase SHA-256")
    for field in ("device_model", "pci_identity", "driver_version", "sdk_compiler_identity"):
        value = identity.get(field)
        if isinstance(value, str) and value.strip().lower() in {
                "unknown", "unset", "n/a", "na", "none"}:
            errors.append(f"run-identity {field} is not a measured explicit identity")
    digest = identity.get("identity_sha256")
    payload = {key: value for key, value in identity.items() if key != "identity_sha256"}
    expected_digest = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    if not isinstance(digest, str) or digest != expected_digest:
        errors.append("run-identity.json digest differs from its exact payload")
    if identity.get("groups") != expected_groups:
        errors.append(
            f"run-identity groups={identity.get('groups')!r} differs from {expected_groups!r}")
    for field in (
            "root_sha", "submodule_status", "actlize_sha", "binary_sha256",
            "device_model", "pci_identity", "driver_version", "sdk_compiler_identity",
            "protocol_sample_count", "groups"):
        if identity.get(field) != provenance.get(field):
            errors.append(f"run-identity {field} differs from provenance")
    if digest != provenance.get("run_identity_sha256"):
        errors.append("run-identity digest differs from provenance run_identity_sha256")
    return errors


def _commands_by_role(provenance: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for item in provenance.get("commands", []):
        if isinstance(item, dict) and isinstance(item.get("role"), str):
            out.setdefault(item["role"], []).append(item)
    return out


def _crosscheck_provenance_files(bundle: pathlib.Path,
                                 provenance: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    command_path = bundle / "commands.jsonl"
    try:
        commands = [json.loads(line, object_pairs_hook=_pairs_without_duplicates)
                    for line in command_path.read_text().splitlines() if line]
        if commands != provenance.get("commands"):
            errors.append("commands.jsonl differs from embedded provenance commands")
    except (OSError, json.JSONDecodeError, PolicyError) as exc:
        errors.append(f"missing/unreadable commands.jsonl: {exc}")
    try:
        submodules = (bundle / "submodule-status.txt").read_text().rstrip("\n")
        if submodules != provenance.get("submodule_status"):
            errors.append("submodule-status.txt differs from provenance")
    except OSError as exc:
        errors.append(f"missing/unreadable submodule-status.txt: {exc}")
    return errors


def _argv_has_exact_option(argv: list[str], name: str, value: Any) -> bool:
    matches = [word for word in argv if word.startswith(name + "=")]
    return matches == [f"{name}={value}"]


def _read_base_census(path: pathlib.Path, wanted: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        value = json.loads(path.read_text(), object_pairs_hook=_pairs_without_duplicates)
    except (OSError, json.JSONDecodeError, PolicyError) as exc:
        return {}, [f"missing/unreadable base-census.json: {exc}"]
    try:
        value = _expect_keys(
            value, ["schema", "total", "legal", "pruned", "prune_reasons"],
            "base-census.json")
        if value["schema"] != "quactlize-gemv-base-census-v1":
            errors.append("base-census.json schema differs")
        observed = {key: value[key] for key in ("total", "legal", "pruned", "prune_reasons")}
        _validate_census(observed, "base-census.json", jobs=False)
        if observed != wanted:
            errors.append("base-census.json differs from preregistration")
    except PolicyError as exc:
        errors.append(str(exc))
    return value, errors


def _one_match(text: str, pattern: str, noun: str) -> tuple[re.Match[str] | None, list[str]]:
    found = list(re.finditer(pattern, text, re.MULTILINE))
    if len(found) != 1:
        return None, [f"expected exactly one {noun}, found {len(found)}"]
    return found[0], []


def _dense_cell_from_log(path: pathlib.Path, bpc: int, policy: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        text = path.read_text()
    except OSError as exc:
        return {"warp_k": 4, "blocks_per_cu": bpc, "status": "MALFORMED"}, [str(exc)]
    dec, found = _one_match(
        text,
        r"^  \[dense marlin decomposition\] real_cu=(\d+) occupancy_api=(\d+) "
        r"blocks_per_cu=(\d+) Q=(\d+) Kt=(\d+) G=(\d+) I=(\d+) active=(\d+) "
        r"idle=(\d+) handoffs=(\d+) max_peers=(\d+) workspace=(\d+)$",
        "Marlin decomposition")
    errors += found
    if dec is not None and int(dec.group(3)) != bpc:
        errors.append(f"filename B={bpc} disagrees with log B={dec.group(3)}")
    if dec is not None:
        expected_dec = policy["decomposition"]
        if int(dec.group(4)) != expected_dec["output_tiles"]:
            errors.append("decomposition Q differs from preregistered problem")
        if int(dec.group(5)) != expected_dec["k_tiles"]:
            errors.append("decomposition Kt differs from preregistered problem")
        real_cu = int(dec.group(1))
        occupancy_api = int(dec.group(2))
        if real_cu <= 0 or occupancy_api <= 0:
            errors.append("decomposition real_cu/occupancy_api must be positive")
        elif bpc > occupancy_api:
            errors.append(
                f"decomposition B={bpc} exceeds occupancy_api={occupancy_api}")
        else:
            expected = _marlin_dense_decomposition(
                expected_dec["output_tiles"], expected_dec["k_tiles"], real_cu, bpc)
            names = (
                ("grid_ctas", 6), ("stripe_iters", 7), ("active_ctas", 8),
                ("idle_ctas", 9), ("handoffs", 10), ("max_peers", 11),
            )
            for name, group in names:
                observed = int(dec.group(group))
                if observed != expected[name]:
                    errors.append(
                        f"decomposition {name}={observed} differs from independent "
                        f"flat-stripe oracle {expected[name]}")
            workspace = int(dec.group(12))
            split = expected["handoffs"] > 0
            if split != (workspace > 0):
                errors.append(
                    "workspace zero/nonzero state disagrees with whether any tile is split")
    timing, found = _one_match(
        text,
        r"^  \[dense kernel-span-upper\] n=(\d+) median=([0-9.]+) us mean=([0-9.]+) us "
        r"min=([0-9.]+) us max=([0-9.]+) us .*distinct-event-pairs=(\d+) ",
        "independent-event timing")
    errors += found
    disposition = re.findall(r"^  Disposition: (Passed|Failed)(?:\b| \()", text, re.MULTILINE)
    repeat_ids = re.findall(
        r"^  \[dense marlin lock fingerprint\] repeat=([1-8])/8 raw_bitdiff=0 .*"
        r"stable=1 same-workspace=1 external-lock-reset=0$", text, re.MULTILINE)
    checks = {
        "correctness": disposition == ["Passed"],
        "shipping_artifact_roundtrip": bool(re.search(
            r"^  \[dense marlin aligned artifact\].*roundtrip_bad=0/16777216$", text, re.MULTILINE)),
        "exact_fixture": bool(re.search(
            r"^  \[streamk fixture exactness\] fixture=a0-exact shape=1x4096x4096 .*"
            r"ORDER-INDEPENDENT\+FP16-EXACT$", text, re.MULTILINE)),
        "lock_fingerprints_8_stable": sorted(map(int, repeat_ids)) == list(range(1, 9)) and
            bool(re.search(r"^  \[dense marlin lock protocol\].*repeats=8 stable=1 "
                           r"all-bitexact=1 .*external-lock-reset=0$", text, re.MULTILINE)),
    }
    if timing is None:
        cell = {"warp_k": 4, "blocks_per_cu": bpc, "status": "MALFORMED"}
    else:
        n, median, _, low, high, pairs = timing.groups()
        cell = {
            "warp_k": 4, "blocks_per_cu": bpc, "status": "RUN",
            "sample_count": int(n), "median_us": median, "min_us": low, "max_us": high,
            "prerequisites": checks,
        }
        if dec is not None:
            cell["decomposition"] = {
                "real_cu": int(dec.group(1)), "occupancy_api": int(dec.group(2)),
                "blocks_per_cu": int(dec.group(3)), "q": int(dec.group(4)),
                "k_tiles": int(dec.group(5)), "grid_ctas": int(dec.group(6)),
                "stripe_iters": int(dec.group(7)), "active_ctas": int(dec.group(8)),
                "idle_ctas": int(dec.group(9)), "handoffs": int(dec.group(10)),
                "max_peers": int(dec.group(11)), "workspace": int(dec.group(12)),
            }
        if n != pairs:
            errors.append("sample count differs from distinct event-pair count")
    return cell, errors


def adjudicate_dense_bundle(loaded: LoadedPolicy, bundle_path: pathlib.Path | str) -> dict[str, Any]:
    bundle = pathlib.Path(bundle_path)
    provenance, admission_errors = _read_provenance(bundle)
    admission_errors.extend(_crosscheck_provenance_files(bundle, provenance))
    admission_errors.extend(_crosscheck_run_identity(
        bundle, provenance, "not-applicable"))
    sample_count = loaded.value["dense"]["sample_count"]
    if provenance.get("protocol_sample_count") != sample_count:
        admission_errors.append("dense provenance sample count differs from preregistration")
    runner_argv = provenance.get("argv")
    if (not isinstance(runner_argv, list) or len(runner_argv) != 1 or
            not runner_argv[0].endswith("tools/run_dense_marlin_wk4_box.sh")):
        admission_errors.append("dense top-level runner argv is not the frozen entry point")
    commands = _commands_by_role(provenance)
    for role in ("wk1-static-target", "wk1-committed-production-delivery", "device-build"):
        rows = commands.get(role, [])
        if len(rows) != 1 or rows[0].get("exit_status") != 0:
            admission_errors.append(f"command journal lacks one successful {role}")
    committed_rows = commands.get("wk1-committed-production-delivery", [])
    if len(committed_rows) == 1:
        committed_argv = committed_rows[0].get("argv", [])
        expected_object = (
            f"{provenance.get('root_sha')}:"
            "dev/fold_derivation/l143_wk4_production_delivery.expected.txt")
        if (len(committed_argv) != 5 or committed_argv[0] != "git" or
                committed_argv[1] != "-C" or committed_argv[3] != "show" or
                committed_argv[4] != expected_object):
            admission_errors.append(
                "WK1 committed production-delivery command is not the exact result-SHA git show")
    runner = bundle / "runner.log"
    try:
        runner_text = runner.read_text()
    except OSError as exc:
        runner_text = ""
        admission_errors.append(f"missing/unreadable runner.log: {exc}")
    for name in ("build.log", "illegal-bpc.log"):
        if not (bundle / name).is_file():
            admission_errors.append(f"required raw artifact {name} is absent")
    root_hits = re.findall(r"^\[marlin-wk4\] root-sha=([0-9a-f]+)$", runner_text, re.MULTILINE)
    act_hits = re.findall(r"^\[marlin-wk4\] actlize-sha=([0-9a-f]+)$", runner_text, re.MULTILINE)
    binary_hits = re.findall(r"^\[marlin-wk4\] binary-sha256=([0-9a-f]{64})$",
                             runner_text, re.MULTILINE)
    if len(root_hits) != 1 or root_hits[0] != provenance.get("root_sha"):
        admission_errors.append("runner root SHA is missing/ambiguous or differs from provenance")
    if len(act_hits) != 1 or act_hits[0] != provenance.get("actlize_sha"):
        admission_errors.append("runner actlize SHA is missing/ambiguous or differs from provenance")
    if len(binary_hits) != 1 or binary_hits[0] != provenance.get("binary_sha256"):
        admission_errors.append("runner binary SHA is missing/ambiguous or differs from provenance")
    runner_terminator = (
        "[marlin-wk4] PASS: classic-aligned WK4 consumer built on shipping bytes; "
        "supported B points passed exact output + 8-launch locks; over-cap B stayed NOT RUN")
    if runner_text.splitlines().count(runner_terminator) != 1:
        admission_errors.append("exact unique runner PASS terminator is absent")

    # WK1 is a locally executable oracle whose exact expected output is committed
    # at the measured result SHA.  The box runner retrieves that immutable file;
    # it does not pretend that the host-only CuTe oracle was freshly compiled or
    # executed on the PPU box.  The evidence is still output, not a caller-provided
    # boolean, and the command journal binds it to the result SHA above.
    wk1_path = bundle / "wk1-admission.log"
    try:
        wk1_text = wk1_path.read_text()
    except OSError as exc:
        wk1_text = ""
        admission_errors.append(f"missing/unreadable wk1-admission.log: {exc}")
    committed_marker, marker_errors = _one_match(
        wk1_text,
        (r"^\[marlin-wk4\] wk1-evidence=committed-local-oracle source-sha=" +
         re.escape(str(provenance.get("root_sha"))) +
         r" path=dev/fold_derivation/l143_wk4_production_delivery\.expected\.txt "
         r"fresh-box-execution=0$"),
        "explicit committed-not-box L143 evidence marker")
    wk1_map, map_errors = _one_match(
        wk1_text,
        r"^L143 WK1 shipping map-diff=(\d+) byte-diff=(\d+) result=BIT-IDENTICAL$",
        "L143 WK1 byte-map result")
    wk1_direct, direct_errors = _one_match(
        wk1_text,
        r"^L143 direct-pair pairs=(\d+)/(\d+) codes=16384/16384 "
        r"destinations=8192/8192 bad-pairs=0 formula-mismatch=0 bad-fragments=0 "
        r"map-diff=0 shipping-hash=[0-9a-f]{16}$",
        "L143 direct-pair exhaustive result")
    wk1_final, final_errors = _one_match(
        wk1_text,
        r"^L143 shipping-pair-scatter=EXACT artifact-order=RED compact-order=RED "
        r"first32=RED wrong-pair=RED source-swap=RED WK1-BYTES=UNCHANGED result=PASS$",
        "L143 final positive/negative result")
    structure, structure_errors = _one_match(
        wk1_text,
        r"^\[dense-marlin-wk4\] PASS: isolated 1Mx2Nx4K "
        r"type/shipping-artifact/CLI; historical target unchanged; "
        r"thirteen structural plants rejected$",
        "structural WK1/WK4 source gate")
    admission_errors += marker_errors + map_errors + direct_errors + final_errors + structure_errors
    wk1_policy = loaded.value["dense"]["wk1_admission"]
    wk1_total = int(wk1_direct.group(2)) if wk1_direct is not None else None
    wk1_ok = bool(
        wk1_map is not None and int(wk1_map.group(1)) == wk1_policy["byte_map_diff"] and
        int(wk1_map.group(2)) == wk1_policy["byte_map_diff"] and
        wk1_direct is not None and int(wk1_direct.group(1)) == wk1_total ==
        wk1_policy["byte_map_total"] and wk1_final is not None and structure is not None and
        committed_marker is not None)
    if not wk1_ok:
        admission_errors.append("WK1 executable structural/byte-map admission evidence did not close")

    raw_cells: list[dict[str, Any]] = [
        {"warp_k": 1, "blocks_per_cu": 1, "status": "ADMISSION_CONTROL"},
        *({"warp_k": 1, "blocks_per_cu": b, "status": "NOT_IN_QUEUED_RUN"}
          for b in (2, 4, 6)),
        *({"warp_k": 2, "blocks_per_cu": b,
           "status": "PREREGISTERED_COMPILE_NEGATIVE_NOT_RERUN"}
          for b in (1, 2, 4, 6)),
    ]
    parse_errors: list[str] = []
    unregistered: list[dict[str, Any]] = []
    registered_command_roles = {
        "wk1-static-target", "wk1-committed-production-delivery", "device-build",
        "dense-wk4-illegal-bpc",
        *(f"dense-wk4-bpc{bpc}" for bpc in (1, 2, 4, 6)),
    }
    unregistered.extend(
        {"command_role": role, "argv": row.get("argv"),
         "exit_status": row.get("exit_status"),
         "reason": "command role is outside the preregistered dense bundle contract"}
        for role, rows in sorted(commands.items()) if role not in registered_command_roles
        for row in rows
    )
    registered_keys = {(c["warp_k"], c["blocks_per_cu"])
                       for c in loaded.value["dense"]["cells"]}
    # B=1 owns the exact instantiated-kernel occupancy cap.  Every registered
    # WK4 rung must then have exactly one timing log or one exact NOT-RUN record.
    b1, b1_errors = _dense_cell_from_log(bundle / "bpc1.log", 1, loaded.value["dense"])
    parse_errors.extend(f"bpc1.log: {error}" for error in b1_errors)
    raw_cells.append(b1)
    try:
        b1_text = (bundle / "bpc1.log").read_text()
    except OSError:
        b1_text = ""
    caps = re.findall(
        r"^  \[dense marlin decomposition\] real_cu=\d+ occupancy_api=(\d+) "
        r"blocks_per_cu=1 ", b1_text, re.MULTILINE)
    cap = int(caps[0]) if len(caps) == 1 else None
    if cap is None or cap <= 0:
        parse_errors.append("bpc1.log: exact occupancy_api cap is missing or ambiguous")
    b1_decomposition = b1.get("decomposition", {})
    b1_real_cu = b1_decomposition.get("real_cu")
    b1_workspace = b1_decomposition.get("workspace")

    problem = loaded.value["dense"]["problem"]
    common_options = {
        "--m": problem["m"], "--n": problem["n"], "--k": problem["k"],
        "--l": problem["l"], "--g": problem["group_size"],
        "--iterations": sample_count,
    }
    invocation = loaded.value["dense"]["invocation"]
    common_options.update({
        "--mode": invocation["options"]["mode"],
        "--alpha": invocation["options"]["alpha"],
        "--beta": invocation["options"]["beta"],
    })
    for bpc in (1, 2, 4, 6):
        role = f"dense-wk4-bpc{bpc}"
        rows = commands.get(role, [])
        should_run = cap is not None and bpc <= cap
        if should_run:
            if len(rows) != 1 or rows[0].get("exit_status") != 0:
                admission_errors.append(f"command journal lacks one successful {role}")
                continue
            argv = rows[0]["argv"]
            for flag in invocation["flags"]:
                if argv.count(flag) != 1:
                    admission_errors.append(f"{role} does not carry exactly one {flag}")
            for option, value in common_options.items():
                if not _argv_has_exact_option(argv, option, value):
                    admission_errors.append(f"{role} does not carry exact {option}={value}")
            override = [word for word in argv if word.startswith("--marlin-blocks-per-cu=")]
            if bpc == 1 and override:
                admission_errors.append("WK4/B1 used an explicit override, so it is not the default path")
            if bpc != 1 and override != [f"--marlin-blocks-per-cu={bpc}"]:
                admission_errors.append(f"{role} lacks its exact blocks-per-CU override")
        elif rows:
            admission_errors.append(f"over-cap {role} was launched")

    illegal_rows = commands.get("dense-wk4-illegal-bpc", [])
    if cap is not None:
        expected_flag = f"--marlin-blocks-per-cu={cap + 1}"
        if (len(illegal_rows) != 1 or illegal_rows[0].get("exit_status") == 0 or
                illegal_rows[0].get("argv", []).count(expected_flag) != 1):
            admission_errors.append("illegal-B command is not the exact nonzero cap+1 launch")

    seen_wk4 = {1}
    for path in sorted(bundle.glob("bpc*.log")):
        match = re.fullmatch(r"bpc(\d+)\.log", path.name)
        if not match:
            unregistered.append({"artifact": path.name, "reason": "unparsed B log filename"})
            continue
        bpc = int(match.group(1))
        if bpc == 1 and path.name == "bpc1.log":
            continue
        if (4, bpc) not in registered_keys:
            unregistered.append({"artifact": path.name, "warp_k": 4, "blocks_per_cu": bpc})
            continue
        if bpc in seen_wk4:
            parse_errors.append(f"duplicate dense cell WK4/B{bpc}")
            continue
        seen_wk4.add(bpc)
        cell, errors = _dense_cell_from_log(path, bpc, loaded.value["dense"])
        raw_cells.append(cell)
        parse_errors.extend(f"{path.name}: {error}" for error in errors)
        observed_cap = cell.get("decomposition", {}).get("occupancy_api")
        if cap is not None and observed_cap != cap:
            parse_errors.append(
                f"{path.name}: occupancy_api={observed_cap} differs from B1 cap={cap}")
        observed_real_cu = cell.get("decomposition", {}).get("real_cu")
        if b1_real_cu is not None and observed_real_cu != b1_real_cu:
            parse_errors.append(
                f"{path.name}: real_cu={observed_real_cu} differs from "
                f"B1 real_cu={b1_real_cu}")
        observed_workspace = cell.get("decomposition", {}).get("workspace")
        if b1_workspace is not None and observed_workspace != b1_workspace:
            parse_errors.append(
                f"{path.name}: workspace={observed_workspace} differs from "
                f"B1 workspace={b1_workspace}")
    for path in sorted(bundle.glob("bpc*.not-run")):
        match = re.fullmatch(r"bpc(\d+)\.not-run", path.name)
        if not match:
            unregistered.append({"artifact": path.name, "reason": "unparsed NOT RUN filename"})
            continue
        bpc = int(match.group(1))
        if (4, bpc) in registered_keys:
            if bpc in seen_wk4:
                parse_errors.append(f"duplicate dense cell WK4/B{bpc}")
                continue
            seen_wk4.add(bpc)
            try:
                not_run = path.read_text()
            except OSError as exc:
                parse_errors.append(f"{path.name}: {exc}")
                not_run = ""
            expected = (None if cap is None else
                        f"[marlin-wk4] NOT RUN: B={bpc} exceeds "
                        f"Gemm::maximum_active_blocks()={cap}\n")
            if expected is None or not_run != expected or bpc <= cap:
                parse_errors.append(
                    f"{path.name}: NOT-RUN evidence is not the exact over-cap rejection")
            raw_cells.append({"warp_k": 4, "blocks_per_cu": bpc, "status": "NOT_RUN"})
        else:
            unregistered.append({"artifact": path.name, "warp_k": 4, "blocks_per_cu": bpc})
    for bpc in (1, 2, 4, 6):
        if bpc not in seen_wk4:
            parse_errors.append(f"WK4/B{bpc}: neither bpc{bpc}.log nor bpc{bpc}.not-run exists")
        elif bpc != 1 and cap is not None:
            has_log = (bundle / f"bpc{bpc}.log").is_file()
            if (bpc <= cap) != has_log:
                parse_errors.append(
                    f"WK4/B{bpc}: log/NOT-RUN choice disagrees with exact cap={cap}")

    try:
        illegal_text = (bundle / "illegal-bpc.log").read_text()
    except OSError:
        illegal_text = ""
    illegal = None if cap is None else cap + 1
    expected_illegal = (None if illegal is None else
                        f"--marlin-blocks-per-cu={illegal} is outside the exact "
                        f"kernel occupancy range 1..{cap}")
    if expected_illegal is None or illegal_text.count(expected_illegal) != 1:
        parse_errors.append("illegal-bpc.log does not contain the exact cap+1 runtime rejection")
    observation = {
        "schema": DENSE_SCHEMA,
        "identity_valid": not admission_errors,
        "wk1_admission": {
            "structural_identity": structure is not None,
            "byte_map_total": wk1_total,
            "byte_map_diff": (int(wk1_map.group(2)) if wk1_map is not None else None),
            "device_control_present": False,
        },
        "cells": raw_cells,
    }
    result = adjudicate_dense(loaded, observation)
    if parse_errors:
        result["reasons"].extend(parse_errors)
        result["verdict"] = "VOID"
    result["reasons"] = admission_errors + result["reasons"]
    if admission_errors:
        result["verdict"] = "VOID"
    result["cell_results"] = result.pop("cells")
    result["registered_verdict"] = result["verdict"]
    known = {
        "provenance.json", "run-identity.json", "commands.jsonl",
        "submodule-status.txt", "runner.log",
        "build.log", "illegal-bpc.log", "wk1-admission.log", "l143",
        *(f"bpc{b}.log" for b in (1, 2, 4, 6)),
        *(f"bpc{b}.not-run" for b in (2, 4, 6)),
    }
    unregistered.extend(
        {"artifact": path.name, "reason": "not covered by preregistered bundle contract"}
        for path in sorted(bundle.iterdir(), key=lambda p: p.name)
        if path.name not in known and not re.fullmatch(r"bpc\d+\.(?:log|not-run)", path.name)
    )
    result["unregistered_observations"] = unregistered
    result["provenance"] = provenance
    return result


def adjudicate_gemv_bundle(loaded: LoadedPolicy, bundle_path: pathlib.Path | str) -> dict[str, Any]:
    bundle = pathlib.Path(bundle_path)
    provenance, admission_errors = _read_provenance(bundle)
    admission_errors.extend(_crosscheck_provenance_files(bundle, provenance))
    admission_errors.extend(_crosscheck_run_identity(bundle, provenance, "all"))
    samples = loaded.value["gemv"]["sample_count"]
    if provenance.get("protocol_sample_count") != samples:
        admission_errors.append("GEMV provenance sample count differs from preregistration")
    runner_argv = provenance.get("argv")
    if (not isinstance(runner_argv, list) or len(runner_argv) != 1 or
            not runner_argv[0].endswith("tools/run_gemv_sweep_box.sh")):
        admission_errors.append("GEMV top-level runner argv is not the frozen entry point")
    commands = _commands_by_role(provenance)
    for role in ("device-build", "base-tactic-census", "manifest", "dry-run-audit",
                 "measured-sweep", "analyse", "analyse-completeness"):
        rows = commands.get(role, [])
        if not rows or rows[-1].get("exit_status") != 0:
            admission_errors.append(f"command journal lacks a final successful {role}")
    registered_command_roles = {
        "device-build", "base-tactic-census", "manifest", "dry-run-audit",
        "measured-sweep", "analyse", "analyse-completeness",
    }
    unregistered_commands = [
        {"command_role": role, "argv": row.get("argv"),
         "exit_status": row.get("exit_status"),
         "reason": "command role is outside the preregistered GEMV bundle contract"}
        for role, rows in sorted(commands.items()) if role not in registered_command_roles
        for row in rows
    ]
    required = ("manifest.json", "raw.jsonl", "progress.jsonl", "result.json", "run.log",
                "base-census.json", "base-census-authority.log", "build.log", "runner.log",
                "commands.jsonl", "submodule-status.txt", "run-identity.json",
                "pending.audit.jsonl", "pending.summary.jsonl")
    for name in required:
        if not (bundle / name).is_file():
            admission_errors.append(f"required raw artifact {name} is absent")
    if not (bundle / "logs").is_dir():
        admission_errors.append("required per-attempt logs/ directory is absent")
    base_census, census_errors = _read_base_census(
        bundle / "base-census.json", loaded.value["gemv"]["base_census"])
    admission_errors.extend(census_errors)
    try:
        authority = (bundle / "base-census-authority.log").read_text()
        wanted = loaded.value["gemv"]["base_census"]
        census_rows: dict[str, int] = {}
        exclusion_rows: dict[str, int] = {}
        result_rows: list[str] = []
        duplicate_rows: list[str] = []
        for line in authority.splitlines():
            fields = line.split(",")
            if len(fields) == 3 and fields[0] in ("CENSUS", "EXCLUSION"):
                target = census_rows if fields[0] == "CENSUS" else exclusion_rows
                if fields[1] in target:
                    duplicate_rows.append(f"{fields[0]},{fields[1]}")
                    continue
                try:
                    target[fields[1]] = int(fields[2])
                except ValueError:
                    duplicate_rows.append(f"malformed {line}")
            elif len(fields) == 2 and fields[0] == "RESULT":
                result_rows.append(fields[1])
        expected_census = {
            "total": wanted["total"], "legal": wanted["legal"],
            "rejected": wanted["pruned"],
        }
        if (duplicate_rows or census_rows != expected_census or
                exclusion_rows != wanted["prune_reasons"] or result_rows != ["PASS"]):
            admission_errors.append(
                "base census authority rows differ from the exact registered histogram: "
                f"duplicates={duplicate_rows} census={census_rows} "
                f"exclusions={exclusion_rows} result={result_rows}")
    except OSError as exc:
        admission_errors.append(f"cannot read base-census-authority.log: {exc}")
    result = adjudicate_gemv(loaded, bundle / "manifest.json", [bundle / "raw.jsonl"])
    # Raw build identity is an independent cross-check on root/binary/protocol.
    try:
        sweep = _load_sweep_module()
        raw = sweep.load_raw([bundle / "raw.jsonl"])
        expected_run_id = (
            f"gemv-{str(provenance.get('root_sha'))[:12]}-"
            f"{str(provenance.get('binary_sha256'))[:16]}-samples{samples}")
        if set(raw.runs) != {expected_run_id}:
            admission_errors.append(
                f"raw run identity {sorted(raw.runs)!r} differs from {expected_run_id!r}")
        builds = {run.build for run in raw.runs.values()}
        expected = (f"{provenance.get('root_sha')}/bin-sha256:"
                    f"{provenance.get('binary_sha256')}/protocol:samples{samples}")
        if builds != {expected}:
            admission_errors.append(f"raw build identity {sorted(builds)!r} differs from {expected!r}")
    except Exception as exc:
        admission_errors.append(f"cannot cross-check raw build identity: {exc}")
    if admission_errors:
        result["verdict"] = "VOID"
        result["analysis_complete"] = False
        result["reasons"] = admission_errors + result["reasons"]
    result["cell_results"] = result.pop("groups")
    result["registered_verdict"] = result["verdict"]
    # run.lock is an expected operational inode, not evidence and not an
    # unregistered observation.  Its contents never enter a verdict.
    registered_artifacts = set(required) | {"provenance.json", "logs", "run.lock"}
    result["unregistered_observations"] = unregistered_commands + [
        {"artifact": path.name, "reason": "not covered by preregistered bundle contract"}
        for path in sorted(bundle.iterdir(), key=lambda p: p.name)
        if path.name not in registered_artifacts
    ] if bundle.is_dir() else []
    result["pruning_accounting"]["observed_base_census"] = base_census
    result["provenance"] = provenance
    return result


def _write_result(result: dict[str, Any], output: str) -> None:
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output == "-":
        sys.stdout.write(text)
    else:
        pathlib.Path(output).write_text(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(ROOT))
    parser.add_argument("--policy")
    parser.add_argument("--fixture-mode", action="store_true",
                        help="allow an explicit policy/normalized observation in CI fixtures only")
    parser.add_argument("--output", default="-")
    sub = parser.add_subparsers(dest="kind", required=True)
    dense = sub.add_parser("dense-bundle")
    dense.add_argument("bundle")
    gemv = sub.add_parser("gemv-bundle")
    gemv.add_argument("bundle")
    fixture_dense = sub.add_parser("dense-fixture")
    fixture_dense.add_argument("observation")
    args = parser.parse_args(argv)
    if args.policy and not args.fixture_mode:
        parser.error("--policy is allowed only with --fixture-mode")
    if args.kind == "dense-fixture" and not args.fixture_mode:
        parser.error("dense-fixture is allowed only with --fixture-mode")
    try:
        if args.fixture_mode:
            policy = load_policy(args.policy or DEFAULT_POLICY)
        else:
            provenance, errors = _read_provenance(pathlib.Path(args.bundle))
            if errors or not provenance.get("root_sha"):
                result = {
                    "schema": "quactlize-box-adjudication-v1", "kind": args.kind,
                    "verdict": "VOID", "registered_verdict": "VOID",
                    "reasons": errors or ["result root SHA is absent"],
                    "cell_results": [], "unregistered_observations": [],
                }
                _write_result(result, args.output)
                return 1
            code_errors = publication_code_errors(
                provenance["root_sha"], args.repo, ROOT)
            if code_errors:
                result = {
                    "schema": "quactlize-box-adjudication-v1", "kind": args.kind,
                    "verdict": "VOID", "registered_verdict": "VOID",
                    "reasons": code_errors,
                    "cell_results": [], "unregistered_observations": [],
                }
                _write_result(result, args.output)
                return 1
            policy = load_policy_from_git(provenance["root_sha"], args.repo)
    except (OSError, PolicyError) as exc:
        result = {
            "schema": "quactlize-box-adjudication-v1", "kind": args.kind,
            "verdict": "VOID", "registered_verdict": "VOID",
            "reasons": [f"adjudication policy VOID: {exc}"],
            "cell_results": [], "unregistered_observations": [],
        }
        _write_result(result, args.output)
        return 1
    if args.kind == "dense-fixture":
        try:
            observation = json.loads(pathlib.Path(args.observation).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            result = {"schema": "quactlize-box-adjudication-v1", "kind": "dense",
                      "verdict": "VOID", "reasons": [f"cannot read observation: {exc}"],
                      "policy": _policy_identity(policy), "cells": []}
        else:
            result = adjudicate_dense(policy, observation)
            result["cell_results"] = result.pop("cells")
            result["registered_verdict"] = result["verdict"]
            result["unregistered_observations"] = []
    elif args.kind == "dense-bundle":
        result = adjudicate_dense_bundle(policy, args.bundle)
    else:
        result = adjudicate_gemv_bundle(policy, args.bundle)
    _write_result(result, args.output)
    return 1 if result["verdict"] == "VOID" else 0


if __name__ == "__main__":
    raise SystemExit(main())
