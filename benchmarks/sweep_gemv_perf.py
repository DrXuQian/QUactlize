#!/usr/bin/env python3
"""Bounded driver and conservative analyser for the low-bit GEMV tactic sweep.

This program deliberately consumes *raw device-event samples*.  A C++ benchmark
that prints one averaged time is not an input to this analyser: averaging hides
the timer lattice, and the lattice is part of the verdict.

There are two versioned JSON contracts.

Raw JSONL (``gemv-sweep-raw-v1``)::

  {"rec":"run", "schema":"gemv-sweep-raw-v1", "run_id":"r", ...,
   "build":"...", "space_id":"...", "partial_space":false}
  {"rec":"attempt", ..., "shape_id":"S068", "shape":{...},
   "format":"int4", "config_id":"...", "config":{...},
   "attempt_id":"0", "pass":0, "expected_samples":20}
  {"rec":"sample",  ..., same complete identity ..., "attempt_id":"0",
   "pass":0, "launch_index":0, "event_ms_bits":1011438492,
   "event_us":12.288}
  {"rec":"excluded", ..., same complete identity ..., "attempt_id":"0", "pass":0,
   "why":"..."}

The complete ``shape`` and ``config`` objects are repeated intentionally.  IDs
are conveniences, not identities: reusing an ID for different JSON, or giving
the same JSON two IDs, is a hard error.  This prevents a newly added tactic axis
from silently disappearing from the analyser's grouping key.

An attempt resolves to exactly one exclusion or to samples numbered
``0..expected_samples-1``.  A sample without its attempt, a hole, a duplicate,
or samples plus an exclusion is incomplete and cannot support a winner.

Driver manifest (``gemv-sweep-manifest-v1``)::

  {"schema":"gemv-sweep-manifest-v1", "space_id":"...",
   "partial_space":false, "counts":{"total":42,"legal":30,"pruned":12,
   "prune_reasons":{"...":12}}, "jobs":[
     {"job_id":"S068", "shape_id":"S068", "shape":{...},
      "formats":["int4",...], "argv":["0"], "env":{},
      "expected":[{"format":"int4","config_id":"...","config":{...}}]}
   ]}

``run --dry-run`` emits the exact pending job manifest without launching.  A
real child receives GEMV_SWEEP_JSONL, GEMV_SWEEP_RUN_ID,
GEMV_SWEEP_JOB_ID, and GEMV_SWEEP_ATTEMPT (names are configurable).  Each job
writes a private temporary JSONL; it is validated before being appended to the
durable raw file, while an invalid/failed attempt is still retained for audit.

Examples::

  python3 benchmarks/sweep_gemv_perf.py analyse raw.jsonl --output result.json
  python3 benchmarks/sweep_gemv_perf.py run plan.json --bin ./test_gemv_perf \
      --raw raw.jsonl --progress progress.jsonl --shape-timeout 900 \
      --deadline-seconds 7200 --resume
  python3 benchmarks/sweep_gemv_perf.py run plan.json --dry-run \
      --dry-run-manifest /tmp/gemv-plan.jsonl
  python3 benchmarks/sweep_gemv_perf.py --self-test
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


RAW_SCHEMA = "gemv-sweep-raw-v1"
MANIFEST_SCHEMA = "gemv-sweep-manifest-v1"
PROGRESS_SCHEMA = "gemv-sweep-progress-v1"
RESULT_SCHEMA = "gemv-sweep-result-v1"

# A smaller inferred divisor is more likely decimal noise than a demonstrated
# hardware quantum.  Rejecting a real finer timer is conservative: the result is
# UNKNOWN and therefore UNRESOLVED, never a false winner.  The threshold is a
# CLI input and is recorded in the output.
DEFAULT_MIN_QUANTUM_US = Decimal("0.01")


class ContractError(ValueError):
    pass


def canonical(value: Any) -> str:
    """Canonical JSON used for identity; NaN/Inf are never valid identities."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"identity is not canonical JSON: {exc}") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _nonempty_string(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ContractError(f"{key} must be a non-empty string")
    return value


def _nonnegative_int(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{key} must be a non-negative integer")
    return value


def _identity_object(record: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict) or not value:
        raise ContractError(f"{key} must be a non-empty JSON object")
    canonical(value)  # validates before the object enters a key
    return value


def _event_decimal(record: Mapping[str, Any]) -> Decimal:
    # Accept the spelling event_us for hand-written fixtures, but never an
    # aggregate or an array.  One record is one raw event observation.
    present = [k for k in ("event_us_raw", "event_us") if k in record]
    if len(present) != 1:
        raise ContractError("sample must carry exactly one of event_us_raw/event_us")
    raw = record[present[0]]
    if isinstance(raw, bool) or isinstance(raw, (list, dict)):
        raise ContractError(f"{present[0]} must be one raw scalar")
    try:
        value = Decimal(str(raw))
    except InvalidOperation as exc:
        raise ContractError(f"{present[0]} is not decimal") from exc
    if not value.is_finite() or value <= 0:
        raise ContractError(f"{present[0]} must be finite and > 0")
    bit_fields = [k for k in ("event_ms_bits", "event_ms_f32_bits") if k in record]
    if len(bit_fields) != 1:
        raise ContractError("sample must carry exactly one event_ms_bits/event_ms_f32_bits raw word")
    bits = record[bit_fields[0]]
    if isinstance(bits, bool) or not isinstance(bits, int) or not 0 <= bits < 2**32:
        raise ContractError(f"{bit_fields[0]} must be a uint32")
    event_ms = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not math.isfinite(event_ms) or event_ms <= 0:
        raise ContractError(f"{bit_fields[0]} does not encode a finite positive float")
    # event_us is the human-readable lattice value; the word is the raw source.
    # Decimal formatting may round at 0.001 us, hence half a last display unit.
    if abs(float(value) - event_ms * 1000.0) > 0.001:
        raise ContractError(
            f"event_us {value} disagrees with raw event_ms_bits ({event_ms * 1000.0:.9g} us)")
    return value


def _aliased_nonnegative_int(record: Mapping[str, Any], primary: str, alias: str) -> int:
    present = [k for k in (primary, alias) if k in record]
    if not present:
        raise ContractError(f"missing {primary}")
    values = [_nonnegative_int(record, k) for k in present]
    if len(set(values)) != 1:
        raise ContractError(f"{primary}/{alias} disagree")
    return values[0]


def _attempt_key(record: Mapping[str, Any]) -> tuple[str, int]:
    # The production writer names these attempt_id/pass.  The integer `attempt`
    # alias keeps small planted fixtures readable; aliases may never disagree.
    present = [k for k in ("attempt_id", "attempt") if k in record]
    if not present:
        raise ContractError("missing attempt_id")
    values = []
    for key in present:
        value = record[key]
        if isinstance(value, bool) or not isinstance(value, (str, int)) or value == "":
            raise ContractError(f"{key} must be a non-empty string or integer")
        values.append(canonical(value))
    if len(set(values)) != 1:
        raise ContractError("attempt_id/attempt disagree")
    pass_index = _nonnegative_int(record, "pass") if "pass" in record else 0
    return values[0], pass_index


@dataclasses.dataclass(frozen=True)
class RunInfo:
    run_id: str
    build: str
    space_id: str
    partial_space: bool


@dataclasses.dataclass(frozen=True)
class CandidateIdentity:
    run_id: str
    shape_id: str
    shape_json: str
    fmt: str
    config_id: str
    config_json: str


@dataclasses.dataclass(frozen=True)
class AttemptIdentity:
    candidate: CandidateIdentity
    attempt_id: str
    pass_index: int


@dataclasses.dataclass
class RawData:
    runs: dict[str, RunInfo]
    attempts: dict[AttemptIdentity, dict[str, Any]]
    samples: dict[AttemptIdentity, dict[int, Decimal]]
    exclusions: dict[AttemptIdentity, dict[str, Any]]
    shapes: dict[str, str]
    configs: dict[tuple[str, str], str]
    shape_ids_by_json: dict[str, str]
    config_ids_by_json: dict[tuple[str, str], str]
    complaints: list[str]

    def complete_attempts(self) -> dict[AttemptIdentity, list[Decimal]]:
        out: dict[AttemptIdentity, list[Decimal]] = {}
        for aid, attempt in self.attempts.items():
            expected = attempt["expected_samples"]
            got = self.samples.get(aid, {})
            exclusion = self.exclusions.get(aid)
            if exclusion is not None:
                if got:
                    continue
                out[aid] = []
            elif set(got) == set(range(expected)):
                out[aid] = [got[i] for i in range(expected)]
        return out


def _candidate(record: Mapping[str, Any], data: RawData) -> CandidateIdentity:
    run_id = _nonempty_string(record, "run_id")
    shape_id = _nonempty_string(record, "shape_id")
    config_id = _nonempty_string(record, "config_id")
    fmt = _nonempty_string(record, "format")
    shape_json = canonical(_identity_object(record, "shape"))
    config_json = canonical(_identity_object(record, "config"))

    def bind(by_id: dict[str, str], by_json: dict[str, str], ident: str,
             value_json: str, noun: str) -> None:
        if ident in by_id and by_id[ident] != value_json:
            raise ContractError(f"{noun} identity collision: {ident!r} names two JSON objects")
        if value_json in by_json and by_json[value_json] != ident:
            raise ContractError(
                f"{noun} identity alias: the same JSON is named {by_json[value_json]!r} and {ident!r}")
        by_id[ident] = value_json
        by_json[value_json] = ident

    bind(data.shapes, data.shape_ids_by_json, shape_id, shape_json, "shape")
    # A tactic ID is scoped by format.  The same axes under int4 and int2 are
    # two candidates, while reusing an ID for two JSON objects *within* int4 is
    # the collision this contract must catch.
    scoped_by_id = (fmt, config_id)
    scoped_by_json = (fmt, config_json)
    if scoped_by_id in data.configs and data.configs[scoped_by_id] != config_json:
        raise ContractError(
            f"config identity collision: {fmt}/{config_id!r} names two JSON objects")
    if scoped_by_json in data.config_ids_by_json and data.config_ids_by_json[scoped_by_json] != config_id:
        raise ContractError(
            f"config identity alias: {fmt} JSON is named "
            f"{data.config_ids_by_json[scoped_by_json]!r} and {config_id!r}")
    data.configs[scoped_by_id] = config_json
    data.config_ids_by_json[scoped_by_json] = config_id
    return CandidateIdentity(run_id, shape_id, shape_json, fmt, config_id, config_json)


def load_raw_lines(lines: Iterable[str], source: str = "<memory>") -> RawData:
    data = RawData({}, {}, collections.defaultdict(dict), {}, {}, {}, {}, {}, [])
    for line_no, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            data.complaints.append(f"{source}:{line_no}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(record, dict):
            data.complaints.append(f"{source}:{line_no}: record is not an object")
            continue
        try:
            if record.get("schema") != RAW_SCHEMA:
                raise ContractError(f"schema must be {RAW_SCHEMA!r}")
            rec = {"a": "attempt", "s": "sample", "x": "excluded"}.get(
                record.get("rec"), record.get("rec"))
            if rec == "run":
                run_id = _nonempty_string(record, "run_id")
                build = _nonempty_string(record, "build")
                space_id = _nonempty_string(record, "space_id")
                partial = record.get("partial_space")
                if not isinstance(partial, bool):
                    raise ContractError("partial_space must be boolean")
                value = RunInfo(run_id, build, space_id, partial)
                if run_id in data.runs and data.runs[run_id] != value:
                    raise ContractError(f"run_id {run_id!r} has conflicting headers")
                data.runs[run_id] = value
                continue
            if rec not in ("attempt", "sample", "excluded"):
                raise ContractError("rec must be run/attempt/sample/excluded")
            candidate = _candidate(record, data)
            if candidate.run_id not in data.runs:
                raise ContractError(f"record precedes run header {candidate.run_id!r}")
            attempt_id, pass_index = _attempt_key(record)
            aid = AttemptIdentity(candidate, attempt_id, pass_index)
            if rec == "attempt":
                expected = _nonnegative_int(record, "expected_samples")
                if expected == 0:
                    raise ContractError("expected_samples must be > 0")
                if aid in data.attempts:
                    raise ContractError("duplicate attempt record")
                copy = dict(record)
                copy["expected_samples"] = expected
                data.attempts[aid] = copy
            elif rec == "sample":
                sample_no = _aliased_nonnegative_int(record, "launch_index", "sample")
                value = _event_decimal(record)
                if sample_no in data.samples[aid]:
                    raise ContractError(f"duplicate sample index {sample_no}")
                data.samples[aid][sample_no] = value
            else:
                why = _nonempty_string(record, "why")
                if aid in data.exclusions:
                    raise ContractError("duplicate exclusion record")
                copy = dict(record)
                copy["why"] = why
                data.exclusions[aid] = copy
        except ContractError as exc:
            data.complaints.append(f"{source}:{line_no}: {exc}")

    # Completeness is checked after the full file, because the writer is free to
    # put an attempt before its later outcome and the process may die between them.
    all_outcomes = set(data.samples) | set(data.exclusions)
    for aid in sorted(all_outcomes - set(data.attempts), key=repr):
        data.complaints.append(f"{source}: outcome has no attempt: {aid}")
    for aid, attempt in data.attempts.items():
        got = data.samples.get(aid, {})
        excluded = aid in data.exclusions
        expected = attempt["expected_samples"]
        if excluded and got:
            data.complaints.append(f"{source}: attempt has both samples and exclusion: {aid}")
        elif excluded:
            continue
        elif set(got) != set(range(expected)):
            missing = sorted(set(range(expected)) - set(got))
            extra = sorted(set(got) - set(range(expected)))
            data.complaints.append(
                f"{source}: incomplete attempt {aid}: expected={expected} got={len(got)} "
                f"missing={missing} extra={extra}")
    return data


def load_raw(paths: Sequence[pathlib.Path]) -> RawData:
    merged_lines: list[str] = []
    for path in paths:
        try:
            merged_lines.extend(path.read_text().splitlines())
        except OSError as exc:
            raise ContractError(f"cannot read {path}: {exc}") from exc
    return load_raw_lines(merged_lines, ",".join(map(str, paths)))


@dataclasses.dataclass(frozen=True)
class Quantum:
    status: str
    value: Decimal | None
    reason: str


def infer_quantum(values: Iterable[Decimal],
                  minimum: Decimal = DEFAULT_MIN_QUANTUM_US) -> Quantum:
    """Infer a demonstrated event lattice with an exact decimal GCD.

    GCD can only overestimate a hidden finer quantum when the observed ticks have
    a common factor.  That is conservative for the <=1-quantum tie rule.  Decimal
    noise collapses the GCD; values below ``minimum`` are UNKNOWN rather than a
    spurious high-resolution timer.
    """
    vals = sorted(set(values))
    if len(vals) < 2:
        return Quantum("UNKNOWN", None, "fewer than two distinct event values")
    if any(not v.is_finite() or v <= 0 for v in vals):
        return Quantum("UNKNOWN", None, "non-positive or non-finite event value")
    places = max(max(0, -v.as_tuple().exponent) for v in vals)
    # More than nanosecond-fraction decimal printing is usually formatter noise.
    # Keep it in the GCD so it forces UNKNOWN instead of rounding it away.
    scale = 10 ** places
    ints = [int(v * scale) for v in vals]
    g = 0
    for value in ints:
        g = math.gcd(g, abs(value))
    if g == 0:
        return Quantum("UNKNOWN", None, "zero GCD")
    quantum = Decimal(g) / Decimal(scale)
    if quantum < minimum:
        return Quantum(
            "UNKNOWN", None,
            f"decimal GCD {quantum} us is below conservative floor {minimum} us")
    ticks = [v / quantum for v in vals]
    if any(t != t.to_integral_value() for t in ticks):
        # Defensive: exact integer construction above should make this impossible.
        return Quantum("UNKNOWN", None, "values are off the inferred lattice")
    return Quantum("KNOWN", quantum,
                   f"exact decimal GCD over {len(vals)} distinct raw event values")


def _decimal_median(values: Sequence[Decimal]) -> Decimal:
    ordered = sorted(values)
    n = len(ordered)
    if n & 1:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2


def _number(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


def _manifest_coverage(manifest: dict[str, Any], data: RawData,
                       selected_run: str | None) -> tuple[list[str], dict[str, Any]]:
    validate_manifest(manifest)
    expected = set()
    for job in manifest["jobs"]:
        shape_json = canonical(job["shape"])
        for item in job["expected"]:
            expected.add((job["shape_id"], shape_json, item["format"],
                          item["config_id"], canonical(item["config"])))
    complete = data.complete_attempts()
    observed = {
        (aid.candidate.shape_id, aid.candidate.shape_json, aid.candidate.fmt,
         aid.candidate.config_id, aid.candidate.config_json)
        for aid in complete
        if selected_run is None or aid.candidate.run_id == selected_run
    }
    missing = expected - observed
    unexpected = observed - expected
    complaints = []
    if missing:
        complaints.append(f"manifest coverage missing {len(missing)} candidate outcome(s)")
    if unexpected:
        complaints.append(f"raw data has {len(unexpected)} candidate(s) absent from manifest")
    return complaints, {
        "counts": manifest["counts"],
        "expected_outcomes": len(expected),
        "observed_outcomes": len(observed),
        "missing_outcomes": len(missing),
        "unexpected_outcomes": len(unexpected),
    }


def analyse_data(data: RawData, *, min_quantum_us: Decimal = DEFAULT_MIN_QUANTUM_US,
                 selected_run: str | None = None,
                 manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    if selected_run is not None and selected_run not in data.runs:
        raise ContractError(f"unknown --run-id {selected_run!r}")
    runs = [r for r in data.runs.values() if selected_run in (None, r.run_id)]
    if not runs:
        raise ContractError("no run header")
    build_space = {(r.build, r.space_id, r.partial_space) for r in runs}
    if len(build_space) != 1:
        raise ContractError(
            "selected records mix build/space/partial_space; select one --run-id: "
            + repr(sorted(build_space)))
    build, space_id, partial_space = next(iter(build_space))
    extra_complaints: list[str] = []
    coverage = None
    if manifest is not None:
        validate_manifest(manifest)
        if manifest["space_id"] != space_id:
            extra_complaints.append(
                f"manifest space_id={manifest['space_id']!r} differs from raw {space_id!r}")
        if manifest["partial_space"] != partial_space:
            extra_complaints.append(
                "manifest partial_space differs from raw run header")
        found, coverage = _manifest_coverage(manifest, data, selected_run)
        extra_complaints.extend(found)

    values_by_candidate: dict[CandidateIdentity, list[Decimal]] = collections.defaultdict(list)
    exclusion_reasons: collections.Counter[str] = collections.Counter()
    for aid, values in data.complete_attempts().items():
        if selected_run is not None and aid.candidate.run_id != selected_run:
            continue
        if aid in data.exclusions:
            exclusion_reasons[data.exclusions[aid]["why"]] += 1
        else:
            values_by_candidate[aid.candidate].extend(values)

    by_group: dict[tuple[str, str, str], list[tuple[CandidateIdentity, list[Decimal]]]] = \
        collections.defaultdict(list)
    shape_values: dict[tuple[str, str], list[Decimal]] = collections.defaultdict(list)
    for candidate, values in values_by_candidate.items():
        if not values:
            continue
        skey = (candidate.shape_id, candidate.shape_json)
        shape_values[skey].extend(values)
        by_group[(candidate.shape_id, candidate.shape_json, candidate.fmt)].append((candidate, values))
    shape_quantum = {k: infer_quantum(v, min_quantum_us) for k, v in shape_values.items()}

    groups: list[dict[str, Any]] = []
    for (shape_id, shape_json, fmt), candidates in sorted(by_group.items()):
        stats = []
        for candidate, values in candidates:
            med = _decimal_median(values)
            stats.append({
                "config_id": candidate.config_id,
                "config": json.loads(candidate.config_json),
                "raw_samples": len(values),
                "median_us": _number(med),
                "band_us": [_number(min(values)), _number(max(values))],
                "_median": med,
                "_lo": min(values),
                "_hi": max(values),
            })
        stats.sort(key=lambda x: (x["_median"], x["config_id"]))
        q = shape_quantum[(shape_id, shape_json)]
        reasons: list[str] = []
        leader = stats[0] if stats else None
        runner = stats[1] if len(stats) > 1 else None
        if runner is None:
            reasons.append("NO_RUNNER_UP")
        if q.status != "KNOWN":
            reasons.append("QUANTUM_UNKNOWN")
        gap: Decimal | None = None
        overlap = None
        if leader is not None and runner is not None:
            gap = runner["_median"] - leader["_median"]
            overlap = max(leader["_lo"], runner["_lo"]) <= min(leader["_hi"], runner["_hi"])
            if overlap:
                reasons.append("BAND_OVERLAP")
            if q.value is not None and gap <= q.value:
                reasons.append("WITHIN_ONE_QUANTUM")
        if partial_space:
            verdict = "LOWEST_IN_PARTIAL_SPACE"
        else:
            verdict = "UNRESOLVED" if reasons else "RESOLVED"
        for stat in stats:
            for internal in ("_median", "_lo", "_hi"):
                del stat[internal]
        groups.append({
            "shape_id": shape_id,
            "shape": json.loads(shape_json),
            "format": fmt,
            "partial_space": partial_space,
            "verdict": verdict,
            "reasons": reasons,
            "quantum": {
                "status": q.status,
                "us": _number(q.value),
                "reason": q.reason,
                "minimum_claimable_us": float(min_quantum_us),
            },
            "leader": None if leader is None else leader["config_id"],
            "runner_up": None if runner is None else runner["config_id"],
            "leader_runner_gap_us": _number(gap),
            "leader_runner_bands_overlap": overlap,
            "candidates": stats,
        })

    return {
        "schema": RESULT_SCHEMA,
        "build": build,
        "space_id": space_id,
        "partial_space": partial_space,
        "complete": not data.complaints and not extra_complaints,
        "complaints": data.complaints + extra_complaints,
        "attempts": len(data.attempts),
        "complete_attempts": len(data.complete_attempts()),
        "excluded_attempts": len(data.exclusions),
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "manifest_coverage": coverage,
        "groups": groups,
    }


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError("manifest root must be an object")
    return value


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ContractError(f"manifest schema must be {MANIFEST_SCHEMA!r}")
    _nonempty_string(manifest, "space_id")
    if not isinstance(manifest.get("partial_space"), bool):
        raise ContractError("manifest partial_space must be boolean")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ContractError("manifest counts must be an object")
    total = _nonnegative_int(counts, "total")
    legal = _nonnegative_int(counts, "legal")
    pruned = _nonnegative_int(counts, "pruned")
    reasons = counts.get("prune_reasons")
    if not isinstance(reasons, dict) or any(
            not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, int) or v < 0
            for k, v in reasons.items()):
        raise ContractError("counts.prune_reasons must map strings to non-negative integers")
    if total != legal + pruned:
        raise ContractError("counts.total must equal legal + pruned")
    if sum(reasons.values()) != pruned:
        raise ContractError("prune reason histogram must sum to counts.pruned")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ContractError("manifest jobs must be a non-empty array")
    seen_jobs: set[str] = set()
    shape_id_to_json: dict[str, str] = {}
    shape_json_to_id: dict[str, str] = {}
    seen_expected: set[tuple[str, str, str, str, str]] = set()
    expected_id_to_json: dict[tuple[str, str], str] = {}
    expected_json_to_id: dict[tuple[str, str], str] = {}
    expected_count = 0
    for job in jobs:
        if not isinstance(job, dict):
            raise ContractError("every manifest job must be an object")
        job_id = _nonempty_string(job, "job_id")
        if job_id in seen_jobs:
            raise ContractError(f"duplicate job_id {job_id!r}")
        seen_jobs.add(job_id)
        shape_id = _nonempty_string(job, "shape_id")
        shape_json = canonical(_identity_object(job, "shape"))
        if shape_id in shape_id_to_json and shape_id_to_json[shape_id] != shape_json:
            raise ContractError(f"manifest shape identity collision for {shape_id!r}")
        if shape_json in shape_json_to_id and shape_json_to_id[shape_json] != shape_id:
            raise ContractError(f"manifest shape identity alias for {shape_id!r}")
        shape_id_to_json[shape_id] = shape_json
        shape_json_to_id[shape_json] = shape_id
        argv = job.get("argv", [])
        env = job.get("env", {})
        if not isinstance(argv, list) or any(not isinstance(x, str) for x in argv):
            raise ContractError(f"job {job_id}: argv must be an array of strings")
        if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                            for k, v in env.items()):
            raise ContractError(f"job {job_id}: env must map strings to strings")
        formats = job.get("formats")
        if not isinstance(formats, list) or not formats or any(not isinstance(x, str) or not x for x in formats):
            raise ContractError(f"job {job_id}: formats must be non-empty strings")
        expected = job.get("expected")
        if not isinstance(expected, list) or not expected:
            raise ContractError(f"job {job_id}: expected must be non-empty")
        for item in expected:
            if not isinstance(item, dict):
                raise ContractError(f"job {job_id}: expected item must be an object")
            fmt = _nonempty_string(item, "format")
            if fmt not in formats:
                raise ContractError(f"job {job_id}: expected format {fmt!r} not in formats")
            cid = _nonempty_string(item, "config_id")
            cjson = canonical(_identity_object(item, "config"))
            by_id = (fmt, cid)
            by_json = (fmt, cjson)
            if by_id in expected_id_to_json and expected_id_to_json[by_id] != cjson:
                raise ContractError(f"manifest config identity collision for {fmt}/{cid!r}")
            if by_json in expected_json_to_id and expected_json_to_id[by_json] != cid:
                raise ContractError(f"manifest config identity alias for {fmt}/{cid!r}")
            expected_id_to_json[by_id] = cjson
            expected_json_to_id[by_json] = cid
            occurrence = (shape_id, shape_json, fmt, cid, cjson)
            if occurrence in seen_expected:
                raise ContractError(
                    f"manifest repeats expected candidate {shape_id}/{fmt}/{cid}")
            seen_expected.add(occurrence)
            expected_count += 1
    if expected_count != legal:
        raise ContractError(f"counts.legal={legal}, but jobs enumerate {expected_count} expected rows")


def manifest_hash(manifest: dict[str, Any]) -> str:
    return digest(manifest)


def _progress_done(path: pathlib.Path, plan_hash: str) -> tuple[set[str], dict[str, int]]:
    done: set[str] = set()
    attempts: dict[str, int] = collections.defaultdict(int)
    if not path.exists():
        return done, attempts
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{path}:{n}: invalid progress JSON: {exc.msg}") from exc
        if record.get("schema") != PROGRESS_SCHEMA:
            raise ContractError(f"{path}:{n}: wrong progress schema")
        if record.get("manifest_sha256") != plan_hash:
            raise ContractError(f"{path}:{n}: progress belongs to a different manifest")
        job_id = _nonempty_string(record, "job_id")
        attempt = _nonnegative_int(record, "attempt")
        attempts[job_id] = max(attempts[job_id], attempt + 1)
        if record.get("status") == "complete":
            done.add(job_id)
    return done, attempts


def _append_line(path: pathlib.Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(canonical(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_file(source: pathlib.Path, destination: pathlib.Path) -> int:
    data = source.read_bytes() if source.exists() else b""
    if not data:
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("ab") as handle:
        handle.write(data)
        if not data.endswith(b"\n"):
            handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return len(data)


def _expected_job_outcomes(job: dict[str, Any], data: RawData,
                           manifest: dict[str, Any], run_id: str) -> tuple[bool, str]:
    expected = {(item["format"], item["config_id"], canonical(item["config"]))
                for item in job["expected"]}
    observed_attempts = {
        (aid.candidate.fmt, aid.candidate.config_id, aid.candidate.config_json)
        for aid in data.attempts
    }
    wrong_shape = [
        aid for aid in data.attempts
        if aid.candidate.shape_id != job["shape_id"]
        or aid.candidate.shape_json != canonical(job["shape"])
    ]
    if wrong_shape:
        return False, f"records use a different shape identity: {wrong_shape[:2]!r}"
    if set(data.runs) != {run_id}:
        return False, f"child run headers are {sorted(data.runs)!r}, expected {run_id!r}"
    run = data.runs.get(run_id)
    if run is not None and (run.space_id != manifest["space_id"]
                            or run.partial_space != manifest["partial_space"]):
        return False, "child run header disagrees with manifest space/partial_space"
    unexpected = observed_attempts - expected
    missing = expected - observed_attempts
    if unexpected:
        return False, f"unexpected candidate identities: {sorted(unexpected)!r}"
    if missing:
        return False, f"missing candidate attempts: {sorted(missing)!r}"
    if data.complaints:
        return False, "; ".join(data.complaints)
    return True, "complete attempt/sample/excluded coverage"


def dry_run_records(manifest: dict[str, Any], binary: pathlib.Path, pending: set[str],
                    shape_timeout: float, raw_env: str, run_id: str) -> list[dict[str, Any]]:
    records = []
    for job in manifest["jobs"]:
        if job["job_id"] not in pending:
            continue
        env = dict(job.get("env", {}))
        env.update({
            raw_env: "<private-jsonl>",
            "GEMV_SWEEP_RUN_ID": run_id,
            "GEMV_SWEEP_JOB_ID": job["job_id"],
            "GEMV_SWEEP_ATTEMPT": "<resume-attempt>",
        })
        records.append({
            "schema": "gemv-sweep-dry-run-v1",
            "job_id": job["job_id"],
            "shape_id": job["shape_id"],
            "shape": job["shape"],
            "formats": job["formats"],
            "command": [str(binary)] + job.get("argv", []),
            "env": env,
            "shape_timeout_seconds": shape_timeout,
            "expected": job["expected"],
        })
    return records


def run_manifest(args: argparse.Namespace) -> int:
    manifest_path = pathlib.Path(args.manifest)
    manifest = _read_json(manifest_path)
    validate_manifest(manifest)
    plan_hash = manifest_hash(manifest)
    binary = pathlib.Path(args.bin).resolve() if args.bin else pathlib.Path("<binary>")
    progress = pathlib.Path(args.progress)
    raw = pathlib.Path(args.raw)
    if not args.resume and (progress.exists() or raw.exists()) and not args.dry_run:
        raise ContractError("raw/progress exists; pass --resume or choose fresh paths")
    done, attempt_numbers = _progress_done(progress, plan_hash) if args.resume else (set(), {})
    pending = {job["job_id"] for job in manifest["jobs"] if job["job_id"] not in done}
    run_id = args.run_id or f"run-{plan_hash[:12]}"

    if args.dry_run:
        records = dry_run_records(manifest, binary, pending, args.shape_timeout,
                                  args.jsonl_env, run_id)
        text = "".join(canonical(r) + "\n" for r in records)
        if args.dry_run_manifest:
            out = pathlib.Path(args.dry_run_manifest)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text)
        sys.stdout.write(text)
        print(canonical({
            "schema": "gemv-sweep-dry-run-summary-v1",
            "manifest_sha256": plan_hash,
            "space_id": manifest["space_id"],
            "partial_space": manifest["partial_space"],
            "counts": manifest["counts"],
            "pending_jobs": len(records),
            "completed_jobs": len(done),
        }))
        return 0
    if not binary.is_file():
        raise ContractError(f"binary does not exist: {binary}")

    start = time.monotonic()
    shape_spent: dict[str, float] = collections.defaultdict(float)
    completed_now = failed = timed_out = never = 0
    for job in manifest["jobs"]:
        job_id = job["job_id"]
        if job_id not in pending:
            continue
        elapsed = time.monotonic() - start
        overall_left = args.deadline_seconds - elapsed
        shape_left = args.shape_timeout - shape_spent[job["shape_id"]]
        if overall_left <= 0:
            never += 1
            continue
        if shape_left <= 0:
            timed_out += 1
            continue
        timeout = min(overall_left, shape_left)
        attempt = int(attempt_numbers.get(job_id, 0))
        raw.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                prefix=f".{raw.name}.{job_id}.", suffix=".jsonl", dir=raw.parent,
                delete=False) as tmp_handle:
            tmp = pathlib.Path(tmp_handle.name)
        env = dict(os.environ)
        env.update(job.get("env", {}))
        env.update({
            args.jsonl_env: str(tmp),
            "GEMV_SWEEP_RUN_ID": run_id,
            "GEMV_SWEEP_JOB_ID": job_id,
            "GEMV_SWEEP_ATTEMPT": str(attempt),
        })
        command = [str(binary)] + job.get("argv", [])
        launched = time.monotonic()
        status = "failed"
        reason = ""
        rc: int | None = None
        stdout = stderr = ""
        try:
            proc = subprocess.run(command, env=env, text=True, capture_output=True, timeout=timeout)
            rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
            parsed = load_raw_lines(tmp.read_text().splitlines(), str(tmp)) if tmp.stat().st_size else \
                load_raw_lines([], str(tmp))
            valid, why = _expected_job_outcomes(job, parsed, manifest, run_id)
            if rc == 0 and valid:
                status, reason = "complete", why
                completed_now += 1
            else:
                status = "invalid" if rc == 0 else "failed"
                reason = (f"rc={rc}; " if rc else "") + why
                failed += 1
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            status, reason = "timeout", f"exceeded {timeout:.3f}s remaining budget"
            timed_out += 1
        finally:
            duration = time.monotonic() - launched
            shape_spent[job["shape_id"]] += duration
            appended = _append_file(tmp, raw)
            tmp.unlink(missing_ok=True)
        if args.logs_dir:
            logs = pathlib.Path(args.logs_dir)
            logs.mkdir(parents=True, exist_ok=True)
            (logs / f"{job_id}.attempt{attempt}.stdout").write_text(stdout)
            (logs / f"{job_id}.attempt{attempt}.stderr").write_text(stderr)
        _append_line(progress, {
            "schema": PROGRESS_SCHEMA,
            "manifest_sha256": plan_hash,
            "run_id": run_id,
            "job_id": job_id,
            "shape_id": job["shape_id"],
            "attempt": attempt,
            "status": status,
            "reason": reason,
            "returncode": rc,
            "duration_seconds": duration,
            "raw_bytes_appended": appended,
        })
        print(f"[gemv-sweep] {job_id}: {status} ({duration:.2f}s): {reason}")
    remaining = len(pending) - completed_now - failed - timed_out - never
    never += max(0, remaining)
    print(f"[gemv-sweep] completed={completed_now} failed={failed} timeout={timed_out} "
          f"never_attempted={never} resumed={len(done)}")
    return 0 if failed == timed_out == never == 0 else 3


def _raw_records(*, values: dict[str, list[str]], partial: bool = False,
                 run_id: str = "r", config_overrides: dict[str, dict[str, Any]] | None = None) -> list[str]:
    """Small exact fixture used by --self-test and the CI wrapper."""
    lines = [canonical({"rec": "run", "schema": RAW_SCHEMA, "run_id": run_id,
                        "build": "b", "space_id": "full", "partial_space": partial})]
    config_overrides = config_overrides or {}
    for ci, (config_id, samples) in enumerate(values.items()):
        config = config_overrides.get(config_id, {
            "format": "int4", "quant_op": "finegrained_scale_zero", "group_size": 32,
            "w_layout": "native", "tile_k": 0, "step_k": 16, "threads": 64,
            "cta_m": 1, "cta_n": 1 + ci, "chunk": 1, "dtype_a": "fp16", "dtype_c": "fp16"})
        base = {"schema": RAW_SCHEMA, "run_id": run_id,
                "shape_id": "shape", "shape": {"m": 1, "n": 2048, "k": 2048,
                                                       "experts": 0, "active": 1},
                "format": "int4", "config_id": config_id, "config": config,
                "attempt_id": "0", "pass": 0}
        lines.append(canonical(dict(base, rec="attempt", expected_samples=len(samples))))
        for i, value in enumerate(samples):
            bits = struct.unpack("<I", struct.pack("<f", float(Decimal(value) / 1000)))[0]
            lines.append(canonical(dict(base, rec="sample", launch_index=i,
                                        event_ms_bits=bits, event_us=value)))
    return lines


def self_test(verbose: bool = True) -> None:
    def check(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    q = infer_quantum(map(Decimal, ("6.144", "8.192", "10.240")))
    check(q.status == "KNOWN" and q.value == Decimal("2.048"), f"2.048 lattice: {q}")

    one = analyse_data(load_raw_lines(_raw_records(values={
        "a": ["10.240", "10.240"], "b": ["12.288", "12.288"]})))
    g = one["groups"][0]
    check(g["verdict"] == "UNRESOLVED" and "WITHIN_ONE_QUANTUM" in g["reasons"],
          f"one tick must be unresolved: {g}")

    separated = analyse_data(load_raw_lines(_raw_records(values={
        "a": ["10.240", "10.240"], "b": ["14.336", "14.336"]})))
    check(separated["groups"][0]["verdict"] == "RESOLVED", ">tick separated must resolve")

    overlap = analyse_data(load_raw_lines(_raw_records(values={
        "a": ["8.192", "8.192", "14.336"],
        "b": ["10.240", "12.288", "12.288"]})))
    check("BAND_OVERLAP" in overlap["groups"][0]["reasons"], "band overlap must be unresolved")

    off = infer_quantum(map(Decimal, ("6.144", "8.193", "10.240")))
    check(off.status == "UNKNOWN", f"off-lattice must be unknown: {off}")
    single = infer_quantum([Decimal("6.144"), Decimal("6.144")])
    check(single.status == "UNKNOWN", f"single-value must be unknown: {single}")

    partial = analyse_data(load_raw_lines(_raw_records(values={
        "a": ["10.240", "10.240"], "b": ["14.336", "14.336"]}, partial=True)))
    check(partial["groups"][0]["verdict"] == "LOWEST_IN_PARTIAL_SPACE",
          "partial space must never publish a winner")

    collision_lines = _raw_records(values={"same": ["10.240"]})
    # Same config ID, different full JSON: the omission of one future axis must
    # fail rather than make two kernels look like repeated samples.
    collision = json.loads(collision_lines[-1])
    collision["config"] = dict(collision["config"], cta_n=99)
    collision["launch_index"] = 1
    collision_lines.append(canonical(collision))
    collided = load_raw_lines(collision_lines)
    check(any("identity collision" in x for x in collided.complaints),
          f"JSON identity collision was not rejected: {collided.complaints}")

    incomplete = _raw_records(values={"a": ["10.240", "12.288"]})
    incomplete.pop()  # advertised sample 1 never arrives
    bad = load_raw_lines(incomplete)
    check(any("incomplete attempt" in x for x in bad.complaints),
          "sample hole must be visible")
    if verbose:
        print("[gemv-sweep-self-test] PASS: 2.048-us lattice; one-tick/band unresolved; "
              ">tick separated; off-lattice/single unknown; partial-space label; "
              "identity collision and incomplete attempt fail closed")


def analyse_command(args: argparse.Namespace) -> int:
    paths = [pathlib.Path(p) for p in args.jsonl]
    data = load_raw(paths)
    manifest = _read_json(pathlib.Path(args.manifest)) if args.manifest else None
    result = analyse_data(data, min_quantum_us=Decimal(args.min_quantum_us),
                          selected_run=args.run_id, manifest=manifest)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        pathlib.Path(args.output).write_text(text)
    else:
        sys.stdout.write(text)
    return 0 if result["complete"] else 2


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", help="run planted analyser tests and exit")
    sub = ap.add_subparsers(dest="command")
    ana = sub.add_parser("analyse", help="validate raw JSONL and rank shape x format")
    ana.add_argument("jsonl", nargs="+", help="gemv-sweep-raw-v1 JSONL file(s)")
    ana.add_argument("--output", help="write result JSON here (default: stdout)")
    ana.add_argument("--run-id", help="select one run when a file contains several")
    ana.add_argument("--manifest", help="require exact candidate coverage from this manifest")
    ana.add_argument("--min-quantum-us", default=str(DEFAULT_MIN_QUANTUM_US),
                     help="smaller decimal GCDs are UNKNOWN (default: %(default)s)")

    run = sub.add_parser("run", help="run/resume a bounded manifest")
    run.add_argument("manifest", help="gemv-sweep-manifest-v1 JSON")
    run.add_argument("--bin", help="benchmark executable (not required for --dry-run)")
    run.add_argument("--raw", required=True, help="durable raw JSONL destination")
    run.add_argument("--progress", required=True, help="durable driver progress JSONL")
    run.add_argument("--run-id", help="stable run id; default derives from manifest hash")
    run.add_argument("--shape-timeout", type=float, default=900.0,
                     help="seconds budget shared by jobs of one shape")
    run.add_argument("--deadline-seconds", type=float, default=7200.0,
                     help="overall wall deadline for this invocation")
    run.add_argument("--jsonl-env", default="GEMV_SWEEP_JSONL",
                     help="environment variable carrying the private child JSONL")
    run.add_argument("--logs-dir", help="keep child stdout/stderr per attempt")
    run.add_argument("--resume", action="store_true", help="reuse matching progress and retry incomplete jobs")
    run.add_argument("--dry-run", action="store_true", help="emit exact pending job manifest; launch nothing")
    run.add_argument("--dry-run-manifest", help="also write dry-run JSONL here")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.self_test:
            self_test()
            return 0
        if args.command == "analyse":
            return analyse_command(args)
        if args.command == "run":
            if args.shape_timeout <= 0 or args.deadline_seconds <= 0:
                raise ContractError("timeouts must be > 0")
            return run_manifest(args)
        parser().print_help(sys.stderr)
        return 2
    except ContractError as exc:
        print(f"[gemv-sweep] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
