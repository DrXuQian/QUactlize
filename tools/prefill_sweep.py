#!/usr/bin/env python3
"""Plan, run, and summarise the first real-format PPU prefill sweep.

This is deliberately a narrow bridge to the existing ``test_scalefirst_bench``
rather than a second benchmark implementation.  The JSON spec supplies tensor
names and expected geometry; qtype is read from the actual GGUF tensor header.
Every measured row therefore has three independently visible identities:

  * the checkpoint tensor and its GGUF qtype;
  * the resident code-plane arrangement, including per-plane FoldN derived
    from (bits, ArtifactTileK);
  * the exact finite candidate denominator present in
    benchmarks/test_scalefirst_bench.cu.

The finite denominator is read from the benchmark source and, for Q8_0, its
shared candidate manifest.  The timing scope is ScaleFirst GEMM only.  It does NOT include the metadata
prepass and does NOT measure direct FullyQuantized GEMM.  Those missing arms are
reported in the plan rather than silently filled with a timing from a different
route.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import statistics
import struct
import subprocess
import shutil
import sys
import time
from typing import BinaryIO

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PLANNER_SOURCE = pathlib.Path(__file__).resolve()

PLAN_SCHEMA = "quactlize-prefill-plan-v2"
RESULT_SCHEMA = "quactlize-prefill-scale-first-result-v1"
SUPPORTED_SPEC_SCHEMA = "quactlize-prefill-smoke-v1"
BENCH_SOURCE = ROOT / "benchmarks" / "test_scalefirst_bench.cu"
Q8_CANDIDATE_SOURCE = ROOT / "benchmarks" / "prefill_q8_candidates.inc"
REGISTRY_SOURCE = ROOT / "quactlize" / "include" / "ppu_format_config.inc"
Q8_ORACLE_SOURCE = ROOT / "dev" / "fold_derivation" / "l208_q8_emit_layout.cu"
Q8_ORACLE_RUNNER = ROOT / "dev" / "fold_derivation" / "run_l208_q8_emit_layout.sh"
MIX_EMIT_SOURCE = (ROOT / "quactlize" / "include" / "quactlize_extensions" / "cutlass" /
                   "quactlize_mix_gemm_convert.h")
XPLANE_SOURCE = ROOT / "quactlize" / "include" / "xplane_offline.hpp"
VENDOR_INT8_CONVERTER = (ROOT / "third_party" / "actlize" / "include" / "cutlass" /
                         "fast_numeric_conversion_for_mix_gemm.h")

# This maps a GGUF semantic format to an explicitly tagged row family.  In
# particular Q4_K maps to q4, never to the ScaleOnly i4 ceiling.
FAMILY_BY_QTYPE = {8: "q8", 10: "i2", 11: "BC", 12: "q4", 13: "q5", 14: "q6"}
DENOMINATOR_KEY_BY_QTYPE = {8: "q8", 10: "q2", 11: "q3", 12: "q4", 13: "q5", 14: "q6"}

SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                7: 1, 10: 8, 11: 8, 12: 8}


def _read_exact(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise ValueError(f"truncated GGUF header: wanted {n} bytes, got {len(b)}")
    return b


def _u32(f: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _gguf_string(f: BinaryIO) -> str:
    n = _u64(f)
    return _read_exact(f, n).decode("utf-8")


def _skip_value(f: BinaryIO, value_type: int) -> None:
    if value_type == 8:  # string
        _ = _gguf_string(f)
        return
    if value_type == 9:  # array
        element_type, count = _u32(f), _u64(f)
        if element_type == 9:
            raise ValueError("GGUF metadata contains a nested array, forbidden by the GGUF format")
        for _ in range(count):
            _skip_value(f, element_type)
        return
    size = SCALAR_SIZES.get(value_type)
    if size is None:
        raise ValueError(f"unknown GGUF metadata value type {value_type}")
    _read_exact(f, size)


def gguf_tensor_headers(path: pathlib.Path) -> list[dict]:
    """Read tensor identity from the file itself; never infer qtype from its name."""
    with path.open("rb") as f:
        if _read_exact(f, 4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version = _u32(f)
        if version not in (2, 3):
            raise ValueError(f"{path}: GGUF version {version} is not supported by this header reader")
        tensor_count, kv_count = _u64(f), _u64(f)
        for _ in range(kv_count):
            _ = _gguf_string(f)
            _skip_value(f, _u32(f))
        out = []
        for _ in range(tensor_count):
            name = _gguf_string(f)
            nd = _u32(f)
            dims = [_u64(f) for _ in range(nd)]
            qtype, offset = _u32(f), _u64(f)
            out.append({"name": name, "dims": dims, "qtype": qtype, "offset": offset})
        return out


Q8_CANDIDATE_RE = re.compile(
    r"^PREFILL_Q8_CANDIDATE\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,"
    r"\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$")


def q8_candidate_tuples(source: pathlib.Path = Q8_CANDIDATE_SOURCE) -> list[tuple[int, ...]]:
    """Parse the shared authority strictly; an ignored active line is a missing candidate."""
    rows = []
    for lineno, line in enumerate(source.read_text().splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "#")):
            continue
        match = Q8_CANDIDATE_RE.fullmatch(stripped)
        if not match:
            raise ValueError(f"{source}:{lineno}: unparsed active Q8 candidate line: {stripped!r}")
        rows.append(tuple(map(int, match.groups())))
    if len(set(rows)) != len(rows):
        raise ValueError(f"{source}: duplicate Q8 candidate tuple")
    if not rows:
        raise ValueError(f"{source}: empty Q8 candidate authority")
    return rows


def q8_candidate_manifest(source: pathlib.Path = Q8_CANDIDATE_SOURCE) -> dict:
    rows = q8_candidate_tuples(source)
    canonical = json.dumps(rows, separators=(",", ":"))
    return {"schema": "quactlize-q8-prefill-candidates-v1", "count": len(rows),
            "tuples_tm_tn_tk_wm_wn_stages": [list(row) for row in rows],
            "tuple_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "source": str(source.resolve()), "source_sha256": sha256_file(source)}


def source_denominators(source: pathlib.Path = BENCH_SOURCE,
                        q8_source: pathlib.Path = Q8_CANDIDATE_SOURCE) -> dict[str, int]:
    """Count calls in the measured source authorities, never a mirrored number."""
    prefixes = {
        "i2": ("I2(", "I2F("),
        "BC": ("BC(", "BCF("),
        "q4": ("Q4(",),
        "q5": ('Q65("q5"',),
        "q6": ('Q65("q6"',),
    }
    counts = {k: 0 for k in prefixes}
    for line in source.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("#define"):
            continue
        for family, starts in prefixes.items():
            if any(s.startswith(p) for p in starts):
                counts[family] += 1
    counts["q8"] = len(q8_candidate_tuples(q8_source))
    return counts


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def recorded_source_hashes() -> list[dict]:
    sources = (PLANNER_SOURCE, BENCH_SOURCE, Q8_CANDIDATE_SOURCE,
               Q8_ORACLE_SOURCE, Q8_ORACLE_RUNNER, MIX_EMIT_SOURCE,
               XPLANE_SOURCE, VENDOR_INT8_CONVERTER)
    return [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in sources]


def verify_recorded_sources(plan: dict) -> None:
    """Bind plan, later build, and direct measure to the same source bytes."""
    recorded = plan.get("source_authorities")
    expected_paths = [str(path.resolve()) for path in
                      (PLANNER_SOURCE, BENCH_SOURCE, Q8_CANDIDATE_SOURCE,
                       Q8_ORACLE_SOURCE, Q8_ORACLE_RUNNER, MIX_EMIT_SOURCE,
                       XPLANE_SOURCE, VENDOR_INT8_CONVERTER)]
    if not isinstance(recorded, list) or [item.get("path") for item in recorded] != expected_paths:
        raise ValueError("plan source_authorities is missing, reordered, or names a different authority set")
    for item in recorded:
        path = pathlib.Path(item["path"])
        got = sha256_file(path)
        if got != item.get("sha256"):
            raise ValueError(f"source authority changed after plan: {path} {item.get('sha256')} -> {got}")
    registry_path = pathlib.Path(plan.get("registry", ""))
    if registry_path.resolve() != REGISTRY_SOURCE.resolve():
        raise ValueError(f"plan registry path is not this checkout's authority: {registry_path}")
    registry_hash = sha256_file(REGISTRY_SOURCE)
    if registry_hash != plan.get("registry_sha256"):
        raise ValueError(f"registry authority changed after plan: {plan.get('registry_sha256')} -> {registry_hash}")
    if plan.get("q8_candidate_manifest") != q8_candidate_manifest():
        raise ValueError("Q8 candidate manifest changed after plan")
    spec = pathlib.Path(plan.get("spec", ""))
    if sha256_file(spec) != plan.get("spec_sha256"):
        raise ValueError(f"prefill spec changed after plan: {spec}")


def fold_for(bits: int, tile_k: int) -> int:
    if bits <= 0 or tile_k <= 0 or (bits * tile_k) % 8:
        raise ValueError(f"invalid plane run: bits={bits} tile_k={tile_k}")
    run_bytes = bits * tile_k // 8
    if run_bytes >= 32:
        return 1
    if not run_bytes or 32 % run_bytes:
        raise ValueError(f"plane run {run_bytes} B cannot be FoldN-packed to the 32 B floor")
    return 32 // run_bytes


def load_registry() -> dict[int, dict]:
    # Parse the shipping X-macro instead of mirroring qtype semantics here.
    from tools.pack_gguf import format_registry
    return format_registry()


def controlled_scalefirst_row(qtype: int) -> dict | None:
    """Describe a proved ScaleFirst-only route without forging a shipping FQ row.

    Q8_0 is not a K-quant format and therefore does not belong in
    ``ppu_format_config.inc``: that registry also promises a native packed
    metadata/FullyQuantized reader.  Its existing int8 collective plus L208's
    independently anchored A32 xplane producer are sufficient for the
    controlled resident GEMM leg measured here.  This does not claim that a
    checkpoint Q8 block split/reorder producer is wired.
    """
    if qtype != 8:
        return None
    from quactlize.formats import BLOCKS, QuantType
    layout = BLOCKS[QuantType.Q8_0]
    if (layout.weights, layout.block_bytes, layout.scale_meta_bytes,
            layout.group_size, layout.has_min) != (32, 34, 2, 32, False):
        raise ValueError(f"Q8_0 block-layout authority drifted: {layout}")
    return {
        "name": "Q8_0",
        "qtype": 8,
        "low_bits": 8,
        "high_bits": 0,
        "group_size": layout.group_size,
        "scale_first_tile_k": 32,
        "metadata_fp16_planes": 1,
        "quant_mode": "FinegrainedScaleOnly",
        "route_scope": "CONTROLLED_RESIDENT_SCALEFIRST_GEMM_ONLY",
        "registry_backed": False,
        "layout_authority": "L208 vendor-int8-emission+xplane-A32-F1",
    }


def qtype_name(qtype: int) -> str:
    """Return the repository's semantic GGUF name, independent of route support.

    The shipping ScaleFirst registry intentionally contains only Q2_K..Q6_K.
    Using that registry as the *name* authority turned recognized Q8_0 (ggml
    type 8) into the misleading string ``qtype-8``.  Format identity and this
    route's capability are separate facts.
    """
    from quactlize.formats import QuantType
    try:
        return QuantType(int(qtype)).name
    except ValueError:
        return f"UNKNOWN_QTYPE_{int(qtype)}"


def support_summary(cells: list[dict]) -> dict:
    supported = sum(c["support"]["state"] == "SUPPORTED" for c in cells)
    by_state: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for cell in cells:
        support = cell["support"]
        by_state[support["state"]] = by_state.get(support["state"], 0) + 1
        code = support["reason_code"]
        by_reason[code] = by_reason.get(code, 0) + 1
    return {
        "total_cells": len(cells),
        "supported_cells": supported,
        "unsupported_cells": len(cells) - supported,
        "by_state": by_state,
        "by_reason_code": by_reason,
    }


def plan_admission(plan: dict) -> tuple[bool, str]:
    """Validate the recorded count and decide whether a binary may be built.

    This is shared by the runner admission command and ``measure`` so bypassing
    the shell script cannot turn an all-unsupported plan into a benchmark run.
    """
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"not a {PLAN_SCHEMA} plan")
    verify_recorded_sources(plan)
    actual = support_summary(plan.get("cells", []))
    recorded = plan.get("support_summary")
    if recorded != actual:
        raise ValueError(f"plan support_summary disagrees with cells: recorded={recorded}, actual={actual}")
    if actual["supported_cells"] == 0:
        return False, (f"NO_SUPPORTED_CELLS: supported=0 unsupported={actual['unsupported_cells']} "
                       f"total={actual['total_cells']}; no binary may be built or measured")
    return True, (f"SUPPORTED_CELLS_PRESENT: supported={actual['supported_cells']} "
                  f"unsupported={actual['unsupported_cells']} total={actual['total_cells']}")


def build_plan(spec_path: pathlib.Path, gguf_path: pathlib.Path) -> dict:
    spec = json.loads(spec_path.read_text())
    if spec.get("schema") != SUPPORTED_SPEC_SCHEMA:
        raise ValueError(f"{spec_path}: schema must be {SUPPORTED_SPEC_SCHEMA!r}")
    m_values = spec.get("m_values")
    if not isinstance(m_values, list) or not m_values or any(not isinstance(m, int) or m <= 0 for m in m_values):
        raise ValueError("m_values must be a nonempty array of positive integers")
    projections = spec.get("projections")
    if not isinstance(projections, list) or not projections:
        raise ValueError("projections must be a nonempty array")

    headers = gguf_tensor_headers(gguf_path)
    registry = load_registry()
    denominators = source_denominators()
    q8_manifest = q8_candidate_manifest()
    # Resolve all projection names before creating cells.  Hybrid models such
    # as Qwen3.5 do not promise that block 0 is a full-attention block, so the
    # manifest captures a numeric layer id and we choose the lowest layer that
    # has q/k/v/o together at the declared geometry.  Independent "first q"
    # and "first k" choices are forbidden: they could silently splice layers.
    resolved = []
    for projection in projections:
        pattern = projection.get("tensor_pattern", "")
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"bad tensor_pattern {pattern!r}: {e}") from e
        expected = int(projection["n"]), int(projection["k"])
        name_matches = []
        for tensor in headers:
            match = rx.fullmatch(tensor["name"])
            if match:
                name_matches.append((tensor, match))
        geometry_matches = []
        for tensor, match in name_matches:
            if len(tensor["dims"]) != 2:
                continue
            # GGUF dimension order is K,N for llama.cpp dense weights.
            actual_n, actual_k = int(tensor["dims"][1]), int(tensor["dims"][0])
            if (actual_n, actual_k) == expected:
                geometry_matches.append((tensor, match.groupdict().get("layer")))
        if not geometry_matches:
            same_shape = [t["name"] for t in headers
                          if len(t["dims"]) == 2 and
                          (int(t["dims"][1]), int(t["dims"][0])) == expected][:12]
            named = [{"name": t["name"], "dims": t["dims"]} for t, _ in name_matches[:12]]
            attention = [t["name"] for t in headers if "attn" in t["name"]][:20]
            raise ValueError(
                f"{projection.get('name')}: pattern {pattern!r} has no tensor with N,K={expected}; "
                f"name_matches={named}; same_shape_candidates={same_shape}; attention_name_examples={attention}")
        resolved.append((projection, geometry_matches))

    if all(all(layer is not None for _, layer in matches) for _, matches in resolved):
        layer_sets = [{layer for _, layer in matches} for _, matches in resolved]
        common_layers = set.intersection(*layer_sets)
        if not common_layers:
            detail = {projection["name"]: sorted({layer for _, layer in matches}, key=int)
                      for projection, matches in resolved}
            raise ValueError(f"projection patterns have no common full-attention layer: {detail}")
        selected_layer = min(common_layers, key=int)
    else:
        if any(len(matches) != 1 for _, matches in resolved):
            detail = {projection["name"]: [t["name"] for t, _ in matches]
                      for projection, matches in resolved}
            raise ValueError(f"patterns without a named layer capture must resolve uniquely: {detail}")
        selected_layer = None

    cells = []
    selected_tensors = {}
    for projection, matches in resolved:
        selected = [(tensor, layer) for tensor, layer in matches
                    if selected_layer is None or layer == selected_layer]
        if len(selected) != 1:
            raise ValueError(f"{projection['name']}: layer {selected_layer!r} resolves to {len(selected)} tensors")
        tensor, tensor_layer = selected[0]
        selected_tensors[projection["name"]] = tensor["name"]
        actual_n, actual_k = int(tensor["dims"][1]), int(tensor["dims"][0])
        qtype = int(tensor["qtype"])
        registry_row = registry.get(qtype)
        row = registry_row or controlled_scalefirst_row(qtype)
        family = FAMILY_BY_QTYPE.get(qtype)
        semantic_name = qtype_name(qtype)
        if row is None:
            support = {
                "state": "UNSUPPORTED",
                "reason_code": "QTYPE_NOT_IN_CURRENT_SCALEFIRST_REGISTRY",
                "reason": (f"{semantic_name} (GGUF qtype {qtype}) is recognized from the tensor header but is "
                           "not registered for the current PPU ScaleFirst benchmark route"),
            }
        elif family is None:
            support = {
                "state": "UNSUPPORTED",
                "reason_code": "NO_BENCHMARK_ROW_FAMILY",
                "reason": f"{row['name']} has no semantic row family in test_scalefirst_bench",
            }
        elif denominators.get(family, 0) == 0:
            support = {
                "state": "UNSUPPORTED",
                "reason_code": "EMPTY_SOURCE_DENOMINATOR",
                "reason": f"row family {family} has an empty source denominator",
            }
        else:
            controlled = not bool(row.get("registry_backed", registry_row is not None))
            support = {"state": "SUPPORTED",
                       "reason_code": ("CONTROLLED_SCALEFIRST_ROW_FAMILY" if controlled
                                       else "REGISTERED_ROW_FAMILY"),
                       "reason": (f"{row['name']} maps to independently anchored ScaleFirst-only row family {family}"
                                  if controlled else
                                  f"{row['name']} maps to nonempty benchmark row family {family}"),
                       "family": family,
                       "source_denominator": denominators[family]}
        for m in m_values:
            cell = {
                "id": f"{projection['name']}-m{m}",
                "projection": projection["name"],
                "tensor": tensor["name"],
                "selected_layer": int(tensor_layer) if tensor_layer is not None else None,
                "qtype_source": "GGUF-tensor-header",
                "qtype": qtype,
                "format": semantic_name,
                "m": m, "n": actual_n, "k": actual_k,
                "group_size": row["group_size"] if row else None,
                "planes": [row["low_bits"], row["high_bits"]] if row else None,
                "quant_mode": row.get("quant_mode", "FinegrainedScaleZero") if row else None,
                "metadata_fp16_planes": row.get("metadata_fp16_planes", 2) if row else None,
                "route_scope": row.get("route_scope", "REGISTERED_KQUANT_SCALEFIRST") if row else None,
                "registry_backed": bool(row.get("registry_backed", registry_row is not None)) if row else False,
                "layout_authority": row.get("layout_authority", "ppu_format_config.inc+xplane") if row else None,
                "canonical_scale_first_arrangement": ({
                    "artifact_tile_k": row["scale_first_tile_k"],
                    "fold_n": [fold_for(row["low_bits"], row["scale_first_tile_k"]),
                               fold_for(row["high_bits"], row["scale_first_tile_k"])
                               if row["high_bits"] else 1],
                } if row else None),
                "support": support,
            }
            cells.append(cell)
    summary = support_summary(cells)
    return {
        "schema": PLAN_SCHEMA,
        "model": spec["model"],
        "scope": spec.get("scope", ""),
        "gguf": str(gguf_path.resolve()),
        "gguf_size": gguf_path.stat().st_size,
        "spec": str(spec_path.resolve()),
        "spec_sha256": sha256_file(spec_path),
        "registry": str(REGISTRY_SOURCE.resolve()),
        "registry_sha256": sha256_file(REGISTRY_SOURCE),
        "candidate_source": str(BENCH_SOURCE.resolve()),
        "candidate_source_sha256": sha256_file(BENCH_SOURCE),
        "source_authorities": recorded_source_hashes(),
        "q8_candidate_manifest": q8_manifest,
        "candidate_denominators": denominators,
        "support_summary": summary,
        "tensor_selection": {
            "policy": "lowest numeric layer present in every declared projection at the declared geometry",
            "selected_layer": int(selected_layer) if selected_layer is not None else None,
            "tensors": selected_tensors,
        },
        "timing_scope": "ScaleFirst GEMM only; metadata prepass and direct FullyQuantized GEMM excluded",
        "input_materialization_scope": (
            "GGUF supplies tensor identity/qtype/shape only; benchmark payloads are controlled synthetic resident "
            "artifacts; checkpoint block split/reorder is not timed or claimed wired"),
        "denominator_scope": "finite manual row families in test_scalefirst_bench; not the generated full tactic space",
        "cells": cells,
    }


CANDIDATE_RE = re.compile(
    r"^\s*(?P<family>BC|i2|q4|q5|q6|q8)\s+"
    r"(?P<tm>\d+)x(?P<tn>\d+):(?P<tk>\d+)\s+"
    r"w(?P<wm>\d+)x(?P<wn>\d+)\s+s(?P<st>\d+)"
    r"(?:\s+\(ScaleOnly\))?"
    r"(?:\s+\[(?P<annotation>[^]]+)\])?\s+"
    r"(?P<us>\d+(?:\.\d+)?)\s+us\s+\|",
    re.M | re.I,
)


def parse_candidates(text: str, family: str) -> list[dict]:
    out = []
    for match in CANDIDATE_RE.finditer(text):
        got_family = match.group("family")
        if got_family.lower() != family.lower():
            continue
        row = match.groupdict()
        for key in ("tm", "tn", "tk", "wm", "wn", "st"):
            row[key] = int(row[key])
        row["us"] = float(row["us"])
        row["family"] = got_family
        row["annotation"] = row.get("annotation") or ""
        row["config"] = (f"{row['tm']}x{row['tn']}x{row['tk']}_w{row['wm']}x{row['wn']}_s{row['st']}"
                         + (f"_[{row['annotation']}]" if row["annotation"] else ""))
        out.append(row)
    return out


DENOMINATOR_RE = re.compile(
    r"^\s*PREFILL_ROW_DENOMINATOR\s+q2=(\d+)\s+q3=(\d+)\s+q4=(\d+)\s+q5=(\d+)\s+q6=(\d+)\s+q8=(\d+)\s*$",
    re.M,
)
LAUNCH_STATUS_RE = re.compile(
    r"^\s*PREFILL_LAUNCH_STATUS\s+failures=(\d+)\s+verdict=(PASS|FAIL)\s*$", re.M)
Q8_CORRECTNESS_RE = re.compile(
    r"^\s*Q8_CORRECTNESS\s+config=(\S+)\s+bad=(\d+)/(\d+)\s+"
    r"fixture=(\S+)\s+verdict=(PASS|FAIL)\s*$", re.M)
Q8_FIXTURE_RE = re.compile(
    r"^\s*Q8_FIXTURE\s+shape=(\d+)x(\d+)x(\d+)\s+selected_k_per_row=(\d+)\s+"
    r"unique_bad=(\d+)\s+fp16_exact_bad=(\d+)\s+scale_values=(\d+)\s+"
    r"fixture=(\S+)\s+verdict=(PASS|FAIL)\s*$", re.M)


def runtime_contract(text: str, cell: dict, parsed_rows: int | list[dict]) -> int:
    """Return the runtime denominator, requiring the bench's fail-closed footer exactly once."""
    denoms = DENOMINATOR_RE.findall(text)
    if len(denoms) != 1:
        raise ValueError(f"expected one PREFILL_ROW_DENOMINATOR, found {len(denoms)}")
    statuses = LAUNCH_STATUS_RE.findall(text)
    if statuses != [("0", "PASS")]:
        raise ValueError(f"launch status is not exactly failures=0/PASS: {statuses}")
    keys = ("q2", "q3", "q4", "q5", "q6", "q8")
    values = dict(zip(keys, map(int, denoms[0])))
    key = DENOMINATOR_KEY_BY_QTYPE[int(cell["qtype"])]
    denominator = values[key]
    parsed_count = len(parsed_rows) if isinstance(parsed_rows, list) else parsed_rows
    if parsed_count != denominator:
        raise ValueError(f"parsed {parsed_count} {key} rows but runtime denominator is {denominator}")
    source_expected = int(cell["support"]["source_denominator"])
    if denominator != source_expected:
        raise ValueError(
            f"runtime denominator {denominator} disagrees with source-call authority {source_expected} for {key}")
    if int(cell["qtype"]) == 8:
        if not isinstance(parsed_rows, list):
            raise ValueError("Q8 runtime contract requires parsed tuple identities, not only a row count")
        emitted = [(row["tm"], row["tn"], row["tk"], row["wm"], row["wn"], row["st"])
                   for row in parsed_rows]
        authority = q8_candidate_tuples()
        if emitted != authority:
            missing = sorted(set(authority) - set(emitted))
            extra = sorted(set(emitted) - set(authority))
            raise ValueError(
                f"Q8 runtime tuple sequence differs from shared authority: missing={missing} extra={extra} "
                f"same_members={set(emitted) == set(authority)}")
        correctness = Q8_CORRECTNESS_RE.findall(text)
        expected_tags = [f"{tm}x{tn}:{tk}_w{wm}x{wn}_s{st}"
                         for tm, tn, tk, wm, wn, st in authority]
        got_tags = [tag for tag, _, _, _, _ in correctness]
        if got_tags != expected_tags:
            raise ValueError(
                f"Q8 correctness witness sequence differs from candidate authority: "
                f"got={got_tags} expected={expected_tags}")
        expected_total = int(cell["m"]) * int(cell["n"])
        if any(int(bad) != 0 or int(total) != expected_total or
               fixture != "ORDER-INDEPENDENT+FP16-EXACT" or verdict != "PASS"
               for _, bad, total, fixture, verdict in correctness):
            raise ValueError(f"Q8 correctness witness is red: {correctness}")
        fixture_rows = Q8_FIXTURE_RE.findall(text)
        expected_fixture = [(str(cell["m"]), str(cell["n"]), str(cell["k"]),
                             "4", "0", "0", "3", "ORDER-INDEPENDENT+FP16-EXACT", "PASS")]
        if fixture_rows != expected_fixture:
            raise ValueError(f"Q8 fixture identity/exactness is not bound to this invocation: {fixture_rows}")
    return denominator


def candidate_layout(candidate: dict, cell: dict) -> dict:
    low_bits, high_bits = cell["planes"]
    family, annotation = cell["support"]["family"], candidate["annotation"]
    # The old non-folded I2/BC macros consume the legacy interleave-256 buffer.
    # Every explicitly folded row and every q4/q5/q6 row calls xplane with TK.
    legacy = family in ("i2", "BC") and not annotation
    # Q8_0's resident file identity is one canonical 32-code/32-byte
    # delivery.  TacticTileK remains a reader axis and may concatenate A32
    # deliveries without changing the bytes on disk.
    artifact_tk = (32 if family == "q8" else 256 if legacy else candidate["tk"])
    folds = [fold_for(low_bits, artifact_tk), fold_for(high_bits, artifact_tk) if high_bits else 1]

    # The source prints the folds it instantiated.  A parser/layout mismatch is
    # a failed result, not a plausible descriptor attached to different bytes.
    annotation_words = annotation.split()
    annotation_pairs = [re.fullmatch(r"([A-Za-z][A-Za-z0-9]*)=(\d+)", word)
                        for word in annotation_words]
    if any(pair is None for pair in annotation_pairs):
        raise ValueError(f"{candidate['config']}: malformed layout annotation {annotation!r}")
    printed = {}
    for pair in annotation_pairs:
        assert pair is not None
        key, value = pair.groups()
        if key in printed:
            raise ValueError(f"{candidate['config']}: duplicate layout token {key}")
        printed[key] = int(value)
    if family == "q8":
        # Q8 is the only family whose tactic TK is deliberately different
        # from its artifact TK.  Therefore A=32 is a required measured token,
        # not a value the planner may silently infer after parsing a row that
        # omitted or contradicted it.
        if printed != {"A": 32, "F": 1}:
            raise ValueError(
                f"{candidate['config']}: Q8 layout annotation must be exactly A=32 F=1, got {annotation!r}")
    if "F" in printed and printed["F"] != folds[0]:
        raise ValueError(f"{candidate['config']}: printed F={printed['F']} but descriptor derives {folds[0]}")
    if "F1" in printed and printed["F1"] != folds[0]:
        raise ValueError(f"{candidate['config']}: printed F1={printed['F1']} but descriptor derives {folds[0]}")
    if "F2" in printed and printed["F2"] != folds[1]:
        raise ValueError(f"{candidate['config']}: printed F2={printed['F2']} but descriptor derives {folds[1]}")
    canonical = cell["canonical_scale_first_arrangement"]
    measured_descriptor = f"bits={low_bits},tile_k={artifact_tk},high_bits={high_bits}"
    # F=1 inside the verified <=256 interleave domain is the only class we
    # merge without materialising and hashing bytes.  Folded maps stay
    # conservatively distinct down to producer function and full tile/warp
    # geometry; equal FoldN is explicitly insufficient evidence of equal
    # resident bytes.
    if family == "q8":
        layout_class = "xplane-q8-a32-f1:l208"
        class_basis = ("dev/fold_derivation/l208_q8_emit_layout.cu: vendor int8 emission anchor + "
                       "byte-identical production placement over every emitted q8 tactic")
        resident_artifact_tk = 32
    elif folds == [1, 1] and artifact_tk <= 256:
        layout_class = f"xplane-tile-free-f1-le256:bits={low_bits}+{high_bits}"
        class_basis = "dev/fold_derivation/l115_artifact_tactic_code_slots.cu verified tile-free F1<=256"
        resident_artifact_tk = canonical["artifact_tile_k"]
    else:
        if family == "BC":
            producer_map = "place_derived+place_int1"
        elif family in ("q5", "q6"):
            producer_map = "place_derived+place_hi"
        elif family in ("i2", "q4", "q8"):
            producer_map = "place_derived"
        else:
            producer_map = "legacy-interleave256"
        layout_class = (f"conservative:{producer_map}:bits={low_bits}+{high_bits}:"
                        f"tm={candidate['tm']}:tn={candidate['tn']}:tk={candidate['tk']}:"
                        f"wm={candidate['wm']}:wn={candidate['wn']}:fold={folds[0]}+{folds[1]}")
        class_basis = "byte hash NOT_EMITTED; folded producer maps are not merged"
        resident_artifact_tk = artifact_tk
    descriptor = f"bits={low_bits},tile_k={resident_artifact_tk},high_bits={high_bits}"
    if legacy:
        producer = "legacy-interleave256"
    elif family == "BC":
        producer = "xplane-place_derived+place_int1"
    elif family in ("q5", "q6"):
        producer = "xplane-place_derived+place_hi"
    elif family == "q8":
        producer = "xplane-place_derived-q8-a32"
    else:
        producer = "xplane-place_derived"
    eligible = not (family == "BC" and legacy)
    return {
        "producer": producer,
        "artifact_tile_k": resident_artifact_tk,
        "measured_artifact_tile_k": artifact_tk,
        "low_bits": low_bits,
        "high_bits": high_bits,
        "fold_n": folds,
        "descriptor": descriptor,
        "descriptor_sha256": hashlib.sha256(descriptor.encode()).hexdigest(),
        "measured_descriptor": measured_descriptor,
        "layout_class": layout_class,
        "layout_class_basis": class_basis,
        "resident_byte_hash": "NOT_EMITTED",
        "shipping_registry_match": (bool(cell.get("registry_backed")) and
                                    resident_artifact_tk == canonical["artifact_tile_k"] and
                                    folds == canonical["fold_n"]),
        "shipping_registry_exact_measured_map": (bool(cell.get("registry_backed")) and
                                                   artifact_tk == canonical["artifact_tile_k"] and
                                                   folds == canonical["fold_n"]),
        "scale_first_contract_match": (resident_artifact_tk == canonical["artifact_tile_k"] and
                                       folds == canonical["fold_n"]),
        "selection_eligible": eligible,
        "selection_exclusion": ("known-wrong Q3 high-plane single-plane producer map; "
                                "BCF/place_int1 is the semantic authority"
                                if not eligible else None),
    }


def traffic(cell: dict, us: float, peak_tflops: float, hbm_gbs: float) -> dict:
    m, n, k, gs = (int(cell[x]) for x in ("m", "n", "k", "group_size"))
    low_bits, high_bits = cell["planes"]
    code = n * k * (low_bits + high_bits) // 8
    metadata_planes = int(cell["metadata_fp16_planes"])
    metadata = n * (k // gs) * 2 * metadata_planes
    act, out = m * k * 2, m * n * 2
    distinct = code + metadata + act + out
    tflops = 2.0 * m * n * k / us / 1.0e6
    gbs = distinct / us / 1000.0
    return {
        "flops": 2 * m * n * k,
        "tflops": tflops,
        "mfu_percent": 100.0 * tflops / peak_tflops,
        "distinct_bytes": distinct,
        "byte_breakdown": {"codes": code, "fp16_metadata": metadata,
                           "activation": act, "output": out},
        "metadata_planes": metadata_planes,
        "distinct_gbs": gbs,
        "mbu_percent": 100.0 * gbs / hbm_gbs,
        "peak_tflops": peak_tflops,
        "hbm_gbs": hbm_gbs,
        "traffic_scope": "one distinct ScaleFirst resident weight + declared fp16 metadata planes + fp16 A/D; cache reuse not multiplied by tiles",
    }


def layout_decisions(results: list[dict]) -> list[dict]:
    """Choose no per-M artifact: report one equal-weight minimax diagnostic per tensor.

    A conflicting per-cell winner remains CONFLICT/UNRESOLVED even when the
    minimax calculation has a numerical argmin.  That argmin is useful for the
    next experiment, not permission to emit an offline artifact.
    """
    grouped: dict[tuple, list[dict]] = {}
    for result in results:
        if result.get("disposition") not in ("RESOLVED", "UNRESOLVED"):
            continue
        cell = result["cell"]
        key = (cell["tensor"], cell["qtype"], cell["n"], cell["k"])
        grouped.setdefault(key, []).append(result)
    decisions = []
    for (tensor, qtype, n, k), cells in sorted(grouped.items()):
        cells.sort(key=lambda r: r["cell"]["m"])
        per_cell = []
        common = None
        leader_classes = []
        for result in cells:
            by_layout: dict[str, list[dict]] = {}
            for candidate in result["candidates"]:
                by_layout.setdefault(candidate["layout"]["layout_class"], []).append(candidate)
            best_by_layout = {layout: min(rows, key=lambda r: r["median_us"])
                              for layout, rows in by_layout.items()}
            per_cell.append((result, best_by_layout))
            layouts = set(best_by_layout)
            common = layouts if common is None else common & layouts
            leader_classes.append(result["leader"]["layout"]["layout_class"])
        conflict = len(set(leader_classes)) != 1
        timing_unresolved = any(result["disposition"] == "UNRESOLVED" for result in cells)
        scored = []
        for layout in sorted(common or ()):
            regrets = []
            tactics = []
            for result, best_by_layout in per_cell:
                selected = best_by_layout[layout]
                optimum = result["leader"]["median_us"]
                regrets.append(selected["median_us"] / optimum - 1.0)
                # Warmup is allowed to search tactics only inside the one
                # resident layout class selected offline.
                shortlist = sorted(
                    (c for c in result["candidates"] if c["layout"]["layout_class"] == layout),
                    key=lambda c: c["median_us"])[:3]
                tactics.append({"m": result["cell"]["m"], "shortlist": [
                    {"config": c["config"], "median_us": c["median_us"]} for c in shortlist]})
            exemplar = per_cell[0][1][layout]["layout"]
            scored.append({"layout_class": layout, "layout_descriptor": exemplar,
                           "max_regret": max(regrets),
                           "mean_regret": statistics.mean(regrets), "per_m_regret": regrets,
                           "warmup_shortlist_within_layout": tactics})
        scored.sort(key=lambda x: (x["max_regret"], x["mean_regret"], x["layout_class"]))
        diagnostic = scored[0] if scored else None
        if timing_unresolved:
            disposition = "TIMING/UNRESOLVED"
        elif conflict:
            disposition = "CONFLICT/UNRESOLVED"
        elif diagnostic:
            disposition = "RESOLVED"
        else:
            disposition = "NO-COMMON-LAYOUT/UNRESOLVED"
        decisions.append({
            "tensor": tensor, "qtype": qtype, "n": n, "k": k,
            "m_values": [r["cell"]["m"] for r in cells],
            "policy": "equal-weight across M; minimize worst per-cell median regret, then mean regret",
            "cell_winner_layouts": leader_classes,
            "disposition": disposition,
            "minimax_diagnostic": diagnostic,
            "all_common_layout_scores": scored,
            "artifact_selection": (diagnostic["layout_class"] if disposition == "RESOLVED" else None),
        })
    return decisions


def measure(plan_path: pathlib.Path, binary: pathlib.Path, out: pathlib.Path,
            repeats: int, timeout: int, peak_tflops: float, hbm_gbs: float) -> int:
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"{plan_path}: not a {PLAN_SCHEMA} plan")
    admitted, admission = plan_admission(plan)
    if not admitted:
        raise ValueError(admission)
    if repeats < 2:
        raise ValueError("repeats must be >=2; one averaged pass cannot establish a ranking band")
    if out.exists():
        raise ValueError(f"refusing to overwrite result directory {out}")
    out.mkdir(parents=True)
    raw_dir = out / "raw"
    raw_dir.mkdir()
    all_results, raw_records = [], []
    any_failure = False
    for cell in plan["cells"]:
        if cell["support"]["state"] != "SUPPORTED":
            all_results.append({"cell": cell, "disposition": "UNSUPPORTED", "reason": cell["support"]["reason"]})
            any_failure = True
            continue
        family = cell["support"]["family"]
        expected = int(cell["support"]["source_denominator"])
        repeat_rows, failure = [], None
        for repeat in range(repeats):
            command = [str(binary), str(cell["m"]), str(cell["n"]), str(cell["k"]), str(cell["group_size"])]
            try:
                proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, timeout=timeout, check=False)
                text = proc.stdout
            except subprocess.TimeoutExpired as e:
                text = (e.stdout or "") + "\nTIMEOUT\n"
                proc = None
            raw_path = raw_dir / f"{cell['id']}.r{repeat}.log"
            raw_path.write_text(text)
            if proc is None:
                failure = f"repeat {repeat} timed out after {timeout}s"
                break
            if proc.returncode:
                failure = f"repeat {repeat} returned rc={proc.returncode}"
                break
            rows = parse_candidates(text, family)
            try:
                runtime_expected = runtime_contract(text, cell, rows)
            except ValueError as e:
                failure = f"repeat {repeat}: {e}"
                break
            if runtime_expected != expected:
                failure = f"repeat {repeat} emitted denominator {runtime_expected}/{expected} {family} rows"
                break
            if len({r["config"] for r in rows}) != expected:
                failure = f"repeat {repeat} has duplicate {family} config identities"
                break
            for row in rows:
                row["layout"] = candidate_layout(row, cell)
                raw_records.append({"schema": RESULT_SCHEMA, "cell_id": cell["id"], "repeat": repeat, **row})
            repeat_rows.append(rows)
        if failure:
            all_results.append({"cell": cell, "disposition": "FAIL", "reason": failure})
            any_failure = True
            continue
        identities = [{r["config"] for r in rows} for rows in repeat_rows]
        if any(ids != identities[0] for ids in identities[1:]):
            all_results.append({"cell": cell, "disposition": "FAIL", "reason": "candidate set changed between repeats"})
            any_failure = True
            continue
        by_config = {}
        representative = {}
        for rows in repeat_rows:
            for row in rows:
                by_config.setdefault(row["config"], []).append(row["us"])
                representative[row["config"]] = row
        stats, excluded = [], []
        for config, samples in by_config.items():
            row = representative[config]
            measured = {"config": config, "median_us": statistics.median(samples),
                        "band_us": [min(samples), max(samples)], "samples_us": samples,
                        "tactic": {k: row[k] for k in ("tm", "tn", "tk", "wm", "wn", "st")},
                        "layout": row["layout"]}
            (stats if row["layout"]["selection_eligible"] else excluded).append(measured)
        if not stats:
            all_results.append({"cell": cell, "disposition": "FAIL",
                                "reason": "every emitted candidate is semantically ineligible",
                                "candidate_denominator": expected,
                                "candidate_excluded": excluded})
            any_failure = True
            continue
        stats.sort(key=lambda x: x["median_us"])
        leader = stats[0]
        ties = [s for s in stats[1:] if s["band_us"][0] <= leader["band_us"][1]]
        leader["metrics"] = traffic(cell, leader["median_us"], peak_tflops, hbm_gbs)
        all_results.append({
            "cell": cell,
            "disposition": "UNRESOLVED" if ties else "RESOLVED",
            "leader": leader,
            "runner_up": stats[1] if len(stats) > 1 else None,
            "ties": ties,
            "candidate_denominator": expected,
            "candidate_measured": len(stats),
            "candidate_excluded": excluded,
            "candidate_excluded_count": len(excluded),
            "repeats": repeats,
            "candidates": stats,
        })

    (out / "samples.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in raw_records))
    decisions = layout_decisions(all_results)
    summary = {
        "schema": RESULT_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "plan": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "q8_candidate_tuple_sha256": plan["q8_candidate_manifest"]["tuple_sha256"],
        "binary": str(binary.resolve()),
        "binary_sha256": sha256_file(binary),
        "overall": "INCOMPLETE" if any_failure else "COMPLETE",
        "results": all_results,
        "layout_decisions": decisions,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    offline_plan = {
        "schema": "quactlize-prefill-offline-layout-plan-v1",
        "source_summary": str((out / "summary.json").resolve()),
        "selection_rule": "one resident layout per tensor across all measured M values",
        "decisions": decisions,
        "missing_comparison_arms": [
            "ScaleFirst metadata prepass",
            "Q8_0 GGUF block to resident A32 code/scale split-reorder producer",
            "direct FullyQuantized prefill GEMM",
        ],
    }
    (out / "offline_layout_plan.json").write_text(json.dumps(offline_plan, indent=2, sort_keys=True) + "\n")
    lines = ["cell\tformat\tM\tN\tK\tdisposition\tleader\tmedian_us\tMFU_pct\t"
             "distinct_MBU_model_pct\tArtifactTileK\tFoldN_low\tFoldN_high\tlayout\tdenominator"]
    for result in all_results:
        cell = result["cell"]
        leader = result.get("leader")
        if leader:
            metrics = leader["metrics"]
            row = [cell["id"], cell["format"], cell["m"], cell["n"], cell["k"], result["disposition"],
                   leader["config"], f"{leader['median_us']:.6f}", f"{metrics['mfu_percent']:.3f}",
                   f"{metrics['mbu_percent']:.3f}", leader["layout"]["artifact_tile_k"],
                   leader["layout"]["fold_n"][0], leader["layout"]["fold_n"][1],
                   leader["layout"]["layout_class"],
                   f"{result['candidate_measured']} eligible + {result['candidate_excluded_count']} excluded / "
                   f"{result['candidate_denominator']} emitted"]
        else:
            row = [cell["id"], cell["format"], cell["m"], cell["n"], cell["k"], result["disposition"],
                   "", "", "", "", "", "", "", "", "0/0"]
        lines.append("\t".join(map(str, row)))
    (out / "summary.tsv").write_text("\n".join(lines) + "\n")
    layout_lines = ["tensor\tqtype\tN\tK\tdisposition\tpolicy\tminimax_layout\tmax_regret"]
    for decision in decisions:
        diagnostic = decision["minimax_diagnostic"]
        layout_lines.append("\t".join(map(str, [
            decision["tensor"], decision["qtype"], decision["n"], decision["k"],
            decision["disposition"], decision["policy"],
            diagnostic["layout_class"] if diagnostic else "",
            f"{diagnostic['max_regret']:.6f}" if diagnostic else "",
        ])))
    (out / "layout-decisions.tsv").write_text("\n".join(layout_lines) + "\n")
    print("\n".join(lines))
    print("\n".join(layout_lines))
    print(f"[prefill-sweep] overall={summary['overall']} results={out / 'summary.json'}")
    return 1 if any_failure else 0


def write_synthetic_gguf(path: pathlib.Path, tensors: list[tuple[str, list[int], int]] | None = None) -> None:
    """A tiny v3 header used by --self-test; contains no tensor payload."""
    tensors = tensors or [
        # Hybrid control: block 0 is Gated DeltaNet and must not be mistaken
        # for a full-attention q projection merely because its shape is close.
        ("blk.0.attn_qkv.weight", [2048, 8192], 12),
        ("blk.0.attn_gate.weight", [2048, 4096], 12),
        # First full-attention layer in the real Qwen3.5 checkpoint.
        ("blk.3.attn_q.weight", [2048, 8192], 12),
        ("blk.3.attn_k.weight", [2048, 512], 10),
        ("blk.3.attn_v.weight", [2048, 512], 13),
        ("blk.3.attn_output.weight", [4096, 2048], 14),
    ]
    with path.open("wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 0))
        for name, dims, qtype in tensors:
            encoded = name.encode()
            f.write(struct.pack("<Q", len(encoded)) + encoded)
            f.write(struct.pack("<I", len(dims)))
            for dim in dims:
                f.write(struct.pack("<Q", dim))
            f.write(struct.pack("<IQ", qtype, 0))


def self_test() -> int:
    # Keep even ephemeral test artifacts under /workspace: box operators use
    # that mount for quota/cleanup and explicitly do not want hidden /tmp
    # trees.  The PID makes mkdir fail closed on an impossible collision.
    td = pathlib.Path(f"/workspace/quactlize-prefill-selftest-{os.getpid()}")
    if td.exists():
        raise ValueError(f"self-test directory already exists: {td}")
    td.mkdir(parents=True)
    try:
        gguf = td / "model.gguf"
        write_synthetic_gguf(gguf)
        plan = build_plan(ROOT / "benchmarks" / "prefill_qwen35_a3b_smoke.json", gguf)
        assert len(plan["cells"]) == 8
        assert [c["qtype"] for c in plan["cells"][::2]] == [12, 10, 13, 14]
        assert all(c["support"]["state"] == "SUPPORTED" for c in plan["cells"])
        assert plan["support_summary"]["supported_cells"] == 8
        assert plan["support_summary"]["unsupported_cells"] == 0
        assert plan_admission(plan)[0]
        assert plan["cells"][0]["qtype_source"] == "GGUF-tensor-header"
        assert plan["tensor_selection"]["selected_layer"] == 3
        assert all(c["selected_layer"] == 3 for c in plan["cells"])
        assert plan["cells"][0]["tensor"] == "blk.3.attn_q.weight"
        assert (plan["cells"][0]["n"], plan["cells"][0]["k"]) == (8192, 2048)
        sample = "    q4 64x64:64 w32x32 s3 [F=1]   12.50 us | 1.0 TFLOP/s\n"
        rows = parse_candidates(sample, "q4")
        assert len(rows) == 1 and rows[0]["tk"] == 64
        layout = candidate_layout(rows[0], plan["cells"][0])
        assert layout["fold_n"] == [1, 1] and layout["artifact_tile_k"] == 64
        assert layout["selection_eligible"]
        # Negative: a one-bit fold typo must not survive as a plausible result descriptor.
        rows[0]["annotation"] = "F=2"
        try:
            candidate_layout(rows[0], plan["cells"][0])
        except ValueError:
            pass
        else:
            raise AssertionError("wrong printed FoldN was accepted")
        d = source_denominators()
        footer = (f"  PREFILL_ROW_DENOMINATOR q2={d['i2']} q3={d['BC']} q4={d['q4']} "
                  f"q5={d['q5']} q6={d['q6']} q8={d['q8']}\n"
                  "  PREFILL_LAUNCH_STATUS failures=0 verdict=PASS\n")
        q4_cell = plan["cells"][0]
        assert runtime_contract(footer, q4_cell, d["q4"]) == d["q4"]
        # Negative: dropping one parsed row must fail rather than shrink the
        # denominator and publish a winner over a subset.
        try:
            runtime_contract(footer, q4_cell, d["q4"] - 1)
        except ValueError:
            pass
        else:
            raise AssertionError("a missing candidate row did not fail closed")
        # A legacy Q3 BC row is still part of the executed denominator, but
        # its independently packed high plane is a known-invalid producer map
        # and must never enter selection.
        q3_cell = {
            **plan["cells"][2], "qtype": 11, "planes": [2, 1],
            "support": {"state": "SUPPORTED", "family": "BC", "source_denominator": d["BC"]},
            "canonical_scale_first_arrangement": {"artifact_tile_k": 256, "fold_n": [1, 1]},
        }
        legacy_q3 = parse_candidates("    BC 64x64:256 w32x32 s3 20.0 us |\n", "BC")[0]
        legacy_layout = candidate_layout(legacy_q3, q3_cell)
        assert not legacy_layout["selection_eligible"]
        assert "high-plane" in legacy_layout["selection_exclusion"]
        # Cross-M winners that require different resident layouts must not
        # silently publish two artifacts.  Minimax is printed as a diagnostic,
        # while artifact_selection remains null.
        def cand(name, layout, us):
            return {"config": name, "median_us": us,
                    "layout": {"layout_class": layout}}
        base = {"tensor": "blk.3.attn_q.weight", "qtype": 12, "n": 8192, "k": 2048}
        a64, b64 = cand("a64", "layout-A", 10.0), cand("b64", "layout-B", 11.0)
        a2k, b2k = cand("a2k", "layout-A", 12.0), cand("b2k", "layout-B", 10.0)
        planted = [
            {"cell": {**base, "m": 64}, "disposition": "RESOLVED", "leader": a64,
             "candidates": [a64, b64]},
            {"cell": {**base, "m": 2048}, "disposition": "RESOLVED", "leader": b2k,
             "candidates": [a2k, b2k]},
        ]
        decisions = layout_decisions(planted)
        assert decisions[0]["disposition"] == "CONFLICT/UNRESOLVED"
        assert decisions[0]["artifact_selection"] is None
        assert decisions[0]["minimax_diagnostic"] is not None

        # Q8_0 is a controlled ScaleFirst-only positive route.  It is not
        # inserted into the K-quant/FullyQuantized registry: support rests on
        # L208's vendor-converter emission anchor and A32 xplane roundtrip.
        q8_gguf = td / "all-q8.gguf"
        write_synthetic_gguf(q8_gguf, [
            ("blk.3.attn_q.weight", [2048, 8192], 8),
            ("blk.3.attn_k.weight", [2048, 512], 8),
            ("blk.3.attn_v.weight", [2048, 512], 8),
            ("blk.3.attn_output.weight", [4096, 2048], 8),
        ])
        q8_plan = build_plan(ROOT / "benchmarks" / "prefill_qwen35_a3b_smoke.json", q8_gguf)
        assert len(q8_plan["cells"]) == 8
        assert all(c["format"] == "Q8_0" for c in q8_plan["cells"])
        assert all(c["support"]["state"] == "SUPPORTED" for c in q8_plan["cells"])
        assert all(c["support"]["reason_code"] == "CONTROLLED_SCALEFIRST_ROW_FAMILY"
                   for c in q8_plan["cells"])
        assert all(not c["registry_backed"] and
                   c["route_scope"] == "CONTROLLED_RESIDENT_SCALEFIRST_GEMM_ONLY"
                   for c in q8_plan["cells"])
        assert all(c["quant_mode"] == "FinegrainedScaleOnly" and c["metadata_fp16_planes"] == 1
                   for c in q8_plan["cells"])
        assert all(c["canonical_scale_first_arrangement"] == {"artifact_tile_k": 32, "fold_n": [1, 1]}
                   for c in q8_plan["cells"])
        assert q8_plan["support_summary"] == {
            "total_cells": 8,
            "supported_cells": 8,
            "unsupported_cells": 0,
            "by_state": {"SUPPORTED": 8},
            "by_reason_code": {"CONTROLLED_SCALEFIRST_ROW_FAMILY": 8},
        }
        admitted, reason = plan_admission(q8_plan)
        assert admitted and reason.startswith("SUPPORTED_CELLS_PRESENT:")
        tampered_hash_plan = json.loads(json.dumps(q8_plan))
        tampered_hash_plan["source_authorities"][1]["sha256"] = "0" * 64
        try:
            plan_admission(tampered_hash_plan)
        except ValueError as e:
            assert "source authority changed after plan" in str(e)
        else:
            raise AssertionError("tampered Q8 source hash was admitted")
        tampered_manifest_plan = json.loads(json.dumps(q8_plan))
        tampered_manifest_plan["q8_candidate_manifest"]["tuples_tm_tn_tk_wm_wn_stages"][0][0] += 1
        try:
            plan_admission(tampered_manifest_plan)
        except ValueError as e:
            assert "candidate manifest changed after plan" in str(e)
        else:
            raise AssertionError("tampered Q8 tuple manifest was admitted")
        q8_row = parse_candidates(
            "    q8 64x64:32 w32x32 s2 [A=32 F=1] 10.0 us | 1.0 TFLOP/s\n", "q8")[0]
        q8_layout = candidate_layout(q8_row, q8_plan["cells"][0])
        assert q8_layout["artifact_tile_k"] == 32 and q8_layout["fold_n"] == [1, 1]
        assert q8_layout["scale_first_contract_match"] and not q8_layout["shipping_registry_match"]
        q8_metrics = traffic(q8_plan["cells"][0], 10.0, 500.0, 2766.0)
        assert q8_metrics["metadata_planes"] == 1
        assert q8_metrics["byte_breakdown"]["fp16_metadata"] == 8192 * (2048 // 32) * 2
        q8_runtime_rows = [
            {"tm": tm, "tn": tn, "tk": tk, "wm": wm, "wn": wn, "st": st}
            for tm, tn, tk, wm, wn, st in q8_candidate_tuples()
        ]
        q8_expected_outputs = int(q8_plan["cells"][0]["m"]) * int(q8_plan["cells"][0]["n"])
        q8_correctness_text = "".join(
            f"  Q8_CORRECTNESS config={tm}x{tn}:{tk}_w{wm}x{wn}_s{st} "
            f"bad=0/{q8_expected_outputs} fixture=ORDER-INDEPENDENT+FP16-EXACT verdict=PASS\n"
            for tm, tn, tk, wm, wn, st in q8_candidate_tuples())
        q8_fixture_text = (
            f"  Q8_FIXTURE shape={q8_plan['cells'][0]['m']}x{q8_plan['cells'][0]['n']}x"
            f"{q8_plan['cells'][0]['k']} selected_k_per_row=4 unique_bad=0 fp16_exact_bad=0 "
            "scale_values=3 fixture=ORDER-INDEPENDENT+FP16-EXACT verdict=PASS\n")
        q8_footer = q8_fixture_text + q8_correctness_text + footer
        assert runtime_contract(q8_footer, q8_plan["cells"][0], q8_runtime_rows) == d["q8"]
        q8_reordered = list(q8_runtime_rows)
        q8_reordered[0], q8_reordered[1] = q8_reordered[1], q8_reordered[0]
        try:
            runtime_contract(q8_footer, q8_plan["cells"][0], q8_reordered)
        except ValueError as e:
            assert "tuple sequence differs" in str(e) and "same_members=True" in str(e)
        else:
            raise AssertionError("reordered Q8 runtime tuples were accepted")
        try:
            runtime_contract(footer, q8_plan["cells"][0], q8_runtime_rows)
        except ValueError as e:
            assert "correctness witness sequence differs" in str(e)
        else:
            raise AssertionError("Q8 rows without numerical witnesses were accepted")
        for planted in (
                q8_fixture_text + q8_correctness_text.replace(
                    f"bad=0/{q8_expected_outputs}", "bad=0/0", 1) + footer,
                q8_fixture_text + q8_correctness_text.replace(
                    "fixture=ORDER-INDEPENDENT+FP16-EXACT", "fixture=ROUNDS-UNKNOWN", 1) + footer):
            try:
                runtime_contract(planted, q8_plan["cells"][0], q8_runtime_rows)
            except ValueError as e:
                assert "correctness witness is red" in str(e)
            else:
                raise AssertionError("Q8 zero-denominator or wrong-fixture witness was accepted")
        try:
            runtime_contract(q8_correctness_text + footer, q8_plan["cells"][0], q8_runtime_rows)
        except ValueError as e:
            assert "fixture identity/exactness" in str(e)
        else:
            raise AssertionError("Q8 rows without an invocation-bound exact fixture were accepted")
        # Q8's A32 token is an ABI assertion, not decoration.  Missing,
        # conflicting, duplicate, or extra tokens must all fail closed.
        for bad_annotation in ("", "A=64 F=1", "A=32 F=2",
                               "A=32 A=32 F=1", "A=32 F=1 X=7"):
            bad_q8 = dict(q8_row, annotation=bad_annotation,
                          config=f"q8-negative-[{bad_annotation}]")
            try:
                candidate_layout(bad_q8, q8_plan["cells"][0])
            except ValueError:
                pass
            else:
                raise AssertionError(f"bad Q8 layout annotation was accepted: {bad_annotation!r}")

        # The pre-build fail-close remains independently live.  Use Q5_1,
        # which is a recognized GGUF identity but has neither this controlled
        # Q8 route nor a registered K-quant row.
        unsupported_gguf = td / "all-unsupported.gguf"
        write_synthetic_gguf(unsupported_gguf, [
            ("blk.3.attn_q.weight", [2048, 8192], 7),
            ("blk.3.attn_k.weight", [2048, 512], 7),
            ("blk.3.attn_v.weight", [2048, 512], 7),
            ("blk.3.attn_output.weight", [4096, 2048], 7),
        ])
        unsupported_plan = build_plan(
            ROOT / "benchmarks" / "prefill_qwen35_a3b_smoke.json", unsupported_gguf)
        assert all(c["format"] == "Q5_1" and c["support"]["state"] == "UNSUPPORTED"
                   for c in unsupported_plan["cells"])
        admitted, reason = plan_admission(unsupported_plan)
        assert not admitted and reason.startswith("NO_SUPPORTED_CELLS:")
        unsupported_plan_path = td / "all-unsupported-plan.json"
        unsupported_plan_path.write_text(json.dumps(unsupported_plan, sort_keys=True) + "\n")
        admission_proc = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()), "admit", "--plan", str(unsupported_plan_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        assert admission_proc.returncode == 3
        assert "NO_SUPPORTED_CELLS" in admission_proc.stdout
        blocked_results = td / "must-not-be-created"
        try:
            measure(unsupported_plan_path, td / "must-not-be-executed", blocked_results,
                    repeats=2, timeout=1, peak_tflops=500.0, hbm_gbs=2766.0)
        except ValueError as e:
            assert "NO_SUPPORTED_CELLS" in str(e)
        else:
            raise AssertionError("an all-unsupported plan reached measurement")
        assert not blocked_results.exists()
    finally:
        shutil.rmtree(td)
    print("[prefill-sweep:self-test] PASS: hybrid blk0-GDN/blk3-attention selection, GGUF qtype/dim authority, "
          "q4 semantic tag, FoldN negative, Q3 legacy exclusion, denominator fail-close, cross-M conflict, "
          "controlled Q8_0 ScaleFirst admission, and all-unsupported pre-build admission")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--spec", type=pathlib.Path, required=True)
    p.add_argument("--gguf", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p = sub.add_parser("measure")
    p.add_argument("--plan", type=pathlib.Path, required=True)
    p.add_argument("--bin", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--peak-tflops", type=float, default=500.0)
    p.add_argument("--hbm-gbs", type=float, default=2766.0)
    p = sub.add_parser("admit")
    p.add_argument("--plan", type=pathlib.Path, required=True)
    sub.add_parser("self-test")
    a = ap.parse_args()
    try:
        if a.command == "self-test":
            return self_test()
        if a.command == "plan":
            plan = build_plan(a.spec, a.gguf)
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
            for cell in plan["cells"]:
                print(f"{cell['id']:<8} {cell['tensor']:<34} {cell['format']:<5} "
                      f"M/N/K={cell['m']}/{cell['n']}/{cell['k']} "
                      f"fold={cell['canonical_scale_first_arrangement']['fold_n'] if cell['canonical_scale_first_arrangement'] else '-'} "
                      f"{cell['support']['state']} reason={cell['support']['reason_code']}:"
                      f"{cell['support']['reason']}")
            summary = plan["support_summary"]
            print(f"[prefill-sweep] plan={a.output} cells={summary['total_cells']} "
                  f"supported={summary['supported_cells']} unsupported={summary['unsupported_cells']}")
            return 0
        if a.command == "admit":
            plan = json.loads(a.plan.read_text())
            admitted, reason = plan_admission(plan)
            print(f"[prefill-sweep] {reason}")
            return 0 if admitted else 3
        return measure(a.plan, a.bin, a.out, a.repeats, a.timeout, a.peak_tflops, a.hbm_gbs)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"[prefill-sweep] FAIL: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
