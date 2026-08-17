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

The timing scope is ScaleFirst GEMM only.  It does NOT include the metadata
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

PLAN_SCHEMA = "quactlize-prefill-plan-v1"
RESULT_SCHEMA = "quactlize-prefill-scale-first-result-v1"
SUPPORTED_SPEC_SCHEMA = "quactlize-prefill-smoke-v1"
BENCH_SOURCE = ROOT / "benchmarks" / "test_scalefirst_bench.cu"
REGISTRY_SOURCE = ROOT / "quactlize" / "include" / "ppu_format_config.inc"

# This maps a GGUF semantic format to an explicitly tagged row family.  In
# particular Q4_K maps to q4, never to the ScaleOnly i4 ceiling.
FAMILY_BY_QTYPE = {10: "i2", 11: "BC", 12: "q4", 13: "q5", 14: "q6"}
DENOMINATOR_KEY_BY_QTYPE = {10: "q2", 11: "q3", 12: "q4", 13: "q5", 14: "q6"}

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


def source_denominators(source: pathlib.Path = BENCH_SOURCE) -> dict[str, int]:
    """Count calls, not comments or macro definitions, in the one measured source."""
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
    return counts


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    cells = []
    for projection in projections:
        pattern = projection.get("tensor_pattern", "")
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"bad tensor_pattern {pattern!r}: {e}") from e
        matches = [t for t in headers if rx.fullmatch(t["name"])]
        if len(matches) != 1:
            names = [t["name"] for t in matches[:8]]
            raise ValueError(f"{projection.get('name')}: pattern {pattern!r} matched {len(matches)} tensors: {names}")
        tensor = matches[0]
        if len(tensor["dims"]) != 2:
            raise ValueError(f"{tensor['name']}: expected a 2-D weight, got dims={tensor['dims']}")
        # GGUF dimension order is K,N for llama.cpp dense weights.
        actual_n, actual_k = int(tensor["dims"][1]), int(tensor["dims"][0])
        expected = int(projection["n"]), int(projection["k"])
        if (actual_n, actual_k) != expected:
            raise ValueError(
                f"{tensor['name']}: GGUF says N,K={actual_n},{actual_k}; spec says {expected}. "
                "The checkpoint header wins; update the geometry spec before measuring.")
        qtype = int(tensor["qtype"])
        row = registry.get(qtype)
        family = FAMILY_BY_QTYPE.get(qtype)
        if row is None:
            support = {"state": "UNSUPPORTED", "reason": f"GGUF qtype {qtype} is absent from the PPU format registry"}
        elif family is None:
            support = {"state": "UNSUPPORTED", "reason": f"{row['name']} has no semantic row family in test_scalefirst_bench"}
        elif denominators.get(family, 0) == 0:
            support = {"state": "UNSUPPORTED", "reason": f"row family {family} has an empty source denominator"}
        else:
            support = {"state": "SUPPORTED", "family": family,
                       "source_denominator": denominators[family]}
        for m in m_values:
            cell = {
                "id": f"{projection['name']}-m{m}",
                "projection": projection["name"],
                "tensor": tensor["name"],
                "qtype_source": "GGUF-tensor-header",
                "qtype": qtype,
                "format": row["name"] if row else f"qtype-{qtype}",
                "m": m, "n": actual_n, "k": actual_k,
                "group_size": row["group_size"] if row else None,
                "planes": [row["low_bits"], row["high_bits"]] if row else None,
                "canonical_scale_first_arrangement": ({
                    "artifact_tile_k": row["scale_first_tile_k"],
                    "fold_n": [fold_for(row["low_bits"], row["scale_first_tile_k"]),
                               fold_for(row["high_bits"], row["scale_first_tile_k"])
                               if row["high_bits"] else 1],
                } if row else None),
                "support": support,
            }
            cells.append(cell)
    return {
        "schema": PLAN_SCHEMA,
        "model": spec["model"],
        "scope": spec.get("scope", ""),
        "gguf": str(gguf_path.resolve()),
        "gguf_size": gguf_path.stat().st_size,
        "spec": str(spec_path.resolve()),
        "registry": str(REGISTRY_SOURCE.resolve()),
        "registry_sha256": sha256_file(REGISTRY_SOURCE),
        "candidate_source": str(BENCH_SOURCE.resolve()),
        "candidate_source_sha256": sha256_file(BENCH_SOURCE),
        "candidate_denominators": denominators,
        "timing_scope": "ScaleFirst GEMM only; metadata prepass and direct FullyQuantized GEMM excluded",
        "denominator_scope": "finite manual row families in test_scalefirst_bench; not the generated full tactic space",
        "cells": cells,
    }


CANDIDATE_RE = re.compile(
    r"^\s*(?P<family>BC|i2|q4|q5|q6)\s+"
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
    r"^\s*PREFILL_ROW_DENOMINATOR\s+q2=(\d+)\s+q3=(\d+)\s+q4=(\d+)\s+q5=(\d+)\s+q6=(\d+)\s*$",
    re.M,
)
LAUNCH_STATUS_RE = re.compile(
    r"^\s*PREFILL_LAUNCH_STATUS\s+failures=(\d+)\s+verdict=(PASS|FAIL)\s*$", re.M)


def runtime_contract(text: str, cell: dict, parsed_rows: int) -> int:
    """Return the runtime denominator, requiring the bench's fail-closed footer exactly once."""
    denoms = DENOMINATOR_RE.findall(text)
    if len(denoms) != 1:
        raise ValueError(f"expected one PREFILL_ROW_DENOMINATOR, found {len(denoms)}")
    statuses = LAUNCH_STATUS_RE.findall(text)
    if statuses != [("0", "PASS")]:
        raise ValueError(f"launch status is not exactly failures=0/PASS: {statuses}")
    keys = ("q2", "q3", "q4", "q5", "q6")
    values = dict(zip(keys, map(int, denoms[0])))
    key = DENOMINATOR_KEY_BY_QTYPE[int(cell["qtype"])]
    denominator = values[key]
    if parsed_rows != denominator:
        raise ValueError(f"parsed {parsed_rows} {key} rows but runtime denominator is {denominator}")
    source_expected = int(cell["support"]["source_denominator"])
    if denominator != source_expected:
        raise ValueError(
            f"runtime denominator {denominator} disagrees with source-call authority {source_expected} for {key}")
    return denominator


def candidate_layout(candidate: dict, cell: dict) -> dict:
    low_bits, high_bits = cell["planes"]
    family, annotation = cell["support"]["family"], candidate["annotation"]
    # The old non-folded I2/BC macros consume the legacy interleave-256 buffer.
    # Every explicitly folded row and every q4/q5/q6 row calls xplane with TK.
    legacy = family in ("i2", "BC") and not annotation
    artifact_tk = 256 if legacy else candidate["tk"]
    folds = [fold_for(low_bits, artifact_tk), fold_for(high_bits, artifact_tk) if high_bits else 1]

    # The source prints the folds it instantiated.  A parser/layout mismatch is
    # a failed result, not a plausible descriptor attached to different bytes.
    printed = {}
    for key, value in re.findall(r"(F|F1|F2)=(\d+)", annotation):
        printed[key] = int(value)
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
    if folds == [1, 1] and artifact_tk <= 256:
        layout_class = f"xplane-tile-free-f1-le256:bits={low_bits}+{high_bits}"
        class_basis = "dev/fold_derivation/l115_artifact_tactic_code_slots.cu verified tile-free F1<=256"
        resident_artifact_tk = canonical["artifact_tile_k"]
    else:
        if family == "BC":
            producer_map = "place_derived+place_int1"
        elif family in ("q5", "q6"):
            producer_map = "place_derived+place_hi"
        elif family in ("i2", "q4"):
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
        "shipping_registry_match": resident_artifact_tk == canonical["artifact_tile_k"] and folds == canonical["fold_n"],
        "shipping_registry_exact_measured_map": artifact_tk == canonical["artifact_tile_k"] and folds == canonical["fold_n"],
        "selection_eligible": eligible,
        "selection_exclusion": ("known-wrong Q3 high-plane single-plane producer map; "
                                "BCF/place_int1 is the semantic authority"
                                if not eligible else None),
    }


def traffic(cell: dict, us: float, peak_tflops: float, hbm_gbs: float) -> dict:
    m, n, k, gs = (int(cell[x]) for x in ("m", "n", "k", "group_size"))
    low_bits, high_bits = cell["planes"]
    code = n * k * (low_bits + high_bits) // 8
    # The measured ScaleZero path materialises fp16 scale and fp16 zero.
    metadata = n * (k // gs) * 4
    act, out = m * k * 2, m * n * 2
    distinct = code + metadata + act + out
    tflops = 2.0 * m * n * k / us / 1.0e6
    gbs = distinct / us / 1000.0
    return {
        "flops": 2 * m * n * k,
        "tflops": tflops,
        "mfu_percent": 100.0 * tflops / peak_tflops,
        "distinct_bytes": distinct,
        "byte_breakdown": {"codes": code, "fp16_scale_zero": metadata, "activation": act, "output": out},
        "distinct_gbs": gbs,
        "mbu_percent": 100.0 * gbs / hbm_gbs,
        "peak_tflops": peak_tflops,
        "hbm_gbs": hbm_gbs,
        "traffic_scope": "one distinct ScaleFirst resident weight + fp16 A/D; cache reuse not multiplied by tiles",
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
                runtime_expected = runtime_contract(text, cell, len(rows))
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
        "missing_comparison_arms": ["ScaleFirst metadata prepass", "direct FullyQuantized prefill GEMM"],
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


def write_synthetic_gguf(path: pathlib.Path) -> None:
    """A tiny v3 header used by --self-test; contains no tensor payload."""
    tensors = [
        ("blk.0.attn_q.weight", [2048, 4096], 12),
        ("blk.0.attn_k.weight", [2048, 512], 10),
        ("blk.0.attn_v.weight", [2048, 512], 13),
        ("blk.0.attn_output.weight", [4096, 2048], 14),
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
        assert plan["cells"][0]["qtype_source"] == "GGUF-tensor-header"
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
                  f"q5={d['q5']} q6={d['q6']}\n"
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
        base = {"tensor": "blk.0.attn_q.weight", "qtype": 12, "n": 4096, "k": 2048}
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
    finally:
        shutil.rmtree(td)
    print("[prefill-sweep:self-test] PASS: GGUF qtype/dim authority, q4 semantic tag, FoldN negative, "
          "Q3 legacy exclusion, denominator fail-close, and cross-M conflict")
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
                      f"{cell['support']['state']}")
            print(f"[prefill-sweep] plan={a.output} cells={len(plan['cells'])}")
            return 0
        return measure(a.plan, a.bin, a.out, a.repeats, a.timeout, a.peak_tflops, a.hbm_gbs)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"[prefill-sweep] FAIL: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
