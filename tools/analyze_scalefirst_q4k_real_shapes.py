#!/usr/bin/env python3
"""Turn one completed real-Q4_K ScaleFirst sweep into deployment evidence.

The device runner deliberately stops at per-shape/per-board winners.  This
tool answers the three questions that must be answered after, rather than
during, measurement:

* one offline layout per concrete model/TP/tensor across every measured M;
* which tactic axes can be conservatively pruned, and whether M alone can
  choose one configuration;
* a model/tensor registry which records only resolved choices and preserves
  unresolved candidates instead of silently picking one.

No kernel is built or run.  Every consumed member is checked against the
completed bundle.json before it contributes to a report.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
from typing import Any

import plan_scalefirst_q4k_real_shapes as planner

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from quactlize import formats as qformats


ANALYSIS_SCHEMA = "quactlize.scalefirst_q4k_real_shapes_analysis.v4"
BUNDLE_SCHEMA = "quactlize.scalefirst_q4k_real_shapes_bundle.v1"
OFFLINE_SCHEMA = "quactlize.scalefirst_q4k_offline_layout_decisions.v3"
HEURISTIC_SCHEMA = "quactlize.scalefirst_q4k_heuristic_evidence.v1"
REGISTRY_SCHEMA = "quactlize.scalefirst_q4k_winner_registry.v4"
AXES = ("tile_m", "tile_n", "tactic_tile_k", "warp_m", "warp_n",
        "stages", "bchunk")


class AnalysisError(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "INF" if value > 0 else "-INF" if value < 0 else "NAN"
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def atomic_json(path: pathlib.Path, value: Any) -> None:
    atomic_text(path, json.dumps(json_safe(value), indent=2, sort_keys=True,
                                 ensure_ascii=False) + "\n")


def atomic_tsv(path: pathlib.Path, rows: list[dict[str, Any]],
               fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t",
                                extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_object(path: pathlib.Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AnalysisError(f"required regular file is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AnalysisError(f"JSON root is not an object: {path}")
    return value


class Authority:
    def __init__(self, root: pathlib.Path):
        self.root = root.resolve(strict=True)
        self.bundle_path = self.root / "bundle.json"
        self.bundle = read_object(self.bundle_path)
        if self.bundle.get("schema") != BUNDLE_SCHEMA:
            raise AnalysisError("bundle schema differs or run did not complete")
        files = self.bundle.get("files")
        if not isinstance(files, dict) or not files:
            raise AnalysisError("bundle has no bound member census")
        self.files: dict[str, str] = files
        self.verified: set[str] = set()

    def path(self, relative: str) -> pathlib.Path:
        candidate = self.root / relative
        if pathlib.PurePosixPath(relative).is_absolute() or \
                ".." in pathlib.PurePosixPath(relative).parts:
            raise AnalysisError(f"unsafe bundle member {relative!r}")
        if relative not in self.files:
            raise AnalysisError(f"unbound bundle member requested: {relative}")
        if candidate.is_symlink() or not candidate.is_file():
            raise AnalysisError(f"bundle member is absent/non-regular: {relative}")
        if digest(candidate) != self.files[relative]:
            raise AnalysisError(f"bundle member hash differs: {relative}")
        self.verified.add(relative)
        return candidate

    def object(self, relative: str) -> dict[str, Any]:
        return read_object(self.path(relative))


def manifest_rows(authority: Authority, artifact: int
                  ) -> dict[str, dict[str, Any]]:
    relative = f"generated/q12-a{artifact}-bc0/manifest.json"
    manifest = authority.object(relative)
    identity = {"qtype": 12, "format": "Q4_K",
                "artifact_tile_k": artifact, "bchunk": 0}
    if manifest.get("identity") != identity:
        raise AnalysisError(f"A{artifact} manifest identity differs")
    rows = manifest.get("typed_rows")
    if not isinstance(rows, list) or not rows:
        raise AnalysisError(f"A{artifact} manifest has no typed rows")
    result: dict[str, dict[str, Any]] = {}
    configs: set[str] = set()
    for row in rows:
        symbol = row.get("symbol")
        if not isinstance(symbol, str) or symbol in result:
            raise AnalysisError(f"A{artifact} manifest symbol duplicate")
        if any(not isinstance(row.get(axis), int) for axis in AXES):
            raise AnalysisError(f"{symbol} lacks integer tactic axes")
        config = tactic_name(row)
        if config in configs:
            raise AnalysisError(f"A{artifact} duplicate tactic config {config}")
        configs.add(config)
        result[symbol] = row
    if manifest.get("denominator", {}).get("typed_rows") != len(result):
        raise AnalysisError(f"A{artifact} manifest denominator differs")
    return result


def tactic_name(row: dict[str, Any]) -> str:
    return (f"{row['tile_m']}x{row['tile_n']}x{row['tactic_tile_k']}_"
            f"w{row['warp_m']}x{row['warp_n']}_s{row['stages']}_"
            f"bc{row['bchunk']}")


def result_path(artifact: int, shape_key: str, name: str) -> str:
    return f"results/a{artifact}/{shape_key}/{name}"


def load_inputs(bundle: pathlib.Path) -> dict[str, Any]:
    authority = Authority(bundle)
    plan_path = authority.path("plan.json")
    plan = read_object(plan_path)
    planner.validate_plan(plan)
    summary = authority.object("summary.json")
    if summary.get("schema") != planner.SUMMARY_SCHEMA or \
            summary.get("plan_sha256") != digest(plan_path) or \
            summary.get("shape_count") != plan.get("shape_count") or \
            summary.get("cell_count") != plan.get("cell_count"):
        raise AnalysisError("root summary identity/denominator differs")
    manifests = {artifact: manifest_rows(authority, artifact)
                 for artifact in planner.ARTIFACTS}
    cell_summaries: dict[tuple[str, int], dict[str, Any]] = {}
    screens: dict[tuple[str, int], dict[str, Any]] = {}
    schedulers: dict[tuple[str, int], dict[str, Any]] = {}
    for cell in plan["cells"]:
        key = str(cell["shape_key"])
        artifact = int(cell["artifact_tile_k"])
        summary_rel = result_path(artifact, key, "summary.json")
        screen_rel = result_path(artifact, key, "screen.json")
        scheduler_rel = result_path(artifact, key, "scheduler.json")
        result = authority.object(summary_rel)
        screen = authority.object(screen_rel)
        scheduler = authority.object(scheduler_rel)
        if result.get("schema") != planner.RESULT_SCHEMA or \
                result.get("phase") != "CONFIRM" or \
                set(result.get("boards", {})) != set(planner.BOARDS):
            raise AnalysisError(f"malformed confirm result {summary_rel}")
        if screen.get("schema") != planner.RESULT_SCHEMA or \
                screen.get("phase") != "SCREEN":
            raise AnalysisError(f"malformed screen result {screen_rel}")
        if scheduler.get("schema") != planner.RESULT_SCHEMA or \
                scheduler.get("phase") != "SCHEDULER":
            raise AnalysisError(f"malformed scheduler result {scheduler_rel}")
        expected = cell.get("policy_sha256")
        if any(item.get("policy_sha256") != expected
               for item in (result, screen, scheduler)):
            raise AnalysisError(f"phase policy binding differs for {key}/A{artifact}")
        measured = screen.get("denominator", {}).get("measured")
        candidates = screen.get("selected", []) + screen.get("screened_out", [])
        if measured != len(candidates) or \
                len({row.get("symbol") for row in candidates}) != len(candidates):
            raise AnalysisError(f"screen candidate denominator differs for {key}/A{artifact}")
        if any(row.get("symbol") not in manifests[artifact]
               for row in candidates):
            raise AnalysisError(f"screen candidate outside manifest for {key}/A{artifact}")
        cell_summaries[(key, artifact)] = result
        screens[(key, artifact)] = screen
        schedulers[(key, artifact)] = scheduler
    if len(cell_summaries) != int(plan["cell_count"]):
        raise AnalysisError("loaded cell denominator differs")
    return {"authority": authority, "plan": plan, "summary": summary,
            "manifests": manifests, "cell_summaries": cell_summaries,
            "screens": screens, "schedulers": schedulers}


def measured_board(cell_summaries: dict[tuple[str, int], dict[str, Any]],
                   key: str, artifact: int, board: str
                   ) -> dict[str, Any] | None:
    result = cell_summaries[(key, artifact)]["boards"][board]
    winner = result.get("winner")
    if winner is None:
        return None
    return {"artifact_tile_k": artifact, "verdict": result["verdict"],
            "winner": winner, "runner_up": result.get("runner_up")}


def physical_layout_class(artifact: int) -> dict[str, Any]:
    """Canonical xplane byte identity, not a CuTe debug-print string.

    L105 groups complete stored buffers and L115 compares the complete logical
    owner -> physical slot map.  Together they prove one normalized F=1 class
    inside the interleave-256/A<=256 domain.  Folded arrangements stay bound
    to their exact producer descriptor unless an equally strong byte-map
    witness merges them.
    """
    if artifact not in planner.ARTIFACTS:
        raise AnalysisError(f"unregistered ArtifactTileK {artifact}")
    arrangement = qformats.PlacedArrangement(4, artifact, 0)
    tile_free = arrangement.layout_is_tile_free() and artifact <= 256
    if tile_free:
        name = "xplane-q4k-tile-free-f1-le256"
        identity = {
            "schema": "quactlize.xplane_canonical_mapping.v1",
            "producer": "xplane::place_derived",
            "logical_code_planes": [4],
            "fold_n": [1],
            "interleave_codes": 256,
            "equivalence_domain": {"artifact_tile_k": [64, 128, 256]},
        }
        basis = (
            "L105 exact stored-byte class plus L115 exact logical-owner/"
            "physical-slot parity for F=1, ArtifactTileK<=256")
    else:
        name = f"xplane-q4k-fold{arrangement.fold}-a{artifact}"
        identity = {
            "schema": "quactlize.xplane_canonical_mapping.v1",
            "producer": "xplane::place_derived",
            "logical_code_planes": [4],
            "artifact_tile_k": artifact,
            "fold_n": [arrangement.fold],
            "interleave_codes": 256,
        }
        basis = (
            "exact folded producer descriptor; no exact byte-map authority "
            "merges it with another descriptor")
    encoded = canonical(identity).encode("utf-8")
    return {
        "name": name,
        "mapping_sha256": hashlib.sha256(encoded).hexdigest(),
        "canonical_mapping_identity": identity,
        "cute_debug_string_role": "DIAGNOSTIC_ONLY_NOT_CANONICAL",
        "basis": basis,
    }


def regret_interval(candidate: dict[str, Any], all_candidates: list[dict[str, Any]]
                    ) -> tuple[float, float]:
    low = float(candidate["winner"]["range_us"][0])
    high = float(candidate["winner"]["range_us"][1])
    best_high = min(float(item["winner"]["range_us"][1])
                    for item in all_candidates)
    best_low = min(float(item["winner"]["range_us"][0])
                   for item in all_candidates)
    return max(0., low / best_high - 1.), max(0., high / best_low - 1.)


def ranked_decision(scored: list[dict[str, Any]]) -> tuple[
        str, dict[str, Any] | None, dict[str, Any] | None,
        dict[str, Any] | None]:
    """Return verdict, objective runner, and the interval blocker.

    `scored` is already ordered by the registered point objective.  Resolution
    is stronger than merely beating that objective runner: the selected upper
    envelope must be below every alternative lower envelope.  A noisy third
    place must therefore keep the decision unresolved.
    """
    if not scored:
        return "NO_COMMON_LAYOUT", None, None, None
    selected = scored[0]
    runner = scored[1] if len(scored) > 1 else None
    alternatives = scored[1:]
    blocker = (None if not alternatives else min(
        alternatives, key=lambda item: (
            float(item["max_regret_interval"][0]),
            float(item["max_regret"]), float(item["mean_regret"]))))
    verdict = ("RESOLVED" if blocker is None or
               float(selected["max_regret_interval"][1]) <
               float(blocker["max_regret_interval"][0]) else "UNRESOLVED")
    return verdict, selected, runner, blocker


def layer_identity(reference: dict[str, Any], tensor: str) -> dict[str, Any]:
    if not isinstance(tensor, str) or not tensor:
        raise AnalysisError("layer reference contains an empty tensor name")
    return {
        "model_id": str(reference["model_id"]),
        "tensor": tensor,
        "tp_world": int(reference["tp_world"]),
        "tp_rank": int(reference["tp_rank"]),
        "tp_partition": str(reference["tp_partition"]),
    }


def layer_key(layer: dict[str, Any]) -> tuple[str, int, int, str, str]:
    return (str(layer["model_id"]), int(layer["tp_world"]),
            int(layer["tp_rank"]), str(layer["tp_partition"]),
            str(layer["tensor"]))


def offline_layout_decisions(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    plan = inputs["plan"]
    cell_summaries = inputs["cell_summaries"]
    grouped: dict[tuple[str, int, int, str, str], list[dict[str, Any]]] = \
        collections.defaultdict(list)
    identities: dict[tuple[str, int, int, str, str], dict[str, Any]] = {}
    for shape in plan["shapes"]:
        seen_in_shape = set()
        references = shape.get("references")
        if not isinstance(references, list) or not references:
            raise AnalysisError(f"{shape.get('shape_key')} has no layer references")
        for reference in references:
            tensors = reference.get("source_tensors")
            if not isinstance(tensors, list) or not tensors:
                raise AnalysisError("layer reference has no source tensors")
            for tensor in tensors:
                identity = layer_identity(reference, tensor)
                key = layer_key(identity)
                if key in seen_in_shape:
                    raise AnalysisError(
                        f"layer {key} occurs twice in {shape['shape_key']}")
                seen_in_shape.add(key)
                prior = identities.setdefault(key, identity)
                if prior != identity:
                    raise AnalysisError(f"layer identity drifted for {key}")
                grouped[key].append(shape)
    decisions = []
    for identity_key, shapes in sorted(grouped.items()):
        shapes.sort(key=lambda item: int(item["m"]))
        layer = identities[identity_key]
        geometries = {(int(shape["n"]), int(shape["k"]),
                       int(shape["group_size"])) for shape in shapes}
        if len(geometries) != 1:
            raise AnalysisError(
                f"one layer changed N/K/group across M: "
                f"{identity_key} -> {geometries}")
        n, k, group_size = next(iter(geometries))
        m_values = [int(shape["m"]) for shape in shapes]
        if len(m_values) != len(set(m_values)):
            raise AnalysisError(
                f"one layer repeats a measured M: {identity_key}")
        per_shape: dict[str, list[dict[str, Any]]] = {}
        for shape in shapes:
            shape_key = str(shape["shape_key"])
            per_shape[shape_key] = [value for artifact in planner.ARTIFACTS
                              if (value := measured_board(
                                  cell_summaries, shape_key, artifact,
                                  "FULL_OUTPUT")) is not None]
            if not per_shape[shape_key]:
                raise AnalysisError(
                    f"FULL_OUTPUT has no layout for {shape_key}")
        scored = []
        for artifact in planner.ARTIFACTS:
            if any(not any(item["artifact_tile_k"] == artifact
                           for item in per_shape[str(shape["shape_key"])])
                   for shape in shapes):
                continue
            regrets, lowers, uppers = [], [], []
            per_m = []
            for shape in shapes:
                key = str(shape["shape_key"])
                candidates = per_shape[key]
                item = next(value for value in candidates
                            if value["artifact_tile_k"] == artifact)
                best = min(float(value["winner"]["median_us"])
                           for value in candidates)
                median = float(item["winner"]["median_us"])
                lower, upper = regret_interval(item, candidates)
                regrets.append(median / best - 1.)
                lowers.append(lower)
                uppers.append(upper)
                per_m.append({"M": int(shape["m"]), "median_us": median,
                              "regret": regrets[-1],
                              "regret_interval": [lower, upper],
                              "config": item["winner"]["config"],
                              "algorithm": item["winner"]["algorithm"],
                              "within_layout_verdict": item["verdict"]})
            scored.append({"artifact_tile_k": artifact,
                           "layout": planner.layout_identity(artifact),
                           "physical_layout_class": physical_layout_class(artifact),
                           "max_regret": max(regrets),
                           "mean_regret": statistics.mean(regrets),
                           "max_regret_interval": [max(lowers), max(uppers)],
                           "per_m": per_m})
        scored.sort(key=lambda item: (item["max_regret"],
                                      item["mean_regret"],
                                      item["artifact_tile_k"]))
        verdict, selected, runner, resolution_competitor = \
            ranked_decision(scored)
        # Physical bytes and the resident reader/copy descriptor are separate
        # decisions.  Within the proven F=1 class A64/A128/A256 can select
        # different readers per M without asking the offline packer for three
        # copies.  Score those byte classes directly; do not inherit a
        # descriptor tie as a fake repack ambiguity.
        physical_scores = []
        class_ids = sorted({physical_layout_class(artifact)["mapping_sha256"]
                            for artifact in planner.ARTIFACTS})
        for class_id in class_ids:
            per_m, regrets, lowers, uppers, used_artifacts = [], [], [], [], set()
            for shape in shapes:
                key = str(shape["shape_key"])
                candidates = per_shape[key]
                in_class = [item for item in candidates
                            if physical_layout_class(item["artifact_tile_k"])[
                                "mapping_sha256"] == class_id]
                if not in_class:
                    break
                item = min(in_class, key=lambda value: (
                    float(value["winner"]["median_us"]),
                    value["artifact_tile_k"]))
                best = min(float(value["winner"]["median_us"])
                           for value in candidates)
                median = float(item["winner"]["median_us"])
                lower, upper = regret_interval(item, candidates)
                regrets.append(median / best - 1.)
                lowers.append(lower); uppers.append(upper)
                used_artifacts.add(item["artifact_tile_k"])
                per_m.append({"M": int(shape["m"]),
                              "reader_artifact_tile_k": item["artifact_tile_k"],
                              "median_us": median, "regret": regrets[-1],
                              "regret_interval": [lower, upper],
                              "config": item["winner"]["config"],
                              "algorithm": item["winner"]["algorithm"]})
            if len(per_m) == len(shapes):
                exemplar = next(physical_layout_class(artifact)
                                for artifact in planner.ARTIFACTS
                                if physical_layout_class(artifact)[
                                    "mapping_sha256"] == class_id)
                physical_scores.append({
                    "physical_layout_class": exemplar,
                    "reader_artifact_tile_k_used": sorted(used_artifacts),
                    "max_regret": max(regrets),
                    "mean_regret": statistics.mean(regrets),
                    "max_regret_interval": [max(lowers), max(uppers)],
                    "per_m": per_m,
                })
        physical_scores.sort(key=lambda item: (
            item["max_regret"], item["mean_regret"],
            item["physical_layout_class"]["mapping_sha256"]))
        physical_verdict, physical_selected, physical_runner, \
            physical_resolution_competitor = ranked_decision(physical_scores)
        point_winners = {}
        point_winner_classes = {}
        for shape in shapes:
            key = str(shape["shape_key"])
            candidates = per_shape[key]
            best = min(float(item["winner"]["median_us"])
                       for item in candidates)
            point_winners[str(shape["m"])] = sorted(
                item["artifact_tile_k"] for item in candidates
                if float(item["winner"]["median_us"]) == best)
            point_winner_classes[str(shape["m"])] = sorted({
                physical_layout_class(item["artifact_tile_k"])[
                    "mapping_sha256"]
                for item in candidates
                if float(item["winner"]["median_us"]) == best})
        decisions.append({"layer": layer,
                          "N": n, "K": k, "group_size": group_size,
                          "M_values": m_values,
                          "shape_keys_by_M": {
                              str(int(shape["m"])): str(shape["shape_key"])
                              for shape in shapes},
                          "verdict": verdict,
                          "per_m_point_winners": point_winners,
                          "per_m_point_physical_layout_classes": point_winner_classes,
                          "descriptor_winner_changes_with_m": len({tuple(value)
                                for value in point_winners.values()}) > 1,
                          "xplane_byte_class_winner_changes_with_m": len({tuple(value)
                                for value in point_winner_classes.values()}) > 1,
                          "xplane_byte_class_decision": {
                              "verdict": physical_verdict,
                              "selected": physical_selected,
                              "runner_up": physical_runner,
                              "resolution_competitor":
                                  physical_resolution_competitor,
                              "all_scores": physical_scores,
                          },
                          "selected": selected, "runner_up": runner,
                          "resolution_competitor": resolution_competitor,
                          "all_common_layout_scores": scored})
    return decisions


def screen_candidates(screen: dict[str, Any], manifest: dict[str, dict[str, Any]]
                     ) -> dict[str, dict[str, Any]]:
    result = {}
    for candidate in screen["selected"] + screen["screened_out"]:
        symbol = str(candidate["symbol"])
        row = manifest[symbol]
        score = float(candidate["score_us"])
        if not math.isfinite(score) or score <= 0:
            raise AnalysisError(f"invalid screen score for {symbol}")
        result[tactic_name(row)] = {"score_us": score, "row": row,
                                    "symbol": symbol}
    if len(result) != screen["denominator"]["measured"]:
        raise AnalysisError("screen config denominator collapsed")
    return result


def screen_heuristics(inputs: dict[str, Any], threshold: float
                     ) -> dict[str, Any]:
    plan = inputs["plan"]
    manifests = inputs["manifests"]
    screens = inputs["screens"]
    shapes = {str(item["shape_key"]): item for item in plan["shapes"]}
    axis_stats: dict[tuple[int, int, str, int], dict[str, Any]] = {}
    common: dict[tuple[int, int], dict[str, list[float]]] = {}
    cell_counts: collections.Counter[tuple[int, int]] = collections.Counter()
    for cell in plan["cells"]:
        key = str(cell["shape_key"])
        artifact = int(cell["artifact_tile_k"])
        m = int(shapes[key]["m"])
        candidates = screen_candidates(screens[(key, artifact)],
                                       manifests[artifact])
        best = min(item["score_us"] for item in candidates.values())
        group_key = (artifact, m)
        cell_counts[group_key] += 1
        regrets = {config: item["score_us"] / best - 1.
                   for config, item in candidates.items()}
        if group_key not in common:
            common[group_key] = {config: [regret, regret, 1]
                                 for config, regret in regrets.items()}
        else:
            state = common[group_key]
            for config in list(state):
                if config not in regrets:
                    del state[config]
                else:
                    state[config][0] = max(state[config][0], regrets[config])
                    state[config][1] += regrets[config]
                    state[config][2] += 1
        leader = min(candidates, key=lambda config: (candidates[config]["score_us"],
                                                     config))
        for axis in AXES:
            # A shape-specific terminal can remove every measured tactic for
            # one value (for example TK that does not divide K).  The value is
            # still part of the compiled axis denominator: represent it as
            # unavailable-for-this-cell (only-value regret = INF), rather than
            # dropping the value and later mistaking a shortened census for a
            # safe heuristic.
            values = sorted({int(item[axis])
                             for item in manifests[artifact].values()})
            for value in values:
                stat_key = (artifact, m, axis, value)
                stat = axis_stats.setdefault(stat_key, {
                    "artifact_tile_k": artifact, "M": m, "axis": axis,
                    "value": value, "cells": 0, "leader_hits": 0,
                    "worst_regret_if_dropped": 0.,
                    "worst_regret_if_only_value": 0.})
                kept_drop = [item["score_us"] for item in candidates.values()
                             if int(item["row"][axis]) != value]
                kept_only = [item["score_us"] for item in candidates.values()
                             if int(item["row"][axis]) == value]
                drop_regret = (float("inf") if not kept_drop else
                               min(kept_drop) / best - 1.)
                only_regret = (float("inf") if not kept_only else
                               min(kept_only) / best - 1.)
                stat["cells"] += 1
                stat["leader_hits"] += int(
                    int(candidates[leader]["row"][axis]) == value)
                stat["worst_regret_if_dropped"] = max(
                    stat["worst_regret_if_dropped"], drop_regret)
                stat["worst_regret_if_only_value"] = max(
                    stat["worst_regret_if_only_value"], only_regret)
    axis_rows = []
    for key in sorted(axis_stats):
        row = axis_stats[key]
        if row["cells"] != cell_counts[(row["artifact_tile_k"], row["M"])]:
            raise AnalysisError(f"axis census differs for {key}")
        row["drop_within_threshold"] = \
            row["worst_regret_if_dropped"] <= threshold
        row["only_value_within_threshold"] = \
            row["worst_regret_if_only_value"] <= threshold
        axis_rows.append(row)
    m_only = []
    for (artifact, m), configs in sorted(common.items()):
        if not configs:
            m_only.append({"artifact_tile_k": artifact, "M": m,
                           "cells": cell_counts[(artifact, m)],
                           "best_single_config": None,
                           "worst_regret": None, "mean_regret": None,
                           "within_threshold": False})
            continue
        scored = sorted((values[0], values[1] / values[2], config)
                        for config, values in configs.items())
        worst, mean, config = scored[0]
        m_only.append({"artifact_tile_k": artifact, "M": m,
                       "cells": cell_counts[(artifact, m)],
                       "common_configs": len(configs),
                       "best_single_config": config,
                       "worst_regret": worst, "mean_regret": mean,
                       "within_threshold": worst <= threshold})
    return {"schema": HEURISTIC_SCHEMA,
            "scope": "NONPERSISTENT complete-screen scores; two samples; diagnostic pruning evidence, not confirmed deployment ranking",
            "regret_threshold": threshold,
            "axis_value_evidence": axis_rows,
            "m_only_config_evidence": m_only}


def ratio_band(n: int, k: int) -> str:
    if n * 2 <= k:
        return "N_LE_HALF_K"
    if n >= k * 2:
        return "N_GE_2K"
    return "BALANCED_NK"


def confirmed_patterns(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    manifests = inputs["manifests"]
    groups: dict[tuple[str, int, str], dict[str, Any]] = {}
    for shape in inputs["summary"]["shapes"]:
        m, n, k = map(int, (shape["m"], shape["n"], shape["k"]))
        for board in planner.BOARDS:
            result = shape["boards"][board]
            key = (board, m, ratio_band(n, k))
            group = groups.setdefault(key, {"resolved": [], "unresolved": 0,
                                            "unavailable": 0})
            winner = result.get("winner")
            if winner is None:
                group["unavailable"] += 1
            elif result["verdict"] != "RESOLVED":
                group["unresolved"] += 1
            else:
                artifact = int(winner["artifact_tile_k"])
                row = manifests[artifact][winner["symbol"]]
                group["resolved"].append({"config": winner["config"],
                                           "algorithm": winner["algorithm"],
                                           **{axis: int(row[axis]) for axis in AXES}})
    rows = []
    for (board, m, band), group in sorted(groups.items()):
        resolved = group["resolved"]
        config_counts = collections.Counter(item["config"] for item in resolved)
        algorithm_counts = collections.Counter(item["algorithm"] for item in resolved)
        axis_modes = {}
        for axis in AXES:
            counts = collections.Counter(item[axis] for item in resolved)
            axis_modes[axis] = (None if not counts else
                                {"value": counts.most_common(1)[0][0],
                                 "count": counts.most_common(1)[0][1],
                                 "distinct": len(counts)})
        mode_config, mode_count = ((None, 0) if not config_counts else
                                   config_counts.most_common(1)[0])
        rows.append({"board": board, "M": m, "ratio_band": band,
                     "resolved": len(resolved),
                     "unresolved": group["unresolved"],
                     "unavailable": group["unavailable"],
                     "mode_config": mode_config,
                     "mode_config_count": mode_count,
                     "mode_config_coverage": (0. if not resolved else
                                              mode_count / len(resolved)),
                     "algorithm_counts": dict(sorted(algorithm_counts.items())),
                     "axis_modes": axis_modes})
    return rows


def is_product_e2e_recordable(board: str, board_recordable: bool) -> bool:
    return board_recordable and board == "FULL_OUTPUT"


def winner_registry(inputs: dict[str, Any], decisions: list[dict[str, Any]]
                   ) -> list[dict[str, Any]]:
    manifests = inputs["manifests"]
    cell_summaries = inputs["cell_summaries"]
    root_shapes = {str(item["shape_key"]): item
                   for item in inputs["summary"]["shapes"]}
    if len(root_shapes) != len(inputs["summary"]["shapes"]):
        raise AnalysisError("root summary shape key duplicate")
    seen_layers = set()
    rows = []
    for decision in decisions:
        layer = decision["layer"]
        key = layer_key(layer)
        if key in seen_layers:
            raise AnalysisError(f"duplicate layer decision {key}")
        seen_layers.add(key)
        selected = decision.get("selected")
        artifact = None if selected is None else int(selected["artifact_tile_k"])
        physical_decision = decision["xplane_byte_class_decision"]
        physical_selected = physical_decision.get("selected")
        for m_text, shape_key in sorted(
                decision["shape_keys_by_M"].items(), key=lambda item: int(item[0])):
            m = int(m_text)
            root_shape = root_shapes.get(shape_key)
            if root_shape is None:
                raise AnalysisError(f"root summary lacks {shape_key}")
            for board in planner.BOARDS:
                board_result = (None if artifact is None else
                                cell_summaries[(shape_key, artifact)][
                                    "boards"][board])
                winner = (None if board_result is None else
                          board_result.get("winner"))
                runner = (None if board_result is None else
                          board_result.get("runner_up"))
                config_verdict = ("UNAVAILABLE" if winner is None else
                                  board_result["verdict"])
                recordable = (decision["verdict"] == "RESOLVED" and
                              config_verdict == "RESOLVED")
                product_e2e_recordable = is_product_e2e_recordable(
                    board, recordable)
                axes = ({axis: None for axis in AXES} if winner is None else
                        {axis: int(manifests[artifact][winner["symbol"]][axis])
                         for axis in AXES})
                cross_winner = root_shape["boards"][board].get("winner")
                regret = (None if winner is None or cross_winner is None else
                          float(winner["median_us"]) /
                          float(cross_winner["median_us"]) - 1.)
                rows.append({
                        **layer,
                        "M": m, "N": decision["N"], "K": decision["K"],
                        "group_size": decision["group_size"],
                        "board": board,
                        "metric_scope": ("FULL_OUTPUT" if board == "FULL_OUTPUT"
                                         else "PRODUCER_ONLY_NO_REDUCER_E2E"),
                        "offline_layout_verdict": decision["verdict"],
                        "xplane_byte_class_verdict": physical_decision["verdict"],
                        "xplane_byte_class_recordable":
                            physical_decision["verdict"] == "RESOLVED",
                        "config_verdict": config_verdict,
                        "recordable": recordable,
                        "product_e2e_recordable": product_e2e_recordable,
                        "artifact_tile_k": artifact,
                        "layout": (None if artifact is None else
                                   planner.layout_identity(artifact)),
                        "physical_layout_class": (None if physical_selected is None else
                            physical_selected["physical_layout_class"]),
                        "config": None if winner is None else winner["config"],
                        "algorithm": None if winner is None else winner["algorithm"],
                        "grid": None if winner is None else winner["grid"],
                        "policy": None if winner is None else winner["policy"],
                        "median_us": None if winner is None else winner["median_us"],
                        "MFU_pct": None if winner is None else winner["MFU_pct_500TF"],
                        "distinct_MBU_pct": (None if winner is None else
                                             winner["distinct_MBU_pct_2766GBs"]),
                        "regret_vs_per_shape_cross_layout_best": regret,
                        "runner_up": runner,
                        "axes": axes,
                    })
    return rows


def flatten_offline(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in decisions:
        selected, runner = item.get("selected"), item.get("runner_up")
        resolution_competitor = item.get("resolution_competitor")
        physical_decision = item["xplane_byte_class_decision"]
        physical_selected = physical_decision.get("selected")
        physical_competitor = physical_decision.get("resolution_competitor")
        rows.append({**item["layer"],
                     "N": item["N"], "K": item["K"],
                     "group_size": item["group_size"],
                     "M_values": ",".join(map(str, item["M_values"])),
                     "descriptor_verdict": item["verdict"],
                     "xplane_byte_class_verdict": physical_decision["verdict"],
                     "descriptor_winner_changes_with_m":
                         item["descriptor_winner_changes_with_m"],
                     "xplane_byte_class_winner_changes_with_m":
                         item["xplane_byte_class_winner_changes_with_m"],
                     "ArtifactTileK": "" if selected is None else
                         selected["artifact_tile_k"],
                     "FoldN_low": "" if selected is None else
                         selected["layout"]["fold_n"]["low"],
                     "layout": "" if selected is None else selected["layout"]["name"],
                     "physical_layout_class": "" if physical_selected is None else
                         physical_selected["physical_layout_class"]["name"],
                     "xplane_mapping_sha256": "" if physical_selected is None else
                         physical_selected["physical_layout_class"][
                             "mapping_sha256"],
                     "xplane_reader_ArtifactTileK_used": "" if
                         physical_selected is None else ",".join(map(
                             str, physical_selected[
                                 "reader_artifact_tile_k_used"])),
                     "physical_max_regret": "" if physical_selected is None else
                         physical_selected["max_regret"],
                     "physical_max_regret_low": "" if
                         physical_selected is None else
                         physical_selected["max_regret_interval"][0],
                     "physical_max_regret_high": "" if
                         physical_selected is None else
                         physical_selected["max_regret_interval"][1],
                     "max_regret": "" if selected is None else selected["max_regret"],
                     "max_regret_low": "" if selected is None else
                         selected["max_regret_interval"][0],
                     "max_regret_high": "" if selected is None else
                         selected["max_regret_interval"][1],
                     "runner_ArtifactTileK": "" if runner is None else
                         runner["artifact_tile_k"],
                     "runner_max_regret": "" if runner is None else
                         runner["max_regret"],
                     "resolution_competitor_ArtifactTileK": "" if
                         resolution_competitor is None else
                         resolution_competitor["artifact_tile_k"],
                     "xplane_resolution_competitor": "" if
                         physical_competitor is None else
                         physical_competitor["physical_layout_class"]["name"]})
    return rows


def flatten_axis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row,
             "worst_regret_if_dropped": ("INF" if math.isinf(
                 row["worst_regret_if_dropped"]) else
                 row["worst_regret_if_dropped"]),
             "worst_regret_if_only_value": ("INF" if math.isinf(
                 row["worst_regret_if_only_value"]) else
                 row["worst_regret_if_only_value"])} for row in rows]


def flatten_registry(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        layout = row["layout"]
        physical = row["physical_layout_class"]
        axes = row["axes"]
        result.append({key: row[key] for key in (
            "model_id", "tensor", "tp_world", "tp_rank", "tp_partition",
            "M", "N", "K", "group_size", "board", "metric_scope",
            "offline_layout_verdict", "xplane_byte_class_verdict",
            "xplane_byte_class_recordable", "config_verdict", "recordable",
            "product_e2e_recordable")}
            | {"ArtifactTileK": "" if layout is None else
                   layout["artifact_tile_k"],
               "FoldN_low": "" if layout is None else layout["fold_n"]["low"],
               "layout": "" if layout is None else layout["name"],
               "physical_layout_class": "" if physical is None else
                   physical["name"],
               "xplane_mapping_sha256": "" if physical is None else
                   physical["mapping_sha256"],
               "config": row["config"] or "", "algorithm": row["algorithm"] or "",
               "grid": "" if row["grid"] is None else row["grid"],
               "policy": row["policy"] or "",
               "median_us": "" if row["median_us"] is None else row["median_us"],
               "MFU_pct": "" if row["MFU_pct"] is None else row["MFU_pct"],
               "distinct_MBU_pct": ("" if row["distinct_MBU_pct"] is None else
                                    row["distinct_MBU_pct"]),
               "regret_vs_cross_layout_best": ("" if row[
                   "regret_vs_per_shape_cross_layout_best"] is None else
                   row["regret_vs_per_shape_cross_layout_best"]),
               **axes})
    return result


def report_text(inputs: dict[str, Any], decisions: list[dict[str, Any]],
                heuristics: dict[str, Any], patterns: list[dict[str, Any]],
                registry: list[dict[str, Any]]) -> str:
    lines = []
    authority: Authority = inputs["authority"]
    descriptor_verdicts = collections.Counter(item["verdict"]
                                               for item in decisions)
    physical_verdicts = collections.Counter(
        item["xplane_byte_class_decision"]["verdict"] for item in decisions)
    descriptors = collections.Counter(
        item["selected"]["artifact_tile_k"] for item in decisions
        if item.get("selected") is not None)
    physical_classes = collections.Counter(
        item["xplane_byte_class_decision"]["selected"][
            "physical_layout_class"]["name"] for item in decisions
        if item["xplane_byte_class_decision"].get("selected") is not None)
    descriptor_changes = sum(bool(item["descriptor_winner_changes_with_m"])
                             for item in decisions)
    physical_changes = sum(bool(item["xplane_byte_class_winner_changes_with_m"])
                           for item in decisions)
    measurement_families = {(item["N"], item["K"], item["group_size"],
                             tuple(item["M_values"])) for item in decisions}
    grouped_decisions: dict[str, list[dict[str, Any]]] = \
        collections.defaultdict(list)
    for item in decisions:
        selected = item.get("selected")
        physical = item["xplane_byte_class_decision"].get("selected")
        signature = canonical({
            "N": item["N"], "K": item["K"],
            "group_size": item["group_size"], "M_values": item["M_values"],
            "descriptor_verdict": item["verdict"],
            "descriptor_A": (None if selected is None else
                               selected["artifact_tile_k"]),
            "descriptor_regret": (None if selected is None else
                                    selected["max_regret"]),
            "xplane_verdict": item["xplane_byte_class_decision"]["verdict"],
            "xplane_mapping": (None if physical is None else
                                physical["physical_layout_class"][
                                    "mapping_sha256"]),
            "xplane_regret": (None if physical is None else
                               physical["max_regret"]),
        })
        grouped_decisions[signature].append(item)
    lines.append("Q4K_POSTPROCESS "
                 f"measurement_sha={authority.bundle.get('git_sha')} "
                 f"shapes={inputs['plan']['shape_count']} "
                 f"layout_cells={inputs['plan']['cell_count']} "
                 f"measurement_families={len(measurement_families)} "
                 f"layer_decisions={len(decisions)} "
                 f"decision_signatures={len(grouped_decisions)}")
    lines.append("OFFLINE_CENSUS layer_descriptor_verdicts=" +
                 canonical(dict(sorted(descriptor_verdicts.items()))) +
                 " layer_xplane_byte_class_verdicts=" +
                 canonical(dict(sorted(physical_verdicts.items()))) +
                 " layer_selected_descriptor_A=" +
                 canonical(dict(sorted(descriptors.items()))) +
                 " layer_selected_xplane_byte_class=" +
                 canonical(dict(sorted(physical_classes.items()))) +
                 f" per_M_descriptor_changes={descriptor_changes} "
                 f"per_M_xplane_byte_class_changes={physical_changes}")
    representatives = [(values[0], len(values))
                       for values in grouped_decisions.values()]
    for item, layer_count in sorted(
            representatives,
            key=lambda pair: (-float(pair[0]["selected"]["max_regret"])
                              if pair[0].get("selected") else float("inf"),
                              pair[0]["N"], pair[0]["K"]))[:24]:
        selected = item.get("selected")
        physical_decision = item["xplane_byte_class_decision"]
        physical = physical_decision.get("selected")
        descriptor_blocker = item.get("resolution_competitor")
        physical_blocker = physical_decision.get("resolution_competitor")
        lines.append("OFFLINE_HOT "
                     f"layers={layer_count} N={item['N']} K={item['K']} "
                     f"descriptor_verdict={item['verdict']} "
                     f"xplane_byte_class_verdict={physical_decision['verdict']} "
                     f"perM_descriptor_change={int(item['descriptor_winner_changes_with_m'])} "
                     f"perM_xplane_byte_class_change={int(item['xplane_byte_class_winner_changes_with_m'])} "
                     f"descriptor_A={('NA' if selected is None else selected['artifact_tile_k'])} "
                     f"descriptor_blocker_A={('NA' if descriptor_blocker is None else descriptor_blocker['artifact_tile_k'])} "
                     f"descriptor_max_regret={('NA' if selected is None else format(selected['max_regret'], '.6f'))} "
                     f"descriptor_interval={('NA' if selected is None else canonical(selected['max_regret_interval']))} "
                     f"xplane_byte_class={('NA' if physical is None else physical['physical_layout_class']['name'])} "
                     f"xplane_mapping_sha256={('NA' if physical is None else physical['physical_layout_class']['mapping_sha256'])} "
                     f"xplane_blocker_class={('NA' if physical_blocker is None else physical_blocker['physical_layout_class']['name'])} "
                     f"xplane_reader_A={('NA' if physical is None else canonical(physical['reader_artifact_tile_k_used']))} "
                     f"xplane_class_max_regret={('NA' if physical is None else format(physical['max_regret'], '.6f'))} "
                     f"xplane_class_interval={('NA' if physical is None else canonical(physical['max_regret_interval']))}")
    for row in heuristics["m_only_config_evidence"]:
        lines.append("HEURISTIC_M_ONLY "
                     f"A={row['artifact_tile_k']} M={row['M']} cells={row['cells']} "
                     f"config={row.get('best_single_config') or 'NONE'} "
                     f"worst_regret={('NA' if row.get('worst_regret') is None else format(row['worst_regret'], '.6f'))} "
                     f"within_{heuristics['regret_threshold']:.3f}={int(row['within_threshold'])}")
    for row in patterns:
        lines.append("CONFIRMED_PATTERN "
                     f"board={row['board']} M={row['M']} ratio={row['ratio_band']} "
                     f"resolved={row['resolved']} unresolved={row['unresolved']} "
                     f"unavailable={row['unavailable']} mode={row['mode_config'] or 'NONE'} "
                     f"coverage={row['mode_config_coverage']:.3f} "
                     f"algorithms={canonical(row['algorithm_counts'])}")
    recordable = sum(bool(row["recordable"]) for row in registry)
    product_e2e_recordable = sum(bool(row["product_e2e_recordable"])
                                 for row in registry)
    physical_recordable = sum(bool(row["xplane_byte_class_recordable"])
                              for row in registry)
    lines.append(f"WINNER_REGISTRY rows={len(registry)} "
                 f"xplane_byte_class_recordable={physical_recordable} "
                 f"board_scoped_recordable={recordable} "
                 f"product_e2e_recordable={product_e2e_recordable} "
                 f"board_scoped_held_back={len(registry)-recordable}")
    lines.append("SCOPE FULL_OUTPUT is product E2E; SPLITK boards are producer-only and cannot be compared with FULL_OUTPUT or recorded as product latency")
    return "\n".join(lines) + "\n"


def analyze(bundle: pathlib.Path, output: pathlib.Path, threshold: float) -> None:
    if output.exists():
        raise AnalysisError(f"refusing existing analysis output {output}")
    if not math.isfinite(threshold) or threshold < 0:
        raise AnalysisError("regret threshold must be finite and nonnegative")
    inputs = load_inputs(bundle)
    decisions = offline_layout_decisions(inputs)
    heuristics = screen_heuristics(inputs, threshold)
    patterns = confirmed_patterns(inputs)
    registry = winner_registry(inputs, decisions)
    authority: Authority = inputs["authority"]
    offline_doc = {"schema": OFFLINE_SCHEMA,
                   "decision_key": "one concrete (model_id,tp_world,tp_rank,tp_partition,tensor); different layers are never storage-coupled merely because N/K match",
                   "measurement_dedup_rule": "identical (M,N,K,gs,format,device) cells reuse one timing result, but measurement reuse does not merge per-layer offline decisions",
                   "descriptor_selection_rule": "one common ArtifactTileK per layer across measured M; minimize maximum median regret, then mean regret; resolution requires non-overlapping conservative max-regret envelopes",
                   "physical_class_rule": "Score one physical byte class per layer across that layer's measured M values. ArtifactTileK is a resident reader/copy descriptor: Q4_K A32/FoldN=2 is one class, while A64/A128/A256 share the proven tile-free F=1/TK<=256 class and may use different readers per M without repacking",
                   "decisions": decisions}
    registry_doc = {"schema": REGISTRY_SCHEMA,
                    "recording_rule": "physical bytes are recordable when the physical class is RESOLVED; a board-scoped winner is recordable only when both the ArtifactTileK descriptor and within-layout config are RESOLVED; product_e2e_recordable additionally requires FULL_OUTPUT, while producer-only boards remain explicitly scoped",
                    "rows": registry}
    analyzer_path = pathlib.Path(__file__).resolve()
    try:
        git_sha = subprocess.check_output(
            ["git", "-C", str(analyzer_path.parent.parent), "rev-parse", "HEAD"],
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "UNKNOWN"
    analysis = {
        "schema": ANALYSIS_SCHEMA,
        "measurement_git_sha": authority.bundle.get("git_sha"),
        "source_bundle_sha256": digest(authority.bundle_path),
        "analyzer_git_sha": git_sha,
        "analyzer_sha256": digest(analyzer_path),
        "verified_input_members": sorted(authority.verified),
        "offline_layout": offline_doc,
        "heuristics": heuristics,
        "confirmed_patterns": patterns,
        "winner_registry": registry_doc,
    }
    output.mkdir(parents=True)
    atomic_json(output / "analysis.json", analysis)
    atomic_json(output / "offline-layout-decisions.json", offline_doc)
    offline_rows = flatten_offline(decisions)
    atomic_tsv(output / "offline-layout-decisions.tsv", offline_rows,
               list(offline_rows[0]))
    atomic_json(output / "heuristic-evidence.json", heuristics)
    axis_rows = flatten_axis(heuristics["axis_value_evidence"])
    atomic_tsv(output / "axis-pruning-evidence.tsv", axis_rows,
               list(axis_rows[0]))
    m_rows = heuristics["m_only_config_evidence"]
    atomic_tsv(output / "m-only-config-evidence.tsv", m_rows,
               list(m_rows[0]))
    atomic_json(output / "winner-registry.json", registry_doc)
    registry_rows = flatten_registry(registry)
    atomic_tsv(output / "winner-registry.tsv", registry_rows,
               list(registry_rows[0]))
    atomic_json(output / "confirmed-patterns.json",
                {"scope": "RESOLVED confirmed winners only", "rows": patterns})
    text = report_text(inputs, decisions, heuristics, patterns, registry)
    atomic_text(output / "report.txt", text)
    print(text, end="")
    print(f"Q4K_POSTPROCESS_PASS output={output}")


def self_test() -> None:
    # Minimax must not quietly choose the layout which wins only one M.
    reference_both = {
        "model_id": "model", "source_tensors": ["layer0", "layer1"],
        "tp_world": 1, "tp_rank": 0, "tp_partition": "replicated",
    }
    reference_layer0 = {
        "model_id": "model", "source_tensors": ["layer0"],
        "tp_world": 1, "tp_rank": 0, "tp_partition": "replicated",
    }
    shapes = [{"shape_key": "m64", "m": 64, "n": 128, "k": 256,
               "group_size": 32, "references": [reference_both]},
              {"shape_key": "m2048", "m": 2048, "n": 128, "k": 256,
               "group_size": 32, "references": [reference_layer0]}]
    def board(us: float) -> dict[str, Any]:
        return {"verdict": "RESOLVED", "winner": {
            "median_us": us, "range_us": [us * .99, us * 1.01],
            "symbol": "c", "config": "c", "algorithm": "NONPERSISTENT",
            "grid": 1, "policy": "ordinary", "MFU_pct_500TF": 1.,
            "distinct_MBU_pct_2766GBs": 1.}, "runner_up": None}
    cells = {}
    values = {("m64", 32): 10., ("m64", 64): 11.,
              ("m2048", 32): 12., ("m2048", 64): 10.}
    for key in ("m64", "m2048"):
        for artifact in planner.ARTIFACTS:
            result = (board(values[(key, artifact)]) if artifact in (32, 64)
                      else {"verdict": "UNAVAILABLE", "winner": None,
                            "runner_up": None})
            cells[(key, artifact)] = {"boards": {"FULL_OUTPUT": result}}
    decisions = offline_layout_decisions(
        {"plan": {"shapes": shapes}, "cell_summaries": cells})
    by_tensor = {item["layer"]["tensor"]: item for item in decisions}
    if len(decisions) != 2 or \
            by_tensor["layer0"]["selected"]["artifact_tile_k"] != 64 or \
            by_tensor["layer1"]["selected"]["artifact_tile_k"] != 32 or \
            not by_tensor["layer0"]["descriptor_winner_changes_with_m"] or \
            not by_tensor["layer0"][
                "xplane_byte_class_winner_changes_with_m"]:
        raise AssertionError(
            "per-layer minimax/common-M decision was shape-coupled")
    duplicate_m = json.loads(json.dumps(shapes[0]))
    duplicate_m["shape_key"] = "m64-duplicate-measurement"
    try:
        offline_layout_decisions({
            "plan": {"shapes": [shapes[0], duplicate_m]},
            "cell_summaries": cells,
        })
    except AnalysisError as error:
        if "repeats a measured M" not in str(error):
            raise
    else:
        raise AssertionError(
            "one layer accepted two offline decisions for the same M")
    registry_cells = {}
    for cell_key, cell in cells.items():
        full = cell["boards"]["FULL_OUTPUT"]
        registry_cells[cell_key] = {
            "boards": {name: full for name in planner.BOARDS}}
    manifest_row = {axis: 1 for axis in AXES}
    manifests = {
        artifact: {"c": {"symbol": "c", **manifest_row}}
        for artifact in planner.ARTIFACTS
    }
    summary_shapes = []
    for shape in shapes:
        root_winner = min(
            (registry_cells[(shape["shape_key"], artifact)]
             ["boards"]["FULL_OUTPUT"]["winner"]
             for artifact in planner.ARTIFACTS
             if registry_cells[(shape["shape_key"], artifact)]
             ["boards"]["FULL_OUTPUT"].get("winner") is not None),
            key=lambda item: item["median_us"])
        summary_shapes.append({
            "shape_key": shape["shape_key"],
            "boards": {name: {"winner": root_winner}
                       for name in planner.BOARDS},
        })
    registry = winner_registry({
        "manifests": manifests,
        "cell_summaries": registry_cells,
        "summary": {"shapes": summary_shapes},
    }, decisions)
    registry_layers = collections.Counter(row["tensor"] for row in registry)
    if registry_layers != {"layer0": 8, "layer1": 4} or \
            {row["artifact_tile_k"] for row in registry
             if row["tensor"] == "layer0"} != {64} or \
            {row["artifact_tile_k"] for row in registry
             if row["tensor"] == "layer1"} != {32}:
        raise AssertionError(
            "winner registry re-coupled same-shape layer decisions")
    if physical_layout_class(64) != physical_layout_class(128) or \
            physical_layout_class(32) == physical_layout_class(64):
        raise AssertionError("physical FoldN class collapsed/separated incorrectly")
    # A reader-descriptor tie inside one proven byte class must not become a
    # fake offline-repack tie.  A64 wins one M and A128 wins the other; their
    # conservative descriptor envelopes overlap, while the shared F=1 bytes
    # remain cleanly separated from A32/F=2.
    tied_values = {
        ("m64", 32): 14., ("m64", 64): 10.,
        ("m64", 128): 10.05, ("m64", 256): 10.10,
        ("m2048", 32): 14., ("m2048", 64): 10.05,
        ("m2048", 128): 10., ("m2048", 256): 10.10,
    }
    tied_cells = {
        (key, artifact): {"boards": {"FULL_OUTPUT": board(
            tied_values[(key, artifact)])}}
        for key in ("m64", "m2048") for artifact in planner.ARTIFACTS
    }
    tied = next(item for item in offline_layout_decisions(
        {"plan": {"shapes": shapes}, "cell_summaries": tied_cells})
                if item["layer"]["tensor"] == "layer0")
    byte_decision = tied["xplane_byte_class_decision"]
    if tied["verdict"] != "UNRESOLVED" or \
            byte_decision["verdict"] != "RESOLVED" or \
            byte_decision["selected"]["physical_layout_class"]["name"] != \
            "xplane-q4k-tile-free-f1-le256" or \
            byte_decision["selected"]["reader_artifact_tile_k_used"] != \
            [64, 128]:
        raise AssertionError(
            "reader descriptor ambiguity infected the xplane byte decision")
    # The point-objective runner may be cleanly separated while a noisier
    # third place still overlaps.  Looking only at row two is fail-open.
    planted_scores = [
        {"max_regret": 0., "mean_regret": 0.,
         "max_regret_interval": [0., .02]},
        {"max_regret": .03, "mean_regret": .03,
         "max_regret_interval": [.03, .04]},
        {"max_regret": .05, "mean_regret": .05,
         "max_regret_interval": [.01, .50]},
    ]
    planted_verdict, _, planted_runner, planted_blocker = \
        ranked_decision(planted_scores)
    if planted_verdict != "UNRESOLVED" or \
            planted_runner is not planted_scores[1] or \
            planted_blocker is not planted_scores[2]:
        raise AssertionError("noisy third-place interval stayed green")
    if not is_product_e2e_recordable("FULL_OUTPUT", True) or \
            is_product_e2e_recordable("SPLITK_S4_PRODUCER", True) or \
            is_product_e2e_recordable("FULL_OUTPUT", False):
        raise AssertionError("producer-only board became product E2E")
    # Dropping an essential value must be red while a dominated value is safe.
    manifest = {
        "a": {"symbol": "a", "tile_m": 8, "tile_n": 64,
              "tactic_tile_k": 64, "warp_m": 8, "warp_n": 32,
              "stages": 2, "bchunk": 0},
        "b": {"symbol": "b", "tile_m": 16, "tile_n": 64,
              "tactic_tile_k": 64, "warp_m": 16, "warp_n": 32,
              "stages": 2, "bchunk": 0},
    }
    screen = {"selected": [{"symbol": "a", "score_us": 10.}],
              "screened_out": [{"symbol": "b", "score_us": 20.}],
              "denominator": {"measured": 2}}
    fake_plan = {"shapes": [{"shape_key": "x", "m": 64}],
                 "cells": [{"shape_key": "x", "artifact_tile_k": 32}]}
    heur = screen_heuristics({"plan": fake_plan,
                              "manifests": {32: manifest},
                              "screens": {("x", 32): screen}}, .05)
    evidence = {(row["axis"], row["value"]): row
                for row in heur["axis_value_evidence"]}
    if evidence[("tile_m", 8)]["drop_within_threshold"] or \
            not evidence[("tile_m", 16)]["drop_within_threshold"]:
        raise AssertionError("axis drop regret direction differs")
    # Missing one emitted candidate must never shrink the denominator green.
    planted = json.loads(json.dumps(screen))
    planted["denominator"]["measured"] = 3
    try:
        screen_candidates(planted, manifest)
    except AnalysisError:
        pass
    else:
        raise AssertionError("missing screen candidate stayed green")
    print("[q4k-postprocess:self-test] PASS per-layer minimax cross-M "
          "descriptor, same-shape layer isolation, duplicate-M RED, "
          "descriptor-tie/byte-class separation, noisy-third-place RED, "
          "essential/dominated axis regret, and missing denominator RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    command = sub.add_parser("analyze")
    command.add_argument("--bundle", type=pathlib.Path, required=True)
    command.add_argument("--output-dir", type=pathlib.Path)
    command.add_argument("--regret-threshold", type=float, default=.05)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
        return 0
    bundle = args.bundle.resolve(strict=True)
    output = (args.output_dir.resolve() if args.output_dir else
              bundle / "analysis")
    analyze(bundle, output, args.regret_threshold)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AnalysisError, planner.PlanError, OSError, ValueError,
            json.JSONDecodeError) as error:
        print(f"[q4k-postprocess] FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)
