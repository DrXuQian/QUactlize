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
import prune_scalefirst_q4k_pilot as pruner

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from quactlize import formats as qformats


ANALYSIS_SCHEMA = "quactlize.scalefirst_q4k_real_shapes_analysis.v5"
BUNDLE_SCHEMA = "quactlize.scalefirst_q4k_real_shapes_bundle.v1"
OFFLINE_SCHEMA = "quactlize.scalefirst_q4k_offline_layout_decisions.v4"
HEURISTIC_SCHEMA = "quactlize.scalefirst_q4k_heuristic_evidence.v1"
REGISTRY_SCHEMA = "quactlize.scalefirst_q4k_winner_registry.v5"
AXES = ("tile_m", "tile_n", "tactic_tile_k", "warp_m", "warp_n",
        "stages", "bchunk")
MODELED_PRODUCT_BOARD = "MODELED_E2E_REDUCER_80PCT_NO_LAUNCH"
REDUCER_BANDWIDTH_FRACTION = .80
REDUCER_NAMEPLATE_GBS = 2766.
REDUCER_EFFECTIVE_GBS = REDUCER_BANDWIDTH_FRACTION * REDUCER_NAMEPLATE_GBS
SPLIT_BY_ALGORITHM = {
    "SPLITK_S2_PRODUCER": 2,
    "SPLITK_S4_PRODUCER": 4,
    "SPLITK_S8_PRODUCER": 8,
}
CONFIRM_ALGORITHMS = {"NONPERSISTENT", "PERSISTENT",
                      *SPLIT_BY_ALGORITHM}


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


def reducer_model(m: int, n: int, split: int,
                  observed_partial_bytes: int | None = None
                  ) -> dict[str, Any]:
    """Model only the reducer's traffic; the measured producer owns its write.

    The producer timing already includes the FP32 partial-workspace write.
    Therefore the reducer adds one FP32 workspace read plus one FP16 output
    write.  Counting two workspace passes here would double-count the producer
    write which is already present in the measured span.
    """
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (m, n)) or split not in (2, 4, 8):
        raise AnalysisError(
            f"invalid reducer geometry M={m} N={n} S={split}")
    partial_bytes = m * n * split * 4
    if observed_partial_bytes is not None and (
            isinstance(observed_partial_bytes, bool) or
            not isinstance(observed_partial_bytes, int) or
            observed_partial_bytes != partial_bytes):
        raise AnalysisError(
            f"Split-K partial byte authority differs: "
            f"observed={observed_partial_bytes!r} expected={partial_bytes}")
    output_bytes = m * n * 2
    logical_bytes = partial_bytes + output_bytes
    modeled_us = logical_bytes / (REDUCER_EFFECTIVE_GBS * 1.e3)
    return {
        "split": split,
        "partial_workspace_read_bytes": partial_bytes,
        "fp16_output_write_bytes": output_bytes,
        "logical_read_write_bytes": logical_bytes,
        "nameplate_gbs": REDUCER_NAMEPLATE_GBS,
        "bandwidth_fraction": REDUCER_BANDWIDTH_FRACTION,
        "effective_gbs": REDUCER_EFFECTIVE_GBS,
        "launch_us": 0.,
        "modeled_us": modeled_us,
        "traffic_scope": "REDUCER_READ_FP32_PARTIAL_PLUS_WRITE_FP16_D",
        "producer_workspace_write": "ALREADY_INCLUDED_IN_MEASURED_PRODUCER",
    }


def q4k_product_metrics(m: int, n: int, k: int, us: float
                        ) -> dict[str, float]:
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (m, n, k)) or k % 32 or \
            not math.isfinite(us) or us <= 0:
        raise AnalysisError("invalid Q4_K product metric geometry/time")
    distinct_bytes = (m * k * 2 + n * k * .5 + n * (k // 32) * 4 +
                      m * n * 2)
    tflops = (2. * m * n * k) / (us * 1.e6)
    return {
        "MFU_pct_500TF": tflops / 500. * 100.,
        "distinct_MBU_pct_2766GBs":
            distinct_bytes / (us * 1.e3) / REDUCER_NAMEPLATE_GBS * 100.,
    }


def modeled_product_candidate(cell: dict[str, Any], m: int, n: int, k: int
                              ) -> dict[str, Any]:
    if cell.get("status") != "MEASURED":
        raise AnalysisError("terminal cell entered modeled product candidates")
    algorithm = str(cell.get("algorithm"))
    producer_us = float(cell.get("median_us"))
    producer_low = float(cell.get("min_us"))
    producer_high = float(cell.get("max_us"))
    if any(not math.isfinite(value) or value <= 0
           for value in (producer_us, producer_low, producer_high)) or \
            not producer_low <= producer_us <= producer_high:
        raise AnalysisError(f"invalid producer envelope for {algorithm}")
    if algorithm in ("NONPERSISTENT", "PERSISTENT"):
        if cell.get("metric_scope") != "FULL_OUTPUT" or \
                int(cell.get("split", -1)) != 1 or \
                int(cell.get("partial_bytes", -1)) != 0:
            raise AnalysisError(
                f"S=1 product cell has split/metric drift: {algorithm}")
        model = {
            "split": 1, "partial_workspace_read_bytes": 0,
            "fp16_output_write_bytes": 0, "logical_read_write_bytes": 0,
            "nameplate_gbs": REDUCER_NAMEPLATE_GBS,
            "bandwidth_fraction": REDUCER_BANDWIDTH_FRACTION,
            "effective_gbs": REDUCER_EFFECTIVE_GBS,
            "launch_us": 0., "modeled_us": 0.,
            "traffic_scope": "NO_SEPARATE_REDUCER",
            "producer_workspace_write": "NOT_APPLICABLE",
        }
        source_board = "FULL_OUTPUT"
    elif algorithm in SPLIT_BY_ALGORITHM:
        split = SPLIT_BY_ALGORITHM[algorithm]
        if cell.get("metric_scope") != "PRODUCER_ONLY_NOT_PRODUCT_E2E" or \
                int(cell.get("split", -1)) != split:
            raise AnalysisError(
                f"{algorithm} lost producer-only/split identity")
        if cell.get("reducer_correctness_untimed") not in (1, True):
            raise AnalysisError(
                f"{algorithm} lacks untimed reducer correctness closure")
        model = reducer_model(m, n, split, cell.get("partial_bytes"))
        source_board = algorithm
    else:
        raise AnalysisError(f"unregistered product candidate {algorithm}")
    reducer_us = float(model["modeled_us"])
    e2e_us = producer_us + reducer_us
    result = {
        "cell": pruner.cell_label(cell),
        "symbol": str(cell["symbol"]), "config": str(cell["config"]),
        "algorithm": algorithm, "source_board": source_board,
        "grid": int(cell["grid"]), "policy": str(cell["policy"]),
        "occupancy": int(cell["occupancy"]),
        "split": int(model["split"]),
        "producer_median_us": producer_us,
        "producer_range_us": [producer_low, producer_high],
        "modeled_reducer_us": reducer_us,
        "modeled_reducer": model,
        "median_us": e2e_us,
        "range_us": [producer_low + reducer_us,
                     producer_high + reducer_us],
        "metric_scope": MODELED_PRODUCT_BOARD,
        "timing_kind": "MEASURED_PRODUCER_PLUS_MODELED_REDUCER",
        **q4k_product_metrics(m, n, k, e2e_us),
    }
    return result


def adjudicate_time_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {"verdict": "UNAVAILABLE", "winner": None,
                "runner_up": None, "resolution_competitor": None,
                "confirmed_candidate_count": 0}
    ordered = sorted(candidates, key=lambda item: (
        float(item["median_us"]), str(item.get("cell", "")),
        int(item.get("artifact_tile_k", 0))))
    winner = ordered[0]
    runner = ordered[1] if len(ordered) > 1 else None
    alternatives = ordered[1:]
    blocker = (None if not alternatives else min(
        alternatives, key=lambda item: (
            float(item["range_us"][0]), float(item["median_us"]),
            str(item.get("cell", "")))))
    resolved = (blocker is None or
                float(winner["range_us"][1]) <
                float(blocker["range_us"][0]))
    runner_payload = None if runner is None else {
        **runner,
        "gap_us": float(runner["median_us"]) -
                  float(winner["median_us"]),
    }
    return {
        "verdict": "RESOLVED" if resolved else "UNRESOLVED",
        "winner": winner,
        "runner_up": runner_payload,
        "resolution_competitor": blocker,
        "confirmed_candidate_count": len(ordered),
    }


def modeled_product_board(groups: dict[tuple[Any, ...], dict[str, Any]],
                          m: int, n: int, k: int) -> dict[str, Any]:
    observed: collections.Counter[str] = collections.Counter()
    measured: collections.Counter[str] = collections.Counter()
    candidates = []
    for cell in groups.values():
        board = pruner.board_of(cell)
        observed[board] += 1
        if cell.get("status") == "MEASURED":
            measured[board] += 1
            candidates.append(modeled_product_candidate(cell, m, n, k))
    if set(observed) != set(planner.BOARDS):
        raise AnalysisError(
            f"modeled product board denominator differs: {sorted(observed)}")
    result = adjudicate_time_candidates(candidates)
    result.update({
        "board": MODELED_PRODUCT_BOARD,
        "metric_scope": MODELED_PRODUCT_BOARD,
        "timing_kind": "MEASURED_PRODUCER_PLUS_MODELED_REDUCER",
        "source_board_census": dict(sorted(observed.items())),
        "measured_source_board_census": dict(sorted(measured.items())),
        "reducer_assumptions": {
            "bandwidth_fraction": REDUCER_BANDWIDTH_FRACTION,
            "nameplate_gbs": REDUCER_NAMEPLATE_GBS,
            "effective_gbs": REDUCER_EFFECTIVE_GBS,
            "launch_us": 0.,
            "logical_bytes": "M*N*S*4 + M*N*2",
            "double_count_guard":
                "producer FP32 partial write stays only in measured producer",
        },
    })
    return result


def verify_confirm_summary(result: dict[str, Any],
                           rebuilt: dict[str, Any]) -> None:
    """Bind the published winners to the raw confirm log without schema lockstep."""
    keys = ("verdict", "measured_cells", "terminal_cells",
            "terminal_reasons")
    winner_keys = ("cell", "symbol", "config", "algorithm", "grid",
                   "policy", "occupancy", "median_us", "range_us")
    runner_keys = ("cell", "symbol", "config", "algorithm", "grid",
                   "policy", "occupancy", "median_us", "range_us")
    for board in planner.BOARDS:
        published = result["boards"][board]
        replayed = rebuilt["boards"][board]
        if any(published.get(key) != replayed.get(key) for key in keys):
            raise AnalysisError(
                f"published {board} census/verdict differs from raw confirm log")
        for name, fields in (("winner", winner_keys),
                             ("runner_up", runner_keys)):
            left, right = published.get(name), replayed.get(name)
            if (left is None) != (right is None) or (
                    left is not None and any(left.get(field) != right.get(field)
                                             for field in fields)):
                raise AnalysisError(
                    f"published {board} {name} differs from raw confirm log")


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
    confirm_groups: dict[tuple[str, int],
                         dict[tuple[Any, ...], dict[str, Any]]] = {}
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
        policy_path = authority.path(str(cell["policy"]))
        policy = pruner.load_policy(policy_path)
        shortlist = pruner.read_symbols(authority.path(
            result_path(artifact, key, "confirm-shortlist.txt")))
        groups = pruner.load_log(
            authority.path(f"raw/a{artifact}/{key}/confirm.log"),
            manifests[artifact], policy, shortlist,
            algorithms=CONFIRM_ALGORITHMS,
            iterations=int(policy["confirm"]["iterations"]))
        rebuilt = pruner.adjudicate(manifests[artifact], groups, policy)
        verify_confirm_summary(result, rebuilt)
        cell_summaries[(key, artifact)] = result
        screens[(key, artifact)] = screen
        schedulers[(key, artifact)] = scheduler
        confirm_groups[(key, artifact)] = groups
    if len(cell_summaries) != int(plan["cell_count"]):
        raise AnalysisError("loaded cell denominator differs")
    modeled_boards = {}
    shapes = {str(item["shape_key"]): item for item in plan["shapes"]}
    for (key, artifact), groups in confirm_groups.items():
        shape = shapes[key]
        modeled_boards[(key, artifact)] = modeled_product_board(
            groups, int(shape["m"]), int(shape["n"]), int(shape["k"]))
    modeled_shape_boards = {}
    for key in shapes:
        candidates = []
        for artifact in planner.ARTIFACTS:
            item = modeled_boards[(key, artifact)]
            if item.get("winner") is not None:
                candidates.append({"artifact_tile_k": artifact,
                                   "layout": planner.layout_identity(artifact),
                                   "within_layout_verdict": item["verdict"],
                                   **item["winner"]})
        modeled_shape_boards[key] = adjudicate_time_candidates(candidates)
        if modeled_shape_boards[key].get("winner") is not None and \
                modeled_shape_boards[key]["winner"][
                    "within_layout_verdict"] != "RESOLVED":
            modeled_shape_boards[key]["verdict"] = "UNRESOLVED"
    return {"authority": authority, "plan": plan, "summary": summary,
            "manifests": manifests, "cell_summaries": cell_summaries,
            "screens": screens, "schedulers": schedulers,
            "confirm_groups": confirm_groups,
            "modeled_boards": modeled_boards,
            "modeled_shape_boards": modeled_shape_boards}


def measured_board(cell_summaries: dict[tuple[str, int], dict[str, Any]],
                   key: str, artifact: int, board: str
                   ) -> dict[str, Any] | None:
    result = cell_summaries[(key, artifact)]["boards"][board]
    winner = result.get("winner")
    if winner is None:
        return None
    return {"artifact_tile_k": artifact, "verdict": result["verdict"],
            "winner": winner, "runner_up": result.get("runner_up")}


def deployment_board(modeled_boards: dict[tuple[str, int], dict[str, Any]],
                     key: str, artifact: int) -> dict[str, Any] | None:
    result = modeled_boards[(key, artifact)]
    winner = result.get("winner")
    if winner is None:
        return None
    return {"artifact_tile_k": artifact, "verdict": result["verdict"],
            "winner": winner, "runner_up": result.get("runner_up"),
            "resolution_competitor": result.get("resolution_competitor")}


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
    modeled_boards = inputs["modeled_boards"]
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
                              if (value := deployment_board(
                                  modeled_boards, shape_key, artifact)) is not None]
            if not per_shape[shape_key]:
                raise AnalysisError(
                    f"modeled product E2E has no layout for {shape_key}")
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
                              "split": item["winner"]["split"],
                              "producer_median_us": item["winner"][
                                  "producer_median_us"],
                              "modeled_reducer_us": item["winner"][
                                  "modeled_reducer_us"],
                              "modeled_reducer_logical_bytes": item[
                                  "winner"]["modeled_reducer"][
                                      "logical_read_write_bytes"],
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
                              "algorithm": item["winner"]["algorithm"],
                              "split": item["winner"]["split"],
                              "producer_median_us": item["winner"][
                                  "producer_median_us"],
                              "modeled_reducer_us": item["winner"][
                                  "modeled_reducer_us"]})
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
                          "selection_board": MODELED_PRODUCT_BOARD,
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
    return board_recordable and board in {"FULL_OUTPUT",
                                          MODELED_PRODUCT_BOARD}


def winner_registry(inputs: dict[str, Any], decisions: list[dict[str, Any]]
                   ) -> list[dict[str, Any]]:
    manifests = inputs["manifests"]
    cell_summaries = inputs["cell_summaries"]
    modeled_boards = inputs["modeled_boards"]
    modeled_shape_boards = inputs["modeled_shape_boards"]
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
            for board in (*planner.BOARDS, MODELED_PRODUCT_BOARD):
                if artifact is None:
                    board_result = None
                elif board == MODELED_PRODUCT_BOARD:
                    board_result = modeled_boards[(shape_key, artifact)]
                else:
                    board_result = cell_summaries[(shape_key, artifact)][
                        "boards"][board]
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
                cross_winner = (modeled_shape_boards[shape_key].get("winner")
                                if board == MODELED_PRODUCT_BOARD else
                                root_shape["boards"][board].get("winner"))
                regret = (None if winner is None or cross_winner is None else
                          float(winner["median_us"]) /
                          float(cross_winner["median_us"]) - 1.)
                if board == MODELED_PRODUCT_BOARD:
                    metric_scope = MODELED_PRODUCT_BOARD
                    timing_kind = "MEASURED_PRODUCER_PLUS_MODELED_REDUCER"
                elif board == "FULL_OUTPUT":
                    metric_scope = "MEASURED_PRODUCT_E2E"
                    timing_kind = "MEASURED_PRODUCT_E2E"
                else:
                    metric_scope = "PRODUCER_ONLY_NOT_PRODUCT_E2E"
                    timing_kind = "MEASURED_PRODUCER_ONLY"
                split = None
                producer_us = None
                reducer_us = None
                reducer_bytes = None
                if winner is not None:
                    split = int(winner.get(
                        "split", SPLIT_BY_ALGORITHM.get(
                            str(winner["algorithm"]), 1)))
                    if board == MODELED_PRODUCT_BOARD:
                        producer_us = winner["producer_median_us"]
                        reducer_us = winner["modeled_reducer_us"]
                        reducer_bytes = winner["modeled_reducer"][
                            "logical_read_write_bytes"]
                    elif board == "FULL_OUTPUT":
                        producer_us = winner["median_us"]
                rows.append({
                        **layer,
                        "M": m, "N": decision["N"], "K": decision["K"],
                        "group_size": decision["group_size"],
                        "board": board,
                        "metric_scope": metric_scope,
                        "timing_kind": timing_kind,
                        "offline_layout_verdict": decision["verdict"],
                        "xplane_byte_class_verdict": physical_decision["verdict"],
                        "xplane_byte_class_recordable":
                            physical_decision["verdict"] == "RESOLVED",
                        "config_verdict": config_verdict,
                        "recordable": recordable,
                        "product_e2e_recordable": product_e2e_recordable,
                        "measured_product_e2e_recordable":
                            recordable and board == "FULL_OUTPUT",
                        "modeled_product_e2e_recordable":
                            recordable and board == MODELED_PRODUCT_BOARD,
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
                        "split": split,
                        "producer_median_us": producer_us,
                        "modeled_reducer_us": reducer_us,
                        "modeled_reducer_logical_bytes": reducer_bytes,
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


def flatten_deployment_scores(decisions: list[dict[str, Any]]
                              ) -> list[dict[str, Any]]:
    rows = []
    for item in decisions:
        selected = item.get("selected")
        physical_selected = item["xplane_byte_class_decision"].get("selected")
        selected_artifact = (None if selected is None else
                             int(selected["artifact_tile_k"]))
        selected_mapping = (None if physical_selected is None else
                            physical_selected["physical_layout_class"][
                                "mapping_sha256"])
        for score in item["all_common_layout_scores"]:
            artifact = int(score["artifact_tile_k"])
            physical = score["physical_layout_class"]
            for per_m in score["per_m"]:
                rows.append({
                    **item["layer"],
                    "M": per_m["M"], "N": item["N"], "K": item["K"],
                    "group_size": item["group_size"],
                    "ArtifactTileK": artifact,
                    "FoldN_low": score["layout"]["fold_n"]["low"],
                    "layout": score["layout"]["name"],
                    "physical_layout_class": physical["name"],
                    "xplane_mapping_sha256": physical["mapping_sha256"],
                    "descriptor_selected": artifact == selected_artifact,
                    "physical_class_selected":
                        physical["mapping_sha256"] == selected_mapping,
                    "offline_descriptor_verdict": item["verdict"],
                    "offline_physical_class_verdict": item[
                        "xplane_byte_class_decision"]["verdict"],
                    "within_layout_verdict": per_m["within_layout_verdict"],
                    "algorithm": per_m["algorithm"],
                    "split": per_m["split"],
                    "config": per_m["config"],
                    "producer_median_us": per_m["producer_median_us"],
                    "modeled_reducer_us": per_m["modeled_reducer_us"],
                    "modeled_reducer_logical_bytes": per_m[
                        "modeled_reducer_logical_bytes"],
                    "modeled_e2e_us": per_m["median_us"],
                    "regret": per_m["regret"],
                    "regret_low": per_m["regret_interval"][0],
                    "regret_high": per_m["regret_interval"][1],
                })
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
            "timing_kind",
            "offline_layout_verdict", "xplane_byte_class_verdict",
            "xplane_byte_class_recordable", "config_verdict", "recordable",
            "product_e2e_recordable", "measured_product_e2e_recordable",
            "modeled_product_e2e_recordable")}
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
               "split": "" if row["split"] is None else row["split"],
               "producer_median_us": ("" if row["producer_median_us"] is None
                                        else row["producer_median_us"]),
               "modeled_reducer_us": ("" if row["modeled_reducer_us"] is None
                                        else row["modeled_reducer_us"]),
               "modeled_reducer_logical_bytes": ("" if row[
                   "modeled_reducer_logical_bytes"] is None else row[
                       "modeled_reducer_logical_bytes"]),
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
    lines.append("DEPLOYMENT_MODEL "
                 f"board={MODELED_PRODUCT_BOARD} "
                 f"reducer_bandwidth_fraction={REDUCER_BANDWIDTH_FRACTION:.2f} "
                 f"nameplate_gbs={REDUCER_NAMEPLATE_GBS:.1f} "
                 f"effective_gbs={REDUCER_EFFECTIVE_GBS:.1f} "
                 "launch_us=0 logical_bytes=M*N*S*4+M*N*2 "
                 "producer_partial_write=ALREADY_IN_MEASURED_PRODUCER")
    deployment_splits = collections.Counter()
    deployment_algorithms = collections.Counter()
    for item in decisions:
        selected = item.get("selected")
        if selected is None:
            continue
        for per_m in selected["per_m"]:
            deployment_splits[str(per_m["split"])] += 1
            deployment_algorithms[str(per_m["algorithm"])] += 1
    lines.append("DEPLOYMENT_WINNER_CENSUS layer_M_choices="
                 f"{sum(deployment_splits.values())} splits="
                 f"{canonical(dict(sorted(deployment_splits.items())))} "
                 f"algorithms={canonical(dict(sorted(deployment_algorithms.items())))}")
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
    measured_product_recordable = sum(
        bool(row["measured_product_e2e_recordable"]) for row in registry)
    modeled_product_recordable = sum(
        bool(row["modeled_product_e2e_recordable"]) for row in registry)
    physical_recordable = sum(bool(row["xplane_byte_class_recordable"])
                              for row in registry)
    lines.append(f"WINNER_REGISTRY rows={len(registry)} "
                 f"xplane_byte_class_recordable={physical_recordable} "
                 f"board_scoped_recordable={recordable} "
                 f"product_e2e_recordable={product_e2e_recordable} "
                 f"measured_product_e2e_recordable={measured_product_recordable} "
                 f"modeled_product_e2e_recordable={modeled_product_recordable} "
                 f"board_scoped_held_back={len(registry)-recordable}")
    lines.append("SCOPE raw SPLITK boards remain producer-only; offline/deployment selection uses measured producer + reducer traffic at 80% of 2766 GB/s with zero reducer launch time. The composite is modeled product E2E, never relabeled as measured product E2E")
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
                   "descriptor_selection_rule": "one common ArtifactTileK per layer across measured M; each M/layout first selects the best measured S=1 product or Split-K producer plus modeled reducer; minimize maximum modeled-product median regret, then mean regret; resolution requires non-overlapping conservative max-regret envelopes",
                   "deployment_timing_scope": MODELED_PRODUCT_BOARD,
                   "reducer_model": {
                       "nameplate_gbs": REDUCER_NAMEPLATE_GBS,
                       "bandwidth_fraction": REDUCER_BANDWIDTH_FRACTION,
                       "effective_gbs": REDUCER_EFFECTIVE_GBS,
                       "launch_us": 0.,
                       "logical_bytes": "M*N*S*4 + M*N*2",
                       "byte_accounting": "read FP32 partial workspace once and write FP16 D once; producer workspace write is already measured",
                   },
                   "physical_class_rule": "Score one physical byte class per layer across that layer's measured M values. ArtifactTileK is a resident reader/copy descriptor: Q4_K A32/FoldN=2 is one class, while A64/A128/A256 share the proven tile-free F=1/TK<=256 class and may use different readers per M without repacking",
                   "decisions": decisions}
    registry_doc = {"schema": REGISTRY_SCHEMA,
                    "recording_rule": "physical bytes are recordable when the physical class is RESOLVED; a board-scoped winner is recordable only when both the ArtifactTileK descriptor and within-layout config are RESOLVED; measured_product_e2e_recordable requires FULL_OUTPUT, modeled_product_e2e_recordable requires the explicit 80%-bandwidth/zero-launch composite, and raw Split-K boards remain producer-only",
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
        "deployment_reducer_model": offline_doc["reducer_model"],
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
    deployment_rows = flatten_deployment_scores(decisions)
    atomic_tsv(output / "modeled-deployment-layout-scores.tsv",
               deployment_rows, list(deployment_rows[0]))
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
            "source_board": "FULL_OUTPUT", "split": 1,
            "producer_median_us": us, "modeled_reducer_us": 0.,
            "modeled_reducer": {"logical_read_write_bytes": 0},
            "grid": 1, "policy": "ordinary", "MFU_pct_500TF": 1.,
            "distinct_MBU_pct_2766GBs": 1.}, "runner_up": None}
    modeled_cells = {}
    values = {("m64", 32): 10., ("m64", 64): 11.,
              ("m2048", 32): 12., ("m2048", 64): 10.}
    for key in ("m64", "m2048"):
        for artifact in planner.ARTIFACTS:
            result = (board(values[(key, artifact)]) if artifact in (32, 64)
                      else {"verdict": "UNAVAILABLE", "winner": None,
                            "runner_up": None})
            modeled_cells[(key, artifact)] = result
    decisions = offline_layout_decisions(
        {"plan": {"shapes": shapes}, "modeled_boards": modeled_cells})
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
            "modeled_boards": modeled_cells,
        })
    except AnalysisError as error:
        if "repeats a measured M" not in str(error):
            raise
    else:
        raise AssertionError(
            "one layer accepted two offline decisions for the same M")
    registry_cells = {}
    for cell_key, full in modeled_cells.items():
        registry_cells[cell_key] = {
            "boards": {name: full for name in planner.BOARDS}}
    manifest_row = {axis: 1 for axis in AXES}
    manifests = {
        artifact: {"c": {"symbol": "c", **manifest_row}}
        for artifact in planner.ARTIFACTS
    }
    summary_shapes = []
    modeled_shape_boards = {}
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
        modeled_shape_boards[shape["shape_key"]] = adjudicate_time_candidates([
            {"artifact_tile_k": artifact,
             "within_layout_verdict": modeled_cells[
                 (shape["shape_key"], artifact)]["verdict"],
             **modeled_cells[(shape["shape_key"], artifact)]["winner"]}
            for artifact in planner.ARTIFACTS
            if modeled_cells[(shape["shape_key"], artifact)].get("winner")
        ])
    registry = winner_registry({
        "manifests": manifests,
        "cell_summaries": registry_cells,
        "modeled_boards": modeled_cells,
        "modeled_shape_boards": modeled_shape_boards,
        "summary": {"shapes": summary_shapes},
    }, decisions)
    registry_layers = collections.Counter(row["tensor"] for row in registry)
    if registry_layers != {"layer0": 10, "layer1": 5} or \
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
        (key, artifact): board(tied_values[(key, artifact)])
        for key in ("m64", "m2048") for artifact in planner.ARTIFACTS
    }
    tied = next(item for item in offline_layout_decisions(
        {"plan": {"shapes": shapes}, "modeled_boards": tied_cells})
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
            not is_product_e2e_recordable(MODELED_PRODUCT_BOARD, True) or \
            is_product_e2e_recordable("SPLITK_S4_PRODUCER", True) or \
            is_product_e2e_recordable("FULL_OUTPUT", False):
        raise AssertionError("producer-only board became product E2E")
    exact_model = reducer_model(1, 4096, 8, 131072)
    if exact_model["logical_read_write_bytes"] != 139264 or \
            not math.isclose(exact_model["modeled_us"],
                             139264 / (2212.8 * 1.e3), rel_tol=1.e-15) or \
            exact_model["logical_read_write_bytes"] == 2 * 131072 + 8192:
        raise AssertionError("reducer byte/time model double-counted producer write")

    def raw_cell(algorithm: str, us: float | None,
                 *, correctness: int = 1,
                 partial_override: int | None = None) -> dict[str, Any]:
        split = SPLIT_BY_ALGORITHM.get(algorithm, 1)
        measured = us is not None
        partial = (0 if split == 1 else 1 * 4096 * split * 4)
        if partial_override is not None:
            partial = partial_override
        return {
            "symbol": f"symbol-{algorithm}", "config": "config",
            "algorithm": algorithm,
            "metric_scope": ("FULL_OUTPUT" if split == 1 else
                             "PRODUCER_ONLY_NOT_PRODUCT_E2E"),
            "policy": "ordinary" if split == 1 else "fixed-split-k",
            "split": split, "grid": 1, "occupancy": 1,
            "capacity_b_mask": "0x0", "balanced_b_mask": "0x0",
            "status": "MEASURED" if measured else "INADMISSIBLE",
            "reason": "MEASURED" if measured else "PIPELINE_DEPTH",
            "partial_bytes": partial,
            "reducer_correctness_untimed": correctness if measured else 0,
            "median_us": us, "min_us": None if us is None else us - .001,
            "max_us": None if us is None else us + .001,
        }

    def raw_groups(split2_us: float | None,
                   *, correctness: int = 1,
                   partial_override: int | None = None
                   ) -> dict[tuple[Any, ...], dict[str, Any]]:
        values = [raw_cell("NONPERSISTENT", 10.),
                  raw_cell("SPLITK_S2_PRODUCER", split2_us,
                           correctness=correctness,
                           partial_override=partial_override),
                  raw_cell("SPLITK_S4_PRODUCER", None),
                  raw_cell("SPLITK_S8_PRODUCER", None)]
        return {(cell["algorithm"],): cell for cell in values}

    # Raw producer 9.99 us looks faster than S=1, but its modeled reducer makes
    # the product slower.  A 9.90 us producer still wins.  This proves Split-K
    # participates without pretending producer-only time is product latency.
    modeled_loss = modeled_product_board(raw_groups(9.99), 1, 4096, 4096)
    modeled_win = modeled_product_board(raw_groups(9.90), 1, 4096, 4096)
    fallback = modeled_product_board(raw_groups(None), 1, 4096, 4096)
    if modeled_loss["winner"]["split"] != 1 or \
            modeled_win["winner"]["split"] != 2 or \
            fallback["winner"]["split"] != 1:
        raise AssertionError("modeled Split-K/S=1 ranking or fallback differs")
    for planted, label in (
            (raw_groups(9.90, correctness=0), "missing reducer correctness"),
            (raw_groups(9.90, partial_override=32764),
             "wrong partial byte denominator")):
        try:
            modeled_product_board(planted, 1, 4096, 4096)
        except AnalysisError:
            pass
        else:
            raise AssertionError(f"{label} stayed green")
    missing_board = raw_groups(9.90)
    del missing_board[("SPLITK_S8_PRODUCER",)]
    try:
        modeled_product_board(missing_board, 1, 4096, 4096)
    except AnalysisError:
        pass
    else:
        raise AssertionError("missing modeled source board stayed green")
    time_candidates = [
        {"median_us": 10., "range_us": [9.99, 10.01], "cell": "a"},
        {"median_us": 10.1, "range_us": [10.08, 10.12], "cell": "b"},
        {"median_us": 10.2, "range_us": [9.98, 10.5], "cell": "c"},
    ]
    if adjudicate_time_candidates(time_candidates)["verdict"] != "UNRESOLVED":
        raise AssertionError("modeled noisy third-place interval stayed green")
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
          "80%-bandwidth/zero-launch reducer bytes, Split-K ranking/fallback, "
          "reducer-correctness/partial-byte/source-board negatives, "
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
