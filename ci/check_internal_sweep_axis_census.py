#!/usr/bin/env python3
"""Bind the internal full-sweep axis census to component source authorities.

``--audit-current`` consumes the ScaleFirst/FullyQuantized matrix and analyzer
contracts directly.  It deliberately does not infer coverage from a committed
legal-table row count: legal rows alone erase static rejects, unsupported
formats, and runtime algorithm expansion from the denominator.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import itertools
import pathlib
import re
import sys
from dataclasses import dataclass


ROOT = pathlib.Path(__file__).resolve().parents[1]
TACTIC = ROOT / "quactlize/include/ppu_tactic_space.hpp"
FORMAT = ROOT / "quactlize/include/ppu_format_config.inc"
FORMAT_HPP = ROOT / "quactlize/include/ppu_format_config.hpp"
SF_MATRIX = ROOT / "tools/scalefirst_internal_matrix.py"
SF_EMITTER = ROOT / "tools/emit_scalefirst_internal_superset.cpp"
SF_RUNNER = ROOT / "tools/run_scalefirst_internal_sweep_box.sh"
SF_ANALYZER = ROOT / "tools/analyze_scalefirst_internal_sweep.py"
FQ_MATRIX = ROOT / "tools/fully_quantized_internal_matrix.py"
FQ_EMITTER = ROOT / "tools/emit_fully_quantized_splitk_superset.cpp"
FQ_BENCH = ROOT / "benchmarks/test_fully_quantized_internal_sweep.cu"
FQ_RUNNER = ROOT / "tools/run_fully_quantized_internal_sweep_box.sh"
FQ_ANALYZER = ROOT / "tools/analyze_fully_quantized_internal_sweep.py"
BC = ROOT / "quactlize/include/gguf_bc_vecdot.hpp"
BC_Q4 = ROOT / "quactlize/include/gguf_bc_q4_gemv.hpp"
OLD_GEMV = ROOT / "quactlize/include/gemv_lowbit/gemv_tactic_space.hpp"


class CensusError(RuntimeError):
    pass


def text(path: pathlib.Path) -> str:
    try:
        return path.read_text()
    except OSError as exc:
        raise CensusError(f"cannot read authority {path.relative_to(ROOT)}: {exc}") from exc


def cpp_int_array(source: str, name: str) -> tuple[int, ...]:
    match = re.search(
        rf"\b{name}\s*\{{\{{(?P<body>.*?)\}}\}}\s*;", source, re.S)
    if match is None:
        raise CensusError(f"cannot find integer axis {name}")
    values = tuple(map(int, re.findall(r"\b\d+\b", match.group("body"))))
    if not values or len(values) != len(set(values)):
        raise CensusError(f"axis {name} is empty or contains duplicates: {values}")
    return values


def python_tuple(path: pathlib.Path, name: str) -> tuple[int, ...]:
    tree = ast.parse(text(path), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = ([target.id for target in node.targets if isinstance(target, ast.Name)]
                     if isinstance(node, ast.Assign)
                     else [node.target.id] if isinstance(node.target, ast.Name) else [])
            if name in names:
                value = ast.literal_eval(node.value)
                if not isinstance(value, tuple) or not value:
                    raise CensusError(f"{path.name}:{name} is not a nonempty tuple")
                return tuple(int(item) for item in value)
    raise CensusError(f"cannot find Python tuple {path.name}:{name}")


@dataclass(frozen=True)
class FormatRow:
    qtype: int
    low_bits: int
    high_bits: int
    group_size: int
    scale_default: int
    fq_default: int


def format_rows() -> dict[int, FormatRow]:
    pattern = re.compile(
        r'X\(\w+,\s*"[^"]+",\s*(\d+),\s*(\d+),\s*(\d+),'
        r'\s*(\d+),\s*(\d+),\s*(\d+),\s*\d+\)')
    rows = {
        int(q): FormatRow(*map(int, (q, lo, hi, group, sf, fq)))
        for q, lo, hi, group, sf, fq in pattern.findall(text(FORMAT))
    }
    if set(rows) != {10, 11, 12, 13, 14}:
        raise CensusError(f"shipping format registry drifted: {sorted(rows)}")
    return rows


def supported_artifacts(qtype: int, artifacts: tuple[int, ...]) -> tuple[int, ...]:
    if qtype == 8:
        return (32,)
    return tuple(a for a in artifacts
                 if not (qtype in {11, 13} and a == 32)
                 and not (qtype == 14 and a == 256))


def coordinate_digest(axes: dict[str, tuple[int, ...]]) -> tuple[int, str]:
    names = tuple(axes)
    digest = hashlib.sha256()
    count = 0
    for values in itertools.product(*(axes[name] for name in names)):
        digest.update(("|".join(f"{name}={value}" for name, value in zip(names, values)) + "\n").encode())
        count += 1
    return count, digest.hexdigest()


def source_census() -> dict:
    tactic = text(TACTIC)
    matrix_artifacts = python_tuple(FQ_MATRIX, "ARTIFACT_TILE_K")
    axes = {
        "tile_m": cpp_int_array(tactic, "kTileM"),
        "tile_n": cpp_int_array(tactic, "kTileN"),
        "tactic_tile_k": cpp_int_array(tactic, "kTileK"),
        "warp_m": cpp_int_array(tactic, "kWarpM"),
        "warp_n": cpp_int_array(tactic, "kWarpN"),
        "stages": python_tuple(FQ_MATRIX, "STAGES"),
        "bchunk_requested": cpp_int_array(tactic, "kBChunkModes"),
    }
    count, digest = coordinate_digest(axes)
    if count != 23040:
        raise CensusError(f"raw topology count drifted: {count}, expected 23040")
    if matrix_artifacts != axes["tactic_tile_k"]:
        raise CensusError(
            f"artifact candidate ladder {matrix_artifacts} differs from TileK axis {axes['tactic_tile_k']}")
    if python_tuple(FQ_MATRIX, "TACTIC_TILE_K") != axes["tactic_tile_k"]:
        raise CensusError("FQ TacticTileK transcription differs from ppu_tactic_space")
    if python_tuple(FQ_MATRIX, "BCHUNK_REQUESTS") != axes["bchunk_requested"]:
        raise CensusError("FQ BChunk transcription differs from ppu_tactic_space")
    splits = python_tuple(FQ_MATRIX, "SPLITS")
    rpw = python_tuple(FQ_MATRIX, "BC_ROWS_PER_WARP")
    if splits != (1, 2, 4, 8) or rpw != (1, 2, 4, 8):
        raise CensusError(f"runtime axes drifted: S={splits} RPW={rpw}")
    artifacts = {8: (32,)}
    artifacts.update({q: supported_artifacts(q, matrix_artifacts) for q in format_rows()})
    return {
        "qtypes": (8, 10, 11, 12, 13, 14),
        "compile_axes": axes,
        "raw_topology_per_format_artifact": count,
        "raw_topology_sha256": digest,
        "artifact_tile_k_by_qtype": artifacts,
        "scale_first_algorithms": (
            "non-persistent", "persistent-capacity", "persistent-balanced",
            "splitk-S2-producer", "splitk-S4-producer", "splitk-S8-producer"),
        "fq_splits": splits,
        "bc_rows_per_warp": rpw,
    }


def require_source_structure(census: dict) -> None:
    sf_emitter = text(SF_EMITTER)
    fq_emitter = text(FQ_EMITTER)
    for loop in ("kTileM", "kTileN", "kWarpM", "kWarpN", "kBChunkModes"):
        if f": {loop})" not in sf_emitter:
            raise CensusError(f"ScaleFirst emitter no longer loops over {loop}")
        if f": {loop})" not in fq_emitter:
            raise CensusError(f"FQ emitter no longer loops over {loop}")
    if "for (int tactic_tk : kTileK)" not in sf_emitter:
        raise CensusError("ScaleFirst emitter no longer loops over shared TacticTileK values")
    if "for (int tactic_tk : kTileK)" not in fq_emitter:
        raise CensusError("FQ emitter no longer loops over the shared TacticTileK axis")
    if "artifact_tile_k_supported" not in text(FORMAT_HPP):
        raise CensusError("finite artifact producer ABI disappeared")

    bc = text(BC)
    for value in census["bc_rows_per_warp"]:
        if not re.search(rf"case\s+{value}\s*:\s*launch_fixed<[^>]*,{value},Grouped>", bc):
            raise CensusError(f"shipping BC dispatch lacks RowsPerWarp={value}")
    if "constexpr int Threads=(T==KType::Q4_K&&RowsPerWarp==4)?128:256;" not in bc:
        raise CensusError("BC Threads policy is no longer the audited derived value")
    if "template <KType T, int ArtifactTileK, int RowsPerWarp, bool Grouped = false>" not in bc:
        raise CensusError("BC kernel template axes drifted")
    bench = text(FQ_BENCH)
    for value in census["bc_rows_per_warp"]:
        if f"FQ_RUN_BC_RPW({value});" not in bench:
            raise CensusError(f"FQ device bench omits BC RowsPerWarp={value}")
    # CTA_N is a real axis only in the separate CUDA Q4 specialization.  Its
    # existence must not be misreported as a generic placed-BC axis.
    if "template <int CTA_N, int WARPS_N, int WARPS_K = 1>" not in text(BC_Q4):
        raise CensusError("separate Q4 CUDA CTA_N branch moved; reclassify the axis")
    if "int cta_n;" not in text(OLD_GEMV):
        raise CensusError("legacy gemv_lowbit CTA_N axis moved; reclassify it")


def bash_int_array(source: str, name: str) -> tuple[int, ...]:
    match = re.search(rf"\b{re.escape(name)}=\((?P<body>[^)]*)\)", source)
    if match is None:
        raise CensusError(f"runner no longer declares {name}")
    values = tuple(map(int, re.findall(r"\b\d+\b", match.group("body"))))
    if not values or len(values) != len(set(values)):
        raise CensusError(f"runner axis {name} is empty or duplicated: {values}")
    return values


def component_modules():
    tools = str(ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    importlib.invalidate_caches()
    modules = (
        importlib.import_module("scalefirst_internal_matrix"),
        importlib.import_module("analyze_scalefirst_internal_sweep"),
        importlib.import_module("fully_quantized_internal_matrix"),
        importlib.import_module("analyze_fully_quantized_internal_sweep"),
    )
    expected = (SF_MATRIX, SF_ANALYZER, FQ_MATRIX, FQ_ANALYZER)
    for module, path in zip(modules, expected):
        if pathlib.Path(module.__file__).resolve() != path.resolve():
            raise CensusError(
                f"component import {module.__name__} resolved to {module.__file__}, "
                f"expected {path.relative_to(ROOT)}")
    if modules[1].matrix is not modules[0] or modules[3].matrix is not modules[2]:
        raise CensusError("analyzer and checker imported different matrix authorities")
    return modules


def unsupported_fixture(grouped: bool) -> dict:
    if not grouped:
        return {
            "qtype": 20, "m": 1, "n": 1, "k": 256,
            "tensor": "future", "format": "IQ4_X",
            "model_id": "future-model", "tp_world": 2, "tp_rank": 0,
            "partition": "n",
        }
    return {
        "qtype": 12, "M": 8, "N": 4096, "K": 4096,
        "tensor": "experts", "format": "Q4_K", "model_id": "moe",
        "route": "grouped_fully_quantized", "route_class": "grouped",
        "E": 256, "active": 8, "ragged": "one-heavy",
    }


def terminal_signature(rows: list[dict]) -> tuple[tuple[str, int, str, str], ...]:
    return tuple(sorted((str(row["algorithm"]), int(row["S"]),
                         str(row["status"]), str(row["problem_route"]))
                        for row in rows))


def current_contract_signature(census: dict) -> dict:
    sf_matrix, sf_analyzer, fq_matrix, fq_analyzer = component_modules()
    sf = sf_matrix.make_manifest(False)
    fq = fq_matrix.make_manifest(False)

    sf_pairs = {(int(row["qtype"]), int(row["artifact_tile_k"])): row
                for row in sf["pairs"]}
    sf_supported_pairs = {
        key for key, row in sf_pairs.items()
        if row["artifact_route"] == "SUPPORTED"
    }
    sf_grouped = tuple(sorted(
        (int(row["qtype"]), str(row["algorithm"]), str(row["status"]),
         str(row["reason"])) for row in sf["grouped_routes"]))
    sf_descriptors = tuple(
        (str(algorithm), int(split), str(scope), str(policy), int(grid))
        for algorithm, split, scope, policy, grid
        in sf_analyzer.algorithm_descriptors())

    fq_q8 = next(fmt for fmt in fq_matrix.parse_formats() if fmt.qtype == 8)
    fq_q8_cells = fq_matrix.expanded_cells([fq_q8])
    fq_algorithms = tuple(
        (str(fq_matrix.algorithm(split)["name"]), int(split),
         str(fq_matrix.algorithm(split)["metric_scope"]))
        for split in fq_matrix.SPLITS)

    sf_unknown = sf_analyzer.unsupported_cells(
        unsupported_fixture(False), "QTYPE_NOT_IN_SCALEFIRST_REGISTRY")
    sf_unknown_grouped = sf_analyzer.unsupported_cells(
        unsupported_fixture(True), "NO_GROUPED_SCALEFIRST_SWEEP_KERNEL")
    fq_unknown = fq_analyzer.unsupported_cells(unsupported_fixture(False))
    fq_unknown_grouped = fq_analyzer.unsupported_cells(
        unsupported_fixture(True),
        "GROUPED_FULLY_QUANTIZED_SWEEP_NOT_REGISTERED")

    sf_runner = text(SF_RUNNER)
    sf_analyzer_source = text(SF_ANALYZER)
    fq_runner = text(FQ_RUNNER)
    fq_analyzer_source = text(FQ_ANALYZER)
    return {
        "sf_schema": sf.get("schema"),
        "fq_schema": fq.get("schema"),
        "sf_qtypes": tuple(sorted(int(row["qtype"]) for row in sf["formats"])),
        "fq_qtypes": tuple(sorted(int(row["qtype"]) for row in fq["formats"])),
        "sf_all_pairs": tuple(sorted(sf_pairs)),
        "sf_supported_pairs": tuple(sorted(sf_supported_pairs)),
        "sf_pair_raw_rows": tuple(sorted(
            (key, int(row["raw_tactic_rows"])) for key, row in sf_pairs.items())),
        "sf_pair_status_totals": tuple(sorted(
            (key, sum(map(int, row["status_counts"].values())))
            for key, row in sf_pairs.items())),
        "sf_pair_static_rejects": sum(
            int(row["status_counts"].get("STATIC_REJECT", 0))
            for row in sf_pairs.values()),
        "sf_supported_raw": int(sf["denominator"]["supported_raw_tactic_rows"]),
        "sf_supported_pair_count": int(
            sf["denominator"]["supported_format_artifact_pairs"]),
        "sf_axes": {name: tuple(values) for name, values in sf["axes"].items()},
        "sf_descriptors": sf_descriptors,
        "sf_metric_boards": dict(sf["metric_boards"]),
        "sf_grouped": sf_grouped,
        "sf_unknown": terminal_signature(sf_unknown),
        "sf_unknown_grouped": terminal_signature(sf_unknown_grouped),
        "fq_axes": {name: tuple(values) if isinstance(values, list) else values
                    for name, values in fq["axes"].items()},
        "fq_algorithms": fq_algorithms,
        "fq_bc_algorithm": str(fq_matrix.BC_ALGORITHM),
        "fq_bc_rpw": tuple(map(int, fq_matrix.BC_ROWS_PER_WARP)),
        "fq_q8": tuple(sorted(
            (str(row["algorithm"]), row.get("split_k_slices"),
             str(row["status"])) for row in fq_q8_cells)),
        "fq_total": int(fq["denominator"]["total_support_cells"]),
        "fq_status_total": sum(map(int, fq["status_counts"].values())),
        "fq_static_rejects": int(fq["status_counts"].get("STATIC_REJECT", 0)),
        "fq_unsupported": int(fq["status_counts"].get("UNSUPPORTED", 0)),
        "fq_unknown": terminal_signature(fq_unknown),
        "fq_unknown_grouped": terminal_signature(fq_unknown_grouped),
        "sf_runner_artifacts": bash_int_array(sf_runner, "artifacts"),
        "sf_runner_bchunks": bash_int_array(sf_runner, "bchunks"),
        "fq_runner_qtypes": bash_int_array(fq_runner, "qtypes"),
        "fq_runner_artifacts": bash_int_array(fq_runner, "artifacts"),
        "fq_runner_bchunks": bash_int_array(fq_runner, "bchunks"),
        "sf_runner_graph": all(marker in sf_runner for marker in (
            "--list-plan", "gen_scalefirst_internal_units.py",
            "test_scalefirst_internal_sweep",
            "analyze_scalefirst_internal_sweep.py")),
        "fq_runner_graph": all(marker in fq_runner for marker in (
            "gen_fully_quantized_splitk_producer_units.py",
            "test_fully_quantized_internal_sweep",
            "analyze_fully_quantized_internal_sweep.py")),
        "sf_reject_publication": all(marker in sf_analyzer_source for marker in (
            'manifest["non_typed_rows"]', "terminal_from_static")),
        "fq_reject_publication": all(marker in fq_analyzer_source for marker in (
            'base["status"] in {"STATIC_REJECT", "UNSUPPORTED"}',
            'status="INADMISSIBLE" if base["status"] == "STATIC_REJECT"')),
        "sf_unsupported_publication": all(
            marker in sf_analyzer_source for marker in (
                "if is_grouped(plan_cell):",
                '"NO_GROUPED_SCALEFIRST_SWEEP_KERNEL"',
                "if fmt is None:", '"QTYPE_NOT_IN_SCALEFIRST_REGISTRY"')),
        "fq_unsupported_publication": all(
            marker in fq_analyzer_source for marker in (
                "if is_grouped(plan_cell):",
                '"GROUPED_FULLY_QUANTIZED_SWEEP_NOT_REGISTERED"',
                "if qtype not in VALID_QTYPES:")),
    }


def require_current_contract(census: dict, contract: dict) -> None:
    expected_qtypes = census["qtypes"]
    if contract["sf_schema"] != "quactlize.scalefirst_internal_support.v2":
        raise CensusError(f"ScaleFirst matrix schema drifted: {contract['sf_schema']}")
    if contract["fq_schema"] != "quactlize-fq-internal-support-v2":
        raise CensusError(f"FullyQuantized matrix schema drifted: {contract['fq_schema']}")
    require_equal("ScaleFirst qtypes", contract["sf_qtypes"], expected_qtypes)
    require_equal("FullyQuantized qtypes", contract["fq_qtypes"], expected_qtypes)

    artifact_axis = census["compile_axes"]["tactic_tile_k"]
    expected_all_pairs = tuple(sorted(itertools.product(expected_qtypes,
                                                        artifact_axis)))
    expected_supported_pairs = tuple(sorted(
        (qtype, artifact)
        for qtype, artifacts in census["artifact_tile_k_by_qtype"].items()
        for artifact in artifacts))
    require_equal("ScaleFirst all format/artifact pairs",
                  contract["sf_all_pairs"], expected_all_pairs)
    require_equal("ScaleFirst supported format/artifact pairs",
                  contract["sf_supported_pairs"], expected_supported_pairs)
    if contract["sf_supported_pair_count"] != 18:
        raise CensusError(
            f"supported format/artifact count is {contract['sf_supported_pair_count']}, expected 18")
    raw = census["raw_topology_per_format_artifact"]
    if any(value != raw for _, value in contract["sf_pair_raw_rows"]):
        raise CensusError("ScaleFirst matrix erased a raw topology coordinate")
    if contract["sf_pair_raw_rows"] != contract["sf_pair_status_totals"]:
        raise CensusError("ScaleFirst terminal status counts do not cover every raw row")
    if contract["sf_supported_raw"] != 18 * raw or \
            contract["sf_pair_static_rejects"] <= 0:
        raise CensusError("ScaleFirst raw rejects no longer occupy the denominator")

    shared_axis_map = {
        "tile_m": "tile_m", "tile_n": "tile_n",
        "tactic_tile_k": "tactic_tile_k", "warp_m": "warp_m",
        "warp_n": "warp_n", "stages": "stages",
        "bchunk_requested": "bchunk",
    }
    for source_name, sf_name in shared_axis_map.items():
        require_equal(f"ScaleFirst shared axis {source_name}",
                      tuple(contract["sf_axes"].get(sf_name, ())),
                      census["compile_axes"][source_name])
    require_equal("ScaleFirst ArtifactTileK",
                  tuple(contract["sf_axes"].get("artifact_tile_k", ())),
                  artifact_axis)
    require_equal("ScaleFirst fixed Split-K",
                  tuple(contract["sf_axes"].get("fixed_split_k", ())),
                  (2, 4, 8))
    expected_sf_descriptors = (
        ("non-persistent", 1, "FULL_OUTPUT", "non-persistent", 0),
        ("persistent", 1, "FULL_OUTPUT", "capacity+balanced", 0),
        ("scale-first-splitk", 2, "PRODUCER_ONLY_REDUCER_EXCLUDED",
         "fixed-split-k", 0),
        ("scale-first-splitk", 4, "PRODUCER_ONLY_REDUCER_EXCLUDED",
         "fixed-split-k", 0),
        ("scale-first-splitk", 8, "PRODUCER_ONLY_REDUCER_EXCLUDED",
         "fixed-split-k", 0),
    )
    require_equal("ScaleFirst runtime algorithms", contract["sf_descriptors"],
                  expected_sf_descriptors)
    if contract["sf_metric_boards"] != {
            "NONPERSISTENT": "FULL_OUTPUT",
            "PERSISTENT": "FULL_OUTPUT_CAPACITY_AND_BALANCED_GRIDS",
            "SPLITK_S2_S4_S8": "PRODUCER_ONLY_NOT_PRODUCT_E2E"}:
        raise CensusError("ScaleFirst metric boards lost NP/P-capacity+balanced/S2/S4/S8")

    expected_sf_grouped = tuple(sorted(
        (qtype, algorithm, "UNSUPPORTED", "NO_GROUPED_SCALEFIRST_SWEEP_KERNEL")
        for qtype in expected_qtypes
        for algorithm in ("NONPERSISTENT", "PERSISTENT",
                          "SPLITK_S2_PRODUCER", "SPLITK_S4_PRODUCER",
                          "SPLITK_S8_PRODUCER")))
    require_equal("ScaleFirst grouped terminals", contract["sf_grouped"],
                  expected_sf_grouped)
    # Spell the two route variants separately; duplicating one fixture cannot
    # accidentally satisfy both unknown-qtype and grouped coverage.
    expected_sf_unknown = tuple(sorted(
        (algorithm, split, "UNSUPPORTED", "dense")
        for algorithm, split in (("non-persistent", 1), ("persistent", 1),
                                 ("scale-first-splitk", 2),
                                 ("scale-first-splitk", 4),
                                 ("scale-first-splitk", 8))))
    expected_sf_group_unknown = tuple(
        (algorithm, split, status, "grouped")
        for algorithm, split, status, _ in expected_sf_unknown)
    require_equal("ScaleFirst unknown-qtype terminals", contract["sf_unknown"],
                  expected_sf_unknown)
    require_equal("ScaleFirst grouped-workload terminals",
                  contract["sf_unknown_grouped"], expected_sf_group_unknown)

    require_equal("FullyQuantized stages",
                  tuple(contract["fq_axes"].get("stages", ())),
                  census["compile_axes"]["stages"])
    require_equal("FullyQuantized BChunk",
                  tuple(contract["fq_axes"].get("bchunk_requested", ())),
                  census["compile_axes"]["bchunk_requested"])
    require_equal("FullyQuantized TacticTileK",
                  tuple(contract["fq_axes"].get("tactic_tile_k", ())),
                  artifact_axis)
    require_equal("FullyQuantized ArtifactTileK",
                  tuple(contract["fq_axes"].get("artifact_tile_k", ())),
                  artifact_axis)
    require_equal("FullyQuantized split-K", tuple(
        contract["fq_axes"].get("split_k_slices", ())), (1, 2, 4, 8))
    require_equal("FullyQuantized TC algorithms", contract["fq_algorithms"], (
        ("FQ_S1", 1, "FULL_END_TO_END_SHIPPING_RESULT"),
        ("SPLITK_S2_PRODUCER", 2,
         "PRODUCER_ONLY_DIAGNOSTIC_NOT_A_PRODUCT_RESULT"),
        ("SPLITK_S4_PRODUCER", 4,
         "PRODUCER_ONLY_DIAGNOSTIC_NOT_A_PRODUCT_RESULT"),
        ("SPLITK_S8_PRODUCER", 8,
         "PRODUCER_ONLY_DIAGNOSTIC_NOT_A_PRODUCT_RESULT")))
    if contract["fq_bc_algorithm"] != "PLACED_BC_GEMV_FULL_OUTPUT":
        raise CensusError("FullyQuantized BC full-output algorithm disappeared")
    require_equal("FullyQuantized BC RowsPerWarp", contract["fq_bc_rpw"],
                  (1, 2, 4, 8))
    require_equal("FullyQuantized Q8 unsupported denominator", contract["fq_q8"], (
        ("FQ_S1", 1, "UNSUPPORTED"),
        ("PLACED_BC_GEMV_FULL_OUTPUT", None, "UNSUPPORTED"),
        ("SPLITK_S2_PRODUCER", 2, "UNSUPPORTED"),
        ("SPLITK_S4_PRODUCER", 4, "UNSUPPORTED"),
        ("SPLITK_S8_PRODUCER", 8, "UNSUPPORTED")))
    if contract["fq_total"] != contract["fq_status_total"] or \
            contract["fq_static_rejects"] <= 0 or contract["fq_unsupported"] != 5:
        raise CensusError("FullyQuantized static/unsupported rows escaped its denominator")

    expected_fq_unknown = tuple(sorted((
        ("bc-gemv", 1, "UNSUPPORTED", "dense"),
        ("tc-s1", 1, "UNSUPPORTED", "dense"),
        ("tc-splitk", 2, "UNSUPPORTED", "dense"),
        ("tc-splitk", 4, "UNSUPPORTED", "dense"),
        ("tc-splitk", 8, "UNSUPPORTED", "dense"))))
    expected_fq_grouped = tuple(
        (algorithm, split, status, "grouped")
        for algorithm, split, status, _ in expected_fq_unknown)
    require_equal("FullyQuantized unknown-qtype terminals", contract["fq_unknown"],
                  expected_fq_unknown)
    require_equal("FullyQuantized grouped terminals",
                  contract["fq_unknown_grouped"], expected_fq_grouped)

    for name in ("sf_runner_artifacts", "fq_runner_artifacts"):
        require_equal(name, contract[name], artifact_axis)
    for name in ("sf_runner_bchunks", "fq_runner_bchunks"):
        require_equal(name, contract[name], (0, 1))
    require_equal("FullyQuantized runner qtypes", contract["fq_runner_qtypes"],
                  (10, 11, 12, 13, 14))
    for name in ("sf_runner_graph", "fq_runner_graph",
                 "sf_reject_publication", "fq_reject_publication",
                 "sf_unsupported_publication", "fq_unsupported_publication"):
        if not contract[name]:
            raise CensusError(f"component runtime graph lost contract seam {name}")


def expect_red(label: str, callback) -> None:
    try:
        callback()
    except CensusError:
        return
    raise AssertionError(f"negative control stayed green: {label}")


def require_equal(label: str, got: tuple[int, ...], want: tuple[int, ...]) -> None:
    if got != want:
        raise CensusError(f"{label}: got={got}, want={want}")


def self_test() -> None:
    census = source_census()
    require_source_structure(census)
    contract = current_contract_signature(census)
    require_current_contract(census, contract)
    assert census["raw_topology_per_format_artifact"] == 23040
    assert sum(len(v) for v in census["artifact_tile_k_by_qtype"].values()) == 18
    axes = census["compile_axes"]
    planted = dict(axes)
    planted["stages"] = tuple(value for value in axes["stages"] if value != 6)
    expect_red("stage ladder loses s6", lambda: require_equal(
        "stages", planted["stages"], axes["stages"]))
    planted = dict(axes)
    planted["bchunk_requested"] = (0,)
    expect_red("BChunk historical default replaces axis", lambda: require_equal(
        "BChunk", planted["bchunk_requested"], axes["bchunk_requested"]))
    canonical_only = {q: ((32,) if q == 8 else (format_rows()[q].scale_default,))
                      for q in census["artifact_tile_k_by_qtype"]}
    if canonical_only == census["artifact_tile_k_by_qtype"]:
        raise AssertionError("artifact negative is not discriminating")
    expect_red("canonical artifact default replaces supported layout axis", lambda: (
        None if canonical_only == census["artifact_tile_k_by_qtype"] else
        (_ for _ in ()).throw(CensusError("artifact map differs"))))
    expect_red("BC default RPW replaces 1/2/4/8 axis", lambda: require_equal(
        "BC RowsPerWarp", (4,), census["bc_rows_per_warp"]))
    expect_red("ScaleFirst S4 producer omitted", lambda: require_equal(
        "ScaleFirst algorithms",
        tuple(x for x in census["scale_first_algorithms"] if "S4" not in x),
        census["scale_first_algorithms"]))
    planted_contract = dict(contract)
    planted_contract["sf_descriptors"] = tuple(
        row for row in contract["sf_descriptors"] if row[1] != 4)
    expect_red("current ScaleFirst S4 terminal omitted",
               lambda: require_current_contract(census, planted_contract))
    planted_contract = dict(contract)
    planted_contract["fq_bc_rpw"] = (1, 2, 4)
    expect_red("current BC RowsPerWarp=8 omitted",
               lambda: require_current_contract(census, planted_contract))
    planted_contract = dict(contract)
    planted_contract["fq_static_rejects"] = 0
    expect_red("current raw rejects erased",
               lambda: require_current_contract(census, planted_contract))
    planted_contract = dict(contract)
    planted_contract["fq_unknown_grouped"] = tuple(
        row for row in contract["fq_unknown_grouped"] if row[1] != 8)
    expect_red("current grouped unsupported S8 omitted",
               lambda: require_current_contract(census, planted_contract))
    print("[internal-axis-census:self-test] PASS "
          f"raw_per_format_artifact={census['raw_topology_per_format_artifact']} "
          f"artifact_pairs={sum(len(v) for v in census['artifact_tile_k_by_qtype'].values())} "
          "source_axes=BOUND current_contract=BOUND "
          "negatives=stage-s6+bchunk1+artifact-set+bc-rpw+scalefirst-s4+"
          "runtime-s4+runtime-rpw8+raw-reject+grouped-s8")


def audit_current() -> int:
    census = source_census()
    require_source_structure(census)
    print("[internal-axis-census] SOURCE " + " ".join(
        f"{name}={','.join(map(str, values))}"
        for name, values in census["compile_axes"].items()))
    print("[internal-axis-census] SOURCE artifacts=" + ";".join(
        f"q{q}:{','.join(map(str, values))}"
        for q, values in sorted(census["artifact_tile_k_by_qtype"].items())))
    print(f"[internal-axis-census] SOURCE raw_per_format_artifact="
          f"{census['raw_topology_per_format_artifact']} "
          f"sha256={census['raw_topology_sha256']}")
    contract = current_contract_signature(census)
    require_current_contract(census, contract)
    print("[internal-axis-census] SCALEFIRST "
          "qtypes=8,10,11,12,13,14 pairs=18/24 "
          "algorithms=NP+P(capacity+balanced)+S2/S4/S8-producer "
          f"raw_rejects={contract['sf_pair_static_rejects']}")
    print("[internal-axis-census] FULLY_QUANTIZED "
          "qtypes=8,10,11,12,13,14 "
          "algorithms=BC-RPW1/2/4/8+TC-S1/S2/S4/S8 "
          f"raw_rejects={contract['fq_static_rejects']} "
          f"unsupported={contract['fq_unsupported']}")
    print("[internal-axis-census] COMPLETE source axes, component matrices, "
          "runners, raw rejects, and grouped/unknown terminals agree")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--audit-current", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        return audit_current()
    except (AssertionError, CensusError, OSError, SyntaxError, ValueError) as exc:
        print(f"[internal-axis-census] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
