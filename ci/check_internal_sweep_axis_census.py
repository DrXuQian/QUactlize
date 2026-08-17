#!/usr/bin/env python3
"""Bind the internal full-sweep axis census to its source authorities.

This checker has two deliberately different modes:

* ``--self-test`` proves the source census and its planted negatives; it is a
  green local check.
* ``--audit-current`` asks whether today's top-level runners already cover the
  complete census.  It is expected to remain red until the integration gaps
  printed by this program are closed.  A historical default is never promoted
  to a complete axis merely to make this mode green.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import pathlib
import re
import sys
from dataclasses import dataclass


ROOT = pathlib.Path(__file__).resolve().parents[1]
TACTIC = ROOT / "quactlize/include/ppu_tactic_space.hpp"
FORMAT = ROOT / "quactlize/include/ppu_format_config.inc"
FORMAT_HPP = ROOT / "quactlize/include/ppu_format_config.hpp"
EMITTER = ROOT / "benchmarks/emit_tactic_configs.cpp"
Q8_TABLE = ROOT / "benchmarks/lowbit_dense_i8_configs.inc"
SF_RUNNER = ROOT / "tools/run_scalefirst_internal_sweep_box.sh"
SF_ANALYZER = ROOT / "tools/analyze_scalefirst_internal_sweep.py"
FQ_MATRIX = ROOT / "tools/fully_quantized_internal_matrix.py"
FQ_EMITTER = ROOT / "tools/emit_fully_quantized_splitk_superset.cpp"
FQ_BENCH = ROOT / "benchmarks/test_fully_quantized_internal_sweep.cu"
FQ_RUNNER = ROOT / "tools/run_fully_quantized_internal_sweep_box.sh"
FQ_PLAN_RUNNER = ROOT / "tools/run_fully_quantized_internal_matrix.sh"
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
    emitter = text(EMITTER)
    fq_emitter = text(FQ_EMITTER)
    for loop in ("kTileM", "kTileN", "kWarpM", "kWarpN", "kBChunkModes"):
        if f": {loop})" not in emitter:
            raise CensusError(f"ScaleFirst emitter no longer loops over {loop}")
        if f": {loop})" not in fq_emitter:
            raise CensusError(f"FQ emitter no longer loops over {loop}")
    if "for (int tactic_tk : tactic_tks)" not in emitter:
        raise CensusError("ScaleFirst emitter no longer loops over explicit TacticTileK values")
    if "for (int tactic_tk : kTileK)" not in fq_emitter:
        raise CensusError("FQ emitter no longer loops over the shared TacticTileK axis")
    # Historical defaults are deliberately present, but they are not the full
    # experiment ladder.  The committed full Q8 table must spell the latter.
    if "std::vector<int> g_stages{2, 3, 4};" not in emitter:
        raise CensusError("historical stage default moved; re-audit its non-authority status")
    table = text(Q8_TABLE)
    match = re.search(r"^//\s+stages:\s+([0-9 ]+)\s+<-", table, re.M)
    if match is None:
        raise CensusError("Q8 table lacks an explicit stage coverage stamp")
    table_stages = tuple(map(int, match.group(1).split()))
    if table_stages != census["compile_axes"]["stages"]:
        raise CensusError(
            f"full Q8 stage stamp {table_stages} differs from experiment ladder "
            f"{census['compile_axes']['stages']}")
    if "--prune=none" not in table or "--tactic-tk=32,64,128,256" not in table:
        raise CensusError("Q8 table regeneration command silently narrows legal tactics")
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


def current_runner_gaps(census: dict) -> list[str]:
    gaps: list[str] = []
    sf_runner = text(SF_RUNNER)
    sf_analyzer = text(SF_ANALYZER)
    q8_table = text(Q8_TABLE)
    if "Full Q8_0 ScaleFirst" in sf_runner and "format=Q8_0 qtype=8" in sf_runner:
        gaps.append("ScaleFirst top-level runner is Q8/A32-only; qtypes 10--14 and their supported ArtifactTileK candidates are absent")
    if "algorithm=np+capacity+balanced" in sf_runner and "splitk" not in sf_runner.lower():
        gaps.append("ScaleFirst runner has no fixed Split-K S2/S4/S8 producer denominator")
    if "noncanonical measured status" in sf_analyzer and "INADMISSIBLE" not in sf_analyzer:
        gaps.append("ScaleFirst analyzer consumes only legal measured rows; raw static rejects do not occupy terminal denominator cells")
    rows = re.findall(r"^\s*X\((\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),B\)", q8_table, re.M)
    if len(rows) != 2501:
        gaps.append(f"committed Q8 legal table count is {len(rows)}, expected audited 2501")
    else:
        values = [tuple(map(int, row)) for row in rows]
        if {row[4] for row in values} != {16, 32, 64}:
            gaps.append("Q8 legal table unexpectedly changed its WN subset; re-audit static rejects")
        # The raw source axes contain WN128 and BChunk1.  Their absence from a
        # legal table is correct, but a *full denominator* must name the rejects.
        gaps.append(
            f"ScaleFirst publishes {len(rows)} legal Q8 rows, not all "
            f"{census['raw_topology_per_format_artifact']} raw coordinates with rejection reasons")
    if not FQ_RUNNER.is_file():
        gaps.append("default FullyQuantized device runner is absent; top-level falls back to a plan-only matrix")
    if "SUPPORT_MATRIX_ONLY_NO_DEVICE_PERFORMANCE" in text(FQ_PLAN_RUNNER):
        gaps.append("FullyQuantized fallback cannot produce measured component cells or a mergeable COMPLETE summary")
    if "GGUF_SET" not in sf_runner:
        gaps.append("ScaleFirst component runner has no multi-GGUF/model_id+TP+grouped input interface")
    return gaps


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
    print("[internal-axis-census:self-test] PASS "
          f"raw_per_format_artifact={census['raw_topology_per_format_artifact']} "
          f"artifact_pairs={sum(len(v) for v in census['artifact_tile_k_by_qtype'].values())} "
          "source_axes=BOUND negatives=stage-s6+bchunk1+artifact-set+bc-rpw+scalefirst-s4")


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
    gaps = current_runner_gaps(census)
    for index, gap in enumerate(gaps, 1):
        print(f"[internal-axis-census] GAP {index}: {gap}")
    if gaps:
        print(f"[internal-axis-census] INCOMPLETE gaps={len(gaps)}")
        return 3
    print("[internal-axis-census] PASS runner denominator covers the source census")
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
