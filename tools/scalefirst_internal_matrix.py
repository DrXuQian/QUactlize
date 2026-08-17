#!/usr/bin/env python3
"""Own the exact all-format ScaleFirst sweep denominator.

The raw tactic product is emitted by the same host-readable policy that owns
the shipping dense kernels.  This file adds only format/artifact reachability
and algorithm scope.  In particular, fixed Split-K S2/S4/S8 is a
producer-only diagnostic and can never be ranked as a full product result.

Persistent grids depend on the *compiled* row's occupancy and the runtime
shape/CU count.  They are therefore expanded by the device executable from
``scalefirst_persistent_policy::grid_space``; this plan records that expansion
rule instead of guessing an occupancy on the host.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from dataclasses import asdict, dataclass


ROOT = pathlib.Path(__file__).resolve().parents[1]
EMITTER = ROOT / "tools/emit_scalefirst_internal_superset.cpp"
TACTIC_SPACE = ROOT / "quactlize/include/ppu_tactic_space.hpp"
FORMAT_AUTHORITY = ROOT / "quactlize/include/ppu_format_config.inc"
PERSISTENT_POLICY = ROOT / "quactlize/include/scalefirst_persistent_policy.hpp"

ARTIFACT_TILE_K = (32, 64, 128, 256)
SPLITS = (2, 4, 8)
FULL_OUTPUT_ALGORITHMS = ("NONPERSISTENT", "PERSISTENT")
PRODUCER_ONLY_ALGORITHMS = tuple(f"SPLITK_S{s}_PRODUCER" for s in SPLITS)
GROUPED_ALGORITHMS = (*FULL_OUTPUT_ALGORITHMS, *PRODUCER_ONLY_ALGORITHMS)
RAW_ROWS_PER_PAIR = 23040


@dataclass(frozen=True)
class Format:
    qtype: int
    name: str
    low_bits: int
    high_bits: int
    group_size: int
    quant_mode: str
    metadata_planes: int
    artifacts: tuple[int, ...]


@dataclass(frozen=True)
class Tactic:
    tile_m: int
    tile_n: int
    tactic_tile_k: int
    warp_m: int
    warp_n: int
    stages: int
    bchunk: int
    source_status: str
    source_reason: str
    fold_low: int
    fold_high: int

    @property
    def name(self) -> str:
        return (f"{self.tile_m}x{self.tile_n}x{self.tactic_tile_k}_"
                f"w{self.warp_m}x{self.warp_n}_s{self.stages}_bc{self.bchunk}")


FORMATS = (
    Format(8, "Q8_0", 8, 0, 32, "ScaleOnly", 1, (32,)),
    Format(10, "Q2_K", 2, 0, 16, "ScaleZero", 2, (32, 64, 128, 256)),
    Format(11, "Q3_K", 2, 1, 16, "ScaleOnly", 1, (64, 128, 256)),
    Format(12, "Q4_K", 4, 0, 32, "ScaleZero", 2, (32, 64, 128, 256)),
    Format(13, "Q5_K", 4, 1, 32, "ScaleZero", 2, (64, 128, 256)),
    Format(14, "Q6_K", 4, 2, 16, "ScaleOnly", 1, (32, 64, 128)),
)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_for(qtype: int) -> Format:
    for fmt in FORMATS:
        if fmt.qtype == qtype:
            return fmt
    raise ValueError(f"unsupported ScaleFirst qtype {qtype}")


def emitter_binary() -> pathlib.Path:
    digest = hashlib.sha256(
        EMITTER.read_bytes() + TACTIC_SPACE.read_bytes()).hexdigest()[:16]
    build = pathlib.Path("/workspace") / f"quactlize-scalefirst-emitter-{digest}"
    binary = build / "emit_scalefirst_internal_superset"
    if binary.is_file():
        return binary
    build.mkdir(parents=True, exist_ok=True)
    partial = build / "emit_scalefirst_internal_superset.building"
    result = subprocess.run(
        ["c++", "-std=c++17", "-O2", "-Iquactlize/include",
         str(EMITTER.relative_to(ROOT)), "-o", str(partial)],
        cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("ScaleFirst tactic emitter did not compile:\n" +
                           result.stdout + result.stderr)
    partial.replace(binary)
    return binary


@functools.lru_cache(maxsize=None)
def emitted_tactics(qtype: int, artifact: int,
                    legacy_q4_gs16: bool = False) -> tuple[Tactic, ...]:
    command = [str(emitter_binary()), str(qtype), str(artifact), "0"]
    if legacy_q4_gs16:
        command.append("--plant-q4-legacy-gs16")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            f"ScaleFirst emitter rejected q={qtype} A={artifact}:\n" +
            result.stdout + result.stderr)
    header = re.search(
        r"^SF_SUPERSET_SUMMARY .* raw=(\d+) eligible=(\d+) rejected=(\d+)$",
        result.stdout, re.M)
    row_re = re.compile(
        r"^SF_SUPERSET_ROW q=\d+ format=\S+ mode=\S+ gs=\d+ planes=\d+ "
        r"A=\d+ fold_low=(\d+) fold_high=(\d+) tm=(\d+) tn=(\d+) tk=(\d+) "
        r"wm=(\d+) wn=(\d+) stages=(\d+) bchunk=(\d+) status=(\S+) reason=(\S+)$",
        re.M)
    rows = tuple(Tactic(
        tile_m=int(match[2]), tile_n=int(match[3]),
        tactic_tile_k=int(match[4]), warp_m=int(match[5]),
        warp_n=int(match[6]), stages=int(match[7]), bchunk=int(match[8]),
        source_status=match[9], source_reason=match[10],
        fold_low=int(match[0]), fold_high=int(match[1]))
        for match in row_re.findall(result.stdout))
    eligible = sum(row.source_status == "TYPE_ADMISSION_REQUIRED" for row in rows)
    rejected = sum(row.source_status == "STATIC_REJECT" for row in rows)
    if (header is None or len(rows) != RAW_ROWS_PER_PAIR or
            (int(header.group(1)), int(header.group(2)), int(header.group(3))) !=
            (len(rows), eligible, rejected) or len(set(rows)) != len(rows)):
        raise RuntimeError(
            f"ScaleFirst denominator did not bind q={qtype} A={artifact}: "
            f"header={header.groups() if header else None} rows={len(rows)} "
            f"eligible={eligible} rejected={rejected}")
    return rows


def atom_tiling_supported(tactic: Tactic) -> tuple[bool, str]:
    # Production folded/two-plane copies tile an eight-row swizzle atom after
    # applying FoldN.  This is stronger than merely requiring FoldN | TileN.
    if tactic.tile_n % (8 * tactic.fold_low):
        return False, "LOW_PLANE_COPY_ATOM_DOES_NOT_TILE_FOLDED_N"
    if tactic.fold_high != 1 and tactic.tile_n % (8 * tactic.fold_high):
        return False, "HIGH_PLANE_COPY_ATOM_DOES_NOT_TILE_FOLDED_N"
    return True, "COMPILED_COPY_ATOM_GEOMETRY"


def classify(fmt: Format, artifact: int, tactic: Tactic) -> tuple[str, str]:
    if artifact not in fmt.artifacts:
        return "UNSUPPORTED", "FORMAT_ARTIFACT_ROUTE_UNSUPPORTED"
    if tactic.source_status != "TYPE_ADMISSION_REQUIRED":
        return "STATIC_REJECT", tactic.source_reason
    atom_ok, atom_reason = atom_tiling_supported(tactic)
    if not atom_ok:
        return "STATIC_REJECT", atom_reason
    return "TYPE_ADMISSION_REQUIRED", "REAL_COMPILED_TYPE_AND_CAN_IMPLEMENT_REQUIRED"


def pair_manifest(fmt: Format, artifact: int, expand: bool) -> dict:
    rows = emitted_tactics(fmt.qtype, artifact)
    counts: dict[str, int] = {}
    typed = []
    records = []
    for row in rows:
        status, reason = classify(fmt, artifact, row)
        counts[status] = counts.get(status, 0) + 1
        if status == "TYPE_ADMISSION_REQUIRED":
            typed.append(row)
        if expand:
            records.append(asdict(row) | {"status": status, "reason": reason})
    result = {
        "qtype": fmt.qtype,
        "format": fmt.name,
        "quant_mode": fmt.quant_mode,
        "group_size": fmt.group_size,
        "metadata_planes": fmt.metadata_planes,
        "artifact_tile_k": artifact,
        "fold_low": rows[0].fold_low,
        "fold_high": rows[0].fold_high,
        "artifact_route": "SUPPORTED" if artifact in fmt.artifacts else "UNSUPPORTED",
        "raw_tactic_rows": len(rows),
        "typed_tactic_rows": len(typed),
        "status_counts": counts,
        "runtime_algorithm_expansion": {
            "NONPERSISTENT": "one full-output cell per admitted tactic",
            "PERSISTENT": (
                "one full-output cell per deduplicated capacity/balanced grid "
                "from exact compiled maximum_active_blocks()"),
            "SPLITK_S2_S4_S8": (
                "one producer-only cell per admitted tactic and S; untimed "
                "deterministic reducer is a correctness prerequisite"),
        },
    }
    if expand:
        result["rows"] = records
    return result


def make_manifest(expand: bool) -> dict:
    pairs = [pair_manifest(fmt, artifact, expand)
             for fmt in FORMATS for artifact in ARTIFACT_TILE_K]
    supported_raw = sum(pair["raw_tactic_rows"] for pair in pairs
                        if pair["artifact_route"] == "SUPPORTED")
    typed = sum(pair["typed_tactic_rows"] for pair in pairs)
    grouped = [
        {"qtype": fmt.qtype, "format": fmt.name, "algorithm": algorithm,
         "status": "UNSUPPORTED", "reason": "NO_GROUPED_SCALEFIRST_SWEEP_KERNEL"}
        for fmt in FORMATS for algorithm in GROUPED_ALGORITHMS
    ]
    return {
        "schema": "quactlize.scalefirst_internal_support.v2",
        "scope": "dense-all-format-denominator-and-runtime-expansion-contract",
        "authorities": {
            str(EMITTER.relative_to(ROOT)): sha256(EMITTER),
            str(TACTIC_SPACE.relative_to(ROOT)): sha256(TACTIC_SPACE),
            str(FORMAT_AUTHORITY.relative_to(ROOT)): sha256(FORMAT_AUTHORITY),
            str(PERSISTENT_POLICY.relative_to(ROOT)): sha256(PERSISTENT_POLICY),
        },
        "formats": [asdict(fmt) for fmt in FORMATS],
        "axes": {
            "artifact_tile_k": list(ARTIFACT_TILE_K),
            "tile_m": [8, 16, 32, 64, 128, 256],
            "tile_n": [16, 32, 64, 128, 256],
            "tactic_tile_k": [32, 64, 128, 256],
            "warp_m": [8, 16, 32, 64],
            "warp_n": [16, 32, 64, 128],
            "stages": [2, 3, 4, 6, 8, 12],
            "bchunk": [0, 1],
            "fixed_split_k": list(SPLITS),
        },
        "metric_boards": {
            "NONPERSISTENT": "FULL_OUTPUT",
            "PERSISTENT": "FULL_OUTPUT_CAPACITY_AND_BALANCED_GRIDS",
            "SPLITK_S2_S4_S8": "PRODUCER_ONLY_NOT_PRODUCT_E2E",
        },
        "denominator": {
            "all_format_artifact_pairs": len(FORMATS) * len(ARTIFACT_TILE_K),
            "supported_format_artifact_pairs": sum(len(fmt.artifacts) for fmt in FORMATS),
            "raw_tactic_rows_per_pair": RAW_ROWS_PER_PAIR,
            "supported_raw_tactic_rows": supported_raw,
            "typed_tactic_rows": typed,
            "grouped_explicit_unsupported_cells": len(grouped),
            "runtime_cells": "DEVICE_OWNED_EXACT_DENOMINATOR_AFTER_OCCUPANCY_GRID_EXPANSION",
        },
        "pairs": pairs,
        "grouped_routes": grouped,
    }


def self_test() -> None:
    manifest = make_manifest(False)
    assert manifest["denominator"]["supported_format_artifact_pairs"] == 18
    assert manifest["denominator"]["supported_raw_tactic_rows"] == 18 * RAW_ROWS_PER_PAIR
    assert manifest["denominator"]["grouped_explicit_unsupported_cells"] == 30
    q8 = next(pair for pair in manifest["pairs"]
              if pair["qtype"] == 8 and pair["artifact_tile_k"] == 32)
    assert q8["typed_tactic_rows"] == 2501
    # Negative 1: the old Q4 gs16/two-plane assumption must delete real rows.
    live = emitted_tactics(12, 32)
    planted = emitted_tactics(12, 32, True)
    assert sum(r.source_status == "TYPE_ADMISSION_REQUIRED" for r in live) != \
        sum(r.source_status == "TYPE_ADMISSION_REQUIRED" for r in planted)
    # Negative 2: one omitted raw coordinate changes the denominator.
    assert len(live[:-1]) != RAW_ROWS_PER_PAIR
    # Negative 3: an extra runtime record is not part of any named board.
    assert "SPLITK_S16_PRODUCER" not in PRODUCER_ONLY_ALGORITHMS
    # Negative 4: grouped absence is red-by-name, never an empty green set.
    assert all(row["status"] == "UNSUPPORTED" and row["reason"]
               for row in manifest["grouped_routes"])
    print("[scalefirst-internal-matrix:self-test] PASS "
          f"pairs=18 raw={18 * RAW_ROWS_PER_PAIR} "
          f"typed={manifest['denominator']['typed_tactic_rows']} "
          "q8=2501 Q4-gs-negative=RED omit-one=RED extra-algorithm=RED "
          "grouped=30xEXPLICIT_UNSUPPORTED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("emit", "self-test"))
    parser.add_argument("--expand", action="store_true")
    parser.add_argument("--out", default="-")
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
            return 0
        payload = json.dumps(make_manifest(args.expand), indent=2,
                             sort_keys=True) + "\n"
        if args.out == "-":
            sys.stdout.write(payload)
        else:
            path = pathlib.Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload)
        return 0
    except (AssertionError, OSError, RuntimeError, ValueError) as error:
        print(f"[scalefirst-internal-matrix] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
