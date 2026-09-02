#!/usr/bin/env python3
"""Generate exhaustive grouped FullyQuantized canonical K-pack shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import fully_quantized_kpack_discovery_matrix as matrix
import fully_quantized_kpack_bundle_index as bundle_index


SCHEMA = "quactlize.fully_quantized_grouped_kpack_shard.v2"


def write(path: pathlib.Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def symbol(qtype: int, row, delivery_n: int, persistent: int) -> str:
    mode = "persistent" if persistent else "nonpersistent"
    return (f"fqg_q{qtype}_l{matrix.layout_for(qtype)}_"
            f"tm{row.tile_m}_tn{row.tile_n}_tk{row.tactic_tile_k}_"
            f"wm{row.warp_m}_wn{row.warp_n}_s{row.stages}_ap0_"
            f"dn{delivery_n}_{mode}")


def macro(name: str, qtype: int,
          rows: list[tuple[object, int, int]]) -> str:
    if not rows:
        return f"#define {name}(X)\n"
    lines = [f"#define {name}(X) \\"]
    layout = matrix.layout_for(qtype)
    for index, (row, delivery_n, algorithm) in enumerate(rows):
        persistent = int(algorithm == "GROUPED_PERSISTENT")
        tail = " \\" if index + 1 < len(rows) else ""
        lines.append(
            f"  X({symbol(qtype, row, delivery_n, persistent)},"
            f"{qtype},{layout},"
            f"{row.tile_m},{row.tile_n},{row.tactic_tile_k},"
            f"{row.warp_m},{row.warp_n},{row.stages},{delivery_n},"
            f"{persistent}){tail}")
    return "\n".join(lines) + "\n"


def row_json(qtype: int, row, delivery_n: int, persistent: int,
             parent_ordinal: int) -> dict:
    algorithm = "GROUPED_PERSISTENT" if persistent else "GROUPED_NONPERSISTENT"
    return {
        "parent_ordinal": parent_ordinal,
        "static_candidate_id": matrix.parent_id(
            qtype, row, 0, delivery_n, "grouped", algorithm),
        "symbol": symbol(qtype, row, delivery_n, persistent),
        "qtype": qtype,
        "weight_layout": matrix.layout_for(qtype),
        "artifact_tile_k": 0,
        "bchunk": 0,
        "tile_m": row.tile_m,
        "tile_n": row.tile_n,
        "tactic_tile_k": row.tactic_tile_k,
        "warp_m": row.warp_m,
        "warp_n": row.warp_n,
        "stages": row.stages,
        "a_provider": 0,
        "resolved_delivery_n": delivery_n,
        "algorithm": algorithm,
        "persistent": bool(persistent),
        "correctness": "REAL_RAGGED_OFFSETS_EMPTY_EXPERT_RAW_FP16_BEFORE_TIMING",
        "runtime_variant_authority": (
            "EXACT_RUNTIME_GRID_SPACE_FROM_COMPILED_KERNEL"
            if persistent else "ONE_NONPERSISTENT_VARIANT_PER_CELL"),
    }


def generate(qtype: int, out: pathlib.Path, per_unit: int,
             plant_drop_middle: bool = False,
             plant_fake_splitk: bool = False,
             parent_begin: int = 0,
             parent_count: int | None = None) -> dict:
    if (qtype not in matrix.QTYPES or per_unit <= 0 or
            per_unit > bundle_index.MAX_PARENTS_PER_BINARY):
        raise ValueError(
            "qtype must be Q2/Q3/Q4/Q5/Q6 and per-unit must be in [1,32]")
    topology = list(matrix.admitted_rows(qtype))
    authority = list(matrix.grouped_rows(qtype))
    expected = len(authority)
    if plant_drop_middle and authority:
        authority.pop(len(authority) // 2)
    if len(authority) != expected:
        raise RuntimeError(f"grouped type denominator {len(authority)}/{expected}; row missing")

    if parent_begin < 0:
        raise ValueError("parent begin must be nonnegative")
    if parent_begin >= expected:
        raise ValueError(
            f"parent range begins outside authority: {parent_begin}/{expected}")
    if parent_count is None:
        parent_count = min(bundle_index.MAX_PARENTS_PER_BINARY,
                           expected - parent_begin)
    if not 1 <= parent_count <= bundle_index.MAX_PARENTS_PER_BINARY:
        raise ValueError("parent count must be in [1,32]")
    parent_end = parent_begin + parent_count
    if parent_end > expected:
        raise ValueError(
            f"parent range exceeds authority: [{parent_begin},{parent_end})/{expected}")
    selected = authority[parent_begin:parent_end]

    layout = matrix.layout_for(qtype)
    registry = (
        "// GENERATED -- exhaustive grouped canonical K-pack FQ registry.\n"
        f"#define FQ_GROUPED_KPACK_GENERATED_QTYPE {qtype}\n"
        f"#define FQ_GROUPED_KPACK_GENERATED_WEIGHT_LAYOUT {layout}\n"
        f"#define FQ_GROUPED_KPACK_GENERATED_TOPOLOGIES {len(topology)}\n"
        f"#define FQ_GROUPED_KPACK_GENERATED_TYPE_ROWS {len(selected)}\n" +
        macro("FQ_GROUPED_KPACK_REGISTRY_ROWS", qtype, selected))
    write(out / "fq_grouped_kpack_registry.inc", registry)

    units = []
    for unit_index, begin in enumerate(range(0, len(selected), per_unit)):
        batch = selected[begin:begin + per_unit]
        path = out / "units" / f"fq_grouped_kpack_unit_{unit_index:05d}.cu"
        write(path,
              "// GENERATED -- grouped canonical K-pack packed-unit types.\n"
              "#ifdef PPU_PACKED_SCALE\n#undef PPU_PACKED_SCALE\n#endif\n"
              "#define PPU_PACKED_SCALE 1\n" +
              macro("FQ_GROUPED_KPACK_UNIT_ROWS", qtype, batch) +
              '#include "fully_quantized_grouped_kpack_discovery_unit.inc"\n')
        units.append(str(path))
    write(out / "units.cmake",
          "# GENERATED -- grouped canonical K-pack FQ discovery sources.\n"
          "set(FQ_GROUPED_KPACK_GENERATED_UNIT_SOURCES\n" +
          "".join(f'  "{path}"\n' for path in units) + ")\n")

    structural = {
        "GROUPED_SPLITK_S2": "STRUCTURAL_UNAVAILABLE",
        "GROUPED_SPLITK_S4": "STRUCTURAL_UNAVAILABLE",
        "GROUPED_SPLITK_S8": "STRUCTURAL_UNAVAILABLE",
        "GROUPED_BC_FULL_OUTPUT": "STRUCTURAL_UNAVAILABLE",
    }
    if plant_fake_splitk:
        structural["GROUPED_SPLITK_S4"] = "AVAILABLE"
    manifest = {
        "schema": SCHEMA,
        "identity": {
            "qtype": qtype,
            "format": matrix.format_for(qtype).name,
            "weight_layout": layout,
            "weight_layout_name": matrix.topology.layout_name(layout),
            "artifact_tile_k": 0,
            "bchunk": 0,
            "packed_scale": 1,
            "metadata": "PACKED_UNITS_SCALE_AND_ZERO",
        },
        "denominator": {
            "source_raw_rows": matrix.topology.RAW_ROWS_PER_PAIR,
            "admitted_topologies": len(topology),
            "delivery_algorithm_expanded_type_rows": expected,
            "compiled_rows": len(selected),
        },
        "parent_range": {
            "begin": parent_begin,
            "end": parent_end,
            "count": len(selected),
            "authority_count": expected,
        },
        "fixture_contract": {
            "ragged_offsets": True,
            "empty_experts": "ALLOWED_ZERO_AND_EXACT_COMPLEMENT",
            "active_experts": "REQUIRED_NONZERO_AND_MAY_EQUAL_EXPERTS",
            "correctness": "FULL_RAW_FP16_BEFORE_ANY_TIMING",
        },
        "confirmation": {
            "all_raw_bit_clean_candidates": True,
            "top_n": None,
            "screen_elimination": "NONE",
        },
        "grouped_parents": [
            row_json(qtype, row, delivery_n, persistent, ordinal)
            for ordinal, (row, delivery_n, algorithm) in enumerate(
                selected, start=parent_begin)
            for persistent in (int(algorithm == "GROUPED_PERSISTENT"),)
        ],
        "structural_algorithms": structural,
        "units": units,
        "authority_sha256": {
            "matrix": hashlib.sha256(pathlib.Path(matrix.__file__).read_bytes()).hexdigest(),
            "topology_matrix": matrix.topology.sha256(
                matrix.topology.ROOT / "tools/scalefirst_internal_matrix.py"),
            "emitter": matrix.topology.sha256(matrix.topology.EMITTER),
        },
    }
    validate_manifest(manifest)
    write(out / "manifest.json", json.dumps(manifest, indent=2,
                                              sort_keys=True) + "\n")
    return manifest


def validate_manifest(document: dict) -> None:
    if document.get("schema") != SCHEMA:
        raise ValueError("grouped FQ K-pack shard schema differs")
    identity = document.get("identity") or {}
    qtype = int(identity.get("qtype", -1))
    if qtype not in matrix.QTYPES or identity != {
            "qtype": qtype, "format": matrix.format_for(qtype).name,
            "weight_layout": matrix.layout_for(qtype),
            "weight_layout_name": matrix.topology.layout_name(matrix.layout_for(qtype)),
            "artifact_tile_k": 0, "bchunk": 0, "packed_scale": 1,
            "metadata": "PACKED_UNITS_SCALE_AND_ZERO"}:
        raise ValueError("grouped FQ K-pack identity differs")
    expected_topologies = len(matrix.admitted_rows(qtype))
    authority = list(matrix.grouped_rows(qtype))
    expected = len(authority)
    parent_range = document.get("parent_range") or {}
    begin = parent_range.get("begin")
    end = parent_range.get("end")
    if (not isinstance(begin, int) or isinstance(begin, bool) or
            not isinstance(end, int) or isinstance(end, bool) or
            begin < 0 or end <= begin or end > expected or
            end - begin > bundle_index.MAX_PARENTS_PER_BINARY or
            parent_range != {"begin": begin, "end": end,
                             "count": end - begin,
                             "authority_count": expected}):
        raise ValueError("grouped FQ K-pack parent range differs")
    expected_rows = [
        row_json(qtype, row, delivery_n,
                 int(algorithm == "GROUPED_PERSISTENT"), ordinal)
        for ordinal, (row, delivery_n, algorithm) in enumerate(
            authority[begin:end], start=begin)]
    rows = document.get("grouped_parents")
    compiled = end - begin
    if (not isinstance(rows, list) or rows != expected_rows or
            len({row.get("static_candidate_id") for row in rows}) != compiled or
            len({row.get("symbol") for row in rows}) != compiled):
        raise ValueError("grouped FQ K-pack compiled denominator differs")
    per_algorithm = {name: 0 for name in matrix.GROUPED_ALGORITHMS}
    for row in rows:
        algorithm = row.get("algorithm")
        if algorithm not in per_algorithm or \
                row.get("weight_layout") != matrix.layout_for(qtype) or \
                row.get("artifact_tile_k") != 0 or row.get("bchunk") != 0 or \
                row.get("a_provider") != 0 or \
                row.get("resolved_delivery_n") not in \
                    matrix.topology.resolved_delivery_ns(row["tile_n"]) or \
                row.get("correctness") != \
                    "REAL_RAGGED_OFFSETS_EMPTY_EXPERT_RAW_FP16_BEFORE_TIMING":
            raise ValueError("grouped FQ K-pack row identity differs")
        per_algorithm[algorithm] += 1
    expected_per_algorithm = {
        algorithm: sum(row["algorithm"] == algorithm for row in expected_rows)
        for algorithm in matrix.GROUPED_ALGORITHMS}
    if per_algorithm != expected_per_algorithm:
        raise ValueError("grouped FQ K-pack algorithm expansion differs")
    if document.get("denominator") != {
            "source_raw_rows": matrix.topology.RAW_ROWS_PER_PAIR,
            "admitted_topologies": expected_topologies,
            "delivery_algorithm_expanded_type_rows": expected,
            "compiled_rows": compiled}:
        raise ValueError("grouped FQ K-pack aggregate denominator differs")
    if document.get("fixture_contract") != {
            "ragged_offsets": True,
            "empty_experts": "ALLOWED_ZERO_AND_EXACT_COMPLEMENT",
            "active_experts": "REQUIRED_NONZERO_AND_MAY_EQUAL_EXPERTS",
            "correctness": "FULL_RAW_FP16_BEFORE_ANY_TIMING"}:
        raise ValueError("grouped FQ K-pack fixture contract differs")
    if document.get("confirmation") != {
            "all_raw_bit_clean_candidates": True, "top_n": None,
            "screen_elimination": "NONE"}:
        raise ValueError("grouped FQ K-pack introduced top-N filtering")
    if document.get("structural_algorithms") != {
            "GROUPED_SPLITK_S2": "STRUCTURAL_UNAVAILABLE",
            "GROUPED_SPLITK_S4": "STRUCTURAL_UNAVAILABLE",
            "GROUPED_SPLITK_S8": "STRUCTURAL_UNAVAILABLE",
            "GROUPED_BC_FULL_OUTPUT": "STRUCTURAL_UNAVAILABLE"}:
        raise ValueError("grouped FQ K-pack fabricated Split-K or BC")


def self_test() -> None:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="fq-kpack-grouped-gen-") as tmp:
        out = pathlib.Path(tmp)
        for qtype in matrix.QTYPES:
            total = len(matrix.grouped_rows(qtype))
            seen = []
            for begin in range(0, total, bundle_index.MAX_PARENTS_PER_BINARY):
                count = min(bundle_index.MAX_PARENTS_PER_BINARY, total - begin)
                manifest = generate(qtype, out / f"{qtype}-{begin}", 4,
                                    parent_begin=begin, parent_count=count)
                validate_manifest(manifest)
                seen.extend(row["parent_ordinal"]
                            for row in manifest["grouped_parents"])
            if seen != list(range(total)):
                raise ValueError(f"q{qtype} grouped range union differs")
        failures = 0
        for kwargs in ({"plant_drop_middle": True},
                       {"plant_fake_splitk": True}):
            try:
                generate(12, out / f"negative-{failures}", 4, **kwargs)
            except (RuntimeError, ValueError):
                failures += 1
        if failures != 2:
            raise ValueError("grouped generator negative plant stayed green")
        total = len(matrix.grouped_rows(12))
        for begin, count in ((-1, 1), (total, 1), (0, 33), (total - 1, 2)):
            try:
                generate(12, out / f"range-negative-{failures}", 4,
                         parent_begin=begin, parent_count=count)
            except ValueError:
                failures += 1
        if failures != 6:
            raise ValueError("grouped out-of-range/oversize plant stayed green")
    print("[fq-kpack-grouped-generate:self-test] PASS formats=5 NP+P "
          "parent-range<=32 exact-union AP0 unique-resolved-DN "
          "ragged+empty+raw-bit "
          "missing-row+fake-availability+range=RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtype", type=int)
    parser.add_argument("--out-dir", type=pathlib.Path)
    parser.add_argument("--per-unit", type=int, default=4)
    parser.add_argument("--parent-begin", type=int, default=0)
    parser.add_argument("--parent-count", type=int)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--plant-drop-middle", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--plant-fake-splitk", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if args.qtype is None or args.out_dir is None:
            raise ValueError("--qtype and --out-dir are required")
        manifest = generate(args.qtype, args.out_dir, args.per_unit,
                            args.plant_drop_middle, args.plant_fake_splitk,
                            args.parent_begin, args.parent_count)
        den = manifest["denominator"]
        print("[fq-kpack-grouped-generate] PASS "
              f"q={args.qtype} layout={manifest['identity']['weight_layout']} "
              f"raw={den['source_raw_rows']} compiled={den['compiled_rows']}/"
              f"{den['delivery_algorithm_expanded_type_rows']} "
              f"units={len(manifest['units'])} "
              f"range=[{manifest['parent_range']['begin']},"
              f"{manifest['parent_range']['end']}) "
              "NP+P splitk+BC=STRUCTURAL_UNAVAILABLE")
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"[fq-kpack-grouped-generate] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
