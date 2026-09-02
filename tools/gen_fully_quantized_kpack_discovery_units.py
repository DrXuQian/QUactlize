#!/usr/bin/env python3
"""Generate exhaustive dense FullyQuantized canonical K-pack type shards.

One invocation owns one qtype and its canonical layout.  It emits every
statically admitted topology, expands the legal AP0/AP1 axis, and never ranks
or truncates the resulting set.  S1/S2/S4/S8 remain runtime variants of one
compiled parent type.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import fully_quantized_kpack_discovery_matrix as matrix
import fully_quantized_kpack_bundle_index as bundle_index


SCHEMA = "quactlize.fully_quantized_kpack_dense_shard.v2"


def write(path: pathlib.Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def symbol(qtype: int, row, provider: int, delivery_n: int) -> str:
    return (f"fqk_tc_q{qtype}_l{matrix.layout_for(qtype)}_a0_"
            f"tm{row.tile_m}_tn{row.tile_n}_tk{row.tactic_tile_k}_"
            f"wm{row.warp_m}_wn{row.warp_n}_s{row.stages}_bc0_"
            f"ap{provider}_dn{delivery_n}")


def macro(name: str, qtype: int,
          rows: list[tuple[object, int, int]]) -> str:
    if not rows:
        return f"#define {name}(X)\n"
    lines = [f"#define {name}(X) \\"]
    for index, (row, provider, delivery_n) in enumerate(rows):
        tail = " \\" if index + 1 < len(rows) else ""
        lines.append(
            f"  X({symbol(qtype, row, provider, delivery_n)},{qtype},0,"
            f"{row.tile_m},{row.tile_n},{row.tactic_tile_k},"
            f"{row.warp_m},{row.warp_n},{row.stages},0,{provider},"
            f"{delivery_n}){tail}")
    return "\n".join(lines) + "\n"


def row_json(qtype: int, row, provider: int, delivery_n: int,
             parent_ordinal: int) -> dict:
    return {
        "parent_ordinal": parent_ordinal,
        "static_candidate_id": matrix.parent_id(
            qtype, row, provider, delivery_n, "dense"),
        "symbol": symbol(qtype, row, provider, delivery_n),
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
        "a_provider": provider,
        "a_provider_name": "packed-row" if provider else "standard-aiu",
        "resolved_delivery_n": delivery_n,
        "runtime_variants": [f"TC_S{split}" for split in matrix.SPLITS],
    }


def generate(qtype: int, out: pathlib.Path, per_unit: int,
             plant_drop_middle: bool = False,
             plant_wrong_layout: bool = False,
             parent_begin: int = 0,
             parent_count: int | None = None) -> dict:
    if (qtype not in matrix.QTYPES or per_unit <= 0 or
            per_unit > bundle_index.MAX_PARENTS_PER_BINARY):
        raise ValueError(
            "qtype must be Q2/Q3/Q4/Q5/Q6 and per-unit must be in [1,32]")
    layout = matrix.layout_for(qtype)
    authority = list(matrix.provider_rows(qtype))
    expected = len(authority)
    if plant_drop_middle and authority:
        authority.pop(len(authority) // 2)
    if len(authority) != expected:
        raise RuntimeError(f"dense typed denominator {len(authority)}/{expected}; row missing")
    emitted_layout = 1 if plant_wrong_layout and layout == 2 else layout
    if emitted_layout != layout:
        raise RuntimeError(
            f"qtype {qtype} generated layout {emitted_layout} differs from canonical {layout}")

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

    registry = (
        "// GENERATED -- exhaustive canonical K-pack FullyQuantized dense registry.\n"
        f"#define FQ_TC_GENERATED_QTYPE {qtype}\n"
        "#define FQ_TC_GENERATED_ARTIFACT_TK 0\n"
        "#define FQ_TC_GENERATED_BCHUNK 0\n"
        f"#define FQ_TC_GENERATED_WEIGHT_LAYOUT {layout}\n"
        f"#define FQ_TC_GENERATED_RAW_ROWS {matrix.topology.RAW_ROWS_PER_PAIR}\n"
        f"#define FQ_TC_GENERATED_TYPED_ROWS {len(selected)}\n" +
        macro("FQ_TC_REGISTRY_ROWS", qtype, selected))
    write(out / "fq_tc_registry.inc", registry)

    units = []
    for unit_index, begin in enumerate(range(0, len(selected), per_unit)):
        batch = selected[begin:begin + per_unit]
        path = out / "units" / f"fq_kpack_dense_unit_{unit_index:05d}.cu"
        write(path,
              "// GENERATED -- canonical K-pack FQ packed-metadata types.\n"
              "#ifdef PPU_PACKED_SCALE\n#undef PPU_PACKED_SCALE\n#endif\n"
              "#define PPU_PACKED_SCALE 1\n"
              f"#define FQ_TC_WEIGHT_LAYOUT {layout}\n" +
              macro("FQ_TC_UNIT_ROWS", qtype, batch) +
              '#include "fully_quantized_splitk_producer_unit.inc"\n')
        units.append(str(path))
    write(out / "units.cmake",
          "# GENERATED -- canonical K-pack dense discovery sources.\n"
          "set(FQ_TC_GENERATED_UNIT_SOURCES\n" +
          "".join(f'  "{path}"\n' for path in units) + ")\n"
          f'set(FQ_TC_GENERATED_REGISTRY "{out / "fq_tc_registry.inc"}")\n'
          f'set(FQ_TC_GENERATED_MANIFEST "{out / "manifest.json"}")\n')

    rows = [row_json(qtype, row, provider, delivery_n, ordinal)
            for ordinal, (row, provider, delivery_n) in enumerate(
                selected, start=parent_begin)]
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
            "admitted_topologies": len(matrix.admitted_rows(qtype)),
            "provider_expanded_rows": expected,
            "compiled_rows": len(selected),
            "runtime_tc_cells": len(selected) * len(matrix.SPLITS),
        },
        "parent_range": {
            "begin": parent_begin,
            "end": parent_end,
            "count": len(selected),
            "authority_count": expected,
        },
        "confirmation": {
            "all_raw_bit_clean_candidates": True,
            "top_n": None,
            "screen_elimination": "NONE",
        },
        "dense_tc_parents": rows,
        "dense_bc": {
            "algorithm": "DENSE_BC_FULL_OUTPUT",
            "status": "STRUCTURAL_UNAVAILABLE",
            "reason": matrix.BC_REASON,
        },
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
        raise ValueError("dense FQ K-pack shard schema differs")
    identity = document.get("identity") or {}
    qtype = int(identity.get("qtype", -1))
    if qtype not in matrix.QTYPES or identity != {
            "qtype": qtype, "format": matrix.format_for(qtype).name,
            "weight_layout": matrix.layout_for(qtype),
            "weight_layout_name": matrix.topology.layout_name(matrix.layout_for(qtype)),
            "artifact_tile_k": 0, "bchunk": 0, "packed_scale": 1,
            "metadata": "PACKED_UNITS_SCALE_AND_ZERO"}:
        raise ValueError("dense FQ K-pack shard identity differs")
    authority = list(matrix.provider_rows(qtype))
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
        raise ValueError("dense FQ K-pack parent range differs")
    expected_rows = [row_json(qtype, row, provider, delivery_n, ordinal)
                     for ordinal, (row, provider, delivery_n) in enumerate(
                         authority[begin:end], start=begin)]
    rows = document.get("dense_tc_parents")
    compiled = end - begin
    if (not isinstance(rows, list) or rows != expected_rows or
            len({row.get("static_candidate_id") for row in rows}) != compiled or
            len({row.get("symbol") for row in rows}) != compiled):
        raise ValueError("dense FQ K-pack compiled denominator differs")
    for row in rows:
        if (row.get("weight_layout") != matrix.layout_for(qtype) or
                row.get("artifact_tile_k") != 0 or row.get("bchunk") != 0 or
                row.get("runtime_variants") != ["TC_S1", "TC_S2", "TC_S4", "TC_S8"]):
            raise ValueError("dense FQ K-pack row identity differs")
        if row.get("a_provider") not in (0, 1):
            raise ValueError("dense FQ K-pack provider differs")
        if row.get("a_provider") == 1 and not (
                qtype in (10, 12) and row.get("tile_m") == 8 and
                row.get("warp_m") == 8):
            raise ValueError("dense FQ K-pack AP1 row is illegal")
        if row.get("resolved_delivery_n") not in \
                matrix.topology.resolved_delivery_ns(row["tile_n"]):
            raise ValueError("dense FQ K-pack delivery row is illegal")
    denominator = document.get("denominator") or {}
    if denominator != {
            "source_raw_rows": matrix.topology.RAW_ROWS_PER_PAIR,
            "admitted_topologies": len(matrix.admitted_rows(qtype)),
            "provider_expanded_rows": expected,
            "compiled_rows": compiled,
            "runtime_tc_cells": compiled * len(matrix.SPLITS)}:
        raise ValueError("dense FQ K-pack aggregate denominator differs")
    if document.get("dense_bc") != {
            "algorithm": "DENSE_BC_FULL_OUTPUT",
            "status": "STRUCTURAL_UNAVAILABLE", "reason": matrix.BC_REASON}:
        raise ValueError("dense FQ K-pack fabricated a BC candidate")
    if document.get("confirmation") != {
            "all_raw_bit_clean_candidates": True, "top_n": None,
            "screen_elimination": "NONE"}:
        raise ValueError("dense FQ K-pack introduced top-N filtering")


def self_test() -> None:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="fq-kpack-dense-gen-") as tmp:
        out = pathlib.Path(tmp)
        for qtype in matrix.QTYPES:
            total = len(matrix.provider_rows(qtype))
            seen = []
            for begin in range(0, total, bundle_index.MAX_PARENTS_PER_BINARY):
                count = min(bundle_index.MAX_PARENTS_PER_BINARY, total - begin)
                manifest = generate(qtype, out / f"{qtype}-{begin}", 4,
                                    parent_begin=begin, parent_count=count)
                validate_manifest(manifest)
                seen.extend(row["parent_ordinal"]
                            for row in manifest["dense_tc_parents"])
            if seen != list(range(total)):
                raise ValueError(f"q{qtype} dense range union differs")
        failures = 0
        for kwargs in ({"plant_drop_middle": True},
                       {"plant_wrong_layout": True}):
            try:
                generate(10, out / f"negative-{failures}", 4, **kwargs)
            except RuntimeError:
                failures += 1
        if failures != 2:
            raise ValueError("dense generator negative plant stayed green")
        planted = generate(11, out / "manifest-negative", 4)
        planted["dense_bc"]["status"] = "AVAILABLE"
        try:
            validate_manifest(planted)
        except ValueError:
            failures += 1
        if failures != 3:
            raise ValueError("fake BC manifest stayed green")
        for begin, count in ((-1, 1), (len(matrix.provider_rows(10)), 1),
                             (0, 33), (len(matrix.provider_rows(10)) - 1, 2)):
            try:
                generate(10, out / f"range-negative-{failures}", 4,
                         parent_begin=begin, parent_count=count)
            except ValueError:
                failures += 1
        if failures != 7:
            raise ValueError("dense out-of-range/oversize plant stayed green")
    print("[fq-kpack-dense-generate:self-test] PASS formats=5 "
          "parent-range<=32 exact-union unique-resolved-DN AP1xDN "
          "layout+missing-row+fake-bc+range negatives=RED")


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
    parser.add_argument("--plant-wrong-layout", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if args.qtype is None or args.out_dir is None:
            raise ValueError("--qtype and --out-dir are required")
        manifest = generate(args.qtype, args.out_dir, args.per_unit,
                            args.plant_drop_middle, args.plant_wrong_layout,
                            args.parent_begin, args.parent_count)
        den = manifest["denominator"]
        print("[fq-kpack-dense-generate] PASS "
              f"q={args.qtype} layout={manifest['identity']['weight_layout']} "
              f"raw={den['source_raw_rows']} compiled={den['compiled_rows']}/"
              f"{den['provider_expanded_rows']} cells={den['runtime_tc_cells']} "
              f"range=[{manifest['parent_range']['begin']},"
              f"{manifest['parent_range']['end']}) "
              f"units={len(manifest['units'])} BC=STRUCTURAL_UNAVAILABLE")
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"[fq-kpack-dense-generate] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
