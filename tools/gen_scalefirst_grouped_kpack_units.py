#!/usr/bin/env python3
"""Generate one canonical K-pack grouped ScaleFirst discovery shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import scalefirst_grouped_kpack_matrix as matrix


def write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def symbol(qtype: int, row, delivery_n: int) -> str:
    return (f"sfg_q{qtype}_tm{row.tile_m}_tn{row.tile_n}_"
            f"tk{row.tactic_tile_k}_wm{row.warp_m}_wn{row.warp_n}_"
            f"s{row.stages}_ap0_dn{delivery_n}")


def macro(name: str, qtype: int, layout: int, rows: list) -> str:
    if not rows:
        return f"#define {name}(X)\n"
    lines = [f"#define {name}(X) \\"]
    for index, (row, delivery_n) in enumerate(rows):
        tail = " \\" if index + 1 < len(rows) else ""
        lines.append(
            f"  X({symbol(qtype, row, delivery_n)},{qtype},{layout},{row.tile_m},"
            f"{row.tile_n},{row.tactic_tile_k},{row.warp_m},{row.warp_n},"
            f"{row.stages},{delivery_n}){tail}")
    return "\n".join(lines) + "\n"


def generate(qtype: int, out: pathlib.Path, per_unit: int,
             drop_last: bool = False, select_symbol: str | None = None,
             parent_begin: int = 0,
             parent_count: int | None = None) -> dict:
    if qtype not in matrix.QTYPES or per_unit <= 0:
        raise ValueError("qtype must be Q2/Q3/Q4/Q5/Q6 and per-unit positive")
    layout = matrix.layout_for(qtype)
    authority = list(matrix.candidate_rows(qtype))
    expected = len(authority)
    if drop_last and authority:
        authority.pop()
    if len(authority) != expected:
        raise RuntimeError(f"typed denominator {len(authority)}/{expected}; row missing")
    if parent_begin < 0 or (parent_count is not None and parent_count <= 0):
        raise ValueError("parent begin must be nonnegative and count positive")
    if select_symbol is not None and (parent_begin != 0 or parent_count is not None):
        raise ValueError("exact-symbol and parent-range selection are mutually exclusive")
    if select_symbol is not None:
        matches = [(index, item) for index, item in enumerate(authority)
                   if symbol(qtype, *item) == select_symbol]
        if len(matches) != 1:
            raise ValueError("selected symbol is not one admitted grouped row")
        parent_begin, item = matches[0]
        parent_end = parent_begin + 1
        rows = [item]
        selection_mode = "exact-symbol"
    else:
        if parent_begin >= expected:
            raise ValueError(
                f"parent range begins outside authority: {parent_begin}/{expected}")
        parent_end = expected if parent_count is None else parent_begin + parent_count
        if parent_end > expected:
            raise ValueError(
                f"parent range exceeds authority: [{parent_begin},{parent_end})/{expected}")
        rows = authority[parent_begin:parent_end]
        selection_mode = ("authority-full" if parent_begin == 0 and
                          parent_end == expected else "parent-range")
    parent_ids = list(range(parent_begin, parent_end))

    registry = (
        "// GENERATED -- canonical K-pack grouped ScaleFirst discovery.\n"
        f"#define SCALEFIRST_GROUPED_GENERATED_QTYPE {qtype}\n"
        f"#define SCALEFIRST_GROUPED_GENERATED_WEIGHT_LAYOUT {layout}\n"
        f"#define SCALEFIRST_GROUPED_GENERATED_TYPED_ROWS {len(rows)}\n" +
        macro("SCALEFIRST_GROUPED_REGISTRY_ROWS", qtype, layout, rows))
    write(out / "scalefirst_grouped_registry.inc", registry)

    units = []
    for unit_index, begin in enumerate(range(0, len(rows), per_unit)):
        batch = rows[begin:begin + per_unit]
        path = out / "units" / f"scalefirst_grouped_unit_{unit_index:05d}.cu"
        write(path,
              "// GENERATED -- grouped full-output K-pack types.\n"
              "#ifdef PPU_PACKED_SCALE\n#undef PPU_PACKED_SCALE\n#endif\n"
              "#define PPU_PACKED_SCALE 0\n" +
              macro("SCALEFIRST_GROUPED_UNIT_ROWS", qtype, layout, batch) +
              '#include "scalefirst_grouped_kpack_discovery_unit.inc"\n')
        units.append(str(path))
    write(out / "units.cmake",
          "# GENERATED -- grouped discovery sources.\n"
          "set(SCALEFIRST_GROUPED_GENERATED_UNIT_SOURCES\n" +
          "".join(f'  "{path}"\n' for path in units) + ")\n")

    row_json = lambda parent_id, item: {
        "parent_id": parent_id,
        "symbol": symbol(qtype, *item), "a_provider": 0,
        "static_candidate_id": symbol(qtype, *item),
        "config_name": (f"{item[0].tile_m}x{item[0].tile_n}x{item[0].tactic_tile_k}_"
                        f"w{item[0].warp_m}x{item[0].warp_n}_s{item[0].stages}_ap0_dn{item[1]}"),
        "resolved_delivery_n": item[1],
        **{name: getattr(item[0], name) for name in (
            "tile_m", "tile_n", "tactic_tile_k", "warp_m", "warp_n",
            "stages", "bchunk")},
    }
    manifest = {
        "schema": "quactlize.scalefirst_grouped_kpack_shard.v2",
        "identity": {
            "qtype": qtype,
            "weight_layout": layout,
            "weight_layout_name": matrix.dense.layout_name(layout),
            "artifact_tile_k": 0,
            "quant_mode": "ScaleZero",
            "metadata_planes": 2,
        },
        "algorithms": {
            "full_output": {
                "nonpersistent": "RAW_BIT_THEN_TIMING",
                "persistent": "RAW_BIT_THEN_TIMING",
            },
            "persistent": "AVAILABLE_RUNTIME_EXACT_OCCUPANCY_CAPACITY_BALANCED",
            "cuda": {
                "status": "STRUCTURAL_UNAVAILABLE",
                "reason": "NO_CANONICAL_KPACK_CUDA_READER",
            },
            "split_k": {f"S{split}": "STRUCTURAL_UNAVAILABLE"
                        for split in matrix.SPLITS},
        },
        "denominator": {
            "source_raw_rows": matrix.dense.RAW_ROWS_PER_PAIR,
            "authority_typed_rows": expected,
            "compiled_rows": len(rows),
        },
        "parent_range": {
            "begin": parent_begin, "end": parent_end,
            "count": len(rows), "authority_count": expected,
        },
        "compiled_parents": [
            {"parent_id": parent_id, "symbol": symbol(qtype, *item),
             "static_candidate_id": symbol(qtype, *item),
             "a_provider": 0, "resolved_delivery_n": item[1]}
            for parent_id, item in zip(parent_ids, rows)],
        "typed_rows": [row_json(parent_id, item)
                       for parent_id, item in zip(parent_ids, rows)],
        "units": units,
        "authority_sha256": {
            "matrix": hashlib.sha256(pathlib.Path(matrix.__file__).read_bytes()).hexdigest(),
            "dense_matrix": matrix.dense.sha256(matrix.dense.ROOT / "tools/scalefirst_internal_matrix.py"),
            "emitter": matrix.dense.sha256(matrix.dense.EMITTER),
        },
    }
    manifest["selection"] = {
        "mode": selection_mode,
        "begin": parent_begin,
        "end": parent_end,
        "authority_typed_rows": expected,
        "compiled_rows": len(rows),
    }
    if select_symbol is not None:
        manifest["selection"]["symbol"] = select_symbol
    write(out / "manifest.json", json.dumps(manifest, indent=2,
                                              sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtype", type=int, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--per-unit", type=int, default=4)
    parser.add_argument("--parent-begin", type=int, default=0)
    parser.add_argument("--parent-count", type=int)
    parser.add_argument("--select-symbol")
    parser.add_argument("--plant-drop-last", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        manifest = generate(args.qtype, args.out_dir, args.per_unit,
                            args.plant_drop_last, args.select_symbol,
                            args.parent_begin, args.parent_count)
        print("[sf-grouped-kpack-generate] PASS "
              f"q={args.qtype} layout={manifest['identity']['weight_layout']} "
              f"typed={manifest['denominator']['compiled_rows']}/"
              f"{manifest['denominator']['authority_typed_rows']} "
              f"range=[{manifest['parent_range']['begin']},"
              f"{manifest['parent_range']['end']}) "
              f"units={len(manifest['units'])} splitk=STRUCTURAL_UNAVAILABLE")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[sf-grouped-kpack-generate] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
