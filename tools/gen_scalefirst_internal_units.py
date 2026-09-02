#!/usr/bin/env python3
"""Generate one exact ScaleFirst (qtype, ArtifactTileK, BChunk) shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import scalefirst_internal_matrix as matrix


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


def symbol(qtype: int, artifact: int, row: matrix.Tactic,
           provider: int = 0, delivery_n: int = 0) -> str:
    return (f"sf_q{qtype}_a{artifact}_tm{row.tile_m}_tn{row.tile_n}_"
            f"tk{row.tactic_tile_k}_wm{row.warp_m}_wn{row.warp_n}_"
            f"s{row.stages}_bc{row.bchunk}_ap{provider}_dn{delivery_n}")


def macro(name: str, rows: list[tuple[matrix.Tactic, int, int]], qtype: int,
          artifact: int) -> str:
    if not rows:
        return f"#define {name}(X)\n"
    lines = [f"#define {name}(X) \\"]
    for index, (row, provider, delivery_n) in enumerate(rows):
        continuation = " \\" if index + 1 < len(rows) else ""
        lines.append(
            f"  X({symbol(qtype, artifact, row, provider, delivery_n)},{qtype},{artifact},"
            f"{row.tile_m},{row.tile_n},{row.tactic_tile_k},"
            f"{row.warp_m},{row.warp_n},{row.stages},{row.bchunk},"
            f"{provider},{delivery_n})"
            f"{continuation}")
    return "\n".join(lines) + "\n"


def row_json(qtype: int, artifact: int, row: matrix.Tactic,
             status: str, reason: str, parent_id: int | None = None,
             provider: int = 0, delivery_n: int = 0) -> dict:
    candidate_symbol = symbol(qtype, artifact, row, provider, delivery_n)
    result = {
        "symbol": candidate_symbol,
        "static_candidate_id": candidate_symbol,
        "config_name": (f"{row.tile_m}x{row.tile_n}x{row.tactic_tile_k}_"
                        f"w{row.warp_m}x{row.warp_n}_s{row.stages}_"
                        f"bc{row.bchunk}_ap{provider}_dn{delivery_n}"),
        "qtype": qtype,
        "artifact_tile_k": artifact,
        "tile_m": row.tile_m, "tile_n": row.tile_n,
        "tactic_tile_k": row.tactic_tile_k,
        "warp_m": row.warp_m, "warp_n": row.warp_n,
        "stages": row.stages, "bchunk": row.bchunk,
        "a_provider": provider, "resolved_delivery_n": delivery_n,
        "fold_low": row.fold_low, "fold_high": row.fold_high,
        "status": status, "reason": reason,
    }
    if parent_id is not None:
        result["parent_id"] = parent_id
    return result


def generate(qtype: int, artifact: int, bchunk: int, out: pathlib.Path,
             per_unit: int, drop_last: bool,
             select_symbol: str | None = None,
             weight_layout: int = matrix.XPLANE_LAYOUT,
             parent_begin: int = 0,
             parent_count: int | None = None) -> dict:
    fmt = matrix.format_for(qtype)
    if not matrix.pair_supported(fmt, artifact, weight_layout):
        raise ValueError(
            "layout tuple must be Xplane/A32..256, Q4 K-pack/A0, or "
            "Q2/Q3/Q5/Q6 K-pack/A0")
    if bchunk not in (0, 1) or per_unit <= 0:
        raise ValueError("bchunk must be 0/1 and per-unit must be positive")
    raw = [row for row in matrix.emitted_tactics(
               qtype, artifact, weight_layout=weight_layout)
           if row.bchunk == bchunk]
    if len(raw) != matrix.RAW_ROWS_PER_PAIR // 2:
        raise RuntimeError(f"raw bchunk denominator {len(raw)}/11520")
    classified = [(row, *matrix.classify(
        fmt, artifact, row, weight_layout)) for row in raw]
    eligible = [row for row, status, _ in classified
                if status == "TYPE_ADMISSION_REQUIRED"]
    typed_topology_count = len(eligible)
    authority_eligible = (matrix.kpack_dense_candidates(qtype)
                          if weight_layout != matrix.XPLANE_LAYOUT else
                          [(row, 0, 0) for row in eligible])
    authority_eligible = [item for item in authority_eligible
                          if item[0].bchunk == bchunk]
    expected = len(authority_eligible)
    if drop_last and authority_eligible:
        authority_eligible.pop()
    if len(authority_eligible) != expected:
        raise RuntimeError(
            f"typed denominator {len(authority_eligible)}/{expected}; candidate missing")

    if parent_begin < 0 or (parent_count is not None and parent_count <= 0):
        raise ValueError("parent begin must be nonnegative and count positive")
    if select_symbol is not None and (parent_begin != 0 or parent_count is not None):
        raise ValueError("exact-symbol and parent-range selection are mutually exclusive")
    if select_symbol is not None:
        matches = [(index, item) for index, item in enumerate(authority_eligible)
                   if symbol(qtype, artifact, *item) == select_symbol]
        if len(matches) != 1:
            raise RuntimeError(
                f"selected symbol is not one exact typed row: {select_symbol}")
        parent_begin, item = matches[0]
        parent_end = parent_begin + 1
        eligible = [item]
        selection_mode = "exact-symbol"
    else:
        if parent_begin >= expected:
            raise ValueError(
                f"parent range begins outside authority: {parent_begin}/{expected}")
        parent_end = expected if parent_count is None else parent_begin + parent_count
        if parent_end > expected:
            raise ValueError(
                f"parent range exceeds authority: [{parent_begin},{parent_end})/{expected}")
        eligible = authority_eligible[parent_begin:parent_end]
        selection_mode = ("authority-full" if parent_begin == 0 and
                          parent_end == expected else "parent-range")
    parent_ids = list(range(parent_begin, parent_end))

    registry = (
        "// GENERATED -- exact all-format ScaleFirst typed registry.\n"
        f"#define SCALEFIRST_GENERATED_QTYPE {qtype}\n"
        f"#define SCALEFIRST_GENERATED_ARTIFACT_TK {artifact}\n"
        f"#define SCALEFIRST_GENERATED_BCHUNK {bchunk}\n"
        f"#define SCALEFIRST_GENERATED_WEIGHT_LAYOUT {weight_layout}\n"
        f"#define SCALEFIRST_GENERATED_RAW_ROWS {len(raw)}\n"
        f"#define SCALEFIRST_GENERATED_TYPED_ROWS {len(eligible)}\n" +
        macro("SCALEFIRST_REGISTRY_ROWS", eligible, qtype, artifact))
    write(out / "scalefirst_registry.inc", registry)

    units = []
    for unit_index, begin in enumerate(range(0, len(eligible), per_unit)):
        batch = eligible[begin:begin + per_unit]
        path = out / "units" / f"scalefirst_unit_{unit_index:05d}.cu"
        body = (
            "// GENERATED -- real ScaleFirst shipping/persistent/Split-K types.\n"
            "#ifdef PPU_PACKED_SCALE\n#undef PPU_PACKED_SCALE\n#endif\n"
            "#define PPU_PACKED_SCALE 0\n" +
            macro("SCALEFIRST_UNIT_ROWS", batch, qtype, artifact) +
            '#include "scalefirst_internal_sweep_unit.inc"\n')
        write(path, body)
        units.append(str(path))

    write(out / "units.cmake",
          "# GENERATED; one authority, no copied registry.\n"
          "set(SCALEFIRST_GENERATED_UNIT_SOURCES\n" +
          "".join(f'  "{path}"\n' for path in units) + ")\n" +
          f'set(SCALEFIRST_GENERATED_REGISTRY "{out / "scalefirst_registry.inc"}")\n' +
          f'set(SCALEFIRST_GENERATED_MANIFEST "{out / "manifest.json"}")\n')

    identity = {"qtype": qtype, "format": fmt.name,
                "artifact_tile_k": artifact, "bchunk": bchunk}
    if weight_layout != matrix.XPLANE_LAYOUT:
        identity.update({
            "weight_layout": weight_layout,
            "weight_layout_name": matrix.layout_name(weight_layout),
        })
    non_typed = [row_json(qtype, artifact, row, status, reason)
                 for row, status, reason in classified
                 if status != "TYPE_ADMISSION_REQUIRED"]
    non_typed_encoded = json.dumps(
        non_typed, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "schema": "quactlize.scalefirst.generated_shard.v3",
        "identity": identity,
        "runtime": {
            "full_output": ["NONPERSISTENT", "PERSISTENT_CAPACITY_BALANCED"],
            "producer_only": list(matrix.PRODUCER_ONLY_ALGORITHMS),
            "shape": "runtime M,N,K",
        },
        "denominator": {
            "raw_rows": len(raw), "typed_rows": len(eligible),
            "authority_typed_rows": len(authority_eligible),
            "non_typed_rows": len(raw) - typed_topology_count,
            "persistent_grid_rows": "runtime exact occupancy expansion",
        },
        "parent_range": {
            "begin": parent_begin, "end": parent_end,
            "count": len(eligible),
            "authority_count": len(authority_eligible),
        },
        "compiled_parents": [
            ({"parent_id": parent_id,
              "symbol": symbol(qtype, artifact, *item)} |
             ({"static_candidate_id": symbol(qtype, artifact, *item),
               "a_provider": item[1], "resolved_delivery_n": item[2]}
              if weight_layout != matrix.XPLANE_LAYOUT else {}))
            for parent_id, item in zip(parent_ids, eligible)],
        "typed_rows": [row_json(
            qtype, artifact, item[0], "TYPE_ADMISSION_REQUIRED", "COMPILED_TYPE",
            parent_id, item[1], item[2])
            for parent_id, item in zip(parent_ids, eligible)],
        # Range shards bind the complete rejection authority without
        # duplicating roughly ten thousand verbose rows in every manifest.
        "non_typed_authority": {
            "count": len(non_typed),
            "sha256": hashlib.sha256(non_typed_encoded).hexdigest(),
            "encoding": "JSON_SORT_KEYS_COMPACT_V1",
        },
        "units": units,
        "authority_sha256": {
            "matrix": hashlib.sha256(
                pathlib.Path(matrix.__file__).read_bytes()).hexdigest(),
            "emitter": matrix.sha256(matrix.EMITTER),
            "tactic_space": matrix.sha256(matrix.TACTIC_SPACE),
        },
    }
    if weight_layout != matrix.XPLANE_LAYOUT:
        manifest["runtime"]["cuda"] = {
            "status": "STRUCTURAL_UNAVAILABLE",
            "reason": "NO_CANONICAL_KPACK_CUDA_READER",
        }
    manifest["selection"] = {
        "mode": selection_mode,
        "begin": parent_begin,
        "end": parent_end,
        "authority_typed_rows": len(authority_eligible),
        "compiled_rows": len(eligible),
    }
    # Preserve the legacy expanded view only for the one full-authority
    # invocation. Parent ranges and exact-symbol diagnostics use the compact
    # count/hash authority above.
    if selection_mode == "authority-full":
        manifest["non_typed_rows"] = non_typed
    if select_symbol is not None:
        manifest["selection"]["symbol"] = select_symbol
    write(out / "manifest.json", json.dumps(manifest, indent=2,
                                              sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtype", type=int, required=True)
    parser.add_argument("--artifact-tk", type=int, required=True)
    parser.add_argument("--bchunk", type=int, required=True)
    parser.add_argument("--weight-layout", type=int, default=0,
                        choices=(0, 1, 2),
                        help="0=Xplane, 1=Q4 K-pack, 2=generic K-pack")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--per-unit", type=int, default=4)
    parser.add_argument("--parent-begin", type=int, default=0)
    parser.add_argument("--parent-count", type=int)
    parser.add_argument("--select-symbol",
                        help="compile exactly one typed row (diagnostic only)")
    parser.add_argument("--plant-drop-last", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        manifest = generate(args.qtype, args.artifact_tk, args.bchunk,
                            args.out_dir, args.per_unit,
                            args.plant_drop_last, args.select_symbol,
                            args.weight_layout, args.parent_begin,
                            args.parent_count)
        den = manifest["denominator"]
        print("[scalefirst-generate] " + " ".join(
            f"{key}={value}" for key, value in manifest["identity"].items()) +
            f" raw={den['raw_rows']} typed={den['typed_rows']} "
            f"authority_typed={den['authority_typed_rows']} "
            f"range=[{manifest['parent_range']['begin']},"
            f"{manifest['parent_range']['end']}) "
            f"non_typed={den['non_typed_rows']} units={len(manifest['units'])}")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[scalefirst-generate] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
