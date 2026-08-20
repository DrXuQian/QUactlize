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
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def symbol(qtype: int, artifact: int, row: matrix.Tactic) -> str:
    return (f"sf_q{qtype}_a{artifact}_tm{row.tile_m}_tn{row.tile_n}_"
            f"tk{row.tactic_tile_k}_wm{row.warp_m}_wn{row.warp_n}_"
            f"s{row.stages}_bc{row.bchunk}")


def macro(name: str, rows: list[matrix.Tactic], qtype: int,
          artifact: int) -> str:
    if not rows:
        return f"#define {name}(X)\n"
    lines = [f"#define {name}(X) \\"]
    for index, row in enumerate(rows):
        continuation = " \\" if index + 1 < len(rows) else ""
        lines.append(
            f"  X({symbol(qtype, artifact, row)},{qtype},{artifact},"
            f"{row.tile_m},{row.tile_n},{row.tactic_tile_k},"
            f"{row.warp_m},{row.warp_n},{row.stages},{row.bchunk})"
            f"{continuation}")
    return "\n".join(lines) + "\n"


def row_json(qtype: int, artifact: int, row: matrix.Tactic,
             status: str, reason: str) -> dict:
    return {
        "symbol": symbol(qtype, artifact, row),
        "qtype": qtype,
        "artifact_tile_k": artifact,
        "tile_m": row.tile_m, "tile_n": row.tile_n,
        "tactic_tile_k": row.tactic_tile_k,
        "warp_m": row.warp_m, "warp_n": row.warp_n,
        "stages": row.stages, "bchunk": row.bchunk,
        "fold_low": row.fold_low, "fold_high": row.fold_high,
        "status": status, "reason": reason,
    }


def generate(qtype: int, artifact: int, bchunk: int, out: pathlib.Path,
             per_unit: int, drop_last: bool,
             select_symbol: str | None = None) -> dict:
    fmt = matrix.format_for(qtype)
    if artifact not in matrix.ARTIFACT_TILE_K:
        raise ValueError("artifact-tk must be 32,64,128,256")
    if bchunk not in (0, 1) or per_unit <= 0:
        raise ValueError("bchunk must be 0/1 and per-unit must be positive")
    raw = [row for row in matrix.emitted_tactics(qtype, artifact)
           if row.bchunk == bchunk]
    if len(raw) != matrix.RAW_ROWS_PER_PAIR // 2:
        raise RuntimeError(f"raw bchunk denominator {len(raw)}/11520")
    classified = [(row, *matrix.classify(fmt, artifact, row)) for row in raw]
    eligible = [row for row, status, _ in classified
                if status == "TYPE_ADMISSION_REQUIRED"]
    expected = len(eligible)
    if drop_last and eligible:
        eligible.pop()
    if len(eligible) != expected:
        raise RuntimeError(
            f"typed denominator {len(eligible)}/{expected}; candidate missing")

    authority_eligible = eligible
    if select_symbol is not None:
        eligible = [row for row in authority_eligible
                    if symbol(qtype, artifact, row) == select_symbol]
        if len(eligible) != 1:
            raise RuntimeError(
                f"selected symbol is not one exact typed row: {select_symbol}")

    registry = (
        "// GENERATED -- exact all-format ScaleFirst typed registry.\n"
        f"#define SCALEFIRST_GENERATED_QTYPE {qtype}\n"
        f"#define SCALEFIRST_GENERATED_ARTIFACT_TK {artifact}\n"
        f"#define SCALEFIRST_GENERATED_BCHUNK {bchunk}\n"
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
            "#define PPU_PACKED_SCALE 0\n"
            "#ifdef PPU_B_CHUNK\n#undef PPU_B_CHUNK\n#endif\n"
            f"#define PPU_B_CHUNK {bchunk}\n" +
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

    manifest = {
        "schema": "quactlize.scalefirst.generated_shard.v2",
        "identity": {"qtype": qtype, "format": fmt.name,
                     "artifact_tile_k": artifact, "bchunk": bchunk},
        "runtime": {
            "full_output": ["NONPERSISTENT", "PERSISTENT_CAPACITY_BALANCED"],
            "producer_only": list(matrix.PRODUCER_ONLY_ALGORITHMS),
            "shape": "runtime M,N,K",
        },
        "denominator": {
            "raw_rows": len(raw), "typed_rows": len(eligible),
            "non_typed_rows": len(raw) - len(authority_eligible),
            "persistent_grid_rows": "runtime exact occupancy expansion",
        },
        "typed_rows": [row_json(qtype, artifact, row,
                                "TYPE_ADMISSION_REQUIRED", "COMPILED_TYPE")
                       for row in eligible],
        "non_typed_rows": [row_json(qtype, artifact, row, status, reason)
                           for row, status, reason in classified
                           if status != "TYPE_ADMISSION_REQUIRED"],
        "units": units,
        "authority_sha256": {
            "matrix": hashlib.sha256(
                pathlib.Path(matrix.__file__).read_bytes()).hexdigest(),
            "emitter": matrix.sha256(matrix.EMITTER),
            "tactic_space": matrix.sha256(matrix.TACTIC_SPACE),
        },
    }
    if select_symbol is not None:
        manifest["selection"] = {
            "mode": "exact-symbol",
            "symbol": select_symbol,
            "authority_typed_rows": len(authority_eligible),
            "compiled_rows": 1,
        }
    write(out / "manifest.json", json.dumps(manifest, indent=2,
                                              sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtype", type=int, required=True)
    parser.add_argument("--artifact-tk", type=int, required=True)
    parser.add_argument("--bchunk", type=int, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--per-unit", type=int, default=4)
    parser.add_argument("--select-symbol",
                        help="compile exactly one typed row (diagnostic only)")
    parser.add_argument("--plant-drop-last", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        manifest = generate(args.qtype, args.artifact_tk, args.bchunk,
                            args.out_dir, args.per_unit,
                            args.plant_drop_last, args.select_symbol)
        den = manifest["denominator"]
        print("[scalefirst-generate] " + " ".join(
            f"{key}={value}" for key, value in manifest["identity"].items()) +
            f" raw={den['raw_rows']} typed={den['typed_rows']} "
            f"non_typed={den['non_typed_rows']} units={len(manifest['units'])}")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[scalefirst-generate] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
