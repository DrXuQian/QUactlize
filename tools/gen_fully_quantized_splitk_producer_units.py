#!/usr/bin/env python3
"""Generate exact qtype10--14 FullyQuantized TC sweep shards.

The raw denominator comes from ``emit_fully_quantized_splitk_superset.cpp``.
Only rows that pass the shared non-smem/static producer guard instantiate a
device type; every omitted raw row remains in ``manifest.json`` with its named
reason.  The generated wrapper then lets the *real* Shipping/Split types decide
shared storage and ``can_implement`` for each runtime shape.

One invocation owns exactly one (qtype, ArtifactTileK, PPU_B_CHUNK) tuple.
That makes the translation-unit policy explicit and gives a shard a stable,
resume-safe identity.  S={1,2,4,8} is runtime data, not four duplicate types.
An optional TileM selection records both the complete source-typed denominator
and every excluded legal row; decode can therefore compile TM8 only without
pretending the generic tactic space was smaller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import fully_quantized_internal_matrix as matrix


def write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def symbol(qtype: int, artifact: int, tactic: matrix.Tactic,
           a_provider: int = 0) -> str:
    return (f"fq_tc_q{qtype}_a{artifact}_tm{tactic.tile_m}_tn{tactic.tile_n}_"
            f"tk{tactic.tactic_tile_k}_wm{tactic.warp_m}_wn{tactic.warp_n}_"
            f"s{tactic.stages}_bc{tactic.bchunk}_ap{a_provider}")


def macro(name: str, rows: list[tuple[matrix.Tactic, int]], qtype: int,
          artifact: int) -> str:
    lines = [f"#define {name}(X) \\"]
    for index, (row, a_provider) in enumerate(rows):
        continuation = " \\" if index + 1 < len(rows) else ""
        lines.append(
            f"  X({symbol(qtype, artifact, row, a_provider)},{qtype},{artifact},"
            f"{row.tile_m},{row.tile_n},{row.tactic_tile_k},"
            f"{row.warp_m},{row.warp_n},{row.stages},{row.bchunk},{a_provider})"
            f"{continuation}")
    if not rows:
        # A zero-row macro must still be syntactically usable as
        # FQ_TC_UNIT_ROWS(FQ_TC_DEFINE_WRAPPER).
        return f"#define {name}(X)\n"
    return "\n".join(lines) + "\n"


def row_json(qtype: int, artifact: int, row: matrix.Tactic,
             a_provider: int = 0) -> dict:
    return {
        "symbol": symbol(qtype, artifact, row, a_provider),
        "qtype": qtype,
        "artifact_tile_k": artifact,
        "tile_m": row.tile_m,
        "tile_n": row.tile_n,
        "tactic_tile_k": row.tactic_tile_k,
        "warp_m": row.warp_m,
        "warp_n": row.warp_n,
        "stages": row.stages,
        "bchunk": row.bchunk,
        "a_provider": "packed-row" if a_provider else "standard-aiu",
        "a_provider_capacity_rows": 1 if a_provider else None,
        "source_status": row.source_status,
        "source_reason": row.source_reason,
    }


def generate(qtype: int, artifact: int, bchunk: int, out: pathlib.Path,
             per_unit: int, drop_last: bool,
             tile_m_filter: int | None = None) -> dict:
    formats = {fmt.qtype: fmt for fmt in matrix.parse_formats()}
    if qtype not in (10, 11, 12, 13, 14):
        raise ValueError("qtype must be one of the shipping FQ formats 10..14")
    if artifact not in matrix.ARTIFACT_TILE_K:
        raise ValueError("artifact-tk must be one of 32,64,128,256")
    if bchunk not in matrix.BCHUNK_REQUESTS:
        raise ValueError("bchunk must be 0 or 1")
    if per_unit <= 0:
        raise ValueError("per-unit must be positive")
    if tile_m_filter is not None and tile_m_filter not in (
            8, 16, 32, 64, 128, 256):
        raise ValueError("tile-m-filter must be one of 8,16,32,64,128,256")

    raw_rows = [r for r in matrix.emitted_tactics(qtype, artifact)
                if r.bchunk == bchunk]
    if len(raw_rows) != 11520:
        raise RuntimeError(
            f"raw tuple denominator drifted: got {len(raw_rows)}, expected 11520")
    fmt = formats[qtype]
    def row_admitted(row: matrix.Tactic) -> bool:
        artifact_ok, _ = matrix.artifact_supported(
            fmt, artifact, row.tactic_tile_k)
        packed_ok, _ = matrix.packed_metadata_tactic_supported(
            fmt, row.tactic_tile_k)
        atom_ok, _ = matrix.collective_atom_tiling_supported(
            fmt, row, artifact)
        chunk_ok, _ = matrix.bchunk_state(fmt, row.bchunk)
        return (row.source_status == "TYPE_ADMISSION_REQUIRED" and
                artifact_ok and packed_ok and atom_ok and chunk_ok)
    all_rows: list[tuple[matrix.Tactic, int]] = []
    for row in raw_rows:
        all_rows.append((row, 0))
        if matrix.packed_a_provider_candidate(fmt, row, artifact):
            all_rows.append((row, 1))
    source_eligible = [(r, ap) for r, ap in all_rows if row_admitted(r)]
    eligible = [(r, ap) for r, ap in source_eligible
                if tile_m_filter is None or r.tile_m == tile_m_filter]
    selection_rejects = [(r, ap) for r, ap in source_eligible
                         if tile_m_filter is not None and
                         r.tile_m != tile_m_filter]
    static_rejects = [(r, ap) for r, ap in all_rows if not row_admitted(r)]
    if drop_last and eligible:
        eligible.pop()

    expected_eligible = sum(
        row_admitted(r) and
        (tile_m_filter is None or r.tile_m == tile_m_filter)
        for r, _ in all_rows)
    if len(eligible) != expected_eligible:
        raise RuntimeError(
            "generated typed denominator "
            f"{len(eligible)}/{expected_eligible}; a candidate row is missing")
    if len(eligible) + len(selection_rejects) + len(static_rejects) != \
            len(all_rows):
        raise RuntimeError("selected/source-eligible/static partition is not exact")

    # Registry is included only by the host main.  It has the complete typed
    # identities while manifest.json retains all static rejects.
    registry = (
        "// GENERATED -- exact FQ tensor-core typed registry.\n"
        f"#define FQ_TC_GENERATED_QTYPE {qtype}\n"
        f"#define FQ_TC_GENERATED_ARTIFACT_TK {artifact}\n"
        f"#define FQ_TC_GENERATED_BCHUNK {bchunk}\n"
        f"#define FQ_TC_GENERATED_RAW_ROWS {len(all_rows)}\n"
        f"#define FQ_TC_GENERATED_TYPED_ROWS {len(eligible)}\n" +
        macro("FQ_TC_REGISTRY_ROWS", eligible, qtype, artifact))
    write(out / "fq_tc_registry.inc", registry)

    units: list[str] = []
    for unit_index, begin in enumerate(range(0, len(eligible), per_unit)):
        batch = eligible[begin:begin + per_unit]
        path = out / "units" / f"fq_tc_unit_{unit_index:05d}.cu"
        body = (
            "// GENERATED -- real FQ Shipping/Split type shard.\n"
            "#ifdef PPU_PACKED_SCALE\n#undef PPU_PACKED_SCALE\n#endif\n"
            "#define PPU_PACKED_SCALE 1\n"
            "#ifdef PPU_PACKED_FORMAT\n#undef PPU_PACKED_FORMAT\n#endif\n"
            f"#define PPU_PACKED_FORMAT {formats[qtype].packed_format}\n"
            "#ifdef PPU_B_CHUNK\n#undef PPU_B_CHUNK\n#endif\n"
            f"#define PPU_B_CHUNK {bchunk}\n" +
            macro("FQ_TC_UNIT_ROWS", batch, qtype, artifact) +
            '#include "fully_quantized_splitk_producer_unit.inc"\n')
        write(path, body)
        units.append(str(path))

    cmake = (
        "# GENERATED; include from fq_internal_sweep.cmake.in.\n"
        "set(FQ_TC_GENERATED_UNIT_SOURCES\n" +
        "".join(f'  "{path}"\n' for path in units) + ")\n"
        f'set(FQ_TC_GENERATED_REGISTRY "{out / "fq_tc_registry.inc"}")\n'
        f'set(FQ_TC_GENERATED_MANIFEST "{out / "manifest.json"}")\n')
    write(out / "units.cmake", cmake)

    rejects = []
    for row, a_provider in static_rejects:
        artifact_ok, artifact_reason = matrix.artifact_supported(
            fmt, artifact, row.tactic_tile_k)
        packed_ok, packed_reason = matrix.packed_metadata_tactic_supported(
            fmt, row.tactic_tile_k)
        atom_ok, atom_reason = matrix.collective_atom_tiling_supported(
            fmt, row, artifact)
        chunk_ok, chunk_reason = matrix.bchunk_state(fmt, row.bchunk)
        reason = row.source_reason
        if row.source_status == "TYPE_ADMISSION_REQUIRED" and not artifact_ok:
            reason = artifact_reason
        elif row.source_status == "TYPE_ADMISSION_REQUIRED" and not packed_ok:
            reason = packed_reason
        elif row.source_status == "TYPE_ADMISSION_REQUIRED" and not atom_ok:
            reason = atom_reason
        elif row.source_status == "TYPE_ADMISSION_REQUIRED" and not chunk_ok:
            reason = chunk_reason
        entry = row_json(qtype, artifact, row, a_provider)
        entry["reason"] = reason
        rejects.append(entry)

    identity = {
        "qtype": qtype,
        "format": formats[qtype].name,
        "artifact_tile_k": artifact,
        "bchunk": bchunk,
    }
    if tile_m_filter is not None:
        identity["tile_m_filter"] = tile_m_filter
    selected_reject_rows = []
    for row, a_provider in selection_rejects:
        entry = row_json(qtype, artifact, row, a_provider)
        entry["reason"] = f"TILE_M_FILTER_NE_{tile_m_filter}"
        selected_reject_rows.append(entry)

    manifest = {
        "schema": "quactlize-fq-tc-generated-shard-v2",
        "identity": identity,
        "selection": {
            "tile_m_filter": tile_m_filter,
            "source_typed_rows": len(source_eligible),
            "selected_typed_rows": len(eligible),
            "selection_reject_rows": len(selection_rejects),
        },
        "runtime": {
            "splits": list(matrix.SPLITS),
            "metric_scope": {
                "S1": "FULL_OUTPUT",
                "S2_S4_S8": "PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS",
            },
            "shape": "runtime M,N,K; one compiled graph serves all shapes",
        },
        "placed_bc": {
            "compiled_in_bchunk0": matrix.bc_artifact_supported(
                fmt, artifact)[0],
            "reason": matrix.bc_artifact_supported(fmt, artifact)[1],
        },
        "tactic_tile_k_axis": {
            "full_legal_space": list(matrix.TACTIC_TILE_K),
            "default_fq_tactic_tile_k": fmt.fully_quantized_tile_k,
            "default_is_annotation_not_denominator": True,
        },
        "denominator": {
            "raw_topology_rows": len(raw_rows),
            "provider_expanded_rows": len(all_rows),
            "source_typed_rows": len(source_eligible),
            "typed_rows": len(eligible),
            "selection_reject_rows": len(selection_rejects),
            "static_reject_rows": len(static_rejects),
            "runtime_tc_cells": len(all_rows) * len(matrix.SPLITS),
            "typed_runtime_tc_cells": len(eligible) * len(matrix.SPLITS),
        },
        "typed_rows": [row_json(qtype, artifact, row, ap)
                       for row, ap in eligible],
        "selection_rejects": selected_reject_rows,
        "static_rejects": rejects,
        "units": units,
        "registry": str(out / "fq_tc_registry.inc"),
        "cmake_fragment": str(out / "units.cmake"),
        "authority_sha256": {
            "matrix": hashlib.sha256(
                pathlib.Path(matrix.__file__).read_bytes()).hexdigest(),
            "emitter": matrix.sha256(matrix.TACTIC_EMITTER),
            "tactic_space": matrix.sha256(matrix.TACTIC_SPACE),
        },
    }
    write(out / "manifest.json",
          json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtype", type=int, required=True)
    parser.add_argument("--artifact-tk", type=int, required=True)
    parser.add_argument("--bchunk", type=int, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--per-unit", type=int, default=1)
    parser.add_argument("--tile-m-filter", type=int,
                        help="compile one TileM while retaining the source denominator")
    parser.add_argument("--plant-drop-last", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        manifest = generate(args.qtype, args.artifact_tk, args.bchunk,
                            args.out_dir, args.per_unit,
                            args.plant_drop_last, args.tile_m_filter)
        den = manifest["denominator"]
        print("[fq-tc-generate] " + " ".join(
            f"{k}={v}" for k, v in manifest["identity"].items()) +
            f" raw={den['raw_topology_rows']} "
            f"provider_expanded={den['provider_expanded_rows']} "
            f"source_typed={den['source_typed_rows']} "
            f"typed={den['typed_rows']} "
            f"selection_reject={den['selection_reject_rows']} "
            f"static_reject={den['static_reject_rows']} "
            f"units={len(manifest['units'])}")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[fq-tc-generate] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
