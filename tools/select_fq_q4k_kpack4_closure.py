#!/usr/bin/env python3
"""Materialize one canonical Q4_K K-pack4 production tactic.

The ordinary A64/TM8 generator remains the tactic-space authority.  This
selector proves the named geometry exists in that complete typed denominator,
then changes only the physical weight-layout identity to ArtifactTileK=0 and
emits a one-unit graph for the K-pack4 production type.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


SOURCE_SYMBOL = "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap0"
SYMBOL = "fq_tc_q12_kp4_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap0"
MAPPING_ID = "0x51344b5034540001"


def select(manifest: dict) -> dict:
    identity = manifest.get("identity", {})
    if (identity.get("qtype"), identity.get("artifact_tile_k"),
            identity.get("bchunk"), identity.get("tile_m_filter")) != \
            (12, 64, 0, 8):
        raise ValueError("source must be the complete q12/A64/bchunk0/TM8 authority")
    rows = manifest.get("typed_rows", [])
    units = manifest.get("units", [])
    denominator = manifest.get("denominator", {})
    selection = manifest.get("selection", {})
    if (len(rows) != 144 or denominator.get("typed_rows") != 144 or
            selection.get("source_typed_rows") != 918 or len(units) != len(rows)):
        raise ValueError("source typed denominator must be exact 144/918 with one unit per row")
    selected = [row for row in rows if row.get("symbol") == SOURCE_SYMBOL]
    if len(selected) != 1:
        raise ValueError(f"exact source row denominator changed: got {len(selected)}")
    row = selected[0]
    expected = {
        "qtype": 12, "artifact_tile_k": 64,
        "tile_m": 8, "tile_n": 64, "tactic_tile_k": 256,
        "warp_m": 8, "warp_n": 16, "stages": 2, "bchunk": 0,
        "a_provider": "standard-aiu",
    }
    contradictions = {key: (row.get(key), value) for key, value in expected.items()
                      if row.get(key) != value}
    if contradictions:
        raise ValueError(f"symbol/axis contradiction: {contradictions}")
    return row


def materialize(source: pathlib.Path, output: pathlib.Path) -> None:
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_row = select(manifest)
    output.mkdir(parents=True, exist_ok=True)
    units = output / "units"
    units.mkdir(parents=True, exist_ok=True)
    unit = units / "fq_tc_kpack4_closure_00.cu"
    unit.write_text(
        "// GENERATED -- one canonical Q4_K K-pack4 production row.\n"
        "#ifdef PPU_PACKED_SCALE\n#undef PPU_PACKED_SCALE\n#endif\n"
        "#define PPU_PACKED_SCALE 1\n"
        "#ifdef PPU_PACKED_FORMAT\n#undef PPU_PACKED_FORMAT\n#endif\n"
        "#define PPU_PACKED_FORMAT 0\n"
        "#ifdef PPU_B_CHUNK\n#undef PPU_B_CHUNK\n#endif\n"
        "#define PPU_B_CHUNK 0\n"
        "#define FQ_TC_UNIT_ROWS(X) " + "\\" + "\n"
        f"  X({SYMBOL},12,0,8,64,256,8,16,2,0,0)\n"
        '#include "fully_quantized_splitk_producer_unit.inc"\n')
    registry = output / "fq_tc_registry.inc"
    registry.write_text(
        "// GENERATED -- one canonical Q4_K K-pack4 production row.\n"
        "#define FQ_TC_GENERATED_QTYPE 12\n"
        "#define FQ_TC_GENERATED_ARTIFACT_TK 0\n"
        "#define FQ_TC_GENERATED_BCHUNK 0\n"
        "#define FQ_TC_GENERATED_RAW_ROWS 1\n"
        "#define FQ_TC_GENERATED_TYPED_ROWS 1\n"
        "#define FQ_TC_REGISTRY_ROWS(X) " + "\\" + "\n"
        f"  X({SYMBOL},12,0,8,64,256,8,16,2,0,0)\n")
    closure = {
        "schema": "quactlize.fq-q4k-kpack4-closure.v1",
        "identity": {
            "qtype": 12,
            "artifact_tile_k": 0,
            "bchunk": 0,
            "weight_layout": 1,
            "layout": "q4-kpack4-transpose-v1",
            "mapping_id": MAPPING_ID,
            "transport_tile_k": 64,
            "group_size": 32,
        },
        "source_manifest": str(manifest_path.resolve()),
        "source_typed_denominator": len(manifest["typed_rows"]),
        "source_global_typed_denominator":
            manifest["selection"]["source_typed_rows"],
        "selection_denominator": 1,
        "source_row": source_row,
        "typed_rows": [{
            **source_row,
            "symbol": SYMBOL,
            "artifact_tile_k": 0,
            "weight_layout": "q4-kpack4-transpose-v1",
        }],
        "units": [str(unit.resolve())],
    }
    (output / "manifest.json").write_text(
        json.dumps(closure, indent=2, sort_keys=True) + "\n")
    (output / "units.cmake").write_text(
        "# GENERATED -- one canonical Q4_K K-pack4 production row.\n"
        "set(FQ_TC_GENERATED_UNIT_SOURCES\n"
        f'  "{unit.resolve()}"\n'
        ")\n"
        f'set(FQ_TC_GENERATED_REGISTRY "{registry.resolve()}")\n'
        f'set(FQ_TC_GENERATED_MANIFEST "{(output / "manifest.json").resolve()}")\n')
    print("[fq-q4k-kpack4-select] PASS "
          f"source_typed={len(manifest['typed_rows'])}/"
          f"{manifest['selection']['source_typed_rows']} selected=1 "
          f"layout=1 mapping={MAPPING_ID} output={output}")


def self_test() -> None:
    row = {
        "symbol": SOURCE_SYMBOL, "qtype": 12, "artifact_tile_k": 64,
        "tile_m": 8, "tile_n": 64, "tactic_tile_k": 256,
        "warp_m": 8, "warp_n": 16, "stages": 2, "bchunk": 0,
        "a_provider": "standard-aiu",
    }
    filler = [{**row, "symbol": f"filler_{i}"} for i in range(143)]
    fixture = {
        "identity": {"qtype": 12, "artifact_tile_k": 64,
                     "bchunk": 0, "tile_m_filter": 8},
        "typed_rows": [row, *filler],
        "units": [f"unit-{i}" for i in range(144)],
        "selection": {"source_typed_rows": 918},
        "denominator": {"typed_rows": 144},
    }
    assert select(fixture)["symbol"] == SOURCE_SYMBOL
    negatives = []
    broken = json.loads(json.dumps(fixture))
    broken["typed_rows"][0]["symbol"] = "missing"
    negatives.append(broken)
    broken = json.loads(json.dumps(fixture))
    broken["typed_rows"][0]["warp_n"] = 32
    negatives.append(broken)
    broken = json.loads(json.dumps(fixture))
    broken["selection"]["source_typed_rows"] = 917
    negatives.append(broken)
    for broken in negatives:
        try:
            select(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("selector RED control stayed green")
    print("[fq-q4k-kpack4-select:self-test] PASS exact source row; "
          "missing-row, axis-contradiction and denominator plants RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=pathlib.Path)
    parser.add_argument("--out-dir", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            if args.source_dir is None or args.out_dir is None:
                parser.error("--source-dir and --out-dir are required")
            materialize(args.source_dir.resolve(), args.out_dir.resolve())
        return 0
    except (OSError, ValueError, AssertionError, KeyError) as exc:
        print(f"[fq-q4k-kpack4-select] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
