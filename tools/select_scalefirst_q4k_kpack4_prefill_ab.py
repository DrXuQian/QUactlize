#!/usr/bin/env python3
"""Materialize an exact persistent ScaleFirst Xplane/K-pack4 prefill A/B.

The two arms share the same three logical tactics and FP16 scale/zero planes.
Only the resident Q4 weight mapping differs.  This is intentionally separate
from the FullyQuantized packed-metadata sweep: it answers whether K-pack4's
transpose costs anything under the historical ScaleFirst prefill contract.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import fully_quantized_internal_matrix as fq_matrix  # noqa: E402
import scalefirst_internal_matrix as sf_matrix  # noqa: E402


MAPPING_ID = "0x51344b5034540001"
CANDIDATES = (
    # Historical ScaleFirst M=2048 winner family.
    (64, 64, 64, 64, 32, 3),
    # Historical large-M A64 family.
    (64, 128, 64, 64, 16, 6),
    # Current K-pack4 ordinary-S1 winner/control.
    (64, 128, 256, 64, 16, 2),
)
SHAPES = ((2048, 1024, 5120), (4096, 1024, 5120))


class SelectError(ValueError):
    pass


def coordinates(row: Any) -> tuple[int, ...]:
    return (row.tile_m, row.tile_n, row.tactic_tile_k,
            row.warp_m, row.warp_n, row.stages)


def source_rows(layout: str) -> tuple[list[Any], int]:
    if layout == "xplane":
        fmt = sf_matrix.format_for(12)
        raw = [row for row in sf_matrix.emitted_tactics(12, 64)
               if row.bchunk == 0]
        eligible = [row for row in raw
                    if sf_matrix.classify(fmt, 64, row)[0] ==
                    "TYPE_ADMISSION_REQUIRED"]
        expected = 1824
    elif layout == "q4-kpack4":
        raw = [row for row in fq_matrix.emitted_tactics(12, 64)
               if row.bchunk == 0]
        eligible = [
            row for row in raw
            if row.source_status == "TYPE_ADMISSION_REQUIRED" and
            row.tactic_tile_k >= 64 and row.tactic_tile_k % 64 == 0 and
            row.tile_n % 16 == 0
        ]
        expected = 2538
    else:
        raise SelectError(f"unknown layout {layout}")
    if len(raw) != 11520 or len(eligible) != expected:
        raise SelectError(
            f"{layout} source denominator differs: raw={len(raw)} "
            f"eligible={len(eligible)}")
    return eligible, len(raw)


def select(layout: str) -> list[Any]:
    eligible, _ = source_rows(layout)
    selected = []
    for wanted in CANDIDATES:
        matches = [row for row in eligible if coordinates(row) == wanted]
        if len(matches) != 1:
            raise SelectError(
                f"{layout} exact tactic {wanted} denominator={len(matches)}")
        selected.append(matches[0])
    return selected


def symbol(layout: str, row: Any) -> str:
    artifact = 64 if layout == "xplane" else 0
    return (f"sf_q12_a{artifact}_tm{row.tile_m}_tn{row.tile_n}_"
            f"tk{row.tactic_tile_k}_wm{row.warp_m}_wn{row.warp_n}_"
            f"s{row.stages}_bc0")


def macro(name: str, layout: str, rows: list[Any]) -> str:
    artifact = 64 if layout == "xplane" else 0
    lines = [f"#define {name}(X) \\"]
    for index, row in enumerate(rows):
        tail = " \\" if index + 1 < len(rows) else ""
        lines.append(
            f"  X({symbol(layout, row)},12,{artifact},{row.tile_m},"
            f"{row.tile_n},{row.tactic_tile_k},{row.warp_m},{row.warp_n},"
            f"{row.stages},0){tail}")
    return "\n".join(lines) + "\n"


def row_json(layout: str, row: Any) -> dict[str, Any]:
    artifact = 64 if layout == "xplane" else 0
    return {
        "symbol": symbol(layout, row),
        "qtype": 12,
        "artifact_tile_k": artifact,
        "tile_m": row.tile_m,
        "tile_n": row.tile_n,
        "tactic_tile_k": row.tactic_tile_k,
        "warp_m": row.warp_m,
        "warp_n": row.warp_n,
        "stages": row.stages,
        "bchunk": 0,
        "config": (f"{row.tile_m}x{row.tile_n}x{row.tactic_tile_k}_"
                   f"w{row.warp_m}x{row.warp_n}_s{row.stages}_bc0"),
    }


def materialize_arm(output: pathlib.Path, layout: str) -> dict[str, Any]:
    rows = select(layout)
    eligible, raw_count = source_rows(layout)
    artifact = 64 if layout == "xplane" else 0
    weight_layout = 0 if layout == "xplane" else 1
    output.mkdir(parents=True, exist_ok=True)
    unit_dir = output / "units"
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit = unit_dir / f"scalefirst_q4k_{layout.replace('-', '_')}.cu"
    unit.write_text(
        "// GENERATED -- exact Q4 ScaleFirst persistent layout A/B rows.\n"
        "#ifdef PPU_PACKED_SCALE\n#undef PPU_PACKED_SCALE\n#endif\n"
        "#define PPU_PACKED_SCALE 0\n"
        "#ifdef PPU_B_CHUNK\n#undef PPU_B_CHUNK\n#endif\n"
        "#define PPU_B_CHUNK 0\n" +
        macro("SCALEFIRST_UNIT_ROWS", layout, rows) +
        '#include "scalefirst_internal_sweep_unit.inc"\n')
    registry = output / "scalefirst_registry.inc"
    registry.write_text(
        "// GENERATED -- exact Q4 ScaleFirst persistent layout A/B registry.\n"
        "#define SCALEFIRST_GENERATED_QTYPE 12\n"
        f"#define SCALEFIRST_GENERATED_ARTIFACT_TK {artifact}\n"
        "#define SCALEFIRST_GENERATED_BCHUNK 0\n"
        f"#define SCALEFIRST_GENERATED_RAW_ROWS {raw_count}\n"
        f"#define SCALEFIRST_GENERATED_TYPED_ROWS {len(rows)}\n" +
        macro("SCALEFIRST_REGISTRY_ROWS", layout, rows))
    manifest = {
        "schema": "quactlize.scalefirst-q4k-kpack4-prefill-arm.v1",
        "layout": layout,
        "weight_layout": weight_layout,
        "weight_mapping_id": "0x0" if weight_layout == 0 else MAPPING_ID,
        "artifact_tile_k": artifact,
        "metadata": "scalefirst-fp16-scale-zero",
        "algorithms": ["PERSISTENT"],
        "source_denominator": {
            "raw_rows": raw_count,
            "eligible_rows": len(eligible),
            "selected_rows": len(rows),
        },
        "typed_rows": [row_json(layout, row) for row in rows],
        "units": [str(unit.resolve())],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2,
                                        sort_keys=True) + "\n")
    (output / "units.cmake").write_text(
        "# GENERATED -- exact Q4 ScaleFirst persistent layout A/B rows.\n"
        "set(SCALEFIRST_GENERATED_UNIT_SOURCES\n"
        f'  "{unit.resolve()}"\n'
        ")\n"
        f'set(SCALEFIRST_GENERATED_REGISTRY "{registry.resolve()}")\n'
        f'set(SCALEFIRST_GENERATED_MANIFEST "{manifest_path.resolve()}")\n')
    return manifest


def validate_bundle(value: dict[str, Any]) -> None:
    if value.get("schema") != \
            "quactlize.scalefirst-q4k-kpack4-prefill-ab.v1":
        raise SelectError("bundle schema differs")
    if value.get("shapes") != [list(shape) for shape in SHAPES]:
        raise SelectError("large-prefill shape denominator differs")
    arms = value.get("arms", [])
    if len(arms) != 2 or {arm.get("layout") for arm in arms} != \
            {"xplane", "q4-kpack4"}:
        raise SelectError("layout arm denominator differs")
    for arm in arms:
        expected_weight = int(arm["layout"] == "q4-kpack4")
        expected_artifact = 0 if expected_weight else 64
        expected_mapping = MAPPING_ID if expected_weight else "0x0"
        if arm.get("weight_layout") != expected_weight or \
                arm.get("artifact_tile_k") != expected_artifact or \
                arm.get("weight_mapping_id") != expected_mapping or \
                arm.get("metadata") != "scalefirst-fp16-scale-zero" or \
                arm.get("algorithms") != ["PERSISTENT"]:
            raise SelectError(f"{arm.get('layout')} identity differs")
        rows = arm.get("typed_rows", [])
        if len(rows) != len(CANDIDATES) or \
                {tuple(row[key] for key in (
                    "tile_m", "tile_n", "tactic_tile_k", "warp_m",
                    "warp_n", "stages")) for row in rows} != set(CANDIDATES):
            raise SelectError(f"{arm.get('layout')} tactic denominator differs")


def materialize(output: pathlib.Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    arms = [materialize_arm(output / layout, layout)
            for layout in ("xplane", "q4-kpack4")]
    value = {
        "schema": "quactlize.scalefirst-q4k-kpack4-prefill-ab.v1",
        "purpose": "isolate weight layout under identical ScaleFirst metadata and persistent driver",
        "shapes": [list(shape) for shape in SHAPES],
        "candidates": [list(row) for row in CANDIDATES],
        "arms": arms,
    }
    validate_bundle(value)
    (output / "manifest.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n")
    print("[sf-kpack4-prefill-select] PASS layouts=2 rows=3 shapes=2 "
          "metadata=ScaleFirst-FP16 algorithm=PERSISTENT "
          f"output={output}")
    return value


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="qz-sf-kpack4-prefill-") as temp:
        value = materialize(pathlib.Path(temp) / "out")
        plants = []
        broken = copy.deepcopy(value)
        broken["shapes"].append([64, 1024, 5120])
        plants.append(broken)
        broken = copy.deepcopy(value)
        broken["arms"][1]["metadata"] = "packed-gguf"
        plants.append(broken)
        broken = copy.deepcopy(value)
        broken["arms"][1]["weight_mapping_id"] = "0x0"
        plants.append(broken)
        broken = copy.deepcopy(value)
        broken["arms"][0]["typed_rows"].pop()
        plants.append(broken)
        for broken in plants:
            try:
                validate_bundle(broken)
            except SelectError:
                pass
            else:
                raise AssertionError("selector negative stayed green")
    print("[sf-kpack4-prefill-select:self-test] PASS exact M2048/M4096, "
          "persistent FP16 metadata and three tactics; four plants RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    run = sub.add_parser("materialize")
    run.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        else:
            materialize(args.out_dir)
        return 0
    except (OSError, SelectError, AssertionError) as error:
        print(f"[sf-kpack4-prefill-select] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
