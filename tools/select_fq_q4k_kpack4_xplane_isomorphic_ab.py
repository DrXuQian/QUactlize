#!/usr/bin/env python3
"""Materialize four exact Q4_K decode A/B arms.

The source manifests remain the complete TM8 authorities.  The selected row is
identical in TM/TN/TK/WM/WN/stages/bchunk and differs only in the two explicit
axes under test:

  weight layout: xplane A64 vs canonical K-pack4
  A provider:    standard-aiu (AP0) vs packed-row (AP1)

Each output directory contains one generated shipping row so hgobjdump can
select one Split-K producer symbol without guessing among a sweep binary.
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

import gen_fully_quantized_splitk_producer_units as generator  # noqa: E402


AXES = {
    "qtype": 12,
    "tile_m": 8,
    "tile_n": 64,
    "tactic_tile_k": 256,
    "warp_m": 8,
    "warp_n": 16,
    "stages": 2,
    "bchunk": 0,
}
MAPPING_ID = "0x51344b5034540001"
ARMS = (
    ("xplane-ap0", "xplane", 64, 0),
    ("kpack4-ap0", "q4-kpack4", 0, 0),
    ("xplane-ap1", "xplane", 64, 1),
    ("kpack4-ap1", "q4-kpack4", 0, 1),
)


class SelectError(ValueError):
    pass


def provider_name(ap: int) -> str:
    return "packed-row" if ap else "standard-aiu"


def validate_source(value: dict[str, Any], *, layout: str) -> None:
    identity = value.get("identity", {})
    expected_identity: dict[str, Any] = {
        "qtype": 12, "format": "Q4_K", "artifact_tile_k": 64 if layout == "xplane" else 0,
        "bchunk": 0, "tile_m_filter": 8,
    }
    if layout == "q4-kpack4":
        expected_identity["weight_layout"] = "q4-kpack4"
    if identity != expected_identity:
        raise SelectError(f"{layout} source identity differs: {identity}")
    denominator = value.get("denominator", {})
    expected = {
        "raw_topology_rows": 11520,
        "provider_expanded_rows": 12000,
        "source_typed_rows": 918,
        "typed_rows": 144,
        "selection_reject_rows": 774,
        "static_reject_rows": 11082,
        "runtime_tc_cells": 48000,
        "typed_runtime_tc_cells": 576,
    }
    if denominator != expected:
        raise SelectError(f"{layout} source denominator differs: {denominator}")
    rows = value.get("typed_rows", [])
    if len(rows) != 144 or len({row.get("symbol") for row in rows}) != 144:
        raise SelectError(f"{layout} typed symbol denominator is not 144 unique rows")
    if layout == "q4-kpack4":
        mapping = value.get("weight_mapping", {})
        if mapping.get("mapping_id") != MAPPING_ID or \
                mapping.get("layout") != "q4-kpack4-transpose-v1" or \
                mapping.get("artifact_tile_k_is_not_an_axis") is not True:
            raise SelectError(f"K-pack4 mapping identity differs: {mapping}")
    elif "weight_mapping" in value:
        raise SelectError("xplane source unexpectedly carries K-pack4 mapping")


def select_row(value: dict[str, Any], *, layout: str, ap: int) -> dict[str, Any]:
    validate_source(value, layout=layout)
    expected = {
        **AXES,
        "artifact_tile_k": 64 if layout == "xplane" else 0,
        "a_provider": provider_name(ap),
    }
    selected = [
        row for row in value["typed_rows"]
        if all(row.get(key) == want for key, want in expected.items())
    ]
    if len(selected) != 1:
        raise SelectError(
            f"{layout}/AP{ap} exact row denominator is {len(selected)}, expected 1")
    return selected[0]


def row_macro(symbol: str, artifact: int, ap: int) -> str:
    return (
        f"  X({symbol},12,{artifact},8,64,256,8,16,2,0,{ap})\n"
    )


def materialize_arm(output: pathlib.Path, *, name: str, layout: str,
                    artifact: int, ap: int, source_path: pathlib.Path,
                    source: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    units = output / "units"
    units.mkdir(parents=True, exist_ok=True)
    unit = units / f"fq_tc_{name.replace('-', '_')}_00.cu"
    symbol = str(row["symbol"])
    macro = row_macro(symbol, artifact, ap)
    unit.write_text(
        "// GENERATED -- exact Q4_K K-pack4/xplane isomorphic A/B row.\n"
        "#ifdef PPU_PACKED_SCALE\n#undef PPU_PACKED_SCALE\n#endif\n"
        "#define PPU_PACKED_SCALE 1\n"
        "#ifdef PPU_PACKED_FORMAT\n#undef PPU_PACKED_FORMAT\n#endif\n"
        "#define PPU_PACKED_FORMAT 0\n"
        "#define FQ_TC_UNIT_ROWS(X) \\\n" + macro +
        '#include "fully_quantized_splitk_producer_unit.inc"\n'
    )
    registry = output / "fq_tc_registry.inc"
    registry.write_text(
        "// GENERATED -- exact Q4_K K-pack4/xplane isomorphic A/B row.\n"
        "#define FQ_TC_GENERATED_QTYPE 12\n"
        f"#define FQ_TC_GENERATED_ARTIFACT_TK {artifact}\n"
        "#define FQ_TC_GENERATED_BCHUNK 0\n"
        "#define FQ_TC_GENERATED_RAW_ROWS 1\n"
        "#define FQ_TC_GENERATED_TYPED_ROWS 1\n"
        "#define FQ_TC_REGISTRY_ROWS(X) \\\n" + macro
    )
    manifest = {
        "schema": "quactlize.fq-q4k-kpack4-xplane-isomorphic-arm.v1",
        "name": name,
        "layout": layout,
        "weight_layout": 0 if layout == "xplane" else 1,
        "artifact_tile_k": artifact,
        "a_provider": provider_name(ap),
        "a_provider_id": ap,
        "selection_denominator": 1,
        "source_manifest": str(source_path.resolve()),
        "source_typed_denominator": source["denominator"]["typed_rows"],
        "source_global_typed_denominator":
            source["denominator"]["source_typed_rows"],
        "row": row,
        "units": [str(unit.resolve())],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "units.cmake").write_text(
        "# GENERATED -- exact Q4_K K-pack4/xplane isomorphic A/B row.\n"
        "set(FQ_TC_GENERATED_UNIT_SOURCES\n"
        f'  "{unit.resolve()}"\n'
        ")\n"
        f'set(FQ_TC_GENERATED_REGISTRY "{registry.resolve()}")\n'
        f'set(FQ_TC_GENERATED_MANIFEST "{manifest_path.resolve()}")\n'
    )
    return manifest


def materialize(xplane_dir: pathlib.Path, kpack4_dir: pathlib.Path,
                output: pathlib.Path) -> None:
    source_paths = {
        "xplane": xplane_dir / "manifest.json",
        "q4-kpack4": kpack4_dir / "manifest.json",
    }
    sources = {
        name: json.loads(path.read_text()) for name, path in source_paths.items()
    }
    output.mkdir(parents=True, exist_ok=True)
    arm_values = []
    for name, layout, artifact, ap in ARMS:
        row = select_row(sources[layout], layout=layout, ap=ap)
        arm_values.append(materialize_arm(
            output / name, name=name, layout=layout, artifact=artifact, ap=ap,
            source_path=source_paths[layout], source=sources[layout], row=row))
    manifest = {
        "schema": "quactlize.fq-q4k-kpack4-xplane-isomorphic-ab.v1",
        "axes": {**AXES, "split": 4},
        "arms": arm_values,
        "claims": {
            "same_tactic": True,
            "same_split": True,
            "ap0_isolates_weight_layout": True,
            "ap1_isolates_weight_layout": True,
            "ap0_vs_ap1_isolates_a_provider_within_each_layout": True,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("[fq-kpack4-xplane-ab-select] PASS source=xplane:144/918,"
          "kpack4:144/918 selected=4 config=8x64x256_w8x16_s2 "
          f"output={output}")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="qz-kpack4-xplane-ab-") as temp:
        root = pathlib.Path(temp)
        xdir, kdir = root / "x", root / "k"
        x = generator.generate(12, 64, 0, xdir, 144, False, 8, "xplane")
        k = generator.generate(12, 0, 0, kdir, 144, False, 8, "q4-kpack4")
        materialize(xdir, kdir, root / "out")
        for layout, value in (("xplane", x), ("q4-kpack4", k)):
            for ap in (0, 1):
                assert select_row(value, layout=layout, ap=ap)["a_provider"] == \
                    provider_name(ap)
        plants = []
        broken = copy.deepcopy(k)
        broken["typed_rows"] = [
            row for row in broken["typed_rows"]
            if not (row["symbol"].endswith("_ap1") and
                    all(row.get(key) == want for key, want in AXES.items()))
        ]
        plants.append(("q4-kpack4", 1, broken))
        broken = copy.deepcopy(x)
        broken["denominator"]["typed_rows"] = 143
        plants.append(("xplane", 0, broken))
        broken = copy.deepcopy(k)
        broken["weight_mapping"]["mapping_id"] = "0x0"
        plants.append(("q4-kpack4", 0, broken))
        for layout, ap, broken in plants:
            try:
                select_row(broken, layout=layout, ap=ap)
            except SelectError:
                pass
            else:
                raise AssertionError("selector negative stayed green")
    print("[fq-kpack4-xplane-ab-select:self-test] PASS four exact arms; "
          "missing AP1, denominator and mapping plants RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    run = sub.add_parser("materialize")
    run.add_argument("--xplane-dir", type=pathlib.Path, required=True)
    run.add_argument("--kpack4-dir", type=pathlib.Path, required=True)
    run.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        else:
            materialize(args.xplane_dir.resolve(), args.kpack4_dir.resolve(),
                        args.out_dir.resolve())
        return 0
    except (OSError, KeyError, SelectError, AssertionError, ValueError) as exc:
        print(f"[fq-kpack4-xplane-ab-select] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
