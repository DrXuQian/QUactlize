#!/usr/bin/env python3
"""Select the exact AP0/AP1 rows for the Q4_K Split-K timing closure."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys


EXPECTED = {
    "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap0",
    "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap1",
}


def select(manifest: dict) -> list[tuple[int, dict]]:
    identity = manifest.get("identity", {})
    if (identity.get("qtype"), identity.get("artifact_tile_k"),
            identity.get("bchunk")) != (12, 64, 0):
        raise ValueError("source must be q12/A64/bchunk0")
    rows = manifest.get("typed_rows", [])
    units = manifest.get("units", [])
    denominator = manifest.get("denominator", {})
    if len(rows) != denominator.get("typed_rows") or len(units) != len(rows):
        raise ValueError("source must use one unit per typed row")
    selected = [(index, row) for index, row in enumerate(rows)
                if row.get("symbol") in EXPECTED]
    got = {row["symbol"] for _, row in selected}
    if got != EXPECTED or len(selected) != len(EXPECTED):
        raise ValueError(
            f"closure denominator changed missing={sorted(EXPECTED-got)} "
            f"extra={sorted(got-EXPECTED)}")
    for _, row in selected:
        if not (row["tile_m"] == 8 and row["tile_n"] == 64 and
                row["tactic_tile_k"] == 256 and row["warp_m"] == 8 and
                row["warp_n"] == 16 and row["stages"] == 2 and
                row["a_provider"] in ("standard-aiu", "packed-row")):
            raise ValueError(f"symbol/axis contradiction: {row}")
    return selected


def registry_macro(rows: list[dict]) -> str:
    lines = ["#define FQ_TC_REGISTRY_ROWS(X) \\"]
    for index, row in enumerate(rows):
        provider = 1 if row["a_provider"] == "packed-row" else 0
        continuation = " \\" if index + 1 < len(rows) else ""
        lines.append(
            f"  X({row['symbol']},{row['qtype']},{row['artifact_tile_k']},"
            f"{row['tile_m']},{row['tile_n']},{row['tactic_tile_k']},"
            f"{row['warp_m']},{row['warp_n']},{row['stages']},"
            f"{row['bchunk']},{provider}){continuation}")
    return "\n".join(lines) + "\n"


def materialize(source: pathlib.Path, output: pathlib.Path) -> None:
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selected = select(manifest)
    output.mkdir(parents=True, exist_ok=True)
    unit_dir = output / "units"
    unit_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    rows: list[dict] = []
    for ordinal, (source_index, row) in enumerate(selected):
        source_unit = pathlib.Path(manifest["units"][source_index]).resolve()
        if not source_unit.is_file() or source_unit.is_symlink():
            raise ValueError(f"source unit missing/symlinked: {source_unit}")
        destination = unit_dir / f"fq_tc_split_timing_{ordinal:02d}.cu"
        shutil.copy2(source_unit, destination)
        copied.append(str(destination.resolve()))
        rows.append(row)
    registry = (
        "// GENERATED -- exact Q4_K Split-K timing closure.\n"
        "#define FQ_TC_GENERATED_QTYPE 12\n"
        "#define FQ_TC_GENERATED_ARTIFACT_TK 64\n"
        "#define FQ_TC_GENERATED_BCHUNK 0\n"
        "#define FQ_TC_GENERATED_RAW_ROWS 2\n"
        "#define FQ_TC_GENERATED_TYPED_ROWS 2\n" + registry_macro(rows))
    (output / "fq_tc_registry.inc").write_text(registry)
    (output / "units.cmake").write_text(
        "# GENERATED -- exact Q4_K Split-K timing closure.\n"
        "set(FQ_TC_GENERATED_UNIT_SOURCES\n" +
        "".join(f'  "{path}"\n' for path in copied) +
        ")\n"
        f'set(FQ_TC_GENERATED_REGISTRY "{(output / "fq_tc_registry.inc").resolve()}")\n'
        f'set(FQ_TC_GENERATED_MANIFEST "{(output / "manifest.json").resolve()}")\n')
    closure = {
        "schema": "quactlize.fq-split-timing-closure.v1",
        "source_manifest": str(manifest_path.resolve()),
        "source_typed_denominator": len(manifest["typed_rows"]),
        "selection_denominator": len(rows),
        "identity": manifest["identity"],
        "typed_rows": rows,
        "units": copied,
    }
    (output / "manifest.json").write_text(
        json.dumps(closure, indent=2, sort_keys=True) + "\n")
    print("[fq-split-timing-select] PASS "
          f"source_typed={len(manifest['typed_rows'])} selected={len(rows)} "
          f"output={output}")


def self_test() -> None:
    rows, units = [], []
    for symbol in sorted(EXPECTED):
        rows.append({
            "symbol": symbol, "qtype": 12, "artifact_tile_k": 64,
            "tile_m": 8, "tile_n": 64, "tactic_tile_k": 256,
            "warp_m": 8, "warp_n": 16, "stages": 2, "bchunk": 0,
            "a_provider": "packed-row" if symbol.endswith("_ap1")
            else "standard-aiu",
        })
        units.append(f"unit-{len(units)}")
    fixture = {
        "identity": {"qtype": 12, "artifact_tile_k": 64, "bchunk": 0},
        "typed_rows": rows, "units": units,
        "denominator": {"typed_rows": 2},
    }
    assert len(select(fixture)) == 2
    missing = json.loads(json.dumps(fixture))
    missing["typed_rows"].pop()
    for broken in (missing, dict(fixture, identity={"qtype": 12,
                                                    "artifact_tile_k": 32,
                                                    "bchunk": 0})):
        try:
            select(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("selector negative stayed green")
    print("[fq-split-timing-select:self-test] PASS exact AP0/AP1; "
          "missing-row and identity negatives RED")


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
    except (AssertionError, OSError, ValueError) as error:
        print(f"[fq-split-timing-select] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
