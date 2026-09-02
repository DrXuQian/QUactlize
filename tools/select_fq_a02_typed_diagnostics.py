#!/usr/bin/env python3
"""Select the exact A02 Q4 provider pair and Q3 effective-BChunk pair."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys


Q4 = (
    "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap0",
    "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap1",
)
Q3 = (
    "fq_tc_q11_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap0",
    "fq_tc_q11_a64_tm8_tn64_tk256_wm8_wn16_s2_bc1_ap0",
)


def load_one(root: Path, qtype: int, bchunk: int, symbol: str) -> tuple[dict, Path]:
    value = json.loads((root / "manifest.json").read_text())
    identity = value.get("identity", {})
    if (identity.get("qtype"), identity.get("artifact_tile_k"), identity.get("bchunk")) != \
       (qtype, 64, bchunk):
        raise ValueError("source shard identity differs")
    rows, units = value.get("typed_rows", []), value.get("units", [])
    if len(rows) != value.get("denominator", {}).get("typed_rows") or len(units) != len(rows):
        raise ValueError("source requires per-unit=1 and exact typed denominator")
    matches = [(row, Path(units[index])) for index, row in enumerate(rows)
               if row.get("symbol") == symbol]
    if len(matches) != 1:
        raise ValueError(f"exact row missing/duplicate: {symbol}")
    row, unit = matches[0]
    expected = {"qtype": qtype, "artifact_tile_k": 64, "tile_m": 8,
                "tile_n": 64, "tactic_tile_k": 256, "warp_m": 8,
                "warp_n": 16, "stages": 2, "bchunk": bchunk,
                "a_provider": "standard-aiu" if symbol.endswith("ap0") else "packed-row"}
    if any(row.get(key) != item for key, item in expected.items()):
        raise ValueError(f"row axes contradict symbol: {symbol}")
    unit = unit.resolve()
    if not unit.is_file() or unit.is_symlink():
        raise ValueError("unit is missing/symlinked")
    return row, unit


def registry(rows: list[dict], qtype: int, bchunk: int) -> str:
    lines = ["#define FQ_TC_REGISTRY_ROWS(X) \\"]
    for index, row in enumerate(rows):
        ap = 1 if row["a_provider"] == "packed-row" else 0
        tail = " \\" if index + 1 < len(rows) else ""
        lines.append(
            f"  X({row['symbol']},{qtype},64,8,64,256,8,16,2,{row['bchunk']},{ap}){tail}")
    return (f"#define FQ_TC_GENERATED_QTYPE {qtype}\n"
            "#define FQ_TC_GENERATED_ARTIFACT_TK 64\n"
            f"#define FQ_TC_GENERATED_BCHUNK {bchunk}\n" + "\n".join(lines) + "\n")


def emit(output: Path, rows_units: list[tuple[dict, Path]], qtype: int,
         bchunk: int, schema: str) -> None:
    output.mkdir(parents=True, exist_ok=False)
    units_dir = output / "units"
    units_dir.mkdir()
    rows, copied = [], []
    for index, (row, source) in enumerate(rows_units):
        target = units_dir / f"fq_a02_unit_{index}.cu"
        text = source.read_text()
        if qtype == 11:
            namespace = f"fq_a02_q3_bc{row['bchunk']}_generated"
            needle = '#include "fully_quantized_splitk_producer_unit.inc"'
            if text.count(needle) != 1:
                raise ValueError("generated unit include seam differs")
            text = text.replace(needle, f"#define FQ_TC_GENERATED_NAMESPACE {namespace}\n{needle}")
        target.write_text(text)
        rows.append(row)
        copied.append(str(target.resolve()))
    (output / "fq_tc_registry.inc").write_text(registry(rows, qtype, bchunk))
    (output / "units.cmake").write_text(
        "set(FQ_TC_GENERATED_UNIT_SOURCES\n" +
        "".join(f'  "{path}"\n' for path in copied) + ")\n")
    (output / "manifest.json").write_text(json.dumps({
        "schema": schema, "identity": {"qtype": qtype, "artifact_tile_k": 64,
                                         "bchunk": bchunk},
        "classification": ["PRODUCT_SHIPPING" if row["symbol"] == Q4[0]
                           else "TYPED_DIAGNOSTIC_NONPRODUCT" for row in rows],
        "typed_rows": rows, "units": copied, "denominator": {"typed_rows": 2},
    }, indent=2, sort_keys=True) + "\n")


def materialize(q4_source: Path, q3_bc0: Path, q3_bc1: Path, output: Path) -> None:
    q4 = [load_one(q4_source, 12, 0, symbol) for symbol in Q4]
    q3 = [load_one(q3_bc0 if bc == 0 else q3_bc1, 11, bc, symbol)
          for bc, symbol in zip((0, 1), Q3)]
    output.mkdir(parents=True, exist_ok=False)
    emit(output / "q4", q4, 12, 0, "quactlize.fq-a02-q4-provider-pair.v1")
    emit(output / "q3", q3, 11, -1, "quactlize.fq-a02-q3-bchunk-pair.v1")


def self_test() -> None:
    rows = []
    for symbol in (*Q4, *Q3):
        rows.append({"symbol": symbol, "classification":
                     "PRODUCT_SHIPPING" if symbol == Q4[0]
                     else "TYPED_DIAGNOSTIC_NONPRODUCT"})
    assert sum(row["classification"] == "PRODUCT_SHIPPING" for row in rows) == 1
    broken = copy.deepcopy(rows)
    broken[1]["classification"] = "PRODUCT_SHIPPING"
    assert sum(row["classification"] == "PRODUCT_SHIPPING" for row in broken) != 1
    print("[fq-a02-select:self-test] PASS exact pairs; AP1/Q3 shipping-label plant RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--q4-source", type=Path)
    parser.add_argument("--q3-bc0", type=Path)
    parser.add_argument("--q3-bc1", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            if None in (args.q4_source, args.q3_bc0, args.q3_bc1, args.output):
                raise ValueError("all source/output paths are required")
            materialize(args.q4_source, args.q3_bc0, args.q3_bc1, args.output)
        return 0
    except (OSError, ValueError, AssertionError) as error:
        print(f"[fq-a02-select] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
