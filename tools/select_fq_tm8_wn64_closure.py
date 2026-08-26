#!/usr/bin/env python3
"""Select the six exact Q4_K packed-metadata closure rows.

The production generator remains the tactic authority.  This tool only makes
a small build graph from its complete A64 manifest so a correctness rerun does
not rebuild or execute the full 918-row shard.  The historical filename is
retained for callers; the denominator now includes the WN16 root-cause row.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys


EXPECTED = {
    "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn64_s3_bc0_ap0",
    "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn64_s3_bc0_ap1",
    "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn64_s4_bc0_ap0",
    "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn64_s4_bc0_ap1",
    "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap0",
    "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap1",
}


def select(manifest: dict) -> list[tuple[int, dict]]:
    identity = manifest.get("identity", {})
    if (identity.get("qtype"), identity.get("artifact_tile_k"),
            identity.get("bchunk")) != (12, 64, 0):
        raise ValueError("source must be the complete q12/A64/bchunk0 authority")
    rows = manifest.get("typed_rows", [])
    units = manifest.get("units", [])
    denominator = manifest.get("denominator", {})
    if len(rows) != denominator.get("typed_rows") or len(units) != len(rows):
        raise ValueError("source must use one unit per typed row with an exact denominator")
    selected = [(i, row) for i, row in enumerate(rows)
                if row.get("symbol") in EXPECTED]
    got = {row["symbol"] for _, row in selected}
    if got != EXPECTED or len(selected) != len(EXPECTED):
        raise ValueError(f"exact closure denominator changed: missing={sorted(EXPECTED-got)} "
                         f"extra={sorted(got-EXPECTED)}")
    for _, row in selected:
        topology = ((row["warp_n"] == 64 and row["stages"] in (3, 4)) or
                    (row["warp_n"] == 16 and row["stages"] == 2))
        if not (row["tile_m"] == 8 and row["tile_n"] == 64 and
                row["tactic_tile_k"] == 256 and row["warp_m"] == 8 and
                topology and
                row["a_provider"] in ("standard-aiu", "packed-row")):
            raise ValueError(f"symbol/axis contradiction: {row}")
    return selected


def macro(rows: list[dict]) -> str:
    lines = ["#define FQ_TC_REGISTRY_ROWS(X) \\"]
    for index, row in enumerate(rows):
        ap = 1 if row["a_provider"] == "packed-row" else 0
        continuation = " \\" if index + 1 < len(rows) else ""
        lines.append(
            f"  X({row['symbol']},{row['qtype']},{row['artifact_tile_k']},"
            f"{row['tile_m']},{row['tile_n']},{row['tactic_tile_k']},"
            f"{row['warp_m']},{row['warp_n']},{row['stages']},"
            f"{row['bchunk']},{ap}){continuation}")
    return "\n".join(lines) + "\n"


def materialize(source: pathlib.Path, output: pathlib.Path) -> None:
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selected = select(manifest)
    output.mkdir(parents=True, exist_ok=True)
    unit_dir = output / "units"
    unit_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    rows = []
    for ordinal, (source_index, row) in enumerate(selected):
        src = pathlib.Path(manifest["units"][source_index]).resolve()
        if not src.is_file() or src.is_symlink():
            raise ValueError(f"source unit missing or symlinked: {src}")
        dst = unit_dir / f"fq_tc_closure_{ordinal:02d}.cu"
        shutil.copy2(src, dst)
        copied.append(str(dst.resolve()))
        rows.append(row)

    registry = (
        "// GENERATED -- exact Q4_K packed-metadata correctness closure.\n"
        "#define FQ_TC_GENERATED_QTYPE 12\n"
        "#define FQ_TC_GENERATED_ARTIFACT_TK 64\n"
        "#define FQ_TC_GENERATED_BCHUNK 0\n"
        "#define FQ_TC_GENERATED_RAW_ROWS 4\n"
        "#define FQ_TC_GENERATED_TYPED_ROWS 4\n" + macro(rows))
    (output / "fq_tc_registry.inc").write_text(registry)
    cmake = (
        "# GENERATED -- exact Q4_K packed-metadata correctness closure.\n"
        "set(FQ_TC_GENERATED_UNIT_SOURCES\n" +
        "".join(f'  "{path}"\n' for path in copied) +
        ")\n"
        f'set(FQ_TC_GENERATED_REGISTRY "{(output / "fq_tc_registry.inc").resolve()}")\n'
        f'set(FQ_TC_GENERATED_MANIFEST "{(output / "manifest.json").resolve()}")\n')
    (output / "units.cmake").write_text(cmake)
    closure = {
        "schema": "quactlize.fq-packed-metadata-closure.v2",
        "source_manifest": str(manifest_path.resolve()),
        "source_typed_denominator": len(manifest["typed_rows"]),
        "selection_denominator": len(rows),
        "identity": manifest["identity"],
        "typed_rows": rows,
        "units": copied,
    }
    (output / "manifest.json").write_text(
        json.dumps(closure, indent=2, sort_keys=True) + "\n")
    print("[fq-tm8-wn64-select] PASS "
          f"source_typed={len(manifest['typed_rows'])} selected={len(rows)} "
          f"output={output}")


def self_test() -> None:
    rows = []
    units = []
    for symbol in sorted(EXPECTED):
        stage = int(symbol.split("_s", 1)[1].split("_", 1)[0])
        warp_n = int(symbol.split("_wn", 1)[1].split("_", 1)[0])
        provider = "packed-row" if symbol.endswith("_ap1") else "standard-aiu"
        rows.append({
            "symbol": symbol, "qtype": 12, "artifact_tile_k": 64,
            "tile_m": 8, "tile_n": 64, "tactic_tile_k": 256,
            "warp_m": 8, "warp_n": warp_n, "stages": stage, "bchunk": 0,
            "a_provider": provider,
        })
        units.append(f"unit-{len(units)}")
    fixture = {"identity": {"qtype": 12, "artifact_tile_k": 64, "bchunk": 0},
               "typed_rows": rows, "units": units,
               "denominator": {"typed_rows": len(rows)}}
    assert len(select(fixture)) == 6
    broken = json.loads(json.dumps(fixture))
    broken["typed_rows"].pop()
    try:
        select(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("missing exact row stayed green")
    broken = json.loads(json.dumps(fixture))
    broken["typed_rows"][0]["warp_n"] = 32
    try:
        select(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("symbol/axis contradiction stayed green")
    print("[fq-tm8-wn64-select:self-test] PASS exact-6; missing-row and axis-contradiction RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=pathlib.Path)
    parser.add_argument("--out-dir", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if args.source_dir is None or args.out_dir is None:
            parser.error("--source-dir and --out-dir are required")
        materialize(args.source_dir.resolve(), args.out_dir.resolve())
        return 0
    except (OSError, ValueError, AssertionError) as exc:
        print(f"[fq-tm8-wn64-select] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
