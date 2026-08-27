#!/usr/bin/env python3
"""Build-graph selector and fail-closed checker for K-pack4 fragment mapping.

The closure contains the four production geometries that L231 and the first
72-row device screen classify as non-identity, plus two identity controls.  A
candidate and the exact legacy loader-stride arm are built from the same Git
tree.  The candidate must make all six rows raw-bit exact; the legacy arm must
leave only the four predicted rows red with their exact output stripe sizes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import sys
from typing import Any


MAPPING_ID = "0x51344b5034540001"
SHAPE = "1x1024x5120"


def symbol(tn: int, wn: int) -> str:
    return f"fq_tc_q12_a0_tm8_tn{tn}_tk256_wm8_wn{wn}_s2_bc0_ap0"


CASES = (
    # Two controls whose loader N stride already equals the compute stride.
    (32, 16, True, 0),
    (128, 32, True, 0),
    # The four geometry groups made red by the legacy destination view.
    (32, 32, False, 512),
    (64, 32, False, 512),
    (64, 64, False, 1024),
    (128, 64, False, 512),
)
EXPECTED = {
    symbol(tn, wn): {
        "tn": tn, "wn": wn, "legacy_clean": clean,
        "legacy_raw_bad": raw_bad,
    }
    for tn, wn, clean, raw_bad in CASES
}


class ClosureError(ValueError):
    pass


def validate_source(value: dict[str, Any]) -> list[dict[str, Any]]:
    if value.get("identity") != {
            "qtype": 12, "format": "Q4_K", "artifact_tile_k": 0,
            "bchunk": 0, "tile_m_filter": 8,
            "weight_layout": "q4-kpack4"}:
        raise ClosureError("source must be the complete native K-pack4 TM8 authority")
    denominator = value.get("denominator", {})
    if denominator.get("typed_rows") != 72 or \
            denominator.get("source_typed_rows") != 846:
        raise ClosureError("source typed denominator must be exact 72/846")
    if value.get("weight_mapping", {}).get("mapping_id") != MAPPING_ID:
        raise ClosureError("source K-pack4 mapping identity differs")
    rows = {row.get("symbol"): row for row in value.get("typed_rows", [])}
    if len(rows) != 72:
        raise ClosureError("source typed symbol denominator is not 72 unique rows")
    selected: list[dict[str, Any]] = []
    for name, expected in EXPECTED.items():
        row = rows.get(name)
        if row is None:
            raise ClosureError(f"missing exact closure row: {name}")
        axes = {
            "qtype": 12, "artifact_tile_k": 0,
            "tile_m": 8, "tile_n": expected["tn"],
            "tactic_tile_k": 256, "warp_m": 8,
            "warp_n": expected["wn"], "stages": 2,
            "bchunk": 0, "a_provider": "standard-aiu",
        }
        contradictions = {
            key: (row.get(key), want) for key, want in axes.items()
            if row.get(key) != want
        }
        if contradictions:
            raise ClosureError(f"symbol/axis contradiction for {name}: {contradictions}")
        selected.append(row)
    return selected


def materialize(source: pathlib.Path, output: pathlib.Path) -> None:
    source_manifest = source / "manifest.json"
    value = json.loads(source_manifest.read_text())
    selected = validate_source(value)
    output.mkdir(parents=True, exist_ok=True)
    units = output / "units"
    units.mkdir(parents=True, exist_ok=True)
    unit = units / "fq_tc_kpack4_fragment_closure_00.cu"

    rows = []
    for i, row in enumerate(selected):
        suffix = " \\" if i + 1 != len(selected) else ""
        rows.append(
            f"  X({row['symbol']},12,0,8,{row['tile_n']},256,8,"
            f"{row['warp_n']},2,0,0){suffix}")
    unit.write_text(
        "// GENERATED -- exact Q4_K K-pack4 fragment-mapping closure.\n"
        "#ifdef PPU_PACKED_SCALE\n#undef PPU_PACKED_SCALE\n#endif\n"
        "#define PPU_PACKED_SCALE 1\n"
        "#ifdef PPU_PACKED_FORMAT\n#undef PPU_PACKED_FORMAT\n#endif\n"
        "#define PPU_PACKED_FORMAT 0\n"
        "#ifdef PPU_B_CHUNK\n#undef PPU_B_CHUNK\n#endif\n"
        "#define PPU_B_CHUNK 0\n"
        "#define FQ_TC_UNIT_ROWS(X) \\\n" + "\n".join(rows) + "\n"
        '#include "fully_quantized_splitk_producer_unit.inc"\n')

    registry = output / "fq_tc_registry.inc"
    registry.write_text(
        "// GENERATED -- exact Q4_K K-pack4 fragment-mapping closure.\n"
        "#define FQ_TC_GENERATED_QTYPE 12\n"
        "#define FQ_TC_GENERATED_ARTIFACT_TK 0\n"
        "#define FQ_TC_GENERATED_BCHUNK 0\n"
        "#define FQ_TC_GENERATED_RAW_ROWS 6\n"
        "#define FQ_TC_GENERATED_TYPED_ROWS 6\n"
        "#define FQ_TC_REGISTRY_ROWS(X) \\\n" + "\n".join(rows) + "\n")

    manifest = {
        "schema": "quactlize.fq-q4k-kpack4-fragment-closure.v1",
        "identity": {
            "qtype": 12, "artifact_tile_k": 0, "bchunk": 0,
            "weight_layout": "q4-kpack4", "mapping_id": MAPPING_ID,
        },
        "source_manifest": str(source_manifest.resolve()),
        "source_typed_denominator": 72,
        "source_global_typed_denominator": 846,
        "selection_denominator": 6,
        "typed_rows": selected,
        "units": [str(unit.resolve())],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "units.cmake").write_text(
        "# GENERATED -- exact Q4_K K-pack4 fragment-mapping closure.\n"
        "set(FQ_TC_GENERATED_UNIT_SOURCES\n"
        f'  "{unit.resolve()}"\n'
        ")\n"
        f'set(FQ_TC_GENERATED_REGISTRY "{registry.resolve()}")\n'
        f'set(FQ_TC_GENERATED_MANIFEST "{manifest_path.resolve()}")\n')
    print("[fq-kpack4-fragment-select] PASS source_typed=72/846 "
          f"selected=6 mapping={MAPPING_ID} output={output}")


def fields(line: str, prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in shlex.split(line.removeprefix(prefix)):
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    return result


def one(text: str, prefix: str) -> dict[str, str]:
    values = [fields(line, prefix) for line in text.splitlines()
              if line.startswith(prefix)]
    if len(values) != 1:
        raise ClosureError(f"{prefix.strip()} denominator is {len(values)}, expected 1")
    return values[0]


def check_fixture(text: str) -> None:
    rows = [fields(line, "FQ_KPACK4_FIXTURE ") for line in text.splitlines()
            if line.startswith("FQ_KPACK4_FIXTURE ")]
    by_phase = {row.get("phase"): row for row in rows}
    if len(rows) != 2 or set(by_phase) != {"prepare", "recover"}:
        raise ClosureError("K-pack4 fixture denominator is not exactly prepare/recover")
    for row in rows:
        if row.get("q") != "12" or row.get("shape") != SHAPE or \
                row.get("mapping_id") != MAPPING_ID or \
                row.get("direct_rc") != "0" or row.get("abi_rc") != "0" or \
                row.get("direct_equal") != "1":
            raise ClosureError(f"K-pack4 fixture contract differs: {row}")
    if by_phase["recover"].get("native_equal") != "1":
        raise ClosureError("K-pack4 recovery is not native-byte exact")


def load_arm(path: pathlib.Path, *, legacy: bool) -> dict[str, dict[str, str]]:
    text = path.read_text()
    check_fixture(text)
    shard = one(text, "FQ_SHARD ")
    done = one(text, "FQ_SHAPE_DONE ")
    common = {
        "q": "12", "A": "0", "bchunk": "0", "shape": SHAPE,
        "weight_layout": "1", "weight_mapping_id": MAPPING_ID,
        "typed_rows": "6", "selected_rows": "6", "only_split": "1",
        "bc_mode": "skip", "bc_batch": "native-grid-y-m-lt8",
    }
    for marker in (shard, done):
        if any(marker.get(key) != value for key, value in common.items()):
            raise ClosureError(f"arm marker identity differs: {marker}")
    if done.get("status") != ("FAIL" if legacy else "PASS"):
        raise ClosureError(f"arm completion status differs: {done}")

    cells = [fields(line, "FQ_TC_CELL ") for line in text.splitlines()
             if line.startswith("FQ_TC_CELL ")]
    by_symbol = {cell.get("symbol"): cell for cell in cells}
    if len(cells) != 6 or set(by_symbol) != set(EXPECTED):
        raise ClosureError(
            f"arm cell denominator differs: rows={len(cells)} symbols={sorted(by_symbol)}")
    for name, expected in EXPECTED.items():
        cell = by_symbol[name]
        axes = {
            "q": "12", "A": "0", "bchunk": "0", "shape": SHAPE,
            "symbol": name, "tm": "8", "tn": str(expected["tn"]),
            "tk": "256", "wm": "8", "wn": str(expected["wn"]),
            "stages": "2", "provider": "standard-aiu", "S": "1",
            "scope": "FULL_OUTPUT", "provider_capacity_rows": "0",
            "reducer_untimed": "0", "partial_bytes": "0",
        }
        if any(cell.get(key) != value for key, value in axes.items()):
            raise ClosureError(f"cell axes differ for {name}: {cell}")
        should_pass = not legacy or bool(expected["legacy_clean"])
        if should_pass:
            required = {
                "state": "MEASURED", "raw_bad": "0",
                "failure_step": "NONE", "failure_repeat": "-1",
            }
            if any(cell.get(key) != value for key, value in required.items()) or \
                    float(cell.get("us", "0")) <= 0 or cell.get("samples") == "[]":
                raise ClosureError(f"clean cell is not measured/raw-bit exact: {cell}")
        else:
            required = {
                "state": "RAW_FP16_MISMATCH",
                "raw_bad": str(expected["legacy_raw_bad"]),
                "failure_step": "RAW_FP16_MISMATCH", "failure_repeat": "0",
                "us": "0.000000000", "samples": "[]",
            }
            if any(cell.get(key) != value for key, value in required.items()):
                raise ClosureError(f"legacy failure signature differs: {cell}")
    return by_symbol


def check(candidate: pathlib.Path, legacy: pathlib.Path,
          candidate_rc: int, legacy_rc: int) -> None:
    if candidate_rc != 0 or legacy_rc != 1:
        raise ClosureError(
            f"arm return codes differ: candidate={candidate_rc} legacy={legacy_rc}")
    load_arm(candidate, legacy=False)
    load_arm(legacy, legacy=True)
    print("[fq-kpack4-fragment-check] PASS candidate=6/6-RAW-BIT "
          "legacy=2/6-clean+4/6-predicted-RED overlap=EXACT-L231")


def fixture(*, legacy: bool) -> str:
    lines = [
        f"FQ_KPACK4_FIXTURE phase=prepare q=12 shape={SHAPE} version=2 "
        "layout=1 bits=4 high_bits=0 artifact_tile_k=0 transport_tile_k=64 "
        f"group_size=32 reserved=0 mapping_id={MAPPING_ID} direct_rc=0 "
        "abi_rc=0 direct_equal=1",
        f"FQ_KPACK4_FIXTURE phase=recover q=12 shape={SHAPE} "
        f"mapping_id={MAPPING_ID} direct_rc=0 abi_rc=0 direct_equal=1 native_equal=1",
        f"FQ_SHARD q=12 A=0 bchunk=0 shape={SHAPE} weight_layout=1 "
        f"weight_mapping_id={MAPPING_ID} typed_rows=6 selected_rows=6 "
        "only_split=1 bc_mode=skip bc_batch=native-grid-y-m-lt8 iterations=1 correctness_repeats=8",
    ]
    for name, expected in EXPECTED.items():
        clean = not legacy or expected["legacy_clean"]
        tail = ("state=MEASURED us=20.000000000 raw_bad=0 "
                "failure_step=NONE failure_repeat=-1 samples=[20.0]" if clean else
                f"state=RAW_FP16_MISMATCH us=0.000000000 raw_bad={expected['legacy_raw_bad']} "
                "failure_step=RAW_FP16_MISMATCH failure_repeat=0 samples=[]")
        lines.append(
            f"FQ_TC_CELL q=12 A=0 bchunk=0 shape={SHAPE} symbol={name} "
            f"tm=8 tn={expected['tn']} tk=256 wm=8 wn={expected['wn']} stages=2 "
            "provider=standard-aiu S=1 scope=FULL_OUTPUT provider_capacity_rows=0 "
            f"reducer_untimed=0 partial_bytes=0 {tail}")
    lines.append(
        f"FQ_SHAPE_DONE q=12 A=0 bchunk=0 shape={SHAPE} weight_layout=1 "
        f"weight_mapping_id={MAPPING_ID} typed_rows=6 selected_rows=6 only_split=1 "
        f"bc_mode=skip bc_batch=native-grid-y-m-lt8 status={'FAIL' if legacy else 'PASS'}")
    return "\n".join(lines) + "\n"


def self_test() -> None:
    source_row = {
        "qtype": 12, "artifact_tile_k": 0, "tile_m": 8,
        "tactic_tile_k": 256, "warp_m": 8, "stages": 2,
        "bchunk": 0, "a_provider": "standard-aiu",
    }
    rows = []
    for i in range(72):
        if i < len(CASES):
            tn, wn, _, _ = CASES[i]
            rows.append({**source_row, "symbol": symbol(tn, wn),
                         "tile_n": tn, "warp_n": wn})
        else:
            rows.append({**source_row, "symbol": f"filler_{i}",
                         "tile_n": 256, "warp_n": 16})
    manifest = {
        "identity": {"qtype": 12, "format": "Q4_K", "artifact_tile_k": 0,
                     "bchunk": 0, "tile_m_filter": 8,
                     "weight_layout": "q4-kpack4"},
        "denominator": {"typed_rows": 72, "source_typed_rows": 846},
        "weight_mapping": {"mapping_id": MAPPING_ID},
        "typed_rows": rows,
    }
    assert len(validate_source(manifest)) == 6
    import tempfile
    with tempfile.TemporaryDirectory(prefix="qz-kpack4-fragment-check-") as temp:
        root = pathlib.Path(temp)
        candidate = root / "candidate.log"
        legacy = root / "legacy.log"
        candidate.write_text(fixture(legacy=False))
        legacy.write_text(fixture(legacy=True))
        check(candidate, legacy, 0, 1)
        negatives = (
            (fixture(legacy=False).replace("raw_bad=0", "raw_bad=32", 1),
             fixture(legacy=True), 0, 1),
            (fixture(legacy=False),
             fixture(legacy=True).replace("raw_bad=512", "raw_bad=32", 1), 0, 1),
            (fixture(legacy=False), fixture(legacy=True), 0, 0),
        )
        for cand_text, legacy_text, cand_rc, legacy_rc in negatives:
            candidate.write_text(cand_text)
            legacy.write_text(legacy_text)
            try:
                check(candidate, legacy, cand_rc, legacy_rc)
            except ClosureError:
                pass
            else:
                raise AssertionError("fragment closure RED control stayed green")
    print("[fq-kpack4-fragment-check:self-test] PASS exact six-row selection; "
          "candidate, legacy-stripe and return-code negatives RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--source-dir", type=pathlib.Path, required=True)
    select_parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    check_parser = sub.add_parser("check")
    check_parser.add_argument("--candidate-log", type=pathlib.Path, required=True)
    check_parser.add_argument("--legacy-log", type=pathlib.Path, required=True)
    check_parser.add_argument("--candidate-rc", type=int, required=True)
    check_parser.add_argument("--legacy-rc", type=int, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "select":
            materialize(args.source_dir.resolve(), args.out_dir.resolve())
        else:
            check(args.candidate_log, args.legacy_log,
                  args.candidate_rc, args.legacy_rc)
        return 0
    except (ClosureError, OSError, KeyError, AssertionError) as error:
        print(f"[fq-kpack4-fragment-check] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
