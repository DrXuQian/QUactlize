#!/usr/bin/env python3
"""Fail-closed local contract for every one-plane dense fixed-Split-K row."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HOST = ROOT / "dev/fold_derivation/l198_dense_splitk_oneplane.cpp"
TYPES = ROOT / "dev/fold_derivation/l198_dense_splitk_oneplane_types.cu"
RUNNER = ROOT / "dev/fold_derivation/run_l198_dense_splitk_oneplane.sh"
HANDLE = ROOT / "quactlize/include/dense_splitk_parallel_ppu.cuh"

ROW_RE = re.compile(
    r"^\s*X\((\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),B\)\s*\\?\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class TableSpec:
    name: str
    path: Path
    prefix: str
    bits: int
    artifact_tile_k: int
    rows: int
    tactic_tile_ks: tuple[int, ...]
    b_chunks: tuple[int, ...]


TABLES = (
    TableSpec("i4", ROOT / "benchmarks/lowbit_dense_configs.inc",
              "LOWBIT_DENSE_CFG", 4, 64, 1772, (64, 128, 256), (0,)),
    TableSpec("i2", ROOT / "benchmarks/lowbit_dense_i2_configs.inc",
              "LOWBIT_DENSE_I2_CFG", 2, 128, 2140, (128, 256), (0, 1)),
    TableSpec("i1", ROOT / "benchmarks/lowbit_dense_i1_configs.inc",
              "LOWBIT_DENSE_I1_CFG", 1, 256, 878, (256,), (0, 1)),
)
SPLITS = (1, 2, 4, 8)
MODES = ("ScaleOnly", "ScaleZero")


@dataclass(frozen=True)
class Counts:
    rows: int
    cells: int
    admitted: int
    inadmissible: int
    per_split: tuple[int, int, int, int]
    per_format: tuple[int, int, int]
    per_b_chunk: tuple[int, int]


EXPECTED = Counts(
    rows=4790,
    cells=38320,
    admitted=33004,
    inadmissible=5316,
    per_split=(9580, 9384, 8088, 5952),
    per_format=(12908, 14556, 5540),
    per_b_chunk=(22956, 10048),
)


def macro_int(text: str, name: str) -> int | None:
    match = re.search(rf"^#define\s+{re.escape(name)}\s+(\d+)\s*$", text, re.M)
    return int(match.group(1)) if match else None


def parse_tables(texts: dict[str, str]) -> dict[str, list[tuple[int, ...]]]:
    return {
        spec.name: [tuple(map(int, match.groups()))
                    for match in ROW_RE.finditer(texts[spec.name])]
        for spec in TABLES
    }


def calculate(rows_by_format: dict[str, list[tuple[int, ...]]],
              splits: tuple[int, ...] = SPLITS,
              mode_count: int = len(MODES)) -> Counts:
    per_split = [0] * len(splits)
    per_format: list[int] = []
    per_b_chunk = [0, 0]
    row_count = sum(len(rows_by_format[spec.name]) for spec in TABLES)
    admitted = 0
    for spec in TABLES:
        format_admitted = 0
        for row in rows_by_format[spec.name]:
            _, _, tactic_k, _, _, stages, b_chunk = row
            k_tiles = 4096 // tactic_k
            for split_index, split in enumerate(splits):
                runnable = (
                    4096 % tactic_k == 0
                    and k_tiles % split == 0
                    and (split == 1 or k_tiles // split >= stages - 1)
                )
                if not runnable:
                    continue
                cells = mode_count
                admitted += cells
                format_admitted += cells
                per_split[split_index] += cells
                if b_chunk in (0, 1):
                    per_b_chunk[b_chunk] += cells
        per_format.append(format_admitted)
    total_cells = row_count * len(splits) * mode_count
    return Counts(row_count, total_cells, admitted, total_cells - admitted,
                  tuple(per_split), tuple(per_format), tuple(per_b_chunk))


def require(text: str, token: str, owner: str, bad: list[str]) -> None:
    if token not in text:
        bad.append(f"{owner}: missing {token!r}")


def audit(rows_by_format: dict[str, list[tuple[int, ...]]],
          texts: dict[str, str], files: dict[str, str],
          splits: tuple[int, ...] = SPLITS,
          mode_count: int = len(MODES)) -> list[str]:
    bad: list[str] = []
    counts = calculate(rows_by_format, splits, mode_count)
    if counts != EXPECTED:
        bad.append(f"denominator: got {counts}, expected {EXPECTED}")

    for spec in TABLES:
        text = texts[spec.name]
        rows = rows_by_format[spec.name]
        if macro_int(text, f"{spec.prefix}_BITS") != spec.bits:
            bad.append(f"{spec.name}: bits stamp drifted")
        if macro_int(text, f"{spec.prefix}_ARTIFACT_TILEK") != spec.artifact_tile_k:
            bad.append(f"{spec.name}: artifact stamp drifted")
        if macro_int(text, f"{spec.prefix}_ROWS") != spec.rows:
            bad.append(f"{spec.name}: row-count stamp drifted")
        if len(rows) != spec.rows or len(set(rows)) != spec.rows:
            bad.append(f"{spec.name}: rows/unique={len(rows)}/{len(set(rows))}, expected {spec.rows}")
        if spec.bits * spec.artifact_tile_k // 8 != 32:
            bad.append(f"{spec.name}: resident run stopped being exact unfolded 32 B")
        if {row[2] for row in rows} != set(spec.tactic_tile_ks):
            bad.append(f"{spec.name}: tactic TileK domain drifted")
        if {row[6] for row in rows} != set(spec.b_chunks):
            bad.append(f"{spec.name}: BChunk domain drifted")
        if any(row[2] < spec.artifact_tile_k or
               row[2] % spec.artifact_tile_k for row in rows):
            bad.append(f"{spec.name}: tactic no longer consumes whole resident artifacts")
        if spec.b_chunks == (0, 1):
            bc0 = {row[:6] for row in rows if row[6] == 0}
            bc1 = {row[:6] for row in rows if row[6] == 1}
            if bc0 != bc1:
                bad.append(f"{spec.name}: BC0/BC1 shipping tactic projections differ")
        for bc in spec.b_chunks:
            witness = (8, 128, spec.artifact_tile_k, 8, 32, 3, bc)
            if witness not in rows:
                bad.append(f"{spec.name}: compiled type witness is not a shipping row: {witness}")

    host = files["host"]
    for token in (
        "LOWBIT_DENSE_CFG_LIST(L198_I4_ROW",
        "LOWBIT_DENSE_I2_CFG_LIST(L198_I2_ROW",
        "LOWBIT_DENSE_I1_CFG_LIST(L198_I1_ROW",
        "fs::work_for_linear(params, linear)",
        "fs::work_matches_params(params, work)",
        "Mode::ScaleOnly, Mode::ScaleZero",
        "fp16_bits(s1[size_t(n)]) == fp16_bits(reduced[size_t(n)])",
        "size_t(split) * size_t(kN) * sizeof(float)",
        "tables=3 rows=%llu modes=2 cells=%llu admitted=%llu",
        "inadmissible_pipeline_depth=%llu",
        "resident_payload_fingerprint(row)",
        "Plant::Bits", "Plant::Mode", "Plant::Artifact", "Plant::Fold",
        "Plant::BChunk", "Plant::Partial",
    ):
        require(host, token, "host oracle", bad)

    types = files["types"]
    for token in (
        "DenseKernelTypes<",
        "DensePackedAKernelTypes<",
        "dense_splitk_parallel_ppu::KernelTypes<Shipping",
        "ExpectedShippingKernel",
        "L198_S1_SHIPPING_TYPE_IDENTITY",
        "L198_S_GT_1_EXACT_COLLECTIVE_REUSE",
        "SeparateKernelCompletion",
        "L198_PARTIAL_ABI_MUST_REMAIN_FP32",
        "ppu_mixed_policy::OrdinaryBProvider",
        "CollectiveBuilderType::",
        "fold_schedule_traits<",
        "L198_PRODUCTION_RESIDENT_ARTIFACT_READER_SEAM",
        "typename Kernel::Arguments args",
        "one_plane_metadata_arguments_valid<Shipping>(args.mainloop)",
        "shipping_group_size_arguments_valid<Shipping>(args.mainloop)",
        "nullptr, StaticGroupSize, so_good_kernel",
        "z, StaticGroupSize, so_bad_kernel",
        "z, StaticGroupSize, sz_good_kernel",
        "nullptr, StaticGroupSize, sz_bad_kernel",
        "nullptr, WrongGroupSize, gs_bad_kernel",
        "I2ScaleOnlyGs16",
        "I2ScaleZeroGs16",
        "arguments=SO:null+/-nonnull- SZ:nonnull+/null-",
        "static_gs=16/32:match+/mismatch-",
        "#if PPU_B_CHUNK == 0",
    ):
        require(types, token, "compiled type/argument oracle", bad)

    handle = files["handle"]
    for token in (
        "one_plane_metadata_arguments_valid(",
        "CollectiveMainloop::has_zero_channel",
        "return CollectiveMainloop::has_zero_channel ? zeros != nullptr",
        "if (!one_plane_metadata_arguments_valid<ShippingTypes>(zeros))",
        "shipping_group_size_arguments_valid(int group_size)",
        "DispatchPolicy::StaticGroupSize",
        "if (!shipping_group_size_arguments_valid<ShippingTypes>(group_size))",
    ):
        require(handle, token, "production fixed-SplitK wrapper", bad)
    if handle.count("if (!one_plane_metadata_arguments_valid<ShippingTypes>(zeros))") != 2:
        bad.append("production fixed-SplitK wrapper: prepared/generic admission is not shared exactly twice")
    if handle.count("!shipping_group_size_arguments_valid<ShippingTypes>(group_size)") != 2:
        bad.append("production fixed-SplitK wrapper: static group-size admission is not shared exactly twice")

    runner = files["runner"]
    for token in (
        "l198_dense_splitk_oneplane.cpp",
        "l198_dense_splitk_oneplane_types.cu",
        "for plant in bits mode artifact fold bchunk partial",
        "for bc in 0 1",
        "L198_FORMAT_BITS_SEAM",
        "L198_FORMAT_MODE_SEAM",
        "L198_FORMAT_ARTIFACT_SEAM",
        "L198_FORMAT_BCHUNK_SEAM",
    ):
        require(runner, token, "runner", bad)
    return bad


def main() -> int:
    paths = [spec.path for spec in TABLES] + [HOST, TYPES, RUNNER, HANDLE]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        print("[dense-splitk-oneplane] FAIL missing: " + ", ".join(missing))
        return 1

    texts = {spec.name: spec.path.read_text() for spec in TABLES}
    rows = parse_tables(texts)
    files = {
        "host": HOST.read_text(), "types": TYPES.read_text(),
        "runner": RUNNER.read_text(), "handle": HANDLE.read_text(),
    }
    bad = audit(rows, texts, files)
    if bad:
        for item in bad:
            print(f"[dense-splitk-oneplane] FAIL: {item}")
        return 1

    controls: list[tuple[str, dict[str, list[tuple[int, ...]]],
                         dict[str, str], dict[str, str], tuple[int, ...], int]] = []
    dropped = {name: list(value) for name, value in rows.items()}
    dropped["i4"] = dropped["i4"][:-1]
    controls.append(("dropped-row", dropped, texts, files, SPLITS, len(MODES)))
    duplicate = {name: list(value) for name, value in rows.items()}
    duplicate["i2"].append(duplicate["i2"][0])
    controls.append(("duplicate-row", duplicate, texts, files, SPLITS, len(MODES)))
    bc_drift = {name: list(value) for name, value in rows.items()}
    first_i1 = bc_drift["i1"][0]
    bc_drift["i1"][0] = (*first_i1[:6], 1 - first_i1[6])
    controls.append(("bchunk-projection", bc_drift, texts, files, SPLITS, len(MODES)))
    artifact_texts = dict(texts)
    artifact_texts["i2"] = artifact_texts["i2"].replace(
        "#define LOWBIT_DENSE_I2_CFG_ARTIFACT_TILEK 128",
        "#define LOWBIT_DENSE_I2_CFG_ARTIFACT_TILEK 64", 1)
    controls.append(("artifact-stamp", rows, artifact_texts, files, SPLITS, len(MODES)))
    controls.append(("missing-S8", rows, texts, files, (1, 2, 4), len(MODES)))
    controls.append(("missing-mode", rows, texts, files, SPLITS, 1))
    for name, owner, token in (
        ("lost-real-mainloop-arguments", "types", "typename Kernel::Arguments args"),
        ("lost-artifact-reader", "types", "L198_PRODUCTION_RESIDENT_ARTIFACT_READER_SEAM"),
        ("lost-zero-channel", "handle", "CollectiveMainloop::has_zero_channel"),
        ("lost-static-group-size", "handle", "DispatchPolicy::StaticGroupSize"),
        ("lost-fp32-partial", "types", "L198_PARTIAL_ABI_MUST_REMAIN_FP32"),
    ):
        planted_files = dict(files)
        planted_files[owner] = planted_files[owner].replace(token, "PLANTED_ABSENT")
        controls.append((name, rows, texts, planted_files, SPLITS, len(MODES)))

    escaped = []
    for name, planted_rows, planted_texts, planted_files, splits, mode_count in controls:
        if not audit(planted_rows, planted_texts, planted_files, splits, mode_count):
            escaped.append(name)
    if escaped:
        print("[dense-splitk-oneplane] FAIL escaped plants: " + ", ".join(escaped))
        return 1

    print(
        "[dense-splitk-oneplane] PASS: tables=3 rows=4790 modes=2 "
        "cells=38320 runnable=33004 inadmissible=5316 "
        "per_split=9580/9384/8088/5952 formats=12908/14556/5540 "
        "bc=22956/10048 source_plants=11"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
