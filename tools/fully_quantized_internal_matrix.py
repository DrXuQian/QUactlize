#!/usr/bin/env python3
"""Emit the fail-closed support matrix for the dense FullyQuantized sweep.

This is deliberately a *plan authority*, not a performance benchmark.  It
reads the shipping format and tactic X-macros, states which existing launch
edge can measure each algorithm, and never promotes L199's local type proof to
a PPU measurement.  Split-K cells use producer-only timing by request; S1 is
always a complete shipping result and therefore has a different metric scope.

Q8_0 is included explicitly as a denominator control even though the shipping
fully-quantized registry contains only GGUF qtypes 10--14.  Every Q8 cell must
say UNSUPPORTED rather than disappearing or being attached to an unrelated
``rows_per_warp`` switch arm.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from dataclasses import asdict, dataclass


ROOT = pathlib.Path(__file__).resolve().parents[1]
FORMAT_INC = ROOT / "quactlize/include/ppu_format_config.inc"
L199 = ROOT / "dev/fold_derivation/l199_dense_splitk_multiformat_type.cu"
TACTIC_SPACE = ROOT / "quactlize/include/ppu_tactic_space.hpp"
TACTIC_EMITTER = ROOT / "tools/emit_fully_quantized_splitk_superset.cpp"

ARTIFACT_TILE_K = (32, 64, 128, 256)
TACTIC_TILE_K = (32, 64, 128, 256)
BCHUNK_REQUESTS = (0, 1)
SPLITS = (1, 2, 4, 8)
STAGES = (2, 3, 4, 6, 8, 12)
BC_ROWS_PER_WARP = (1, 2, 4, 8)
BC_ALGORITHM = "PLACED_BC_GEMV_FULL_OUTPUT"


@dataclass(frozen=True)
class Format:
    ident: str
    name: str
    qtype: int
    low_bits: int
    high_bits: int
    group_size: int
    scale_first_tile_k: int | None
    fully_quantized_tile_k: int | None
    packed_format: int | None
    shipping_route: str
    tensor_core_fully_quantized: bool


@dataclass(frozen=True)
class Tactic:
    tile_m: int
    tile_n: int
    tactic_tile_k: int
    warp_m: int
    warp_n: int
    stages: int
    bchunk: int
    source_status: str
    source_reason: str

    @property
    def name(self) -> str:
        return (f"{self.tile_m}x{self.tile_n}x{self.tactic_tile_k}_"
                f"w{self.warp_m}x{self.warp_n}_s{self.stages}_bc{self.bchunk}")


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_formats() -> list[Format]:
    pattern = re.compile(
        r'X\((\w+),\s*"([^"]+)",\s*(\d+),\s*(\d+),\s*(\d+),'
        r'\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
    )
    rows = [
        Format(m[0], m[1], *map(int, m[2:]),
               shipping_route="dense_fully_quantized_arrangement_v1",
               tensor_core_fully_quantized=True)
        for m in pattern.findall(FORMAT_INC.read_text())
    ]
    if len(rows) != 5 or {row.qtype for row in rows} != {10, 11, 12, 13, 14}:
        raise RuntimeError(f"shipping FQ format authority drifted: {rows}")
    # Q8_0 has no shipping fully-quantized reader in either the tensor-core
    # registry or the BC GEMV qtype switch (which contains only 10--14).
    # Keeping it in this list is a denominator control that prevents the
    # unsupported route from looking like an unmeasured winner.
    q8 = Format(
        "Q8_0", "Q8_0", 8, 8, 0, 32, 32, None, None,
        "bc_vecdot_q8", False,
    )
    return [q8, *rows]


def emitter_binary() -> pathlib.Path:
    digest = hashlib.sha256(
        TACTIC_EMITTER.read_bytes() + TACTIC_SPACE.read_bytes()).hexdigest()[:16]
    build = pathlib.Path("/workspace") / f"quactlize-fq-emitter-{digest}"
    binary = build / "emit_tactic_configs"
    if binary.is_file():
        return binary
    build.mkdir(parents=True, exist_ok=True)
    partial = build / "emit_tactic_configs.building"
    result = subprocess.run(
        ["c++", "-std=c++17", "-Iquactlize/include",
         str(TACTIC_EMITTER.relative_to(ROOT)), "-o", str(partial)],
        cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("shared tactic emitter did not compile:\n" +
                           result.stdout + result.stderr)
    partial.replace(binary)
    return binary


@functools.lru_cache(maxsize=None)
def emitted_tactics(qtype: int, artifact_tk: int) -> tuple[Tactic, ...]:
    formats = {row.qtype: row for row in parse_formats()}
    fmt = formats[qtype]
    if qtype == 8:
        return ()
    command = [str(emitter_binary()), str(qtype), str(artifact_tk), "0",
               *(str(stage) for stage in STAGES)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            f"shared tactic emitter rejected qtype={qtype} A={artifact_tk}:\n" +
            result.stdout + result.stderr)
    header = re.search(
        r"FQ_SUPERSET_SUMMARY .*? raw=(\d+) eligible=(\d+) rejected=(\d+)",
        result.stdout)
    row_re = re.compile(
        r"^FQ_SUPERSET_ROW q=\d+ A=\d+ tm=(\d+) tn=(\d+) tk=(\d+) "
        r"wm=(\d+) wn=(\d+) stages=(\d+) bchunk=(\d+) "
        r"status=(\S+) reason=(.*)$",
        re.M)
    rows = tuple(Tactic(*map(int, match[:7]), match[7], match[8])
                 for match in row_re.findall(result.stdout))
    eligible = sum(row.source_status == "TYPE_ADMISSION_REQUIRED" for row in rows)
    rejected = sum(row.source_status == "STATIC_REJECT" for row in rows)
    if (not header or int(header.group(1)) != len(rows) or
            int(header.group(2)) != eligible or int(header.group(3)) != rejected or
            len(set(rows)) != len(rows)):
        raise RuntimeError(
            f"cannot bind emitted denominator for qtype={qtype} A={artifact_tk}: "
            f"stamp={header.groups() if header else 'missing'} rows={len(rows)} "
            f"eligible={eligible} rejected={rejected}")
    return rows


def artifact_supported(fmt: Format, artifact_tk: int,
                       tactic_tk: int) -> tuple[bool, str]:
    if not fmt.tensor_core_fully_quantized:
        return False, "Q8_HAS_NO_TENSORCORE_FULLY_QUANTIZED_READER"
    if (fmt.qtype in (11, 13) and artifact_tk == 32) or (
            fmt.qtype == 14 and artifact_tk == 256):
        return False, "FORMAT_ARTIFACT_UNSUPPORTED"
    if artifact_tk > tactic_tk or tactic_tk % artifact_tk:
        return False, "ARTIFACT_DOES_NOT_DIVIDE_TACTIC_TILE_K"
    low_bytes = fmt.low_bits * artifact_tk // 8
    if fmt.high_bits == 0 and low_bytes < 32:
        # Q4/A32 is the one proved sub-32B single-plane artifact: Fold=2 turns
        # each logical N64 x K32 pair into the physical N32 x K64 AIU run,
        # while the folded collective restores logical N x K semantics.  The
        # packed-metadata gate below independently requires logical TK=256.
        if not (fmt.qtype == 12 and artifact_tk == 32):
            return False, "PACKED_SINGLE_PLANE_READER_UNSUPPORTED"
    return True, "SUPPORTED_BY_DESCRIPTOR_PREDICATE"


def packed_metadata_tactic_supported(fmt: Format,
                                     tactic_tk: int) -> tuple[bool, str]:
    """Mirror the active packed-unit collective, not its historical default.

    Single-plane Q2/Q4 currently activate packed metadata only when one
    k-tile consumes exactly one 256-code superblock.  The two-plane collective
    explicitly supports integral group runs inside a superblock, so every
    shared TileK whose group count divides the format's 16/8 groups is a real
    typed candidate.  This distinction is an implementation capability, not a
    reason to erase the other shared TileK values from the denominator.
    """
    if tactic_tk not in TACTIC_TILE_K or tactic_tk % fmt.group_size:
        return False, "PACKED_METADATA_GROUP_RUN_NONINTEGRAL"
    groups = 256 // fmt.group_size
    tile_groups = tactic_tk // fmt.group_size
    if fmt.high_bits == 0:
        ok = tactic_tk == 256
        return (ok, "SUPPORTED_BY_SINGLE_PLANE_PACKED_COLLECTIVE") if ok else (
            False, "SINGLE_PLANE_PACKED_METADATA_REQUIRES_ONE_SUPERBLOCK")
    ok = tile_groups > 0 and groups % tile_groups == 0
    return (ok, "SUPPORTED_BY_TWO_PLANE_PACKED_COLLECTIVE") if ok else (
        False, "TWO_PLANE_PACKED_METADATA_TILE_DOES_NOT_DIVIDE_SUPERBLOCK")


def q4_kpack4_subsuperblock_tactic_supported(
        fmt: Format, tactic_tk: int) -> tuple[bool, str]:
    """Admission for the explicit K-pack4 sub-superblock experiment.

    The offline byte map composes fixed K64 transports and has no tactic-K
    axis. Its one-plane collective can therefore decode an integral run of
    Q4_K's eight gs32 groups while retaining the same 16-byte source unit.
    Keep this separate from ``packed_metadata_tactic_supported`` until the
    anchored device closure is complete so historical Xplane manifests and
    resume authorities do not change implicitly.
    """
    if fmt.qtype != 12 or fmt.high_bits != 0 or fmt.group_size != 32:
        return False, "Q4_KPACK4_SUBSUPERBLOCK_REQUIRES_Q4_K"
    if tactic_tk not in TACTIC_TILE_K or tactic_tk < 64 or tactic_tk % 64:
        return False, "Q4_KPACK4_SUBSUPERBLOCK_REQUIRES_K64_MULTIPLE"
    groups = 256 // fmt.group_size
    tile_groups = tactic_tk // fmt.group_size
    ok = tile_groups > 0 and groups % tile_groups == 0
    return (ok, "SUPPORTED_BY_Q4_KPACK4_SUBSUPERBLOCK_COLLECTIVE") if ok else (
        False, "Q4_KPACK4_TILE_DOES_NOT_DIVIDE_SUPERBLOCK")


def collective_atom_tiling_supported(fmt: Format, tactic: Tactic,
                                     artifact_tk: int) -> tuple[bool, str]:
    """Mirror the sub-byte collective's real 8-row copy-atom divisibility.

    ``common_static_sweep_exclusion`` proves only that each artifact fold
    divides TileN.  The production folded/two-plane collectives have a
    stronger typed requirement: the *physical* N extent must contain an
    integral 8-row PPU swizzle atom.  A real type-only compile caught the
    missing factor for Q3/A64/TK256/TN16; keeping this predicate separate
    records that compiler-derived rejection instead of silently shrinking the
    shared tactic axes.
    """
    low_fold = fold_for_plane(fmt.low_bits, artifact_tk)
    if low_fold <= 0 or tactic.tile_n % (8 * low_fold):
        return False, "LOW_PLANE_COPY_ATOM_DOES_NOT_TILE_FOLDED_N"
    if fmt.high_bits:
        high_fold = fold_for_plane(fmt.high_bits, artifact_tk)
        if high_fold <= 0 or tactic.tile_n % (8 * high_fold):
            return False, "HIGH_PLANE_COPY_ATOM_DOES_NOT_TILE_FOLDED_N"
    return True, "SUPPORTED_BY_COMPILED_COLLECTIVE_ATOM_GEOMETRY"


def bc_artifact_supported(fmt: Format, artifact_tk: int) -> tuple[bool, str]:
    """Mirror bc_vecdot::arrangement_supported_v, not the TC descriptor.

    The distinction is load-bearing: the scalar/whole-word BC reader supports
    folded Q2 A32/A64 and Q4 A32 arrangements that the current tensor-core
    packed reader rejects.  Reusing artifact_supported() here would silently
    erase three real full-output decode candidates.
    """
    if fmt.qtype in (10, 12):
        ok = artifact_tk in (32, 64, 128, 256)
    elif fmt.qtype in (11, 13):
        ok = artifact_tk in (64, 128, 256)
    elif fmt.qtype == 14:
        ok = artifact_tk in (32, 64, 128)
    else:
        ok = False
    return (ok, "SUPPORTED_BY_BC_ARRANGEMENT") if ok else (
        False, "BC_ARRANGEMENT_UNSUPPORTED")


def bchunk_state(fmt: Format, requested: int) -> tuple[bool, str]:
    if requested == 0:
        return True, "0"
    if fmt.qtype == 10:
        return True, "0"  # requested but inert, exactly as L199 reports
    if fmt.qtype == 12:
        return False, "BCHUNK_UNSUPPORTED_BITS"
    return True, "1"


def algorithm(split: int) -> dict:
    if split == 1:
        return {
            "name": "FQ_S1",
            "split_k_slices": 1,
            "metric_scope": "FULL_END_TO_END_SHIPPING_RESULT",
            "reducer": "NOT_APPLICABLE",
        }
    return {
        "name": f"SPLITK_S{split}_PRODUCER",
        "split_k_slices": split,
        "metric_scope": "PRODUCER_ONLY_DIAGNOSTIC_NOT_A_PRODUCT_RESULT",
        "reducer": "EXCLUDED_BY_EXPERIMENT_CONTRACT",
    }


def support_cell(fmt: Format, tactic: Tactic | None, artifact_tk: int | None,
                 split: int, a_provider: str = "standard-aiu") -> dict:
    base = {
        "qtype": fmt.qtype,
        "format": fmt.name,
        "algorithm": algorithm(split)["name"],
        "split_k_slices": split,
        "metric_scope": algorithm(split)["metric_scope"],
        "tactic": asdict(tactic) if tactic else None,
        "artifact_tile_k": artifact_tk,
        "bchunk_requested": tactic.bchunk if tactic else None,
        "a_provider": a_provider,
        "a_provider_capacity_rows": 1 if a_provider == "packed-row" else None,
    }
    if fmt.qtype == 8:
        return base | {
            "status": "UNSUPPORTED",
            "route": None,
            "candidate_denominator": "EXPLICIT_UNSUPPORTED_CELL",
            "reason": "Q8_HAS_NO_SHIPPING_FULLY_QUANTIZED_READER_OR_SPLITK_PRODUCER",
        }

    assert tactic is not None and artifact_tk is not None
    if a_provider not in ("standard-aiu", "packed-row"):
        raise ValueError(f"unknown A provider {a_provider}")
    if tactic.source_status != "TYPE_ADMISSION_REQUIRED":
        return base | {"status": "STATIC_REJECT", "route": None,
                       "reason": tactic.source_reason}
    artifact_ok, artifact_reason = artifact_supported(
        fmt, artifact_tk, tactic.tactic_tile_k)
    packed_ok, packed_reason = packed_metadata_tactic_supported(
        fmt, tactic.tactic_tile_k)
    atom_ok, atom_reason = collective_atom_tiling_supported(
        fmt, tactic, artifact_tk)
    chunk_ok, effective_chunk = bchunk_state(fmt, tactic.bchunk)
    base["bchunk_effective"] = effective_chunk
    if not artifact_ok:
        return base | {"status": "STATIC_REJECT", "route": None,
                       "reason": artifact_reason}
    if not packed_ok:
        return base | {"status": "STATIC_REJECT", "route": None,
                       "reason": packed_reason}
    if not atom_ok:
        return base | {"status": "STATIC_REJECT", "route": None,
                       "reason": atom_reason}
    if not chunk_ok:
        return base | {"status": "STATIC_REJECT", "route": None,
                       "reason": effective_chunk}
    if split == 1:
        return base | {
            "status": "FQ_COMPILED_TYPE_ADMISSION_REQUIRED",
            "route": "quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v1",
            "inventory": "quactlize_ppu_list_valid_dense_fully_quantized_configs_for_arrangement_v1",
            "reason": "REAL_FQ_SHIPPING_SHARED_STORAGE_AND_CAN_IMPLEMENT_REQUIRED",
        }
    return base | {
        "status": "FQ_COMPILED_TYPE_ADMISSION_REQUIRED",
        "route": None,
        "local_evidence": "L199_PACKED_METADATA",
        "reason": "REAL_FQ_SPLIT_SHARED_STORAGE_AND_CAN_IMPLEMENT_REQUIRED",
    }


def packed_a_provider_candidate(fmt: Format, tactic: Tactic,
                                artifact_tk: int) -> bool:
    return (fmt.high_bits == 0 and tactic.tile_m == 8 and tactic.warp_m == 8
            and fold_for_plane(fmt.low_bits, artifact_tk) == 1)


def fold_for_plane(bits: int, artifact_tk: int) -> int:
    if bits == 0:
        return 1
    run = bits * artifact_tk // 8
    if bits * artifact_tk % 8 or run <= 0 or (run < 32 and 32 % run):
        return 0
    return 1 if run >= 32 else 32 // run


def expanded_cells(formats: list[Format]) -> list[dict]:
    cells: list[dict] = []
    for fmt in formats:
        if fmt.qtype == 8:
            for split in SPLITS:
                cells.append(support_cell(fmt, None, None, split))
            cells.append({
                "qtype": fmt.qtype,
                "format": fmt.name,
                "algorithm": BC_ALGORITHM,
                "split_k_slices": None,
                "metric_scope": "FULL_END_TO_END_BC_GEMV_RESULT",
                "tactic": None,
                "artifact_tile_k": None,
                "bchunk_requested": None,
                "status": "UNSUPPORTED",
                "route": None,
                "candidate_denominator": "EXPLICIT_UNSUPPORTED_CELL",
                "reason": "Q8_HAS_NO_SHIPPING_PLACED_BC_READER",
            })
            continue
        for artifact in ARTIFACT_TILE_K:
            for tactic in emitted_tactics(fmt.qtype, artifact):
                for split in SPLITS:
                    cells.append(support_cell(
                        fmt, tactic, artifact, split, "standard-aiu"))
                    if packed_a_provider_candidate(fmt, tactic, artifact):
                        cells.append(support_cell(
                            fmt, tactic, artifact, split, "packed-row"))
        # The placed BC reader consumes the same artifact but is not a tensor-
        # core tactic.  It therefore owns one full-output cell per legal
        # arrangement, outside the (TM,TN,TK,WM,WN,stage,BChunk) product.
        for artifact in ARTIFACT_TILE_K:
            supported, reason = bc_artifact_supported(fmt, artifact)
            for rows_per_warp in BC_ROWS_PER_WARP:
                # Threads is a derived implementation property, not a search
                # axis: only Q4/RPW4 uses its measured 128-thread topology.
                threads = 128 if fmt.qtype == 12 and rows_per_warp == 4 else 256
                cells.append({
                    "qtype": fmt.qtype,
                    "format": fmt.name,
                    "algorithm": BC_ALGORITHM,
                    "split_k_slices": None,
                    "metric_scope": "FULL_END_TO_END_BC_GEMV_RESULT",
                    "tactic": {"rows_per_warp": rows_per_warp,
                               "threads": threads},
                    "artifact_tile_k": artifact,
                    "bchunk_requested": None,
                    "status": "BC_COMPILED_TYPE_ADMISSION_REQUIRED" if supported
                              else "STATIC_REJECT",
                    "route": "gguf_scale::bc_vecdot::launch_fixed",
                    "reason": "REAL_PLACED_BC_READER_REQUIRED" if supported else reason,
                })
    return cells


def make_manifest(expand: bool) -> dict:
    formats = parse_formats()
    cells = expanded_cells(formats)
    tensor_cells = sum(cell["algorithm"] != BC_ALGORITHM and
                       cell["qtype"] != 8 for cell in cells)
    # Keep the Q8 BC sentinel in q8_explicit_route_cells, not also in the
    # supported-format BC subtotal.  The old field counted that one cell in
    # both named subtotals even though total_support_cells itself was correct.
    bc_cells = sum(cell["algorithm"] == BC_ALGORITHM and cell["qtype"] != 8
                   for cell in cells)
    result = {
        "schema": "quactlize-fq-internal-support-v2",
        "scope": "support-and-denominator-only-no-device-performance",
        "authorities": {
            str(FORMAT_INC.relative_to(ROOT)): sha256(FORMAT_INC),
            str(TACTIC_SPACE.relative_to(ROOT)): sha256(TACTIC_SPACE),
            str(TACTIC_EMITTER.relative_to(ROOT)): sha256(TACTIC_EMITTER),
            str(L199.relative_to(ROOT)): sha256(L199),
        },
        "measurement_contract": {
            "FQ_S1": "full end-to-end shipping result",
            "SPLITK_S2_S4_S8": "producer-only diagnostic; reducer excluded; never rank against S1 as product E2E",
            "PLACED_BC_GEMV": "full-output decode result; may rank against FQ S1",
            "performance_values": "none are generated by this manifest",
        },
        "formats": [asdict(row) for row in formats],
        "candidate_source": (
            "shared ppu_tactic_space raw axes plus common_static_sweep_exclusion; "
            "ScaleFirst shared-memory exclusion deliberately deferred to real FQ compiled types"),
        "axes": {
            "artifact_tile_k": list(ARTIFACT_TILE_K),
            "tactic_tile_k": list(TACTIC_TILE_K),
            "default_fq_tactic_tile_k": {
                str(fmt.qtype): fmt.fully_quantized_tile_k
                for fmt in formats if fmt.tensor_core_fully_quantized
            },
            "bchunk_requested": list(BCHUNK_REQUESTS),
            "split_k_slices": list(SPLITS),
            "stages": list(STAGES),
        },
        "denominator": {
            "tensor_core_fq_cells": tensor_cells,
            "q8_explicit_route_cells": len(SPLITS) + 1,
            "placed_bc_cells": bc_cells,
            "total_support_cells": len(cells),
            "formula": "standard-A: 5 formats * 4 artifacts * (4 TacticTileK * 5760 topology rows) * 4 TC algorithms; plus an explicit packed-row-A candidate for every structural TM8/WM8 unfolded one-plane row; plus 5 Q8 unsupported cells (BC + TC S1/S2/S4/S8); plus 20 supported-format BC arrangements * 4 RowsPerWarp; every unsupported cell remains named",
        },
        "status_counts": {},
        "q8_splitk_verdict": "UNSUPPORTED_EXPLICIT_NOT_OMITTED",
    }
    for cell in cells:
        result["status_counts"][cell["status"]] = (
            result["status_counts"].get(cell["status"], 0) + 1)
    if expand:
        result["cells"] = cells
    return result


def self_test() -> None:
    formats = parse_formats()
    manifest = make_manifest(True)
    cells = manifest["cells"]
    assert len(formats) == 6 and formats[0].qtype == 8
    for qtype in (10, 11, 12, 13, 14):
        for artifact in ARTIFACT_TILE_K:
            rows = emitted_tactics(qtype, artifact)
            assert len(rows) == 23040
            assert {row.tactic_tile_k for row in rows} == set(TACTIC_TILE_K)
    # TM8 selects the m8 atom through WM8 only.  WN remains a generic tactic
    # axis; all producer-supported widths must survive on the exact Q4/A64
    # decode family instead of using WN pruning as a correctness workaround.
    q4_a64 = emitted_tactics(12, 64)
    m8_wn = {
        row.warp_n for row in q4_a64
        if (row.tile_m, row.tile_n, row.tactic_tile_k, row.warp_m,
            row.stages, row.bchunk, row.source_status) ==
           (8, 64, 256, 8, 3, 0, "TYPE_ADMISSION_REQUIRED")
    }
    assert m8_wn == {16, 32, 64}
    q4 = next(fmt for fmt in formats if fmt.qtype == 12)
    q2 = next(fmt for fmt in formats if fmt.qtype == 10)
    assert artifact_supported(q4, 32, 256) == (
        True, "SUPPORTED_BY_DESCRIPTOR_PREDICATE")
    assert packed_metadata_tactic_supported(q4, 256) == (
        True, "SUPPORTED_BY_SINGLE_PLANE_PACKED_COLLECTIVE")
    assert {
        tk: q4_kpack4_subsuperblock_tactic_supported(q4, tk)[0]
        for tk in TACTIC_TILE_K
    } == {32: False, 64: True, 128: True, 256: True}
    assert artifact_supported(q2, 32, 256) == (
        False, "PACKED_SINGLE_PLANE_READER_UNSUPPORTED")
    q4_a32_tm8 = [
        row for row in emitted_tactics(12, 32)
        if row.bchunk == 0 and row.tile_m == 8 and
        row.source_status == "TYPE_ADMISSION_REQUIRED" and
        artifact_supported(q4, 32, row.tactic_tile_k)[0] and
        packed_metadata_tactic_supported(q4, row.tactic_tile_k)[0] and
        collective_atom_tiling_supported(q4, row, 32)[0]
    ]
    assert len(q4_a32_tm8) == 12
    assert {(row.tile_n, row.warp_n) for row in q4_a32_tm8} == {
        (64, 32), (128, 64)}
    assert {row.stages for row in q4_a32_tm8} == set(STAGES)
    expected_supported_bc = 5 * 4 * 4
    expected_bc = expected_supported_bc + 1  # physical cell list includes Q8 sentinel
    expected_standard = 5 * 4 * 23040 * 4
    assert manifest["denominator"]["tensor_core_fq_cells"] >= expected_standard
    assert manifest["denominator"]["placed_bc_cells"] == expected_supported_bc
    assert len(cells) == (manifest["denominator"]["tensor_core_fq_cells"] +
                          manifest["denominator"]["q8_explicit_route_cells"] +
                          manifest["denominator"]["placed_bc_cells"])
    q8 = [cell for cell in cells if cell["qtype"] == 8]
    assert len(q8) == 5
    assert all(cell["status"] == "UNSUPPORTED" for cell in q8)
    assert all(cell["route"] is None for cell in q8)
    expected_q8_algorithms = {
        "FQ_S1", "SPLITK_S2_PRODUCER", "SPLITK_S4_PRODUCER",
        "SPLITK_S8_PRODUCER", BC_ALGORITHM,
    }
    assert {cell["algorithm"] for cell in q8} == expected_q8_algorithms
    # Negative: deleting Q8 BC while forging the declared arithmetic total
    # still fails the algorithm identity denominator.
    planted_q8 = [cell for cell in q8 if cell["algorithm"] != BC_ALGORITHM]
    assert len(planted_q8) == 4
    assert {cell["algorithm"] for cell in planted_q8} != expected_q8_algorithms
    assert all(cell["metric_scope"] ==
               "PRODUCER_ONLY_DIAGNOSTIC_NOT_A_PRODUCT_RESULT"
               for cell in cells if isinstance(cell["split_k_slices"], int)
               and cell["split_k_slices"] > 1)
    assert all(cell["metric_scope"] == "FULL_END_TO_END_SHIPPING_RESULT"
               for cell in cells if cell["split_k_slices"] == 1)
    bc = [cell for cell in cells if cell["algorithm"] == BC_ALGORITHM]
    assert len(bc) == expected_bc
    assert sum(cell["status"] == "BC_COMPILED_TYPE_ADMISSION_REQUIRED"
               for cell in bc) == 17 * 4
    assert {cell["tactic"]["rows_per_warp"] for cell in bc
            if cell["tactic"] is not None} == set(BC_ROWS_PER_WARP)
    assert all(cell["metric_scope"] == "FULL_END_TO_END_BC_GEMV_RESULT"
               for cell in bc)
    # Negative denominator controls: the exact denominator is provider-expanded.
    # Dropping one emitted provider row or one complete Split-K plane must change
    # the total rather than remain a plausible green manifest.
    tensor_total = manifest["denominator"]["tensor_core_fq_cells"]
    assert tensor_total - 1 + 4 + expected_bc != len(cells)
    assert tensor_total - tensor_total // len(SPLITS) + 4 + expected_bc != len(cells)
    assert any(cell["reason"] == "PACKED_SINGLE_PLANE_READER_UNSUPPORTED"
               for cell in cells)
    assert any(cell["reason"] == "BCHUNK_UNSUPPORTED_BITS"
               for cell in cells)
    assert any(cell.get("a_provider") == "packed-row" for cell in cells)
    assert any(cell.get("a_provider") == "standard-aiu" and
               cell["status"] == "FQ_COMPILED_TYPE_ADMISSION_REQUIRED"
               for cell in cells)
    assert all(cell.get("a_provider_capacity_rows") == 1
               for cell in cells if cell.get("a_provider") == "packed-row")
    assert all(cell.get("a_provider_capacity_rows") is None
               for cell in cells if cell.get("a_provider") == "standard-aiu")
    print("[fq-internal-matrix:self-test] PASS formats=6 tensor_fq=5 "
          f"cells={len(cells)} raw_axes=BOUND fq_smem=DEFERRED_TO_COMPILED_TYPE "
          "tacticTK=FULL_SHARED_SPACE/default-marked "
          "TM8=WM8/WN16+32+64 "
          "bc=80-supported+1-q8-sentinel/68-legal-rpw-expanded "
          "q8=5xUNSUPPORTED metric_scope=BOUND")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("emit", "self-test"))
    parser.add_argument("--out", default="-", help="JSON output path; '-' is stdout")
    parser.add_argument("--expand", action="store_true",
                        help="include all 1,764 support cells")
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
            return 0
        manifest = make_manifest(args.expand)
        payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if args.out == "-":
            sys.stdout.write(payload)
        else:
            path = pathlib.Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload)
        return 0
    except (AssertionError, OSError, RuntimeError, ValueError) as exc:
        print(f"[fq-internal-matrix] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
