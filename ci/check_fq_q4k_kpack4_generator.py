#!/usr/bin/env python3
"""Fail-closed contract for the native Q4_K K-pack4 tactic generator.

K-pack4 deliberately has no ArtifactTileK axis.  Until the common tactic
emitter grows a layout-neutral entry point, the generator uses the complete
Q4/A64 raw topology as an enumeration source and applies an independent
K-pack4 admission predicate.  This check proves that the resulting TM8
geometry and AP0/AP1 provider product are exactly the established A64
denominator, while keeping the legacy xplane manifest schema unchanged.
"""

from __future__ import annotations

import copy
import pathlib
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import gen_fully_quantized_splitk_producer_units as generator  # noqa: E402


MAPPING_ID = "0x51344b5034540001"
GEOMETRY_KEYS = (
    "tile_m", "tile_n", "tactic_tile_k", "warp_m", "warp_n", "stages",
    "bchunk",
)


class CheckError(ValueError):
    pass


def geometry(row: dict) -> tuple[int, ...]:
    return tuple(int(row[key]) for key in GEOMETRY_KEYS)


def validate(xplane: dict, kpack4: dict) -> None:
    x_identity = xplane.get("identity", {})
    if x_identity != {
            "qtype": 12, "format": "Q4_K", "artifact_tile_k": 64,
            "bchunk": 0, "tile_m_filter": 8}:
        raise CheckError(f"legacy xplane identity/schema drifted: {x_identity}")
    if "weight_mapping" in xplane or "weight_layout" in x_identity:
        raise CheckError("default xplane manifest gained a resume-breaking field")
    x_den = xplane.get("denominator", {})
    expected_x = {
        "raw_topology_rows": 11520,
        "provider_expanded_rows": 12000,
        "source_typed_rows": 918,
        "typed_rows": 144,
        "selection_reject_rows": 774,
        "static_reject_rows": 11082,
        "runtime_tc_cells": 48000,
        "typed_runtime_tc_cells": 576,
    }
    if x_den != expected_x:
        raise CheckError(f"legacy xplane denominator drifted: {x_den}")

    identity = kpack4.get("identity", {})
    expected_identity = {
        "qtype": 12, "format": "Q4_K", "artifact_tile_k": 0,
        "bchunk": 0, "tile_m_filter": 8,
        "weight_layout": "q4-kpack4",
    }
    if identity != expected_identity:
        raise CheckError(f"K-pack4 identity drifted: {identity}")
    mapping = kpack4.get("weight_mapping", {})
    expected_mapping = {
        "layout": "q4-kpack4-transpose-v1",
        "mapping_id": MAPPING_ID,
        "artifact_tile_k_is_not_an_axis": True,
        "transport_tile_k": 64,
        "transport_tile_n": 16,
    }
    if mapping != expected_mapping:
        raise CheckError(f"K-pack4 mapping identity drifted: {mapping}")
    den = kpack4.get("denominator", {})
    expected_den = {
        "raw_topology_rows": 11520,
        "provider_expanded_rows": 12000,
        "source_typed_rows": 918,
        "typed_rows": 144,
        "selection_reject_rows": 774,
        "static_reject_rows": 11082,
        "runtime_tc_cells": 48000,
        "typed_runtime_tc_cells": 576,
    }
    if den != expected_den:
        raise CheckError(f"K-pack4 denominator drifted: {den}")
    if (den["typed_rows"] + den["selection_reject_rows"] +
            den["static_reject_rows"] != den["provider_expanded_rows"]):
        raise CheckError("K-pack4 selected/rejected partition is incomplete")

    x_rows = xplane.get("typed_rows", [])
    rows = kpack4.get("typed_rows", [])
    if len(x_rows) != 144 or len(rows) != 144:
        raise CheckError(
            f"TM8 provider denominator changed: xplane={len(x_rows)} "
            f"kpack4={len(rows)}")
    x_product = {(geometry(row), row.get("a_provider")) for row in x_rows}
    kpack_product = {(geometry(row), row.get("a_provider")) for row in rows}
    if len(x_product) != 144 or len(kpack_product) != 144:
        raise CheckError("typed geometry/provider product contains a duplicate")
    if x_product != kpack_product:
        raise CheckError(
            "K-pack4 geometry/provider product differs from xplane A64")
    for row in rows:
        if (row.get("qtype"), row.get("artifact_tile_k"),
                row.get("bchunk")) != (12, 0, 0) or \
                row.get("a_provider") not in ("standard-aiu", "packed-row"):
            raise CheckError(f"K-pack4 row carries a foreign axis: {row}")
        if (row.get("tile_m") != 8 or row.get("warp_m") != 8 or
                row.get("tactic_tile_k") != 256 or
                int(row.get("tile_n", 0)) % 16 != 0):
            raise CheckError(f"K-pack4 transport predicate was bypassed: {row}")
        if not str(row.get("symbol", "")).startswith("fq_tc_q12_a0_"):
            raise CheckError(f"K-pack4 symbol retained an artifact identity: {row}")
    if kpack4.get("placed_bc") != {
            "compiled_in_bchunk0": False,
            "reason": "Q4_KPACK4_HAS_NO_BC_READER"}:
        raise CheckError("K-pack4 accidentally advertised a BC reader")


def expect_invalid(*, qtype: int, artifact: int, bchunk: int) -> None:
    with tempfile.TemporaryDirectory(prefix="qz-kpack4-invalid-") as temp:
        try:
            generator.generate(
                qtype, artifact, bchunk, pathlib.Path(temp), 4, False, 8,
                "q4-kpack4")
        except ValueError:
            return
    raise CheckError(
        f"invalid K-pack4 identity stayed green: q={qtype} A={artifact} "
        f"bc={bchunk}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="qz-kpack4-generator-") as temp:
        root = pathlib.Path(temp)
        xplane = generator.generate(12, 64, 0, root / "xplane", 4,
                                    False, 8, "xplane")
        kpack4 = generator.generate(12, 0, 0, root / "kpack4", 4,
                                    False, 8, "q4-kpack4")
        validate(xplane, kpack4)

        plants: list[dict] = []
        broken = copy.deepcopy(kpack4)
        broken["weight_mapping"]["mapping_id"] = "0x0"
        plants.append(broken)
        broken = copy.deepcopy(kpack4)
        broken["typed_rows"].pop()
        plants.append(broken)
        broken = copy.deepcopy(kpack4)
        broken["typed_rows"][0]["a_provider"] = "foreign-provider"
        plants.append(broken)
        broken = copy.deepcopy(kpack4)
        broken["identity"]["artifact_tile_k"] = 64
        plants.append(broken)
        for broken in plants:
            try:
                validate(xplane, broken)
            except CheckError:
                pass
            else:
                raise CheckError("K-pack4 generator negative control stayed green")

        try:
            generator.generate(12, 0, 0, root / "drop-last", 4,
                               True, 8, "q4-kpack4")
        except RuntimeError:
            pass
        else:
            raise CheckError("K-pack4 dropped-row denominator stayed green")

    expect_invalid(qtype=11, artifact=0, bchunk=0)
    expect_invalid(qtype=12, artifact=64, bchunk=0)
    expect_invalid(qtype=12, artifact=0, bchunk=1)
    print("[fq-q4k-kpack4-generator:self-test] PASS "
          "xplane=144/918 unchanged Kpack4=144/918 raw=11520 "
          "geometry=A64/AP0+AP1-exact mapping=0x51344b5034540001; "
          "four manifest and four generator negatives RED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CheckError, OSError, RuntimeError, AssertionError) as error:
        print(f"[fq-q4k-kpack4-generator:self-test] FAIL: {error}")
        raise SystemExit(2)
