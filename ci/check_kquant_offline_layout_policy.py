#!/usr/bin/env python3
"""Guard the canonical five-format K-pack policy and resident boundary."""

from __future__ import annotations

import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quactlize import formats  # noqa: E402


def require(text: str, tokens: tuple[str, ...], label: str) -> None:
    missing = [token for token in tokens if token not in text]
    if missing:
        raise AssertionError(f"{label} lacks {missing}")


def main() -> int:
    try:
        expected = {
            formats.QuantType.Q2_K: "kquant-kpack",
            formats.QuantType.Q3_K: "kquant-kpack",
            formats.QuantType.Q4_K: "q4-kpack4",
            formats.QuantType.Q5_K: "kquant-kpack",
            formats.QuantType.Q6_K: "kquant-kpack",
        }
        observed = {
            qtype: formats.canonical_fully_quantized_layout(qtype)
            for qtype in expected
        }
        if observed != expected:
            raise AssertionError(f"canonical layout table differs: {observed}")
        if any(formats.archived_fully_quantized_layouts(qtype) !=
               frozenset({"xplane"}) for qtype in expected):
            raise AssertionError("explicit development Xplane archive does not cover exactly five qtypes")

        for qtype in expected:
            k = 512 if qtype in (formats.QuantType.Q3_K,
                                 formats.QuantType.Q6_K) else 256
            formats.validate_fully_quantized_resident_geometry(qtype, 256, k)

        routes = (ROOT / "quactlize/routes.py").read_text()
        packer = (ROOT / "quactlize/pack_gguf.py").read_text()
        runner = (ROOT / "tools/run_nonq4_xplane_correctness_box.sh").read_text()
        placed = (ROOT / "quactlize/include/ppu_placed_arrangement.hpp").read_text()
        require(routes, (
            "layout = canonical_fully_quantized_layout(qtype)",
            "tile_k is an explicit Xplane compatibility setting",
            "validate_fully_quantized_resident_geometry(qtype, n, k)",
            "q4-kpack4 is defined only for Q4_K",
            "kquant-kpack has no artifact TileK axis",
        ), "route policy")
        require(packer, (
            "return F.canonical_fully_quantized_layout",
            "F.validate_fully_quantized_resident_geometry(qtype, n, k)",
            '"layout_policy": "canonical"',
        ), "whole-model packer")
        if "--layout-policy" in packer or "all-kpack" in packer:
            raise AssertionError("whole-model packer still exposes a redundant layout policy")
        require(runner, (
            'QUACTLIZE_PPU_LIB="$base_so"',
            '"QUACTLIZE_PPU_LIB_FMT${fmt}=$format_so"',
            "PPU_ARCHS=ppu0010",
            '"$sdk_root/bin/hgobjdump" -lelf "$so"',
        ), "non-Q4 device runner")
        if 'QUACTLIZE_PPU_LIB="$format_so"' in runner:
            raise AssertionError(
                "non-Q4 runner merged the format reader into the independent base/oracle arm")
        require(placed, (
            "return qtype == 12 && arrangement->bits == 4",
            "arrangement->mapping_id == q4_kpack4::kMappingId",
            "kquant_kpack_transpose_v1(qtype)",
            "QUACTLIZE_PPU_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1",
        ), "device descriptor predicate")

        subprocess.run([
            sys.executable, "-m", "pytest", "-q",
            "tests/test_placed_artifact_abi.py",
            "-k", "canonical_offline_layout_policy or whole_model_packer",
        ], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        print("[kquant-offline-policy] PASS canonical=Q4/K-pack4 + "
              "Q2/Q3/Q5/Q6/per-plane-K-pack; Xplane explicit-development-only; "
              "resident N/K boundary shared by routes and packer")
        return 0
    except (AssertionError, OSError, subprocess.CalledProcessError) as error:
        print(f"[kquant-offline-policy] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
