#!/usr/bin/env python3
"""Guard the Q4 K-pack4 transition without retiring non-Q4 Xplane."""

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
            formats.QuantType.Q2_K: "xplane",
            formats.QuantType.Q3_K: "xplane",
            formats.QuantType.Q4_K: "q4-kpack4",
            formats.QuantType.Q5_K: "xplane",
            formats.QuantType.Q6_K: "xplane",
        }
        observed = {
            qtype: formats.canonical_fully_quantized_layout(qtype)
            for qtype in expected
        }
        if observed != expected:
            raise AssertionError(f"canonical layout table differs: {observed}")
        if formats.archived_fully_quantized_layouts(formats.QuantType.Q4_K) != \
                frozenset({"xplane"}) or any(
                    formats.archived_fully_quantized_layouts(qtype)
                    for qtype in expected if qtype != formats.QuantType.Q4_K):
            raise AssertionError("archive scope escaped Q4 Xplane")

        routes = (ROOT / "quactlize/routes.py").read_text()
        packer = (ROOT / "tools/pack_gguf.py").read_text()
        runner = (ROOT / "tools/run_nonq4_xplane_correctness_box.sh").read_text()
        placed = (ROOT / "quactlize/include/ppu_placed_arrangement.hpp").read_text()
        require(routes, (
            "layout = (canonical_fully_quantized_layout(qtype)",
            "layout = canonical_fully_quantized_layout(qtype)",
            "q4-kpack4 is defined only for Q4_K",
        ), "route policy")
        require(packer, (
            "F.canonical_fully_quantized_layout(int(t.tensor_type))",
            "layout = F.canonical_fully_quantized_layout(qtype)",
        ), "whole-model packer")
        if "--q4-layout" in packer:
            raise AssertionError("whole-model packer still exposes archived Q4 Xplane")
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
        ), "device descriptor predicate")

        subprocess.run([
            sys.executable, "-m", "pytest", "-q",
            "tests/test_placed_artifact_abi.py",
            "-k", "canonical_offline_layout_policy or whole_model_packer",
        ], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        print("[kquant-offline-policy] PASS canonical=Q4/K-pack4; "
              "Q2/Q3/Q5/Q6=Xplane; Q4/Xplane archived from automatic "
              "and whole-model selection only")
        return 0
    except (AssertionError, OSError, subprocess.CalledProcessError) as error:
        print(f"[kquant-offline-policy] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
