#!/usr/bin/env python3
"""Source contracts for the 15-shape ScaleFirst K-pack4 prefill A/B."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "tools" / "select_scalefirst_q4k_kpack4_prefill_real_shapes.py"
ANALYZER = ROOT / "tools" / "analyze_scalefirst_q4k_kpack4_prefill_real_shapes.py"
RUNNER = ROOT / "tools" / "run_scalefirst_q4k_kpack4_prefill_real_shapes_box.sh"


def require(text: str, tokens: tuple[str, ...], label: str) -> None:
    missing = [token for token in tokens if token not in text]
    if missing:
        raise AssertionError(f"{label} lacks {missing}")


def main() -> int:
    try:
        runner = RUNNER.read_text()
        require(runner, (
            "INTERNAL_SWEEP_SPEC must name the COMPLETE inventory-v2 JSON",
            "quactlize.scalefirst_q4k_real_shapes_plan.v1",
            "shape_count')!=15",
            "--algorithm=persistent", "--fixture=exact",
            "SCALEFIRST_SWEEP_WEIGHT_LAYOUT=\"$layout\"",
            "-DPPU_PACKED_SCALE=0", "rows=7", "PERF_ROUNDS:-2",
            "order='xplane q4-kpack4'", "order='q4-kpack4 xplane'",
        ), "real-shape runner")
        for forbidden in ("--algorithm=split", "--algorithm=all",
                          "--fixture=transport-only"):
            if forbidden in runner:
                raise AssertionError(f"runner admitted {forbidden}")
        subprocess.run([sys.executable, "-B", str(SELECTOR), "self-test"],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, "-B", str(ANALYZER), "self-test"],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        with tempfile.TemporaryDirectory(prefix="qz-sf-kpack4-real-ci-") as temp:
            output = pathlib.Path(temp) / "generated"
            subprocess.run([sys.executable, "-B", str(SELECTOR),
                            "materialize", "--out-dir", str(output)],
                           cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            value = json.loads((output / "manifest.json").read_text())
            if value.get("schema") != \
                    "quactlize.scalefirst-q4k-kpack4-prefill-real.v2" or \
                    len(value.get("shapes", [])) != 15 or \
                    len(value.get("candidates", [])) != 7:
                raise AssertionError("real-shape manifest denominator differs")
            expected_m = {64, 2048, 4096}
            expected_families = {
                (1024, 5120), (5120, 8192), (5120, 25600),
                (8192, 5120), (25600, 5120),
            }
            observed = {tuple(map(int, row)) for row in value["shapes"]}
            if {m for m, _, _ in observed} != expected_m or \
                    {(n, k) for _, n, k in observed} != expected_families:
                raise AssertionError("real-shape axes differ")
            for arm, artifact, layout in (
                    ("xplane", 64, 0), ("q4-kpack4", 0, 1)):
                manifest = json.loads((output / arm / "manifest.json").read_text())
                registry = (output / arm / "scalefirst_registry.inc").read_text()
                if manifest.get("artifact_tile_k") != artifact or \
                        manifest.get("weight_layout") != layout or \
                        manifest.get("algorithms") != ["PERSISTENT"] or \
                        len(manifest.get("typed_rows", [])) != 7:
                    raise AssertionError(f"{arm} manifest identity differs")
                require(registry, (
                    f"#define SCALEFIRST_GENERATED_ARTIFACT_TK {artifact}",
                    "#define SCALEFIRST_GENERATED_TYPED_ROWS 7",
                ), f"{arm} registry")
        print("[sf-kpack4-prefill-real:self-test] PASS inventory-owned "
              "15-shape denominator; paired persistent exact A/B, seven "
              "matched tactics and alternating-order authority bound")
        return 0
    except (AssertionError, OSError, subprocess.CalledProcessError) as error:
        print(f"[sf-kpack4-prefill-real] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
