#!/usr/bin/env python3
"""Source and host contracts for the ScaleFirst K-pack4 persistent A/B."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "tools" / "select_scalefirst_q4k_kpack4_prefill_ab.py"
ANALYZER = ROOT / "tools" / "analyze_scalefirst_q4k_kpack4_prefill_ab.py"
RUNNER = ROOT / "tools" / "run_scalefirst_q4k_kpack4_prefill_ab_box.sh"
BENCH = ROOT / "benchmarks" / "scalefirst_internal_sweep_bench.hpp"
DRIVER = ROOT / "benchmarks" / "test_scalefirst_internal_sweep.cu"
CMAKE = ROOT / "quactlize" / "csrc" / "scalefirst_internal_sweep.cmake.in"
BUILD = ROOT / "build.sh"


def require(text: str, tokens: tuple[str, ...], label: str) -> None:
    missing = [token for token in tokens if token not in text]
    if missing:
        raise AssertionError(f"{label} lacks {missing}")


def main() -> int:
    try:
        bench = BENCH.read_text()
        driver = DRIVER.read_text()
        cmake = CMAKE.read_text()
        build = BUILD.read_text()
        runner = RUNNER.read_text()
        require(bench, (
            "ScaleFirstShippingSelector<true",
            "DenseQ4KPack4KernelTypes<",
            "WeightLayout = SCALEFIRST_SWEEP_WEIGHT_LAYOUT",
            "MainloopDescriptor::q4_kpack4_transpose",
            "Mainloop::kQ4KPack4Transpose",
            "MainloopUsesPackedMetadata",
            "PersistentMixedInputKernel<",
        ), "ScaleFirst type seam")
        require(driver, (
            '#include "ppu_placed_arrangement.hpp"',
            "#if SCALEFIRST_SWEEP_WEIGHT_LAYOUT == 1",
            "q4_kpack4::prepare(", "q4_kpack4::recover(",
            "quactlize_ppu_prepare_dense_for_arrangement_v2(",
            "quactlize_ppu_recover_dense_for_arrangement_v2(",
            '"weight_layout=%d weight_mapping_id=0x%016llx "',
        ), "fixture/identity seam")
        require(cmake, (
            "SCALEFIRST_SWEEP_WEIGHT_LAYOUT",
            '"-DPPU_PACKED_SCALE=0"',
            "K-pack4 requires q=12 A=0 bc=0",
        ), "CMake seam")
        require(build, ("SCALEFIRST_SWEEP_WEIGHT_LAYOUT",), "build seam")
        require(runner, (
            "2048\t1024\t5120", "4096\t1024\t5120",
            "--algorithm=persistent", "--fixture=exact",
            "SCALEFIRST_SWEEP_WEIGHT_LAYOUT=\"$layout\"",
            "-DPPU_PACKED_SCALE=0",
            "metadata=ScaleFirst-FP16 algorithm=PERSISTENT",
        ), "runner seam")
        if "--algorithm=split" in runner or "--algorithm=all" in runner or \
                "64\t1024\t5120" in runner:
            raise AssertionError("runner admitted M64 or Split-K")
        subprocess.run([sys.executable, "-B", str(SELECTOR), "self-test"],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, "-B", str(ANALYZER), "self-test"],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        with tempfile.TemporaryDirectory(prefix="qz-sf-kpack4-ci-") as temp:
            output = pathlib.Path(temp) / "generated"
            subprocess.run([sys.executable, "-B", str(SELECTOR),
                            "materialize", "--out-dir", str(output)],
                           cwd=ROOT, check=True,
                           stdout=subprocess.DEVNULL)
            for arm, artifact in (("xplane", 64), ("q4-kpack4", 0)):
                registry = (output / arm / "scalefirst_registry.inc").read_text()
                unit = next((output / arm / "units").glob("*.cu")).read_text()
                require(registry, (
                    f"#define SCALEFIRST_GENERATED_ARTIFACT_TK {artifact}",
                    "#define SCALEFIRST_GENERATED_TYPED_ROWS 3",
                ), f"{arm} registry")
                require(unit, (
                    "#define PPU_PACKED_SCALE 0",
                    "#define PPU_B_CHUNK 0",
                    '#include "scalefirst_internal_sweep_unit.inc"',
                ), f"{arm} unit")
        print("[sf-kpack4-prefill-ab:self-test] PASS M=2048/4096 only; "
              "same FP16 metadata, persistent driver and three tactics; "
              "layout/build/fixture authority bound")
        return 0
    except (AssertionError, OSError, subprocess.CalledProcessError) as error:
        print(f"[sf-kpack4-prefill-ab] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
