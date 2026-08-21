#!/usr/bin/env python3
"""Fail-closed source contract for the DeepGEMM-style grouped scheduler port.

The port is allowed to replace only the work-tile driver.  The resident
artifact, mixed-input collective, and epilogue remain the existing quactlize
types; the historical non-persistent kernel remains the default control.
"""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load() -> dict[str, str]:
    paths = {
        "launcher": ROOT / "quactlize/include/moe_grouped_ppu.cuh",
        "kernel": ROOT / (
            "quactlize/include/actlize_extensions/cutlass/gemm/kernel/"
            "ppu_aiu_gemm_mixed_input_group_persistent.hpp"
        ),
        "directory": ROOT / (
            "quactlize/include/actlize_extensions/cutlass/gemm/kernel/"
            "ppu_moe_block_directory.hpp"
        ),
        "backend": ROOT / "quactlize/csrc/device/ppu_dense_backend.cu",
        "benchmark": ROOT / "benchmarks/test_lowbit_moe_bench.cu",
        "harness": ROOT / "benchmarks/lowbit_moe_bench.hpp",
        "runner": ROOT / "tools/run_moe_directory_persistent_ab_box.sh",
    }
    return {name: path.read_text(encoding="utf-8") for name, path in paths.items()}


def violations(src: dict[str, str]) -> list[str]:
    out: list[str] = []
    launcher = src["launcher"]
    kernel = src["kernel"]
    directory = src["directory"]
    backend = src["backend"]
    benchmark = src["benchmark"]
    harness = src["harness"]
    runner = src["runner"]

    requirements = {
        "default remains non-persistent":
            "inline constexpr bool kPersistentBuild = false;" in launcher,
        "old GemmUniversal kernel remains an explicit control":
            "using NonPersistentKernel" in launcher and
            "GemmUniversal<GroupProblemShape" in launcher,
        "generated filter selects only through the build switch":
            "ArtifactTileK, kPersistentBuild>" in launcher,
        "device ABI selects through the same build switch":
            "moe_grouped_ppu::kPersistentBuild>(" in backend,
        "persistent kernel reuses the existing mainloop type":
            "using CollectiveMainloop = CollectiveMainloop_;" in kernel and
            "collective_mainloop.load_init" in kernel,
        "persistent kernel reuses the existing epilogue type":
            "using CollectiveEpilogue = CollectiveEpilogue_;" in kernel and
            "CollectiveEpilogue epilogue" in kernel,
        "directory record is exactly 16 bytes":
            "static_assert(sizeof(BlockEntry) == 16" in directory,
        "directory uses a persistent grid-stride walk":
            "current_iter_++" in directory and "gridDim.x" in directory and
            "blockIdx.x" in directory,
        "workspace ABI explicitly covers both schedulers":
            "std::max(legacy_scheduler_bytes, persistent_directory_bytes)" in backend,
        "benchmark run identity separates scheduler arms":
            "scheduler=%s timing=%s" in benchmark and
            '"persistent-directory" : "non-persistent"' in harness and
            "moe_scheduler_name()" in benchmark,
        "persistent sample label includes directory construction":
            '"event-scheduler-span-upper-v1"' in harness and
            '"scheduler-span-upper" : "kernel-span-upper"' in harness and
            "s.timing = moe_timing_identity();" in harness and
            "moe_span_label()" in benchmark,
        "dirty-tree box provenance is reconstructable":
            "tracked-worktree.patch" in runner and
            "untracked-files.tar.gz" in runner and
            "source-bundle.sha256" in runner and
            "binary.sha256" in runner,
        "box restriction preserves the smallest legal decode sentinel":
            "MOE_TM_LIST='8;64'" in runner and
            "MOE_TN_LIST=64" in runner and
            "MOE_WM_LIST='8;64'" in runner and
            "MOE_STAGES=3" in runner,
    }
    out.extend(name for name, ok in requirements.items() if not ok)

    # Directory construction is a measured scheduler cost.  This order check
    # prevents a later refactor from moving it outside the event span while
    # continuing to label the number end-to-end.
    start = launcher.find("hggcEventRecord(kernel_span->start")
    build = launcher.find("moe_directory::launch_build", start)
    run = launcher.find("gemm.run(stream)", build)
    stop = launcher.find("hggcEventRecord(kernel_span->stop", run)
    if min(start, build, run, stop) < 0 or not start < build < run < stop:
        out.append("measured order must be start -> directory build -> GEMM -> stop")

    # A scheduler file growing a format converter is the exact architectural
    # regression this port avoids.  Names of C++ template elements are fine;
    # representation-specific machinery is not.
    lower = kernel.lower()
    for forbidden in ("gguf", "dequant", "xplane", "packed_scale", "converter"):
        if forbidden in lower:
            out.append(f"scheduler kernel acquired forbidden representation logic: {forbidden}")
    return out


def self_test(src: dict[str, str]) -> list[str]:
    failures: list[str] = []
    plants = [
        ("drop build switch", "ArtifactTileK, kPersistentBuild>", "ArtifactTileK>"),
        ("change record ABI", "sizeof(BlockEntry) == 16", "sizeof(BlockEntry) == 32"),
        ("drop scheduler identity", "scheduler=%s timing=%s", "timing=%s"),
        ("collapse span label", "event-scheduler-span-upper-v1", "event-kernel-span-upper-v1"),
        ("drop source bundle", "tracked-worktree.patch", "tracked-worktree.missing"),
        ("drop band-coverage sentinel", "MOE_TM_LIST='8;64'", "MOE_TM_LIST=64"),
        ("move build after GEMM", "gemm.run(stream);", "gemm.run(stream);\n  // plant\n"),
        ("add format logic", "namespace cutlass::gemm::kernel {",
         "namespace cutlass::gemm::kernel {\n// xplane converter plant"),
    ]
    for name, old, new in plants:
        planted = dict(src)
        target = "kernel" if "format" in name else (
            "directory" if "record" in name else (
                "benchmark" if "identity" in name else (
                    "harness" if "span" in name else (
                        "runner" if "bundle" in name or "band-coverage" in name else "launcher"))))
        if old not in planted[target]:
            failures.append(f"self-test plant seam missing: {name}")
            continue
        if name == "move build after GEMM":
            # Invert the two calls, rather than merely adding a comment.
            text = planted[target]
            build_pos = text.find("  if constexpr (UsePersistent) {", text.find("THE MEASURED INTERVAL"))
            run_pos = text.find("  gemm.run(stream);", build_pos)
            if build_pos < 0 or run_pos < 0:
                failures.append(f"self-test plant seam missing: {name}")
                continue
            block = text[build_pos:run_pos]
            planted[target] = text[:build_pos] + text[run_pos:run_pos + len("  gemm.run(stream);\n")] + block + text[run_pos + len("  gemm.run(stream);\n"):]
        elif name == "drop source bundle":
            planted[target] = planted[target].replace(old, new)
        else:
            planted[target] = planted[target].replace(old, new, 1)
        if not violations(planted):
            failures.append(f"negative plant was incorrectly green: {name}")
    return failures


def main() -> int:
    src = load()
    bad = violations(src)
    bad.extend(self_test(src))
    if bad:
        for item in bad:
            print(f"[moe-directory-contract] FAIL: {item}", file=sys.stderr)
        return 1
    print("[moe-directory-contract] PASS: driver-only port, common collective/epilogue, "
          "16-byte directory, scheduler/source-bound identity, dual-band build coverage, measured build order, "
          "and eight negative plants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
