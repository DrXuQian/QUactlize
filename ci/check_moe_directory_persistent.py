#!/usr/bin/env python3
"""Fail-closed source contract for the DeepGEMM-style grouped scheduler port.

The port is allowed to replace only the work-tile driver.  The resident
artifact, mixed-input collective, and epilogue remain the existing quactlize
types; the historical non-persistent kernel remains the default control.
"""

from __future__ import annotations

import pathlib
import re
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
        "matrix_runner": ROOT / "tools/run_moe_directory_multiformat_ab_box.sh",
        "format_registry": ROOT / "quactlize/include/ppu_format_config.inc",
        "shipping_type": ROOT / "dev/fold_derivation/l216_moe_directory_shipping_type.cu",
        "shipping_type_runner": ROOT / "dev/fold_derivation/run_l216_moe_directory_shipping_type.sh",
    }
    return {name: path.read_text(encoding="utf-8") for name, path in paths.items()}


def format_contract_violations(src: dict[str, str]) -> list[str]:
    """Cross-check every shell/type projection against the one format registry."""
    out: list[str] = []
    registry_re = re.compile(
        r'^\s*X\(\s*([A-Za-z0-9_]+)\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,'
        r'\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)',
        re.M)
    registry = {}
    for match in registry_re.finditer(src["format_registry"]):
        ident, name, qtype, low, high, group, sf_tk, fq_tk, packed = match.groups()
        registry[int(qtype)] = {
            "ident": ident, "name": name, "low": int(low), "high": int(high),
            "group": int(group), "sf_tk": int(sf_tk), "fq_tk": int(fq_tk),
            "packed": int(packed),
        }
    if set(registry) != {10, 11, 12, 13, 14}:
        return [f"format registry denominator is {sorted(registry)}, expected qtypes 10..14"]

    short = {10: "i2", 11: "q3", 12: "i4", 13: "q5", 14: "q6"}
    qmode = {10: 0, 11: 1, 12: 0, 13: 0, 14: 1}
    mode_name = {0: "ScaleZero", 1: "ScaleOnly"}

    case_re = re.compile(
        r"^  (i2|q3|i4|q5|q6)\)\n"
        r"\s+QTYPE=(\d+); FORMAT_NAME=([A-Za-z0-9_]+); GROUP_SIZE=(\d+); ARTIFACT_TILEK=(\d+)\n"
        r"\s+LOW_BITS=(\d+); HIGH_BITS=(\d+); PLANES=(\d+)\n"
        r"\s+DEFAULT_ROW='([^']+)'", re.M)
    cases = {}
    for match in case_re.finditer(src["runner"]):
        tag, qtype, name, group, sf_tk, low, high, planes, row = match.groups()
        cases[int(qtype)] = (tag, name, int(group), int(sf_tk), int(low), int(high), int(planes), row)
    if set(cases) != set(registry):
        out.append(f"base runner format denominator is {sorted(cases)}, expected qtypes 10..14")
    for qtype, fmt in registry.items():
        wn = 64 if qtype in (11, 13) else 32
        want = (short[qtype], fmt["name"], fmt["group"], fmt["sf_tk"],
                fmt["low"], fmt["high"], 2 if fmt["high"] else 1,
                f'{short[qtype]} 64x64:{fmt["sf_tk"]} w64x{wn} s3 bc0->0')
        if cases.get(qtype) != want:
            out.append(f"base runner projection for qtype {qtype} differs from registry/tactic contract")

    matrix_re = re.compile(
        r"^(Q[2-6]_K)\|(i2|q3|i4|q5|q6)\|(\d+)\|([01])\|"
        r"(ScaleOnly|ScaleZero)\|(\d+)\|(\d+\+\d+)\|(\d+)\|(\d+)\|(.+)$", re.M)
    matrix = {}
    for match in matrix_re.finditer(src["matrix_runner"]):
        name, tag, qtype, qm, quant, group, bits, planes, sf_tk, row = match.groups()
        matrix[int(qtype)] = (name, tag, int(qm), quant, int(group), bits,
                              int(planes), int(sf_tk), row)
    if set(matrix) != set(registry):
        out.append(f"multiformat A/B denominator is {sorted(matrix)}, expected qtypes 10..14")
    for qtype, fmt in registry.items():
        wn = 64 if qtype in (11, 13) else 32
        want = (fmt["name"], short[qtype], qmode[qtype], mode_name[qmode[qtype]],
                fmt["group"], f'{fmt["low"]}+{fmt["high"]}',
                2 if fmt["high"] else 1, fmt["sf_tk"],
                f'{short[qtype]} 64x64:{fmt["sf_tk"]} w64x{wn} s3 bc0->0')
        if matrix.get(qtype) != want:
            out.append(f"multiformat A/B row for qtype {qtype} differs from real ScaleFirst semantics")

    shipping = src["shipping_type"]
    shipping_runner = src["shipping_type_runner"]
    for qtype in registry:
        if f"struct L216Format<{qtype}>" not in shipping:
            out.append(f"L216 lacks qtype {qtype} type projection")
        if f"{qtype}:{registry[qtype]['name']}" not in shipping_runner:
            out.append(f"L216 runner lacks an isolated qtype {qtype} compiler arm")
    for qtype, wanted in qmode.items():
        block = re.search(rf"struct L216Format<{qtype}> \{{(.*?)\n\}};", shipping, re.S)
        mode = "FinegrainedScaleOnly" if wanted else "FinegrainedScaleZero"
        if block is None or mode not in block.group(1):
            out.append(f"L216 qtype {qtype} does not bind semantic mode {mode}")
    return out


def violations(src: dict[str, str]) -> list[str]:
    out: list[str] = []
    launcher = src["launcher"]
    kernel = src["kernel"]
    directory = src["directory"]
    backend = src["backend"]
    benchmark = src["benchmark"]
    harness = src["harness"]
    runner = src["runner"]
    matrix_runner = src["matrix_runner"]
    shipping_type = src["shipping_type"]

    requirements = {
        "default remains non-persistent":
            "inline constexpr bool kPersistentBuild = false;" in launcher,
        "old GemmUniversal kernel remains an explicit control":
            "using NonPersistentKernel" in launcher and
            "GemmUniversal<GroupProblemShape" in launcher,
        "generated filter selects only through the build switch":
            "ArtifactTileK, kPersistentBuild, void, BChunk>" in launcher,
        "device ABI selects through the same build switch":
            "QueryOnly, RequireUniversalFallback, ArtifactTileK,\n"
            "                          moe_grouped_ppu::kPersistentBuild," in backend,
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
        "one-row runner routes format and quant mode into the real build":
            'MOE_FORMATS="$FORMAT"' in runner and
            'PPU_DEFS="LOWBIT_QMODE=$QMODE PPU_MOE_PERSISTENT=$enabled"' in runner and
            'quant=$QMODE_NAME' in runner,
        "multiformat runner remains explicitly ScaleFirst":
            "scope=ScaleFirst" in matrix_runner and
            "NOT a FullyQuantized/packed-metadata benchmark" in matrix_runner and
            "fp16-${quant}" in matrix_runner,
        "multiformat runner requires one raw-bit-gated A/B per format":
            "each A/B is full-D raw-bit gated" in matrix_runner and
            'bash "$ROOT/tools/run_moe_directory_persistent_ab_box.sh"' in matrix_runner,
        "shipping type binds registry-owned ScaleFirst identity":
            "ppu_formats::for_qtype(L216_QTYPE)" in shipping_type and
            "kRegistry.scale_first_tile_k" in shipping_type and
            "!L216Policy::Descriptor::packed_metadata" in shipping_type,
    }
    out.extend(name for name, ok in requirements.items() if not ok)
    out.extend(format_contract_violations(src))

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
        ("drop build switch", "ArtifactTileK, kPersistentBuild, void, BChunk>",
         "ArtifactTileK, false, void, BChunk>"),
        ("change record ABI", "sizeof(BlockEntry) == 16", "sizeof(BlockEntry) == 32"),
        ("drop scheduler identity", "scheduler=%s timing=%s", "timing=%s"),
        ("collapse span label", "event-scheduler-span-upper-v1", "event-kernel-span-upper-v1"),
        ("drop source bundle", "tracked-worktree.patch", "tracked-worktree.missing"),
        ("drop band-coverage sentinel", "MOE_TM_LIST='8;64'", "MOE_TM_LIST=64"),
        ("drop multiformat row", "Q6_K|q6|14|1|ScaleOnly", "Q6_X|q6|14|1|ScaleOnly"),
        ("drift multiformat layout", "Q4_K|i4|12|0|ScaleZero|32|4+0|1|64|",
         "Q4_K|i4|12|0|ScaleZero|32|4+0|1|128|"),
        ("drop shipping format", "struct L216Format<14>", "struct L216Format<15>"),
        ("move build after GEMM", "auto const run_status = gemm.run(stream);",
         "auto const run_status = gemm.run(stream);\n  // plant\n"),
        ("add format logic", "namespace cutlass::gemm::kernel {",
         "namespace cutlass::gemm::kernel {\n// xplane converter plant"),
    ]
    for name, old, new in plants:
        planted = dict(src)
        if "multiformat" in name:
            target = "matrix_runner"
        elif "shipping format" in name:
            target = "shipping_type"
        elif "format logic" in name:
            target = "kernel"
        elif "record" in name:
            target = "directory"
        elif "identity" in name:
            target = "benchmark"
        elif "span" in name:
            target = "harness"
        elif "bundle" in name or "band-coverage" in name:
            target = "runner"
        else:
            target = "launcher"
        if old not in planted[target]:
            failures.append(f"self-test plant seam missing: {name}")
            continue
        if name == "move build after GEMM":
            # Invert the two calls, rather than merely adding a comment.
            text = planted[target]
            build_pos = text.find("  if constexpr (UsePersistent) {", text.find("THE MEASURED INTERVAL"))
            run_line = "  auto const run_status = gemm.run(stream);\n"
            run_pos = text.find(run_line, build_pos)
            if build_pos < 0 or run_pos < 0:
                failures.append(f"self-test plant seam missing: {name}")
                continue
            block = text[build_pos:run_pos]
            planted[target] = (text[:build_pos] + text[run_pos:run_pos + len(run_line)] +
                               block + text[run_pos + len(run_line):])
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
          "16-byte directory, scheduler/source-bound identity, dual-band build coverage, five ScaleFirst "
          "format/layout projections, measured build order, and eleven negative plants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
