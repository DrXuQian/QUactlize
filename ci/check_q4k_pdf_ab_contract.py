#!/usr/bin/env python3
"""Fail-closed contract for the INBOX 132B reconstruction/timing evidence."""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
HARNESS = ROOT / "benchmarks" / "q4k_pdf_5090_ab.cu"
KERNEL = ROOT / "benchmarks" / "q4k_pdf_reconstruction.cuh"
FIXTURE = ROOT / "benchmarks" / "q4k_pdf_ab_fixture.hpp"
RUNNER = ROOT / "benchmarks" / "q4k_pdf_5090_ab.py"


def audit(harness: str, kernel: str, fixture: str, runner: str) -> list[str]:
    errors = []
    need_kernel = [
        "this is not an upstream/exact source file", "PairMetadata",
        '"r"(top_magic)', "dim3 const grid(m, n / cols_per_cta, 1)",
        "q4k_gemv_kernel<CTA_N, WARPS_N, WARPS_K, PairMetadata>",
    ]
    for token in need_kernel:
        if token not in kernel:
            errors.append(f"reconstruction lost source/variant/launcher seam: {token}")
    for row in [
        '{"D-EXT-O", 1, 5120, 8192, 2, 8, 1, true}',
        '{"D-EXT-K1024", 1, 5120, 1024, 2, 8, 1, true}',
        '{"D-EXT-Q", 1, 8192, 5120, 4, 8, 1, true}',
        '{"H-G8-2048", 8, 2048, 2048, 2, 8, 1, false}',
    ]:
        if row not in fixture:
            errors.append(f"shape/config authority drift: {row}")
    for token in [
        "Independent decode from raw blocks", "verify_representation",
        "raw != affine", "p.scales[si]", "p.zeros[si]",
    ]:
        if token not in fixture:
            errors.append(f"independent raw/native anchor missing: {token}")
    for token in [
        "raw(upload_repeated(h.raw, count))", "low(upload_repeated(h.low, count))",
        "scales(upload_repeated(h.scales, count))", "zeros(upload_repeated(h.zeros, count))",
        "std::max(raw_bytes, ours_bytes)", "std::min(64, int(budget / maximum))",
        'sample.state == "weight_metadata_cold" ? b : 0',
        "event_ms_bits", "float_bits(sample.elapsed_ms)",
        "cudaEventQuery(sample.stop)", "nvmlDeviceGetClockInfo",
        "nvml_ok(nvmlSystemGetDriverVersion", "device_name,samples_requested,warmup,precondition_ms",
        'add("ours_native_grouped1"', 'add(s.l == 1 ? "ours_native_dense1" : "ours_native_dense8"',
        "correctness_gate(device, arms);", "max_conditioned", "1.f / 128.f",
    ]:
        if token not in harness:
            errors.append(f"timing/correctness contract missing: {token}")
    if harness.find("correctness_gate(device, arms);") > harness.find("measure_state(f, options"):
        errors.append("timing starts before correctness")
    for token in [
        "dirty tree", "binary_sha", "event_ms_bits", "infer_quantum",
        "raw CSV mixes protocol field", "len(group) != requested",
        'verdict = "UNRESOLVED"', "topology-inclusive 1-vs-8", "not an exact-paper reproduction",
    ]:
        if token not in runner:
            errors.append(f"runner/report fail-close missing: {token}")
    return errors


def main() -> int:
    texts = [p.read_text() for p in (HARNESS, KERNEL, FIXTURE, RUNNER)]
    errors = audit(*texts)
    if errors:
        print("FAIL:", errors[0])
        return 1

    plants = []
    def plant(file_index: int, old: str, new: str, label: str) -> None:
        modified = list(texts)
        if old not in modified[file_index]:
            raise RuntimeError(f"control target vanished: {label}")
        modified[file_index] = modified[file_index].replace(old, new, 1)
        if not audit(*modified):
            raise RuntimeError(f"planted fault escaped: {label}")
        plants.append(label)

    plant(0, "zeros(upload_repeated(h.zeros, count))", "zeros(upload_repeated(h.zeros, 1))",
          "cold S/Z copy count")
    plant(0, 'sample.state == "weight_metadata_cold" ? b : 0', "0",
          "cold distinct representation")
    plant(0, 'add("ours_native_grouped1"', 'add("ours_native_dense8_again"',
          "grouped-vs-dense topology label")
    plant(0, "float_bits(sample.elapsed_ms)", "0u", "raw binary32 event authority")
    plant(0, "nvml_ok(nvmlSystemGetDriverVersion", "nvml_ok(cudaDriverGetVersion",
          "actual driver identity")
    plant(1, "this is not an upstream/exact source file", "exact upstream source",
          "reconstruction source boundary")
    plant(2, "Independent decode from raw blocks", "Decode from the affine artifact",
          "independent raw golden")
    plant(3, 'verdict = "UNRESOLVED"', 'verdict = "TIE"', "timer-resolution fail-close")
    plant(3, "len(group) != requested", "len(group) < 0", "declared sample-count fail-close")

    nvcc = shutil.which("nvcc")
    if not nvcc:
        print(f"SKIP: {len(plants)} semantic controls passed; nvcc absent, full sm_120 build unverified")
        return 3
    with tempfile.TemporaryDirectory(prefix="q4k-pdf-ab-contract.") as td:
        out = pathlib.Path(td) / "q4k_pdf_ab"
        cmd = [
            nvcc, "-std=c++17", "-O3", "-arch=sm_120", "--expt-relaxed-constexpr",
            f"-I{ROOT / 'quactlize' / 'include'}",
            f"-I{ROOT / 'third_party' / 'cutlass' / 'include'}",
            f"-I{ROOT / 'third_party' / 'actlize' / 'include'}",
            str(HARNESS), "-lnvidia-ml", "-o", str(out),
        ]
        built = subprocess.run(cmd, text=True, capture_output=True)
        if built.returncode:
            log = (built.stdout + built.stderr).splitlines()
            unsupported = any("Unsupported gpu architecture" in line or "not found" in line for line in log)
            if unsupported:
                print(f"SKIP: {len(plants)} semantic controls passed; compiler cannot build sm_120/NVML")
                return 3
            print("FAIL: full reconstruction build failed")
            print("\n".join(log[-20:]))
            return 1
    print(f"PASS: {len(plants)} planted evidence faults red; full sm_120 reconstruction links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
