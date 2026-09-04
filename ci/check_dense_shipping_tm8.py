#!/usr/bin/env python3
"""Pin the shipped TM8 family, broad and exact censuses, and shape-selected default."""
from __future__ import annotations

import re
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "quactlize/include/ppu_dense_configs.inc"
POLICY = ROOT / "quactlize/include/ppu_dense_shipping_policy.hpp"
SPACE = ROOT / "quactlize/include/ppu_tactic_space.hpp"
BACKEND = ROOT / "quactlize/csrc/device/ppu_dense_backend.cu"
BROAD_ORACLE = ROOT / "dev/fold_derivation/l147_dense_shipping_tm8.cpp"
EXACT_ORACLE = ROOT / "dev/fold_derivation/l148_dense_shipping_tm8_compiled.cu"
EXACT_RUNNER = ROOT / "dev/fold_derivation/run_l148_dense_shipping_tm8_compiled.sh"
LAUNCHER = ROOT / "quactlize/include/fpA_intB_ppu.cuh"


def compile_oracle(extra_include: Path | None = None) -> tuple[int, str]:
    with tempfile.TemporaryDirectory(prefix="quactlize-l147-build-") as td:
        binary = Path(td) / "l147"
        includes = [f"-I{extra_include}"] if extra_include else []
        cmd = ["c++", "-std=c++17", *includes, f"-I{ROOT / 'quactlize/include'}",
               str(BROAD_ORACLE), "-o", str(binary)]
        built = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
        if built.returncode:
            return built.returncode, built.stdout
        run = subprocess.run([str(binary)], cwd=ROOT, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        return run.returncode, run.stdout


def source_errors(backend: str) -> list[str]:
    bad: list[str] = []
    required = (
        "DenseSpace::kernel_exclusion(kTactic)",
        "kProducerExclusion",
        "ID == kDefaultDenseConfig ||",
        "ID == kDecodeDefaultDenseConfig",
        "(DenseConfigId::ID == kDecodeDefaultDenseConfig), QueryOnly",
        "find_dense_config(char const* name, int m, DenseConfigId& config)",
        "ppu_dense_shipping::find_config(name, m, config)",
        "resolve_dense_config(char const* name, int m)",
        "resolve_dense_config(config_name, m)",
        "find_dense_config(config_name, m, config)",
    )
    for token in required:
        if token not in backend:
            bad.append("backend missing " + token)
    if "DenseSpace::topology_exclusion(kTactic, Stages)" in backend:
        bad.append("backend uses the approximate host smem model instead of compiled SharedStorageSize")
    if backend.count("resolve_dense_config(config_name, m)") != 3:
        bad.append("all three config-resolving dense launch ABIs must use the M-aware policy")
    if backend.count("find_dense_config(config_name, m, config)") != 5:
        bad.append("all three validity and two arrangement ABIs must use the M-aware policy")
    return bad


def exact_source_errors(launcher: str, oracle: str) -> list[str]:
    bad: list[str] = []
    required_launcher = (
        "struct DenseKernelTypes",
        "using KernelTypes = DenseKernelTypes<",
        "using GemmKernel = typename KernelTypes::GemmKernel;",
        "ppu_tactics::fits_block_smem(",
    )
    required_oracle = (
        "using Kernel = fpa_intb_ppu::DenseKernelTypes<",
        "shared_bytes = Kernel::SharedStorageSize;",
        "ppu_tactics::fits_block_smem(Kernel::SharedStorageSize)",
        "Kernel::CollectiveMainloop::is_packed_scale",
    )
    for token in required_launcher:
        if token not in launcher:
            bad.append("launcher missing exact authority seam " + token)
    for token in required_oracle:
        if token not in oracle:
            bad.append("exact oracle missing production authority seam " + token)
    return bad


def run_exact_oracle(out: Path) -> tuple[int, str]:
    env = os.environ.copy()
    env["QUACTLIZE_L148_OUT"] = str(out)
    run = subprocess.run(["bash", str(EXACT_RUNNER)], cwd=ROOT, env=env, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
    return run.returncode, run.stdout


def cell_key(line: str) -> tuple[str, str, str]:
    fields = dict(re.findall(r"([a-z_]+)=([^ ]+)", line))
    return fields["format"], fields["mode"], fields["config"]


def main() -> int:
    paths = (INVENTORY, POLICY, SPACE, BACKEND, BROAD_ORACLE, EXACT_ORACLE, EXACT_RUNNER, LAUNCHER)
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    if missing:
        print("[dense-shipping-tm8] FAIL missing: " + ", ".join(missing))
        return 1

    bad = source_errors(BACKEND.read_text())
    bad += exact_source_errors(LAUNCHER.read_text(), EXACT_ORACLE.read_text())
    if bad:
        print("[dense-shipping-tm8] FAIL: " + "; ".join(bad))
        return 1

    rc, output = compile_oracle()
    summary = "broad-summary family_rows=6 cells=60 legal=52 illegal=8 "
    rows = [line for line in output.splitlines() if line.startswith("broad-cell ")]
    rejected = [line for line in rows if "verdict=ILLEGAL" in line]
    if rc or len(rows) != 60 or len(rejected) != 8 or summary not in output:
        print("[dense-shipping-tm8] FAIL broad oracle:\n" + output[-6000:])
        return 1
    if any("physical_a_rows=16" not in row for row in rows):
        print("[dense-shipping-tm8] FAIL: at least one logical TM8 cell stopped paying for 16 A rows")
        return 1
    expected_reasons = {"reason=" + "the conservative gs16 scale+zero footprint exceeds the 256KB block limit"}
    found_reasons = {row.split(" reason=", 1)[1] for row in rejected}
    if found_reasons != {next(iter(expected_reasons)).removeprefix("reason=")}:
        print("[dense-shipping-tm8] FAIL unexpected rejection reasons: " + repr(found_reasons))
        return 1

    broad_illegal = {cell_key(row) for row in rejected}
    with tempfile.TemporaryDirectory(prefix="quactlize-l148-build-") as td:
        exact_rc, exact_output = run_exact_oracle(Path(td))
    exact_rows = [line for line in exact_output.splitlines() if line.startswith("compiled format=")]
    exact_rejected = [line for line in exact_rows if " verdict=ILLEGAL" in line]
    expected_exact_illegal = {
        ("Q2_K", "fully-quantized", "8x128:8x32:s12"),
        ("Q3_K", "scale-first", "8x128:8x32:s12"),
        ("Q3_K", "fully-quantized", "8x128:8x32:s12"),
        ("Q4_K", "fully-quantized", "8x128:8x32:s12"),
        ("Q5_K", "scale-first", "8x128:8x32:s8"),
        ("Q5_K", "scale-first", "8x128:8x32:s12"),
        ("Q5_K", "fully-quantized", "8x128:8x32:s8"),
        ("Q5_K", "fully-quantized", "8x128:8x32:s12"),
        ("Q6_K", "fully-quantized", "8x128:8x32:s12"),
    }
    exact_illegal = {cell_key(row) for row in exact_rejected}
    if (exact_rc or len(exact_rows) != 60 or len(exact_rejected) != 9 or
            exact_illegal != expected_exact_illegal):
        print("[dense-shipping-tm8] FAIL exact compiled oracle:\n" + exact_output[-12000:])
        print("expected exact illegal:", sorted(expected_exact_illegal))
        print("actual exact illegal:", sorted(exact_illegal))
        return 1
    if exact_illegal - broad_illegal != {("Q6_K", "fully-quantized", "8x128:8x32:s12")}:
        print("[dense-shipping-tm8] FAIL: broad/exact delta stopped isolating packed Q6 s12")
        return 1
    if not any("format=Q5_K mode=scale-first" in row and "config=8x128:8x32:s8" in row and
               "shared_bytes=262160" in row for row in exact_rejected):
        print("[dense-shipping-tm8] FAIL: 16-byte zero-member boundary witness disappeared")
        return 1

    # Structural negative controls compile the SAME oracle against a mutated first-authority header. They must fail
    # through the predeclared assertions; a token-only checker would let both changes escape.
    plants = (
        (INVENTORY, "  X(ShortWideM8S12, \"8x128:8x32:s12\",   8, 128,  8, 32, 12)\n", ""),
        (POLICY, "inline constexpr int kDecodeDefaultExclusiveM = 8;",
                 "inline constexpr int kDecodeDefaultExclusiveM = 0;"),
        (SPACE, "return c.tm < 16 ? 16 : c.tm;", "return c.tm;"),
    )
    for source, old, new in plants:
        text = source.read_text()
        if text.count(old) != 1:
            print(f"[dense-shipping-tm8] FAIL cannot plant {source.name}")
            return 1
        with tempfile.TemporaryDirectory(prefix="quactlize-l147-plant-") as td:
            td = Path(td)
            # Quote includes resolve beside the copied policy, so copy both policy and inventory for every plant.
            (td / INVENTORY.name).write_text(
                text.replace(old, new) if source == INVENTORY else INVENTORY.read_text())
            (td / POLICY.name).write_text(
                text.replace(old, new) if source == POLICY else POLICY.read_text())
            if source == SPACE:
                (td / SPACE.name).write_text(text.replace(old, new))
            rc, log = compile_oracle(td)
            if rc == 0:
                print(f"[dense-shipping-tm8] FAIL {source.name} plant escaped")
                return 1

    # Source plants prove that the device adapter really consumes the host-proved policy and stage-aware guard.
    backend = BACKEND.read_text()
    backend_plants = (
        ("DenseSpace::kernel_exclusion(kTactic)", "DenseSpace::topology_exclusion(kTactic, Stages)"),
        ("resolve_dense_config(config_name, m)", "resolve_dense_config(config_name)"),
        ("find_dense_config(config_name, m, config)", "find_dense_config(config_name, config)"),
        ("(DenseConfigId::ID == kDecodeDefaultDenseConfig), QueryOnly", "false, QueryOnly"),
    )
    for old, new in backend_plants:
        if old not in backend:
            print("[dense-shipping-tm8] FAIL cannot plant backend seam " + old)
            return 1
        if not source_errors(backend.replace(old, new, 1)):
            print("[dense-shipping-tm8] FAIL backend plant escaped: " + old)
            return 1

    print("[dense-shipping-tm8] PASS family=6 broad=52/8 exact=51/9 "
          "reason=compiled SharedStorageSize; M1/M7->8x128:8x32:s3 M8->64x64:32x32:s3; "
          "7 plants red")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
