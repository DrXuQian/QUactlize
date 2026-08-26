#!/usr/bin/env python3
"""Fail closed unless the extension include namespace has one canonical owner.

The public include root is ``actlize_extensions``.  This check is deliberately
separate from check_extension_additive.py: additive ownership answers whether
the extension definitions overlap actlize, while this file answers whether the
renamed include tree exists and whether active first-party source still names
the retired root.  A missing canonical tree is therefore a failure, never a
skip.

Coordinator transcripts and committed ACU reports are historical provenance;
they describe old source states and are not active include consumers.  Vendor
source is also outside this repository-owned namespace contract.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent.parent
CANONICAL_REL = Path("quactlize/include/actlize_extensions")
LEGACY_TOKEN = "quactlize" + "_extensions"

# This is the migration inventory, not a count sampled from the tree being
# checked.  A missing header and an unrelated replacement must not cancel out
# into the same total of 39.
MIGRATED_HEADERS = frozenset(
    {
        "cutlass/detail/quactlize_mixed_dtype.hpp",
        "cutlass/gemm/collective/builders/quactlize_mma_builder.inl",
        "cutlass/gemm/collective/detail/ppu_2plane_source_layout.hpp",
        "cutlass/gemm/collective/detail/ppu_a_pack.hpp",
        "cutlass/gemm/collective/detail/ppu_mixed_a_schedule.hpp",
        "cutlass/gemm/collective/detail/ppu_mixed_argument_contract.hpp",
        "cutlass/gemm/collective/detail/ppu_mixed_metadata_policy.hpp",
        "cutlass/gemm/collective/detail/ppu_mixed_pipeline.hpp",
        "cutlass/gemm/collective/marlin_collective_ppu.hpp",
        "cutlass/gemm/collective/marlin_dequant_ppu.hpp",
        "cutlass/gemm/collective/marlin_load_ppu.hpp",
        "cutlass/gemm/collective/marlin_mma_ppu.hpp",
        "cutlass/gemm/collective/ppu_mma_aiu_fold.hpp",
        "cutlass/gemm/collective/ppu_mma_aiu_mixed_input_2plane.hpp",
        "cutlass/gemm/collective/quactlize_mma_mixed_input.hpp",
        "cutlass/gemm/device/marlin_gemm_ppu.hpp",
        "cutlass/gemm/device/ppu_mixed_input_splitk_parallel.hpp",
        "cutlass/gemm/kernel/marlin_kernel_ppu.hpp",
        "cutlass/gemm/kernel/marlin_output_map_ppu.hpp",
        "cutlass/gemm/kernel/marlin_scheduler_ppu.hpp",
        "cutlass/gemm/kernel/ppu_accumulator_residue_mask.hpp",
        "cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_group_marlin.hpp",
        "cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_group_streamk.hpp",
        "cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_marlin.hpp",
        "cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_persistent.hpp",
        "cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_splitk_parallel.hpp",
        "cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_streamk.hpp",
        "cutlass/gemm/kernel/ppu_fixed_splitk_completion_protocol.hpp",
        "cutlass/gemm/kernel/ppu_fixed_splitk_last_arriver.hpp",
        "cutlass/gemm/kernel/ppu_fixed_splitk_partition.hpp",
        "cutlass/gemm/kernel/ppu_grouped_ragged_geometry.hpp",
        "cutlass/gemm/quactlize_dispatch_policy.hpp",
        "cutlass/gguf_packed_scale.h",
        "cutlass/quactlize_mix_gemm_convert.h",
    }
)

PRUNED_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "third_party",
}
PRUNED_DIR_PREFIXES = (".moe_units_check",)
HISTORICAL_PREFIXES = (".coord/", "dev/acu/")
TEXT_SUFFIXES = {
    "",
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".in",
    ".inc",
    ".inl",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _active_files(root: Path):
    """Yield repository-owned text candidates, excluding generated/history roots."""

    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        relative_dir = current_path.relative_to(root)
        relative_text = relative_dir.as_posix()
        if relative_text != "." and any(
            relative_text == prefix.rstrip("/") or relative_text.startswith(prefix)
            for prefix in HISTORICAL_PREFIXES
        ):
            dirs[:] = []
            continue
        dirs[:] = [
            name
            for name in dirs
            if name not in PRUNED_DIRS
            and not any(name.startswith(prefix) for prefix in PRUNED_DIR_PREFIXES)
        ]
        for name in files:
            path = current_path / name
            relative = path.relative_to(root)
            relative_text = relative.as_posix()
            if any(relative_text.startswith(prefix) for prefix in HISTORICAL_PREFIXES):
                continue
            yield path, relative


def audit(root: Path) -> tuple[list[str], int]:
    errors: list[str] = []
    canonical = root / CANONICAL_REL
    if not canonical.is_dir():
        errors.append(f"canonical tree missing: {CANONICAL_REL}")
        actual_headers: set[str] = set()
    else:
        actual_headers = {
            path.relative_to(canonical).as_posix()
            for path in canonical.rglob("*")
            if path.is_file()
        }
        missing = sorted(MIGRATED_HEADERS - actual_headers)
        if missing:
            errors.append("migrated canonical headers missing: " + ", ".join(missing))

    scanned = 0
    legacy_bytes = LEGACY_TOKEN.encode()
    for path, relative in _active_files(root):
        relative_text = relative.as_posix()
        if LEGACY_TOKEN in relative_text:
            errors.append(f"legacy token in active path: {relative_text}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        scanned += 1
        try:
            payload = path.read_bytes()
        except OSError as exc:
            errors.append(f"cannot read active source {relative_text}: {exc}")
            continue
        if legacy_bytes in payload:
            errors.append(f"legacy token in active content: {relative_text}")
    return errors, scanned


def self_test() -> list[str]:
    """Prove the two fail-closed edges without mutating the checkout."""

    broken: list[str] = []
    base = Path(
        os.environ.get(
            "ACTLIZE_EXTENSION_NAMESPACE_SELFTEST_OUT",
            "/workspace/actlize-extension-namespace-selftest",
        )
    ).resolve()
    root = base / f"run-{os.getpid()}"
    root.mkdir(parents=True, exist_ok=False)
    try:
        canonical = root / CANONICAL_REL
        for relative in MIGRATED_HEADERS:
            path = canonical / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("#pragma once\n")

        plant = root / "tests" / "legacy_include_plant.cu"
        plant.parent.mkdir(parents=True)
        plant.write_text(f'#include "{LEGACY_TOKEN}/cutlass/planted.hpp"\n')
        errors, _ = audit(root)
        if not any("legacy token in active content" in error for error in errors):
            broken.append("old-include plant was not rejected")

        plant.unlink()
        clean_errors, _ = audit(root)
        if clean_errors:
            broken.append("synthetic canonical tree is not accepted: " + "; ".join(clean_errors))

        shutil.rmtree(canonical)
        errors, _ = audit(root)
        if not any("canonical tree missing" in error for error in errors):
            broken.append("missing canonical tree was not rejected")
    finally:
        # The target is created above as one PID-named child of the resolved
        # workspace base.  Refuse cleanup if that relationship ever changes.
        if root.parent == base and root.name == f"run-{os.getpid()}" and root.is_dir():
            shutil.rmtree(root)
    return broken


def main() -> int:
    broken = self_test()
    if broken:
        print("[actlize-extension-namespace] ERROR: self-test failed: " + "; ".join(broken))
        return 1

    errors, scanned = audit(ROOT)
    if errors:
        print(f"[actlize-extension-namespace] FAIL: {len(errors)} violation(s)")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(
        "[actlize-extension-namespace] PASS: "
        f"migrated_headers={len(MIGRATED_HEADERS)} active_text_files={scanned} "
        "legacy_active_refs=0 plants=old-include+missing-tree/EXPECTED_RED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
