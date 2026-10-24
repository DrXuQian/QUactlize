"""Product CI inventory. Historical experiments are not release admissions.

Keep paths here, not in CI-provider YAML or shell command strings. The host
suite uses real host code and explicit device doubles; it never launches a
GPU. Its SDK dependency is headers only. Accuracy consumes an installed
public package and must run on a real PPU.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HOST = (
    "test_ci_entry.py",
    "test_reference_gguf_kpack.py", "test_reference_gguf_kpack_file.py",
    "test_pack_gguf_cli.py", "test_kpack_device_pack.py", "test_q8_kpack2.py",
    "test_kpack_public_contract.py", "test_kquant_any_m_abi.py",
    "test_kquant_selected_config_abi.py", "test_kquant_measured_policy.py",
    "test_kpack_policy.py", "test_kpack_native_dispatch.py", "test_kpack_jit.py",
    "test_kpack_dynamic.py", "test_dense_provider.py", "test_ppu_library_package.py",
    "test_kpack_execution.py", "test_kpack_indexed.py", "test_bf16_compute.py",
    "test_bf16_metadata.py", "test_bf16_fastpath.py", "test_kpack_dequant.py",
    "test_kpack_prefill_runtime.py", "test_kpack_prefill_composition.py",
    "test_simt_common.py", "test_simt_scale_fields.py",
    "test_decode_selection.py", "test_decode_refresh_selection.py",
    "test_prefill_refresh_policy.py", "test_selection_refactor.py",
    "test_dense_prefill_policy.py", "test_dense_helper_admission.py",
    "test_prefill_dense.py", "test_prefill_production_helpers.py", "test_dense_full_helpers.py",
    "test_dense_cont.py", "test_dense_compact.py",
    "test_dense_q8_paired.py",
)

# These are retained, not silently deleted or counted as product CI passes.
# They depend on the old Torch/six-library API, private caller trees, archived
# measurements, or a separate NVIDIA/PPU1.7 toolchain. List them explicitly in
# `--list` so an integrator cannot mistake product CI for every experiment.
SEPARATE = {
    "test_formats.py": "legacy Torch extension",
    "test_layouts.py": "legacy Torch extension and NVIDIA compiler",
    "test_gguf_golden.py": "legacy Torch CPU/device arms and external fixtures",
    "test_gguf_routes.py": "legacy six-library/Torch device routes",
    "test_preprocess_ops.py": "legacy Torch extension",
    "test_kpack_host_abi.py": "legacy six-library bundle",
    "test_q4_scalefirst_prepass_abi.py": "legacy six-library bundle",
    "test_placed_artifact_abi.py": "legacy Torch extension ABI",
    "test_kpack_simt_intake.py": "private llama PoC integration",
    "test_kpack_moe_chain.py": "private llama PoC integration",
    "test_native_components_retry.py": "optional historical result archives",
    "test_bench_floor_links.py": "NVIDIA compiler/layout development probe",
    "test_q36_zero_redundancy.py": "NVIDIA compiler/layout development probe",
}

ACCURACY = ("ppu/test_public_library.py", "ppu/test_moe_library.py")


def inventory(root=ROOT):
    """Every Python test file is visible, including work outside product CI."""
    files = {str(p.relative_to(root / "tests")) for p in (root / "tests").rglob("test_*.py")}
    declared = set(HOST) | set(ACCURACY) | set(SEPARATE)
    missing = declared - files
    if missing:
        raise ValueError("CI inventory names missing tests: " + ", ".join(sorted(missing)))
    if len(HOST) != len(set(HOST)) or set(HOST) & (set(ACCURACY) | set(SEPARATE)):
        raise ValueError("duplicate CI test ownership")
    return {
        "host": sorted(HOST),
        "accuracy": sorted(ACCURACY),
        "tooling": sorted(files - declared - {p for p in files if p.startswith("ppu17/")}),
        "separate": dict(sorted({**SEPARATE, **{
            p: "separate PPU1.7/Hopper checkout and runner" for p in files if p.startswith("ppu17/")
        }}.items())),
    }
