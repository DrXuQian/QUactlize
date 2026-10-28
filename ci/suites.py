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
    "test_product_tree.py",
    "test_reference_gguf_kpack.py", "test_reference_gguf_kpack_file.py",
    "test_pack_gguf_cli.py", "test_kpack_device_pack.py", "test_q8_kpack2.py",
    "test_kpack_public_contract.py", "test_kquant_any_m_abi.py",
    "test_kquant_selected_config_abi.py", "test_kquant_measured_policy.py",
    "test_kpack_policy.py", "test_kpack_native_dispatch.py", "test_kpack_jit.py",
    "test_kpack_dynamic.py", "test_dense_provider.py", "test_ppu_library_package.py",
    "test_kpack_execution.py", "test_kpack_indexed.py", "test_bf16_compute.py",
    "test_bf16_metadata.py", "test_bf16_fastpath.py", "test_kpack_dequant.py",
    "test_kpack_prefill_runtime.py", "test_simt_scale_fields.py",
    "test_decode_selection.py", "test_moe_decode_policy.py",
    "test_prefill_refresh_policy.py", "test_dense_prefill_policy.py",
    "test_dense_helper_admission.py",
    "test_prefill_dense.py", "test_dense_full_helpers.py",
    "test_dense_cont.py", "test_dense_compact.py",
    "test_dense_q8_paired.py",
)

# Sweep/importer tests, probes and legacy backends live on experiments/ppu-tuning.

ACCURACY = ("ppu/test_public_library.py", "ppu/test_moe_library.py")


def inventory(root=ROOT):
    """Every product test has an explicit owner; no implicit tooling suite."""
    files = {str(p.relative_to(root / "tests")) for p in (root / "tests").rglob("test_*.py")}
    declared = set(HOST) | set(ACCURACY)
    missing = declared - files
    if missing:
        raise ValueError("CI inventory names missing tests: " + ", ".join(sorted(missing)))
    if files - declared:
        raise ValueError("unregistered product tests: " + ", ".join(sorted(files - declared)))
    if len(HOST) != len(set(HOST)) or set(HOST) & set(ACCURACY):
        raise ValueError("duplicate CI test ownership")
    return {
        "host": sorted(HOST),
        "accuracy": sorted(ACCURACY),
    }
