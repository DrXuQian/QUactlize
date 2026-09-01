"""Public loader contract for canonical fully-quantized K-pack artifacts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _normalized_header(name):
    text = (ROOT / "quactlize" / "include" / name).read_text()
    lines = [
        line.lstrip()[2:] if line.lstrip().startswith("//") else line
        for line in text.splitlines()
    ]
    return " ".join("\n".join(lines).split())


def test_host_abi_documents_independent_expert_major_slices():
    header = _normalized_header("quactlize_ppu_packed.h")
    for contract in (
            "blocks_e/recovered_e = base + e * (N*(K/256)*GGUF-block-bytes)",
            "low_e = low + e * (N*K*bits/8)",
            "high_e (when present) = high + e * (N*K*high_bits/8)",
            "units_e = units + e * quactlize_ppu_units_bytes(N,K,qtype)",
            "same e names the same expert in every allocation",
            "Experts are not interleaved inside any allocation",
            "experts=1 is byte-for-byte equivalent",
            "mutually disjoint slices may execute concurrently when each call's immutable arrangement descriptor",
            "stored outside every participating tensor range",
            "high must be null exactly when arrangement->high_bits==0"):
        assert contract in header


def test_device_abi_documents_pointer_and_library_selected_config_contracts():
    header = _normalized_header("quactlize_ppu_device.h")
    for contract in (
            "Every canonical K-pack arrangement-v2 device consumer uses the exact expert-major artifact",
            "For canonical K-pack, high must be null exactly when arrangement->high_bits==0",
            "Xplane v2 compatibility retains its legacy pointer contract",
            "a null or empty config_name delegates tactic selection to the loaded library",
            "canonical K-quant K-pack v2 first uses an exact measured (qtype,m,n,k) selection",
            "a null or empty config_name selects the one compiled grouped default",
            "grouped does not consult the dense exact-measurement table",
            "unknown non-empty name returns 39 rather than falling back"):
        assert contract in header
