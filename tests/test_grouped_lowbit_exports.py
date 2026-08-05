import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "quactlize/csrc/device/ppu_dense_backend.cu"


def test_grouped_lowbit_has_the_full_grouped_operator_surface():
    """One token export is insufficient: inference needs workspace plus both async device entries."""
    source = BACKEND.read_text()
    expected = {
        "quactlize_ppu_grouped_lowbit",
        "quactlize_ppu_grouped_lowbit_config_v1",
        "quactlize_ppu_grouped_lowbit_config_valid_v1",
        "quactlize_ppu_grouped_lowbit_workspace_bytes_v1",
        "quactlize_ppu_grouped_lowbit_dev_v1",
        "quactlize_ppu_grouped_lowbit_dev_v2",
    }
    definitions = set(re.findall(
        r'extern\s+"C"\s+(?:int|int32_t|int64_t)\s+(quactlize_ppu_grouped_lowbit(?:\w*))\s*\(', source))
    assert definitions == expected, f"grouped_lowbit ABI mismatch: missing={expected-definitions}, extra={definitions-expected}"

    config_header = (ROOT / "quactlize/include/quactlize_ppu_config.h").read_text()
    device_header = (ROOT / "quactlize/include/quactlize_ppu_device.h").read_text()
    assert expected - {"quactlize_ppu_grouped_lowbit"} <= set(re.findall(
        r"(quactlize_ppu_grouped_lowbit(?:\w*))\s*\(", config_header + device_header))


def test_grouped_lowbit_names_the_scale_first_provider_not_packed_metadata():
    source = BACKEND.read_text()
    body = source[source.index("// SCALE_FIRST x GROUPED"):
                  source.index("// FULLY_QUANTIZED x GROUPED")]
    assert "GQM::FinegrainedScaleOnly, false" in body
    assert "SelectedPackedUnit" not in body
    assert "uint16_t const* scale" in body
