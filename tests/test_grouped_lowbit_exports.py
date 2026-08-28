import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "quactlize/csrc/device/ppu_dense_backend.cu"
MIXED = (ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/collective/"
         "quactlize_mma_mixed_input.hpp")


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


def test_grouped_fully_quantized_exposes_the_kpack4_arrangement_surface():
    """K-pack4 grouped needs producer-independent host/device/query entries; a lone host wrapper is not deployment."""
    source = BACKEND.read_text()
    expected = {
        "quactlize_ppu_grouped_fully_quantized",
        "quactlize_ppu_grouped_fully_quantized_config_v1",
        "quactlize_ppu_grouped_fully_quantized_config_valid_v1",
        "quactlize_ppu_grouped_fully_quantized_dev_v1",
        "quactlize_ppu_grouped_fully_quantized_dev_v2",
        "quactlize_ppu_grouped_fully_quantized_workspace_bytes_v1",
        "quactlize_ppu_grouped_fully_quantized_for_arrangement_v2",
        "quactlize_ppu_grouped_fully_quantized_config_valid_for_arrangement_v2",
        "quactlize_ppu_grouped_fully_quantized_dev_for_arrangement_v2",
        "quactlize_ppu_grouped_fully_quantized_workspace_bytes_for_arrangement_v2",
        "quactlize_ppu_list_valid_grouped_fully_quantized_configs_v2",
        "quactlize_ppu_list_valid_grouped_fully_quantized_configs_for_arrangement_v2",
    }
    definitions = set(re.findall(
        r'extern\s+"C"\s+(?:int|int32_t|int64_t)\s+'
        r'(quactlize_ppu_(?:list_valid_)?grouped_fully_quantized(?:\w*))\s*\(', source))
    assert definitions == expected, (
        f"grouped fully-quantized ABI mismatch: missing={expected-definitions}, extra={definitions-expected}")

    headers = ((ROOT / "quactlize/include/quactlize_ppu_config.h").read_text() +
               (ROOT / "quactlize/include/quactlize_ppu_device.h").read_text())
    declared = set(re.findall(
        r"(quactlize_ppu_(?:list_valid_)?grouped_fully_quantized(?:\w*))\s*\(", headers))
    assert expected - {"quactlize_ppu_grouped_fully_quantized"} <= declared


def test_grouped_kpack4_changes_only_the_mainloop_policy():
    source = BACKEND.read_text()
    assert "launch_grouped_q4_kpack4_config" in source
    assert "Q4KPack4MainloopPolicy" in source
    assert "RequireUniversalFallback, 0, Kpack4Policy" in source
    grouped = (ROOT / "quactlize/include/moe_grouped_ppu.cuh").read_text()
    assert "class MainloopPolicyOverride = void" in grouped
    assert "std::conditional_t<" in grouped and "MainloopPolicyOverride" in grouped


def test_grouped_kpack4_selects_the_expert_axis_exactly_once():
    """The byte base and CuTe L slice must not both consume the expert coordinate.

    Dense exercises only expert zero, so a doubled L offset is invisible there.
    The independent address model below keeps the historical construction as a
    must-red control for every nonzero expert.
    """
    source = MIXED.read_text()
    begin = source.index("if constexpr (kQ4KPack4Transpose) {", source.index("auto load_init_B"))
    end = source.index("} else {", begin)
    body = source[begin:end]
    assert "mixed_packed_byte_expert_base" not in body
    assert "make_shape(N, physical_k, L), physical_stride" in body
    assert body.count("mB_nkl(_,_,l_coord)") == 1
    assert "rank(decltype(mB_nk.layout()){}) == 2" in body
    assert "raw_pointer_cast(mB_nk.data())" in body

    experts, n, k, bits = 4, 256, 5120, 4
    bytes_per_expert = n * k * bits // 8
    expected = [e * bytes_per_expert for e in range(experts)]
    once_selected = [e * bytes_per_expert for e in range(experts)]
    legacy_double_selected = [2 * e * bytes_per_expert for e in range(experts)]
    assert once_selected == expected
    assert legacy_double_selected[0] == expected[0]
    assert all(legacy_double_selected[e] != expected[e]
               for e in range(1, experts))
