import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def load(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


select = load("select_fq_a02_typed_diagnostics")
check = load("check_fq_a02_typed_diagnostics")
prebuilt = load("fq_a02_prebuilt")


def shard(root, qtype, bchunk, symbol, provider):
    root.mkdir()
    unit = root / "one.cu"
    unit.write_text(
        symbol + '\n#include "fully_quantized_splitk_producer_unit.inc"\n')
    row = {
        "symbol": symbol, "qtype": qtype, "artifact_tile_k": 64,
        "tile_m": 8, "tile_n": 64, "tactic_tile_k": 256,
        "warp_m": 8, "warp_n": 16, "stages": 2,
        "bchunk": bchunk, "a_provider": provider,
    }
    (root / "manifest.json").write_text(json.dumps({
        "identity": {"qtype": qtype, "artifact_tile_k": 64,
                     "bchunk": bchunk},
        "typed_rows": [row], "units": [str(unit)],
        "denominator": {"typed_rows": 1},
    }))


def test_selector_materializes_exact_pairs_and_labels_only_q4_ap0_shipping(tmp_path):
    q4, q3_bc0, q3_bc1 = tmp_path / "q4", tmp_path / "q30", tmp_path / "q31"
    q4.mkdir()
    rows, units = [], []
    for index, symbol in enumerate(select.Q4):
        unit = q4 / f"u{index}.cu"
        unit.write_text(symbol)
        units.append(str(unit))
        rows.append({
            "symbol": symbol, "qtype": 12, "artifact_tile_k": 64,
            "tile_m": 8, "tile_n": 64, "tactic_tile_k": 256,
            "warp_m": 8, "warp_n": 16, "stages": 2, "bchunk": 0,
            "a_provider": "standard-aiu" if index == 0 else "packed-row",
        })
    (q4 / "manifest.json").write_text(json.dumps({
        "identity": {"qtype": 12, "artifact_tile_k": 64, "bchunk": 0},
        "typed_rows": rows, "units": units, "denominator": {"typed_rows": 2},
    }))
    shard(q3_bc0, 11, 0, select.Q3[0], "standard-aiu")
    shard(q3_bc1, 11, 1, select.Q3[1], "standard-aiu")
    output = tmp_path / "output"
    select.materialize(q4, q3_bc0, q3_bc1, output)
    q4_manifest = json.loads((output / "q4/manifest.json").read_text())
    q3_manifest = json.loads((output / "q3/manifest.json").read_text())
    assert q4_manifest["classification"] == [
        "PRODUCT_SHIPPING", "TYPED_DIAGNOSTIC_NONPRODUCT"]
    assert q3_manifest["classification"] == ["TYPED_DIAGNOSTIC_NONPRODUCT"] * 2
    assert "FQ_TC_GENERATED_BCHUNK -1" in (
        output / "q3/fq_tc_registry.inc").read_text()
    assert "fq_a02_q3_bc0_generated" in (
        output / "q3/units/fq_a02_unit_0.cu").read_text()
    assert "fq_a02_q3_bc1_generated" in (
        output / "q3/units/fq_a02_unit_1.cu").read_text()


def a01_result():
    stability = {
        "phase": "same-row-explicit-config", "launches": 1,
        "raw_bits_stable": True, "raw_sha256": "a" * 64,
        "first_launch_status": {}, "last_launch_status": {},
    }
    grouped = {
        "shape": [6, 256, 512, 4], "rows_per_expert": [2, 0, 3, 1],
        "null_config_error": 0, "explicit_config_error": 0,
        "expert0_rebind_error": {}, "units_only_metadata_error": {},
        "raw_bit_stability": copy.deepcopy(stability),
        "workspace_bytes": 1, "launch_status": [],
    }
    format_row = {
        "qtype": 10, "role": "fmt2", "packed_format": 2,
        "arrangement": [], "any_m": {}, "selected_config": {},
        "shipping_policy": None, "host_prepare_recover": "BYTE_EXACT",
        "frozen_host_artifact": {}, "bad_mapping": {},
        "dense": {
            "shape": [1, 1024, 5120], "null_config_error": 0,
            "explicit_config_error": 0, "zeroed_scale_unit_error": {},
            "raw_bit_stability": copy.deepcopy(stability),
            "workspace_bytes": 1, "launch_status": [],
        },
        "grouped": grouped,
    }
    formats = {name: copy.deepcopy(format_row) for name in prebuilt.FORMAT_NAMES}
    for name, (qtype, role, packed_format) in prebuilt.FORMAT_IDENTITIES.items():
        formats[name]["qtype"] = qtype
        formats[name]["role"] = role
        formats[name]["packed_format"] = packed_format
    formats["Q4_K"]["shipping_policy"] = copy.deepcopy(
        prebuilt.Q4_PER_FORMAT_POLICY)
    for route in ("dense", "grouped"):
        formats["Q4_K"][route]["raw_bit_stability"]["launches"] = 8192
    libraries = [
        {"role": role, "filename": f"{role}.so", "size": 1,
         "sha256": str(index) * 64}
        for index, role in enumerate(sorted(prebuilt.LIBRARY_ROLES), 1)
    ]
    return {
        "schema": prebuilt.A01_SCHEMA,
        "schema_version": prebuilt.A01_SCHEMA_VERSION,
        "status": "PASS",
        "execution": {"device_library_builds": 0, "host_compilations": 0,
                      "runner": "python-ctypes",
                      "library_load_mode": "six DSOs, RTLD_LOCAL, one process"},
        "bundle": {
            "manifest_sha256": "a" * 64,
            "source": {"bundle_source_commit": "b" * 40,
                       "checkout_head": "b" * 40, "runner_sha256": "b" * 64},
            "sdk": {"root": "/sdk", "release": "test",
                    "archive_sha256": "c" * 64,
                    "release_receipt_sha256": "c" * 64,
                    "hgobjdump_sha256": "c" * 64, "runtime_path": "/runtime",
                    "runtime_sha256": "c" * 64, "device": {}},
            "default_library_identity": {
                "path": "default.so", "packed_format": -1,
                "any_m": "REJECTS_ALL_CANONICAL_DESCRIPTORS"},
            "libraries": libraries,
        },
        "python": {"numpy": "test", "gguf": "0.19.0"},
        "formats": formats,
        "coverage": {
            "dense_exact_shape": [1, 1024, 5120],
            "grouped_shape": [6, 256, 512, 4],
            "empty_expert_rows": [2, 0, 3, 1],
            "null_and_explicit_launches": sorted(prebuilt.FORMAT_NAMES),
            "bad_mapping_workspace_queries": "EXPECTED_MINUS_ONE",
            "zeroed_scale_unit_fault": "EXPECTED_NUMERIC_RED",
            "grouped_expert0_rebind_fault": "EXPECTED_NUMERIC_RED",
            "grouped_units_only_metadata_fault": "FMT0_EXPECTED_NUMERIC_RED",
            "q4_correctness_repeats": 8192,
            "q4_product_policy": copy.deepcopy(prebuilt.Q4_PRODUCT_POLICY),
            "q4_product_policy_source": {}, "numeric_reference": "official",
            "host_prepare_recover_scope": "independent",
        },
    }


def write_result(path, value):
    path.write_text(json.dumps(value))
    return path


def test_a01_result_validator_accepts_exact_product_evidence(tmp_path):
    summary = prebuilt.validate_a01_result(
        write_result(tmp_path / "result.json", a01_result()))
    assert summary["libraries"] == 6
    assert summary["formats"] == 5
    assert summary["q4_correctness_repeats"] == 8192
    assert summary["q4_product_policy"] == prebuilt.Q4_PRODUCT_POLICY


@pytest.mark.parametrize("plant", [
    "unknown", "missing", "schema", "status", "libraries", "formats",
    "empty-expert", "grouped-stability", "product-policy", "q4-repeats",
    "format-identity", "per-format-policy", "default-library", "execution",
])
def test_a01_result_authority_plants_fail_closed(tmp_path, plant):
    value = a01_result()
    if plant == "unknown": value["unknown"] = True
    elif plant == "missing": value.pop("coverage")
    elif plant == "schema": value["schema_version"] += 1
    elif plant == "status": value["status"] = "FAIL"
    elif plant == "libraries": value["bundle"]["libraries"].pop()
    elif plant == "formats": value["formats"].pop("Q3_K")
    elif plant == "empty-expert": value["coverage"]["empty_expert_rows"] = [6]
    elif plant == "grouped-stability": (
        value["formats"]["Q2_K"]["grouped"]["raw_bit_stability"].update(
            raw_bits_stable=False))
    elif plant == "product-policy": value["coverage"]["q4_product_policy"]["dense"]["bchunk"] = 1
    elif plant == "q4-repeats": value["coverage"]["q4_correctness_repeats"] = 1
    elif plant == "format-identity": value["formats"]["Q3_K"]["qtype"] = 12
    elif plant == "per-format-policy": value["formats"]["Q4_K"]["shipping_policy"]["dense"]["bchunk"] = 1
    elif plant == "default-library": value["bundle"]["default_library_identity"]["packed_format"] = 0
    else: value["execution"]["library_load_mode"] = "global"
    with pytest.raises(prebuilt.Error):
        prebuilt.validate_a01_result(write_result(tmp_path / "result.json", value))


def test_duplicate_json_keys_are_rejected(tmp_path):
    result = tmp_path / "result.json"
    result.write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(prebuilt.Error, match="duplicate JSON key"):
        prebuilt.validate_a01_result(result)


@pytest.mark.parametrize("keys", [
    {"schema", "schema_version", "source", "sdk", "build", "artifacts"},
    {"schema", "schema_version", "source", "sdk", "build", "artifacts",
     "classification", "unknown"},
])
def test_prebuilt_manifest_unknown_or_missing_top_level_keys_are_rejected(keys):
    with pytest.raises(prebuilt.Error, match="manifest keys differ"):
        prebuilt.require_keys(
            {key: None for key in keys},
            {"schema", "schema_version", "source", "sdk", "build",
             "artifacts", "classification"},
            "manifest")


def test_checker_red_controls():
    check.self_test()


def test_build_authority_resume_rejects_any_identity_drift(tmp_path, monkeypatch):
    authority = {
        "schema": "quactlize.fq-a02-build-authority.v1",
        "source": {"commit": "a" * 40, "submodules": [], "inputs": {}},
        "sdk": {"release": "test", "compiler_sha256": "b" * 64,
                "inspector_sha256": "c" * 64},
        "build": {"arch": "ppu0010", "q4_target": "q4", "q3_target": "q3"},
    }
    monkeypatch.setattr(prebuilt, "current_build_authority",
                        lambda root, sdk: copy.deepcopy(authority))
    path = tmp_path / "authority.json"
    prebuilt.write_build_authority(path, tmp_path, tmp_path)
    prebuilt.verify_build_authority(path, tmp_path, tmp_path)
    planted = copy.deepcopy(authority)
    planted["sdk"]["compiler_sha256"] = "d" * 64
    path.write_text(json.dumps(planted))
    with pytest.raises(prebuilt.Error, match="build authority differs"):
        prebuilt.verify_build_authority(path, tmp_path, tmp_path)


def test_runner_is_execute_only_and_product_labels_are_narrow():
    runner = (ROOT / "tools/run_fq_a02_prebuilt_box.sh").read_text()
    builder = (ROOT / "tools/build_fq_a02_prebuilt.sh").read_text()
    checker = (ROOT / "tools/check_fq_a02_typed_diagnostics.py").read_text()
    assert "build.sh" not in runner and "PPU_BUILD_DIR" not in runner
    assert "verify-a01" in runner and "grouped_execution=NONE" in runner
    assert "a01-product-gate-result.json" in runner
    assert checker.count("classification=PRODUCT_SHIPPING") == 1
    assert "TARGET=test_fq_a02_q3_bchunk_aggregate" in builder
    assert "RESUME" in builder and "verify-build-authority" in builder
    assert "PPU_BUILD_RESUME" in builder and 'work="$out.work"' in builder
    assert "assert_replaceable_regular_file" in builder
    assert 'out.work' in builder and 'RESUME:-0' in builder
    assert 'PPU_BUILD_RESUME="$q4_resume"' in builder
    assert 'PPU_BUILD_RESUME="$q3_resume"' in builder
    assert "env -u CC -u CXX" in builder
    assert "assert_build_identity q4" in builder
    assert "assert_build_identity q3" in builder
    assert 'mv -- "$publish" "$out"' in builder
    assert "\n  rm " not in builder and "\nrm " not in builder
    subprocess.run([
        "bash", "-n", str(ROOT / "tools/build_fq_a02_prebuilt.sh"),
        str(ROOT / "tools/run_fq_a02_prebuilt_box.sh")], check=True)


def test_namespace_macro_is_cleaned_and_shipping_sources_do_not_reach_a02():
    unit = (ROOT / "benchmarks/fully_quantized_splitk_producer_unit.inc").read_text()
    main = (ROOT / "benchmarks/test_fully_quantized_internal_sweep.cu").read_text()
    assert "#undef FQ_TC_GENERATED_NAMESPACE" in unit
    assert "#undef FQ_TC_DECLARE_NS_0" in main
    assert "#undef FQ_TC_DECLARE_NS_1" in main
    for path in (ROOT / "quactlize/csrc/device").glob("*.cu"):
        assert "FQ_A02" not in path.read_text()
    assert "test_fq_a02_q3_bchunk_aggregate" not in (
        ROOT / "quactlize/ppu_bundle.py").read_text()
