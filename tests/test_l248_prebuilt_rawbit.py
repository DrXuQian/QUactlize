import ast
import ctypes
import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "tools" / "run_l248_q4_n16k64_prebuilt.py"
SPEC = importlib.util.spec_from_file_location("l248_prebuilt_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_self_test_closes_authority_one_device_and_probe_reds(tmp_path):
    environment = dict(os.environ)
    environment["TMPDIR"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-B", str(RUNNER_PATH), "--self-test"],
        check=False, capture_output=True, text=True, env=environment)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == (
        "L248_Q4_N16K64_PREBUILT_SELF_TEST PASS "
        "authority=PASS duplicate_json=RED one_device=PASS/RED "
        "sdk_same_release_tools=PASS sdk_release_runtime=RED/RED "
        "positive=PASS wrong_oracle=RED reds=6\n")


def test_external_source_binary_and_manifest_authorities_are_mandatory():
    base = [
        "/bundle", "--ppu-sdk", "/sdk", "--source-tree", "/source",
        "--verify-only",
    ]
    with pytest.raises(SystemExit):
        runner.parse_args(base)
    args = runner.parse_args(base + [
        "--expect-source-sha", "a" * 40,
        "--expect-binary-sha256", "b" * 64,
        "--expect-manifest-sha256", "c" * 64,
    ])
    assert args.expect_source_sha == "a" * 40
    assert args.expect_binary_sha256 == "b" * 64
    assert args.expect_manifest_sha256 == "c" * 64
    with pytest.raises(SystemExit):
        runner.parse_args(base + [
            "--expect-source-sha", "A" * 40,
            "--expect-binary-sha256", "b" * 64,
            "--expect-manifest-sha256", "c" * 64,
        ])


def test_one_device_contract_rejects_multi_visibility_and_runtime_count():
    assert runner._visible_device_ordinal(
        {"CUDA_VISIBLE_DEVICES": "12"}) == "12"
    for value in ("", "0,1", "GPU-deadbeef", "-1", " 0"):
        with pytest.raises(runner.GateError, match="exactly one numeric"):
            runner._visible_device_ordinal({"CUDA_VISIBLE_DEVICES": value})

    def get_one(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 1
        return 0

    def get_two(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 2
        return 0

    def get_zero(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 0
        return 0

    assert runner._runtime_device_identity_from_calls(
        get_one, get_zero, {"CUDA_VISIBLE_DEVICES": "7"}) == {
            "CUDA_VISIBLE_DEVICES": "7",
            "hggc_device_count": 1,
            "hggc_current_device": 0,
        }
    with pytest.raises(runner.GateError, match="exactly one visible"):
        runner._runtime_device_identity_from_calls(
            get_two, get_zero, {"CUDA_VISIBLE_DEVICES": "7"})


def test_positive_and_wrong_oracle_have_exact_raw_bit_contract():
    positive_text, negative_text = runner._sample_probe_lines()
    positive = runner._parse_probe(positive_text)
    negative = runner._parse_probe(negative_text)
    runner._assert_positive(0, positive)
    runner._assert_negative(1, negative, positive)

    launch_drift = dict(positive)
    launch_drift["sync"] = 1
    with pytest.raises(runner.GateError, match="sync differs"):
        runner._assert_positive(0, launch_drift)
    escaped_red = dict(negative)
    escaped_red["raw_bad"] = 0
    with pytest.raises(runner.GateError, match="raw_bad differs"):
        runner._assert_negative(1, escaped_red, positive)
    with pytest.raises(runner.GateError, match="exact RED rc=1"):
        runner._assert_negative(0, negative, positive)


def test_same_release_tool_byte_drift_is_green_but_release_runtime_drift_is_red():
    sdk = {
        "release": runner.SDK_RELEASE,
        "files": {
            "bin/hgcc": "1" * 64,
            "bin/hgobjdump": "2" * 64,
            "lib/libhggc_wrapper.so": runner.RUNTIME_SHA256,
        },
    }
    evidence = runner._assert_sdk_runtime_compatibility(
        runner.SDK_RELEASE, runner.RUNTIME_SHA256, sdk)
    assert evidence["build_tool_byte_identity_required_on_box"] is False
    assert evidence["build_tool_hashes_recorded"] == {
        "bin/hgcc": "1" * 64,
        "bin/hgobjdump": "2" * 64,
    }
    with pytest.raises(runner.GateError, match="release differs"):
        runner._assert_sdk_runtime_compatibility(
            "2.1.1-drift", runner.RUNTIME_SHA256, sdk)
    with pytest.raises(runner.GateError, match="runtime digest differs"):
        runner._assert_sdk_runtime_compatibility(
            runner.SDK_RELEASE, "0" * 64, sdk)


def test_box_sdk_validation_does_not_require_build_tools(monkeypatch, tmp_path):
    sdk_root = tmp_path / "sdk"
    (sdk_root / "lib").mkdir(parents=True)
    (sdk_root / "release.yaml").write_text(
        f"version: {runner.SDK_RELEASE}\n", encoding="utf-8")
    runtime = sdk_root / "lib/libhggc_wrapper.so"
    runtime.write_bytes(b"same admitted runtime fixture")
    manifest = {
        "release": runner.SDK_RELEASE,
        "files": {
            # Neither build tool exists in this execution-only SDK fixture.
            "bin/hgcc": "3" * 64,
            "bin/hgobjdump": "4" * 64,
            "lib/libhggc_wrapper.so": runner.RUNTIME_SHA256,
        },
    }
    monkeypatch.setattr(
        runner, "_sha256",
        lambda path: runner.RUNTIME_SHA256 if path == runtime else "f" * 64)
    resolved, evidence = runner._validate_sdk(sdk_root, manifest)
    assert resolved == sdk_root.resolve()
    assert evidence["build_tool_byte_identity_required_on_box"] is False

    (sdk_root / "release.yaml").write_text(
        "version: 2.1.1-drift\n", encoding="utf-8")
    with pytest.raises(runner.GateError, match="release differs"):
        runner._validate_sdk(sdk_root, manifest)
    (sdk_root / "release.yaml").write_text(
        f"version: {runner.SDK_RELEASE}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_sha256", lambda _path: "0" * 64)
    with pytest.raises(runner.GateError, match="runtime digest differs"):
        runner._validate_sdk(sdk_root, manifest)


def test_builder_default_allows_long_hgcc_compile():
    tools = str(ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    builder_path = ROOT / "tools/build_l248_q4_n16k64_prebuilt.py"
    spec = importlib.util.spec_from_file_location("l248_prebuilt_builder", builder_path)
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    with pytest.raises(SystemExit):
        builder.parse_args([
            "--ppu-sdk", "/sdk", "--sdk-archive", "/sdk.tar.gz",
            "--output", "/root/autodl-tmp/l248-test-output",
        ])
    args = builder.parse_args([
        "--ppu-sdk", "/sdk", "--sdk-archive", "/sdk.tar.gz",
        "--output", "/root/autodl-tmp/l248-test-output",
        "--source-sha", runner.REQUIRED_ANCESTOR,
    ])
    assert args.build_timeout_seconds == 7200


def test_box_runner_has_no_build_subprocess_surface():
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    parents: list[str] = []
    call_owners = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            parents.append(node.name)
            self.generic_visit(node)
            parents.pop()

        def visit_Call(self, node):
            if (isinstance(node.func, ast.Attribute) and
                    isinstance(node.func.value, ast.Name) and
                    node.func.value.id == "subprocess" and
                    node.func.attr == "run"):
                call_owners.append(parents[-1] if parents else "<module>")
            self.generic_visit(node)

    Visitor().visit(tree)
    assert sorted(call_owners) == sorted([
        "_execute_binary", "_run_git", "_verify_source_authority"])
    source = RUNNER_PATH.read_text(encoding="utf-8")
    for forbidden in ("build.sh", "cmake --build", "ninja", "make -j"):
        assert forbidden not in source


def test_device_evidence_refuses_overwrite(tmp_path):
    with pytest.raises(runner.GateError, match="refusing to overwrite"):
        runner._create_output(tmp_path)
