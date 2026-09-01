from __future__ import annotations

import ctypes
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from quactlize import ppu_bundle


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "tools" / "run_prebuilt_ppu_box_gate.py"
SPEC = importlib.util.spec_from_file_location("quactlize_prebuilt_box_gate", GATE_PATH)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


class _Function:
    def __init__(self, implementation):
        self.implementation = implementation
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.implementation(*args)


class _FakeRuntime:
    HOST_TO_DEVICE = 1
    DEVICE_TO_HOST = 2

    def __init__(self):
        self.memory = {}
        self.next_address = 0x10000

    def malloc(self, output, size):
        address = self.next_address
        self.next_address += int(size) + 0x100
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p)).contents.value = address
        self.memory[address] = bytearray(int(size))
        return 0

    def free(self, pointer):
        self.memory.pop(pointer.value)
        return 0

    def memcpy(self, destination, source, size, kind):
        size = int(size)
        if kind == self.HOST_TO_DEVICE:
            self.memory[destination.value][:] = ctypes.string_at(source.value, size)
        elif kind == self.DEVICE_TO_HOST:
            ctypes.memmove(destination.value, bytes(self.memory[source.value]), size)
        else:
            raise AssertionError(kind)
        return 0

    @staticmethod
    def get_last_error():
        return 0

    @staticmethod
    def synchronize():
        return 0

    @staticmethod
    def _error(code):
        return f"error-{code}"

    def check(self, operation, code):
        assert code == 0, (operation, code)

    def store(self, pointer, value):
        payload = np.ascontiguousarray(value).tobytes()
        assert len(payload) == len(self.memory[pointer.value])
        self.memory[pointer.value][:] = payload


def _write_config(pointer, ctype, record):
    output = ctypes.cast(pointer, ctypes.POINTER(ctype)).contents
    ctypes.memset(ctypes.byref(output), 0, ctypes.sizeof(output))
    values = list(record)
    values[1] = values[1].encode("ascii")
    for (field, _field_type), value in zip(ctype._fields_, values):
        setattr(output, field, value)


def _selected_library(spec):
    dense_name = spec.expected_dense_config.encode("ascii")
    default_name = gate.DECODE_DEFAULT_NAME.encode("ascii")
    grouped_name = gate.GROUPED_DEFAULT_NAME.encode("ascii")

    def dense(output, m, _n, _k, _group, _qtype, _arrangement, requested):
        if requested == b"stale-config":
            ctypes.memset(output, 0, ctypes.sizeof(gate.ConfigV4))
            return 0
        selected = dense_name
        if spec.layout == 2 and (m == 3 or requested == default_name):
            selected = default_name
        if requested not in (None, dense_name, default_name):
            return 0
        record = (spec.expected_decode_default_record
                  if selected == default_name else spec.expected_dense_record)
        _write_config(output, gate.ConfigV4, record)
        return 1

    def grouped(output, _total, _n, _k, _group, _experts, _max_rows,
                _qtype, _arrangement, requested):
        if requested not in (None, grouped_name):
            return 0
        _write_config(output, gate.ConfigV3, spec.expected_grouped_record)
        return 1

    def grouped_any_m(_n, _k, _experts, qtype, arrangement):
        if not arrangement or qtype != spec.qtype:
            return 0
        value = ctypes.cast(arrangement, gate.ARRP).contents
        return int(value.mapping_id == spec.mapping_id)

    def dense_any_m(_n, _k, qtype, arrangement):
        if not arrangement or qtype != spec.qtype:
            return 0
        value = ctypes.cast(arrangement, gate.ARRP).contents
        return int(value.mapping_id == spec.mapping_id)

    return SimpleNamespace(
        dense_selected=_Function(dense), grouped_selected=_Function(grouped),
        dense_any_m=_Function(dense_any_m),
        grouped_any_m=_Function(grouped_any_m))


def _manifest(tmp_path):
    libraries = []
    for role in ppu_bundle.LIBRARY_ROLES:
        path = tmp_path / role.filename
        path.write_bytes(role.role.encode("ascii"))
        libraries.append({
            "role": role.role,
            "filename": role.filename,
            "size": path.stat().st_size,
            "sha256": "0" * 64,
        })
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    return {
        "source": {"commit": "a" * 40, "submodules": []},
        "toolchain": {
            "sdk_release": ppu_bundle.SDK_RELEASE,
            "sdk_archive_sha256": ppu_bundle.SDK_ARCHIVE_SHA256,
        },
        "libraries": libraries,
    }


def test_box_gate_has_no_compiler_or_host_extension_build_surface():
    source = GATE_PATH.read_text(encoding="utf-8")
    for forbidden in (
            "setup.py", "build_ext", "build.sh", "cmake", "ninja",
            "hgcc", "nvcc", "import torch", "from quactlize"):
        assert forbidden not in source
    assert source.count("subprocess.run(") == 1
    assert "def _run_git(" in source
    assert gate.parse_args([
        "/bundle", "--ppu-sdk", "/sdk", "--output", "/result",
    ]).bundle == Path("/bundle")


def test_format_table_is_exactly_the_five_canonical_kpack_roles():
    assert [spec.name for spec in gate.FORMATS] == [
        "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]
    assert [(spec.qtype, spec.role, spec.packed_format) for spec in gate.FORMATS] == [
        (10, "fmt2", 2), (11, "fmt3", 3), (12, "fmt0", 0),
        (13, "fmt1", 1), (14, "fmt4", 4),
    ]
    assert gate.DENSE_SHAPE == (1, 1024, 5120)
    assert gate.GROUPED_ROWS == (2, 0, 3, 1)
    assert all(spec.expected_arrangement[0] == 2 for spec in gate.FORMATS)
    assert {spec.expected_arrangement[1] for spec in gate.FORMATS} == {1, 2}


def test_default_sixth_library_runtime_identity_must_be_minus_one(
        tmp_path, monkeypatch):
    path = tmp_path / "libquactlize_ppu.so"
    path.write_bytes(b"fake")

    def library(identity):
        return SimpleNamespace(
            _name=str(path),
            quactlize_ppu_build_packed_format_v1=_Function(lambda: identity))

    monkeypatch.setattr(gate.ctypes, "CDLL", lambda *_args, **_kwargs: library(-1))
    assert gate._assert_default_library_identity(path) == {
        "path": str(path), "packed_format": -1}
    monkeypatch.setattr(gate.ctypes, "CDLL", lambda *_args, **_kwargs: library(0))
    with pytest.raises(gate.GateError, match="expected -1"):
        gate._assert_default_library_identity(path)


@pytest.mark.parametrize("spec", gate.FORMATS, ids=lambda spec: spec.name)
def test_frozen_host_artifact_hashes_come_from_independent_reference(spec):
    import torch
    from reference import gguf_kpack

    n, k = gate.HOST_ABI_FIXTURE_SHAPE
    rows = n * (k // 256)
    raw = ((np.arange(rows * spec.block_bytes, dtype=np.uint64) * 37 +
            spec.qtype) & 0xFF).astype(np.uint8).reshape(rows, spec.block_bytes)
    artifact = gguf_kpack.prepare_dense(
        torch.from_numpy(raw.copy()), n, k, spec.name)
    got = tuple(gate._array_sha256(value.numpy()) for value in (
        artifact.low, artifact.high, artifact.units))
    assert got == gate.HOST_ABI_FIXTURE_SHA256[spec.qtype]


@pytest.mark.parametrize("spec", gate.FORMATS, ids=lambda spec: spec.name)
def test_selected_config_contract_checks_null_explicit_and_stale(spec):
    arrangement = gate.ArrangementV2(*spec.expected_arrangement)
    dense, grouped, evidence = gate._assert_selected_config_contract(
        _selected_library(spec), spec, arrangement)
    assert dense.decode() == spec.expected_dense_config
    assert grouped.decode() == gate.GROUPED_DEFAULT_NAME
    assert evidence["dense"][1] == spec.expected_dense_config
    assert evidence["grouped"][1] == gate.GROUPED_DEFAULT_NAME
    assert evidence["stale"] == "FAIL_CLOSED"
    if spec.layout == 2:
        assert evidence["dense_unmeasured"][1] == gate.DECODE_DEFAULT_NAME
        assert evidence["dense_explicit_override"][1] == gate.DECODE_DEFAULT_NAME
    else:
        assert evidence["dense_unmeasured"] is None


def test_selected_config_contract_rejects_policy_drift():
    spec = gate.FORMATS[0]
    library = _selected_library(spec)

    def drift(output, *_arguments):
        _write_config(output, gate.ConfigV4, spec.expected_decode_default_record)
        return 1

    library.dense_selected = _Function(drift)
    arrangement = gate.ArrangementV2(*spec.expected_arrangement)
    with pytest.raises(gate.GateError, match="null dense selector differs"):
        gate._assert_selected_config_contract(library, spec, arrangement)


def test_selected_config_contract_rejects_geometry_drift_with_same_name():
    spec = gate.FORMATS[2]
    library = _selected_library(spec)
    correct = spec.expected_dense_record

    def drift(output, _m, _n, _k, _group, _qtype, _arrangement, requested):
        record = list(correct)
        record[2] += 24
        _write_config(output, gate.ConfigV4, tuple(record))
        return 1

    library.dense_selected = _Function(drift)
    arrangement = gate.ArrangementV2(*spec.expected_arrangement)
    with pytest.raises(gate.GateError, match="null dense selector differs"):
        gate._assert_selected_config_contract(library, spec, arrangement)


@pytest.mark.parametrize("spec", gate.FORMATS, ids=lambda spec: spec.name)
def test_any_m_contract_admits_shapes_and_rejects_identity_plants(spec):
    arrangement = gate.ArrangementV2(*spec.expected_arrangement)
    assert gate._assert_any_m_contract(
        _selected_library(spec), spec, arrangement) == {
            "dense": {
                "valid": 1,
                "bad_mapping": 0,
                "null_arrangement": 0,
                "foreign_format": 0,
            },
            "grouped": {
                "valid": 1,
                "bad_mapping": 0,
                "null_arrangement": 0,
                "foreign_format": 0,
            },
        }


@pytest.mark.parametrize("field", ["dense_any_m", "grouped_any_m"])
def test_any_m_contract_rejects_a_permissive_export(field):
    spec = gate.FORMATS[0]
    library = _selected_library(spec)
    setattr(library, field, _Function(lambda *_arguments: 1))
    arrangement = gate.ArrangementV2(*spec.expected_arrangement)
    with pytest.raises(gate.GateError, match="did not fail closed"):
        gate._assert_any_m_contract(library, spec, arrangement)


@pytest.mark.parametrize("spec", gate.FORMATS, ids=lambda spec: spec.name)
def test_bad_mapping_workspace_queries_must_return_exact_minus_one(spec):
    def dense(_m, _n, _k, _qtype, arrangement):
        value = ctypes.cast(arrangement, gate.ARRP).contents
        return -1 if value.mapping_id != spec.mapping_id else 16

    def grouped(_total, _max_rows, _n, _k, _experts, _qtype, arrangement):
        value = ctypes.cast(arrangement, gate.ARRP).contents
        return -1 if value.mapping_id != spec.mapping_id else 16

    library = SimpleNamespace(
        dense_workspace=_Function(dense), grouped_workspace=_Function(grouped))
    arrangement = gate.ArrangementV2(*spec.expected_arrangement)
    assert gate._mapping_red(library, spec, arrangement) == {
        "dense_workspace": -1, "grouped_workspace": -1}

    library.dense_workspace = _Function(lambda *_arguments: 0)
    with pytest.raises(gate.GateError, match="bad mapping was not rejected"):
        gate._mapping_red(library, spec, arrangement)


def test_grouped_reference_skips_empty_expert_without_readdressing_rows():
    total, n, k, experts = gate.GROUPED_SHAPE
    activation = np.ones((total, k), dtype=np.float16)
    weights = np.stack([
        np.full((n, k), expert + 1, dtype=np.float64)
        for expert in range(experts)
    ])
    reference, denominator = gate._grouped_reference(activation, weights)
    expected_experts = np.repeat(np.arange(experts), gate.GROUPED_ROWS)
    assert expected_experts.tolist() == [0, 0, 2, 2, 2, 3]
    assert np.array_equal(reference[:, 0], (expected_experts + 1) * k)
    assert np.array_equal(denominator, reference)


def test_grouped_fault_rebinds_only_other_experts_that_have_rows():
    value = np.arange(4 * 3, dtype=np.uint8).reshape(4, 3)
    planted = gate._plant_grouped_expert_zero(value.reshape(-1), 4).reshape(4, 3)
    assert np.array_equal(planted[0], value[0])
    assert np.array_equal(planted[1], value[1])  # the empty expert is untouched
    assert np.array_equal(planted[2], value[0])
    assert np.array_equal(planted[3], value[0])


def test_conditioned_error_makes_zeroed_scale_unit_fault_visible():
    reference = np.array([[2.0, -4.0]], dtype=np.float64)
    denominator = np.array([[2.0, 4.0]], dtype=np.float64)
    close = np.array([[2.0, -4.0]], dtype=np.float16)
    planted = np.zeros((1, 2), dtype=np.float16)
    assert gate._conditioned_error(close, reference, denominator) == 0.0
    assert gate._conditioned_error(planted, reference, denominator) == 1.0
    assert gate._conditioned_error(
        np.full((1, 2), np.nan, dtype=np.float16), reference, denominator,
    ) == float("inf")


def test_source_authority_allows_later_runner_commit_but_rejects_runtime_diff(
        monkeypatch):
    source = "a" * 40
    head = "b" * 40
    manifest = {
        "source": {
            "commit": source,
            "submodules": [
                {"path": "third_party/actlize", "commit": "c" * 40},
                {"path": "third_party/cutlass", "commit": "d" * 40},
            ],
        },
    }

    def git(arguments):
        if arguments == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=head + "\n")
        if arguments[:2] == ["cat-file", "-e"]:
            assert arguments[2] == f"{source}^{{commit}}"
            return SimpleNamespace(returncode=0, stdout="")
        if arguments[:2] == ["diff", "--quiet"]:
            assert "HEAD" in arguments
            if source in arguments:
                assert "tools/run_prebuilt_ppu_box_gate.py" not in arguments
            return SimpleNamespace(returncode=0, stdout="")
        if arguments[:2] == ["ls-files", "--error-unmatch"]:
            return SimpleNamespace(
                returncode=0, stdout="tools/run_prebuilt_ppu_box_gate.py\n")
        if arguments[:2] == ["diff", "--cached"]:
            return SimpleNamespace(returncode=0, stdout="")
        if arguments[:3] == ["ls-files", "--others", "--exclude-standard"]:
            return SimpleNamespace(returncode=0, stdout="")
        if arguments[:2] == ["-C", "third_party/actlize"] or \
                arguments[:2] == ["-C", "third_party/cutlass"]:
            return SimpleNamespace(returncode=0, stdout="")
        assert arguments == ["submodule", "status", "--recursive"]
        return SimpleNamespace(
            returncode=0,
            stdout=(f" {'c' * 40} third_party/actlize\n"
                    f" {'d' * 40} third_party/cutlass\n"))

    monkeypatch.setattr(gate, "_run_git", git)
    authority = gate._assert_source_authority(manifest)
    assert authority["bundle_source_commit"] == source
    assert authority["checkout_head"] == head
    assert len(authority["runner_sha256"]) == 64

    def runtime_diff(arguments):
        result = git(arguments)
        if arguments[:2] == ["diff", "--quiet"]:
            result.returncode = 1
        return result

    monkeypatch.setattr(gate, "_run_git", runtime_diff)
    with pytest.raises(gate.GateError, match="runtime implementation differs"):
        gate._assert_source_authority(manifest)


def test_source_authority_rejects_missing_bundle_commit(monkeypatch):
    def git(arguments):
        if arguments == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="b" * 40 + "\n")
        assert arguments[:2] == ["cat-file", "-e"]
        return SimpleNamespace(returncode=1, stdout="missing")

    monkeypatch.setattr(gate, "_run_git", git)
    with pytest.raises(gate.GateError, match="absent from this checkout"):
        gate._assert_source_authority({
            "source": {"commit": "a" * 40, "submodules": []}})


@pytest.mark.parametrize(
    "plant,diagnostic",
    [
        ("unstaged", "unstaged runtime/gate inputs"),
        ("staged", "staged runtime/gate inputs"),
        ("untracked", "untracked runtime inputs"),
        ("dirty-submodule", "submodule worktree is dirty"),
    ],
)
def test_source_authority_rejects_every_worktree_drift(
        monkeypatch, plant, diagnostic):
    source, head, submodule = "a" * 40, "b" * 40, "c" * 40
    manifest = {"source": {"commit": source, "submodules": [
        {"path": "third_party/actlize", "commit": submodule}]}}

    def git(arguments):
        if arguments == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=head + "\n")
        if arguments[:2] == ["cat-file", "-e"]:
            return SimpleNamespace(returncode=0, stdout="")
        if arguments[:2] == ["diff", "--quiet"]:
            recorded_comparison = source in arguments
            assert (gate.RUNNER_INPUT in arguments) is not recorded_comparison
            return SimpleNamespace(
                returncode=(1 if plant == "unstaged" and not recorded_comparison else 0),
                stdout="")
        if arguments[:2] == ["ls-files", "--error-unmatch"]:
            return SimpleNamespace(returncode=0, stdout=gate.RUNNER_INPUT + "\n")
        if arguments[:2] == ["diff", "--cached"]:
            assert gate.RUNNER_INPUT in arguments
            return SimpleNamespace(
                returncode=1 if plant == "staged" else 0, stdout="")
        if arguments[:3] == ["ls-files", "--others", "--exclude-standard"]:
            return SimpleNamespace(
                returncode=0,
                stdout="quactlize/untracked.cpp\n" if plant == "untracked" else "")
        if arguments == ["submodule", "status", "--recursive"]:
            return SimpleNamespace(
                returncode=0,
                stdout=f" {submodule} third_party/actlize\n")
        assert arguments[:2] == ["-C", "third_party/actlize"]
        return SimpleNamespace(
            returncode=0,
            stdout=" M include/header.hpp\n" if plant == "dirty-submodule" else "")

    monkeypatch.setattr(gate, "_run_git", git)
    with pytest.raises(gate.GateError, match=diagnostic):
        gate._assert_source_authority(manifest)


def test_runtime_requires_one_explicit_visible_device_and_records_it():
    def write(value):
        def implementation(pointer):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = value
            return 0
        return implementation

    identity = gate._runtime_device_identity(
        write(1), write(0), lambda _operation, code: code == 0 or pytest.fail(),
        {"CUDA_VISIBLE_DEVICES": "7"})
    assert identity == {
        "CUDA_VISIBLE_DEVICES": "7",
        "hggc_device_count": 1,
        "hggc_current_device": 0,
    }
    with pytest.raises(gate.GateError, match="exactly one visible"):
        gate._runtime_device_identity(
            write(8), write(0), lambda _operation, _code: None,
            {"CUDA_VISIBLE_DEVICES": "0"})
    for environment in ({}, {"CUDA_VISIBLE_DEVICES": "0,1"},
                        {"CUDA_VISIBLE_DEVICES": "GPU-deadbeef"}):
        with pytest.raises(gate.GateError, match="exactly one numeric"):
            gate._visible_device_ordinal(environment)


def test_runtime_digest_mismatch_is_red(tmp_path, monkeypatch):
    runtime = tmp_path / "libhggc_wrapper.so"
    runtime.write_bytes(b"admitted runtime")
    monkeypatch.setattr(gate, "HGG_RUNTIME_SHA256", gate._sha256(runtime))
    assert gate._require_runtime_digest(runtime) == gate.HGG_RUNTIME_SHA256
    runtime.write_bytes(b"mutated runtime")
    with pytest.raises(gate.GateError, match="digest differs"):
        gate._require_runtime_digest(runtime)


def test_launch_status_checks_before_immediate_sync_and_deferred_errors():
    class Runtime:
        def __init__(self, errors, sync=0):
            self.errors = iter(errors)
            self.sync = sync

        def get_last_error(self):
            return next(self.errors)

        def synchronize(self):
            return self.sync

        @staticmethod
        def _error(code):
            return f"error-{code}"

    assert gate._checked_device_launch(
        Runtime([0, 0, 0]), "test", lambda: 0) == {
            "before": 0, "device_call": 0, "immediate": 0,
            "synchronize": 0, "deferred": 0,
        }
    with pytest.raises(gate.GateError, match="immediate"):
        gate._checked_device_launch(
            Runtime([0, 29, 0]), "test", lambda: 0)
    with pytest.raises(gate.GateError, match="deferred"):
        gate._checked_device_launch(
            Runtime([0, 0, 31]), "test", lambda: 0)


def test_ctypes_device_runner_uses_null_explicit_and_both_fault_plants():
    runtime = _FakeRuntime()
    spec = gate.FORMATS[0]
    arrangement = gate.ArrangementV2(*spec.expected_arrangement)

    def dense(_act, _low, high, units, output, m, n, k, qtype,
              _workspace, workspace_bytes, stream, config, _arrangement):
        assert (m, n, k, qtype) == (*gate.DENSE_SHAPE, spec.qtype)
        assert high is None and workspace_bytes == 32 and stream is None
        planted = not any(runtime.memory[units.value])
        runtime.store(output, np.full((m, n), 0 if planted else 3,
                                      dtype=np.float16))
        assert config in (None, spec.expected_dense_config.encode())
        return 0

    dense_library = SimpleNamespace(dense_device=_Function(dense))
    dense_activation = np.ones(
        (gate.DENSE_SHAPE[0], gate.DENSE_SHAPE[2]), dtype=np.float16)
    dense_outputs = gate._dense_launches(
        runtime, dense_library, spec, arrangement,
        np.arange(16, dtype=np.uint8), np.empty(0, dtype=np.uint8),
        np.ones(8, dtype=np.uint8), dense_activation,
        spec.expected_dense_config.encode(), 32)
    automatic, explicit, planted, dense_status = dense_outputs
    assert np.array_equal(automatic.view(np.uint16), explicit.view(np.uint16))
    assert np.all(automatic == 3) and np.all(planted == 0)
    assert [row["phase"] for row in dense_status] == [
        "null-config", "same-row-explicit-config", "zeroed-scale-unit-plant"]

    original_low = np.arange(16, dtype=np.uint8)
    planted_low = gate._plant_grouped_expert_zero(original_low, 4)

    def grouped(_act, low, high, _units, _offsets, output,
                total, n, k, experts, max_rows, qtype,
                _workspace, workspace_bytes, stream, config, _arrangement):
        assert (total, n, k, experts, max_rows, qtype) == (
            gate.GROUPED_SHAPE[0], gate.GROUPED_SHAPE[1],
            gate.GROUPED_SHAPE[2], gate.GROUPED_SHAPE[3],
            max(gate.GROUPED_ROWS), spec.qtype)
        assert high is None and workspace_bytes == 64 and stream is None
        rebound = bytes(runtime.memory[low.value]) == planted_low.tobytes()
        runtime.store(output, np.full((total, n), 0 if rebound else 5,
                                      dtype=np.float16))
        assert config in (None, gate.GROUPED_DEFAULT_NAME.encode())
        return 0

    grouped_library = SimpleNamespace(grouped_device=_Function(grouped))
    grouped_activation = np.ones(
        (gate.GROUPED_SHAPE[0], gate.GROUPED_SHAPE[2]), dtype=np.float16)
    grouped_outputs = gate._grouped_launches(
        runtime, grouped_library, spec, arrangement, original_low,
        np.empty(0, dtype=np.uint8), np.arange(8, dtype=np.uint8),
        grouped_activation, gate.GROUPED_DEFAULT_NAME.encode(), 64)
    automatic, explicit, planted, grouped_status = grouped_outputs
    assert np.array_equal(automatic.view(np.uint16), explicit.view(np.uint16))
    assert np.all(automatic == 5) and np.all(planted == 0)
    assert [row["phase"] for row in grouped_status] == [
        "null-config", "same-row-explicit-config", "expert0-rebind-plant"]


def test_evidence_publish_is_atomic_strict_json_and_never_overwrites(tmp_path):
    output = tmp_path / "result.json"
    gate._write_json(output, {"status": "PASS"})
    before = output.read_bytes()
    with pytest.raises(gate.GateError, match="cannot write gate evidence"):
        gate._write_json(output, {"status": "REPLACED"})
    assert output.read_bytes() == before
    assert not list(tmp_path.glob(".result.json.*"))
    with pytest.raises(gate.GateError, match="strict JSON"):
        gate._write_json(tmp_path / "bad.json", {"error": float("inf")})
    assert not (tmp_path / "bad.json").exists()

    directory = tmp_path / "evidence"
    directory.mkdir()
    with pytest.raises(gate.GateError, match="refusing to overwrite"):
        gate._create_output(directory)


def test_sdk_evidence_binds_runtime_bytes_and_visible_device(tmp_path):
    sdk = tmp_path / "sdk"
    (sdk / "bin").mkdir(parents=True)
    (sdk / "lib").mkdir()
    (sdk / "release.yaml").write_bytes(b"version: admitted\n")
    (sdk / "bin" / "hgobjdump").write_bytes(b"objdump")
    runtime_path = sdk / "lib" / "libhggc_wrapper.so"
    runtime_path.write_bytes(b"runtime bytes")
    device = {
        "CUDA_VISIBLE_DEVICES": "5",
        "hggc_device_count": 1,
        "hggc_current_device": 0,
    }
    evidence = gate._sdk_evidence(
        sdk, SimpleNamespace(path=runtime_path, device_identity=device), {
            "toolchain": {
                "sdk_release": "admitted",
                "sdk_archive_sha256": "a" * 64,
            },
        })
    assert evidence["archive_sha256"] == "a" * 64
    assert evidence["runtime_sha256"] == gate._sha256(runtime_path)
    assert evidence["release_receipt_sha256"] == gate._sha256(
        sdk / "release.yaml")
    assert evidence["device"] == device


def test_run_gate_verifies_elf_and_executes_all_five_without_building(
        tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = _manifest(bundle)
    output = tmp_path / "result"
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    calls = []

    def verify(root, *, sdk_root, inspect_binaries):
        assert root == bundle
        assert sdk_root == sdk
        assert inspect_binaries is True
        return manifest

    monkeypatch.setattr(gate.ppu_bundle, "verify_bundle", verify)
    monkeypatch.setattr(gate, "_assert_source_authority", lambda _manifest: {
        "bundle_source_commit": "a" * 40,
        "checkout_head": "b" * 40,
        "runner_sha256": "c" * 64,
    })
    monkeypatch.setattr(gate, "_dependency_versions", lambda: {"numpy": "test", "gguf": "0.19.0"})
    monkeypatch.setattr(
        gate, "HggcRuntime",
        lambda _sdk: SimpleNamespace(
            path=_sdk / "lib" / "libhggc_wrapper.so",
            device_identity={"CUDA_VISIBLE_DEVICES": "3",
                             "hggc_device_count": 1,
                             "hggc_current_device": 0}))
    monkeypatch.setattr(gate, "_sdk_evidence", lambda *_arguments: {
        "archive_sha256": "e" * 64,
        "runtime_sha256": "f" * 64,
        "device": {"CUDA_VISIBLE_DEVICES": "3",
                   "hggc_device_count": 1, "hggc_current_device": 0},
    })
    monkeypatch.setattr(
        gate, "_assert_default_library_identity",
        lambda path: {"path": str(path), "packed_format": -1})
    monkeypatch.setattr(
        gate, "_bind_format_library", lambda path: SimpleNamespace(path=path))

    def run_format(runtime, library, spec):
        assert runtime.path == sdk / "lib" / "libhggc_wrapper.so"
        assert library.path == bundle / next(
            role.filename for role in ppu_bundle.LIBRARY_ROLES
            if role.role == spec.role)
        calls.append(spec.name)
        return {"qtype": spec.qtype, "status": "PASS"}

    monkeypatch.setattr(gate, "_run_format_gate", run_format)
    result = gate.run_gate(bundle, sdk, output)
    assert calls == ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]
    assert result["status"] == "PASS"
    assert result["execution"] == {
        "device_library_builds": 0,
        "host_compilations": 0,
        "runner": "python-ctypes",
        "library_load_mode": "six DSOs, RTLD_LOCAL, one process",
    }
    assert set(result["formats"]) == set(calls)
    assert json.loads((output / "result.json").read_text()) == result
    assert (output / "bundle.json").is_file()
