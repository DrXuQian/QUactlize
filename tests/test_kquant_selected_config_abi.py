from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path
import sys

import pytest

from quactlize import ppu_bundle


ORACLE_PATH = Path(__file__).resolve().parents[1] / "tools" / "verify_kquant_selected_config.py"
ORACLE_SPEC = importlib.util.spec_from_file_location(
    "quactlize_verify_kquant_selected_config", ORACLE_PATH)
assert ORACLE_SPEC is not None and ORACLE_SPEC.loader is not None
oracle = importlib.util.module_from_spec(ORACLE_SPEC)
sys.modules[ORACLE_SPEC.name] = oracle
ORACLE_SPEC.loader.exec_module(oracle)


class _Function:
    def __init__(self, implementation):
        self.implementation = implementation
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.implementation(*args)


_STRING_STORAGE = []


def _write_config(pointer, ctype, values):
    out = ctypes.cast(pointer, ctypes.POINTER(ctype)).contents
    ctypes.memset(ctypes.byref(out), 0, ctypes.sizeof(out))
    for name, value in values.items():
        if name == "name":
            storage = ctypes.create_string_buffer(value)
            _STRING_STORAGE.append(storage)
            value = ctypes.cast(storage, ctypes.c_char_p)
        setattr(out, name, value)


class _Fmt2PolicyLibrary:
    _name = "libquactlize_ppu_fmt2.so"

    def __init__(
            self, *, exact_name=b"32x32:16x16:s3", qtype=10,
            arrangement=(2, 2, 2, 0, 0, 128, 16, 0,
                         0x514B504B54000001)):
        self._exact_name = exact_name
        self._qtype = qtype
        self._arrangement = arrangement
        self.quactlize_ppu_canonical_arrangement_v2 = _Function(self._canonical)
        self.quactlize_ppu_dense_fully_quantized_any_m_valid_for_arrangement_v2 = (
            _Function(self._dense_any_m))
        self.quactlize_ppu_dense_fully_quantized_selected_config_for_arrangement_v2 = (
            _Function(self._dense))
        self.quactlize_ppu_grouped_fully_quantized_selected_config_for_arrangement_v2 = (
            _Function(self._grouped))
        self.quactlize_ppu_grouped_fully_quantized_any_m_valid_for_arrangement_v2 = (
            _Function(self._grouped_any_m))

    def _canonical(self, qtype, output):
        if self._qtype is None or qtype != self._qtype:
            return 29
        out = ctypes.cast(output, ctypes.POINTER(oracle.ArrangementV2)).contents
        for (name, _ctype), value in zip(
                oracle.ArrangementV2._fields_, self._arrangement):
            setattr(out, name, value)
        return 0

    def _dense(self, output, m, n, k, group_size, qtype, arrangement, requested):
        del n, k, group_size, qtype, arrangement
        if requested == b"stale-config":
            ctypes.memset(output, 0, ctypes.sizeof(oracle.ConfigV4))
            return 0
        if requested:
            expected = oracle.COMPILED_DECODE_DEFAULT
        elif m == 1:
            expected = oracle.ExpectedConfig(
                self._exact_name.decode(), 32, 32, 256, 0, 16, 16, 3, 1)
        else:
            expected = oracle.COMPILED_DECODE_DEFAULT
        _write_config(output, oracle.ConfigV4, {
            "enable_cuda_kernel": False,
            "name": expected.name.encode(),
            "tile_m": expected.tile_m,
            "tile_n": expected.tile_n,
            "tactic_tile_k": expected.tactic_tile_k,
            "artifact_tile_k": expected.artifact_tile_k,
            "warp_m": expected.warp_m,
            "warp_n": expected.warp_n,
            "stages": expected.stages,
            "split_k_slices": expected.split_k_slices,
        })
        return 1

    @staticmethod
    def _grouped(output, total_rows, n, k, group_size, experts, max_rows,
                 qtype, arrangement, requested):
        del total_rows, n, k, group_size, experts, max_rows, qtype, arrangement
        if requested:
            return 0
        expected = oracle.GROUPED_DEFAULT
        _write_config(output, oracle.ConfigV3, {
            "enable_cuda_kernel": False,
            "name": expected.name.encode(),
            "tile_m": expected.tile_m,
            "tile_n": expected.tile_n,
            "tactic_tile_k": expected.tactic_tile_k,
            "artifact_tile_k": expected.artifact_tile_k,
            "warp_m": expected.warp_m,
            "warp_n": expected.warp_n,
            "stages": expected.stages,
        })
        return 1

    def _matches(self, qtype, arrangement):
        if not arrangement or self._qtype is None or qtype != self._qtype:
            return False
        value = ctypes.cast(
            arrangement, ctypes.POINTER(oracle.ArrangementV2)).contents
        return oracle._arrangement_tuple(value) == self._arrangement

    def _grouped_any_m(self, n, k, experts, qtype, arrangement):
        if n <= 0 or k <= 0 or experts <= 0:
            return 0
        return int(self._matches(qtype, arrangement))

    def _dense_any_m(self, n, k, qtype, arrangement):
        if n <= 0 or k <= 0:
            return 0
        return int(self._matches(qtype, arrangement))


def _policy_libraries(fmt2=None):
    libraries = {
        "default": _Fmt2PolicyLibrary(qtype=None, arrangement=None),
    }
    for role, qtype, arrangement in oracle.ANY_M_FORMATS:
        libraries[role] = _Fmt2PolicyLibrary(
            qtype=qtype, arrangement=arrangement)
    if fmt2 is not None:
        libraries["fmt2"] = fmt2
    return libraries


def test_current_runtime_bundle_contract_requires_selected_config_exports():
    assert ppu_bundle.REQUIRED_EXPORTS == (
        ppu_bundle.LEGACY_REQUIRED_EXPORTS |
        ppu_bundle.SELECTED_CONFIG_REQUIRED_EXPORTS |
        ppu_bundle.ANY_M_REQUIRED_EXPORTS)
    assert ppu_bundle.SELECTED_CONFIG_REQUIRED_EXPORTS == {
        "quactlize_ppu_dense_fully_quantized_selected_config_for_arrangement_v2",
        "quactlize_ppu_grouped_fully_quantized_selected_config_for_arrangement_v2",
    }
    assert ppu_bundle.ANY_M_REQUIRED_EXPORTS == {
        "quactlize_ppu_dense_fully_quantized_any_m_valid_for_arrangement_v2",
        "quactlize_ppu_grouped_fully_quantized_any_m_valid_for_arrangement_v2",
    }


def test_ctypes_records_match_public_x86_64_abi():
    oracle._require_layouts()


def test_selected_config_oracle_covers_exact_default_override_stale_and_grouped(
        monkeypatch, tmp_path):
    library = _Fmt2PolicyLibrary()
    monkeypatch.setattr(
        oracle, "_load_libraries", lambda _bundle: _policy_libraries(library))
    assert oracle.verify_selected_config(tmp_path) == {
        "dense_exact": "32x32:16x16:s3",
        "dense_unmeasured": "8x128:8x32:s3",
        "dense_explicit": "8x128:8x32:s3",
        "dense_stale": "FAIL_CLOSED",
        "dense_any_m": "ALL_M_VALID",
        "grouped_default": "16x128:16x16:s2",
        "grouped_any_m": "ALL_M_VALID",
        "any_m_formats": "FMT0..FMT4_VALID",
        "any_m_default": "REJECTS_ALL",
    }


def test_selected_config_oracle_rejects_policy_drift(monkeypatch, tmp_path):
    library = _Fmt2PolicyLibrary(exact_name=b"8x128:8x32:s3")
    monkeypatch.setattr(
        oracle, "_load_libraries", lambda _bundle: _policy_libraries(library))
    with pytest.raises(oracle.OracleError, match="exact measured dense differs"):
        oracle.verify_selected_config(tmp_path)


@pytest.mark.parametrize("field", [
    "quactlize_ppu_dense_fully_quantized_any_m_valid_for_arrangement_v2",
    "quactlize_ppu_grouped_fully_quantized_any_m_valid_for_arrangement_v2",
])
@pytest.mark.parametrize("role", [
    "default", "fmt0", "fmt1", "fmt2", "fmt3", "fmt4",
])
def test_selected_config_oracle_rejects_permissive_any_m(
        monkeypatch, tmp_path, field, role):
    libraries = _policy_libraries()
    setattr(libraries[role], field, _Function(lambda *_arguments: 1))
    monkeypatch.setattr(oracle, "_load_libraries", lambda _bundle: libraries)
    with pytest.raises(
            oracle.OracleError,
            match="did not fail closed|default library admitted"):
        oracle.verify_selected_config(tmp_path)


@pytest.mark.parametrize("symbol", [
    "quactlize_ppu_dense_fully_quantized_selected_config_for_arrangement_v2",
    "quactlize_ppu_grouped_fully_quantized_selected_config_for_arrangement_v2",
    "quactlize_ppu_dense_fully_quantized_any_m_valid_for_arrangement_v2",
    "quactlize_ppu_grouped_fully_quantized_any_m_valid_for_arrangement_v2",
])
def test_missing_policy_symbol_is_called_legacy_not_current(symbol):
    class Legacy:
        _name = str(Path("libquactlize_ppu_fmt2.so"))

    with pytest.raises(oracle.OracleError, match="legacy runtime bundle"):
        oracle._symbol(Legacy(), symbol)
