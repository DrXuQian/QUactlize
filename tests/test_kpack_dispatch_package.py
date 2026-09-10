import ctypes as C
from pathlib import Path
import pytest
from quactlize.dispatch.native import Request, Choice
from tools.verify_kpack_dispatch import verify
from tests import test_kpack_native_dispatch as host_tests


def test_ctypes_agrees_with_host_binding_tests():
    assert C.sizeof(Request) == C.sizeof(host_tests.Request)
    assert C.sizeof(Choice) == C.sizeof(host_tests.Choice)


def test_missing_package_does_not_pass(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify(tmp_path)


def test_llama_headers_are_exact_copies():
    root = Path(__file__).resolve().parents[1]
    llama = Path("/root/llama.cpp/ggml/src/ggml-cuda/quactlize")
    if not llama.exists():
        pytest.skip("external llama checkout not available")
    for source, target in (
        (root / "quactlize/runtime/abi.h", "kpack_module.h"),
        (root / "quactlize/execution/api.h", "kpack_execution.h"),
        (root / "quactlize/dispatch/api.h", "kpack_dispatch.h"),
    ):
        text = source.read_text().replace("../runtime/abi.h", "kpack_module.h").replace(
            "../integrations/llama/indexed.h", "kpack_indexed.h")
        assert text == (llama / target).read_text()
