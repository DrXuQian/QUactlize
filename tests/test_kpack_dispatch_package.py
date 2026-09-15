import ctypes as C
import json
import os
from pathlib import Path
import pytest
from quactlize.dispatch.native import Request, Choice
from tools.verify_kpack_dispatch import verify
from tools.verify_kpack_dispatch import prefill_paths
from tools.build_kpack_dispatch import attach_prefill
from quactlize.runtime.compiler import sha
from tests import test_kpack_native_dispatch as host_tests


def test_ctypes_agrees_with_host_binding_tests():
    assert C.sizeof(Request) == C.sizeof(host_tests.Request)
    assert C.sizeof(Choice) == C.sizeof(host_tests.Choice)


def test_missing_package_does_not_pass(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify(tmp_path)


def test_llama_headers_are_exact_copies():
    from tools.sync_kpack_llama_headers import INCLUDES
    root = Path(__file__).resolve().parents[1]
    llama = Path(os.environ.get('LLAMA_SOURCE', '/root/autodl-tmp/llama-v0.3.0')) / 'ggml/src/ggml-cuda/quactlize'
    if not llama.exists():
        pytest.skip("external llama checkout not available")
    for source, target in (
        (root / "quactlize/runtime/abi.h", "kpack_module.h"),
        (root / "quactlize/execution/api.h", "kpack_execution.h"),
        (root / "quactlize/dispatch/api.h", "kpack_dispatch.h"),
    ):
        text = source.read_text()
        for before, after in INCLUDES.items():
            text = text.replace('"'+before+'"', '"'+after+'"')
        assert text == (llama / target).read_text()


@pytest.mark.parametrize('fault', [None, 'image', 'receipt', 'helper', 'sdk', 'symlink'])
def test_optional_prefill_closure_is_bound_to_build_and_helper(tmp_path, fault):
    source = tmp_path/'build'; source.mkdir()
    output = tmp_path/'package'; output.mkdir()
    sdk = tmp_path/'sdk'; (sdk/'lib').mkdir(parents=True)
    (sdk/'lib/libhggc_wrapper.so').write_bytes(b'sdk-runtime')
    library = source/'libquactlize_ppu_prefill.so'; library.write_bytes(b'host-test-image')
    (source/'manifest.json').write_text(json.dumps(dict(schema='quactlize.prefill-runtime.v1',
        library=library.name, sha256=sha(library), runtime={'libhggc_wrapper.so':sha(sdk/'lib/libhggc_wrapper.so')})))
    receipt = attach_prefill(output, source, sdk)
    if fault in ('image', 'receipt', 'helper'):
        field = 'library' if fault=='image' else fault
        (output/receipt[field]).write_bytes(b'changed')
    elif fault=='sdk':
        (sdk/'lib/libhggc_wrapper.so').write_bytes(b'wrong-runtime')
    elif fault=='symlink':
        image = output/receipt['library']
        image.rename(output/'moved.so'); image.symlink_to(output/'moved.so')
    if fault:
        with pytest.raises(ValueError):
            prefill_paths(output, receipt, sdk=sdk)
    else:
        assert set(prefill_paths(output, receipt, sdk=sdk)) == {
            'libquactlize_ppu_prefill.so', 'prefill-runtime.json', 'kpack_deepgemm_prewarm.py'}
