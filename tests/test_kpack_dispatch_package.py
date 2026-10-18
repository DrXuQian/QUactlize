import ctypes as C
import json
import os
from pathlib import Path
import re
import subprocess
import pytest
from quactlize.dispatch.native import Request, Choice
from tools.verify_kpack_dispatch import verify
from tools.verify_kpack_dispatch import prefill_paths
from tools.verify_kpack_dispatch import PREFILL_MODEL_EXPORTS
from tools.build_kpack_dispatch import attach_prefill
from quactlize.runtime.compiler import sha
from tests import test_kpack_native_dispatch as host_tests


def test_ctypes_agrees_with_host_binding_tests():
    assert C.sizeof(Request) == C.sizeof(host_tests.Request)
    assert C.sizeof(Choice) == C.sizeof(host_tests.Choice)


def test_missing_package_does_not_pass(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify(tmp_path)


def test_explicit_llama_caller_abi_matches_library(tmp_path):
    # A caller may omit newer, unused additive functions and format its own
    # declarations. Compare compiler-checked ABI, not an arbitrary stale
    # checkout at a hardcoded /root path or byte-identical header comments.
    if not os.environ.get('LLAMA_SOURCE'):
        pytest.skip('set LLAMA_SOURCE to the full caller being admitted')
    from quactlize.dispatch.native import (PrefillChoice, ConfirmedPrefill, SmallmChoice,
        MatchedSmallmChoice, DeviceCall, GroupedCompute, GroupedMetadata)
    from quactlize.runtime.native import Call
    root = Path(__file__).resolve().parents[1]
    llama = Path(os.environ['LLAMA_SOURCE']) / 'ggml/src/ggml-cuda/quactlize'
    types={'qks_request_v1':Request,'qks_choice_v1':Choice,'qk_call_v1':Call,
        'qks_prefill_choice_v1':PrefillChoice,'qks_prefill_choice_v2':ConfirmedPrefill,
        'qks_smallm_choice_v1':SmallmChoice,'qks_smallm_choice_v2':MatchedSmallmChoice,
        'qk_device_call_v2':DeviceCall,'qk_compute_device_call_v3':GroupedCompute,
        'qk_compute_device_call_v4':GroupedMetadata}
    body=['#include <cstddef>', '#include <cstdio>', '#include <type_traits>',
        'static_assert(std::is_same_v<decltype(&quactlize_kpack_dispatch_prefill_v2),'
        'int(*)(qks_request_v1 const*,uint32_t,int32_t,int32_t,qks_prefill_choice_v2*)>);',
        'static_assert(std::is_same_v<decltype(&quactlize_kpack_dispatch_query_prefill_v1),'
        'int(*)(void*,qks_request_v1 const*,int32_t,int32_t,qks_choice_v1*)>);',
        'static_assert(QKS_DECODE_COMPONENT_MEASURED==18 && QKS_PREFILL_COMPONENT_MEASURED==19);',
        'int main() {']
    expected=[]
    for name,cls in types.items():
        body.append(f'printf("%zu %zu ",sizeof({name}),alignof({name}));')
        expected.extend([C.sizeof(cls),C.alignment(cls)])
        for field,_ in cls._fields_:
            body.append(f'printf("%zu ",offsetof({name},{field}));')
            expected.append(getattr(cls,field).offset)
    body.append('}')
    def layout(name,header):
        source=tmp_path/(name+'.cpp');binary=tmp_path/name
        source.write_text('#include "'+str(header)+'"\n'+'\n'.join(body))
        subprocess.run(['g++','-std=c++17','-I'+str(llama),str(source),'-o',str(binary)],check=True)
        return list(map(int,subprocess.check_output([str(binary)],text=True).split()))
    assert layout('provider',root/'quactlize/dispatch/api.h')==expected
    header=llama/'kpack_dispatch.h'
    assert layout('caller',header)==expected
    # Same-sized int fields exchanged is ABI drift despite unchanged sizeof.
    changed,count=re.subn(r'policy,\s*compute_type,\s*channels',
                         'channels, compute_type, policy',header.read_text())
    assert count==1
    bad=tmp_path/'bad-header.h';bad.write_text(changed)
    assert layout('bad-caller',bad)!=expected


@pytest.mark.parametrize('fault', [None, 'image', 'receipt', 'helper', 'sdk', 'symlink'])
def test_optional_prefill_closure_is_bound_to_build_and_helper(tmp_path, fault):
    source = tmp_path/'build'; source.mkdir()
    output = tmp_path/'package'; output.mkdir()
    sdk = tmp_path/'sdk'; (sdk/'lib').mkdir(parents=True)
    (sdk/'lib/libhggc_wrapper.so').write_bytes(b'sdk-runtime')
    library = source/'libquactlize_ppu_prefill.so'
    stub = source/'stub.c'
    stub.write_text('\n'.join('int '+name+'(void) {return 0;}' for name in sorted(PREFILL_MODEL_EXPORTS)))
    subprocess.run(['cc','-shared','-fPIC',str(stub),'-o',str(library)],check=True)
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


@pytest.mark.parametrize('fault', [None, 'canonical', 'pair', 'q8'])
def test_model_packer_requires_current_entries_and_formats(tmp_path, fault):
    from tools.verify_kpack_dispatch import pack_paths, PACK_MODEL_EXPORTS
    pack = tmp_path / 'pack'
    pack.mkdir()
    names = set(PACK_MODEL_EXPORTS)
    if fault in ('canonical', 'pair'):
        names.remove('quactlize_ppu_' + ('kpack_canonical_arrangement_v1' if fault == 'canonical' else
            'prepare_gate_up_dev_for_arrangement_v1'))
    source, library = pack / 'stub.c', pack / 'libquactlize_ppu_pack.so'
    source.write_text('\n'.join('int ' + name + '(void) {return 0;}' for name in sorted(names)))
    subprocess.run(['cc', '-shared', '-fPIC', str(source), '-o', str(library)], check=True)
    formats = [8, 10, 11, 12, 13, 14]
    manifest = pack / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema='quactlize.kpack-device-pack-build.v1',
        library=library.name, sha256=sha(library), formats=formats[1:] if fault == 'q8' else formats)))
    receipt = dict(schema='quactlize.kpack-producer-package.v1', files={
        'pack/' + p.name: sha(p) for p in (library, manifest)})
    if fault:
        with pytest.raises(ValueError):
            pack_paths(tmp_path, receipt, model_formats=formats)
    else:
        assert len(pack_paths(tmp_path, receipt, model_formats=formats)) == 2
