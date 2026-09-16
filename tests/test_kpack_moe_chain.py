from pathlib import Path
import os
import subprocess
import json
import pytest

ROOT=Path(__file__).resolve().parents[1]
LLAMA=Path(os.environ.get('LLAMA_CI_DIR','/root/llama.cpp'))


def test_mixed_dispatch_phases_and_pointer_contract(tmp_path):
    (tmp_path/'catalog.inc').write_text('static std::vector<Image> const kImages{};\nstatic char const kJitSource[]="";\n')
    exe=tmp_path/'mixed'
    subprocess.run(['g++','-std=c++17','-O1','-pthread',f'-I{ROOT}',f'-I{tmp_path}',
        str(ROOT/'tests/kpack_moe_mixed_host.cpp'),'-ldl','-o',str(exe)],check=True)
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'KPACK_MOE_MIXED_HOST PASS' in result.stdout
    assert 'KPACK_MOE_BF16_Q4_HOST PASS chains=96 typed-only all masks tokens=1..8' in result.stdout
    assert 'KPACK_MOE_REUSE_HOST PASS chains=1920 formats=6 tokens=1..8 splits=1/2/4/8' in result.stdout


def test_composition_and_exact_ggml_graph(tmp_path):
    exe=tmp_path/'moe-host'
    library=Path(os.environ.get('LLAMA_HOST_LIB_DIR','/root/autodl-tmp/q8-kpack2-llama-host/bin'))
    caller=(LLAMA/'ggml/src/ggml-cuda/ggml-cuda.cu').read_text()
    memory_check=caller.split('static bool ggml_cuda_check_fusion_memory_ranges(',1)[1].split(
        '\nstatic bool ggml_cuda_can_fuse(',1)[0]
    source=tmp_path/'moe-host.cpp'
    source.write_text('#include "ggml-impl.h"\n#include "ggml-backend-impl.h"\n'
        'static bool ggml_cuda_check_fusion_memory_ranges('+memory_check+
        '\n#include "'+str(ROOT/'tests/kpack_moe_chain_host.cpp')+'"\n')
    result=subprocess.run(['g++','-std=c++17','-O2',f'-I{ROOT}',
        f'-I{LLAMA/"ggml/include"}',f'-I{LLAMA/"ggml/src"}',
        str(source),f'-L{library}',
        f'-Wl,-rpath,{library}','-lggml-base','-o',str(exe)],text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    result=subprocess.run([str(exe)],text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'KPACK_MOE_COMPOSITION PASS' in result.stdout
    assert 'KPACK_MOE_GRAPH PASS' in result.stdout
    assert 'KPACK_MOE_GRAPH_TOKEN_SCOPE PASS' in result.stdout
    assert 'KPACK_MOE_FINISH_GRAPH PASS' in result.stdout
    assert 'KPACK_MOE_ROUTER_FINISH_GRAPH PASS' in result.stdout
    assert 'KPACK_MOE_ROUTER_MEMORY PASS' in result.stdout
    assert 'span.count,outputs,3,false,logits)' in caller


def test_moe_graph_mirror():
    assert (ROOT/'quactlize/integrations/llama/moe_graph.hpp').read_bytes()==(
        LLAMA/'ggml/src/ggml-cuda/quactlize/moe_graph.hpp').read_bytes()


def test_chain_is_explicit_and_capture_run_does_not_prepare():
    binding=(ROOT/'quactlize/dispatch/binding.cpp').read_text()
    run=binding.split('int quactlize_kpack_dispatch_moe_run_v1(')[1].split(
        'extern "C" void quactlize_kpack_dispatch_moe_destroy_v1')[0]
    assert 'QK_MOE_PREPARE' in run and 'QK_MOE_ACTIVATE' in run
    assert 'query' not in run and 'prepare(' not in run and 'Memcpy' not in run
    adapter=(LLAMA/'ggml/src/ggml-cuda/quactlize-execution.cu').read_text()
    assert 'prepare_moe(ctx,graph,start,false)' in adapter
    assert 'owner.private_storage(head + p->choice.workspace_bytes)' in adapter


def test_router_dispatcher_refresh_is_host_only():
    from tools.refresh_kpack_model_execution import dispatcher_refresh_scope
    assert dispatcher_refresh_scope(['quactlize/dispatch/moe.hpp']) == 'M1_ROUTER_SNAPSHOT_ALIAS_ADMISSION'
    for changes, caller in (([], False), (['quactlize/runtime/moe_chain.cuh'], False),
                            (['quactlize/dispatch/moe.hpp', 'quactlize/dispatch/policy.hpp'], False),
                            (['quactlize/dispatch/moe.hpp'], True)):
        with pytest.raises(ValueError):
            dispatcher_refresh_scope(changes, caller)


def test_router_alias_gate_rejects_stale_binary_and_wrong_scope(tmp_path):
    from tools.verify_kpack_dispatch import router_alias_paths
    from quactlize.runtime.compiler import sha
    gate=tmp_path/'router-alias';gate.mkdir()
    binary=gate/'bench';binary.write_bytes(b'isolated-test-payload')
    manifest=gate/'manifest.json'
    manifest.write_text(json.dumps(dict(platform='ppu',binary_sha256=sha(binary))))
    receipt=dict(path='router-alias/manifest.json',sha256=sha(manifest),binary_sha256=sha(binary),
                 cases=360,device_validated=False)
    assert router_alias_paths(tmp_path,receipt)==['router-alias/manifest.json','router-alias/bench']
    for edit in (dict(cases=359),dict(device_validated=True),dict(path='../manifest.json')):
        with pytest.raises(ValueError):
            router_alias_paths(tmp_path,receipt|edit)
    binary.write_bytes(b'changed')
    with pytest.raises(ValueError):
        router_alias_paths(tmp_path,receipt)
