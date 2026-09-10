from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
LLAMA=Path('/root/llama.cpp')


def test_composition_and_exact_ggml_graph(tmp_path):
    exe=tmp_path/'moe-host'
    library=Path('/root/autodl-tmp/q8-kpack2-llama-host/bin')
    result=subprocess.run(['g++','-std=c++17','-O2',f'-I{ROOT}',
        f'-I{LLAMA/"ggml/include"}',f'-I{LLAMA/"ggml/src"}',
        str(ROOT/'tests/kpack_moe_chain_host.cpp'),f'-L{library}',
        f'-Wl,-rpath,{library}','-lggml-base','-o',str(exe)],text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    result=subprocess.run([str(exe)],text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'KPACK_MOE_COMPOSITION PASS' in result.stdout
    assert 'KPACK_MOE_GRAPH PASS' in result.stdout
    assert 'KPACK_MOE_GRAPH_TOKEN_SCOPE PASS' in result.stdout


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
