"""Real llama policy parser and prepared adapter source/compile contracts."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]
LLAMA=Path('/root/llama.cpp')


@pytest.fixture(scope='module')
def policy_probe(tmp_path_factory):
    target=tmp_path_factory.mktemp('simt-policy')/'probe'
    library=Path('/root/autodl-tmp/q8-kpack2-llama-host/bin')
    source=LLAMA/'ggml/src/ggml-cuda'
    subprocess.run(['g++','-std=c++17','-O2','-x','c++','-DGGML_NCP_QUACTLIZE',
        f'-I{source}',f'-I{LLAMA}/ggml/include',f'-I{LLAMA}/ggml/src',
        str(source/'quactlize-execution-lib.cu'),str(ROOT/'tests/kpack_simt_policy_host.cpp'),
        '-o',str(target),f'-L{library}',f'-Wl,-rpath,{library}','-lggml-base','-ldl'],check=True)
    return target


@pytest.mark.parametrize('negative',[None,'unknown-format','duplicate','input-quantized','missing'])
def test_exact_measured_q8_and_kquant_recipes(policy_probe,tmp_path,negative):
    lines=['KPACK_GEMV_POLICY_V1','8 512 2048 1 0 1 1 1 1 32 4 4','12 512 2048 1 0 1 1 1 1 16 8 1']
    if negative=='unknown-format': lines[1]=lines[1].replace('8 512','9 512',1)
    if negative=='duplicate': lines.append(lines[1])
    if negative=='input-quantized': lines[1]='8 512 2048 1 0 1 1 1 2 32 4 4'
    if negative=='missing': lines=lines[:1]
    policy=tmp_path/'policy.tsv'; policy.write_text('\n'.join(lines)+'\n')
    result=subprocess.run([str(policy_probe)],env=dict(os.environ,QUACTLIZE_KPACK_GEMV_POLICY=str(policy)),
        text=True,capture_output=True,timeout=15)
    assert (result.returncode==0)==(negative is None),result.stdout+result.stderr


def test_auto_queries_simt_before_tc_and_does_not_force_q8_to_tc():
    source=(LLAMA/'ggml/src/ggml-cuda/quactlize-execution.cu').read_text()
    assert 'if (mode == RouteMode::Auto || mode == RouteMode::Gemv)' in source
    assert source.index('p->direct = ggml_quactlize_gemv_config')<source.index('qks_request_v1 r{')
    assert 'mode == RouteMode::Gemv && art.qtype != GGML_TYPE_Q8_0' not in source
    # A SIMT member cannot be admitted as a grouped-handle chain.
    assert 'plans[j]->legacy || plans[j]->direct' in source
