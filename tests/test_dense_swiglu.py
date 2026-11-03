from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def test_dense_swiglu_geometry_and_alignment(tmp_path):
    binary=tmp_path/'dense-swiglu'
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-fsanitize=undefined',
                    '-I'+str(ROOT),str(ROOT/'tests/dense_swiglu_host.cpp'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)


def test_dense_only_wiring_and_build():
    source=(ROOT/'quactlize/ffn/ffn.cpp').read_text()
    assert 'if(a.kind==QUACTLIZE_FFN_DENSE) {\n        auto g=' in source
    assert 'a.merged?g+a.down.k:static_cast<float const*>(up.data)' in source
    assert 'else transition_rc=retain?moe::transition_snapshot' in source
    assert "'quactlize/ffn/dense_swiglu.cu'" in (ROOT/'tools/build_ppu_library.py').read_text()
    kernel=(ROOT/'quactlize/ffn/dense_swiglu.cu').read_text()
    assert 'namespace quactlize::ffn' in kernel and 'quactlize::moe' not in kernel
    assert '(gate/(1.f+expf(-gate)))*up' in kernel
    assert '__expf' not in kernel and '__half' not in kernel
