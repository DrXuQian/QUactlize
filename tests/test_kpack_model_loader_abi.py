import json
import os
from pathlib import Path
import re
import subprocess

import pytest

from quactlize.runtime.compiler import sha
from tools.build_kpack_dispatch import attach_prefill
from tools.verify_kpack_dispatch import PREFILL_MODEL_EXPORTS, SMALLM_MODEL_EXPORTS, COMPUTE_MODEL_EXPORTS, require_exports

ROOT=Path(__file__).resolve().parents[1]
LLAMA=Path(os.environ.get('LLAMA_CI_DIR','/root/autodl-tmp/llama-v0.3.0'))
EXPORTS={name:required|COMPUTE_MODEL_EXPORTS[name] for name,required in SMALLM_MODEL_EXPORTS.items()}
EXPORTS['libquactlize_kpack_dispatch.so'] |= {'quactlize_kpack_dispatch_query_smallm_v3'}
EXPORTS['libquactlize_ppu_execution.so'] |= {'quactlize_kpack_q4_decode_select_v2', 'quactlize_kpack_q4_decode_run_v2'}
EXPORTS['libquactlize_ppu_prefill.so']=PREFILL_MODEL_EXPORTS
MISSING='quactlize_kpack_prefill_provider_image_v1'


def stub(path,names):
    source=path.with_suffix('.c')
    source.write_text('\n'.join('int '+name+'(void) {return 0;}' for name in sorted(names)))
    subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(path)],check=True)


@pytest.mark.parametrize('missing',[None,*sorted(PREFILL_MODEL_EXPORTS)])
def test_prefill_attach_rejects_self_consistent_old_build(tmp_path,missing):
    build,out,sdk=(tmp_path/x for x in ('build','out','sdk'))
    for p in (build,out,sdk/'lib'):p.mkdir(parents=True)
    library=build/'libquactlize_ppu_prefill.so'
    stub(library,PREFILL_MODEL_EXPORTS-{missing})
    (build/'manifest.json').write_text(json.dumps(dict(schema='quactlize.prefill-runtime.v1',
        library=library.name,sha256=sha(library),runtime={})))
    if missing:
        with pytest.raises(ValueError,match=missing):attach_prefill(out,build,sdk)
        assert not (out/library.name).exists()
    else:assert attach_prefill(out,build,sdk)['library_sha256']==sha(library)


def test_profile_covers_every_actual_caller_lookup():
    source=LLAMA/'ggml/src/ggml-cuda/quactlize-execution-lib.cu'
    if not source.exists():pytest.skip('companion caller source is not installed')
    text=source.read_text()
    for tag,name in [('host','libquactlize_kpack_dispatch.so'),('device','libquactlize_ppu_execution.so'),
                     ('prefill','libquactlize_ppu_prefill.so')]:
        requested=set(re.findall(r'QZ_BIND\([^,]+,\s*'+tag+r',\s*"([^"]+)"\)',text))
        requested.update(re.findall(r'dlsym\('+tag+r',\s*"([^"]+)"\)',text))
        assert requested==EXPORTS[name]


@pytest.fixture(scope='module')
def loader(tmp_path_factory):
    caller=LLAMA/'ggml/src/ggml-cuda/quactlize-execution-lib.cu'
    if not caller.exists():pytest.skip('companion caller source is not installed')
    binary=tmp_path_factory.mktemp('caller-loader')/'loader'
    subprocess.run(['g++','-std=c++17','-O1','-DGGML_NCP_QUACTLIZE',
        f'-I{LLAMA/"ggml/include"}',f'-I{LLAMA/"ggml/src"}',f'-I{caller.parent}',
        '-x','c++',str(caller),str(ROOT/'tests/kpack_model_loader_host.cpp'),'-ldl','-o',str(binary)],check=True)
    return binary


@pytest.mark.parametrize('missing',[None,MISSING])
def test_actual_loader_old_prefill_red_current_green(loader,tmp_path,missing):
    for name,required in EXPORTS.items():stub(tmp_path/name,required-{missing})
    result=subprocess.run([str(loader)],env=dict(os.environ,QUACTLIZE_KPACK_EXECUTION=str(tmp_path),
        QUACTLIZE_KPACK_JIT_HELPER='host-only-test'),capture_output=True,text=True)
    if missing:
        assert result.returncode==86 and 'native package missing '+missing in result.stderr
        with pytest.raises(ValueError,match=missing):require_exports(tmp_path/'libquactlize_ppu_prefill.so',PREFILL_MODEL_EXPORTS)
    else:
        assert result.returncode==0,result.stderr
        for name,required in EXPORTS.items():require_exports(tmp_path/name,required)
