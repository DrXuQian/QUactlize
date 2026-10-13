"""Relocatable header/JIT payload tests; fake SDK bytes are host-only."""
import os
from pathlib import Path
import subprocess
import sys

from quactlize.runtime.compiler import Compiler, LIBRARIES, source_contract
from tools.build_ppu_library import ROOT, stage_jit


def test_jit_snapshot_has_the_same_source_identity_without_checkout(tmp_path):
    sdk = tmp_path / 'host-identity-only-sdk'
    for name in ('bin/hgcc', 'bin/hgobjdump', *('lib/lib' + n + '.so' for n in LIBRARIES)):
        path = sdk / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'HOST TEST ONLY, NOT A RUNTIME OR COMPILER\n')
    compiler = Compiler(sdk, tmp_path / 'cache')
    snapshot = tmp_path / 'installed/quactlize-runtime/jit'
    stage_jit(compiler, snapshot)
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    command = [sys.executable, '-c',
        'from pathlib import Path; import sys; '
        'from quactlize.runtime.compiler import Compiler, source_contract; '
        'print(source_contract(Compiler(Path(sys.argv[1]),Path(sys.argv[2])).identity)); '
        'assert "torch" not in sys.modules', str(sdk), str(tmp_path / 'snapshot-cache')]
    actual = subprocess.check_output(command, cwd=snapshot, env=env, text=True).strip()
    assert actual == source_contract(compiler.identity)
    subprocess.run([sys.executable, str(snapshot / 'tools/kpack_jit.py'), '--help'],
                   cwd=tmp_path, env=env, check=True, stdout=subprocess.PIPE)
    assert (snapshot / 'third_party/actlize/LICENSE.txt').is_file()


def test_exported_headers_need_no_private_source_include(tmp_path):
    import shutil
    for source, target in (
            ('quactlize/dense/api.h', 'quactlize/dense/api.h'),
            ('quactlize/packing/api.h', 'quactlize/packing/api.h'),
            ('quactlize/include/quactlize_ppu_config.h', 'quactlize_ppu_config.h')):
        output = tmp_path / 'include' / target
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, output)
    for compiler, language in [('gcc', 'c'), ('g++', 'c++')]:
        subprocess.run([compiler, '-x', language, '-fsyntax-only', '-I' + str(tmp_path / 'include'), '-'],
            cwd=tmp_path, input='#include "quactlize/dense/api.h"\n#include "quactlize/packing/api.h"\n',
            text=True, check=True)
