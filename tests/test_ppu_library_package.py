"""Relocatable header/JIT payload tests; fake SDK bytes are host-only."""
import os
import copy
import json
from pathlib import Path
import subprocess
import sys
import pytest

from quactlize.runtime.compiler import Compiler, LIBRARIES, source_contract, validate_actlize_headers
from tools.build_ppu_library import ROOT, stage_jit, validate_actlize_checkout


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
    # Builder .inl files used to be copied but omitted from the fingerprint.
    builder = snapshot / 'quactlize/include/actlize_extensions/cutlass/gemm/collective/builders/quactlize_mma_builder.inl'
    builder.write_text(builder.read_text() + '\n// regression: changed builder\n')
    changed = subprocess.check_output(command, cwd=snapshot, env=env, text=True).strip()
    assert changed != actual


def test_installed_snapshot_rejects_missing_bf16_atom_before_compiling(tmp_path):
    import shutil
    paths = ('cute/arch/mma_ppu0010.hpp', 'cute/atom/mma_traits_ppu0010.hpp',
             'cutlass/gemm/config/gemm_operands.hpp')
    for name in paths:
        dst = tmp_path / 'third_party/actlize/include' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'third_party/actlize/include' / name, dst)
    validate_actlize_headers(tmp_path)
    for name in paths:
        dst = tmp_path / 'third_party/actlize/include' / name
        original = dst.read_text()
        dst.write_text(original.replace('PPU0010_8x16x16_F32BF16BF16F32_TN', 'OLD_HEADER_NO_BF16_M8'))
        with pytest.raises(ValueError, match='incompatible actlize headers.*BF16 M8'):
            validate_actlize_headers(tmp_path)
        dst.write_text(original)


def test_library_build_checks_exact_gitlink_and_dirty_vendor_headers(tmp_path):
    import shutil
    root = tmp_path / 'source'
    vendor = root / 'third_party/actlize'
    vendor.mkdir(parents=True)

    def git(directory, *args):
        return subprocess.check_output(['git', '-C', str(directory), '-c', 'user.name=Test',
            '-c', 'user.email=test@example.invalid', *args], text=True).strip()

    git(root, 'init', '-q')
    git(vendor, 'init', '-q')
    for name in ('cute/arch/mma_ppu0010.hpp', 'cute/atom/mma_traits_ppu0010.hpp',
                 'cutlass/gemm/config/gemm_operands.hpp'):
        dst = vendor / 'include' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'third_party/actlize/include' / name, dst)
    git(vendor, 'add', 'include')
    git(vendor, 'commit', '-qm', 'vendor')
    commit = git(vendor, 'rev-parse', 'HEAD')
    git(root, 'update-index', '--add', '--cacheinfo', '160000', commit, 'third_party/actlize')
    git(root, 'commit', '-qm', 'pin')
    validate_actlize_checkout(root)
    header = vendor / 'include/cute/arch/mma_ppu0010.hpp'
    header.write_text(header.read_text() + '\n// dirty vendor\n')
    with pytest.raises(ValueError, match='headers differ from its checked-out commit'):
        validate_actlize_checkout(root)
    git(vendor, 'add', 'include')
    git(vendor, 'commit', '-qm', 'another vendor commit')
    with pytest.raises(ValueError, match='checkout differs from Quactlize gitlink'):
        validate_actlize_checkout(root)


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


@pytest.mark.parametrize('fault',[None,'header','payload','table'])
def test_dense_prefill_receipt_binds_generated_runtime_selection(tmp_path,fault):
    from tools.build_ppu_library import PUBLIC_EXPORTS,verify
    from quactlize.dispatch.dense_prefill import EVIDENCE,HEADER
    from quactlize.runtime.compiler import sha
    source=tmp_path/'stub.c';library=tmp_path/'libquactlize_ppu.so'
    source.write_text('\n'.join('int '+name+'(void){return 0;}' for name in PUBLIC_EXPORTS))
    subprocess.run(['cc','-shared','-fPIC',source,'-o',library],check=True)
    path=tmp_path/'quactlize-runtime/dense-prefill.json';path.parent.mkdir()
    path.write_bytes(EVIDENCE.read_bytes())
    if fault=='table':
        data=json.loads(path.read_text());data['points'][0]['us']*=2
        path.write_text(json.dumps(data))
    m=dict(schema='quactlize.ppu-library.v1',
        payloads={str(p.relative_to(tmp_path)):sha(p) for p in (library,path)},
        source_hashes={'policies/dense_prefill_v1.json':sha(path),
                      'quactlize/dispatch/dense_prefill.inc':sha(HEADER)},
        dense_prefill=dict(path=str(path.relative_to(tmp_path)),header_sha256=sha(HEADER)))
    if fault=='header':m['dense_prefill']['header_sha256']='b'*64
    if fault=='payload':m['source_hashes']['policies/dense_prefill_v1.json']='b'*64
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    if fault:
        with pytest.raises(ValueError):verify(tmp_path)
    else:
        assert verify(tmp_path)['dense_prefill']==m['dense_prefill']


@pytest.mark.parametrize('fault',[None,'policy','private-policy','image','full-table','private-full-table','full-evidence'])
def test_public_dense_helper_receipt_binds_policy_and_private_runtime(tmp_path,fault):
    from tools.build_ppu_library import PUBLIC_EXPORTS,verify
    from quactlize.runtime.compiler import sha
    source=tmp_path/'host.c';library=tmp_path/'libquactlize_ppu.so'
    source.write_text('\n'.join('int '+name+'(void){return 0;}' for name in PUBLIC_EXPORTS))
    subprocess.run(['cc','-shared','-fPIC',source,'-o',library],check=True)
    runtime=tmp_path/'quactlize-runtime';runtime.mkdir()
    image=runtime/'libquactlize_ppu_prefill.so';image.write_bytes(b'HOST RECEIPT TEST ONLY')
    header='quactlize/prefill/dense_helpers.hpp'
    table='quactlize/prefill/dense_full_helpers.inc'
    evidence='policies/dense_full_helpers_20261023.json'
    receipt=runtime/'prefill-runtime.json'
    data=dict(source_hashes={header:'a'*64,table:'c'*64},sha256=sha(image))
    if fault=='private-full-table':data['source_hashes'][table]='b'*64
    if fault=='private-policy':data['source_hashes'][header]='b'*64
    if fault=='image':data['sha256']='b'*64
    receipt.write_text(json.dumps(data))
    m=dict(schema='quactlize.ppu-library.v1',source_hashes={header:'a'*64,table:'c'*64,evidence:'d'*64},
        payloads={str(p.relative_to(tmp_path)):sha(p) for p in (library,image,receipt)},
        dense_helpers=dict(policy_sha256='b'*64 if fault=='policy' else 'a'*64,
                           full_table_sha256='b'*64 if fault=='full-table' else 'c'*64,
                           full_evidence_sha256='b'*64 if fault=='full-evidence' else 'd'*64,
                           runtime_receipt='quactlize-runtime/prefill-runtime.json'))
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    if fault:
        with pytest.raises(ValueError):verify(tmp_path)
    else:assert verify(tmp_path)['dense_helpers']==m['dense_helpers']


@pytest.mark.parametrize('fault', [None, 'missing-q4', 'stale-execution', 'missing-seal', 'capabilities'])
def test_public_library_binds_actual_selected_execution(tmp_path, fault):
    from tools.build_ppu_library import PUBLIC_EXPORTS, verify
    from quactlize.dispatch.planning import validate_inventory
    from quactlize.runtime.compiler import sha
    source=tmp_path/'stub.c';library=tmp_path/'libquactlize_ppu.so'
    source.write_text('\n'.join('int '+name+'(void) {return 0;}' for name in PUBLIC_EXPORTS))
    subprocess.run(['cc','-shared','-fPIC',source,'-o',library],check=True)
    path=tmp_path/'quactlize-runtime/final-selection.json';path.parent.mkdir()
    config=dict(reader=2,variant=7,warps=8,values=4,columns=4)
    plan=dict(schema='quactlize.final-selection.v1',requests=[dict(status=0,kind=2,
        request=[12,2,1024,2048,256,8,1,1,1],config=config)])
    path.write_text(json.dumps(plan))
    execution=dict(q4_decode_policy_sha256='host-test',smallm_effective_sha256='a'*64,
        q4_decode_configs={'1024x2048':[[2,7,8,4,4]]},q4_decode_compute_v2=dict(compute=['f16','bf16']))
    m=dict(schema='quactlize.ppu-library.v1',payloads={str(p.relative_to(tmp_path)):sha(p) for p in (path,library)},
        execution_receipt=copy.deepcopy(execution),source_hashes={'policies/kpack_smallm_effective_v1.hpp':'a'*64},
        final_selection=dict(path=str(path.relative_to(tmp_path)),capabilities=validate_inventory(plan,execution,jit=True)))
    if fault=='missing-q4':m['execution_receipt']['q4_decode_configs']={}
    elif fault=='stale-execution':m['execution_receipt']['smallm_effective_sha256']='b'*64
    elif fault=='missing-seal':m['payloads'].pop(str(path.relative_to(tmp_path)))
    elif fault=='capabilities':m['final_selection']['capabilities']['required']['q4']=[]
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    if fault:
        with pytest.raises(ValueError):verify(tmp_path)
    else:assert verify(tmp_path)['schema']==m['schema']
