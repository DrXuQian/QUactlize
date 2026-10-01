"""Host-only tests of source-build transactions; no fake device admission."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import build as product


@pytest.mark.parametrize('script',[
    'build_kpack_pack.py', 'build_kpack_execution.py',
    'build_kpack_prefill_runtime.py', 'build_kpack_gate_up.py',
    'build_kpack_dispatch.py'])
def test_stage_command_line(script):
    help_text=subprocess.check_output(
        [sys.executable,str(product.ROOT/'tools'/script),'--help'],text=True)
    for argument in ('--sdk','--output','--jobs'):
        assert argument in help_text


@pytest.fixture
def build_driver(tmp_path,monkeypatch):
    sdk=tmp_path/'sdk';sdk.mkdir()
    args=argparse.Namespace(sdk=sdk,output=tmp_path/'out',jobs=3,resume=False)
    calls=[]
    monkeypatch.setattr(product,'source_identity',lambda:'test-source')
    monkeypatch.setattr(product,'sdk_identity',lambda sdk:{'test-sdk':'identity'})
    names={
        'build_kpack_pack.py':'libquactlize_ppu_pack.so',
        'build_kpack_execution.py':'libquactlize_ppu_execution.so',
        'build_kpack_prefill_runtime.py':'libquactlize_ppu_prefill.so',
        'build_kpack_gate_up.py':'libquactlize_ppu_gate_up.so',
        'build_kpack_dispatch.py':'libquactlize_kpack_dispatch.so',
    }
    def run(command,check):
        assert check
        script=Path(command[1]).name
        calls.append(script)
        assert command[command.index('--jobs')+1]=='3'
        output=Path(command[command.index('--output')+1]);output.mkdir()
        payload=output/names[script];payload.write_bytes(b'host-test-not-an-ELF')
        manifest=dict(library=payload.name,sha256=product.sha(payload),layout_id=5)
        (output/'manifest.json').write_text(json.dumps(manifest))
    def verify(runtime,sdk):
        manifest=product.read_json(runtime/'manifest.json')
        assert manifest['pack']['files']['pack/libquactlize_ppu_pack.so']
        assert manifest['paired_gate_up']['device_validated'] is False
        return manifest
    monkeypatch.setattr(product.subprocess,'run',run)
    monkeypatch.setattr(product,'verify',verify)
    return args,calls,verify


def test_resume_reuses_only_complete_stages(build_driver):
    args,calls,_=build_driver
    product.build(args)
    receipt=product.read_json(args.output/'build.json')
    assert receipt['complete'] and not receipt['device_validated']
    assert len(calls)==5
    args.resume=True
    product.build(args)
    assert len(calls)==5
    payload=args.output/receipt['stages']['pack']['path']/'libquactlize_ppu_pack.so'
    payload.write_bytes(b'modified')
    with pytest.raises(ValueError,match='payload changed'):
        product.build(args)


def test_failed_attachment_is_not_a_completed_stage(build_driver,monkeypatch):
    args,calls,verify=build_driver
    def fail(*args,**kwargs):
        raise ValueError('attachment rejected')
    monkeypatch.setattr(product,'verify',fail)
    with pytest.raises(ValueError,match='attachment rejected'):
        product.build(args)
    receipt=product.read_json(args.output/'build.json')
    assert 'runtime' not in receipt['stages']
    assert not receipt.get('complete')
    monkeypatch.setattr(product,'verify',verify)
    args.resume=True
    product.build(args)
    assert len(calls)==6
    assert product.read_json(args.output/'build.json')['runtime']=='runtime-2'


def test_resume_rejects_changed_source(build_driver,monkeypatch):
    args,calls,_=build_driver
    product.build(args)
    args.resume=True
    monkeypatch.setattr(product,'source_identity',lambda:'changed')
    with pytest.raises(ValueError,match='source or SDK changed'):
        product.build(args)
    assert len(calls)==5
