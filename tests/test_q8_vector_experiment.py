from dev.gemv_simt.q8_vector_access import pattern
from dev.gemv_simt.q8_vector_build import source
from quactlize.execution.simt_codegen import inventory
import pytest


def test_q8_inventory_keeps_all_incumbents_and_compute_types():
    text=source()
    for c in inventory(8):
        for arm in range(2):
            for compute in range(2):
                assert f'invoke<{arm},{compute},{c.variant},{c.columns},{c.warps},{c.values}>' in text
    assert 'd->call.input_type!=QKG_F32' in text
    assert 'query_v2(*d,*f,&a,sizes)' in text
    assert 'register_reuse_reduce<8>' in text


def test_vector_metadata_footprint_and_minimum_abi_alignment():
    for c in inventory(8):
        aligned=pattern(c,256,512)
        assert aligned['metadata_vectorized']
        assert aligned['candidate_metadata_requests']==1
        v=next(s for s in aligned['streams'] if s['name']=='metadata-vector')
        assert v['width_bytes']==2*c.values
        for lane,address in zip(v['lanes'],v['addresses']):
            group=aligned['lane_groups'][lane]
            column=aligned['lane_columns'][lane]
            assert address==2*(group*256+column)
        weak=pattern(c,256,512,bases=dict(A=0,low=0,high=0,units=2))
        assert not weak['metadata_vectorized']
        assert weak['candidate_metadata_requests']==c.values
        assert aligned['packed_b_live_words_per_thread']['candidate']*2==aligned['packed_b_live_words_per_thread']['baseline']


def test_same_logical_k_order_and_exhaustive_two_code_values():
    baseline=[s*8+h*4+r for h in range(2) for s in range(4) for r in range(4)]
    candidate=[seg*16+s*8+h*4+r for h in range(2) for seg in range(2) for s in range(2) for r in range(4)]
    assert baseline==candidate
    assert sorted(candidate)==list(range(32))
    for slot in (0,1):
        for x in range(256):
            for y in range(256):
                word=(x|(y<<16))<<(8*slot)
                bits=((word>>(8*slot))&0x00ff00ff)|0x64006400
                # Half exponent 0x6400 has an exact unit-quantized mantissa.
                assert (bits&1023)-128==x-128
                assert ((bits>>16)&1023)-128==y-128


def test_missing_sdk_l2_requires_verified_override():
    from dev.gemv_simt.q8_vector_run import l2_identity
    assert l2_identity(dict(l2_bytes=0))['l2_source']=='UNAVAILABLE'
    assert l2_identity(dict(l2_bytes=0),64*1024**2)['l2_bytes']==64*1024**2
    assert l2_identity(dict(l2_bytes=48*1024**2))['l2_source']=='RUNTIME_QUERY'
    with pytest.raises(ValueError):l2_identity(dict(l2_bytes=48*1024**2),64*1024**2)


def test_ppu_link_retains_runtime_without_preload(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from dev.gemv_simt.q8_vector_build import link_command
    from quactlize.runtime import compiler
    monkeypatch.setattr(compiler, "LIBRARIES", ("example_runtime",))
    lib = tmp_path / "lib"
    lib.mkdir()
    runtime, caller, obj = tmp_path / "runtime.cpp", tmp_path / "caller.cpp", tmp_path / "caller.o"
    runtime.write_text('extern "C" int launch_runtime(){return 17;}\n')
    caller.write_text('extern "C" int launch_runtime();\nextern "C" int probe(){return launch_runtime();}\n')
    subprocess.run(["g++", "-shared", "-fPIC", runtime, "-o", lib / "libexample_runtime.so"], check=True)
    subprocess.run(["g++", "-c", "-fPIC", caller, "-o", obj], check=True)
    command = link_command("ppu", tmp_path, "unused", [obj], tmp_path / "good.so")
    subprocess.run(command, check=True)
    subprocess.run(["g++", "-shared", "-Wl,--as-needed", f"-L{lib}", "-lexample_runtime",
                    obj, "-o", tmp_path / "bad.so"], check=True)
    code = 'import ctypes,sys; assert ctypes.CDLL(sys.argv[1]).probe()==17'
    env = dict(os.environ, LD_LIBRARY_PATH=str(lib))
    for name, succeeds in (("good.so", True), ("bad.so", False)):
        result = subprocess.run([sys.executable, "-c", code, str(tmp_path / name)],
                                env=env, text=True, capture_output=True)
        assert (result.returncode == 0) == succeeds, result.stderr
        if not succeeds:
            assert "undefined symbol: launch_runtime" in result.stderr
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(["g++", "-shared", "-Wl,-z,defs", obj, "-o", tmp_path / "missing.so"],
                       capture_output=True, check=True)


def test_q8_runner_loads_sdk_before_module_and_closes_on_load_failure(tmp_path, monkeypatch):
    import json
    import sys
    from dev.gemv_simt import q8_vector_run as runner
    (tmp_path / "manifest.json").write_text(json.dumps(dict(platform="ppu")))
    calls = []
    class Runtime:
        def __init__(self, sdk, platform):
            assert platform == "ppu"
            calls.append("runtime")
        def close(self):
            calls.append("close")
    def library(*args):
        calls.append("module")
        raise OSError("planted module failure")
    monkeypatch.setattr(runner, "Runtime", Runtime)
    monkeypatch.setattr(runner, "Library", library)
    monkeypatch.setattr(sys, "argv", ["gate", "--bundle", str(tmp_path),
                                     "--sdk", str(tmp_path), "--output", str(tmp_path / "result.json")])
    with pytest.raises(OSError, match="planted module failure"):
        runner.main()
    assert calls == ["runtime", "module", "close"]
