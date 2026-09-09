import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_kpack_native_dispatch import build_stub, probe, query, Choice, Call
from quactlize.runtime.compiler import Compiler, sha
from tools.kpack_jit import parent_tuple, model_requests

ROOT = Path(__file__).resolve().parents[1]


class JitOptions(C.Structure):
    _fields_ = [("version", C.c_uint32), ("size", C.c_uint32)] + [
        (f, C.c_char_p) for f in ("python", "helper", "sdk", "cache")]


def enable(f, r, tmp, helper):
    options = JitOptions(1, C.sizeof(JitOptions), os.fsencode(sys.executable),
                         os.fsencode(helper), os.fsencode(tmp), os.fsencode(tmp / "modules"))
    return f["enable_jit"](r, C.byref(options))


def helper_file(tmp, body):
    path = tmp / "helper with spaces.py"
    path.write_text("import sys\nfrom pathlib import Path\n" + body)
    return path


def test_compile_import_does_not_load_torch():
    subprocess.run([sys.executable, "-c", "import sys; import quactlize.runtime.compiler; "
                    "assert 'torch' not in sys.modules"], cwd=ROOT, check=True)


def test_jit_only_resolves_selected_parent_once_and_run_never_compiles(tmp_path, probe):
    f, r, req = build_stub(tmp_path, probe, jit=True)
    receipt = f"QK_JIT_V1 {'a'*64} {'b'*64} {'d'*64}"
    helper = helper_file(tmp_path,
        f"count=Path({str(tmp_path / 'calls')!r})\n"
        "count.write_text(count.read_text()+'1' if count.exists() else '1')\n"
        f"assert sys.argv[sys.argv.index('--parent')+1]=={query(probe, [(12,2,528,3072,512,256,129)])[0].split()[0]!r}\n"
        f"print({receipt!r})\n")
    choice = Choice()
    assert f["query"](r, C.byref(req), C.byref(choice)) == 1  # No implicit JIT.
    assert enable(f, r, tmp_path, helper) == 0
    for _ in range(3):
        assert f["query"](r, C.byref(req), C.byref(choice)) == 0
    assert (tmp_path / "calls").read_text() == "1"
    call = Call(version=1, size=C.sizeof(Call), m=req.m, n=req.n, k=req.k,
        experts=req.experts, group_size=32, device=0, compute_units=72,
        mapping_id=req.mapping_id, a=4096, low=8192, metadata=12288, output=16384,
        offsets_device=20480, workspace=24576, workspace_bytes=256)
    handle = C.c_void_p()
    assert f["prepare"](r, C.byref(choice), C.byref(call), C.byref(handle)) == 0
    assert enable(f, r, tmp_path, helper) == 2
    f["close"](r)
    # Live handles retain the module after runtime destruction.
    for _ in range(10):
        assert f["run"](handle, None) == 0
    assert (tmp_path / "calls").read_text() == "1"
    f["destroy"](handle)


@pytest.mark.parametrize("body,reason", [
    ("sys.exit(9)\n", "JIT failed"),
    ("print('bad')\n", "receipt identity"),
    (f"print('QK_JIT_V1 {'a'*64} {'b'*64} extra')\n", "receipt identity"),
    (f"print('QK_JIT_V1 {'/'*64} {'b'*64}')\n", "receipt identity"),
    (f"print('QK_JIT_V1 {'a'*64} {'b'*64} {'e'*64}')\n", "receipt identity"),
    ("print('x'*10000)\n", "oversized"),
])
def test_jit_failures_do_not_return_a_tactic(tmp_path, probe, body, reason):
    f, r, req = build_stub(tmp_path, probe, jit=True)
    assert enable(f, r, tmp_path, helper_file(tmp_path, body)) == 0
    choice = Choice()
    assert f["query"](r, C.byref(req), C.byref(choice)) == 3
    assert reason.encode() in f["error"]() and not choice.ticket
    f["close"](r)


def test_jit_library_must_report_selected_identity(tmp_path, probe):
    f, r, req = build_stub(tmp_path, probe, jit=True, wrong_key=True)
    helper = helper_file(tmp_path, f"print('QK_JIT_V1 {'a'*64} {'b'*64} {'d'*64}')\n")
    assert enable(f, r, tmp_path, helper) == 0
    assert f["query"](r, C.byref(req), C.byref(Choice())) == 3
    assert b"loaded parent/build identity" in f["error"]()
    f["close"](r)


def test_jit_rejects_library_symlink_outside_cache(tmp_path, probe):
    f, r, req = build_stub(tmp_path, probe, jit=True)
    module = tmp_path / "modules" / ("a"*64) / "kernel.so"
    elsewhere = tmp_path / "outside.so"
    module.rename(elsewhere)
    module.symlink_to(elsewhere)
    helper = helper_file(tmp_path, f"print('QK_JIT_V1 {'a'*64} {'b'*64} {'d'*64}')\n")
    assert enable(f, r, tmp_path, helper) == 0
    assert f["query"](r, C.byref(req), C.byref(Choice())) == 3
    assert b"escapes" in f["error"]()
    f["close"](r)


def test_jit_resolve_rejects_invalid_source_fields():
    with pytest.raises(ValueError):
        parent_tuple('bad";code', [12, 0, 8, 64, 64, 8, 16, 2, 0, 16, -1])
    with pytest.raises(ValueError):
        parent_tuple("parent", [12, 9, 8, 64, 64, 8, 16, 2, 0, 16, -1])


def test_header_only_model_plan_deduplicates_and_does_not_admit_q8(tmp_path):
    from tools.gguf_internal_shape_inventory import _synthetic_gguf
    path = tmp_path / "model.gguf"
    # Header only: attempting to read/convert tensor bytes would fail.
    path.write_bytes(_synthetic_gguf([
        ("general.architecture", "qwen35moe"), ("qwen35moe.expert_count", 256),
        ("qwen35moe.expert_used_count", 8)], [
        ("blk.0.ffn_up_exps.weight", (2048,512,256), 12),
        ("blk.1.ffn_up_exps.weight", (2048,512,256), 12),
        ("blk.0.attn_output.weight", (2048,2048), 8),
        ("token_embd.weight", (2048,248320), 14),
        ("output.weight", (2048,248320), 14)]))
    requests, proof = model_requests(path, [1,128])
    assert requests == [(12,2,8,512,2048,256,1),(12,2,1024,512,2048,256,128),
                        (14,0,1,248320,2048,1,1),(14,0,128,248320,2048,1,128)]
    assert {x["name"] for x in proof["omitted"]} == {"token_embd.weight", "blk.0.attn_output.weight"}
    assert len(model_requests(path,[1,128],True)[0]) == 8


def fake_compiler(tmp):
    # Exercise the real locked/atomic cache with an inexpensive fake toolchain.
    sdk = tmp / "sdk"
    (sdk / "bin").mkdir(parents=True)
    (sdk / "lib").mkdir()
    for name in ("hggc_wrapper", "hggcrt1", "hggc", "hg_wrapper"):
        (sdk / "lib" / f"lib{name}.so").write_bytes(b"sdk")
    for name in ("hgcc", "hgobjdump", "g++"):
        p = sdk / "bin" / name
        p.write_text("#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\n"
                     "if '--version' in sys.argv: print('host compiler')\n"
                     "else: Path(sys.argv[sys.argv.index('-o')+1]).write_bytes(b'compiled')\n")
        p.chmod(0o755)
    return sdk


def test_cache_concurrency_corruption_relocation_and_key_invalidation(tmp_path, monkeypatch):
    sdk = fake_compiler(tmp_path)
    monkeypatch.setenv("PATH", str(sdk / "bin") + os.pathsep + os.environ["PATH"])
    parent = parent_tuple("parent", [12, 0, 8, 64, 64, 8, 16, 2, 0, 16, -1])
    compiler = Compiler(sdk, tmp_path / "cache")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(compiler.build, [parent, parent]))
    assert sorted(r["cache_hit"] for r in results) == [False, True]
    key = results[0]["key"]
    assert compiler.build(parent)["key"] == key
    import shutil
    relocated = tmp_path / "moved"
    shutil.copytree(tmp_path / "cache", relocated)
    sdk2 = tmp_path / "sdk-moved"
    shutil.copytree(sdk, sdk2)
    assert Compiler(sdk2, relocated).build(parent)["cache_hit"]
    changed = parent | {"stages": 3}
    assert compiler.build(changed)["key"] != key
    (tmp_path / "cache" / key / "kernel.so").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="identity/payload"):
        compiler.build(parent)
    assert not list((tmp_path / "cache" / key).glob("build-*"))


def test_helper_rejects_foreign_dispatcher_source_before_compile(tmp_path, monkeypatch):
    sdk = fake_compiler(tmp_path)
    monkeypatch.setenv("PATH", str(sdk / "bin") + os.pathsep + os.environ["PATH"])
    cache = tmp_path / "not-built"
    result = subprocess.run([sys.executable, str(ROOT / "tools/kpack_jit.py"), "resolve",
        "--sdk", str(sdk), "--cache", str(cache), "--source-contract", "0"*64,
        "--parent", "parent", "--tuple", "12", "0", "8", "64", "64", "8", "16", "2", "0", "16", "-1"],
        text=True, capture_output=True)
    assert result.returncode == 1 and "source differs" in result.stderr
    assert not result.stdout and not cache.exists()
