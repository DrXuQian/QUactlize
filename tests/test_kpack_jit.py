"""Production JIT and cache contract tests."""
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from tests.test_kpack_native_dispatch import build_stub, probe, query, Choice, Call
from quactlize.runtime.compiler import Compiler, sha, source_contract
from quactlize.runtime.jit_cache import ParentCache, name as cache_name
from quactlize.runtime.tuning import digest
from tools.kpack_jit import parent_tuple, model_requests, inspect_modules
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


def test_header_only_model_plan_deduplicates_and_q8_is_w8a16(tmp_path):
    from ci.support.gguf import _synthetic_gguf
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
    assert requests == [(8,1,1,2048,2048,1,1),(8,1,128,2048,2048,1,128),
                        (12,2,8,512,2048,256,1),(12,2,1024,512,2048,256,128),
                        (14,0,1,248320,2048,1,1),(14,0,128,248320,2048,1,128)]
    assert {x["name"] for x in proof["omitted"]} == {"token_embd.weight"}
    assert len(model_requests(path,[1,128],True)[0]) == 10
    filtered, proof = model_requests(path, [1,128], tensor_pattern=r"^blk\.\d+\.ffn_up_exps\.weight$")
    assert filtered == requests[2:4]
    assert next(r for r in proof["omitted"] if r["name"] == "output.weight")["reason"] == "OUTSIDE_CONSUMER_TENSOR_PATTERN"


def test_recorded_model_prewarm_keeps_normal_and_typed_endpoints_and_one_pool(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace
    import tools.prewarm_kpack_model as pre
    barrier = threading.Barrier(2)
    identities, built = [], []
    def compiler(sdk, cache, jobs, dense, compute):
        assert sdk == 'sdk' and cache == 'cache' and jobs == 2
        identities.append((dense, compute))
        def build(parent):
            barrier.wait(timeout=5)
            built.append((parent['symbol'], dense, compute))
            return dict(parent=parent, cache_hit=True)
        return SimpleNamespace(identity={'base_source_contract': 'contract'}, build=build)
    monkeypatch.setattr(pre, 'compiler_for', compiler)
    monkeypatch.setattr(pre, 'source_contract', lambda identity: 'unused')
    module = dict(symbol='parent', tuple=[12,0,8,64,64,8,16,2,0,16,-1], dense_io=False, compute_type='f16')
    closure = dict(schema='quactlize.model-prewarm.v1', source_contract='contract',
                   modules=[module, module.copy(), module | {'dense_io': True}])
    receipt = pre.prewarm(closure, 'sdk', 'cache', 2)
    assert receipt['parents'] == receipt['workers'] == receipt['cache_hits'] == 2
    assert set(identities) == {(False,'f16'), (True,'f16')}
    assert set(built) == {('parent',False,'f16'), ('parent',True,'f16')}
    closure['source_contract'] = 'foreign'
    with pytest.raises(ValueError, match='differs from current'):
        pre.prewarm(closure, 'sdk', 'cache', 2)


@pytest.mark.parametrize('fault', [None, 'model', 'tokens', 'bundle', 'shape', 'grouped', 'tp'])
def test_recorded_prewarm_is_bound_to_geometry_workload_and_runtime(tmp_path, monkeypatch, fault):
    import tools.prewarm_kpack_model as pre
    model_path = tmp_path/'model.gguf'
    model_path.touch()
    monkeypatch.setattr(pre, 'sha', lambda p: 'other' if fault=='bundle' else 'bundle')
    monkeypatch.setattr(pre, 'inventory', lambda p: dict(eligible=['weight'], files=[str(model_path)]))
    monkeypatch.setattr(pre, 'read_gguf_header', lambda *a: dict(tensors=[dict(
        name='weight', qtype=12, dims_gguf=[5120, 1024 if fault!='shape' else 2048] + ([256] if fault=='grouped' else []))]))
    model = dict(name='other' if fault=='model' else 'model', path=str(model_path), split='tensor' if fault=='tp' else 'none')
    plan = dict(prompts=[128 if fault=='tokens' else 2048])
    closure = dict(model='model', tokens=[1,16,2048], bundle_manifest_sha256='bundle', shapes=[[12,1024,5120]])
    reason = pre.select_closure(model, plan, tmp_path, closure)
    assert (reason is None) == (fault is None)


def test_qwen32_recorded_parent_closure_is_nine_not_a_new_sweep():
    closure = json.loads((ROOT/'tools/kpack_model_prewarm_qwen3_32b.json').read_text())
    assert closure['tokens'] == [1,16,2048]
    modules = closure['modules']
    assert len(modules) == 9 and sum(m['dense_io'] for m in modules) == 3
    assert len({(m['symbol'],m['dense_io'],m['compute_type']) for m in modules}) == 9
    for module in modules:
        assert module['compute_type'] == 'f16'
        assert parent_tuple(module['symbol'],module['tuple'])['route'].endswith('dense')


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


@pytest.fixture(scope='module')
def cache_probe(tmp_path_factory):
    """CPU-only probe of the exact C++ disk-hit code used by dispatch."""
    out = tmp_path_factory.mktemp('jit-cache-probe')
    source = out / 'probe.cpp'
    source.write_text(r'''
#include "quactlize/dispatch/jit_cache.hpp"
#include <chrono>
#include <iostream>
int main(int argc, char** argv) {
    using namespace quactlize::dispatch;
    if (argc!=18 && argc!=19) return 2;
    Config c{}; c.symbol=argv[4];
    int* fields[]={&c.qtype,&c.route,&c.tm,&c.tn,&c.tk,&c.wm,&c.wn,
                   &c.stages,&c.ap,&c.dn,&c.parent_persistent};
    for (int i=0;i<11;++i) *fields[i]=std::stoi(argv[7+i]);
    auto environment=cache_environment(argv[1],argv[3]);
    int count=argc==19 ? std::stoi(argv[18]) : 1;
    auto start=std::chrono::steady_clock::now();
    std::string receipt;
    for(int i=0;i<count;++i) {
        receipt=cached_parent(argv[2],environment,argv[3],c,std::stoi(argv[5]),std::stoi(argv[6]));
        if (receipt.empty()) return 4;
    }
    std::cerr<<std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/count;
    std::cout<<receipt;
}
''')
    binary = out / 'probe'
    subprocess.run(['g++', '-std=c++17', '-O2', f'-I{ROOT}', str(source), '-o', str(binary)], check=True)
    return binary


def lookup(probe, cache, sdk, parent, dense=False, compute='f16', repeat=1):
    from quactlize.runtime.tuning import ROUTES
    values = [parent['qtype'], ROUTES.index(parent['route'])] + [parent[f] for f in
              ('tm', 'tn', 'tk', 'wm', 'wn', 'stages', 'ap', 'dn', 'persistent')]
    return subprocess.run([str(probe), str(sdk), str(cache.cache), cache.source, parent['symbol'],
                           str(int(dense)), str(int(compute=='bf16')), *map(str,values), str(repeat)],
                          text=True, capture_output=True)


@pytest.mark.parametrize('fault', [None, 'parent', 'endpoint', 'compute', 'source', 'sdk',
                                 'module', 'manifest', 'library-link', 'index-link', 'request'])
def test_native_cache_matches_python_and_invalidates_without_hashing(tmp_path, monkeypatch, cache_probe, fault):
    sdk = fake_compiler(tmp_path)
    monkeypatch.setenv('PATH', str(sdk/'bin') + os.pathsep + os.environ['PATH'])
    compiler = Compiler(sdk, tmp_path/'cache')
    parent = parent_tuple('parent', [12,0,8,64,64,8,16,2,0,16,-1])
    record = compiler.build(parent)
    cache = ParentCache(sdk, compiler.cache, source_contract(compiler.identity))
    cache.publish(record)
    dense, compute = False, 'f16'
    if fault == 'parent': parent = parent | {'stages': 3}
    if fault == 'endpoint': dense = True
    if fault == 'compute': compute = 'bf16'
    if fault == 'source': cache = ParentCache(sdk, compiler.cache, '0'*64)
    if fault == 'sdk':
        (sdk/'lib/libhggc.so').write_bytes(b'new SDK')
        cache = ParentCache(sdk, compiler.cache, cache.source)
    if fault in ('module', 'manifest', 'library-link'):
        path = compiler.cache/record['key']/('manifest.json' if fault=='manifest' else 'kernel.so')
        if fault == 'library-link':
            path.rename(tmp_path/'elsewhere')
            path.symlink_to(tmp_path/'elsewhere')
        else:
            path.write_bytes(path.read_bytes()+b'changed')
    if fault in ('index-link', 'request'):
        path = cache.directory/(cache_name(cache.request(parent,False,'f16'))+'.txt')
        if fault == 'index-link':
            path.rename(tmp_path/'elsewhere')
            path.symlink_to(tmp_path/'elsewhere')
        else:
            path.write_text(path.read_text().replace('\nparent\n','\nanother\n'))
    # A hit must not read/hash the source, compiler or .so contents in Python.
    monkeypatch.setattr('quactlize.runtime.compiler.sha', lambda *_: pytest.fail('hot path hashed a payload'))
    receipt = cache.read(parent,dense,compute)
    result = lookup(cache_probe,cache,sdk,parent,dense,compute)
    assert result.returncode == (4 if fault else 0), result.stderr
    assert receipt is None if fault else receipt == result.stdout


def test_native_lookup_latency_is_host_only(tmp_path, monkeypatch, cache_probe):
    sdk = fake_compiler(tmp_path)
    monkeypatch.setenv('PATH',str(sdk/'bin')+os.pathsep+os.environ['PATH'])
    compiler = Compiler(sdk,tmp_path/'cache')
    parent = parent_tuple('parent',[12,0,8,64,64,8,16,2,0,16,-1])
    cache = ParentCache(sdk,compiler.cache,source_contract(compiler.identity))
    cache.publish(compiler.build(parent))
    result = lookup(cache_probe,cache,sdk,parent,repeat=1000)
    assert result.returncode == 0, result.stderr
    print(f'HOST_JIT_INDEX us_per_lookup={float(result.stderr):.3f} calls=1000 '
          'python_children=0 scope=INDEX_ONLY_NOT_DLOPEN_OR_GPU')


def test_prewarm_indexes_all_endpoint_kinds_and_compact_reducers(tmp_path, monkeypatch, cache_probe):
    from tools.kpack_jit import compiler_for
    sdk = fake_compiler(tmp_path)
    monkeypatch.setenv('PATH',str(sdk/'bin')+os.pathsep+os.environ['PATH'])
    parent = parent_tuple('parent',[12,0,8,64,64,8,16,2,0,16,-1])
    keys = set()
    for dense,compute,grouped,reducer in [(False,'f16',False,False),(False,'f16',True,False),
            (True,'f16',False,False),(True,'bf16',False,False),(False,'bf16',True,False),
            (False,'f16',False,True)]:
        selected = parent
        if grouped: selected = parent | {'route':'fq-grouped','persistent':0}
        if reducer: selected = parent | {'symbol':'parent_rc8','dense_reducer':8}
        compiler = compiler_for(sdk,tmp_path/'cache',2,dense,compute)
        record, = compiler.compile_only([selected])
        keys.add(record['key'])
        result = lookup(cache_probe,compiler.cache,sdk,selected,dense,compute)
        assert result.returncode == 0 and result.stdout.split()[1] == record['key'], result.stderr
    assert len(keys) == 6


def test_existing_cache_import_keeps_keys_and_next_parent_skips_compiler(tmp_path, monkeypatch, cache_probe):
    from tools import kpack_jit as jit
    sdk = fake_compiler(tmp_path)
    monkeypatch.setenv('PATH', str(sdk/'bin') + os.pathsep + os.environ['PATH'])
    compiler = Compiler(sdk,tmp_path/'cache')
    first = parent_tuple('first',[12,0,8,64,64,8,16,2,0,16,-1])
    second = first | {'symbol':'second','stages':3}
    records = [compiler.build(p) for p in (first,second)]  # Legacy cache: no fast indices.
    cache = ParentCache(sdk,compiler.cache,source_contract(compiler.identity))
    argv = ['resolve','--sdk',str(sdk),'--cache',str(compiler.cache),'--source-contract',cache.source,
            '--parent','first','--tuple','12','0','8','64','64','8','16','2','0','16','-1']
    result = subprocess.run([sys.executable,str(ROOT/'tools/kpack_jit.py'),*argv],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert 'imported=2 compile=0' in result.stderr
    assert result.stdout.split()[1] == records[0]['key']
    assert lookup(cache_probe,cache,sdk,second).stdout.split()[1] == records[1]['key']
    # Direct Python resolve also takes the index hit, without constructing a compiler.
    argv[argv.index('first')] = 'second'
    argv[-4] = '3'  # stages
    monkeypatch.setattr(sys,'argv',['kpack_jit.py',*argv])
    monkeypatch.setattr(jit,'compiler_for',lambda *_: pytest.fail('cache hit started compiler'))
    jit.main()


@pytest.mark.parametrize('wrong_key', [False, True])
def test_native_disk_hit_never_starts_python_and_still_checks_loaded_abi(tmp_path, probe, wrong_key):
    f,r,req = build_stub(tmp_path,probe,jit=True,wrong_key=wrong_key)
    sdk = fake_compiler(tmp_path)
    fields = query(probe,[(12,2,528,3072,512,256,129)])[0].split()
    parent = parent_tuple(fields[0],list(map(int,fields[1:12])))
    identity = dict(kernel='kernel',generator='generator',flags=[],base_source_contract='d'*64)
    record = dict(key='a'*64,parent=parent,identity=identity)
    (tmp_path/'modules'/('a'*64)/'manifest.json').write_text(json.dumps(record))
    cache = ParentCache(sdk,tmp_path/'modules','d'*64)
    cache.publish(record)
    helper = helper_file(tmp_path,"raise RuntimeError('disk hit started Python')\n")
    options = JitOptions(1,C.sizeof(JitOptions),os.fsencode(sys.executable),os.fsencode(helper),
                        os.fsencode(sdk),os.fsencode(cache.cache))
    for _ in range(2):  # A new runtime/process can reuse disk entries, not just the memory map.
        assert f['enable_jit'](r,C.byref(options)) == 0
        assert f['query'](r,C.byref(req),C.byref(Choice())) == (3 if wrong_key else 0)
        if wrong_key: assert b'loaded parent/build identity' in f['error']()
        f['close'](r)
        r = C.c_void_p()
        assert f['open'](os.fsencode(tmp_path),C.byref(r)) == 0
    f['close'](r)


def published_module(tmp_path):
    parent = parent_tuple("parent", [12, 0, 8, 64, 64, 8, 16, 2, 0, 16, -1])
    identity = dict(kernel="kernel", generator="generator", flags=[], sdk="sdk", host="host")
    key = digest(dict(parent=parent, identity=identity, source=Compiler.source(None, parent, "")))
    directory = tmp_path / key
    directory.mkdir()
    library = directory / "kernel.so"
    library.write_bytes(b"module fixture")
    receipt = dict(key=key, parent=parent, identity=identity, sha256=sha(library))
    (directory / "manifest.json").write_text(json.dumps(receipt))
    return receipt, directory


def test_inspect_selected_cache_is_read_only_and_relocatable(tmp_path):
    receipt, directory = published_module(tmp_path)
    contract = source_contract(receipt["identity"])
    before = {p.name: sha(p) for p in directory.iterdir()}
    records = inspect_modules(tmp_path, [receipt["key"], receipt["key"]], contract)
    assert len(records) == 1 and records[0]["parent"] == receipt["parent"]
    assert Path(records[0]["path"]) == directory / "kernel.so"
    output = subprocess.check_output([sys.executable, str(ROOT / "tools/kpack_jit.py"), "inspect",
        "--cache", str(tmp_path), "--keys", receipt["key"], "--source-contract", contract], text=True)
    assert json.loads(output)["modules"] == records
    assert before == {p.name: sha(p) for p in directory.iterdir()}
    moved = tmp_path / "moved"
    moved.mkdir()
    directory.rename(moved / receipt["key"])
    assert inspect_modules(moved, [receipt["key"]], contract)[0]["path"].startswith(str(moved))


@pytest.mark.parametrize("fault", ["payload", "key", "parent", "source", "manifest-link", "library-link", "directory-link", "traversal"])
def test_inspect_cache_rejects_stale_tampered_or_escaped_receipts(tmp_path, fault):
    receipt, directory = published_module(tmp_path)
    contract, key = source_contract(receipt["identity"]), receipt["key"]
    if fault in ("manifest-link", "library-link"):
        path = directory / ("manifest.json" if fault == "manifest-link" else "kernel.so")
        moved = tmp_path / "outside"
        path.rename(moved)
        path.symlink_to(moved)
    elif fault == "directory-link":
        moved = tmp_path / "elsewhere"
        directory.rename(moved)
        directory.symlink_to(moved)
    elif fault == "traversal":
        key = "../" + key
    elif fault == "payload":
        (directory / "kernel.so").write_bytes(b"different")
    else:
        if fault == "key":
            receipt["key"] = "0"*64
        elif fault == "parent":
            receipt["parent"]["stages"] = 3
        else:
            receipt["identity"]["kernel"] = "different"
        (directory / "manifest.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        inspect_modules(tmp_path, [key], contract)


def test_prewarm_source_mismatch_rejected_before_build(tmp_path, monkeypatch):
    sdk = fake_compiler(tmp_path)
    monkeypatch.setenv("PATH", str(sdk / "bin") + os.pathsep + os.environ["PATH"])
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(parents=[parent_tuple("parent", [12,0,8,64,64,8,16,2,0,16,-1])])))
    result = subprocess.run([sys.executable, str(ROOT / "tools/kpack_jit.py"), "prewarm", "--sdk", str(sdk),
        "--cache", str(tmp_path / "cache"), "--plan", str(plan), "--source-contract", "0"*64],
        text=True, capture_output=True)
    assert result.returncode == 1 and "source differs" in result.stderr
    assert not (tmp_path / "cache").exists()
