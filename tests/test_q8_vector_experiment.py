from dev.gemv_simt.q8_vector_access import pattern
from dev.gemv_simt.q8_vector_build import source
from quactlize.execution.simt_codegen import inventory
import pytest


def test_production_q8_policy_preserves_tc_indexed_and_different_incumbents(tmp_path):
    import copy
    import json
    import subprocess
    from pathlib import Path
    from tools.fit_kpack_q8_vector import fit, header, append_model_topology, append_model_readers, EVIDENCE, MATCHED, MODEL, READERS, OUTPUT, ROOT
    evidence, matched = json.loads(EVIDENCE.read_text()), json.loads(MATCHED.read_text())
    policy = fit(evidence, matched)
    assert len(policy['rows']) == 11 and len(policy['retained']) == 39
    assert all(r['key'][1] == 0 and r['candidate'][0] in (4,5) and r['new_us']<r['old_us'] for r in policy['rows'])
    assert sum(r['reason']=='KEEP_INDEXED_E16_NOT_MODEL_E256' for r in policy['retained'])==16
    model=json.loads(MODEL.read_text())
    policy=append_model_topology(policy,matched,model)
    assert len(policy['rows'])==12
    assert policy['rows'][-1]['candidate']==[5,8,4,4,8]
    readers=json.loads(READERS.read_text())
    previous=copy.deepcopy(policy)
    policy=append_model_readers(policy,matched,readers)
    assert len(policy['tc_rows'])==2
    assert {tuple(r['key'][2:4]) for r in policy['tc_rows']}=={(8192,2048),(4096,2048)}
    assert policy['rows'] == json.loads(OUTPUT.read_text())['rows']
    assert header(policy) == OUTPUT.with_suffix('.hpp').read_text()
    for mutation in ('compute','shape','winner','round','tc'):
        bad=copy.deepcopy(readers);m=copy.deepcopy(matched)
        row=next(r for r in bad['records'] if r['point']=='q8-qkv')
        if mutation=='compute':row['compute']='BF16'
        elif mutation=='shape':row['point_definition']['n']=4096
        elif mutation=='winner':row['config']['split']=8
        elif mutation=='round':row['round_deltas_pct'][0]=1.
        else:next(r for r in m['exact'] if r['key']==[8,0,8192,2048,1,1,1,1,0])['config']['split']=4
        with pytest.raises(ValueError):append_model_readers(copy.deepcopy(previous),m,bad)
    for mutation in ('numeric','duplicate','nan','delta'):
        bad=copy.deepcopy(evidence)
        if mutation=='numeric': bad['q8']['numeric']='FAIL'
        elif mutation=='duplicate': bad['q8']['performance'][1]=bad['q8']['performance'][0]
        elif mutation=='nan': bad['q8']['performance'][0]['best']['1']['median_us']=float('nan')
        else: bad['q8']['performance'][0]['delta_pct']+=1
        with pytest.raises(ValueError): fit(bad,matched)
    for mutation in ('numeric','samples','scope','negative','reducer'):
        bad=copy.deepcopy(model);r=bad['rows'][-1]
        if mutation=='numeric':r['numeric']=[]
        elif mutation=='samples':r['samples'][r['arm']][0][0]=float('nan')
        elif mutation=='scope':r['arm']='v5-c4-w8-p4-s1-r0'
        elif mutation=='negative':r['negatives']={}
        else:
            next(x for x in r['numeric'] if x.get('key')==r['arm'])['reducer_matched_bits']=False
        with pytest.raises(ValueError):append_model_topology(fit(evidence,matched),matched,bad)
    source=tmp_path/'policy.cpp'
    source.write_text('''#include "quactlize/dispatch/q8_vector.hpp"
#include <cassert>
int main() {
  using quactlize::dispatch::q8_vector::select;
  for(auto const& r:quactlize::q8_vector_data::kRows) {
    qkg_simt_call_v2 d{};auto& c=d.call;
    c.qtype=8;c.input_type=QKG_F32;c.mode=r.mode;c.n=r.n;c.k=r.k;c.experts=r.experts;
    c.topk=r.topk;c.channels=r.channels;c.rows=r.tokens;d.compute_type=r.compute;
    auto a=r.baseline;qkg_simt_config_v1 f{1,sizeof(f),a[0],a[1],a[2],a[3],a[4]};
    auto wrong=f;wrong.variant^=1;assert(!select(d,wrong));
    c.input_type=QKG_F16;assert(!select(d,f));c.input_type=QKG_F32;
    c.topk=0;assert(!select(d,f));c.topk=r.topk;
    c.experts=256;assert(!select(d,f));c.experts=r.experts;
    assert(select(d,f));a=r.candidate;
    assert(f.variant==a[0] && f.columns==a[1] && f.warps==a[2] && f.values==a[3] && f.split==a[4]);
    assert(!select(d,f));
  }
}''')
    exe=tmp_path/'policy'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT),str(source),'-o',str(exe)],check=True)
    subprocess.run([exe],check=True)


def test_q8_inventory_keeps_all_incumbents_and_compute_types():
    text=source()
    for c in inventory(8, legacy=True):
        for arm in range(2):
            for compute in range(2):
                assert f'invoke<{arm},{compute},{c.variant},{c.columns},{c.warps},{c.values}>' in text
    assert 'd->call.input_type!=QKG_F32' in text
    assert 'query_v2(*d,*f,&a,sizes)' in text
    assert 'register_reuse_reduce<8>' in text


def test_vector_metadata_footprint_and_minimum_abi_alignment():
    for c in inventory(8, legacy=True):
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


def test_hoist_numerics_bind_the_actual_arm_and_reject_changed_bits():
    import numpy as np
    from tools.run_kpack_q8_hoist import numeric_pair
    libraries = dict(baseline=object(), candidate=object())
    class Bench:
        bad = False
        lib = None
        def __init__(self): self.calls = []
        def update(self, repeat): pass
        def correctness(self, config):
            self.calls.append(('correct', self.lib))
            return np.array([2 if self.bad and self.lib is libraries['candidate'] else 1], dtype='f4'), 0
        def replay_and_negative(self, config):
            self.calls.append(('replay', self.lib))
            return dict(negative='ZERO_A_REJECTED')
        def invalid_id_negative(self, config): self.calls.append(('ids', self.lib))
    bench = Bench()
    assert len(numeric_pair(bench, libraries, None)) == 10
    for lib in libraries.values():
        assert bench.calls.count(('correct', lib)) == 4
        assert bench.calls.count(('replay', lib)) == 1
        assert bench.calls.count(('ids', lib)) == 1
    bench.bad = True
    with pytest.raises(ValueError, match='FP32 output bits'):
        numeric_pair(bench, libraries, None)


def test_hoist_only_accepts_one_source_change_and_no_selector_changes():
    import copy
    from tools.run_kpack_q8_hoist import compare_contracts, KERNEL, POINTS
    baseline = dict(execution_receipt=dict(source_hashes={KERNEL:'old','other':'same'},
        runtime={'wrapper':'same'}, compiler_sha256='compiler', flags=['-O3'], simt_configs=[1]), policy_hashes={'p':'a'})
    candidate = copy.deepcopy(baseline)
    candidate['execution_receipt']['source_hashes'][KERNEL] = 'new'
    assert compare_contracts(baseline, candidate) == [KERNEL]
    for plant in ('source','compiler','flags','runtime','configs','policy'):
        bad = copy.deepcopy(candidate)
        if plant == 'source': bad['execution_receipt']['source_hashes']['other'] = 'bad'
        elif plant == 'policy': bad['policy_hashes']['p'] = 'bad'
        else: bad['execution_receipt'][{'compiler':'compiler_sha256','configs':'simt_configs'}.get(plant,plant)] = 'bad'
        with pytest.raises(ValueError): compare_contracts(baseline, bad)
    assert [(n,k,c.split*n//c.tile_n) for n,k,c in POINTS] == [(512,2048,32),(2048,512,64),(2048,4096,512)]


def test_hoist_keeps_dot_order_and_retains_f16_m1_guards():
    from pathlib import Path
    from dev.gemv_simt.model_followup import kernel_body, candidate_body
    root = Path(__file__).resolve().parents[1]
    text = (root/'quactlize/execution/simt_q8_vector.cuh').read_text()
    assert 'bool Hoist=false' in kernel_body(True)
    assert 'bool Hoist=false' in candidate_body(True)
    assert 'uint32_t words[2][2][4][Pairs]' in text
    assert 'split==1 && c.mode==QKG_DENSE && c.rows==1 && c.experts==1' in text
    assert 'Variant==5 && P==4 && Columns==4 && Warps==8' in text
    assert 'c.n==512 && c.k==2048' in text
    assert 'Variant==5 && Columns==8 && Warps==4 && P==4' in text
    assert 'model_gemv::dense_m1(c,2048,512)' in text
    bf16 = text[text.index('} else if(d.compute_type==QKG_COMPUTE_BF16)'):]
    assert ',P,true>' not in bf16
    # Same low-plane addresses, code slot and FMA order; only the load window changes.
    old = [s*8+h*4+r for h in range(2) for s in range(4) for r in range(4)]
    hoist = [s*16+t*8+h*4+r for h in range(2) for s in range(2) for t in range(2) for r in range(4)]
    assert old == hoist and sorted(hoist)==list(range(32))
