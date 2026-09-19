"""Final public decisions must match the immutable pre-refactor git tree."""
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from quactlize.dispatch.planning import plan_smallm,requirements,validate_inventory,prewarm_groups

ROOT=Path(__file__).resolve().parents[1]
BASELINE='5168b7136c40434bef01b6ac48732b0cce5468a1'


@pytest.fixture(scope='module')
def snapshots(tmp_path_factory):
    directory=tmp_path_factory.mktemp('selection-shadow')
    baseline=directory/'baseline';baseline.mkdir()
    archive=subprocess.check_output(['git','archive',BASELINE,'quactlize','policies'],cwd=ROOT)
    # Extract only regular source files/directories into the fresh test fixture.
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        members=tar.getmembers()
        assert all((m.isdir() or m.isfile()) and not Path(m.name).is_absolute() and
                   '..' not in Path(m.name).parts for m in members)
        tar.extractall(baseline,members=members,filter='data')
    (directory/'catalog.inc').write_text('static std::vector<Image> kImages; static char kJitSource[]="";\n')
    binaries=[]
    for i,root in enumerate((baseline,ROOT)):
        binary=directory/f'probe-{i}'
        subprocess.run(['g++','-std=c++17','-O2','-pthread','-I'+str(root),'-I'+str(directory),
            str(ROOT/'tests/selection_snapshot_host.cpp'),'-ldl','-o',str(binary)],check=True)
        binaries.append(binary)
    return binaries


def case_keys():
    policy=json.loads((ROOT/'policies/kpack_smallm_matched_v1.json').read_text())
    keys=[r['key']+[0] for r in policy['exact']]
    for r in policy['exact']:
        for axis in (2,3):
            for multiplier,offset in ((1,256),(2,0),(2,256)):
                key=r['key']+[0];key[axis]=key[axis]*multiplier+offset;keys.append(key)
    keys += [r['key'][:-1]+[int(r['key'][-1]=='bf16'),0] for r in policy['excluded']]
    keys += [[q,mode,768,2048,256 if mode else 1,8 if mode else 1,ch,t,c,0]
             for q in (8,10,11,12,13,14) for mode in (0,2) for ch in ((1,8) if mode else (1,))
             for t in range(0,10) for c in (0,1)]
    keys += [r['key']+[mutation] for r in policy['exact'][::19] for mutation in range(1,6)]
    from dev.tp2_decode.fallback_plan import MODEL_POINTS,UNSEEN
    keys += [[p.q,p.mode,p.n,p.k,p.experts,8 if p.mode else 1,p.channels,t,p.compute,0]
             for p in MODEL_POINTS+UNSEEN for t in range(1,9)]
    return sorted(set(map(tuple,keys)))


def test_public_final_decisions_equal_frozen_baseline(snapshots):
    keys=case_keys();text=''.join(' '.join(map(str,k))+'\n' for k in keys)
    results=[subprocess.check_output([str(p)],input=text,text=True).splitlines() for p in snapshots]
    assert len(results[0])==len(results[1])==len(keys)
    assert results[0]==results[1],next((k,a,b) for k,a,b in zip(keys,*results) if a!=b)
    print(f'SELECTION_SHADOW PASS requests={len(keys)} baseline={BASELINE} public_abi=unchanged')


@pytest.fixture(scope='module')
def final_plan(tmp_path_factory):
    return plan_smallm(tmp_path_factory.mktemp('final-plan'))


def execution_for(plan):
    need=requirements(plan);configs={}
    fields=('variant','columns','warps','values','split')
    for q,compute,*config in need['simt']:
        configs.setdefault(str(q),[]).append(dict(zip(fields,config)))
    return dict(simt_configs=configs,simt_compute_v2=dict(compute=['f16','bf16'],formats=[8,10,11,12,13,14]),
                q4_decode_policy_sha256='host-fixture')


def test_final_catalog_is_exact_and_checks_full_tuple(final_plan):
    assert len(final_plan['requests'])==1842
    assert all(r['status']==0 for r in final_plan['requests'])
    assert len(requirements(final_plan)['simt'])==189
    execution=execution_for(final_plan)
    validate_inventory(final_plan,execution,jit=True)
    for mutate in ('config','compute','q4'):
        bad=copy.deepcopy(execution)
        if mutate=='config':bad['simt_configs']['8']=[]
        elif mutate=='compute':bad['simt_compute_v2']['compute']=['f16']
        else:bad.pop('q4_decode_policy_sha256')
        with pytest.raises(ValueError):validate_inventory(final_plan,bad,jit=True)
    with pytest.raises(ValueError,match='unpackaged typed TC'):
        validate_inventory(final_plan,execution,jit=False)


def test_prewarm_keeps_compute_and_endpoint_identity(final_plan):
    groups=prewarm_groups(final_plan)
    assert {(c,d) for c,d,_ in groups}=={('f16',False),('f16',True),('bf16',False),('bf16',True)}
    assert sum(len(p) for _,_,p in groups)==len(requirements(final_plan)['tc'])==62
    assert all(p['route'].endswith('dense')==dense for _,dense,parents in groups for p in parents)


def test_cache_profiles_and_channels_do_not_alias(tmp_path):
    source=tmp_path/'cache.cpp';binary=tmp_path/'cache'
    (tmp_path/'catalog.inc').write_text('static std::vector<Image> kImages; static char kJitSource[]="";\n')
    source.write_text('''#include "quactlize/dispatch/binding.cpp"
#include <cassert>
#include <set>
int main(){using namespace quactlize::dispatch;qks_request_v1 r{};std::set<Key> keys;
for(auto p:{SelectionProfile::General,SelectionProfile::Decode,SelectionProfile::LegacySmallM,SelectionProfile::MatchedSmallM})
for(int ch:{0,1,8})for(int endpoint:{0,1,2})for(int compute:{0,1})assert(keys.insert(key(r,{p,ch},endpoint,compute)).second);
assert(keys.size()==72);}
''')
    subprocess.run(['g++','-std=c++17','-O1','-pthread','-I'+str(ROOT),'-I'+str(tmp_path),source,'-ldl','-o',binary],check=True)
    subprocess.run([binary],check=True)


def test_launcher_strategy_matches_old_predicates(tmp_path):
    binary=tmp_path/'strategy'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT),
                    ROOT/'tests/simt_strategy_host.cpp','-o',binary],check=True)
    subprocess.run([binary],check=True)


def test_real_generated_inventory_covers_final_selection(final_plan):
    from quactlize.execution import simt_codegen
    execution=execution_for(final_plan)
    execution['simt_configs']={str(q):[c.record() for c in simt_codegen.runtime_inventory(q)]
                               for q in simt_codegen.QTYPES}
    validate_inventory(final_plan,execution,jit=True)


def test_implementation_identity_follows_actual_shape(tmp_path):
    plan=plan_smallm(tmp_path,[[8,0,n,k,1,1,1,1,0] for n,k in
                              ((2048,4096),(8192,2048),(4096,2048),(768,2048))])
    rows={tuple(r['request'][2:4]):r for r in plan['requests']}
    assert rows[2048,4096]['implementation']['producer']=='q8-vector-fixed'
    assert rows[8192,2048]['implementation']['producer']=='q8-vector-fixed-hoisted'
    assert rows[4096,2048]['implementation']['producer']=='q8-vector-s1-narrow'
    assert rows[768,2048]['implementation']['hoist']
    assert not rows[768,2048]['implementation']['fixed']


def test_cli_plan_and_prewarm_skip_all_simt_compilation(tmp_path):
    output=tmp_path/'plan'
    subprocess.run([sys.executable,ROOT/'tools/kpack_jit.py','plan-smallm','--output',output,
        '--request','8','0','2048','4096','1','1','1','1','0'],check=True)
    receipt=tmp_path/'prewarm.json'
    # A SIMT-only plan must not even inspect a compiler/SDK installation.
    subprocess.run([sys.executable,ROOT/'tools/kpack_jit.py','prewarm',
        '--plan',output/'plan.json','--sdk',tmp_path/'no-sdk','--cache',tmp_path/'no-cache',
        '--receipt',receipt],check=True)
    assert json.loads(receipt.read_text())['modules']==[]
    assert not (tmp_path/'no-cache').exists()


def test_typed_prewarm_only_builds_selected_parents(final_plan,tmp_path,monkeypatch):
    from types import SimpleNamespace
    import tools.kpack_jit as jit
    recorded=[]
    def compiler(sdk,cache,jobs,dense,compute):
        def compile_only(parents,progress):
            recorded.extend((p['symbol'],compute,dense) for p in parents)
            return [dict(parent=p,device_validated=False) for p in parents]
        return SimpleNamespace(identity={'base_source_contract':'contract'},compile_only=compile_only)
    monkeypatch.setattr(jit,'compiler_for',compiler)
    monkeypatch.setattr(jit,'source_contract',lambda identity:'contract')
    path=tmp_path/'plan.json';path.write_text(json.dumps(final_plan))
    monkeypatch.setattr(sys,'argv',['kpack_jit.py','prewarm','--sdk','sdk','--cache','cache',
        '--plan',str(path),'--source-contract','contract'])
    jit.main()
    expected={(symbol,'bf16' if compute else 'f16',dense) for symbol,compute,dense in requirements(final_plan)['tc']}
    assert set(recorded)==expected and len(recorded)==len(expected)
    bad=copy.deepcopy(final_plan);bad['requests'][0]['status']=1
    with pytest.raises(ValueError,match='policy misses'):prewarm_groups(bad)
