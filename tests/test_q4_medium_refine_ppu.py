"""Exact host fold, bounded ownership, immutable launch seams and PPU ISA."""
import ast
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from dev.gemv_ppu import medium_refine as spec
from dev.gemv_ppu import run_medium_refine as runner

ROOT=spec.ROOT
PACKAGE=ROOT/'prebuilt/ppu0010/q4-medium-refine-v1'


def test_only_one_shape_and_both_actual_near_tied_anchors():
    assert spec.SHAPES==((1024,5120),)
    assert len(spec.inventory())==22
    assert len({c.key for c in spec.inventory()})==22
    assert spec.anchor('kpack-p2').key=='affine-h1-l1-a1-w10-c4-p2'
    assert spec.anchor('kpack-p4').key=='affine-h1-l0-a0-w20-c4-p4'
    assert len(spec.plan()['prior_closed_shapes'])==5 and spec.plan()['reference_limit_pct']==5
    assert all((c.values,c.warps) in ((2,10),(4,20)) for c in spec.inventory() if c.unsigned)
    for c in spec.inventory():
        g=c.geometry()
        assert g['grid']*g['tile_n']==1024 and g['threads']<=1024 and g['inter_cta_split']==1
        assert 0<g['last_pass_workers']<=g['k_workers'] and g['last_pass_workers']%8==0
        assert g['shared_a_bytes']==0
        assert spec.lookup(c.key)==c
    with pytest.raises(ValueError):runner.recipe('reader',512,2048,spec.inventory()[0].key)
    with pytest.raises(ValueError):spec.lookup('p2-w10-r9-u0')


def test_source_keeps_dot_dequant_scatter_and_reference_body():
    old=spec.prior.affine_source();new=spec.kernel_source();full=spec.source()
    assert old in full
    start='        uint4 metadata[Early ? P : 1];';end='    constexpr int TileN=Columns*P;'
    assert old[old.index(start):old.index(end)]==new[new.index(start):new.index(end)]
    assert old[old.index('        for(int d=TileN;'):]==new[new.index('        for(int d=TileN;'):]
    assert 'for(int w=tid/TileN;w<Warps;w+=32/TileN) sum+=partial[w*TileN+tid%TileN];' in new
    assert 'q4_medium_fold<0,Warps,TileN>(sum,partial,unsigned(tid));' in new
    assert full.count('if(control) q4_affine_pipeline<')==22
    assert full.count('else q4_medium_refined<')==22


@pytest.mark.parametrize('plant',['none','wrong-column','zero-fold'])
def test_actual_cpp_fold_exact_fp32_and_two_negative_controls(tmp_path,plant):
    header=(ROOT/'dev/gemv_ppu/medium_reduce.hpp').read_text()
    if plant=='wrong-column':header=header.replace('(lane&(TileN-1))','((lane+1)&(TileN-1))')
    if plant=='zero-fold':header=header.replace('sum+=partial[','sum+=0.f*partial[')
    prefix='''#include <cstdint>
#include <cstring>
#include <memory>
#include <cstdio>
#define __host__
#define __device__
#define __forceinline__ inline
'''
    test='''
template<int Warps,int TileN> int check() {
    std::unique_ptr<float[]> partial(new float[Warps*TileN]);
    uint32_t state=0x549731u;
    for(int round=0;round<128;++round) {
        for(int i=0;i<Warps*TileN;++i) {
            state^=state<<13;state^=state>>17;state^=state<<5;
            float value=float(int32_t(state)%65537)*0.00391f;
            if(round%3==0 && i%7==0) value=1e20f;
            if(round%3==0 && i%7==1) value=-1e20f;
            partial[i]=value;
        }
        for(unsigned lane=0;lane<32;++lane) {
            float expected=0,actual=0;
            for(int w=lane/TileN;w<Warps;w+=32/TileN) expected+=partial[w*TileN+lane%TileN];
            q4_medium_fold<0,Warps,TileN>(actual,partial.get(),lane);
            if(std::memcmp(&expected,&actual,4)!=0) return 1;
        }
    }
    return 0;
}
int main() {
'''
    geometries=sorted({(c.warps,c.geometry()['tile_n']) for c in spec.inventory()})
    test+='\n'.join(f'    if(check<{w},{t}>()) return 1;' for w,t in geometries)+'\n    return 0;\n}\n'
    source=tmp_path/'fold.cpp';source.write_text(prefix+header+test)
    binary=tmp_path/'fold'
    subprocess.run(['g++','-std=c++17','-O2','-fsanitize=address,undefined','-fno-omit-frame-pointer',str(source),'-o',str(binary)],check=True)
    result=subprocess.run([str(binary)],capture_output=True,text=True)
    assert result.returncode==(0 if plant=='none' else 1),result.stderr


def test_all_warp_geometries_cover_B_once_and_fold_reads_identical():
    for c in spec.inventory():
        g=c.geometry();tile=g['tile_n'];seen=np.zeros((5120//4,tile),dtype=np.uint8)
        for turn in range(g['k_passes']):
            for tid in range(g['threads']):
                group=turn*g['k_workers']+tid//4
                if group>=160:continue
                col=tid%4*c.values
                seen[group*8:group*8+8,col:col+c.values]+=1
        assert np.all(seen==1),c.key
        for lane in range(32):
            old=list(range(lane//tile,c.warps,32//tile))
            new=[lane//tile+i*(32//tile) for i in range(g['cta_fold_rounds']) if lane//tile+i*(32//tile)<c.warps]
            assert new==old
        a=spec.access(c,dict(A=16,B=32,metadata=48))
        base=spec.prior.access(c.parent,1024,5120,dict(A=16,B=32,metadata=48))
        assert a['streams']==base['streams']
        assert a['logical_global_lane_bytes']==base['logical_global_lane_bytes']
        assert a['logical_global_lane_bytes']['B']==1024*5120//2


def row(arm,key,phase,samples,profile=False):
    weight=1024*5120*9//16;copies=math.ceil(2.25*67108864/weight);value=10.
    r=dict(status='PASS',arm=arm,variant=arm,config_key=key,phase=phase,shape=[1,1024,5120],mode='rotating',
        recipe=list(runner.recipe(arm,1024,5120,key)),error=1e-8,zero_code_negative='PASS',zero_a_check='PASS',
        copies=copies,weight_bytes=weight,device=dict(l2_bytes=67108864),calls_per_graph=max(2,math.ceil(32/copies))*copies,
        output_type='F32',inter_cta_split=1,launches_per_call=1,weight_arithmetic=runner.arithmetic(arm),
        storage='XPLANE' if arm=='xplane' else 'RAW_GGUF' if arm=='raw-reference' else 'CANONICAL_KPACK4',
        timing_scope='RESIDENT_FULL_CALL_NO_HOST_PACK_NO_JIT',cache_scope='ACU_FORCED_COLD' if profile else 'ROTATING_AT_LEAST_2_25_L2')
    if arm=='reader':
        c=spec.lookup(key);value=9.+spec.inventory().index(c)*.001
        r.update(geometry=c.geometry(),matched_fp32_sha256='a'*64,code_dequant='LOP3_HALF2',
            immutable_control_sha256='a'*64 if c.immutable_parent else None,
            same_geometry_control='UNCHANGED_AFFINE_SAME_DOT_AND_CTA_ADD_ORDER',access_models=[spec.access(c)])
    r.update(samples_us=[value]*samples,median_us=value if samples else None)
    return r


@pytest.mark.parametrize('arm,key',[('kpack-p2','control'),('kpack-p4','control'),('raw-reference','control'),
    ('reader','p2-w10-r1-u0'),('reader','p4-w24-r1-u0')])
def test_parser_checks_immutable_scope_exact_order_and_cache(arm,key):
    for profile,samples in ((False,5),(False,15),(True,0)):
        r=row(arm,key,'screen',samples,profile)
        assert runner.parse_row(r,arm,1024,5120,key,'screen',samples,profile)==r
    for field,bad in [('recipe',[0]),('error',float('nan')),('zero_a_check','SKIP'),('samples_us',[0.]*5)]:
        r=row(arm,key,'screen',5);r[field]=bad
        with pytest.raises(ValueError):runner.parse_row(r,arm,1024,5120,key,'screen',5)
    if arm=='reader':
        for field,bad in [('matched_fp32_sha256',None),('access_models',[]),('same_geometry_control','self')]:
            r=row(arm,key,'screen',5);r[field]=bad
            with pytest.raises(ValueError):runner.parse_row(r,arm,1024,5120,key,'screen',5)
        r=row(arm,key,'screen',5);r['immutable_control_sha256']=None if spec.lookup(key).immutable_parent else 'a'*64
        with pytest.raises(ValueError):runner.parse_row(r,arm,1024,5120,key,'screen',5)


@pytest.mark.parametrize('key',['p2-w10-r1-u0','p4-w20-r0-u1','p4-w24-r1-u0'])
@pytest.mark.parametrize('dirty',[False,True])
def test_ffi_blue_and_actual_immutable_parent_then_candidate(key,dirty):
    c=spec.lookup(key);b=runner.MediumBench.__new__(runner.MediumBench)
    b.medium_allowed={c.recipe:c};b.n=1024;b.k=5120;b.a=1;b.output=116;b.output_base=100;b.output_bytes=8;b.copies=2
    b.weight_pointers=[(10,20),(30,40)];b.r=SimpleNamespace(stream=3,fill=lambda *args:None);b.error=lambda:0.
    gold=np.array([1.,2.],dtype='<f4').tobytes();state=[b''];calls=[]
    def launch(*args):
        calls.append(args);state[0]=gold if args[0] or not dirty else bytes([gold[0]^1])+gold[1:];return 0
    def frozen(*args):calls.append(('immutable',*args));state[0]=gold;return 0
    b.medium_launch=launch;b.latency_launch={'affine':frozen};b.sdk=SimpleNamespace(download=lambda *args:state[0])
    if dirty:
        with pytest.raises(ValueError,match='same-order'):b.direct(c.recipe)
    else:assert b.direct(c.recipe)==(gold,0.)
    assert calls[0]==(1,*c.recipe,1,10,20,116,3)
    assert calls[-1]==(0,*c.recipe,1,10,20,116,3)
    assert len(calls)==(3 if c.immutable_parent else 2)
    if c.immutable_parent:assert calls[1]==('immutable',0,*c.parent.recipe[1:],1,10,20,116,3)
    b.invoke(c.recipe,3);assert calls[-1]==(0,*c.recipe,1,30,40,116,3)


def test_profiles_avoid_same_geometry_aliases_but_keep_two_anchors():
    selected=['p2-w10-r1-u0','p2-w10-r0-u1','p4-w20-r1-u0'];med=dict(zip(selected,[5.4,5.41,5.42]))
    assert runner.profile_candidates(selected,med)==[selected[0],selected[2]]
    assert {'kpack-p2','kpack-p4'}.issubset(runner.ANCHORS)


@pytest.mark.parametrize('fault',[False,True])
def test_complete_one_shape_campaign_and_failed_cell_only_resume(tmp_path,monkeypatch,fault):
    (tmp_path/'q12-n1024-k5120-e1-c1.npz').write_bytes(b'host fixture')
    p=ROOT/'prebuilt/ppu0010'
    a=SimpleNamespace(sdk=tmp_path,candidate=PACKAGE,latency=p/'q4-small-latency-v1',followup=p/'q4-reader-followup-v1',
        reuse=p/'q4-reader-reuse-v1',config_bundle=p/'q4-config-sweep-v1',previous=p/'q4-cold-shapes-v1',
        controls=p/'q4-h800-port-v1',bundle=p/'q4-simt-ab-v1',fixtures=tmp_path,output=tmp_path/'results',
        l2_bytes=67108864,skip_acu=False,acu=tmp_path/'acu')
    monkeypatch.setattr(spec,'verify',lambda *args,**kw:dict(runtime={},source_hashes={}))
    monkeypatch.setattr(runner,'probe_device',lambda a:dict(l2_bytes=67108864))
    monkeypatch.setattr(runner,'acu_launch_command',lambda acu,prefix,cmd:cmd+['--test-report',str(prefix)])
    calls=[];planted=[False]
    def fake(cmd,stream):
        assert '--latency' in cmd and '--followup' in cmd
        arm=cmd[cmd.index('--variant')+1];phase=cmd[cmd.index('--phase')+1];keys=json.loads(cmd[cmd.index('--keys')+1])
        assert Path(cmd[cmd.index('--fixture')+1]).name=='q12-n1024-k5120-e1-c1.npz'
        profile='--profile' in cmd;calls.append((arm,phase,keys))
        for key in keys:
            if fault and not planted[0] and phase=='screen' and key=='p4-w24-r1-u0':
                planted[0]=True;stream.write(runner.FAIL_PREFIX+json.dumps(dict(variant=arm,key=key,phase=phase,shape=[1,1024,5120],error='plant'))+'\n');return 1
            stream.write(runner.PREFIX+json.dumps(row(arm,key,phase,0 if profile else 5 if phase=='screen' else 15,profile))+'\n')
        if profile:Path(cmd[cmd.index('--test-report')+1]+'.acurep').write_bytes(b'host report')
        return 0
    monkeypatch.setattr(runner,'execute',fake)
    assert runner.run(a)==int(fault)
    s=json.loads((a.output/'summary.json').read_text());assert len(s['cases'])==1 and len(s['profiles'])==6
    c=s['cases'][0]
    assert len(c['screen'])+len(c['screen_anchors'])+sum(len(x) for x in c['records'].values())==68-int(fault)
    previous=len(calls);assert runner.run(a)==0
    assert len(calls)==previous+int(fault)
    if fault:assert calls[-1]==('reader','screen',['p4-w24-r1-u0'])


def test_prebuilt_has_exact_fold_bounds_and_preserves_frozen_packages():
    p=ROOT/'prebuilt/ppu0010'
    m=spec.verify(PACKAGE,*(p/x for x in ('q4-small-latency-v1','q4-reader-followup-v1','q4-reader-reuse-v1',
        'q4-config-sweep-v1','q4-cold-shapes-v1','q4-h800-port-v1','q4-simt-ab-v1')))
    stats=json.loads((PACKAGE/'isa-stats.json').read_text())
    assert len(stats)==22 and len(m['payloads'])==1 and not m['device_validated'] and not m['production_changed']
    for c in spec.inventory():
        r=stats[c.key];assert r['code_fastpath_present'] and r['fp32_fma_present']
        assert r['operations']['s.blksyn.defer']==1
        if c.reduction or c.unsigned:assert r['cta_fold_shared_load_instructions']==c.geometry()['cta_fold_rounds']
    before=stats['p2-w10-r0-u0'];after=stats['p2-w10-r1-u0']
    assert (before['cta_fold_shared_load_instructions'],after['cta_fold_shared_load_instructions'])==(16,3)
    assert (before['cta_fold_conditional_branches'],after['cta_fold_conditional_branches'])==(18,3)
    exports=subprocess.check_output(['nm','-D','--defined-only',str(PACKAGE/spec.PAYLOAD)],text=True)
    assert ' q4_medium_run' in exports and ' q4_ppu_probe' in exports
    script=ROOT/'tools/run_q4_medium_refine_ppu_box.sh';subprocess.run(['bash','-n',str(script)],check=True)
    assert '\n(\n' in script.read_text() and 'compile=NONE JIT=NONE' in script.read_text()
    tree=ast.parse((ROOT/'dev/gemv_ppu/run_medium_refine.py').read_text())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    assert all(not (isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='SDK') for n in ast.walk(run))
