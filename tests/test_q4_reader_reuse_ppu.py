"""Reader transport/metadata proofs and host runner tests, not PPU admission."""
import ast
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from dev.gemv_ppu import reader_reuse as spec
from dev.gemv_ppu import run_reader_reuse as runner
from dev.gemv_ppu.config_space import source as old_source
from dev.gemv_ppu.access_pattern import warp_pattern

ROOT=spec.ROOT


def test_bounded_factorial_retains_both_confirmed_winners():
    assert sum(len(spec.inventory(n,k)) for n,k in spec.SHAPES)==32
    assert spec.selected(5120,8192).args==(4,8,4)
    assert spec.selected(8192,5120).args==(8,10,4)
    for case in spec.plan()['cases']:
        assert len(case['readers'])==16
        assert {r['switches']['header32'] for r in case['readers']}=={True,False}
        assert len({r['key'] for r in case['readers']})==16


@pytest.mark.parametrize('n,k',spec.SHAPES)
def test_generated_body_preserves_original_and_dot_order(n,k):
    old=old_source(n,k);s=spec.source(n,k)
    seam='template<int Columns,int Warps,int P,int N,int K,bool Early>'
    start=old.index(seam);end=old.index('\n}\n',start)+3
    assert s.startswith(old[:end])
    clone=s[s.index('template<int Variant,int Columns'):]
    entry=s[s.index(f'extern "C" int q4_reader_run_{n}_{k}('):]
    assert entry.count('>>>(')==16
    for marker in ('float2 dot[Pairs]{}','a_sum+=(av.x+av.y)+(av.z+av.w);',
        'dot[p].x=fmaf(ax[r],v.x,dot[p].x);','dot[p].y=fmaf(ax[r],v.y,dot[p].y);',
        'total[p].x+=fmaf(s0.x,dot[p].x,s0.y*a_sum);','total[p].y+=fmaf(s1.x,dot[p].y,s1.y*a_sum);'):
        assert marker in clone
    # The entire output reduction and packed-B load block are unchanged.
    base=old[start:end]
    assert base[base.index('    constexpr int TileN=Columns*P;'):] in clone
    bstart=base.index('            uint32_t words[4][Pairs];')
    bend=base.index('            #pragma unroll\n            for(int slot',bstart)
    assert base[bstart:bend] in clone
    for r in spec.inventory(n,k):
        v,c,w,p=r.args
        assert f'q4_reader_reuse<{v},{c},{w},{p},{n},{k}><<<{n//(c*p)},{w*32},0,' in clone


@pytest.mark.parametrize('n,k,config',[(n,k,c) for n,k in spec.SHAPES for c in spec.GEOMETRIES[n,k]])
def test_cooperative_A_and_metadata_owners_for_every_warp_pass(n,k,config):
    c,w,p=config;workers=w*32//c;tile=c*p;groups=k//32
    # Tags are bit payloads, not arithmetic values; a bad shuffle owner cannot
    # hide behind a numerically degenerate A or matching producer layout.
    a=(np.arange(k,dtype=np.uint16)^np.uint16(0x936d))
    units=np.arange((k//256)*n*4,dtype=np.uint32).reshape(k//256,n,4)^np.uint32(0xa5371286)
    faults=0
    for block in (0,n//tile-1):
        for pas in range((groups+workers-1)//workers):
            for warp in range(w):
                gs=np.array([pas*workers+(warp*32+lane)//c for lane in range(32)])
                assert np.all(gs<groups)  # declared geometries have whole active warps
                assert len(set((gs//8).tolist()))==1
                chunks=np.array([a[g*32+(lane%c)*(32//c):g*32+(lane%c+1)*(32//c)] for lane,g in enumerate(gs)])
                loaded_units=np.zeros((32,4),dtype=np.uint32)
                for lane in range(tile):loaded_units[lane]=units[gs[lane]//8,block*tile+lane]
                for lane,g in enumerate(gs):
                    for off in range(0,32,4):
                        owner=(lane&~(c-1))+off//(32//c);offset=off%(32//c)
                        actual=chunks[owner,offset:offset+4]
                        assert np.array_equal(actual,a[g*32+off:g*32+off+4])
                        faults+=int(not np.array_equal(chunks[owner^1,offset:offset+4],actual))
                    for pp in range(p):
                        owner=(lane%c)*p+pp
                        assert np.array_equal(loaded_units[owner],units[g//8,block*tile+owner])
    assert faults>0


def test_header32_matches_all_eight_packed_bit_fields_and_boundary_negatives():
    u=np.random.default_rng(3137).integers(0,2**32,size=(131072,4),dtype=np.uint32)
    wide=u.astype(np.uint64);negatives=0
    for g in range(8):
        run=(wide[:,2]>>16)|(wide[:,3]<<16) if g&4 else wide[:,1]|((wide[:,2]&65535)<<32)
        shift=6*(g&3);want_sc=(run>>shift)&63;want_mn=(run>>(24+shift))&63
        scales=(u[:,2]>>16)|(u[:,3]<<16) if g&4 else u[:,1]
        mins=u[:,3]>>8 if g&4 else (u[:,1]>>24)|(u[:,2]<<8)
        assert np.array_equal((scales>>shift)&63,want_sc)
        assert np.array_equal((mins>>shift)&63,want_mn)
        assert set(want_sc)==set(range(64)) and set(want_mn)==set(range(64))
        negatives+=np.count_nonzero(((mins>>(shift+1))&63)!=want_mn)
    assert negatives>0


@pytest.mark.parametrize('n,k',spec.SHAPES)
def test_pattern_load_models_change_only_selected_operands(n,k):
    for c in spec.GEOMETRIES[n,k]:
        cfg=spec.Config(*c);base=spec.access(spec.Reader(0,cfg),n,k)
        for v in range(8):
            r=spec.Reader(v,cfg);model=spec.access(r,n,k)
            assert model['first_warp']['streams']['B_vector']==base['first_warp']['streams']['B_vector']
            traffic=model['logical_lane_bytes_per_call'];before=base['logical_lane_bytes_per_call']
            assert before['A']/traffic['A']==(cfg.columns if v&1 else 1)
            assert before['metadata']/traffic['metadata']==(32//cfg.columns if v&2 else 1)
            if v&1:
                a=model['first_warp']['streams']['A_cooperative_chunk']['granules']['64']
                assert a['duplicate_factor']==1 and a['unique_sector_utilization']==1
            if v&2:
                u=model['first_warp']['streams']['metadata_cooperative_unit']['granules']['64']
                assert u['duplicate_factor']==1 and u['unique_sector_utilization']==1
            offset=spec.access(r,n,k,dict(A=16,B=16,metadata=16))
            assert offset['first_warp']['base_mod128']==dict(A=16,B=16,metadata=16)


def row(variant,n,k,key,phase,samples,profile=False):
    weight=n*k*9//16;copies=math.ceil(2.25*67108864/weight)
    r=dict(status='PASS',arm=variant,variant=variant,config_key=key,phase=phase,shape=[1,n,k],mode='rotating',
        recipe=list(runner.recipe(variant,n,k,key)),error=1e-8,zero_code_negative='PASS',zero_a_check='PASS',
        copies=copies,weight_bytes=weight,device=dict(l2_bytes=67108864),
        calls_per_graph=max(2,math.ceil(32/copies))*copies,output_type='F32',inter_cta_split=1,launches_per_call=1,
        weight_arithmetic='FP32_GROUP_AFFINE' if variant in ('reader','baseline') else 'PER_WEIGHT_FP16',
        storage='RAW_GGUF' if variant=='raw-reference' else 'XPLANE' if variant=='xplane' else 'CANONICAL_KPACK4',
        timing_scope='RESIDENT_FULL_CALL_NO_HOST_PACK_NO_JIT',cache_scope='ACU_FORCED_COLD' if profile else 'ROTATING_AT_LEAST_2_25_L2',
        samples_us=[10.]*samples,median_us=10. if samples else None)
    if variant=='reader':
        reader=spec.lookup(n,k,key)
        r.update(geometry=reader.config.geometry(n,k),reader_switches=reader.switches,matched_fp32_sha256='a'*64,
            same_config_control=reader.config.key,code_dequant='LOP3_HALF2',access_models=[spec.access(reader,n,k)])
        r['samples_us']=[10.-reader.variant*.05]*samples;r['median_us']=r['samples_us'][0] if samples else None
    return r


@pytest.mark.parametrize('variant',(*runner.ANCHORS,'reader'))
def test_receipts_require_independent_correctness_and_matched_control(variant):
    n,k=spec.SHAPES[0];key=spec.inventory(n,k)[0].key if variant=='reader' else 'control'
    for profile,samples in ((False,15),(True,0)):
        r=row(variant,n,k,key,'r0',samples,profile)
        assert runner.parse_row(r,variant,n,k,key,'r0',samples,profile)==r
    for field,bad in [('error',float('nan')),('zero_code_negative','SKIP'),('samples_us',[float('nan')]*15),('recipe',[0,4,8,4,1])]:
        r=row(variant,n,k,key,'r0',15);r[field]=bad
        with pytest.raises(ValueError):runner.parse_row(r,variant,n,k,key,'r0',15)
    if variant=='reader':
        for field,bad in [('matched_fp32_sha256','self'),('same_config_control','wrong'),('access_models',[]),('reader_switches',{})]:
            r=row(variant,n,k,key,'r0',15);r[field]=bad
            with pytest.raises(ValueError):runner.parse_row(r,variant,n,k,key,'r0',15)


def test_same_geometry_effects_and_zero_regression_rule():
    n,k=spec.SHAPES[0]
    records={key:[dict(median_us=10.)]*6 for key in (*runner.ANCHORS,*(r.key for r in spec.inventory(n,k)))}
    key=spec.inventory(n,k)[1].key;records[key]=[dict(median_us=9.)]*6
    r=runner.summarize(n,k,records)
    assert r['selected']==key and r['reference_verdict']=='NOT_SLOWER_THAN_REFERENCE'
    assert r['same_geometry_delta_pct'][key]==pytest.approx(-10.)
    records['raw-reference']=[dict(median_us=8.99)]*6
    assert runner.summarize(n,k,records)['reference_verdict']=='REF_REGRESSION'
    records[key]=records[key][:-1]
    assert runner.summarize(n,k,records)['reference_verdict']=='INCOMPLETE'


def test_real_variant_config_and_rotated_pointers_are_passed():
    b=runner.ReuseBench.__new__(runner.ReuseBench);b.reader_allowed={(7,4,8,4)}
    b.copies=2;b.a=1;b.output=2;b.r=SimpleNamespace(stream=3);b.weight_pointers=[(10,20),(30,40)]
    calls=[];b.reuse_launch=lambda *args:calls.append(args) or 0
    assert b.invoke((7,4,8,4),3)==0 and calls==[(7,4,8,4,1,30,40,2,3)]
    with pytest.raises(ValueError):b.invoke((8,4,8,4))


@pytest.mark.parametrize('corrupt',[False,True])
def test_raw_bit_gate_really_reads_the_immutable_control_before_candidate(corrupt):
    b=runner.ReuseBench.__new__(runner.ReuseBench)
    b.output_base=100;b.output=116;b.output_bytes=8;b.a=1;b.weight_pointers=[(10,20)]
    b.r=SimpleNamespace(stream=3,fill=lambda *args:None);b.error=lambda:0.
    state=[b''];calls=[]
    gold=np.array([1.,2.],dtype='<f4').tobytes()
    def old(*args):calls.append(('old',args));state[0]=gold;return 0
    def new(*args,**kw):
        calls.append(('new',args));state[0]=bytes([gold[0]^1])+gold[1:] if corrupt else gold;return 0
    b.config_launch=old;b.invoke=new;b.sdk=SimpleNamespace(download=lambda *args:state[0]);b.matched=None
    if corrupt:
        with pytest.raises(ValueError,match='immutable same-config'):b.direct((7,4,8,4))
    else:
        data,err=b.direct((7,4,8,4));assert data==gold and err==0 and len(b.matched)==64
    assert calls[0]==('old',(4,8,4,1,10,20,116,3))
    assert calls[1][0]=='new'


@pytest.mark.parametrize('fault',[False,True])
def test_two_shape_whole_campaign_profiles_and_exact_missing_cell_resume(tmp_path,monkeypatch,fault):
    for n,k in spec.SHAPES:(tmp_path/f'q12-n{n}-k{k}-e1-c1.npz').write_bytes(b'host-only')
    a=SimpleNamespace(sdk=tmp_path,candidate=ROOT/'prebuilt/ppu0010/q4-reader-reuse-v1',
        config_bundle=ROOT/'prebuilt/ppu0010/q4-config-sweep-v1',previous=ROOT/'prebuilt/ppu0010/q4-cold-shapes-v1',
        controls=ROOT/'prebuilt/ppu0010/q4-h800-port-v1',bundle=ROOT/'prebuilt/ppu0010/q4-simt-ab-v1',
        fixtures=tmp_path,output=tmp_path/'results',l2_bytes=67108864,skip_acu=False,acu=tmp_path/'fake-acu')
    monkeypatch.setattr(runner,'verify',lambda *a,**kw:dict(runtime={},source_hashes={}))
    monkeypatch.setattr(runner,'probe_device',lambda a:dict(l2_bytes=67108864))
    monkeypatch.setattr(runner,'acu_launch_command',lambda acu,prefix,cmd:cmd+['--test-report',str(prefix)])
    jobs=[];planted=[False]
    def fake_execute(cmd,stream):
        v=cmd[cmd.index('--variant')+1];phase=cmd[cmd.index('--phase')+1];keys=json.loads(cmd[cmd.index('--keys')+1])
        fixture=Path(cmd[cmd.index('--fixture')+1]);n,k=next((n,k) for n,k in spec.SHAPES if fixture.name==f'q12-n{n}-k{k}-e1-c1.npz')
        profile='--profile' in cmd;jobs.append((n,k,v,phase,keys))
        for key in keys:
            if fault and not planted[0] and phase=='r0' and v=='reader' and key.startswith('v1-'):
                planted[0]=True
                stream.write(runner.FAIL_PREFIX+json.dumps(dict(variant=v,key=key,phase=phase,shape=[1,n,k],error='plant'))+'\n')
                return 1
            stream.write(runner.PREFIX+json.dumps(row(v,n,k,key,phase,0 if profile else 15,profile))+'\n')
        if profile:Path(cmd[cmd.index('--test-report')+1]+'.acurep').write_bytes(b'host profile receipt')
        return 0
    monkeypatch.setattr(runner,'execute',fake_execute)
    assert runner.run(a)==int(fault)
    r=json.loads((a.output/'summary.json').read_text())
    assert len(r['profiles'])==24 and sum(len(rs) for c in r['cases'] for rs in c['records'].values())==228-int(fault)
    previous=len(jobs);assert runner.run(a)==0
    assert len(jobs)==previous+int(fault)
    if fault:assert len(jobs[-1][-1])==1 and jobs[-1][3]=='r0'


def test_prebuilt_native_variants_and_noncompiling_box_handoff():
    p=ROOT/'prebuilt/ppu0010/q4-reader-reuse-v1'
    m=spec.verify(p,ROOT/'prebuilt/ppu0010/q4-config-sweep-v1',ROOT/'prebuilt/ppu0010/q4-cold-shapes-v1',ROOT/'prebuilt/ppu0010/q4-h800-port-v1',ROOT/'prebuilt/ppu0010/q4-simt-ab-v1')
    assert not m['device_validated'] and not m['production_changed']
    stats=json.loads((p/'isa-stats.json').read_text());assert len(stats)==32
    for n,k in spec.SHAPES:
        exports=subprocess.check_output(['nm','-D','--defined-only',str(p/spec.payload(n,k))],text=True)
        assert f' q4_reader_run_{n}_{k}' in exports and ' q4_ppu_probe' in exports
        for reader in spec.inventory(n,k):
            st=stats[f'n{n}-k{k}-{reader.key}']
            assert st['code_fastpath_present'] and st['fp32_fma_present']
            if reader.variant&4:assert st['operations'].get('v.shrl.b64',0)==0
            if reader.variant&3:assert st['operations'].get('v.shuffle.idx.b32',0)>0
    script=ROOT/'tools/run_q4_reader_reuse_ppu_box.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    assert '\n(\n' in script.read_text() and script.read_text().rstrip().endswith(')')
    parsed=ast.parse((ROOT/'dev/gemv_ppu/run_reader_reuse.py').read_text())
    run=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    names=[n.func.id for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    assert 'SDK' not in names and 'probe_device' in names
