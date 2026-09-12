"""Host/ISA checks for small Q4 readers; actual PPU timing remains a box gate."""
import ast
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from dev.gemv_ppu import small_latency as spec
from dev.gemv_ppu import run_small_latency as runner

ROOT=spec.ROOT
PACKAGE=ROOT/'prebuilt/ppu0010/q4-small-latency-v1'


def test_only_two_remaining_shapes_and_actual_latest_baseline():
    assert spec.SHAPES==((512,2048),(1024,5120))
    assert [len(spec.inventory(n,k)) for n,k in spec.SHAPES]==[32,48]
    assert spec.baseline(512,2048).key=='meta-h0-n8-w16'
    assert spec.baseline(1024,5120).key=='v4-c4-w20-p4'
    assert runner.recipe('baseline',1024,5120,'control')==(4,4,20,4)
    assert spec.plan()['reference_limit_pct']==5
    assert len(spec.plan()['prior_closed_shapes'])==4
    for n,k in spec.SHAPES:
        keys=[c.key for c in spec.inventory(n,k)]
        assert len(keys)==len(set(keys))
        for key in keys:
            c=spec.lookup(n,k,key)
            assert c.geometry(n,k)['grid']*c.geometry(n,k)['tile_n']==n
            assert c.geometry(n,k)['threads']<=1024
            assert c.geometry(n,k)['inter_cta_split']==1


def test_mux_truth_table_and_all_packed_fields_match_independent_64bit_decode():
    u=np.random.default_rng(9113).integers(0,2**32,size=(32768,4),dtype=np.uint32)
    a=u.astype(np.uint64)
    for group in range(8):
        mask=np.uint32(0xffffffff if group&4 else 0)
        hi=(u[:,2]>>16)|(u[:,3]<<16);lo=u[:,1]
        # 0xca: (A & B) | (~A & C), with A=mask, B=high, C=low.
        scales=(mask&hi)|(~mask&lo)
        mins=(mask&(u[:,3]>>8))|(~mask&((u[:,1]>>24)|(u[:,2]<<8)))
        wide=(a[:,2]>>16)|(a[:,3]<<16) if group&4 else a[:,1]|((a[:,2]&65535)<<32)
        shift=6*(group&3)
        assert np.array_equal((scales>>shift)&63,(wide>>shift)&63)
        assert np.array_equal((mins>>shift)&63,(wide>>(24+shift))&63)
        assert np.count_nonzero(((mins>>(shift+1))&63)!=((wide>>(24+shift))&63))>0
    for mask in (0,1):
        for hi in (0,1):
            for lo in (0,1):
                assert (0xca>>((mask<<2)|(hi<<1)|lo))&1 == (hi if mask else lo)


def perm(a,b,selector):
    ab=[(a>>(8*i))&255 for i in range(4)]+[(b>>(8*i))&255 for i in range(4)]
    return sum(ab[(selector>>(4*i))&7]<<(8*i) for i in range(4))


def test_register_A_transpose_matches_direct_strided_half_values_and_negatives():
    rng=np.random.default_rng(19983)
    raw=rng.integers(0,65536,size=(4,32),dtype=np.uint16)
    x=np.array([int(raw[l//8,4*(l%8)])|(int(raw[l//8,4*(l%8)+1])<<16) for l in range(32)],dtype=np.uint32)
    y=np.array([int(raw[l//8,4*(l%8)+2])|(int(raw[l//8,4*(l%8)+3])<<16) for l in range(32)],dtype=np.uint32)
    def swap(v,bit):
        return np.array([perm(int(v[l]),int(v[l^bit]),0x3276 if l&bit else 0x5410) for l in range(32)],dtype=np.uint32)
    x=swap(x,1);y=swap(y,1)
    exchanged=np.array([(x if (l^2)&2 else y)[l^2] for l in range(32)],dtype=np.uint32)
    for l in range(32):
        if l&2:x[l]=exchanged[l]
        else:y[l]=exchanged[l]
    x=swap(x,4);y=swap(y,4)
    got=np.stack([x&65535,y&65535,x>>16,y>>16],axis=1)
    want=np.array([[raw[l//8,l%8+8*s] for s in range(4)] for l in range(32)])
    assert np.array_equal(got,want)
    assert not np.array_equal(got,want[:,[0,2,1,3]])


@pytest.mark.parametrize('n,k',spec.SHAPES)
def test_each_geometry_covers_the_canonical_words_once(n,k):
    geometries={(c.family,c.warps,c.columns,c.values):c for c in spec.inventory(n,k)}
    for c in geometries.values():
        tile=c.geometry(n,k)['tile_n'];seen=np.zeros((k//4,tile),dtype=np.uint8)
        for pass_id in range(c.geometry(n,k)['k_passes']):
            for tid in range(c.warps*32):
                if c.family=='affine':
                    group=pass_id*(c.warps*32//c.columns)+tid//c.columns
                    if group>=k//32:continue
                    cols=range((tid%c.columns)*c.values,(tid%c.columns+1)*c.values)
                    indices=range(group*8,group*8+8)
                else:
                    width=8 if c.family=='meta' else 4
                    chunk=pass_id*c.warps+tid//32
                    if chunk>=k//(32*(32//width)):continue
                    group=chunk*(32//width)+(tid%32)//width
                    residue=(tid%width)*(8//width)
                    indices=range(group*8+residue,group*8+residue+8//width)
                    cols=range(width)
                for ix in indices:
                    for col in cols:seen[ix,col]+=1
        assert np.all(seen==1),(n,k,c.key,np.unique(seen))
        # First and last N tile are aligned and stay inside the same fixed artifact.
        for start in (0,n-tile):
            assert start%tile==0 and 2*((k//4-1)*n+start+tile)<=n*k//2


def test_source_preserves_controls_and_exact_dot_reduction_order():
    for n,k in spec.SHAPES:
        for family in {c.family for c in spec.inventory(n,k)}:
            source=spec.source(n,k,family)
            assert spec.original_kernel(family) in source
            assert source.count('if(control) ')==sum(c.family==family for c in spec.inventory(n,k))
    old=spec.original_kernel('residue2');new=spec.residue_source()
    # All arithmetic from the per-output dot through group/warp/CTA reduction is untouched.
    assert old[old.index('        float dot[4]{};'):]==new[new.index('        float dot[4]{};'):]
    old=spec.original_kernel('affine');new=spec.affine_source()
    assert old[old.index('    constexpr int TileN=Columns*P;'):]==new[new.index('    constexpr int TileN=Columns*P;'):]
    for call in ('a_sum+=(av.x+av.y)+(av.z+av.w);','dot[p].x=fmaf(ax[r],v.x,dot[p].x);',
        'dot[p].y=fmaf(ax[r],v.y,dot[p].y);','total[p].x+=fmaf(s0.x,dot[p].x,s0.y*a_sum);'):
        assert call in old and call in new
    assert '(act,g*32+slot*8+h*4)' in new


def test_address_models_name_shared_offsets_separately_and_keep_B_bytes():
    for n,k in spec.SHAPES:
        for c in spec.inventory(n,k):
            zero=spec.access(c,n,k);off=spec.access(c,n,k,dict(A=16,B=32,metadata=48))
            assert zero['logical_global_lane_bytes']['B']==n*k//2
            for name,r in zero['streams'].items():
                delta=0 if r['space']=='SHARED' else 32 if name.startswith('B_') else 48 if name.startswith('units') else 16
                assert off['streams'][name]['lane_byte_addresses']==[x+delta for x in r['lane_byte_addresses']]
            if c.family=='meta':assert zero['streams']['B_r0']['granules']['64']['unique_sector_utilization']==.25
            if c.family=='residue2':assert zero['streams']['B_r0']['granules']['64']['unique_sector_utilization']==.125


def row(variant,n,k,key,phase,samples,profile=False):
    weight=n*k*9//16;copies=math.ceil(2.25*67108864/weight)
    value=10.
    r=dict(status='PASS',arm=variant,variant=variant,config_key=key,phase=phase,shape=[1,n,k],mode='rotating',
        recipe=list(runner.recipe(variant,n,k,key)),error=1e-8,zero_code_negative='PASS',zero_a_check='PASS',
        copies=copies,weight_bytes=weight,device=dict(l2_bytes=67108864),calls_per_graph=max(2,math.ceil(32/copies))*copies,
        output_type='F32',inter_cta_split=1,launches_per_call=1,weight_arithmetic=runner.arithmetic(variant,n,k,key),
        storage='RAW_GGUF' if variant=='raw-reference' else 'XPLANE' if variant=='xplane' else 'CANONICAL_KPACK4',
        timing_scope='RESIDENT_FULL_CALL_NO_HOST_PACK_NO_JIT',cache_scope='ACU_FORCED_COLD' if profile else 'ROTATING_AT_LEAST_2_25_L2')
    if variant=='reader':
        c=spec.lookup(n,k,key);value=9.+spec.inventory(n,k).index(c)*.001
        r.update(family=c.family,geometry=c.geometry(n,k),matched_fp32_sha256='a'*64,code_dequant='LOP3_HALF2',
            immutable_control_sha256='a'*64 if runner.immutable_match_required(c,n,k) else None,
            same_geometry_control='UNCHANGED_BODY_IDENTICAL_DOT_REDUCTION',access_models=[spec.access(c,n,k)])
    r.update(samples_us=[value]*samples,median_us=value if samples else None)
    return r


@pytest.mark.parametrize('variant,n,k,key',[('baseline',1024,5120,'control'),('raw-reference',512,2048,'control'),
    ('reader',512,2048,'meta-h1-l1-a0-w16'),('reader',1024,5120,'meta-h1-l1-a0-w20'),
    ('reader',512,2048,'residue2-h1-l1-a0-w8'),('reader',1024,5120,'affine-h1-l1-a0-w20-c4-p4')])
def test_parser_requires_real_correctness_current_winner_and_immutable_scope(variant,n,k,key):
    for profile,samples in ((False,5),(False,15),(True,0)):
        r=row(variant,n,k,key,'screen',samples,profile)
        assert runner.parse_row(r,variant,n,k,key,'screen',samples,profile)==r
    for field,bad in [('recipe',[0]),('zero_a_check','SKIP'),('error',float('nan')),('samples_us',[0.]*5)]:
        r=row(variant,n,k,key,'screen',5);r[field]=bad
        with pytest.raises(ValueError):runner.parse_row(r,variant,n,k,key,'screen',5)
    if variant=='reader':
        for field,bad in [('matched_fp32_sha256',None),('same_geometry_control','self'),('access_models',[])]:
            r=row(variant,n,k,key,'screen',5);r[field]=bad
            with pytest.raises(ValueError):runner.parse_row(r,variant,n,k,key,'screen',5)
        r=row(variant,n,k,key,'screen',5)
        r['immutable_control_sha256']='a'*64 if r['immutable_control_sha256'] is None else None
        with pytest.raises(ValueError):runner.parse_row(r,variant,n,k,key,'screen',5)


@pytest.mark.parametrize('family,n,k',[('meta',512,2048),('meta',1024,5120),('residue2',512,2048),('affine',1024,5120)])
@pytest.mark.parametrize('corrupt',[False,True])
def test_actual_FFI_control_then_immutable_then_candidate(family,n,k,corrupt):
    c=next(c for c in spec.inventory(n,k) if c.family==family)
    b=runner.LatencyBench.__new__(runner.LatencyBench)
    b.latency_allowed={c.recipe:c};b.n=n;b.k=k;b.a=1;b.output=116;b.output_base=100;b.output_bytes=8;b.copies=2
    b.weight_pointers=[(10,20),(30,40)];b.r=SimpleNamespace(stream=3,fill=lambda *args:None);b.error=lambda:0.
    state=[b''];calls=[];gold=np.array([1.,2.],dtype='<f4').tobytes()
    def fn(*args):
        calls.append(args);state[0]=gold if args[0] else (bytes([gold[0]^1])+gold[1:]) if corrupt else gold;return 0
    def immutable(*args):calls.append(('immutable',*args));state[0]=gold;return 0
    b.latency_launch={family:fn};b.followup_launch={'affine':immutable,'small':immutable}
    b.sdk=SimpleNamespace(download=lambda *args:state[0])
    if corrupt:
        with pytest.raises(ValueError,match='same-geometry'):b.direct(c.recipe)
    else:
        data,err=b.direct(c.recipe);assert data==gold and err==0
    assert calls[0]==(1,*c.recipe[1:],1,10,20,116,3)
    assert calls[-1]==(0,*c.recipe[1:],1,10,20,116,3)
    assert len(calls)==(3 if runner.immutable_match_required(c,n,k) else 2)
    b.invoke(c.recipe,3)
    assert calls[-1]==(0,*c.recipe[1:],1,30,40,116,3)


@pytest.mark.parametrize('fault',[False,True])
def test_whole_campaign_profiles_and_only_missing_resume(tmp_path,monkeypatch,fault):
    for n,k in spec.SHAPES:(tmp_path/f'q12-n{n}-k{k}-e1-c1.npz').write_bytes(b'host only')
    p=ROOT/'prebuilt/ppu0010'
    a=SimpleNamespace(sdk=tmp_path,candidate=PACKAGE,followup=p/'q4-reader-followup-v1',reuse=p/'q4-reader-reuse-v1',
        config_bundle=p/'q4-config-sweep-v1',previous=p/'q4-cold-shapes-v1',controls=p/'q4-h800-port-v1',bundle=p/'q4-simt-ab-v1',
        fixtures=tmp_path,output=tmp_path/'results',l2_bytes=67108864,skip_acu=False,acu=tmp_path/'acu')
    monkeypatch.setattr(spec,'verify',lambda *args,**kw:dict(runtime={},source_hashes={}))
    monkeypatch.setattr(runner,'probe_device',lambda a:dict(l2_bytes=67108864))
    monkeypatch.setattr(runner,'acu_launch_command',lambda acu,prefix,cmd:cmd+['--test-report',str(prefix)])
    jobs=[];planted=[False]
    def fake(cmd,stream):
        assert '--followup' in cmd
        v=cmd[cmd.index('--variant')+1];phase=cmd[cmd.index('--phase')+1];keys=json.loads(cmd[cmd.index('--keys')+1])
        fn=Path(cmd[cmd.index('--fixture')+1]).name;n,k=next((n,k) for n,k in spec.SHAPES if fn==f'q12-n{n}-k{k}-e1-c1.npz')
        profile='--profile' in cmd;jobs.append((n,k,v,phase,keys))
        for key in keys:
            if fault and not planted[0] and phase=='screen' and key=='residue2-h1-l1-a2-w4':
                planted[0]=True;stream.write(runner.FAIL_PREFIX+json.dumps(dict(variant=v,key=key,phase=phase,shape=[1,n,k],error='plant'))+'\n');return 1
            stream.write(runner.PREFIX+json.dumps(row(v,n,k,key,phase,0 if profile else 5 if phase=='screen' else 15,profile))+'\n')
        if profile:Path(cmd[cmd.index('--test-report')+1]+'.acurep').write_bytes(b'host report')
        return 0
    monkeypatch.setattr(runner,'execute',fake)
    assert runner.run(a)==int(fault)
    s=json.loads((a.output/'summary.json').read_text())
    assert len(s['profiles'])==10 and len(s['cases'])==2
    assert sum(len(c['screen'])+len(c['screen_anchors'])+sum(len(v) for v in c['records'].values()) for c in s['cases'])==158-int(fault)
    previous=len(jobs);assert runner.run(a)==0
    assert len(jobs)==previous+int(fault)
    if fault:assert jobs[-1][-1]==['residue2-h1-l1-a2-w4']


def test_prebuilt_and_native_load_wait_seam_not_just_source_assertion():
    p=ROOT/'prebuilt/ppu0010'
    m=spec.verify(PACKAGE,p/'q4-reader-followup-v1',p/'q4-reader-reuse-v1',p/'q4-config-sweep-v1',
        p/'q4-cold-shapes-v1',p/'q4-h800-port-v1',p/'q4-simt-ab-v1')
    stats=json.loads((PACKAGE/'isa-stats.json').read_text())
    assert len(stats)==80 and len(m['payloads'])==5 and not m['device_validated'] and not m['production_changed']
    for row in stats.values():assert row['code_fastpath_present'] and row['fp32_fma_present']
    before=stats['n512-k2048-meta-h0-l0-a0-w16'];after=stats['n512-k2048-meta-h0-l1-a0-w16']
    assert before['global_loads_before_first_vld_wait']==1
    assert after['global_loads_before_first_vld_wait']==3
    assert before['operations']==after['operations']
    assert before['operations']['s.blksyn.defer']==1
    for name in m['payloads']:
        exports=subprocess.check_output(['nm','-D','--defined-only',str(PACKAGE/name)],text=True)
        assert ' q4_latency_run_' in exports and ' q4_ppu_probe' in exports
    script=ROOT/'tools/run_q4_small_latency_ppu_box.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    assert '\n(\n' in script.read_text() and 'compile=NONE JIT=NONE' in script.read_text()
    tree=ast.parse((ROOT/'dev/gemv_ppu/run_small_latency.py').read_text())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    assert all(not (isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='SDK') for n in ast.walk(run))
