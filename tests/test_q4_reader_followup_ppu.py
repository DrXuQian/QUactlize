"""Followup inventory, lane contracts, native compilation and orchestration; not PPU admission."""
import ast
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from dev.gemv_ppu import reader_followup as spec
from dev.gemv_ppu import run_reader_followup as runner

ROOT = spec.ROOT
PACKAGE = ROOT/'prebuilt/ppu0010/q4-reader-followup-v1'


def test_bounded_inventory_keeps_exact_old_winners_and_revised_gate():
    assert spec.SHAPES == ((512,2048),(1024,5120),(4096,2048),(4096,4096))
    assert [len(spec.inventory(n,k)) for n,k in spec.SHAPES] == [44,34,40,40]
    assert spec.REFERENCE_LIMIT_PCT == 5
    assert spec.baseline(512,2048).recipe == (0,8,16)  # Width8, not Columns2.
    assert spec.baseline(1024,5120).recipe == (0,4,10,2)
    for n,k in spec.SHAPES:
        rows = spec.inventory(n,k)
        assert spec.baseline(n,k) in rows
        assert len({c.key for c in rows}) == len(rows)
        for c in rows:
            assert spec.lookup(n,k,c.key) == c
            if c.family == 'affine' and c.recipe[1] == 2:
                assert c.recipe[0] in (0,4)
    p = spec.plan()
    assert p['prior_closed_shapes'] == [[5120,8192],[8192,5120]]
    assert p['reference_limit_pct'] == 5
    assert len(p['cases']) == 4


@pytest.mark.parametrize('n,k',spec.SHAPES)
def test_affine_body_unchanged_except_explicit_c2_guard(n,k):
    old = spec.frozen.source(5120,8192)
    body = old[:old.index('extern "C" int q4_reader_run_5120_8192(')]
    body = body.replace('static_assert((Columns==4 || Columns==8) && Workers%8==0 && K%256==0);',
        'static_assert(((Columns==4 || Columns==8) || (Variant&3)==0) && Workers%8==0 && K%256==0);')
    source = spec.source(n,k)
    assert source.startswith(body)
    entry = source[len(body):]
    for c in spec.inventory(n,k):
        if c.family == 'affine':
            v,cols,w,p = c.recipe
            assert f'q4_reader_reuse<{v},{cols},{w},{p},{n},{k}><<<{n//(cols*p)},{w*32},0,' in entry


@pytest.mark.parametrize('n,k',spec.SHAPES)
def test_shuffle_groups_are_whole_warps_including_tail_passes(n,k):
    for cols,warps,p in spec.GEOMETRIES[n,k]:
        workers = warps*32//cols
        groups = k//32
        visited = np.zeros((groups,n),dtype=np.uint8)
        for block in range(n//(cols*p)):
            for pas in range((groups+workers-1)//workers):
                for warp in range(warps):
                    gs = np.array([pas*workers+(warp*32+lane)//cols for lane in range(32)])
                    if np.all(gs >= groups):
                        continue
                    assert np.all(gs < groups)  # only whole inactive tail warps permitted
                    if cols in (4,8):
                        assert len(set(gs//8)) == 1
                        for lane,g in enumerate(gs):
                            for pp in range(p):
                                owner = (lane%cols)*p+pp
                                assert owner < 32 and gs[owner]//8 == g//8
                            for off in range(0,32,4):
                                owner = (lane & ~(cols-1))+off//(32//cols)
                                offset = off%(32//cols)
                                assert gs[owner]*32+(owner%cols)*(32//cols)+offset == g*32+off
                    for lane,g in enumerate(gs):
                        for pp in range(p):
                            visited[g,block*cols*p+(lane%cols)*p+pp] += 1
        assert np.all(visited == 1)


def test_small_clones_only_header_extraction_not_half_rounding_or_reduction():
    old = spec.candidate_source('small')
    start = old.index('template<int Width, int Warps, bool StageA')
    end = old.index('\n}\n',start)+3
    kernel = old[start:end]
    source = spec.small_source()
    assert kernel in source
    expected = kernel.replace('__global__ void q4_cooperative_metadata(', '__global__ void q4_cooperative_header32(')
    expected = expected.replace('auto sz = aligned_scale_zero(unit, g & 7);', 'auto sz = followup_scale_zero32(unit, g & 7);')
    assert expected in source
    start = old.index('__device__ __forceinline__ ScaleZero aligned_scale_zero')
    finish = old.index('\n}\n',start)+3
    half_math = old[start:finish].split('    __half2_raw codes_raw,header_raw;')[1]
    new_helper = source[source.index('followup_scale_zero32'):source.index('template<int Width',source.index('followup_scale_zero32'))]
    assert half_math in new_helper and 'uint64_t' not in new_helper
    for c in spec.inventory(512,2048):
        if c.family == 'small':
            h,w,warps = c.recipe
            name = 'q4_cooperative_header32' if h else 'q4_cooperative_metadata'
            assert f'{name}<{w},{warps},false,true,512,2048><<<{512//w},{warps*32},0,' in source


def test_small_coverage_and_observed_coalescing_model():
    for width,warps in spec.SMALL_GEOMETRIES:
        groups = []
        for pas in range((2048//128+warps-1)//warps):
            for warp in range(warps):
                chunk = pas*warps+warp
                if chunk >= 2048//128:
                    continue
                groups += [chunk*4+lane//8 for lane in range(0,32,8)]
        assert sorted(groups) == list(range(2048//32))
        c = spec.Candidate('small',(1,width,warps))
        m = spec.access(c,512,2048,dict(A=16,B=16,metadata=16))
        assert m['logical_lane_bytes_per_call'] == dict(B=512*2048//2,A=2*512*2048//width,metadata=512*2048//2)
        for vector in range(width//8):
            b = m['streams'][f'B_v{vector}']
            assert b['width_bytes'] == 16
            assert len(set(b['lane_byte_addresses'])) == 32
            assert len(set(m['streams'][f'units_v{vector}']['lane_byte_addresses'])) == 8


def row(variant,n,k,key,phase,samples,profile=False):
    weight = n*k*9//16
    copies = math.ceil(2.25*67108864/weight)
    r = dict(status='PASS',arm=variant,variant=variant,config_key=key,phase=phase,shape=[1,n,k],mode='rotating',
        recipe=list(runner.recipe(variant,n,k,key)),error=1e-8,zero_code_negative='PASS',zero_a_check='PASS',
        copies=copies,weight_bytes=weight,device=dict(l2_bytes=67108864),calls_per_graph=max(2,math.ceil(32/copies))*copies,
        output_type='F32',inter_cta_split=1,launches_per_call=1,weight_arithmetic=runner.arithmetic(variant,n,k,key),
        storage='RAW_GGUF' if variant == 'raw-reference' else 'XPLANE' if variant == 'xplane' else 'CANONICAL_KPACK4',
        timing_scope='RESIDENT_FULL_CALL_NO_HOST_PACK_NO_JIT',cache_scope='ACU_FORCED_COLD' if profile else 'ROTATING_AT_LEAST_2_25_L2',
        samples_us=[10.]*samples,median_us=10. if samples else None)
    if variant == 'reader':
        c = spec.lookup(n,k,key)
        r.update(family=c.family,geometry=c.geometry(n,k),matched_fp32_sha256='a'*64,
            same_geometry_control=runner.control_kind(c),code_dequant='LOP3_HALF2',access_models=[spec.access(c,n,k)],
            historical_small_sha256='a'*64 if c.family == 'small' and c.recipe[1:] == (8,16) else None)
        r['samples_us'] = [9.+spec.inventory(n,k).index(c)*.001]*samples
        r['median_us'] = r['samples_us'][0] if samples else None
    return r


@pytest.mark.parametrize('variant,key', [('xplane','control'),('raw-reference','control'),('baseline','control'),
    ('reader','v7-c4-w8-p2'),('reader','meta-h1-n8-w16'),('reader','meta-h1-n16-w4')])
def test_receipts_decline_wrong_precision_or_control(variant,key):
    n,k = 512,2048
    for profile,samples in ((False,5),(False,15),(True,0)):
        r = row(variant,n,k,key,'screen',samples,profile)
        assert runner.parse_row(r,variant,n,k,key,'screen',samples,profile) == r
    for field,bad in [('error',float('nan')),('samples_us',[float('nan')]*5),('zero_a_check','SKIP'),('weight_arithmetic','FP16_ACCUMULATE')]:
        r = row(variant,n,k,key,'screen',5); r[field] = bad
        with pytest.raises(ValueError): runner.parse_row(r,variant,n,k,key,'screen',5)
    if variant == 'reader':
        for field,bad in [('matched_fp32_sha256',None),('same_geometry_control','self'),('access_models',[])]:
            r = row(variant,n,k,key,'screen',5); r[field] = bad
            with pytest.raises(ValueError): runner.parse_row(r,variant,n,k,key,'screen',5)


@pytest.mark.parametrize('corrupt',[False,True])
@pytest.mark.parametrize('cfg',[(7,4,8,2),(1,8,16),(1,16,4)])
def test_direct_checks_matched_control_and_small_historical_binary(cfg,corrupt):
    b = runner.FollowupBench.__new__(runner.FollowupBench)
    b.reader_allowed = {c.recipe:c for c in spec.inventory(512,2048)}
    b.output_base=100; b.output=116; b.output_bytes=8; b.a=1; b.weight_pointers=[(10,20)]; b.n=512; b.k=2048
    b.r=SimpleNamespace(stream=3,fill=lambda *args:None); b.error=lambda:0.
    state=[b'']; calls=[]; gold=np.array([1.,2.],dtype='<f4').tobytes()
    def old(*args): calls.append(('control',args)); state[0]=gold; return 0
    def historical(*args): calls.append(('historical',args)); state[0]=gold; return 0
    def new(*args,**kw):
        calls.append(('new',args)); state[0]=(bytes([gold[0]^1])+gold[1:]) if corrupt else gold; return 0
    b.config_launch=old; b.followup_launch={'small':old}; b.current_launch=historical; b.invoke=new
    b.sdk=SimpleNamespace(download=lambda *args:state[0]); b.matched=None
    if corrupt:
        with pytest.raises(ValueError,match='same-geometry'): b.direct(cfg)
    else:
        data,err=b.direct(cfg); assert data==gold and err==0 and len(b.matched)==64
    assert [call[0] for call in calls] == (['control','historical','new'] if cfg==(1,8,16) else ['control','new'])


def test_invoke_uses_each_real_family_config_and_rotating_weight_pointer():
    b=runner.FollowupBench.__new__(runner.FollowupBench)
    b.reader_allowed={c.recipe:c for c in spec.inventory(512,2048)}
    b.copies=2; b.a=1; b.output=2; b.r=SimpleNamespace(stream=3); b.weight_pointers=[(10,20),(30,40)]
    calls=[]; b.followup_launch={family:lambda *args,f=family:calls.append((f,args)) or 0 for family in ('affine','small')}
    for cfg in ((7,4,8,2),(1,8,16)):
        assert b.invoke(cfg,3)==0
    assert calls==[('affine',(7,4,8,2,1,30,40,2,3)),('small',(1,8,16,1,30,40,2,3))]
    with pytest.raises(ValueError): b.invoke((7,2,4,2))


def test_five_percent_gate_keeps_baseline_and_requires_complete_screen():
    n,k=spec.SHAPES[0]; screen={c.key:{} for c in spec.inventory(n,k)}; keys=list(screen)[:3]
    records={key:[dict(median_us=10.5)]*6 for key in (*runner.ANCHORS,*keys)}
    records['raw-reference']=[dict(median_us=10.)]*6
    r=runner.summarize(n,k,screen,records,keys)
    assert r['reference_verdict']=='WITHIN_5_PERCENT' and r['selected']=='baseline'
    records['raw-reference']=[dict(median_us=9.99)]*6
    assert runner.summarize(n,k,screen,records,keys)['reference_verdict']=='PARITY_OPEN'
    screen.pop(keys[0])
    assert runner.summarize(n,k,screen,records,keys)['reference_verdict']=='INCOMPLETE'


@pytest.mark.parametrize('fault',[False,True])
def test_full_screen_confirm_profile_campaign_and_resume_only_missing(tmp_path,monkeypatch,fault):
    for n,k in spec.SHAPES: (tmp_path/f'q12-n{n}-k{k}-e1-c1.npz').write_bytes(b'host-only fixture')
    a=SimpleNamespace(sdk=tmp_path,candidate=PACKAGE,reuse=ROOT/'prebuilt/ppu0010/q4-reader-reuse-v1',
        config_bundle=ROOT/'prebuilt/ppu0010/q4-config-sweep-v1',previous=ROOT/'prebuilt/ppu0010/q4-cold-shapes-v1',
        controls=ROOT/'prebuilt/ppu0010/q4-h800-port-v1',bundle=ROOT/'prebuilt/ppu0010/q4-simt-ab-v1',
        fixtures=tmp_path,output=tmp_path/'results',l2_bytes=67108864,skip_acu=False,acu=tmp_path/'fake-acu')
    monkeypatch.setattr(spec,'verify',lambda *a,**kw:dict(runtime={},source_hashes={}))
    monkeypatch.setattr(runner,'probe_device',lambda a:dict(l2_bytes=67108864))
    monkeypatch.setattr(runner,'acu_launch_command',lambda acu,prefix,cmd:cmd+['--test-report',str(prefix)])
    jobs=[]; planted=[False]
    def fake(cmd,stream):
        v=cmd[cmd.index('--variant')+1]; phase=cmd[cmd.index('--phase')+1]; keys=json.loads(cmd[cmd.index('--keys')+1])
        fn=Path(cmd[cmd.index('--fixture')+1]).name
        n,k=next((n,k) for n,k in spec.SHAPES if fn==f'q12-n{n}-k{k}-e1-c1.npz')
        profile='--profile' in cmd; jobs.append((n,k,v,phase,keys))
        for key in keys:
            if fault and not planted[0] and phase=='screen' and key=='v7-c4-w8-p4':
                planted[0]=True
                stream.write(runner.FAIL_PREFIX+json.dumps(dict(variant=v,key=key,phase=phase,shape=[1,n,k],error='plant'))+'\n')
                return 1
            samples=0 if profile else 5 if phase=='screen' else 15
            stream.write(runner.PREFIX+json.dumps(row(v,n,k,key,phase,samples,profile))+'\n')
        if profile: Path(cmd[cmd.index('--test-report')+1]+'.acurep').write_bytes(b'host profile receipt')
        return 0
    monkeypatch.setattr(runner,'execute',fake)
    assert runner.run(a)==int(fault)
    r=json.loads((a.output/'summary.json').read_text())
    assert len(r['profiles'])==20
    assert sum(len(c['screen'])+len(c['screen_anchors'])+sum(len(v) for v in c['records'].values()) for c in r['cases'])==314-int(fault)
    previous=len(jobs); assert runner.run(a)==0
    assert len(jobs)==previous+int(fault)
    if fault: assert jobs[-1][-1]==['v7-c4-w8-p4'] and jobs[-1][3]=='screen'
    # Results from a different checkout cannot quietly become resume input.
    receipt=a.output/'authority.json'; doc=json.loads(receipt.read_text()); doc['top_k']=999
    receipt.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='resume source'): runner.run(a)


def test_prebuilt_inventory_fastpath_and_noncompiling_handoff():
    m=spec.verify(PACKAGE,ROOT/'prebuilt/ppu0010/q4-reader-reuse-v1',ROOT/'prebuilt/ppu0010/q4-config-sweep-v1',
        ROOT/'prebuilt/ppu0010/q4-cold-shapes-v1',ROOT/'prebuilt/ppu0010/q4-h800-port-v1',ROOT/'prebuilt/ppu0010/q4-simt-ab-v1')
    assert not m['device_validated'] and not m['production_changed'] and len(m['payloads'])==5
    stats=json.loads((PACKAGE/'isa-stats.json').read_text()); assert len(stats)==158
    for name in m['payloads']:
        exports=subprocess.check_output(['nm','-D','--defined-only',str(PACKAGE/name)],text=True)
        assert ' q4_followup_' in exports and ' q4_ppu_probe' in exports
    for n,k in spec.SHAPES:
        for c in spec.inventory(n,k):
            st=stats[f'n{n}-k{k}-{c.key}']
            assert st['code_fastpath_present'] and st['fp32_fma_present']
            h32=c.recipe[0]==1 if c.family=='small' else bool(c.recipe[0]&4)
            if h32: assert st['operations'].get('v.shrl.b64',0)==0
    script=ROOT/'tools/run_q4_reader_followup_ppu_box.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    assert '\n(\n' in script.read_text() and script.read_text().rstrip().endswith(')')
    assert 'compile=NONE JIT=NONE' in script.read_text()
    tree=ast.parse((ROOT/'dev/gemv_ppu/run_reader_followup.py').read_text())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    names=[n.func.id for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    assert 'probe_device' in names and 'SDK' not in names
