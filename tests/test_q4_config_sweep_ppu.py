"""Host mapping, dequant, inventory and retry tests; PPU speed is not inferred."""
import ast
import json
import math
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from dev.gemv_cuda.build import digest
from dev.gemv_ppu.access_pattern import analyze,footprint,isa_histograms,observed_patterns,warp_pattern
from dev.gemv_ppu.config_space import Config,ROOT,SHAPES,disposition,inventory,lookup,payload,plan,source,verify
from dev.gemv_ppu.cold_shapes import source as old_source
from dev.gemv_ppu import run_config_sweep as runner


def test_explicit_inventory_and_preserved_best_baselines():
    assert [len(inventory(n,k)) for n,k in SHAPES]==[52,71,52,68,79,71]
    p=plan()
    assert p['reference_regression_limit_pct']==0
    assert sum(len(c['candidates']) for c in p['cases'])==393
    for case in p['cases']:
        assert len(case['candidates'])+len(case['excluded'])==5*9*3
        assert len({r['key'] for r in case['candidates']+case['excluded']})==5*9*3
    assert disposition(Config(16,8,8),512,2048)=='INVALID_REDUCE_SCATTER_TILE'
    assert disposition(Config(1,32,2),512,2048)=='PRUNED_IDLE_K_WORKERS'
    assert disposition(Config(16,1,2),5120,8192)=='PRUNED_SERIAL_PASS_BUDGET'
    history=runner.baselines()
    assert [history[n,k]['baseline'] for n,k in SHAPES]==['kpack-current']*5+['kpack-c8']
    assert runner.recipe('baseline',512,2048,'control')==(1,16,1)
    assert runner.recipe('baseline',1024,5120,'control')==(2,10,1)
    assert runner.recipe('baseline',8192,5120,'control')==(8,8,1)


@pytest.mark.parametrize('n,k',SHAPES)
def test_kernel_body_unchanged_and_each_config_has_an_exact_entry(n,k):
    previous=old_source().split('extern "C" int q4_cold_shapes_run(')[0]
    stub='extern "C" int qkg_launch_12(qkg_call_v1 const&,qkg_config_v1 const&) {return QKG_INVALID;}'
    prefix=previous.replace(stub,'')
    current=source(n,k)
    assert current.startswith(prefix)
    launches=current[len(prefix):]
    assert launches.count('>>>(')==len(inventory(n,k))
    for c in inventory(n,k):
        assert f'q4_group_affine<{c.columns},{c.warps},{c.values},{n},{k},true><<<{n//(c.columns*c.values)},{c.warps*32},0,' in launches
    assert 'return QKG_INVALID' in launches and 'hggcGetLastError' in launches


@pytest.mark.parametrize('n,k,c',[(n,k,c) for n,k in SHAPES for c in inventory(n,k)])
def test_all_config_K_coverage_and_reduce_scatter_output_ownership(n,k,c):
    g=c.geometry(n,k);groups=k//32;tile=g['tile_n'];warps=c.warps;p=c.values
    visited=np.zeros((groups,tile),dtype=np.int16)
    values=np.zeros((warps*32,p),dtype=np.int64)
    for tid in range(warps*32):
        col=tid%c.columns*p
        for group in range(tid//c.columns,groups,g['k_workers']):
            visited[group,col:col+p]+=1
            values[tid]+=(group+1)*np.arange(col+1,col+p+1)
    assert np.all(visited==1)
    lane=np.arange(32);values=values.reshape(warps,32,p);count,stride=p,c.columns
    while count>1:
        reduced=np.empty((warps,32,count//2),dtype=np.int64);odd=(lane&stride)!=0
        for i in range(count//2):
            keep=np.where(odd[None,:],values[:,:,2*i+1],values[:,:,2*i])
            send=np.where(odd[None,:],values[:,:,2*i],values[:,:,2*i+1])
            reduced[:,:,i]=keep+send[:,lane^stride]
        values=reduced;count//=2;stride*=2
    reduced=values[:,:,0]
    while stride<32:
        reduced=reduced+reduced[:,lane^stride];stride*=2
    partial=np.zeros((warps,tile),dtype=np.int64)
    for l in range(tile):partial[:,(l%c.columns)*p+l//c.columns]=reduced[:,l]
    sums=np.array([partial[t//tile::32//tile,t%tile].sum() for t in range(32)])
    d=tile
    while d<32:sums=sums+sums[lane^d];d*=2
    assert np.array_equal(sums[:tile],np.arange(1,tile+1)*groups*(groups+1)//2)
    assert g['grid']*tile==n and g['threads']<=1024 and g['inter_cta_split']==1


def test_fast_dequant_all_codes_all_slots_both_halves():
    # Independent scalar nibble oracle vs generated lop3 + half2 construction.
    # No FP16 dot: only the exact small integers are constructed in half.
    rng=np.random.default_rng(2903)
    words=rng.integers(0,2**32,65536,dtype=np.uint32)
    for slot in range(4):
        pos=(slot&1)*4;src=words>>8 if slot>=2 else words
        bits=(src & np.uint32(0x000f000f<<pos)) | np.uint32(0x64006400)
        encoded=np.stack((bits&0xffff,bits>>16),axis=1).astype('<u2').view('<f2').astype(np.float32)
        decoded=(encoded/(1<<pos)-float(1024>>pos)).astype(np.float16)
        expected=np.stack(((words>>(4*slot))&15,(words>>(16+4*slot))&15),axis=1).astype(np.float16)
        assert np.array_equal(decoded,expected)
        assert set(decoded.reshape(-1))==set(range(16))


def test_coalescing_patterns_account_for_B_A_metadata_and_alignment():
    c4=analyze(Config(4,8,4),8192,5120);c8=analyze(Config(8,8,4),8192,5120)
    b4=c4['first_warp']['streams']['B_vector']['granules']['64']
    b8=c8['first_warp']['streams']['B_vector']['granules']['64']
    assert (b4['unique_bytes'],b4['sector_bytes'],b4['duplicate_factor'])==(256,512,1)
    assert (b8['unique_bytes'],b8['sector_bytes'],b8['duplicate_factor'])==(256,256,1)
    assert c4['first_warp']['streams']['A_half2']['granules']['64']['duplicate_factor']==4
    assert c8['first_warp']['streams']['A_half2']['granules']['64']['duplicate_factor']==8
    assert c4['first_warp']['streams']['metadata_p0']['granules']['64']['duplicate_factor']==8
    assert c8['first_warp']['streams']['metadata_p0']['granules']['64']['duplicate_factor']==4
    shifted=warp_pattern(Config(8,8,4),8192,5120,base_mod128=dict(A=0,B=16,metadata=0))
    assert shifted['streams']['B_vector']['granules']['64']['sector_bytes']==512
    assert footprint([0,0],4,64)['unique_bytes']==4
    rows=observed_patterns(Config(8,8,4),8192,5120,128,[(256,512),(512,768),(16,32)])
    assert len(rows)==2 and rows[1]['base_mod128']['B']==16
    with pytest.raises(ValueError):observed_patterns(Config(8,8,4),8192,5120,128,[(3,32)])
    for n,k in SHAPES:
        for c in inventory(n,k):
            m=analyze(c,n,k)
            assert m['logical_lane_bytes_per_call']['A']==2*n*k//c.values
            assert m['last_active_warp']['active_lanes'] in range(1,33)


def record(variant,n,k,key,phase,samples,*,profile=False,us=10.):
    impl='affine-fast' if variant=='config' else runner.baselines()[n,k]['implementation'] if variant=='baseline' else variant
    values=[us]*samples;weight_bytes=n*k*9//16;l2=67108864;copies=math.ceil(2.25*l2/weight_bytes)
    r=dict(status='PASS',arm=variant,variant=variant,config_key=key,phase=phase,
        shape=[1,n,k],mode='rotating',recipe=list(runner.recipe(variant,n,k,key)),error=1e-6,
        zero_code_negative='PASS',zero_a_check='PASS',output_type='F32',implementation=impl,
        weight_arithmetic='FP32_GROUP_AFFINE' if impl.startswith('affine') else 'PER_WEIGHT_FP16',
        storage='XPLANE' if variant=='xplane' else 'RAW_GGUF' if variant=='raw-reference' else 'CANONICAL_KPACK4',
        inter_cta_split=1,launches_per_call=1,timing_scope='RESIDENT_FULL_CALL_NO_HOST_PACK_NO_JIT',
        cache_scope='ACU_FORCED_COLD' if profile else 'ROTATING_AT_LEAST_2_25_L2',
        copies=copies,weight_bytes=weight_bytes,device=dict(l2_bytes=l2),
        calls_per_graph=max(2,math.ceil(32/copies))*copies,samples_us=values,median_us=statistics.median(values) if values else None)
    if variant=='config':
        c=lookup(n,k,key);r.update(geometry=c.geometry(n,k),dequant='LOP3_HALF2_CODES_FP32_GROUP_AFFINE',
                                  observed_warp_patterns=observed_patterns(c,n,k,128,[(256,512)]))
    if variant=='baseline':r['baseline_arm']=runner.baselines()[n,k]['baseline']
    return r


@pytest.mark.parametrize('variant',[*runner.ANCHORS,'config'])
@pytest.mark.parametrize('n,k',SHAPES)
def test_receipts_and_profiler_are_not_interchanged(variant,n,k):
    key=inventory(n,k)[0].key if variant=='config' else 'control'
    for profile,samples in ((False,5),(True,0)):
        r=record(variant,n,k,key,'screen',samples,profile=profile)
        assert runner.parse_row(r,variant,n,k,key,'screen',samples,profile=profile)==r


@pytest.mark.parametrize('field,value',[('recipe',[4,8,1]),('geometry',{}),('phase','r0'),
    ('error',float('nan')),('samples_us',[float('nan')]*5),('median_us',float('nan')),
    ('zero_code_negative','SKIP'),('zero_a_check','SKIP'),('inter_cta_split',2),
    ('observed_warp_patterns',[]),('dequant','scalar'),('cache_scope','ACU_FORCED_COLD')])
def test_wrong_or_missing_result_is_not_measured(field,value):
    n,k=SHAPES[0];key=inventory(n,k)[0].key
    r=record('config',n,k,key,'screen',5);r[field]=value
    with pytest.raises(ValueError):runner.parse_row(r,'config',n,k,key,'screen',5)


def test_observed_model_is_rederived_not_trusted():
    n,k=SHAPES[0];key=inventory(n,k)[0].key;r=record('config',n,k,key,'screen',5)
    r['observed_warp_patterns'][0]['streams']['B_vector']['granules']['64']['sector_bytes']+=64
    with pytest.raises(ValueError):runner.parse_row(r,'config',n,k,key,'screen',5)


def test_no_five_percent_relaxation_against_reference_and_keep_prior_winner():
    n,k=SHAPES[0];keys=[c.key for c in inventory(n,k)];chosen=keys[:3]
    screen={key:{'median_us':10.} for key in keys}
    rows={key:[{'median_us':10.} for _ in range(6)] for key in (*runner.ANCHORS,*chosen)}
    rows['baseline']=[{'median_us':9.} for _ in range(6)]
    assert runner.summarize(n,k,screen,rows,chosen)['selected']=='baseline'
    rows['baseline']=[{'median_us':11.} for _ in range(6)]
    for key in chosen:rows[key]=[{'median_us':10.01} for _ in range(6)]
    out=runner.summarize(n,k,screen,rows,chosen)
    assert out['status']=='PASS' and out['reference_verdict']=='REF_REGRESSION'
    assert out['xplane_within_5pct']
    rows[chosen[0]]=[{'median_us':10.} for _ in range(6)]
    assert runner.summarize(n,k,screen,rows,chosen)['reference_verdict']=='NOT_SLOWER_THAN_REFERENCE'
    screen.pop(keys[-1])
    assert runner.summarize(n,k,screen,rows,chosen)['reference_verdict']=='INCOMPLETE'


def test_real_config_and_rotated_pointers_reach_the_native_entry():
    b=runner.SweepBench.__new__(runner.SweepBench);b.allowed={(8,8,4)}
    b.copies=2;b.a=1;b.output=2;b.r=SimpleNamespace(stream=3);b.weight_pointers=[(10,20),(30,40)]
    calls=[];b.config_launch=lambda *args:calls.append(args) or 0
    assert b.invoke((8,8,4),3)==0 and calls==[(8,8,4,1,30,40,2,3)]
    with pytest.raises(ValueError):b.invoke((8,8,1))  # P is not the old Split-K field.


@pytest.mark.parametrize('fault',[None,'typed_failure','bad_numeric'])
def test_all_393_screens_confirmations_and_resume_only_failed_cell(tmp_path,monkeypatch,fault):
    for n,k in SHAPES:(tmp_path/f'q12-n{n}-k{k}-e1-c1.npz').write_bytes(b'orchestration fixture')
    args=SimpleNamespace(sdk=tmp_path,candidate=ROOT/'prebuilt/ppu0010/q4-config-sweep-v1',
        previous=ROOT/'prebuilt/ppu0010/q4-cold-shapes-v1',controls=ROOT/'prebuilt/ppu0010/q4-h800-port-v1',
        bundle=ROOT/'prebuilt/ppu0010/q4-simt-ab-v1',fixtures=tmp_path,output=tmp_path/'results',
        top_k=3,l2_bytes=67108864,skip_acu=True)
    monkeypatch.setattr(runner,'verify',lambda *args,**kw:dict(runtime={}))
    monkeypatch.setattr(runner,'probe_device',lambda args:dict(l2_bytes=67108864))
    jobs=[];emitted=[];planted=[False]
    def fake_child(cmd,stream):
        v=cmd[cmd.index('--variant')+1];phase=cmd[cmd.index('--phase')+1]
        keys=json.loads(cmd[cmd.index('--keys')+1]);samples=int(cmd[cmd.index('--samples')+1])
        fixture=Path(cmd[cmd.index('--fixture')+1]);n,k=next((n,k) for n,k in SHAPES if fixture.name==f'q12-n{n}-k{k}-e1-c1.npz')
        jobs.append((v,n,k,phase,keys))
        for key in keys:
            fail=fault and not planted[0] and v=='config' and phase=='screen' and key==inventory(n,k)[1].key
            r=record(v,n,k,key,phase,samples)
            if fail:
                planted[0]=True
                if fault=='typed_failure':
                    stream.write(runner.FAIL_PREFIX+json.dumps(dict(variant=v,key=key,shape=[1,n,k],phase=phase,error='planted'))+'\n')
                    return 1
                r['error']=float('nan')
            stream.write(runner.PREFIX+json.dumps(r)+'\n');emitted.append((v,n,k,phase,key))
        return 0
    monkeypatch.setattr(runner,'execute',fake_child)
    assert runner.run(args)==(0 if fault is None else 1)
    summary=json.loads((args.output/'summary.json').read_text())
    assert sum(len(c['screen']) for c in summary['cases'])==393-int(fault is not None)
    assert sum(len(rows) for c in summary['cases'] for rows in c['records'].values())==216
    old_jobs=len(jobs);assert runner.run(args)==0
    if fault is None:assert len(jobs)==old_jobs
    else:
        # Top-three is stable (the failed key was not a tie-selected winner).
        assert len(jobs)==old_jobs+1
        assert len(jobs[-1][-1])==1 and jobs[-1][3]=='screen'
    saved=json.loads((args.output/'config-winners.json').read_text())
    assert not saved['production_changed'] and len(saved['winners'])==6


def test_native_package_isa_and_exports_and_noncompiling_box_script():
    candidate=ROOT/'prebuilt/ppu0010/q4-config-sweep-v1'
    m=verify(candidate,ROOT/'prebuilt/ppu0010/q4-cold-shapes-v1',ROOT/'prebuilt/ppu0010/q4-h800-port-v1',ROOT/'prebuilt/ppu0010/q4-simt-ab-v1')
    assert not m['production_changed'] and not m['device_validated']
    stats=json.loads((candidate/'isa-stats.json').read_text())
    assert len(stats)==393 and all(s['code_fastpath_present'] and s['fp32_fma_present'] for s in stats.values())
    for n,k in SHAPES:
        symbols=subprocess.check_output(['nm','-D','--defined-only',str(candidate/payload(n,k))],text=True)
        assert f' q4_config_run_{n}_{k}' in symbols and ' q4_ppu_probe' in symbols
    script=ROOT/'tools/run_q4_config_sweep_ppu_box.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    assert '\n(\n' in script.read_text() and script.read_text().rstrip().endswith(')')
    parsed=ast.parse((ROOT/'dev/gemv_ppu/run_config_sweep.py').read_text())
    run=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    calls=[x.func.id for x in ast.walk(run) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name)]
    assert 'probe_device' in calls and 'SDK' not in calls and 'load_library' not in calls


def test_isa_statistics_do_not_confuse_fp16_codes_with_fp16_dot():
    text='Disassembly of section .text.kernel.xyzq4_group_affineILi8ELi8ELi4ELi8192ELi5120ELb1EE:\n'
    text+='\tv.lop3.b32 x\n\tv.fma.f16x2 y\n\tv.fma.f32.rtte z\n\tvmem.ld.b32x2 x\n'
    rows=isa_histograms(text);r=rows['n8192-k5120-c8-w8-p4']
    assert r['code_fastpath_present'] and r['fp32_fma_present'] and r['loads']=={'vmem.ld.b32x2':1}
    r=isa_histograms(text.replace('\tv.fma.f32.rtte z\n',''))['n8192-k5120-c8-w8-p4']
    assert r['code_fastpath_present'] and not r['fp32_fma_present']
