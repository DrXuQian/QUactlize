"""Small-M row ownership, bounded TC controls and fail-closed result receipts."""
import copy
import ctypes as C
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dev.gemv_ppu import smallm as spec, smallm_bench as bench, run_smallm as runner


def manifest():
    parent = dict(symbol='fq_test',qtype=12,route='fq-dense',tm=8,tn=64,tk=256,wm=8,wn=16,
                  stages=2,ap=0,dn=64,persistent=-1)
    return dict(modules=[dict(parent=parent,key='b'*64,path='kernel.so')],
                tc_selection=[dict(request=list(r),status='SELECTED',parent='fq_test',split=4,
                                   policy=2,grid_mode=0,grid_b=0) for r in spec.requests()])


DEVICE = dict(name='PPU-ZW810',ordinal=0,l2_bytes=67108864)


def row(arm,n,k,m,key,phase='screen',profile=False):
    data = manifest()
    c = spec.lookup(n,k,key) if arm=='kpack' else None
    if arm=='tc':
        split = int(key.rsplit(':s',1)[1])
        p = data['modules'][0]['parent']
        reason = bench.tc_reason(p,k,split)
        if reason:
            return dict(arm=arm,shape=[m,n,k],key=key,phase=phase,status='STRUCTURAL',reason=reason,samples_us=[],median_us=None)
    copies = max(2,int(np.ceil(2.25*DEVICE['l2_bytes']/(n*k*9//16))))
    values = [] if profile else [2.]*(5 if phase=='screen' else 15)
    out = dict(status='PASS',arm=arm,shape=[m,n,k],key=key,phase=phase,error=1e-5,
        zero_code_negative='PASS',zero_a_check='PASS',output_guard='PASS',row_alias_negative='PASS',replay_check='PASS',
        row_alias_negative_scope='HOST_ORACLE_PLANT',
        output_type='F16' if arm=='tc' else 'F32',accumulator='F32',input_sha256='a'*64,
        timing_scope='RESIDENT_FULL_CALL_INCLUDING_TC_REDUCER_NO_HOST_SETUP_OR_JIT',
        cache_scope='ACU_FORCED_COLD' if profile else 'ROTATING_AT_LEAST_2_25_L2',
        storage='RAW_GGUF' if arm=='reference' else 'CANONICAL_KPACK4',
        device=DEVICE,copies=copies,weight_bytes=n*k*9//16,calls_per_graph=max(2,int(np.ceil(32/copies)))*copies,
        samples_us=values,median_us=None if profile else 2.,selection=None)
    if c:
        out.update(geometry=c.geometry(n,k,m),weight_arithmetic=c.arithmetic,
            matched_fp32_sha256='c'*64,immutable_m1_sha256='c'*64 if key==spec.selected(n,k).key else None,
            multirow_scope='ONE_LAUNCH_ROW_GRID',access_models=[spec.access(c,n,k,m,dict(A=0,B=0,metadata=0))])
    elif arm=='tc':
        out.update(weight_arithmetic='FQ_TC_F16_RECONSTRUCTION',multirow_scope='TC_COMPLETE_CALL',selection=dict(
            parent=p,build_key='b'*64,split=split,algorithm='ORDINARY',grid=0,
            current_policy_key=runner.policy_key(data,m,n,k),is_current_policy=key==runner.policy_key(data,m,n,k)))
    else:
        out.update(weight_arithmetic='PER_WEIGHT_FP16',multirow_scope='ONE_LAUNCH_ROW_GRID')
    return out


def test_scope_and_previous_winner_recall():
    assert spec.MS==tuple(range(2,9)) and len(spec.requests())==42
    assert spec.plan()['denominator']==42
    assert sum(len(spec.inventory(n,k)) for n,k in spec.SHAPES)==52
    for n,k in spec.SHAPES:
        assert spec.selected(n,k) in spec.inventory(n,k)
        assert len({c.key for c in spec.inventory(n,k)})==len(spec.inventory(n,k))
        for c in spec.inventory(n,k):
            for m in spec.MS:
                g=c.geometry(n,k,m)
                assert g['grid_y']==m and g['grid_x']*g['tile_n']==n
                assert g['grid']==g['grid_x']*m and g['threads']<=1024
                assert g['rows_per_cta']==1 and g['inter_cta_split']==1


@pytest.mark.parametrize('n,k',spec.SHAPES)
def test_lift_changes_only_row_bases_and_name(n,k):
    prefix,original,lifted,name=spec.kernel_parts(n,k)
    c=spec.selected(n,k)
    output='output' if c.family=='meta' else 'out_ptr'
    bases=('\n    a_ptr=static_cast<__half const*>(a_ptr)+size_t(blockIdx.y)*K;\n'
           f'    {output}+=size_t(blockIdx.y)*N;\n')
    restored=lifted.replace(bases,'').replace('q4_smallm_'+c.family+'(',name+'(')
    assert restored==original and original in prefix
    text=spec.source(n,k)
    assert 'rows<1 || rows>8' in text and 'if(control)' in text
    assert f'dim3({c.geometry(n,k,1)["grid_x"]},rows)' in text
    assert 'for(int row=0;row<rows;++row)' in text  # explicitly untimed blue path
    for m in spec.MS:
        a=np.arange(m*k).reshape(m,k)
        out=np.arange(m*n).reshape(m,n)
        for rr in range(m):
            np.testing.assert_array_equal(a.reshape(-1)[rr*k:(rr+1)*k],a[rr])
            np.testing.assert_array_equal(out.reshape(-1)[rr*n:(rr+1)*n],out[rr])


def test_raw_reference_retains_multirow_launch_and_fp32_dot():
    text=spec.reference_source()
    assert 'float2 fp32_dot' in text and 'rows,n,k,static_cast<hggcStream_t>' in text
    assert 'rows<1 || rows>8' in text
    assert 'q4k_gemv_fp32::launch_q4k_gemv<' in text


def test_multirow_lane_request_totals_do_not_multiply_unique_weights():
    for n,k in spec.SHAPES:
        c=spec.selected(n,k)
        for m in (2,8):
            value=spec.access(c,n,k,m)
            name='logical_lane_bytes_per_call' if c.family=='reuse' else 'logical_global_lane_bytes'
            assert value[name]=={key:m*b for key,b in value['logical_lane_bytes_one_row'].items()}
            assert value['unique_weight_bytes']==n*k*9//16 and value['weight_row_offset_bytes']==0


def test_tc_structural_matches_existing_module_alignment_and_pipeline():
    p=manifest()['modules'][0]['parent']
    assert bench.tc_reason(p,5120,8)=='K_SPLIT_ALIGNMENT'
    assert bench.tc_reason(p,2048,8) is None
    assert bench.tc_reason(p|dict(stages=3),2048,8)=='PIPELINE_FILL'
    text=(spec.ROOT/'quactlize/runtime/module.cuh').read_text()
    assert 'c.k % (tk * r.split)' in text
    assert 'c.k / (tk * r.split) < stages - 1' in text


@pytest.mark.parametrize('arm',runner.ARMS)
@pytest.mark.parametrize('m',[2,3,7,8])
def test_result_parser_binds_requested_m_output_and_complete_scope(arm,m):
    n,k=1024,5120
    key=runner.keys_for(arm,n,k,manifest())[0]
    value=row(arm,n,k,m,key)
    assert runner.parse_row(value,arm,n,k,m,key,'screen',5,manifest())==value
    for change in (dict(shape=[1,n,k]),dict(output_type='BF16'),dict(accumulator='F16'),
                   dict(zero_code_negative='NONE'),dict(replay_check='FAIL'),dict(calls_per_graph=1),
                   dict(timing_scope='PRODUCER_ONLY'),dict(samples_us=[1.,2.]),dict(error=float('nan'))):
        with pytest.raises(ValueError):
            runner.parse_row(value|change,arm,n,k,m,key,'screen',5,manifest())


def test_tc_unsupported_is_not_a_numeric_failure_or_fake_timing():
    key='fq_test:s8'
    r=row('tc',1024,5120,8,key)
    assert r['status']=='STRUCTURAL'
    runner.parse_row(r,'tc',1024,5120,8,key,'screen',5,manifest())
    for changed in (r|dict(reason='OUT_OF_MEMORY'),r|dict(samples_us=[1.],median_us=1.),r|dict(status='PASS')):
        with pytest.raises(ValueError):runner.parse_row(changed,'tc',1024,5120,8,key,'screen',5,manifest())


def test_tc_current_policy_is_confirmed_even_when_not_screen_best():
    n,k,m=512,2048,2
    values={}
    for s,t in ((1,1.),(2,1.1),(4,4.),(8,4.1)):
        r=row('tc',n,k,m,f'fq_test:s{s}');r['median_us']=t
        values[runner.item_id(m,r['key'])]=r
    assert runner.shortlist(values,m,'tc',n,k,manifest())==['fq_test:s1','fq_test:s2','fq_test:s4']


def test_one_prepared_tc_call_not_a_producer_only_or_per_row_loop():
    b=bench.Bench.__new__(bench.Bench)
    calls=[]
    b.args=SimpleNamespace(arm='tc');b.copies=2;b.weight_pointers=[(11,22),(33,44)]
    b.handles=[1,2];b.r=SimpleNamespace(stream=7)
    b.module=SimpleNamespace(run=lambda h,s:calls.append((h,s)) or 0)
    assert b.invoke(3)==0 and calls==[(2,7)]
    with pytest.raises(ValueError):b.invoke(control=True)


def test_independent_fixture_rejects_row_zero_alias(monkeypatch):
    rng=np.random.default_rng(4493);n,k=512,2048
    raw=rng.integers(0,256,(n,k//256,144),dtype='u1')
    raw[:,:,:4]=np.array([.002,.001],dtype='<f2').view('u1')
    monkeypatch.setattr(bench,'read_fixture',lambda _: (n,k,dict(raw=raw,a=(rng.standard_normal(k)*.2).astype('<f2'))))
    nn,kk,data=bench.fixture(Path('unused'))
    assert (nn,kk)==(n,k) and data['golden'].shape==(8,n)
    for m in spec.MS:
        assert bench.conditioned(np.broadcast_to(data['golden'][0],(m,n)),data['golden'][:m],data['denom'][:m])>.005
    with pytest.raises(ValueError):bench.conditioned(np.zeros((1,n)),data['golden'],data['denom'])


@pytest.mark.parametrize('fault',[False,True])
def test_whole_campaign_retains_success_and_retries_only_missing_cells(tmp_path,monkeypatch,fault):
    data=manifest();data.update(source_hashes={},runtime={})
    package=tmp_path/'package';package.mkdir();(package/'manifest.json').write_text(json.dumps(data))
    fixtures=tmp_path/'fixtures';fixtures.mkdir()
    for n,k in spec.SHAPES:(fixtures/f'q12-n{n}-k{k}-e1-c1.npz').write_bytes(b'fixture')
    args=SimpleNamespace(candidate=package,bundle=package,sdk=tmp_path,fixtures=fixtures,output=tmp_path/'results',
                         l2_bytes=67108864,skip_acu=False,acu=tmp_path/'acu')
    monkeypatch.setattr(spec,'verify',lambda *a,**kw:data)
    monkeypatch.setattr(spec.medium_refine,'verify',lambda *a,**kw:None)
    monkeypatch.setattr(runner,'probe_device',lambda _:DEVICE)
    executions=[];planted=[False]
    def execute(cmd,stream):
        def argument(k):return cmd[cmd.index(k)+1]
        n,k=int(argument('--n')),int(argument('--k'));arm=argument('--arm');phase=argument('--phase')
        items=json.loads(argument('--items'));profile='--profile' in cmd
        executions.append((arm,n,k,phase,items))
        for m,key in items:
            if fault and not planted[0] and arm=='kpack' and phase=='screen':
                planted[0]=True
                stream.write(runner.FAIL_PREFIX+json.dumps(dict(arm=arm,shape=[m,n,k],key=key,phase=phase,error='planted'))+'\n')
                return 1
            value=row(arm,n,k,m,key,phase,profile)
            # The recovered missing candidate really changes the shortlist;
            # stale valid confirmation entries must not block its admission.
            if arm=='kpack' and m==2 and key==spec.inventory(n,k)[0].key and not profile:
                value['samples_us']=[1.]*len(value['samples_us']);value['median_us']=1.
            stream.write(runner.PREFIX+json.dumps(value)+'\n')
        if profile:Path(argument('--export')+'.acurep').write_bytes(b'profile')
        return 0
    monkeypatch.setattr(runner,'execute',execute)
    rc=runner.run(args)
    assert rc==int(fault)
    summary=json.loads((args.output/'summary.json').read_text())
    assert len(summary['cases'])==42
    before=len(executions)
    assert runner.run(args)==0
    new=executions[before:]
    if fault:
        assert new and all(len(x[-1])==1 for x in new if x[3]=='screen')
    else:assert not new
    summary=json.loads((args.output/'summary.json').read_text())
    assert summary['status']=='PASS' and len(summary['cases'])==42 and len(summary['profiles'])>=36
    for c in summary['cases']:
        assert c['comparison']['tc_current_us']==2.
        m,n,k=c['shape']
        if m==2:
            assert c['comparison']['best']['kpack']==spec.inventory(n,k)[0].key
            assert c['comparison']['kpack_us']==1.
