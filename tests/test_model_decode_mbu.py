import copy
import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dev.gemv_simt import model_followup as spec
from dev.gemv_simt.run_model_followup import summarize,validate_profile,verify_candidate
from tools.kpack_model_profile_bench import check_choice,select_choice
from tools.profile_kpack_model_decode import profile_plan
from tools.run_kpack_model_mbu import validate_plan,option


def rows():
    out=[]
    for p in spec.POINTS:
        out.append(dict(point=dict(q=p.q,n=p.n,k=p.k,mode=p.mode,channels=p.channels,
            compute=p.compute,tokens=1,experts=256 if p.mode else 1,topk=8 if p.mode else 1,
            kind='simt',**asdict(p.config))))
    for q,n,k,s in ((8,8192,2048,8),(8,4096,2048,8),(14,248320,2048,1)):
        out.append(dict(point=dict(q=q,n=n,k=k,split=s,compute=0,kind='tc')))
    return out


def test_complete_inventory_and_wrong_previous_run_rejected():
    assert sum(len(p.arms) for p in spec.POINTS)==14
    validate_plan(rows())
    for fault in ('missing','duplicate','compute','geometry','tc'):
        bad=copy.deepcopy(rows())
        if fault=='missing':bad.pop()
        elif fault=='duplicate':bad.append(bad[0])
        elif fault=='compute':bad[0]['point']['compute']=1
        elif fault=='geometry':bad[0]['point']['warps']=4
        else:bad[-1]['point']['split']=8
        with pytest.raises(ValueError):validate_plan(bad)


def test_tc_profile_uses_exact_asys_module_and_no_fixed_top_three():
    selection=dict(plans=[]);matched=[];names=[]
    for i in range(5):
        symbol=f'void cutlass::device_kernel<Test{i}SplitKParallel>(Args)'
        name=f'weight{i}';build=str(i)*64
        selection['plans'].append(dict(op='dense',route='sf',rows='1',activation='FP16',build=build,
            q='8',n=str(1024*2**i),k='2048',split='8',algorithm='0',grid='0',policy='12',parent=name,tensor=name))
        names.append(dict(name=symbol));matched.append(dict(name=symbol,libraries=['jit/'+build]))
    args=(selection,dict(kernels=names),'',(None,None,None),matched)
    result=profile_plan(*args)
    assert len(result)==5 and all(r['point']['kind']=='tc' for r in result)
    for fault in ('unobserved','ambiguous','wrong-build','untyped'):
        s,k,log,helpers,m=copy.deepcopy(args)
        if fault=='unobserved':k['kernels'].pop()
        elif fault=='ambiguous':m.append(dict(name=names[1]['name'],libraries=m[0]['libraries']))
        elif fault=='wrong-build':s['plans'][0]['build']='f'*64
        else:s['plans'][0]['activation']='UNKNOWN'
        with pytest.raises(ValueError):profile_plan(s,k,log,helpers,m)


def test_tc_matched_policy_must_use_same_entry_as_caller():
    point=dict(q=8,n=8192,k=2048,compute=0,endpoint=1,policy=12,parent='observed',build='f'*64,
               split=8,algorithm=0,grid=0,route=1)
    choice=SimpleNamespace(**{k:v for k,v in point.items() if k not in ('parent','build')},
                           parent=b'observed',build_key=b'f'*64)
    calls=[]
    class Dispatcher:
        def query_compute(self,*args):raise AssertionError('wrong selector')
        def query_smallm_matched(self,call,arr,compute):
            calls.append(call)
            assert (call.qtype,call.mode,call.rows,call.input_type,call.a_row_stride)==(8,0,1,1,2048)
            return SimpleNamespace(base=SimpleNamespace(kind=0,tc=choice))
    assert select_choice(Dispatcher(),point,SimpleNamespace(mapping_id=42)) is choice
    assert len(calls)==1
    for key in ('parent','build','split','algorithm','grid','policy'):
        wrong=point|{key: 'wrong' if key in ('parent','build') else 0}
        if wrong[key]==point[key]:wrong[key]=5
        with pytest.raises(ValueError):check_choice(choice,wrong)


def test_frozen_body_clones_and_isolated_changes():
    for p in spec.POINTS:
        source=spec.source(p)
        assert '#include "quactlize/execution/simt_validation.hpp"' in source
        assert 'q4_affine_header32' in source
        assert source.count('model_followup::candidate<')==len(p.arms)
        assert f'<<<blocks,{p.config.warps*32},0,stream>>>' in source
        assert 'split!=1' in source and 'input_type!=QKG_F32' in source
        body=source if p.config.variant>=4 else (spec.ROOT/'quactlize/execution/simt_kernel.cuh').read_text()
        assert 'std::conditional_t<(Changes&2)!=0,unsigned,int>' in body
        if p.q==8:
            assert 'bytes/16' in source and 'bytes/256' in source
            assert 'sum^=v.x^v.y^v.z^v.w' in source
            assert 'static_cast<uint32_t*>(c.workspace)' in source
        else:assert 'q4_s1::q4_medium_fold<0,Warps,TileN>' in body
    with pytest.raises(ValueError):spec.once('changed','expected','new')


def test_h32_fields_and_static_fold_keep_integer_mapping_and_add_order():
    u=np.random.default_rng(103).integers(0,2**32,(4096,4),dtype='u4')
    wide=u.astype('u8')
    for g in range(8):
        shift=6*(g&3)
        run=(wide[:,2]>>16)|(wide[:,3]<<16) if g&4 else wide[:,1]|((wide[:,2]&65535)<<32)
        sc=(u[:,2]>>16)|(u[:,3]<<16) if g&4 else u[:,1]
        mn=u[:,3]>>8 if g&4 else (u[:,1]>>24)|(u[:,2]<<8)
        assert np.array_equal((sc>>shift)&63,(run>>shift)&63)
        assert np.array_equal((mn>>shift)&63,(run>>(shift+24))&63)
    for p in spec.POINTS:
        c=p.config;stripes=32//c.tile_n
        for lane in range(32):
            original=list(range(lane//c.tile_n,c.warps,stripes))
            candidate=[lane//c.tile_n+r*stripes for r in range(c.warps)
                       if lane//c.tile_n+r*stripes<c.warps]
            assert original==candidate
            assert lane%c.tile_n==lane&(c.tile_n-1)


def test_confirmation_mbu_and_failed_samples_never_admitted():
    samples={'shipping':[[10.]*15 for _ in range(6)],'1':[[8.]*15 for _ in range(6)]}
    result=summarize(samples,21600000,60)
    assert result['1']['effective_weight_MBU_pct']==100
    assert result['1']['delta_pct']==pytest.approx(-20)
    for fault in ('round','count','nan','zero'):
        bad=copy.deepcopy(samples)
        if fault=='round':bad['1'].pop()
        elif fault=='count':bad['1'][0].pop()
        else:bad['1'][0][0]=float('nan') if fault=='nan' else 0
        with pytest.raises(ValueError):summarize(bad,21600000,60)


def profile_csv(p,arm):
    c=p.config
    if c.variant>=4:
        args=[1,p.compute,c.variant-4,c.columns,c.warps,c.values]
        name='quactlize::execution::simt::q8_vector::kernel'
    else:
        args=[p.q,1,c.variant,c.columns,c.warps,c.values,p.compute]
        name='quactlize::execution::simt::register_reuse'
    if arm!='shipping':args.insert(0,int(arm));name='quactlize::execution::model_followup::candidate'
    row={'ID':'0','Kernel Name':'void '+name+'<'+', '.join(map(str,args))+'>(qkg_call_v1, int)',
         'Kernel Mangled Name':'mangled','Block Size':f'({c.warps*32},1,1)',
         'Grid Size':f'({(8 if p.mode==2 else 1)*p.n//c.tile_n},1,1)'}
    out=io.StringIO();w=csv.DictWriter(out,fieldnames=list(row),quoting=csv.QUOTE_ALL)
    w.writeheader();w.writerow(row)
    return out.getvalue()


def test_acu_requires_exact_s1_kernel_and_geometry():
    for p in spec.POINTS:
        for arm in ('shipping',*map(str,p.arms)):
            raw=profile_csv(p,arm)
            assert validate_profile(raw,p,arm)['mangled']=='mangled'
            for bad in (raw.replace('(qkg_call_v1, int)','(Other)'),
                        raw.replace(f'({p.config.warps*32},1,1)','(32,1,1)'),
                        raw+raw.splitlines()[-1]+'\n'):
                with pytest.raises(ValueError):validate_profile(bad,p,arm)


def test_candidate_pin_cannot_silently_use_another_shipping_image(tmp_path):
    from quactlize.runtime.compiler import sha
    candidate=tmp_path/'candidate';candidate.mkdir();shipping=tmp_path/'shipping';shipping.mkdir()
    image=shipping/'libquactlize_ppu_execution.so';image.write_bytes(b'original')
    records=[]
    for p in spec.POINTS:
        lib=candidate/(p.name+'.so');lib.write_bytes(b'isolated')
        records.append(dict(point=asdict(p),arms=list(p.arms),library=lib.name,sha256=sha(lib)))
    data=dict(schema=spec.SCHEMA,platform='ppu',records=records,source_hashes={},shipping_execution_sha256=sha(image))
    (candidate/'manifest.json').write_text(json.dumps(data))
    verify_candidate(candidate,shipping)
    image.write_bytes(b'other')
    with pytest.raises(ValueError):verify_candidate(candidate,shipping)


def test_previous_command_option_must_be_unique(tmp_path):
    assert option(['--bundle',str(tmp_path)],'--bundle')==tmp_path
    for argv in ([],['--bundle'],['--bundle',str(tmp_path),'--bundle',str(tmp_path)]):
        with pytest.raises(ValueError):option(argv,'--bundle')
