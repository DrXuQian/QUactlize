"""Replay the measured policy, actual C ABI, selected closure and old-reader body."""
import copy
import ctypes as C
import gzip
import json
from pathlib import Path
import re
import subprocess

import pytest

from tools import fit_q4_decode_policy as fit
from quactlize.execution import q4_decode_codegen as codegen
from quactlize.execution.native import Call, Arrangement, Sizes, arrangement
from quactlize.dispatch.native import Request
from quactlize.runtime.compiler import sha

ROOT=fit.ROOT
EVIDENCE=ROOT/'docs/measurements/q4_decode_policy_20260913.json.gz'


class Config(C.Structure):
    _fields_=[('version',C.c_uint32),('size',C.c_uint32)]+[
        (name,C.c_int32) for name in ('reader','variant','warps','values','columns')]


@pytest.fixture(scope='module')
def evidence():
    return json.loads(gzip.decompress(EVIDENCE.read_bytes()))


@pytest.fixture(scope='module')
def policy():
    return json.loads(codegen.POLICY.read_text())


def test_regenerate_exact_policy(evidence,policy):
    assert fit.fit(evidence)|{'evidence_sha256':sha(EVIDENCE)}==policy
    assert fit.header(policy,evidence)==codegen.POLICY.with_suffix('.hpp').read_text()
    assert len(policy['replay'])==372
    assert max(r['previous_delta_pct'] for r in policy['replay'])<=5
    dense=[r for r in policy['replay'] if r['case'].startswith('dense')]
    assert len(dense)==96 and max(r['regret_pct'] for r in dense)<5
    # Do not quietly turn a public-feature compromise into universal 5% parity.
    open_rows=[r for r in policy['replay'] if r['regret_pct']>5]
    assert open_rows and all('grouped-' in r['case'] and '-t8-' in r['case'] for r in open_rows)


@pytest.mark.parametrize('fault',['duplicate','missing','median','best','no-control','no-repeat','nan'])
def test_bad_measurement_cannot_fit(evidence,fault):
    data=copy.deepcopy(evidence);c=data['cases'][0]
    if fault=='duplicate':data['cases'][1]=copy.deepcopy(c)
    if fault=='missing':data['cases'].pop()
    if fault=='median':c['confirmed'][c['current']]*=1.1
    if fault=='best':c['best']['simt']['median_us']*=.9
    if fault=='no-control':c['current']='tc:missing'
    if fault=='no-repeat':c['round_medians'][c['current']].pop()
    if fault=='nan':c['round_medians'][c['current']][0]=float('nan')
    with pytest.raises(ValueError):fit.fit(data)


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    out=tmp_path_factory.mktemp('q4-decode-host')/'host.so'
    subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC',f'-I{ROOT}',
        f'-I{ROOT}/quactlize/include',f'-I{ROOT}/third_party/actlize/include',
        str(ROOT/'quactlize/execution/q4_decode.cpp'),str(ROOT/'tests/q4_decode_policy_host.cpp'),
        '-o',str(out)],check=True)
    lib=C.CDLL(str(out))
    lib.quactlize_kpack_q4_decode_select_v1.argtypes=[C.POINTER(Call),C.POINTER(Arrangement),C.POINTER(Config),C.POINTER(Sizes)]
    lib.quactlize_kpack_q4_decode_run_v1.argtypes=[C.POINTER(Call),C.POINTER(Config),C.POINTER(Arrangement)]
    lib.q4_decode_tc.argtypes=[C.POINTER(Request),C.c_char_p,C.c_int]
    return lib


def make_call(w):
    indexed=w['operator']=='grouped'
    return Call(version=1,size=C.sizeof(Call),qtype=12,n=w['n'],k=w['k'],experts=w['experts'],
        rows=w['rows'],mode=2 if indexed else 0,input_type=1,channels=w['channels'],topk=8 if indexed else 1,
        a_row_stride=w['k']+8,a_token_stride=(w['k']+8)*w['channels'],ids_stride=11 if indexed else 0,
        out_row_stride=w['n']+8,a=0x2000000000,low=0x1000000000,units=0x1800000000,
        ids=0x4000000000 if indexed else None,output=0x3000000000)


def select(lib,c,arr=None):
    a=arrangement(12) if arr is None else arr;config=Config();sizes=Sizes()
    rc=lib.quactlize_kpack_q4_decode_select_v1(C.byref(c),C.byref(a),C.byref(config),C.byref(sizes))
    return rc,config,sizes


def test_all_372_actual_c_choices(library,policy,evidence):
    replay={r['case']:r for r in policy['replay']}
    for c in evidence['cases']:
        w=c['workload'];call=make_call(w);rc,f,s=select(library,call)
        wanted=replay[w['id']]['recipe']
        if wanted.startswith('simt:'):
            assert rc==0
            assert f'simt:r{f.reader}-v{f.variant}-w{f.warps}-p{f.values}-c{f.columns}'==wanted
            assert s.workspace_bytes==0 and s.high_bytes==0
            assert library.quactlize_kpack_q4_decode_run_v1(C.byref(call),C.byref(f),C.byref(arrangement(12)))==123
        else:
            assert rc==24 and not any(bytes(f)) and not any(bytes(s))
            r=Request(1,C.sizeof(Request),12,2 if call.mode==2 else 0,call.rows,call.n,call.k,call.experts,w['tokens'],0x51344b5034540001)
            out=C.create_string_buffer(512)
            assert library.q4_decode_tc(C.byref(r),out,len(out))==7
            assert out.value.decode()==wanted


@pytest.mark.parametrize('field,value',[('qtype',13),('mode',1),('rows',9),('input_type',0),
    ('channels',2),('version',2),('size',0),('n',768),('k',1024),('experts',2),('a',0x2000000004),
    ('a_row_stride',2049)])
def test_unmeasured_or_invalid_requests_do_not_arm(library,field,value):
    w=dict(operator='dense',n=512,k=2048,experts=1,rows=1,channels=1)
    c=make_call(w);setattr(c,field,value)
    assert select(library,c)[0]!=0


def test_indexed_scope_and_recipe_buffer_guards(library):
    w=dict(operator='grouped',n=512,k=2048,experts=256,rows=8,channels=1)
    c=make_call(w);rc,f,_=select(library,c);assert rc==0
    for field,value in [('experts',128),('topk',4),('channels',2),('rows',72),('ids_stride',7)]:
        d=Call.from_buffer_copy(c);setattr(d,field,value);assert select(library,d)[0]!=0
    arr=arrangement(12)
    for field in ['mapping_id','artifact_tile_k','bits','high_bits','reserved','transport_tile_k']:
        a=Arrangement.from_buffer_copy(arr);setattr(a,field,getattr(a,field)^1)
        assert select(library,c,a)[0]!=0
    run=library.quactlize_kpack_q4_decode_run_v1
    changed=Config.from_buffer_copy(f);changed.warps+=1
    assert run(C.byref(c),C.byref(changed),C.byref(arr))!=123
    for field,value in [('ids',None),('units',None),('high',0x8000),('output',c.a),('output',c.low)]:
        d=Call.from_buffer_copy(c);setattr(d,field,value)
        assert run(C.byref(d),C.byref(f),C.byref(arr))!=123


def body(text,start):
    pos=text.index('{',text.index(start));depth=1;end=pos+1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return re.sub(r'\s+','',text[pos:end])


def test_numerically_admitted_kernel_bodies_are_unchanged():
    previous=(ROOT/'dev/gemv_ppu/decode_kernel.cuh').read_text()
    current=(ROOT/'quactlize/execution/q4_decode_kernel.cuh').read_text()
    assert body(previous,'__global__ void kernel')==body(current,'__global__ void kernel')
    current=(ROOT/'quactlize/execution/q4_decode_io.cu').read_text()
    for file,function in [('decode_io.cu','convert'),('moe_compare_io.cu','prepare'),('moe_compare_io.cu','finish')]:
        old=(ROOT/'dev/gemv_ppu'/file).read_text()
        assert body(old,'__global__ void '+function)==body(current,'__global__ void '+function)


def test_compile_only_selected_closure(policy):
    shapes=codegen.recipes();sources=codegen.sources()
    expected={(r['n'],r['k'],tuple(int(x[1:]) for x in r['recipe'][5:].split('-')))
        for r in policy['ranges'] if r['role']=='auto' and r['recipe'].startswith('simt:')}
    assert {(n,k,tuple(c)) for (n,k),rs in shapes.items() for c in rs}==expected
    assert len(sources)==len(shapes)+1
    assert all('dev/' not in s and 'cuda' not in s for s in sources.values())


def test_llama_automatic_binding_and_abi_copy():
    path=Path('/root/llama.cpp/ggml/src/ggml-cuda')
    if not path.is_dir():pytest.skip('private llama integration checkout is not present')
    code=(path/'quactlize-execution.cu').read_text()
    assert 'p->direct = p->q4_decode = rc == QKG_OK' in code
    assert 'if (!p->sf && !p->q4_tc)' in code
    assert 'p.api->q4_prepare(&p.gemv, p.a, p.bounds, p.ids_dst)' in code
    assert code.index('p.api->q4_run(')<code.index('if (p.indexed || p.dense_io)')
    # The existing fused-chain matcher cannot treat a direct SIMT plan as TC.
    assert 'plans[j]->legacy || plans[j]->direct' in code
    assert (path/'quactlize/kpack_q4_decode.h').read_text().strip()==(
        ROOT/'quactlize/execution/q4_decode.h').read_text().replace('"api.h"','"kpack_execution.h"').strip()


def test_selected_gate_control_and_boundaries(policy):
    from tools import run_q4_decode_policy_gate as gate
    production=json.loads((ROOT/'prebuilt/ppu0010/q4-decode-policy-v1/manifest.json').read_text())
    assert gate.control_identity(gate.sweep.BUNDLE,production)
    work=gate.workloads(policy)
    assert len(work)==233 and len({w['id'] for w in work})==233
    for r in policy['ranges']:
        if r['role']=='auto':
            for token in (r['first'],r['last']):
                assert any((w['operator'],w['n'],w['k'],w['tokens'])==
                    (r['operator'],r['n'],r['k'],token) for w in work)


@pytest.mark.parametrize('fault',['identity','device','key','samples','nan','error','median','ring'])
def test_selected_gate_rejects_stale_or_invalid_resume(fault):
    from tools.run_q4_decode_policy_gate import validate_result
    w={'id':'one'};device={'pci':'0:1'}
    record=dict(status='PASS',workload=w,selected='one',device=device,
        scope='PRODUCTION_F32_CALL_NOT_MODEL',first_launch='EXCLUDED',
        samples_us=[2.]*15,error=.0001,median_us=2.,copies=3,calls_per_graph=33)
    assert validate_result(record,w,'one',device)==record
    record=copy.deepcopy(record)
    if fault=='identity':record['workload']['id']='different'
    if fault=='device':record['device']['pci']='different'
    if fault=='key':record['selected']='different'
    if fault=='samples':record['samples_us'].pop()
    if fault=='nan':record['samples_us'][0]=float('nan')
    if fault=='error':record['error']=.1
    if fault=='median':record['median_us']=1.
    if fault=='ring':record['calls_per_graph']=32
    with pytest.raises(ValueError):validate_result(record,w,'one',device)
