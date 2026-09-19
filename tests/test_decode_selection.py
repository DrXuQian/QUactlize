"""The public production decision must not omit a confirmed winner."""
import copy
import ctypes as C
import json
from pathlib import Path
import subprocess

import pytest

from tools import generate_decode_selection as generate
from quactlize.dispatch.planning import plan_smallm, requirements
from quactlize.dispatch.native import Dispatch
from quactlize.execution.native import Call, arrangement

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def evidence():return json.loads(generate.EVIDENCE.read_text())


def test_generated_catalog_and_implementation_are_current(evidence):
    policy=generate.effective(evidence)
    assert generate.header(policy)==generate.OUTPUT.read_text()
    assert generate.implementations(evidence)==generate.IMPLEMENTATIONS.read_text()
    assert len(evidence['rows'])==19
    assert len(policy['exact'])==len({tuple(r['key']) for r in policy['exact']})==1852
    assert all(w['key'][7]==1 for w in evidence['rows'])
    # Reducer controls are not selected minima, and M1 timing is not M2..8 timing.
    assert not any(w['point']['name'].startswith('reducer-') for w in evidence['rows'])


@pytest.mark.parametrize('field',('config','body','key','median'))
def test_edited_winner_is_rejected(evidence,field):
    e=copy.deepcopy(evidence);r=e['rows'][0]
    if field=='config':r['config']['split']=1
    elif field=='body':r['body']['hoist']=True
    elif field=='key':r['key'][8]=1
    else:next(iter(r['measurements'].values()))['median_us']*=.1
    with pytest.raises(ValueError,match='edited'):generate.effective(e)


@pytest.fixture(scope='module')
def selected(tmp_path_factory,evidence):
    rows=[w for w in evidence['rows'] if not w['point']['paired'] and w['automatic']]
    plan=plan_smallm(tmp_path_factory.mktemp('winner-query'),[r['key'] for r in rows])
    return rows,plan


def assert_winner(row,w):
    assert row['status']==0 and row['policy']==12 and row['donor']==w['key'][2:4]+[1]
    if w['config']['kind']=='tc':
        assert row['kind']==0 and row['parent']['symbol']==w['config']['symbol']
        assert row['split']==w['config']['split']
    else:
        assert row['kind']==1
        assert row['config']=={k:w['config'][k] for k in generate.FIELDS}
        if w['implementation']=='constant-split':
            impl=row['implementation']
            assert impl['measured']==w['point']['name'].replace('-','_')
            for field in ('hoist','fixed','changes'):assert impl[field]==w['body'][field]


def test_all_confirmed_winners_are_selected(selected):
    rows,plan=selected
    for r,w in zip(plan['requests'],rows):assert_winner(r,w)
    head=next(r for r in plan['requests'] if r['request'][0]==14)
    assert head['kind']==0  # Removing a TC winner to simplify policy is a regression.


def test_actual_c_abi_matches_the_selected_simt_winners(tmp_path,selected):
    (tmp_path/'catalog.inc').write_text('static std::vector<Image> kImages; static char const kJitSource[]="";\n')
    subprocess.run(['g++','-std=c++17','-O2','-pthread','-shared','-fPIC','-Wl,-Bsymbolic',
        '-I'+str(tmp_path),ROOT/'quactlize/dispatch/binding.cpp','-ldl',
        '-o',tmp_path/'libquactlize_kpack_dispatch.so'],check=True)
    d=Dispatch(tmp_path)
    try:
        for w in selected[0]:
            if w['config']['kind']!='simt':continue
            q,mode,n,k,e,top,ch,m,compute=w['key']
            c=Call(version=1,size=C.sizeof(Call),qtype=q,mode=mode,n=n,k=k,experts=e,topk=top,
                channels=ch,rows=m*top if mode else m,input_type=1,a_row_stride=k,
                a_token_stride=k*ch,ids_stride=top,out_row_stride=n)
            got=d.query_smallm_matched(c,arrangement(q),compute)
            assert got and got.base.kind==1 and got.base.policy==12
            assert {f:getattr(got.base.simt,f) for f in generate.FIELDS}=={
                f:w['config'][f] for f in generate.FIELDS}
    finally:d.close()


def test_measured_implementation_does_not_leak_to_another_semantic_key(tmp_path,evidence):
    source=tmp_path/'scope.cpp';binary=tmp_path/'scope'
    source.write_text('''#include "quactlize/execution/measured_decode.hpp"
#include <cassert>
using namespace quactlize::execution::simt;
int main(){
#define QDM(Id,Q,Mode,N,K,E,Top,Ch,Compute,V,C,W,P,S,Changes,Hoist,Fixed) \
{qkg_simt_call_v2 d{};auto& c=d.call;d.compute_type=Compute; \
c.qtype=Q;c.mode=Mode;c.n=N;c.k=K;c.experts=E;c.topk=Top;c.channels=Ch; \
c.rows=Top;c.input_type=QKG_F32;qkg_simt_config_v1 f{1,sizeof(f),V,C,W,P,S}; \
assert(measured_decode(d,f)==MeasuredDecode::Id); \
c.rows*=2;assert(measured_decode(d,f)==MeasuredDecode::None);c.rows=Top; \
d.compute_type^=1;assert(measured_decode(d,f)==MeasuredDecode::None);d.compute_type=Compute; \
c.input_type=QKG_F16;assert(measured_decode(d,f)==MeasuredDecode::None);c.input_type=QKG_F32; \
c.channels+=1;assert(measured_decode(d,f)==MeasuredDecode::None);c.channels=Ch; \
c.n+=256;assert(measured_decode(d,f)!=MeasuredDecode::Id);}
#include "quactlize/execution/measured_decode.inc"
#undef QDM
}
''')
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT),source,'-o',binary],check=True)
    subprocess.run([binary],check=True)


def test_fallback_uses_updated_donor_but_not_its_fixed_dimensions(tmp_path):
    plan=plan_smallm(tmp_path,[[8,0,6656,3072,1,1,1,1,0]])
    r=plan['requests'][0]
    assert r['status']==0 and r['policy']==13
    assert r['implementation']['measured']=='none' and not r['implementation']['fixed']
    assert r['config']['variant']==5


def test_device_handoff_is_numerical_only_and_keeps_tc_winners():
    from tools.run_selected_decode_numerics import gate_rows
    rows=gate_rows()
    assert len(rows)==17 and all(r['config']['kind']=='simt' for r in rows)
    assert sum(r['point']['paired'] for r in rows)==3
    assert not any(r['point']['name']=='tp2-q4-paired-routed' for r in rows)
    text=(ROOT/'tools/run_selected_decode_numerics.py').read_text()
    assert 'timing_graph' not in text and 'token_controls=range(1, 9)' in text
    box=(ROOT/'tools/run_kpack_tp2_box.sh').read_text()
    assert box.index('stage=selected-entry-numerics')<box.index('stage=model-perf')


def test_tc_source_does_not_depend_on_simt_row_reducer():
    # TC and SIMT partial layouts differ. Preserve old TC module identities
    # rather than rebuilding or relabeling them for a SIMT-only improvement.
    old=subprocess.check_output(['git','show',
        'b04cb96347beccb682dd298486394ea4e99cd7a1:quactlize/decode/reducer.cuh'],cwd=ROOT)
    assert (ROOT/'quactlize/decode/reducer.cuh').read_bytes()==old
    body=(ROOT/'quactlize/execution/simt_kernel.cuh').read_text()
    assert '#include "simt_reducer.cuh"' in body


def test_instruction_comparison_is_not_only_opcode_counts():
    from tools.inspect_selected_decode import instruction_bytes
    text='Disassembly of section .text.kernel.a:\n 0: 00 01 02 03 \ts.foo r0\n'
    assert instruction_bytes(text,'a')==bytes((0,1,2,3))
    assert instruction_bytes(text.replace('00 01','00 02'),'a')!=instruction_bytes(text,'a')
    with pytest.raises(ValueError):instruction_bytes(text,'b')
    with pytest.raises(ValueError):instruction_bytes(text+text,'a')
