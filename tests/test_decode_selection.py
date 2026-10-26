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
    assert len(policy['exact'])==len({tuple(r['key']) for r in policy['exact']})==2095
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
    catalog={tuple(r['key']):r for r in generate.effective(evidence)['exact']}
    rows=[copy.deepcopy(w) for w in evidence['rows'] if not w['point']['paired'] and w['automatic']]
    for w in rows:
        current=catalog[tuple(w['key'])]
        # Only later confirmed component evidence may supersede an old winner.
        if current['policy']==18:
            w.update(config=current['config'],implementation='existing')
        else:
            assert w['config']==current['config']
        w['policy']=current['policy']
    plan=plan_smallm(tmp_path_factory.mktemp('winner-query'),[r['key'] for r in rows])
    return rows,plan


def assert_winner(row,w):
    assert row['status']==0 and row['policy']==w['policy'] and row['donor']==w['key'][2:4]+[1]
    if w['config']['kind']=='tc':
        assert row['kind']==0 and row['parent']['symbol']==w['config']['symbol']
        assert row['split']==w['config']['split']
    elif w['config']['kind']=='q4':
        assert row['kind']==2
        assert row['config']=={k:v for k,v in w['config'].items() if k!='kind'}
    else:
        assert row['kind']==1
        assert row['config']=={k:w['config'][k] for k in generate.FIELDS}
        if w['implementation']=='constant-split':
            impl=row['implementation']
            if impl['producer']=='common-packet-3d':
                # The admitted common reader superseded non-fixed historical
                # bodies, without changing this measured geometry/selection.
                q, mode, n, k, e, top, ch, m, compute = w['key']
                f = w['config']
                assert 10 <= q <= 14 and compute in (0, 1) and f['variant'] == 3
                assert (f['columns'], f['warps'], f['values']) in ((4, 2, 8), (4, 8, 8), (8, 4, 4)) or (
                    q == 12 and (f['columns'], f['warps'], f['values']) == (4, 4, 4))
                assert not w['body']['fixed'] and impl['measured'] == 'none'
                return
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
            assert got and got.base.kind==1 and got.base.policy==w['policy']
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
    assert len(rows)==17 and sum(r['config']['kind']=='simt' for r in rows)==16
    assert sum(r['config']['kind']=='q4' for r in rows)==1
    assert sum(r['point']['paired'] for r in rows)==3
    assert not any(r['point']['name']=='tp2-q4-paired-routed' for r in rows)
    from tools.run_selected_decode_numerics import entry_checks
    catalog={tuple(r['key']):r for r in generate.effective(json.loads(generate.EVIDENCE.read_text()))['exact']}
    for row in rows:
        if not row['point']['paired']:
            assert row['config']==catalog[tuple(row['key'])]['config']
    assert all(entry=='v2' for row,entry in entry_checks(rows) if row['config']['kind']=='q4')
    text=(ROOT/'tools/run_selected_decode_numerics.py').read_text()
    assert 'timing_graph' not in text and 'token_controls=(1,) if provider.q4 else range(1, 9)' in text
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


def test_actual_generated_v1_and_v2_dispatch_share_implementation(tmp_path):
    from quactlize.execution import simt_codegen
    stub = tmp_path/'measured_decode.cuh'
    stub.write_text(r'''#pragma once
#include "quactlize/execution/measured_decode.hpp"
#include <cstring>
inline qkg_simt_call_v2 observed{};
inline int implementation=-1, launches=0;
namespace quactlize::execution::simt {
template<int Q> int measured_decode_launch(qkg_simt_call_v2 const& d,qkg_simt_config_v1 const& f) {
    observed=d;implementation=int(measured_decode(d,f));++launches;return QKG_OK;
}
template<int Q,int V,int C,int W,int P> int launch_v2(qkg_simt_call_v2 const& d,int) {
    observed=d;implementation=0;++launches;return QKG_OK;
}
template<int Q,int V,int C,int W,int P> int launch(qkg_call_v1 const& c,int s) {
    return launch_v2<Q,V,C,W,P>({2,sizeof(qkg_simt_call_v2),c,QKG_COMPUTE_F16},s);
}
namespace q8_vector { using simt::launch;using simt::launch_v2; }
}
''')
    (tmp_path/'simt_common.cuh').write_text(r'''#pragma once
#include "measured_decode.cuh"
#include "quactlize/execution/simt_common_scope.hpp"
namespace quactlize::execution::simt {
template<int Q,int C,int W,int P> int common_reader_launch(qkg_simt_call_v2 const& d,int) {
    observed=d;implementation=-2;++launches;return QKG_OK;
}
}
''')
    (tmp_path/'simt_q8_refresh.cuh').write_text(r'''#pragma once
#include "measured_decode.cuh"
#include "quactlize/execution/simt_q8_refresh.hpp"
namespace quactlize::execution::simt::q8_refresh {
inline int launch(qkg_simt_call_v2 const& d,qkg_simt_config_v1 const& f) {
    observed=d;implementation=-100-q8_refresh_body(d,f);++launches;return QKG_OK;
}
}
''')
    token_stub = r'''
#include "measured_decode.cuh"
namespace quactlize::execution::simt {
template<int T,int C,int W,int P,int Q=14>
int token_cta_launch(qkg_simt_call_v2 const& d,int) {
    observed=d;implementation=-200-T;++launches;return QKG_OK;
}
template<int T,int C,int W,int P,int Window>
int q8_token_cta_launch(qkg_simt_call_v2 const& d,int) {
    observed=d;implementation=-200-T;++launches;return QKG_OK;
}
}
'''
    token_source=(ROOT/'quactlize/execution/simt_token_cta_dispatch.cuh').read_text()
    token_source=token_source.replace('#include "simt_token_cta_scope.hpp"',
        '#include "quactlize/execution/simt_token_cta_scope.hpp"')
    token_source=token_source.replace('#include "simt_token_cta.cuh"',token_stub)
    token_source=token_source.replace('#include "simt_q8_token_cta.cuh"','')
    token_source=token_source.replace('#include "token_cta.inc"','#include "quactlize/execution/token_cta.inc"')
    (tmp_path/'simt_token_cta_dispatch.cuh').write_text(token_source)
    generated=[]
    for q in simt_codegen.QTYPES:
        path=tmp_path/f'launch{q}.cpp'
        path.write_text(simt_codegen.source(q))
        generated.append(path)
    main=tmp_path/'entry.cpp'
    main.write_text(r'''#include "measured_decode.cuh"
#include "quactlize/execution/simt_validation.hpp"
#include "quactlize/execution/simt_common_scope.hpp"
#include "quactlize/execution/simt_q8_refresh.hpp"
#include "quactlize/execution/simt_token_cta_scope.hpp"
#include <cassert>
using namespace quactlize::execution;
static void check(qkg_call_v1 c,qkg_simt_config_v1 f,int compute,bool measured) {
    auto a=c.qtype==8?q8_kpack2::arrangement():c.qtype==12?
        ppu_arrangements::q4_kpack4_transpose_v1():ppu_arrangements::kquant_kpack_transpose_v1(c.qtype);
    c.version=1;c.size=sizeof(c);c.a_row_stride=c.k+8;c.a_token_stride=c.channels*c.a_row_stride;
    c.out_row_stride=c.n+8;c.ids_stride=c.topk+3;
    c.a=(void*)(1ull<<36);c.low=(uint8_t*)(2ull<<36);c.units=(uint8_t*)(4ull<<36);
    c.output=(float*)(6ull<<36);c.workspace=(void*)(7ull<<36);c.workspace_bytes=1ull<<32;
    c.ids=c.mode==QKG_INDEXED?(int32_t*)(5ull<<36):nullptr;
    qkg_simt_call_v2 d{2,sizeof(d),c,compute};qkg_sizes_v1 sizes{};
    assert(quactlize_kpack_simt_query_v2(&d,&f,&a,&sizes)==QKG_OK);
    d.call.high=sizes.high_bytes?(uint8_t*)(3ull<<36):nullptr;c=d.call;
    launches=0;
    assert(quactlize_kpack_simt_run_v2(&d,&f,&a)==QKG_OK && launches==1);
    assert((implementation!=0)==measured);
    if (simt::token_cta_window(d,f)>=0) assert(implementation==-200-c.rows);
    else if (simt::q8_refresh_body(d,f)) assert(implementation==-100-simt::q8_refresh_body(d,f));
    else if (simt::common_reader_scope(d,f)) assert(implementation==-2);
    else if (measured) assert(implementation==int(simt::measured_decode(d,f)));
    assert(observed.compute_type==compute && !memcmp(&observed.call,&c,sizeof(c)));
    int expected=implementation;
    auto rebound=d;
    rebound.call.a=(void*)((uintptr_t)c.a+4096);
    rebound.call.output=(float*)((uintptr_t)c.output+4096);
    rebound.call.stream=(void*)1234;
    assert(quactlize_kpack_simt_run_v2(&rebound,&f,&a)==QKG_OK);
    assert(implementation==expected && !memcmp(&observed.call,&rebound.call,sizeof(c)));
    if(compute==QKG_COMPUTE_F16) {
        launches=0;implementation=-1;
        assert(quactlize_kpack_simt_run_v1(&c,&f,&a)==QKG_OK && launches==1);
        assert(implementation==expected && observed.compute_type==QKG_COMPUTE_F16);
        assert(!memcmp(&observed.call,&c,sizeof(c)));
    }
    launches=0;
    auto bad=c;bad.a=nullptr;
    assert(quactlize_kpack_simt_run_v1(&bad,&f,&a)==QKG_INVALID && launches==0);
    bad=c;bad.input_type=QKG_SIMT_BF16;
    assert(quactlize_kpack_simt_run_v1(&bad,&f,&a)==QKG_INVALID && launches==0);
    if(f.split>1) {
        bad=c;bad.workspace_bytes=sizes.workspace_bytes-1;
        assert(quactlize_kpack_simt_run_v1(&bad,&f,&a)==QKG_CAPACITY && launches==0);
    }
}
int main() {
#define QDM(Id,Q,Mode,N,K,E,Top,Ch,Compute,V,C,W,P,S,Changes,Hoist,Fixed) \
{qkg_call_v1 c{};c.qtype=Q;c.mode=Mode;c.n=N;c.k=K;c.experts=E;c.topk=Top; \
c.channels=Ch;c.rows=Top;c.input_type=QKG_F32; \
check(c,{1,sizeof(qkg_simt_config_v1),V,C,W,P,S},Compute,true);}
#include "quactlize/execution/measured_decode.inc"
#undef QDM
#define Q8R(N,K,M,C,W,Body) \
{qkg_call_v1 c{};c.qtype=8;c.mode=QKG_DENSE;c.n=N;c.k=K;c.experts=1;c.topk=1; \
c.channels=1;c.rows=M;c.input_type=QKG_F32; \
check(c,{1,sizeof(qkg_simt_config_v1),5,C,W,4,1},0,true);}
#include "quactlize/execution/q8_refresh.inc"
#undef Q8R
#define QTC(Q,N,K,M,V,C,W,P,S,Window) \
{qkg_call_v1 c{};c.qtype=Q;c.mode=QKG_DENSE;c.n=N;c.k=K;c.experts=1;c.topk=1; \
c.channels=1;c.rows=M;c.input_type=QKG_F32; \
check(c,{1,sizeof(qkg_simt_config_v1),V,C,W,P,S},0,true);}
#include "quactlize/execution/token_cta.inc"
#undef QTC
for(int q:{8,10,11,12,13,14}) for(int tokens=1;tokens<=8;++tokens) for(int mode:{0,2})
for(int compute:{0,1}) for(int split:{1,2,4,8}) {
    qkg_call_v1 c{};c.qtype=q;c.mode=mode;c.n=256;c.k=512;c.experts=mode?256:1;
    c.topk=mode?8:1;c.channels=mode?8:1;c.rows=tokens*c.topk;c.input_type=QKG_F32;
    check(c,{1,sizeof(qkg_simt_config_v1),1,4,4,4,split},compute,false);
}
}
''')
    exe=tmp_path/'entries'
    subprocess.run(['g++','-std=c++17','-O1','-I'+str(tmp_path),'-I'+str(ROOT),
                    ROOT/'quactlize/execution/simt.cpp',*generated,main,'-o',exe],check=True)
    subprocess.run([exe],check=True)
