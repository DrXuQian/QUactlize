import json
from pathlib import Path
import subprocess

from tools.import_kpack_cost_policy import candidate, choose_profiles, header

ROOT=Path(__file__).resolve().parents[1]


def test_generated_cost_table_matches_audited_receipt():
    report=json.loads((ROOT/'docs/measurements/kpack_component_policy_20260914.json').read_text())
    assert len(report['points'])==502 and report['components_audited']==2385
    assert len(report['knots'])==430
    assert (ROOT/'policies/kpack_zw810_cost_v1.hpp').read_text()==header(report)
    assert not report['small_m_full_dequant'] and not report['external_adapters_included']
    for row in report['knots']:
        assert row['tokens']>=128
        for mask,c in row['choices'].items():
            assert int(mask)&(1<<c['route'])
            if c['route']==0:assert c['dequant_us']==0
            if c['route']==2 and row['experts']>1:assert c['dequant_config'] in (4,5,10,11)
            if c['route']<2:
                assert c['tactic']==row['choices'][str(1<<c['route'])]['tactic']


def test_tie_prefers_fq_and_unstable_times_do_not_win():
    choices=[candidate(0,103,tactic={'parent':'fq'}),candidate(1,90,10,tactic={'parent':'sf'}),
             candidate(2,1,stable_timing=False)]
    c=choose_profiles([dict(id='a',profile='real',candidates=choices)])
    assert c['route']==0 and c['dequant_us']==0


def test_grouped_profile_specific_tactic_cannot_leak_to_unknown_router():
    a=[candidate(0,20,tactic={'parent':'common'}),candidate(0,10,tactic={'parent':'a'})]
    b=[candidate(0,20,tactic={'parent':'common'}),candidate(0,10,tactic={'parent':'b'})]
    c=choose_profiles([dict(id='a',profile='real',candidates=a),dict(id='b',profile='active32',candidates=b)])
    assert c['tactic']['parent']=='common'
    assert c['worst_measured_regret_pct']==100


def test_host_cost_selector_and_small_m_boundaries(tmp_path):
    source=tmp_path/'test.cpp';exe=tmp_path/'test'
    source.write_text(r'''
#include "quactlize/dispatch/policy.hpp"
#include <cassert>
int main() {
 using namespace quactlize::dispatch;
 for (auto const& k:cost::data::kKnots) {
   qks_request_v1 r{1,sizeof(r),k.q,k.experts==1 ? 0:2,k.tokens*(k.experts==1 ? 1:8),k.n,k.k,k.experts,k.tokens,
     k.q==12 ? UINT64_C(0x51344b5034540001):UINT64_C(0x514b504b54000001)};
   for (int mask:{1,3,7}) {
     qks_prefill_choice_v1 c{};assert(cost::query(r,mask,c)==0);
     auto want=k.choices[mask==7 ? 3:mask==3 ? 2:0];
     assert(!c.predicted && c.measured_tokens==k.tokens && c.route==want.route && c.dequant_config==want.dequant);
   }
   assert(select(r).config && select(r).policy==QKS_COMPONENT_MEASURED);
   r.m=r.max_rows=1;assert(cost::knot(r)==nullptr);
   r.m=r.max_rows=8;assert(cost::knot(r)==nullptr);
   r.max_rows=4097;r.m=4097*(k.experts==1 ? 1:8);assert(cost::knot(r)==nullptr);
 }
}
''')
    subprocess.run(['g++','-std=c++17','-O2',f'-I{ROOT}',str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


def test_dense_prefill_n_extension_retains_recipe_and_measured_domains(tmp_path):
    source=tmp_path/'extension.cpp';exe=tmp_path/'extension'
    source.write_text(r'''
#include "quactlize/dispatch/policy.hpp"
#include <cassert>
int main() {
 using namespace quactlize::dispatch;
 for (int q:{10,11,12,13,14}) for (int route:{0,1})
 for (int k:{2048,5120}) for (int m:{128,512,1024,1025,2048,4096}) {
   qks_request_v1 r{1,sizeof(r),q,route,m,248320,k,1,m,
     q==12 ? UINT64_C(0x51344b5034540001):UINT64_C(0x514b504b54000001)};
   assert(!select_same_family(r).config && !cost::knot(r));
   auto s=select(r);
   assert(s.config && s.policy==QKS_PREDICTED && s.config->split==1 && !s.config->ap);
   int largest=0; Config const* want=nullptr;
   for (auto const& knot:cost::data::kKnots) {
     if (knot.q!=q || knot.k!=k || knot.experts!=1 || knot.n>=r.n || knot.n<=largest) continue;
     auto donor=r; donor.n=knot.n;
     auto c=cost::fixed_route(donor);
     if (!c || c->split!=1 || c->ap || r.n%c->tn || k%c->tk || k/c->tk<c->stages-1) continue;
     largest=knot.n; want=c;
   }
   assert(s.config==want);
   auto recipe_real=recipe(*s.config,r,4);
   assert(recipe_real.split==1 && recipe_real.grid>=0);
   // The cost API must not invent measured head or full-dequant timings.
   qks_prefill_choice_v1 cost{};assert(cost::query(r,7,cost)==QKS_MISS);
   for (int outside:{1,8,127,4097}) {
     auto bad=r; bad.m=bad.max_rows=outside; assert(!select(bad).config);
   }
   auto unknown=r; unknown.k=3584; assert(!select(unknown).config);
   auto invalid=r; invalid.mapping_id=0; assert(!select(invalid).config);
   invalid=r; invalid.n--; assert(!select(invalid).config);
   invalid=r; invalid.n=151936; assert(!select(invalid).config);
 }
 // Exact and interpolated component choices remain higher priority.
 for (auto const& knot:cost::data::kKnots) if (knot.experts==1)
 for (int route:{0,1}) {
   qks_request_v1 r{1,sizeof(r),knot.q,route,knot.tokens,knot.n,knot.k,1,knot.tokens,
     knot.q==12 ? UINT64_C(0x51344b5034540001):UINT64_C(0x514b504b54000001)};
   auto c=cost::fixed_route(r);
   if (c) {assert(select(r).config==c);assert(select(r).policy==QKS_COMPONENT_MEASURED);}
 }
}
''')
    subprocess.run(['g++','-std=c++17','-O2',f'-I{ROOT}',str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
