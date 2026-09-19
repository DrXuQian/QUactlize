"""Exact measured-model scope, including guard failures and retained fallbacks."""
import subprocess
from pathlib import Path
from types import SimpleNamespace
import json
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_tc_replacement_does_not_leak_to_other_domains(tmp_path):
    code=r'''
#include "quactlize/dispatch/smallm_matched.hpp"
#include "tests/legacy_q8_overlay.hpp"
#include <cassert>
int main() {
  using namespace quactlize::dispatch;
  int found=0;
  for(auto const& row:matched::data::kExact) {
    auto const& choice=matched::data::kChoices[row.choice];
    if(choice.kind!=QKS_SMALLM_TC) continue;
    qkg_simt_call_v2 d{2,sizeof(d)};auto& c=d.call;
    c.version=1;c.size=sizeof(c);c.qtype=row.q;c.input_type=QKG_F32;
    c.mode=row.mode;c.rows=row.tokens*(row.mode==QKG_INDEXED?row.topk:1);
    c.n=row.n;c.k=row.k;c.experts=row.experts;c.topk=row.topk;c.channels=row.channels;
    d.compute_type=row.compute;
    qkg_simt_config_v1 f{};
    if(!q8_vector::select_tc(d,choice.tc,f)) continue;
    ++found;assert(row.q==8 && row.compute==0 && row.tokens==1 && f.split==1);
    for(int field=0;field<9;++field) {
      auto wrong=d;
      switch(field) {
        case 0:wrong.compute_type=QKG_COMPUTE_BF16;break;
        case 1:wrong.call.input_type=QKG_F16;break;
        case 2:wrong.call.rows=2;break;
        case 3:wrong.call.n+=256;break;
        case 4:wrong.call.k+=256;break;
        case 5:wrong.call.experts=256;break;
        case 6:wrong.call.topk=8;break;
        case 7:wrong.call.channels=8;break;
        default:wrong.call.mode=QKG_INDEXED;
      }
      assert(!q8_vector::select_tc(wrong,choice.tc,f));
    }
    auto tc=choice.tc;tc.split=4;assert(!q8_vector::select_tc(d,tc,f));
    tc=choice.tc;tc.dn=16;assert(!q8_vector::select_tc(d,tc,f));
    tc=choice.tc;tc.symbol="another-parent";assert(!q8_vector::select_tc(d,tc,f));
  }
  assert(found==2);
  qkg_call_v1 c{};c.input_type=QKG_F32;c.mode=QKG_INDEXED;c.rows=8;
  c.experts=256;c.topk=8;c.channels=1;c.n=1024;c.k=2048;
  using quactlize::execution::model_gemv::indexed_m1;
  assert(indexed_m1(c,1024,2048,1));
  c.rows=16;assert(!indexed_m1(c,1024,2048,1));c.rows=8;
  c.channels=8;assert(!indexed_m1(c,1024,2048,1));
}
'''
    source=tmp_path/'scope.cpp';source.write_text(code)
    binary=tmp_path/'scope'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT),str(source),'-o',str(binary)],check=True)
    subprocess.run([binary],check=True)


def test_integrated_body_preserves_measured_arithmetic():
    fusion=(ROOT/'quactlize/fusion/simt.cu').read_text()
    store=(ROOT/'quactlize/fusion/store.cuh').read_text()
    simt=(ROOT/'quactlize/execution/simt_kernel.cuh').read_text()
    q8=(ROOT/'quactlize/execution/simt_q8_vector.cuh').read_text()
    assert 'register_reuse_body<12,1,3,4,8,8,1,1,SimtFinish>(c,1)' in fusion
    assert 'kernel_body<1,0,1,4,8,4,true,SimtFinish>(c,1)' in fusion
    assert 'c.output_type==QKG_F32 && f.split==1 && f.warps==8' in fusion
    assert 'TileN == 16 || TileN == 32' in store
    assert 'gate += partial[w*TileN+ng]; up += partial[w*TileN+ng+4]' in store
    assert 'Q==13 && Input==QKG_F32 && Compute==QKG_COMPUTE_BF16' in simt
    assert 'kernel_model<1,0,1,8,4,4,false,2048,4096,8>' in q8
    assert 'kernel_model<1,0,1,8,4,4,true,8192,2048,1>' in q8
    assert 'quactlize::decode::reduce_decode<8>' in q8
    assert 'kernel_s1<1,0,1,4,2,4,true>' in q8
    assert 'kernel_s1<1,0,1,4,8,4,false>' in q8


def test_device_gate_requires_actual_selected_recipe():
    from tools.run_model_gemv_integration import selected_simt, SIMT_FIELDS
    config=SimpleNamespace(variant=5,columns=8,warps=4,values=4,split=1)
    def choice(**changes):
        return SimpleNamespace(base=SimpleNamespace(kind=1,policy=15,
            simt=SimpleNamespace(**(vars(config)|changes))))
    assert selected_simt(choice(),config)==dict(policy=15,config=vars(config))
    for field in SIMT_FIELDS:
        with pytest.raises(ValueError,match='selection differs'):
            selected_simt(choice(**{field:getattr(config,field)+1}),config)
    for missing in (None,SimpleNamespace(base=SimpleNamespace(kind=0))):
        with pytest.raises(ValueError,match='selection differs'):selected_simt(missing,config)


def test_frozen_reference_keeps_old_source_but_binds_all_payloads(tmp_path,monkeypatch):
    from tools import run_model_gemv_integration as gate
    (tmp_path/'tools').mkdir()
    bundle=tmp_path/'bundle';bundle.mkdir()
    (bundle/'reader.so').write_bytes(b'host-test-not-an-ELF')
    data=dict(inventory=json.loads(json.dumps(gate.inventory())),
              source_hashes={'old.cuh':'deliberately-not-the-current-source'},
              payloads={'reader.so':gate.sha(bundle/'reader.so')})
    def pin(value):
        (bundle/'manifest.json').write_text(json.dumps(value))
        (tmp_path/'tools/kpack_model_gemv_artifact.json').write_text(json.dumps(
            dict(manifest_sha256=gate.sha(bundle/'manifest.json'))))
    pin(data);monkeypatch.setattr(gate,'ROOT',tmp_path)
    assert gate.reference_manifest(bundle)==data
    (bundle/'reader.so').write_bytes(b'changed')
    with pytest.raises(ValueError,match='payload differs'):gate.reference_manifest(bundle)
    data['payloads']['reader.so']=gate.sha(bundle/'reader.so');pin(data)
    data['inventory']=[];pin(data)
    with pytest.raises(ValueError,match='inventory differs'):gate.reference_manifest(bundle)
    (bundle/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='manifest differs'):gate.reference_manifest(bundle)


def test_native_opcodes_ignore_noninstruction_headers():
    from tools.inspect_model_gemv_integration import operations
    text='''reader.so: file format elf64-unknown
        100: 00 00 00 00\tv.fma.f32.rtte vreg0, vreg1, vreg2, vreg0
        108: 11 22 33 44\tvmem.ld.b32x4 vreg[4:7], [vreg0]
Func 0 reader RESOURCE INFO:
'''
    assert operations(text)=={'v.fma.f32.rtte':1,'vmem.ld.b32x4':1}


def test_parameterized_symbols_do_not_relabel_new_dimensions_as_old_evidence():
    from tools.inspect_model_gemv_integration import SYMBOLS, PARAMETERIZED_SYMBOLS, matches_measured
    for point,new in PARAMETERIZED_SYMBOLS.items():
        assert matches_measured(point,new+'(call)')
        assert matches_measured(point,SYMBOLS[point][1]+'(call)')
        assert not matches_measured(point,new.replace('2048','3072')+'(call)')
