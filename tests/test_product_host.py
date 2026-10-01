import ctypes
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quactlize.gguf import read_gguf_header
from quactlize.packing.native import Arrangement, canonical_arrangement
from quactlize.runtime.compiler import Compiler, validate_parent
from quactlize.runtime.identity import digest
from tools import paired_decode_selection


@pytest.fixture(scope='module')
def selector(tmp_path_factory):
    binary = tmp_path_factory.mktemp('selector') / 'query'
    subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(ROOT),
                    str(ROOT/'tools/kpack_selection.cpp'), '-o', str(binary)], check=True)
    return binary


def test_measured_inventory_decisions(selector):
    output = subprocess.check_output([selector, '--inventory'])
    assert len(output.splitlines()) == 1852
    assert hashlib.sha256(output).hexdigest() == '83973cc53afbffc0eb6dc0eb0445e41bc89a1814eed32522f1fb7a5f28ab7012'


def test_bucket_tp_and_m_boundaries(selector):
    shapes = [(256,512),(512,2048),(1024,3072),(1024,5120),(2048,1024),
              (3072,512),(4096,2048),(5120,8192),(5120,25600),(8192,5120),
              (25600,5120),(512,768),(1536,4096),(768,512),(512,256)]
    requests = [[q,mode,n,k,e,top,ch,m,compute]
                for q in (8,10,11,12,13,14) for compute in (0,1)
                for n,k in shapes for m in range(1,10)
                for mode,e,top,ch in ((0,1,1,1),(2,256,8,1),(2,256,8,8),(2,128,4,1))]
    data = ''.join(' '.join(map(str,r))+'\n' for r in requests).encode()
    output = subprocess.check_output([selector], input=data)
    assert len(output.splitlines()) == len(requests) == 6480
    assert hashlib.sha256(output).hexdigest() == 'b33d88f72afd555948774cc2676b356a863dbb1261d5596cafbcb3a4e07a2072'


def test_canonical_descriptor_abi(tmp_path):
    assert ctypes.sizeof(Arrangement) == 40
    code = r'''
#include "ppu_placed_arrangement.hpp"
#include "q8_kpack2.hpp"
#include <cstddef>
#include <iostream>
int main() {
  static_assert(sizeof(quactlize_ppu_placed_arrangement_v2)==40);
  static_assert(offsetof(quactlize_ppu_placed_arrangement_v2,mapping_id)==32);
  for(int q: {8,10,11,12,13,14}) {
    auto a=q==8?q8_kpack2::arrangement():q==12?ppu_arrangements::q4_kpack4_transpose_v1():
      ppu_arrangements::kquant_kpack_transpose_v1(q);
    std::cout<<q<<' '<<a.version<<' '<<a.layout<<' '<<a.bits<<' '<<a.high_bits<<' '
      <<a.artifact_tile_k<<' '<<a.transport_tile_k<<' '<<a.group_size<<' '<<a.reserved<<' '<<a.mapping_id<<'\n';
    if(q!=8) {
      if(!ppu_arrangements::matches_canonical_kpack(&a,q)) return 1;
      for(int i=0;i<9;++i) {
        auto bad=a;
        if(i==8) ++bad.mapping_id; else reinterpret_cast<int*>(&bad)[i]++;
        if(ppu_arrangements::matches_canonical_kpack(&bad,q)) return 2;
      }
    }
  }
}
'''
    source, binary = tmp_path/'abi.cpp', tmp_path/'abi'
    source.write_text(code)
    subprocess.run(['g++', '-std=c++17', '-I'+str(ROOT/'quactlize/include'),
                    str(source), '-o', str(binary)], check=True)
    for line in subprocess.check_output([binary], text=True).splitlines():
        q, *values = map(int, line.split())
        actual = canonical_arrangement(q)
        assert values == [getattr(actual, name) for name,_ in Arrangement._fields_]
    for q in (-1,0,9,15):
        with pytest.raises(ValueError):
            canonical_arrangement(q)


def test_parent_validation_has_no_layout_guess():
    parent = dict(symbol='q3', qtype=11, route='fq-dense',tm=16,tn=64,tk=256,
                  wm=16,wn=16,stages=2,ap=0,dn=16,persistent=-1)
    validate_parent(parent)
    assert '#include "module.cuh"' in Compiler.source(None, parent, 'identity')
    for change in ({'qtype':9}, {'tk':128}, {'symbol':'x;bad'}, {'dn':48}, {'ap':1}):
        with pytest.raises(ValueError):
            validate_parent(parent | change)
    with pytest.raises(ValueError):
        digest({'cost':float('nan')})


def test_paired_catalog_matches_compiled_recipes():
    catalog = json.loads(paired_decode_selection.EVIDENCE.read_text())
    assert paired_decode_selection.header(catalog) == paired_decode_selection.OUTPUT.read_text()


def test_header_reader_fails_on_truncation_and_unknown_version():
    with pytest.raises(ValueError):
        read_gguf_header(io.BytesIO(b'GGUF'), 'truncated')
    with pytest.raises(ValueError):
        read_gguf_header(io.BytesIO(b'GGUF'+struct.pack('<I',99)), 'unknown')
    header = read_gguf_header(io.BytesIO(b'GGUF'+struct.pack('<IQQ',3,0,0)), 'empty')
    assert header['tensors'] == []


def test_imports_do_not_load_torch_or_a_device():
    code = '''
import sys
import quactlize
import quactlize.dispatch.native
import quactlize.runtime.compiler
import quactlize.packing.native
assert 'torch' not in sys.modules
assert 'numpy' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)


def test_grouped_regression_denominator():
    from tools.compare_grouped import cases
    rows=cases()
    assert len(rows)==48
    assert {r['q'] for r in rows}=={8,10,11,12,13,14}
    assert len({(r['q'],r['compute']) for r in rows})==12
    for row in rows:
        validate_parent(row['parent'])
        assert row['k']%(row['parent']['tk']*row['recipe']['split'])==0
        assert row['m']//row['experts']>row['parent']['tm'] or row['parent']['tm']==16


def test_device_gate_recipes_are_compiled():
    from quactlize.execution.simt_codegen import runtime_inventory
    for q in (8,10,11,12,13,14):
        inventory={(r.variant,r.columns,r.warps,r.values,r.split) for r in runtime_inventory(q)}
        for split in (1,2,4):
            assert (5 if q==8 else 3,4,4,4,split) in inventory
