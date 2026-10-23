"""Replay all measured full-only choices without changing TC/SF semantics."""
import copy
import json
import subprocess

import pytest

from tools.generate_dense_full_helpers import EVIDENCE, HEADER, ROOT, render


def test_complete_typed_table_is_generated():
    data = json.loads(EVIDENCE.read_text())
    assert render(data) == HEADER.read_text()
    assert len(data['rows']) == 350
    assert {(r['q'], r['n'], r['k']) for r in data['rows'] if r['q'] == 8} == {
        (8,512,2048), (8,2048,512), (8,4096,2048), (8,8192,2048), (8,2048,4096), (8,1024,5120)}
    for field, value in [('confirmation_us', [0, 0]), ('dequant', 24), ('storage', 2)]:
        bad = copy.deepcopy(data); bad['rows'][0][field] = value
        with pytest.raises(ValueError): render(bad)
    bad = copy.deepcopy(data); bad['rows'][1] = bad['rows'][0]
    with pytest.raises(ValueError): render(bad)


def test_all_exact_cpp_lookups_and_misses(tmp_path):
    # Compile the production header, not a Python reimplementation of lookup.
    source = tmp_path/'lookup.cpp'
    source.write_text('''#include "quactlize/prefill/dense_helpers.hpp"
#include <cstdio>
int main() { int q,m,n,k,s;
    while(std::scanf("%d%d%d%d%d",&q,&m,&n,&k,&s)==5) {
        auto a=quactlize::prefill::dense_full_helpers(q,m,n,k,s);
        auto b=quactlize::prefill::dense_helpers(q,m,n,k,s);
        std::printf("%d %d %d %d\\n",a.dequant,a.io,b.dequant,b.io);
    }
}''')
    exe=tmp_path/'lookup'
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(ROOT),str(source),'-o',str(exe)],check=True)
    rows=json.loads(EVIDENCE.read_text())['rows']
    inputs=[tuple(r[x] for x in ('q','m','n','k','storage')) for r in rows]
    misses=[(8,127,512,2048,0),(8,128,513,2048,0),(8,128,512,2048,2),
            (8,128,512,2304,0),(9,128,512,2048,0)]
    output=subprocess.check_output([str(exe)],input=''.join(' '.join(map(str,r))+'\n' for r in inputs+misses),text=True)
    parsed=[tuple(map(int,line.split())) for line in output.splitlines()]
    assert len(parsed)==len(inputs)+len(misses)
    old={(12,25600,5120):(21,3,3),(12,8192,5120):(21,3,3),(12,5120,8192):(20,1,1),
         (12,5120,25600):(8,1,1),(12,1024,5120):(20,2,2),
         (14,5120,25600):(21,1,2),(14,1024,5120):(20,2,2)}
    for r,(dq,io,old_dq,old_io) in zip(rows,parsed):
        assert (dq,io)==(r['dequant'],r['io'])
        expected=old.get((r['q'],r['n'],r['k'])) if r['m']==2048 else None
        assert (old_dq,old_io)==((expected[0],expected[1+r['storage']]) if expected else (-1,-1))
    assert all(r[:2]==(-1,-1) for r in parsed[len(rows):])
