"""The common-reader port keeps measured arithmetic and existing winners."""
import re
import subprocess
from pathlib import Path

from quactlize.execution.simt_codegen import source
from dev.gemv_simt.split_source import body
from dev.gemv_simt.general_reader import codes_body

ROOT=Path(__file__).resolve().parents[1]


def rename(text):
    for old,new in (('GeneralFormat','CommonFormat'),('general_locate','common_locate'),
                    ('general_metadata','common_metadata'),('general_codes','common_codes'),
                    ('split_reuse_body','common_reuse_body')):
        text=text.replace(old,new)
    return text


def test_exact_measured_producer_body_and_codes():
    text=(ROOT/'quactlize/execution/simt_common.cuh').read_text()
    assert rename(body()) in text
    assert rename(codes_body()) in text
    assert 'dev/' not in text
    assert 'launch_reduction<Q>(c,split,stream);' in text
    assert 'dim3 grid(c.n/(C*P),c.rows,split);' in text


def test_codegen_common_guard_precedes_measured_but_does_not_change_q8():
    for q in range(10,15):
        text=source(q)
        assert text.index('common_reader_scope')<text.index('measured_decode(*c')
        assert len(re.findall(r'return quactlize::execution::simt::common_reader_launch<',text))==(4 if q==12 else 3)
    assert 'simt_common' not in source(8) and 'common_reader' not in source(8)


def test_host_scope_and_explain(tmp_path):
    code=r'''
#include "quactlize/execution/simt_strategy.hpp"
#include <cassert>
#include <cstring>
using namespace quactlize::execution::simt;
int main() {
    for(int q:{8,10,11,12,13,14})for(int compute:{0,1})
    for(int input:{0,1,2})for(int split:{1,2,4,8})for(int mode:{0,1,2}) {
        qkg_simt_call_v2 d{};d.compute_type=compute;
        auto& c=d.call;c.qtype=q;c.input_type=input;c.mode=mode;
        c.n=4096;c.k=2048;c.rows=8;c.experts=mode?256:1;c.topk=mode?8:1;c.channels=1;
        qkg_simt_config_v1 f{1,sizeof(f),3,4,8,8,split};
        bool expected=q!=8 && input==QKG_F32;
        assert(common_reader_scope(d,f)==expected);
        assert((std::strcmp(implementation(d,f).producer,"common-packet-3d")==0)==expected);
        f.variant=1;assert(!common_reader_scope(d,f));
        f.variant=3;f.values=2;assert(!common_reader_scope(d,f));
        f.values=8;c.rows=65536;assert(!common_reader_scope(d,f));
    }
    qkg_simt_call_v2 d{};d.compute_type=1;auto& c=d.call;
    c.input_type=QKG_F32;c.mode=QKG_INDEXED;c.rows=8;c.experts=256;c.topk=8;
    c.qtype=13;c.n=2048;c.k=512;c.channels=8;
    qkg_simt_config_v1 f{1,sizeof(f),3,4,2,8,1};
    assert(!common_reader_scope(d,f));
    f.split=2;assert(common_reader_scope(d,f));
    f.split=1;c.n=3072;assert(common_reader_scope(d,f));
    c.qtype=12;c.n=1024;c.k=3072;c.channels=1;f.warps=4;f.values=4;
    assert(!common_reader_scope(d,f));
    f.split=4;assert(common_reader_scope(d,f));
}
'''
    unit=tmp_path/'scope.cpp';unit.write_text('#include <initializer_list>\n'+code)
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT),str(unit),'-o',str(tmp_path/'scope')],check=True)
    subprocess.run([str(tmp_path/'scope')],check=True)
