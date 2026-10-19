import json
from pathlib import Path
import subprocess
import sys

import pytest

from dev.gemv_simt.q6_sparse import ROOT, ARMS, sparse_body, source
from dev.gemv_simt.q6_metadata import source as previous_source


def test_actual_packet_fields_and_cross_lane_ownership(tmp_path):
    exe=tmp_path/'packet'
    subprocess.run(['g++','-std=c++17','-O2',f'-I{ROOT}',str(ROOT/'tests/q6_sparse_host.cpp'),
                    '-o',str(exe)],check=True)
    output=subprocess.check_output([str(exe)],text=True)
    assert 'fields=2064384 exchanges=10240 negative=DETECTED' in output


def test_control_unchanged_and_candidate_only_metadata_seams():
    for compute in ('f16','bf16'):
        a1=previous_source(compute).split('extern "C"')[0]
        assert source(compute).startswith(a1)
        assert 'if(arm==0) simt::q6_register_fields<<<blocks,256,0,stream>>>(call,1);' in source(compute)
        assert 'else simt::q6_sparse_fields<<<blocks,256,0,stream>>>(call,1);' in source(compute)
    from dev.gemv_simt.q6_metadata import candidate_body
    text=(ROOT/'quactlize/execution/simt_kernel.cuh').read_text()
    body=sparse_body(text)
    body=body.replace('q6_sparse_fields_body(', 'q6_register_fields_body(').replace(
        '    static_assert(Q==14 && Variant==3 && Columns==4 && Warps==8 && P==8);\n','')
    body=body.replace('q6_sparse::Packet cooperative{};', 'Meta<F::words> cooperative{};').replace(
        'q6_sparse::load(units+F::unit_offset(call.n,tile*TileN+lane,g),unsigned(g))',
        'load_meta<Q>(units,call.n,tile*TileN+lane,g)').replace(
        'q6_sparse::affine(q6_sparse::share(cooperative,(lane%Columns)*P+p,unsigned(g)),unsigned(g))',
        'q6_metadata::affine(share_meta(cooperative,(lane%Columns)*P+p),g)')
    assert body==candidate_body(text)
    with pytest.raises(ValueError):
        sparse_body(text.replace('Meta<F::words> cooperative{};', 'wrong_metadata_type;'))


def test_all_phases_transactions_and_active_warp_contract():
    from dev.gemv_simt.q6_sparse_access import pattern, WORDS
    from quactlize.execution.simt_codegen import Config
    config=Config(3,4,8,8,1)
    phases=set()
    for n,k in ((32,256),(512,512),(512,2048),(1024,5120)):
        for warp in range(8):
            for iteration in (0,1,4):
                pair=pattern(14,config,n,k,warp=warp,iteration=iteration,
                             bases=dict(A=0,low=0,high=0,units=36))
                old,new=(pair[a] for a in ARMS)
                assert [s for s in old['streams'] if not s['name'].startswith('metadata')]==[
                    s for s in new['streams'] if not s['name'].startswith('metadata')]
                metadata=[s for s in new['streams'] if s['name'].startswith('metadata')]
                if any(s['lanes'] for s in metadata):
                    phase=old['lane_groups'][0]//8%4;phases.add(phase)
                    assert len(metadata)==len(WORDS[phase])
                for s in metadata:
                    assert s in old['streams']
                    assert set(s['footprint'])=={'32','64','128'}
                for size in ('32','64','128'):
                    assert sum(s['footprint'][size]['sectors'] for s in metadata)<=sum(
                        s['footprint'][size]['sectors'] for s in old['streams'] if s['name'].startswith('metadata'))
    assert phases=={0,1,2,3}
    with pytest.raises(ValueError,match='requires Q6'):
        pattern(14,Config(3,8,8,8,1),512,2048)


@pytest.mark.parametrize('points,count',[('0',1),('all',8)])
def test_plan_is_bounded_and_labels_control_correctly(tmp_path,points,count):
    output=tmp_path/'plan'
    subprocess.run([sys.executable,str(ROOT/'tools/run_q6_sparse.py'),'--sdk',str(tmp_path),
                    '--output',str(output),'--points',points,'--plan-only'],check=True)
    plan=json.loads((output/'plan.json').read_text())
    assert plan['arms']==list(ARMS) and len(plan['points'])==count
    assert plan['configs']==1 and plan['production_changed'] is False
    assert not (output/'build').exists()


def test_box_no_external_checkout_or_llama():
    script=ROOT/'tools/run_q6_sparse_box.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    text=script.read_text()
    assert 'git submodule update' not in text and 'LLAMA_CI' not in text
    assert '${POINTS:-0}' in text and '${PROFILE_ACU:-1}' in text
    runner=(ROOT/'tools/run_q6_sparse.py').read_text()
    assert 'profile_arms=(1,)' in runner
    assert 'source_inputs=inputs,pattern_model=pattern' in runner


def test_profile_dispatches_frozen_sparse_manifest():
    text=(ROOT/'tools/profile_q6_metadata.py').read_text()
    assert "manifest.get('schema')=='q6-sparse-v1'" in text
    assert "import_module('dev.gemv_simt.build_q6_sparse').inputs" in text
