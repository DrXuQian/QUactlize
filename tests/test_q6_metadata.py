from pathlib import Path
import subprocess

import pytest

from dev.gemv_simt.q6_metadata import ROOT, candidate_body, source


def test_same_body_one_semantic_seam():
    text = (ROOT/'quactlize/execution/simt_kernel.cuh').read_text()
    body = candidate_body(text)
    assert body.count('q6_metadata::affine(') == 2
    for compute in ('f16', 'bf16'):
        generated = source(compute)
        assert '<<<blocks,256,0,stream>>>(call,1)' in generated
        assert 'simt::register_reuse<14,1,3,4,8,8,' in generated
    with pytest.raises(ValueError, match='seam changed'):
        candidate_body(text.replace('affine_selected<Q,Changes>(', 'changed_affine(', 1))
    with pytest.raises(ValueError):
        source('int8')


def test_actual_extractor_exhaustive_fields(tmp_path):
    exe = tmp_path/'fields'
    subprocess.run(['g++', '-std=c++17', '-O2', f'-I{ROOT}',
                    str(ROOT/'tests/q6_metadata_host.cpp'), '-o', str(exe)], check=True)
    result = subprocess.check_output([str(exe)], text=True)
    assert 'Q6_METADATA_HOST PASS' in result
    assert 'negative=DETECTED' in result


def test_existing_both_plane_address_model():
    from dev.gemv_simt.access import pattern
    from quactlize.execution.simt_codegen import Config
    c = Config(3, 4, 8, 8, 1)
    for n, k in ((512, 2048), (1024, 5120)):
        for warp in range(8):
            for iteration in (0, 1):
                p = pattern(14, c, n, k, warp=warp, iteration=iteration)
                assert p['low_word_group_reuse'] == 2
                assert p['high_word_group_reuse'] == 4
                assert len([x for x in p['streams'] if x['name'].startswith('metadata')]) == 9
                assert all(set(x['footprint']) == {'32', '64', '128'} for x in p['streams'])


def test_bounded_inventory_and_raw_equality():
    import numpy as np
    from tools.run_q6_metadata import workloads, raw_equal
    rows = workloads()
    assert len(rows) == 8
    assert (rows[0].m, rows[0].n, rows[0].k, rows[0].experts) == (8,512,2048,256)
    assert {w.m for w in rows} == {1,2,8}
    assert {w.compute for w in rows} == {'f16','bf16'}
    assert {w.operation for w in rows} == {'dense','indexed'}
    assert all(w.qtype==14 for w in rows)
    a=np.array([0.,1.,-2.],dtype='<f4')
    raw_equal(a,a.copy())
    b=a.copy();b.view('<u4')[1]^=1
    with pytest.raises(ValueError,match='candidate differs'):raw_equal(a,b)
    b=a.copy();b[0]=-0.
    with pytest.raises(ValueError,match='candidate differs'):raw_equal(a,b)


def test_fixture_varies_group_scales_and_is_repeatable():
    import numpy as np
    from dev.gemv_simt.q6_metadata_fixture import Weights
    a,b=Weights(256,512,2),Weights(256,512,2)
    assert a.record==b.record
    assert np.array_equal(a.sums,b.sums)
    assert a.planes['units'].shape[-1]==36
    unit=a.planes['units'].reshape(-1,36)
    fields=unit[:,[2,3,4,20,21,22]]
    assert np.all(fields==np.array([128,0,127,128,0,127]))
    assert len(np.unique(unit[:,5:18]))>100
    assert np.any(a.sums>0) and np.any(a.sums<0)
    for compute in ('f16','bf16'):
        data=a.inputs(dict(tokens=2,channels=1,topk=1,mode=2,router='real',compute=compute))
        assert np.isfinite(data['gold']).all() and np.any(data['gold'])


def test_script_shell_and_no_llama_dependency():
    path=ROOT/'tools/run_q6_metadata_box.sh'
    subprocess.run(['bash','-n',str(path)],check=True)
    text=path.read_text()
    assert 'git submodule update' not in text
    assert 'LLAMA_CI' not in text and 'PREVIOUS_RUN' not in text
    assert 'PROFILE_ACU' in text
