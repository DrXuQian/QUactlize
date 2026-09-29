"""Bounded caller matching and additive fusion ABI checks; not device admission."""
import os
from pathlib import Path
import subprocess
import sys
import json

import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_graph_selector_and_row_aliases(tmp_path):
    llama=Path(os.environ['LLAMA_CI_DIR'])
    libraries=Path(os.environ['LLAMA_HOST_LIB_DIR'])
    binary=tmp_path/'gate-up'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT),
                    '-I'+str(llama/'ggml/include'),'-I'+str(llama/'ggml/src'),
                    str(ROOT/'tests/gate_up_integration_host.cpp'),'-L'+str(libraries),
                    '-Wl,-rpath,'+str(libraries),'-lggml-base','-o',str(binary)],check=True)
    subprocess.run([binary],check=True)


def test_shared_mirror_and_no_per_token_allocation():
    llama=Path(os.environ['LLAMA_CI_DIR'])
    assert (ROOT/'quactlize/fusion/llama_graph.hpp').read_bytes()==(
        llama/'ggml/src/ggml-cuda/quactlize/gate_up_graph.hpp').read_bytes()
    source=(llama/'ggml/src/ggml-cuda/quactlize-execution.cu').read_text()
    body=source.split('bool ggml_quactlize_execution_shared_run(',1)[1].split('\nint ggml_',1)[0]
    assert 'prepare_shared(ctx,graph,start,false)' in body
    for forbidden in ('Malloc','Synchronize','paired_repack','private_storage'):
        assert forbidden not in body


def trace_module():
    llama=Path(os.environ['LLAMA_CI_DIR'])
    sys.path.insert(0,str(llama/'tests'))
    import quactlize_native
    return quactlize_native


def test_asys_fusion_is_not_a_selection_only_claim():
    mod=trace_module()
    for q,op,e,compute in ((8,'dense',1,'FP16'),(12,'grouped',256,'BF16')):
        for tokens in range(1,9):
            backend='simt' if q==8 or tokens<=2 or tokens==4 else 'tc'
            split=2 if q==12 and tokens==3 else 1
            tile=0 if backend=='simt' else 16 if tokens<=6 else 8
            warps=(4 if q==12 and tokens==2 else 8) if backend=='simt' else 0
            line=f'[quactlize-paired-plan] tensor=test op={op} q={q} tokens={tokens} n=512 k=2048 experts={e} backend={backend} split={split} tile_m={tile} warps={warps} activation={compute} layout=0x47554e3400000001'
            record=mod.paired_selection(line,{'paired_gate_up':{'layout_id':'0x47554e3400000001'}})[0]
            name=(f'void quactlize::fusion::simt_gate_up<{q}, 1, {int(compute=="BF16")}, {warps}>' if backend=='simt'
                  else f'void quactlize::fusion::tc_gate_up<quactlize::fusion::TcTypes<{q}, {tile}, cutlass::bfloat16_t, float> >')
            assert mod.paired_matches_plan(mod.paired_symbol_recipe(name),record)
            assert not mod.paired_symbol_recipe('void quactlize::runtime::moe_chain_prepare_m1<cutlass::half_t>')
            for bad in (line.replace('n=512','n=1024'),line.replace(f'activation={compute}','activation=BAD')):
                with pytest.raises(Exception):mod.paired_selection(bad,{'paired_gate_up':{'layout_id':'0x47554e3400000001'}})


def test_model_proof_requires_both_actual_fusions(tmp_path):
    from tools.check_kpack_paired_model import check
    path=tmp_path/'trace/model/native';path.mkdir(parents=True)
    proof=dict(paired_observed_ops=['dense','grouped'],paired_missing_ops=[],kernel_execution='PASS_SHORT_REQUEST',
               capture_scope='SECOND_REQUEST_SAME_PROCESS_FIRST_USE_EXCLUDED')
    (path/'proof.json').write_text(json.dumps(proof));check(tmp_path)
    for bad in (dict(paired_observed_ops=['dense']),dict(paired_missing_ops=['grouped']),
                dict(kernel_execution='PARTIAL_SHORT_REQUEST'),dict(capture_scope='FIRST_REQUEST')):
        (path/'proof.json').write_text(json.dumps(proof|bad))
        with pytest.raises(ValueError):check(tmp_path)


@pytest.mark.parametrize('ops,paired',[(['dense'],[]),(['dense','grouped'],['grouped']),
                                    (['dense','grouped'],['dense','grouped'])])
@pytest.mark.parametrize('fault',[None,'compute','missing','first-use','fusion'])
def test_extended_models_prove_their_selected_scope(tmp_path,ops,paired,fault):
    from tools.check_kpack_paired_model import check
    path=tmp_path/'trace/model/native';path.mkdir(parents=True)
    proof=dict(paired_observed_ops=paired,paired_missing_ops=[],kernel_execution='PASS_SHORT_REQUEST',
               capture_scope='SECOND_REQUEST_SAME_PROCESS_FIRST_USE_EXCLUDED',
               expected_ops=ops,observed_ops=ops,missing_ops=[],
               selection=dict(fully_selected=True,paired_plans=[dict(op=op) for op in paired]))
    if fault=='compute':proof['selection']['fully_selected']=False
    if fault=='missing':proof['observed_ops']=[]
    if fault=='first-use':proof['capture_scope']='FIRST_REQUEST'
    if fault=='fusion':proof['paired_missing_ops']=['dense']
    (path/'proof.json').write_text(json.dumps(proof))
    if fault:
        with pytest.raises(ValueError):check(tmp_path,selected_scope=True)
    else:
        check(tmp_path,selected_scope=True)
        result=json.loads((tmp_path/'paired-model-proof.json').read_text())
        assert result['traces'][0]['paired_status']==('DEVICE_OBSERVED' if paired else 'NOT_SELECTED')
