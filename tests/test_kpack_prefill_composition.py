import copy
import hashlib
import json
import math

import numpy as np
import pytest

from quactlize.dequant.native import selected_configs, traffic
from tools.compose_kpack_prefill_costs import Results, combine_cell, compose, write_outputs
from tools.kpack_bf16_fixture import row_domain
from tools.kpack_bf16_providers import DeepGemm
from tools.run_kpack_bf16_gate import cost_record, prefill_candidates


DEVICE = dict(ordinal=0, pci='0000:08:00.0', visible_devices='0',
              l2_bytes=64*1024**2, compute_units=72, warp=32)


def study(path, records, authority):
    path.mkdir()
    records = records | {'authority.json': authority}
    files = {}
    for name, value in records.items():
        data = json.dumps(value).encode()
        (path / name).write_bytes(data)
        files[name] = hashlib.sha256(data).hexdigest()
    (path / 'result.json').write_text(json.dumps(dict(
        status='INCOMPLETE', failed=[{'case': 'other/acu', 'rc': 6}], files=files)))
    return Results(path)


def dequant(w, generation, inventory, best_config, best_us):
    sizes = traffic(w['q'], w['n'], w['k'], w['experts'], w['operation'])
    copies = max(2, math.ceil(2.25*DEVICE['l2_bytes']/sizes['reads']))
    rows = []
    for config in selected_configs(w['q'], w['operation'], generation, inventory):
        us = best_us if config == best_config else best_us+10
        rows.append(dict(config=config, median_us=us, samples_us=[us]*15,
            round_medians_us=[us]*3, effective_gbps=sizes['useful_bytes']/us/1000,
            effective_pct=sizes['useful_bytes']/us/1000/2700*100,
            proof=dict(cells=w['n']*w['k']*w['experts'], bad=0,
                       signed_zero_differences=0, negative_bad=10, guard='PASS')))
    return dict(status='PASS', workload=w, device=DEVICE, gemm_calls=0,
        scope='DEQUANT_ONLY_NOT_COMBINED_OR_REUSABLE_CACHE', peak_gbps=2700,
        bytes=sizes, rows=rows, config_generation=generation, inventory=inventory,
        fixture_hashes=dict(low='a'*64, high='b'*64, units='c'*64), golden_sha256='d'*64,
        copies=copies, input_ring_bytes=sizes['reads']*copies,
        output_ring_bytes=sizes['writes']*copies, calls_per_graph=copies*2)


@pytest.fixture
def evidence(tmp_path):
    return make_evidence(tmp_path)


def make_evidence(tmp_path, experts=1, tokens=2048):
    w = dict(q=12, n=1024, k=5120, experts=experts, operation=1, smoke=False,
             id=f'q12-n1024-k5120-e{experts}-full')
    sf = w | dict(operation=0, id=w['id'].removesuffix('-full')+'-sf')
    authority = dict(device=DEVICE, peak_gbps=2700, manifest_sha256='1'*64,
                     workloads=[w, sf], runtime={'lib': '2'*64}, python_packages={'numpy': 'test'})
    old = study(tmp_path/'old', {w['id']+'.json': dequant(w, 2, 'all', 4, 5.),
        sf['id']+'.json': dequant(sf, 2, 'all', 4, 2.)}, authority)
    new = study(tmp_path/'new', {w['id']+'.json': dequant(w, 4, 'full-packed', 11, 3.)},
        authority | dict(manifest_sha256='3'*64, workloads=[w]))
    rows, _, routes_hash = row_domain(tokens, experts)
    copies = max(2, math.ceil(2.25*DEVICE['l2_bytes']/(2*w['n']*w['k']*experts)))
    active = int(np.count_nonzero(rows))
    provider_identity = (dict(provider='CUBLAS_PPU_SDK', entry='cublasGemmEx', a='BF16_M_K',
        b='BF16_N_K', output='BF16_M_N', compute='CUBLAS_COMPUTE_32F') if experts == 1 else
        dict(provider='DEEPGEMM_INSTALLED', implementation='PYTHON_JIT', entry_module=DeepGemm.MODULE,
            entry='m_grouped_gemm_bf16_bf16_bf16_nt_nopad', a='BF16_SORTED_ROWS_K', b='BF16_E_N_K',
            output='BF16_SORTED_ROWS_N', benchmark_stream='TORCH_CURRENT_NONBLOCKING_STREAM', entry_kind='function'))
    gemm = dict(status='PASS', workload=w, tokens=tokens, samples_us=[10.]*15, median_us=10., error=.001,
        scope='GEMM_PROVIDER_ONLY_DEQUANT_NEVER_INSIDE_TIMED_GRAPH', cost=cost_record(10., 5.),
        guards='PASS', zero_a='PASS', changed_a_graph='PASS', changed_a_sequence='PASS',
        timing=dict(method='CAPTURED_COMPLETE_RING_EVENTS', stream=7, graph_capture=True, host_enqueue_gaps='GRAPH_REPLAY'),
        rows=rows.tolist(), routes_sha256=routes_hash, total_rows=int(rows.sum()),
        expanded_experts=experts, active_experts=active, max_rows=int(rows.max()), topk=1 if experts==1 else 8,
        round_medians_us=[10.]*3,
        identity=dict(copies=copies, weight_ring_bytes=2*w['n']*w['k']*experts*copies,
            device=DEVICE, provider=provider_identity, golden_sha256='d'*64,
            dequant_result_sha256=old.files[w['id']+'.json'], dequant_config=4),
        active_weight_ring_bytes=2*w['n']*w['k']*active*copies, calls_per_graph=copies*2,
        cache='ROTATING_BF16_WEIGHT_COMPLETE_RING_TRAVERSALS', first_use_excluded_seconds=123.,
        production_changed=False, selection_admission='PENDING_MATCHED_FQ_SF_GEMM', sf_dequant_us=2.,
        candidates=prefill_candidates(sf_dequant=2., full_dequant=5., bf16_gemm=10.), loaded_images={})
    provider = study(tmp_path/'gemm', {w['id']+f'-m{tokens}.json': gemm},
        authority | dict(workloads=[w], ms=[tokens], dequant_authority_sha256=old.files['authority.json']))
    return old, new, provider, gemm


def test_composition_reuses_gemm_without_changing_raw_receipts(evidence, tmp_path):
    old, new, provider, gemm = evidence
    before = {p: p.read_bytes() for s in (old, new, provider) for p in s.folder.iterdir()}
    result = compose(old, new, [provider])
    assert result['status']=='COMPONENTS_COMPLETE' and result['complete']==1
    row = result['rows'][0]
    assert row['previous_cost']['sum_estimate_us']==15
    assert row['cost']['sum_estimate_us']==13  # Not 136: first use is not recharged.
    assert row['bf16_gemm']['us']==10 and row['full_dequant']['config']==11
    assert row['candidates']['fq']['status']=='UNMEASURED'
    assert row['candidates']['sf']['cost_us'] is None
    assert row['selection'] is None and not result['production_changed']
    assert row['break_even_us']['sf_gemm']==11
    write_outputs(tmp_path/'out', result)
    plan = json.loads((tmp_path/'out/missing-gemm-plan.json').read_text())
    assert plan['cells']==2 and plan['status']=='PLAN_ONLY_NOT_EXECUTABLE'
    assert all(p.read_bytes()==v for p, v in before.items())
    with pytest.raises(FileExistsError):write_outputs(tmp_path/'out', result)


@pytest.mark.parametrize('mutation', ['device', 'original_hash', 'output', 'config', 'cold', 'samples', 'sparse', 'prepass', 'provider_precision'])
def test_gemm_identity_and_cost_fail_closed(evidence, mutation):
    old, new, provider, original = evidence
    r = copy.deepcopy(original)
    if mutation=='device':r['identity']['device']['pci']='0000:09:00.0'
    if mutation=='original_hash':r['identity']['dequant_result_sha256']='0'*64
    if mutation=='output':r['identity']['golden_sha256']='0'*64
    if mutation=='config':r['identity']['dequant_config']=5
    if mutation=='cold':r['cache']='WARM'
    if mutation=='samples':r['samples_us'][0]=float('nan')
    if mutation=='sparse':r['active_experts']=0
    if mutation=='prepass':r['sf_dequant_us']=99
    if mutation=='provider_precision':r['identity']['provider']['a']='FP16_M_K'
    with pytest.raises(ValueError):combine_cell(old, new, provider, r)


def test_current_sparse_router_cannot_be_priced_as_fraction_of_e256(tmp_path):
    old, new, provider, r = make_evidence(tmp_path, experts=256, tokens=128)
    assert r['active_experts']==247
    with pytest.raises(ValueError, match='sparse expert domain'):
        combine_cell(old, new, provider, r)


def test_duplicate_provider_records_are_not_silently_selected(evidence):
    old, new, provider, _ = evidence
    with pytest.raises(ValueError, match='duplicate provider'):
        compose(old, new, [provider, provider])


def test_missing_record_cannot_become_zero_cost(evidence):
    old, new, provider, r = evidence
    provider.files.pop(r['workload']['id']+f'-m{r["tokens"]}.json')
    result = compose(old, new, [provider])
    assert result['status']=='INCOMPLETE' and result['complete']==0
    assert result['expected']==1 and len(result['missing'])==1


def test_provider_record_must_match_its_authority_key(evidence, monkeypatch):
    old, new, provider, original = evidence
    wrong = copy.deepcopy(original)
    wrong['tokens'] = 4096
    monkeypatch.setattr(provider, 'read', lambda name:wrong)
    with pytest.raises(ValueError, match='declared workload'):
        compose(old, new, [provider])


@pytest.mark.parametrize('mutation', ['bytes', 'golden', 'runtime', 'dtype_package'])
def test_replacement_does_not_cross_weight_or_environment_domains(evidence, mutation, monkeypatch):
    old, new, provider, r = evidence
    records = {name:new.read(name) for name in new.files}
    if mutation=='bytes':records[r['workload']['id']+'.json']['fixture_hashes']['high']='0'*64
    if mutation=='golden':records[r['workload']['id']+'.json']['golden_sha256']='0'*64
    if mutation=='runtime':new.authority['runtime']['lib']='0'*64
    if mutation=='dtype_package':new.authority['python_packages']['numpy']='other'
    monkeypatch.setattr(new, 'read', lambda name:records[name])
    with pytest.raises(ValueError):combine_cell(old, new, provider, r)


def test_result_checksums_are_not_merely_copied(evidence):
    _, new, _, r = evidence
    path = new.folder/(r['workload']['id']+'.json')
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match='checksum/path'):
        Results(new.folder)
