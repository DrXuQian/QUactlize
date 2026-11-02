"""Public projection/selection ABI; device accuracy is in ppu/test_moe_library.py."""
import copy
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('source',['moe_prefill_policy_host','moe_route_host','moe_scaffold_host','moe_matmul_host','ffn_host'])
def test_host_contract(tmp_path,source):
    extra=[]
    if source=='moe_scaffold_host':extra=[str(ROOT/'quactlize/moe/selection.cpp')]
    if source in ('moe_matmul_host','ffn_host'):
        (tmp_path/'catalog.inc').write_text('static std::vector<Image> kImages{};\nstatic char const kJitSource[]="";\n')
        extra=[str(ROOT/p) for p in ('quactlize/packing/sizes.cpp','quactlize/execution/simt.cpp',
               'quactlize/execution/q4_decode.cpp','quactlize/moe/selection.cpp')]
    sdk=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK'))
    exe=tmp_path/source
    subprocess.run(['g++','-std=c++17','-O1','-pthread','-rdynamic','-I'+str(ROOT),'-I'+str(tmp_path),
        '-I'+str(sdk/'include'),'-I'+str(ROOT/'quactlize/include'),'-I'+str(ROOT/'third_party/actlize/include'),
        str(ROOT/'tests'/f'{source}.cpp'),*extra,'-ldl','-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


def test_measured_table_generated_and_fresh_evidence():
    from quactlize.dispatch.moe_prefill_policy import load,header,HEADER,validate
    from quactlize.runtime.tuning import digest
    data=load()
    assert HEADER.read_text()==header(data)
    fresh=[r for r in data['rows'] if r.get('evidence',{}).get('baseline')=='LLAMA_NATIVE']
    assert len(fresh)==2 and all(r['key'][0]==13 and r['key'][-1]==5120 for r in fresh)
    assert all(len(p['selected_samples_us'])==12 for r in fresh for p in r['profiles'])
    bad=copy.deepcopy(data);bad['rows'][-1]['profiles'][0]['selected_samples_us'][0][0]*=2
    bad['sha256']=digest({k:v for k,v in bad.items() if k!='sha256'})
    with pytest.raises(ValueError,match='evidence differs'):validate(bad)


def test_concat_prefill_evidence():
    from quactlize.dispatch.moe_prefill_policy import CONCAT_FAMILIES,MS,load,validate
    from quactlize.runtime.tuning import digest
    data=load()
    rows=[r for r in data['rows'] if tuple(r['key'][:4]) in CONCAT_FAMILIES]
    assert [r['key'] for r in rows]==[[*family,m] for family in CONCAT_FAMILIES for m in MS]
    assert all([p['router'] for p in r['profiles']]==['cluster','spread'] for r in rows)
    assert sum(r['selected']=='original' for r in rows)==2
    assert all(r['recipe']['route']==('fq' if r['key'][-1]<=512 else 'sf') for r in rows)
    for fault in ('cohort','source','router','regression','component'):
        bad=copy.deepcopy(data);row=bad['rows'][-1];p=row['profiles'][0]
        if fault=='cohort':row.pop('evidence')
        elif fault=='source':p['latency_provenance']['quactlize_commit']='0'*40
        elif fault=='router':row['profiles'].pop()
        elif fault=='regression':p['original_us']=p['selected_us']*.5
        else:p['components'][0]['mean_us']*=2
        bad['sha256']=digest({k:v for k,v in bad.items() if k!='sha256'})
        with pytest.raises(ValueError):validate(bad)


def test_ffn_ctypes_layout():
    import ctypes as C
    from ci.support.ffn import Weight,Params,Info,Value,Transition
    assert C.sizeof(Weight)==112 and C.sizeof(Params)==408 and C.sizeof(Info)==656
    assert Params.gate.offset==72 and Info.projections.offset==32
    assert C.sizeof(Value)==48 and Value.round_type.offset==40
    assert C.sizeof(Transition)==168 and Transition.gate.offset==24 and Transition.stream.offset==160


@pytest.mark.parametrize('kind,m,route,compute,policy,expected', [
    (0, 1, 2, 0, 0, None), (0, 8, 0, 0, 18, None),
    (0, 1, 0, 0, 17, 'f16'), (0, 9, 0, 0, 17, 'f16'),
    (0, 128, 1, 0, 20, 'f16'), (0, 128, 0, 1, 0, None),
    (0, 128, 3, 0, 20, 'f16'), (0, 128, 3, 1, 0, 'bf16'),
    (1, 1, 2, 1, 0, None), (1, 4, 0, 1, 0, 'bf16'),
    (1, 512, 1, 1, 19, 'bf16'),
])
def test_ffn_projection_output_precision(kind, m, route, compute, policy, expected):
    from types import SimpleNamespace
    from ci.support.ffn import projection_output_compute
    info = SimpleNamespace(route=route, compute_type=compute, policy=policy)
    assert projection_output_compute(kind, m, info) == expected


def test_ffn_q2_subnormal_down_oracle():
    """Exact CPU replay of the reported M9/lane0 independent Down check."""
    import numpy as np
    from ci.support.fixture import compare, matmul_reference, raw_weight, round_compute
    q, m, k, hidden, n = 10, 9, 512, 512, 256
    weights = [raw_weight(q, n if i == 2 else hidden, k, 913 + q + i*97)[1] for i in range(3)]
    source = np.random.default_rng(301 + q + m).uniform(-.05, .05, (m, k)).astype('f4')
    gate, _ = matmul_reference(source, weights[0], 'f16', output_compute='f16')
    up, _ = matmul_reference(source, weights[1], 'f16', output_compute='f16')
    gate, up = gate.astype('f4'), up.astype('f4')
    activation = (gate/(1 + np.exp(-gate))*up).astype('f4')
    ideal, denom = matmul_reference(activation, weights[2], 'f16')
    got = round_compute(ideal, 'f16')
    np.testing.assert_allclose(ideal[0, :3], [1.4479155652225018e-7, 2.5954796001315117e-7,
                                           3.7488644011318684e-7], rtol=1e-14, atol=0)
    np.testing.assert_array_equal(got[0, :3], np.ldexp([2., 4., 6.], -24))
    np.testing.assert_allclose(denom[0, :3], [8.922943379729986e-7, 1.6829580999910831e-6,
                                           1.341104507446289e-6], rtol=1e-14, atol=0)
    # Original checker falsely rejects legitimate F16 C rounding. Keep this
    # negative: an F32-output path must still reject this precision loss.
    with pytest.raises(ValueError, match='typed oracle mismatch'):
        compare(got, ideal, denom)
    golden, rounded_denom = matmul_reference(activation, weights[2], 'f16', output_compute='f16')
    np.testing.assert_array_equal(denom, rounded_denom)
    assert compare(got, golden, denom)['relative_l1_error'] == 0
    for fault in ('zero', 'row', 'ulp', 'nan'):
        bad = got.copy()
        if fault == 'zero': bad.fill(0)
        elif fault == 'row': bad = np.roll(bad, 1, axis=0)
        elif fault == 'ulp': bad[0, 0] += np.ldexp(1., -24)
        else: bad[0, 0] = np.nan
        with pytest.raises(ValueError, match='typed oracle mismatch'):
            compare(bad, golden, denom)


def test_ffn_output_precision_rejects_unknown_contract():
    from types import SimpleNamespace
    from ci.support.ffn import projection_output_compute
    from ci.support.fixture import matmul_reference
    for kind, route, compute in ((1, 3, 1), (1, 0, 0), (0, 4, 0), (0, 0, 2)):
        with pytest.raises(ValueError, match='projection|provider|compute'):
            projection_output_compute(kind, 9, SimpleNamespace(route=route, compute_type=compute, policy=0))
    with pytest.raises(ValueError, match='precision contract'):
        matmul_reference([[1.]], [[1.]], 'f16', output_compute='unknown')


def test_ffn_q2_bf16_midpoint_oracle():
    """M513/repeat1/lane1/expert0: a legal F32 order crosses a BF16 midpoint.

    Replays local expert row 202 / column 116, original token 406, from the
    reported fixture. This is a checker counterexample, not a reproduction of
    the PPU's still-unreported failing coordinate or exact accumulation order.
    """
    import numpy as np
    from ci.support.fixture import compare, matmul_reference, raw_weight, round_compute
    source = np.random.default_rng(832).uniform(-.05, .05, (513, 512)).astype('f4')[406]
    x = round_compute(source, 'bf16').astype('f8')
    gate_up = []
    for stage in (0, 1):
        weight = raw_weight(10, 512, 512, 923 + stage*97)[1].astype('f8')
        gate_up.append(round_compute(x @ weight.T, 'bf16'))
    gate, up = gate_up
    activation = (gate/(1 + np.exp(-gate))*up).astype('f4')
    weight = raw_weight(10, 256, 512, 1117)[1][116:117]
    x = round_compute(activation, 'bf16').astype('f8')
    terms = x * weight[0].astype('f8')
    # Products are exact F32 here; only the legal accumulation order varies.
    np.testing.assert_array_equal(terms, terms.astype('f4'))
    acc = np.float32(0)
    for i in np.argsort(abs(terms))[::-1]:
        acc = np.float32(np.float64(acc) + terms[i])
    got = round_compute(np.array([[acc]]), 'bf16')
    golden, denom = matmul_reference(activation[None, :], weight, 'bf16', output_compute='bf16')
    np.testing.assert_allclose(golden[0, 0], 2.1830205292339144e-6, rtol=1e-14)
    np.testing.assert_allclose(denom[0, 0], 2.9500172167296057e-6, rtol=1e-14)
    assert got[0, 0] == 2.175569534301758e-6
    rounded = round_compute(golden, 'bf16')
    assert rounded[0, 0] == 2.1904706954956055e-6
    with pytest.raises(ValueError, match='typed oracle mismatch'):
        compare(got, rounded, denom)
    result = compare(got, golden, denom)
    assert result['tolerance'] == .005 and result['relative_l1_error'] < .002526
    assert abs(float(acc) - golden[0, 0]) / denom[0, 0] < 1.5e-7
    # Rounding is not permission to accept a zero, nonfinite or farther value.
    for bad in (np.zeros_like(got), got - np.float32(2**-26), np.full_like(got, np.nan)):
        with pytest.raises(ValueError, match='typed oracle mismatch'):
            compare(bad, golden, denom)


def test_ffn_q3_activation_midpoint_oracle():
    """Uploaded Q3 merged dense M128/repeat2/lane1, output (120,251).

    A reciprocal formulation is a constructive counterexample, not proof of
    the GPU's exp/divide instructions. It reproduces the exact output while
    changing one half-input cell and remaining accurate to F32 at SwiGLU.
    """
    import numpy as np
    from ci.support.ffn import check_swiglu
    from ci.support.fixture import compare, raw_weight, round_compute
    source = np.random.default_rng(455).uniform(-.05, .05, (128, 512)).astype('f4')[120:121]
    x = round_compute(source, 'f16').astype('f8')
    gate, up = [round_compute(x @ raw_weight(11, 512, 512, seed)[1].astype('f8').T, 'f16')
                for seed in (924, 730)]
    cpu = gate / (1 + np.exp(-gate)) * up
    reciprocal = gate * np.reciprocal(1 + np.exp(-gate)) * up
    assert cpu[0, 286].view('u4') == 0xb6defffe
    assert reciprocal[0, 286].view('u4') == 0xb6df0000
    for value in (cpu, reciprocal):
        assert check_swiglu(value, gate, up)['relative_l1_error'] < 3e-7
    typed_cpu, typed_reciprocal = [round_compute(h, 'f16').astype('f8') for h in (cpu, reciprocal)]
    assert np.flatnonzero(typed_cpu != typed_reciprocal).tolist() == [286]
    weight = raw_weight(11, 256, 512, 1118)[1][251:252].astype('f8')
    gold = round_compute(typed_cpu @ weight.T, 'f16')
    got = round_compute(typed_reciprocal @ weight.T, 'f16')
    denom = abs(typed_cpu) @ abs(weight).T
    assert got[0, 0] == 0 and gold[0, 0] == -2**-24
    np.testing.assert_allclose(denom[0, 0], 5.254842108115554e-6, rtol=1e-14)
    # Keep the final .005 gate unchanged: substituting CPU's activation
    # creates this false rejection, even though both F32 activations pass.
    with pytest.raises(ValueError, match='typed oracle mismatch'):
        compare(got, gold, denom)
    for fault in ('zero', 'column', 'scale', 'nan'):
        bad = reciprocal.copy()
        if fault == 'zero': bad.fill(0)
        elif fault == 'column': bad = np.roll(bad, 1, axis=-1)
        elif fault == 'scale': bad *= 1.001
        else: bad[0, 286] = np.nan
        with pytest.raises(ValueError, match='typed oracle mismatch'):
            check_swiglu(bad, gate, up)
