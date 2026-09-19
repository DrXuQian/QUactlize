import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from tools import run_kpack_tp2_chain as chain


def raw_block(q):
    raw = np.zeros((1, 144 if q == 12 else 176), dtype=np.uint8)
    raw[:, :4] = np.array([1, 1], dtype='<f2').view('u1')
    raw[:, 4:8] = 1
    raw[:, 8:12] = 2
    raw[:, 12:16] = 0x21
    raw[:, 16 if q == 12 else 48:] = 0xa3
    return raw


def test_bf16_rounding():
    source = np.array([0x3f808000, 0x3f818000, 0xbf808000, 0x80000000], dtype='<u4').view('<f4')
    assert np.array_equal(chain.bf16(source).view('<u4'), [0x3f800000, 0x3f820000, 0xbf800000, 0x80000000])
    with pytest.raises(ValueError, match='nonfinite'):
        chain.bf16([float('nan')])


def test_q4_independent_fields_and_codes():
    decoded, native = chain.decode(raw_block(12), 12)
    expected = np.tile(np.repeat(np.array([1, 8], dtype='<f4'), 32), 4)[None]
    assert np.array_equal(decoded, expected)
    assert np.array_equal(native, expected)
    changed = raw_block(12)
    changed[0, 14] &= 0xf0
    negative, _ = chain.decode(changed, 12)
    assert np.array_equal(negative[0, :192], decoded[0, :192])
    assert np.count_nonzero(negative != decoded) == 32


def test_q5_high_plane():
    raw = raw_block(13)
    raw[:, 16:48] = 0b10100101
    decoded, native = chain.decode(raw, 13)
    expected = np.tile(np.repeat(np.array([1, 8], dtype='<f4'), 32), 4)
    expected += np.repeat(np.array([16, 0, 16, 0, 0, 16, 0, 16]), 32)
    assert np.array_equal(decoded[0], expected)
    assert np.array_equal(native[0], expected)


@pytest.mark.parametrize('q', [12, 13])
def test_independent_decoder_matches_ggml(q):
    path = os.environ.get('TP2_GGML_ORACLE_LIBRARY')
    if not path:
        pytest.skip('optional independent GGML host library')
    lib = ctypes.CDLL(path)
    function = getattr(lib, 'dequantize_row_q%d_K' % (q-8))
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    raw = np.random.default_rng(91+q).integers(0, 256, (67, 144 if q == 12 else 176), dtype=np.uint8)
    raw[:, :4] = np.array([.015625, .0078125], dtype='<f2').view('u1')
    expected = np.empty((67, 256), dtype='<f4')
    function(raw.ctypes.data, expected.ctypes.data, expected.size)
    actual, _ = chain.decode(raw, q)
    assert np.array_equal(actual, expected)


def test_capture_rejects_bad_size(tmp_path):
    data = tmp_path/'value.f32'
    data.write_bytes(b'')
    with pytest.raises(ValueError, match='size differs'):
        chain.read_array(data, '<f4', (2, 3))


def test_full_capture_parser_and_changed_graph_negative(tmp_path):
    path = os.environ.get('TP2_GGML_ORACLE_LIBRARY')
    if not path:
        pytest.skip('optional independent GGML fixture producer')
    lib = ctypes.CDLL(path)
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    roots = [tmp_path/'ordinary', tmp_path/'retained']
    for root in roots:
        root.mkdir()
    weights = []
    for number, (q, n, size) in enumerate(((12, 2048, 144), (13, 512, 176))):
        index = np.arange(4*n*1024)
        source = (((index*19+index//23)%71-35)/256).astype('<f4')
        raw = np.empty((4, n, 4, size), 'u1')
        importance = np.ones(1024, '<f4')
        assert lib.ggml_quantize_chunk(q, source.ctypes.data, raw.ctypes.data,
            0, n*4, 1024, importance.ctypes.data) == raw.nbytes
        for root in roots:
            raw.tofile(root/f'weight-{number}.bin')
        weights.append(chain.decode(raw, q)[1].reshape(4, n, 1024))
    for replay in range(3):
        suffix = f'-i{replay}'
        index = np.arange(32*1024).reshape(32, 1024)
        a = (((index*13+index//31+replay*11)%61-30)/128).astype('<f4')
        ids = ((np.arange(64)+replay)%4).astype('<i4')
        pair = chain.bf16(chain.indexed_dot(np.repeat(a, 2, axis=0), weights[0], ids))
        middle = chain.swiglu(pair)
        down = [chain.bf16(chain.indexed_dot(chain.bf16(middle[:, r*512:(r+1)*512]),
                weights[1][..., r*512:(r+1)*512], ids)) for r in range(2)]
        total = down[0]+down[1]
        for root in roots:
            a.tofile(root/f'input{suffix}.f32')
            ids.tofile(root/f'ids{suffix}.i32')
            (total*total).tofile(root/f'result{suffix}.f32')
        for rank in range(2):
            local_pair = np.concatenate((pair[:, rank*512:(rank+1)*512],
                pair[:, 1024+rank*512:1024+(rank+1)*512]), axis=1)
            local_pair.tofile(roots[1]/f'pair-r{rank}{suffix}.f32')
            middle[:, rank*512:(rank+1)*512].tofile(roots[1]/f'activation-r{rank}{suffix}.f32')
            total.tofile(roots[1]/f'reduced-r{rank}{suffix}.f32')
    result = chain.analyze(tmp_path)
    assert result['admission'] == 'PENDING' and len(result['replays']) == 3
    assert result['retained_outputs_match_original_bits'] is True
    assert result['replays'][0]['original_gate']['relative_l2'] > .02
    for row in result['replays']:
        assert row['actual_vs_simulated_bf16']['max_abs'] == 0
        assert all(field['max_abs'] == 0 for field in row['stage_local_errors'].values())
    changed = np.fromfile(roots[1]/'result-i0.f32', dtype='<f4')
    changed[0] += 1
    changed.tofile(roots[1]/'result-i0.f32')
    with pytest.raises(ValueError, match='capture changed'):
        chain.analyze(tmp_path)


def test_diagnostic_cannot_admit_model():
    root = Path(__file__).resolve().parents[1]
    source = (root/'tools/run_kpack_tp2_chain.py').read_text()
    assert "admission='PENDING'" in source
    assert "threshold_unchanged=.02" in source
    assert 'capture changed the original graph result or input' in source
    script = (root/'tools/run_kpack_tp2_box.sh').read_text()
    assert 'TP2_MODE == chain' in script and 'run_kpack_tp2_chain.py' in script
    assert "KPACK_TP2_CACHE PASS mode=cold cases=72 chains=6" in script
