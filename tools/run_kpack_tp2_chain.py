#!/usr/bin/env python3
"""Capture the fixed TP2 Q4/Q5 chain without changing its numerical gate."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np


EXPERTS, K, HIDDEN, N, TOKENS, TOPK = 4, 1024, 1024, 512, 32, 2


def bf16(value):
    result = np.asarray(value, dtype='<f4').copy()
    if not np.isfinite(result).all():
        raise ValueError('nonfinite input to the finite BF16 oracle')
    bits = result.view('<u4')
    bits += np.uint32(0x7fff) + ((bits >> 16) & 1)
    bits &= np.uint32(0xffff0000)
    return result


def decode(raw, qtype):
    """Independent GGUF block arithmetic; no K-pack placement or reader code."""
    block_bytes = {12: 144, 13: 176}[qtype]
    if raw.shape[-1] != block_bytes or raw.dtype != np.uint8:
        raise ValueError('GGUF block size/type differs')
    head = raw[..., :4].copy().view('<f2').astype('<f4')
    fields = raw[..., 4:16]
    scales = np.empty(raw.shape[:-1] + (8,), dtype=np.uint8)
    mins = np.empty_like(scales)
    scales[..., :4] = fields[..., :4] & 63
    mins[..., :4] = fields[..., 4:8] & 63
    scales[..., 4:] = (fields[..., 8:12] & 15) | ((fields[..., :4] >> 6) << 4)
    mins[..., 4:] = (fields[..., 8:12] >> 4) | ((fields[..., 4:8] >> 6) << 4)
    low = raw[..., 16 if qtype == 12 else 48:].reshape(raw.shape[:-1] + (4, 32))
    codes = np.stack((low & 15, low >> 4), axis=-2).reshape(raw.shape[:-1] + (256,))
    if qtype == 13:
        high = raw[..., 16:48]
        for group in range(8):
            codes[..., group*32:(group+1)*32] |= ((high >> group) & 1) << 4
    scale = np.repeat(head[..., 0, None] * scales, 32, axis=-1)
    minimum = np.repeat(head[..., 1, None] * mins, 32, axis=-1)
    decoded = scale * codes - minimum
    # This is the present BF16 metadata + separate multiply/add contract,
    # not a claim that it has the same error as rounding the final weight once.
    native_scale = bf16(scale)
    native_zero = bf16(bf16(-minimum) + bf16(8 * native_scale))
    native = bf16(bf16((codes.astype('<f4') - 8) * native_scale) + native_zero)
    return decoded, native


def read_array(path, dtype, shape):
    if path.stat().st_size != int(np.prod(shape)) * np.dtype(dtype).itemsize:
        raise ValueError(f'capture size differs: {path}')
    value = np.fromfile(path, dtype=dtype).reshape(shape)
    if not np.isfinite(value).all():
        raise ValueError(f'nonfinite capture: {path}')
    return value


def metric(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError('oracle shape differs')
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    return dict(relative_l2=float(np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-30)),
                max_abs=float(np.max(np.abs(a-b))))


def indexed_dot(activation, weights, ids):
    out = np.empty((len(ids), weights.shape[1]), dtype=np.float64)
    for expert in range(EXPERTS):
        rows = np.flatnonzero(ids == expert)
        out[rows] = activation[rows].astype(np.float64) @ weights[expert].astype(np.float64).T
    return out


def swiglu(pair):
    gate, up = np.split(pair.astype(np.float64), 2, axis=-1)
    return (gate / (1 + np.exp(-gate)) * up).astype('<f4')


def compose(activation, ids, gate, down, rounded):
    pair = indexed_dot(activation, gate, ids)
    if rounded:
        pair = bf16(pair)
    middle = swiglu(pair)
    if rounded:
        middle = bf16(middle)
    partials = [indexed_dot(middle[:, rank*512:(rank+1)*512],
                down[..., rank*512:(rank+1)*512], ids) for rank in range(2)]
    if rounded:
        partials = [bf16(part) for part in partials]
    total = partials[0] + partials[1]
    return total * total


def analyze(output):
    ordinary, retained = output/'ordinary', output/'retained'
    gate, gate_native = decode(read_array(ordinary/'weight-0.bin', 'u1', (4, 2048, 4, 144)), 12)
    down, down_native = decode(read_array(ordinary/'weight-1.bin', 'u1', (4, 512, 4, 176)), 13)
    gate, gate_native = (v.reshape(4, 2048, 1024) for v in (gate, gate_native))
    down, down_native = (v.reshape(4, 512, 1024) for v in (down, down_native))
    for name in ('weight-0.bin', 'weight-1.bin'):
        if (ordinary/name).read_bytes() != (retained/name).read_bytes():
            raise ValueError('ordinary/retained fixture bytes differ')
    result = dict(status='DIAGNOSTIC_COMPLETE', admission='PENDING', timing_valid=False,
        scope='Q4_GATE_UP_Q5_DOWN_E4_TOP2_T32_TP2', threshold_unchanged=.02,
        arithmetic_oracle='GGUF_DECODE_FP64_DOT; BF16_RNE_METADATA_MUL_ADD_AND_STORAGE', replays=[])
    for replay in range(3):
        suffix = f'-i{replay}'
        for name in (f'input{suffix}.f32', f'ids{suffix}.i32', f'result{suffix}.f32'):
            if (ordinary/name).read_bytes() != (retained/name).read_bytes():
                raise ValueError(f'capture changed the original graph result or input: {name}')
        activation = read_array(ordinary/f'input{suffix}.f32', '<f4', (32, 1024))
        ids = read_array(ordinary/f'ids{suffix}.i32', '<i4', (64,))
        if not np.array_equal(ids, (np.arange(64)+replay)%4):
            raise ValueError('router replay differs')
        index = np.arange(32*1024).reshape(32, 1024)
        want_input = (((index*13+index//31+replay*11)%61-30)/128).astype('<f4')
        if not np.array_equal(activation, want_input):
            raise ValueError('activation replay differs')
        activation = np.repeat(activation, TOPK, axis=0)
        actual = read_array(ordinary/f'result{suffix}.f32', '<f4', (64, 512))
        pairs = [read_array(retained/f'pair-r{r}{suffix}.f32', '<f4', (64, 1024)) for r in range(2)]
        pair = np.concatenate([p[:, :512] for p in pairs] + [p[:, 512:] for p in pairs], axis=-1)
        middle = np.concatenate([read_array(retained/f'activation-r{r}{suffix}.f32', '<f4', (64, 512))
                                 for r in range(2)], axis=-1)
        sums = [read_array(retained/f'reduced-r{r}{suffix}.f32', '<f4', (64, 512)) for r in range(2)]
        if not np.array_equal(sums[0].view('<u4'), sums[1].view('<u4')):
            raise ValueError('all-reduce ranks disagree')
        exact = compose(activation, ids, gate, down, False)
        native = compose(activation, ids, gate_native, down_native, True)
        dot = indexed_dot(activation, gate_native, ids)
        down_parts = [bf16(indexed_dot(bf16(middle[:, r*512:(r+1)*512]),
                      down_native[..., r*512:(r+1)*512], ids)) for r in range(2)]
        # Independent coordinate negatives stay visible beside the precision
        # comparison. This diagnostic cannot promote a failed model gate.
        wrong_expert = compose(activation, (ids+1)%4, gate, down, False)
        missing_rank = compose(activation, ids, gate, np.concatenate(
            (down[..., :512], np.zeros_like(down[..., 512:])), axis=-1), False)
        row = dict(replay=replay, original_gate=metric(actual, exact),
            simulated_bf16_vs_reference=metric(native, exact),
            actual_vs_simulated_bf16=metric(actual, native),
            stage_local_errors=dict(gate_up=metric(pair, bf16(dot)),
                swiglu=metric(middle, swiglu(pair)),
                down_and_reduce=metric(sums[0], down_parts[0]+down_parts[1]),
                square=metric(actual, sums[0]*sums[0])),
            wrong_expert_negative=metric(wrong_expert, exact),
            missing_rank_negative=metric(missing_rank, exact))
        if min(row['wrong_expert_negative']['relative_l2'], row['missing_rank_negative']['relative_l2']) <= .02:
            raise ValueError('coordinate negative did not reject')
        result['replays'].append(row)
    result['retained_outputs_match_original_bits'] = True
    return result


def run(binary, output):
    output.mkdir(parents=True, exist_ok=False)
    keys = ('CUDA_VISIBLE_DEVICES', 'PPU_SDK', 'QUACTLIZE_KPACK_EXECUTION',
            'QUACTLIZE_KPACK_COMPUTE', 'QUACTLIZE_KPACK_ROUTE', 'QUACTLIZE_PPU_BUNDLE')
    env = dict(os.environ)
    if env.get('QUACTLIZE_KPACK_COMPUTE') != 'bf16' or env.get('QUACTLIZE_KPACK_ROUTE') != 'auto':
        raise ValueError('keep the original BF16 auto-selected arithmetic')
    (output/'environment.json').write_text(json.dumps({k:env.get(k) for k in keys}, indent=2)+'\n')
    for arm in ('ordinary', 'retained'):
        directory = output/arm
        directory.mkdir()
        command = [str(binary), '--tp2-chain-dump', str(directory), arm]
        (output/(arm+'.command.json')).write_text(json.dumps(command)+'\n')
        print(f'KPACK_TP2_CHAIN_START arm={arm}', flush=True)
        started = time.monotonic()
        log = output/(arm+'.log')
        with log.open('x') as stream:
            process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            try:
                while True:
                    try:
                        rc = process.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        print(f'KPACK_TP2_CHAIN_WAIT arm={arm} seconds={time.monotonic()-started:.0f}', flush=True)
                        if time.monotonic()-started > 600:
                            raise TimeoutError('chain capture exceeded ten minutes')
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        text = log.read_text(errors='replace')
        rows = re.findall(r'^KPACK_TP2_CHAIN_CAPTURE replay=([012]) retain=([01]) relative=(\S+) threshold=0.02 admitted=0$', text, re.M)
        if rc or [(r, keep) for r, keep, _ in rows] != [(str(r), str(int(arm=='retained'))) for r in range(3)]:
            raise ValueError(f'{arm}: incomplete capture rc={rc}; see {log}')
        print(f'KPACK_TP2_CHAIN_CAPTURED arm={arm} seconds={time.monotonic()-started:.1f}', flush=True)
    result = analyze(output)
    result['files'] = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(output.glob('*/*')) if p.is_file()}
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    for row in result['replays']:
        print('KPACK_TP2_CHAIN_PRECISION '+json.dumps(row), flush=True)
    print(f'KPACK_TP2_CHAIN_DONE status=DIAGNOSTIC_COMPLETE admission=PENDING results={output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.binary.resolve(strict=True), args.output.resolve())
