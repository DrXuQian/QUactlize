#!/usr/bin/env python3
"""Compare bounded first-token snapshots without declaring model admission."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


PATTERN = re.compile(
    r'^LLAMA_NUMERICAL_TENSOR role=node-(\d+) tensor="([^"]+)" op=(\S+) type=(f32|f16|bf16) '
    r'ne=([\d,]+) nb=([\d,]+) count=(\d+) ', re.M)


def range_summary(values):
    finite = np.isfinite(values)
    with np.errstate(over='ignore', invalid='ignore'):
        narrowed = values.astype(np.float16)
    overflow = np.flatnonzero(finite & ~np.isfinite(narrowed))
    return dict(elements=int(values.size), nonfinite=int((~finite).sum()),
                min=float(np.min(values[finite])) if finite.any() else None,
                max=float(np.max(values[finite])) if finite.any() else None,
                max_abs=float(np.max(np.abs(values[finite]))) if finite.any() else None,
                f16_new_nonfinite=int(overflow.size),
                overflow_examples=[dict(index=int(i), value=float(values[i])) for i in overflow[:8]])


def scalar(value):
    value = np.float32(value)
    with np.errstate(over='ignore', invalid='ignore'):
        narrowed = value.astype(np.float16)
    return dict(value=float(value) if np.isfinite(value) else None,
                f32_bits=f'0x{int(value.view(np.uint32)):08x}',
                f16_bits=f'0x{int(narrowed.view(np.uint16)):04x}')


def paired_points(want, got, reference, native):
    a, b = want['values'], got['values']
    with np.errstate(over='ignore', invalid='ignore'):
        indices = set(np.flatnonzero(np.isfinite(a) & ~np.isfinite(a.astype(np.float16)))[:8])
        indices.update(np.flatnonzero(np.isfinite(b) & ~np.isfinite(b.astype(np.float16)))[:8])
    for values in (a, b):
        finite = np.flatnonzero(np.isfinite(values))
        if finite.size:
            indices.add(finite[np.argmax(values[finite])])
    layer = re.fullmatch(r'ffn_swiglu-(\d+)', got['name'])
    points = []
    for index in sorted(indices):
        point = dict(index=int(index), reference=scalar(a[index]), native=scalar(b[index]))
        if layer:
            for arm, records in (('reference', reference), ('native', native)):
                gate = records.get((f'ffn_gate-{layer[1]}', got['occurrence']))
                up = records.get((f'ffn_up-{layer[1]}', got['occurrence']))
                if gate is None or up is None:
                    point[arm]['operands'] = 'MISSING_SNAPSHOT'
                    continue
                if gate['shape'] != got['shape'] or up['shape'] != got['shape']:
                    raise ValueError('SwiGLU operand shape differs from the output')
                g, u = gate['values'][index], up['values'][index]
                with np.errstate(over='ignore', invalid='ignore'):
                    product = np.float32(g / (np.float32(1) + np.exp(-g))) * u
                point[arm].update(gate=scalar(g), up=scalar(u),
                                  swiglu_recomputed_f32=scalar(product),
                                  recompute_scope='HOST_F32_NOT_DEVICE_EXP_ORACLE')
        points.append(point)
    return points


def snapshots(results, arm):
    results = Path(results)
    text = (results / f'{arm}.log').read_text(errors='replace')
    matches = list(PATTERN.finditer(text))
    if not matches or [int(m[1]) for m in matches] != list(range(1, len(matches) + 1)):
        raise ValueError(f'{arm}: snapshot sequence is incomplete')
    seen, records = {}, []
    for m in matches:
        node, name, op, kind = int(m[1]), m[2], m[3], m[4]
        shape, strides = tuple(map(int, m[5].split(','))), tuple(map(int, m[6].split(',')))
        if len(shape) != 4 or len(strides) != 4 or min(shape) <= 0 or np.prod(shape) != int(m[7]):
            raise ValueError(f'{arm}/{node}: shape/count mismatch')
        path = results / f'{arm}-tensors' / f'node-{node}.bin'
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError(f'{arm}/{node}: snapshot exceeds the producer limit')
        data = path.read_bytes()
        dtype = '<f4' if kind == 'f32' else '<f2' if kind == 'f16' else '<u2'
        end = sum((s - 1) * d for s, d in zip(shape, strides)) + np.dtype(dtype).itemsize
        if end != len(data):
            raise ValueError(f'{arm}/{node}: strided snapshot byte extent differs')
        array = np.ndarray(shape[::-1], dtype=dtype, buffer=data, strides=strides[::-1]).reshape(-1)
        if kind == 'bf16':
            array = (array.astype(np.uint32) << 16).view(np.float32)
        else:
            array = array.astype(np.float32)
        occurrence = seen.get(name, 0)
        seen[name] = occurrence + 1
        records.append(dict(node=node, name=name, occurrence=occurrence, op=op, type=kind,
                            shape=shape, sha256=hashlib.sha256(data).hexdigest(), values=array))
    return records


def compare(results, target):
    reference, native = [snapshots(results, arm) for arm in ('reference-tensors', 'native-tensors')]
    if reference[-1]['name'] != target:
        raise ValueError('reference snapshot did not reach the requested tensor')
    ref = {(r['name'], r['occurrence']): r for r in reference}
    nat = {(r['name'], r['occurrence']): r for r in native}
    rows, unmatched = [], []
    for r in native:
        want = ref.get((r['name'], r['occurrence']))
        if want is None:
            unmatched.append(dict(name=r['name'], occurrence=r['occurrence']))
            continue
        if (want['shape'], want['type'], want['op']) != (r['shape'], r['type'], r['op']):
            raise ValueError(f'{r["name"]}: reference/native endpoint differs')
        a, b = want['values'].astype(np.float64), r['values'].astype(np.float64)
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        maximum = float(np.max(np.abs(a - b))) if finite else None
        rows.append(dict(name=r['name'], occurrence=r['occurrence'], shape=r['shape'],
                         reference=range_summary(a), native=range_summary(b),
                         reference_sha256=want['sha256'], native_sha256=r['sha256'],
                         max_abs_error=maximum,
                         relative_linf=maximum / max(1e-30, float(np.max(np.abs(a)))) if finite else None,
                         paired_points=paired_points(want, r, ref, nat)))
    if not rows:
        raise ValueError('no comparable intermediate snapshots')
    return dict(scope='PAIRED_FIRST_TOKEN_INTERMEDIATES_NOT_ACCURACY_OR_TIMING_ADMISSION',
                target=target, native_target_reached=native[-1]['name'] == target,
                callback_changes_graph_partition=True, snapshots='AT_PRODUCER_BEFORE_DOWNSTREAM_CONSUMERS',
                unmatched_native=unmatched, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--target', default='ffn_out-2')
    args = parser.parse_args()
    print(json.dumps(compare(args.results, args.target), indent=2, allow_nan=False))
