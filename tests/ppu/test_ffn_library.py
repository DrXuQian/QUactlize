"""FFN versus independent projections + CPU activation/weighted sum.

Each unfused projection is also checked against original GGUF weights, not
device-packed planes. No timers, performance admission, or llama dependency.
"""
import ctypes as C

import numpy as np
import pytest

from ci.support.dense import Params as DenseParams, Info as DenseInfo, Options
from ci.support.moe import Params as MoEParams, Info as MoEInfo, Public as MoE
from ci.support.ffn import Weight, Params, Info, Public as FFN
from ci.support.fixture import compare, round_compute
from ci.support.graph import Replay
from .support import Arena, Public, QTYPES, packed, check_output


@pytest.mark.parametrize('kind', [0, 1], ids=['dense', 'moe'])
@pytest.mark.parametrize('merged', [False, True], ids=['separate', 'canonical-2n'])
@pytest.mark.parametrize('q', QTYPES)
def test_public_ffn(package, sdk, kind, merged, q):
    public = Public(package)
    api = FFN(public.lib)
    projection = MoE(public.lib) if kind else public
    context, control = C.c_void_p(), C.c_void_p()
    api.check(api.create(0, C.byref(context)))
    try:
        if kind:
            projection.check(projection.create(0, C.byref(control)))
        else:
            options = Options(1, C.sizeof(Options), 0, 0)
            projection.check(projection.create(0, C.byref(options), C.byref(control)))
        with Arena(sdk) as storage:
            experts, topk, hidden, k, n = (4 if kind else 1), (2 if kind else 1), 512, 512, 256
            descriptors, gold = [], []
            for stage in range(3):
                if merged and stage == 1:
                    descriptors.append(Weight()); gold.append(None)
                    continue
                columns = n if stage == 2 else hidden
                arr, sizes, ptrs, weight = packed(storage, public, q, columns, k, experts,
                    pair=bool(merged and stage == 0), seed=stage*97)
                descriptors.append(Weight(q, columns*(2 if merged and stage == 0 else 1), k, experts, arr, sizes, *ptrs))
                gold.append(weight)
            # Cover each decode M, the former cutoff, prefill and an unsampled
            # non-power-of-two M. A large-M witness is shared, not multiplied
            # across every dtype; host shape tests cover up to M8192 for all.
            ms = (*range(1, 10), 16, 128, 513) + ((5120,) if q == 12 and merged else ())
            for m in ms:
                channels = topk if kind and m % 2 == 0 else 1
                params = Params(1, C.sizeof(Params), kind, m, topk, channels, int(merged), 0,
                    k, k*channels, topk if kind else 0, topk if kind else 0, n, *descriptors)
                assert api.supported(C.byref(params)) == 1
                plan, info = C.c_void_p(), Info(1, C.sizeof(Info))
                api.check(api.prepare(context, C.byref(params), C.byref(plan), C.byref(info)))
                handles, control_info = [], []
                for stage, w in enumerate(descriptors):
                    if merged and stage == 1:
                        handles.append(None); control_info.append(None)
                        continue
                    ch = topk if stage == 2 else channels
                    if kind:
                        a = MoEParams(1, C.sizeof(MoEParams), q, w.n, k, experts, m, topk, ch, 0,
                            k, k*ch, topk, w.n, w.arrangement, w.sizes)
                        chosen = MoEInfo(1, C.sizeof(MoEInfo))
                    else:
                        a = DenseParams(1, C.sizeof(DenseParams), q, w.n, k, m, k, w.n, w.arrangement, w.sizes)
                        chosen = DenseInfo(1, C.sizeof(DenseInfo))
                    handle = C.c_void_p()
                    projection.check(projection.prepare(control, C.byref(a), w.low, w.high, w.units,
                        C.byref(handle), C.byref(chosen)))
                    selected = info.projections[stage]
                    assert (selected.route, selected.split, selected.compute_type, bytes(selected.config)) == (
                        chosen.route, chosen.split, chosen.compute_type, bytes(chosen.config))
                    handles.append(handle); control_info.append(chosen)
                # Retire each shape's activations/scratch before the next M.
                with Arena(sdk) as arena:
                    lanes = []
                    graphs = []
                    for lane in range(2):
                        lanes.append(dict(stream=arena.stream(), a=arena.buffer(m*channels*k*4),
                            ids=arena.buffer(m*topk*4), weights=arena.buffer(m*topk*4),
                            out=arena.buffer(m*n*4), scratch=arena.buffer(info.workspace_bytes)))
                    tmp = [arena.buffer(m*topk*w.n*4) if w.n else None for w in descriptors]
                    work = [arena.buffer(i.workspace_bytes) if i else None for i in control_info]
                    activation = arena.buffer(m*topk*hidden*4)
                    try:
                        for repeat in range(3):
                            expected, denominators = [], []
                            for lane, io in enumerate(lanes):
                                rng = np.random.default_rng(301+q+m+repeat*7+lane)
                                a = rng.uniform(-.05, .05, (m, channels, k)).astype('f4')
                                ids = ((np.arange(m)[:, None]*3+np.arange(topk)+repeat+lane) % experts).astype('i4')
                                weights = rng.uniform(.1, 1, (m, topk)).astype('f4')
                                io['a'].write(a); io['ids'].write(ids); io['weights'].write(weights); io['out'].poison()
                                arena.sync()

                                def project(stage, ptr, source):
                                    if kind:
                                        rc = projection.run(handles[stage], ptr, io['ids'].ptr, tmp[stage].ptr,
                                            work[stage].ptr, work[stage].size, io['stream'])
                                    else:
                                        rc = projection.run(handles[stage], ptr, tmp[stage].ptr,
                                            work[stage].ptr, work[stage].size, io['stream'])
                                    projection.check(rc); arena.sync()
                                    out = np.frombuffer(tmp[stage].bytes(), 'f4').reshape(m, topk, -1)
                                    ch = source.shape[1]
                                    # Check all rows against their actual expert, grouping on
                                    # CPU only for the independent oracle, never the runtime.
                                    x = source[:, np.arange(topk) % ch].reshape(-1, k)
                                    y = out.reshape(m*topk, -1)
                                    for expert in range(experts):
                                        mask = ids.reshape(-1) == expert
                                        if mask.any():
                                            check_output(y[mask], x[mask], gold[stage][expert], control_info[stage].compute_type)
                                    return out.copy()

                                gate = project(0, io['a'].ptr, a)
                                if merged:
                                    gate, up = gate[..., :hidden], gate[..., hidden:]
                                else:
                                    up = project(1, io['a'].ptr, a)
                                hidden_value = (gate/(1+np.exp(-gate))*up).astype('f4')
                                activation.write(hidden_value); arena.sync()
                                down = project(2, activation.ptr, hidden_value)
                                # No reduction reordering in the CPU reference.
                                weight_values = weights if kind else np.ones_like(weights)
                                want = np.zeros((m, n), 'f4')
                                for slot in range(topk):
                                    want += down[:, slot]*weight_values[:, slot, None]
                                # Absolute-dot normalization remains meaningful near zero.
                                x = round_compute(hidden_value, 'bf16' if kind else 'f16').astype('f8')
                                denom = np.zeros((m, n), 'f8')
                                for expert in range(experts):
                                    token, slot = np.nonzero(ids == expert)
                                    if len(token):
                                        terms = np.abs(x[token, slot]) @ np.abs(gold[2][expert].astype('f8')).T
                                        np.add.at(denom, token, terms*np.abs(weight_values[token, slot, None]))
                                expected.append(want); denominators.append(denom)
                            arena.sync()
                            for lane, io in enumerate(lanes):
                                def run(io=io):
                                    return api.run(plan, io['a'].ptr, io['ids'].ptr if kind else None,
                                        io['weights'].ptr if kind else None, io['out'].ptr,
                                        io['scratch'].ptr, io['scratch'].size, io['stream'])
                                if repeat == 1:
                                    graphs.append(Replay(sdk, io['stream'], run, 1))
                                api.check(graphs[lane]() if graphs else run())
                            arena.sync()
                            for lane, io in enumerate(lanes):
                                got = np.frombuffer(io['out'].bytes(), 'f4').reshape(m, n)
                                compare(got, expected[lane], denominators[lane])
                                for key in ('a', 'ids', 'weights', 'scratch'):
                                    io[key].guard()
                    finally:
                        arena.sync()
                        for graph in graphs:
                            graph.close()
    finally:
        if control:
            projection.destroy(control)
        api.destroy(context)
