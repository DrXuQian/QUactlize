"""FFN versus independently checked projections/activation + CPU weighted sum.

Each unfused projection is also checked against original GGUF weights, not
device-packed planes; the F32 activation is checked against F64 CPU math.
No timers, performance admission, or llama dependency.
"""
import ctypes as C

import numpy as np
import pytest

from ci.support.dense import Params as DenseParams, Info as DenseInfo, Options
from ci.support.moe import Params as MoEParams, Info as MoEInfo, Public as MoE
from ci.support.ffn import (Weight, Params, Info, Public as FFN, Value, Transition,
                            check_swiglu, projection_output_compute)
from ci.support.fixture import compare, round_compute
from ci.support.graph import Replay
from .support import Arena, Public, QTYPES, packed, check_output


@pytest.mark.parametrize('kind', [0, 1], ids=['dense', 'moe'])
@pytest.mark.parametrize('merged', [False, True], ids=['separate', 'canonical-2n'])
@pytest.mark.parametrize('q', QTYPES)
def test_public_ffn(package, sdk, kind, merged, q):
    run_ffn(package, sdk, kind, merged, q)


@pytest.mark.parametrize('kind', [0, 1], ids=['dense', 'moe'])
@pytest.mark.parametrize('merged', [False, True], ids=['separate', 'canonical-2n'])
@pytest.mark.parametrize('q', QTYPES)
def test_public_ffn_alias(package, sdk, kind, merged, q):
    ms = (1, 8, 9, 128, 513) + ((5120,) if q == 12 and merged else ())
    run_ffn(package, sdk, kind, merged, q, ms=ms, aliases=True)


def test_public_ffn_alias_qwen35(package, sdk):
    run_ffn(package, sdk, 1, True, 12, k=2048, hidden=512, n=2048,
            experts=256, topk=8, ms=(1, 2048), down_q=13, input_channels=1, aliases=True)


@pytest.mark.parametrize('kind,q,k,hidden,down_q', [
    (0, 8, 2048, 512, 8),
    (0, 8, 3072, 1024, 8),
    (1, 12, 2048, 512, 14),
    (1, 12, 3072, 512, 13),
], ids=['shared-35b', 'shared-122b', 'moe-35b', 'moe-122b'])
def test_public_ffn_model_shapes(package, sdk, kind, q, k, hidden, down_q):
    # Real shared/routed cohorts retain the measured 2N projection at every M.
    # SwiGLU is a separate transition even where a historical fused rule exists.
    run_ffn(package, sdk, kind, True, q, hidden=hidden, k=k, n=k,
            experts=256 if kind else 1, topk=8 if kind else 1,
            ms=tuple(range(1, 9)), down_q=down_q, input_channels=1)


def run_ffn(package, sdk, kind, merged, q, *, hidden=512, k=512, n=256,
            experts=None, topk=None, ms=None, down_q=None, input_channels=None, aliases=False):
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
            experts = experts or (4 if kind else 1)
            topk = topk or (2 if kind else 1)
            descriptors, gold = [], []
            for stage in range(3):
                if merged and stage == 1:
                    descriptors.append(Weight()); gold.append(None)
                    continue
                columns = n if stage == 2 else hidden
                reduction = hidden if stage == 2 else k
                dtype = (down_q or q) if stage == 2 else q
                arr, sizes, ptrs, weight = packed(storage, public, dtype, columns, reduction, experts,
                    pair=bool(merged and stage == 0), seed=stage*97)
                descriptors.append(Weight(dtype, columns*(2 if merged and stage == 0 else 1), reduction,
                                          experts, arr, sizes, *ptrs))
                gold.append(weight)
            # Cover each decode M, the former cutoff, prefill and an unsampled
            # non-power-of-two M. A large-M witness is shared, not multiplied
            # across every dtype; host shape tests cover up to M8192 for all.
            ms = ms or (*range(1, 10), 16, 128, 513) + ((5120,) if q == 12 and merged else ())
            for m in ms:
                context_label = (f'kind={kind} q={q} down_q={down_q or q} m={m} '
                                 f'n={hidden} k={k} merged={int(merged)} swiglu=transition')
                print('FFN_ACCURACY_START ' + context_label, flush=True)
                channels = input_channels if input_channels is not None else topk if kind and m % 2 == 0 else 1
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
                        a = MoEParams(1, C.sizeof(MoEParams), w.qtype, w.n, w.k, experts, m, topk, ch, 0,
                            w.k, w.k*ch, topk, w.n, w.arrangement, w.sizes)
                        chosen = MoEInfo(1, C.sizeof(MoEInfo))
                    else:
                        a = DenseParams(1, C.sizeof(DenseParams), w.qtype, w.n, w.k, m, w.k, w.n,
                                        w.arrangement, w.sizes)
                        chosen = DenseInfo(1, C.sizeof(DenseInfo))
                    handle = C.c_void_p()
                    projection.check(projection.prepare(control, C.byref(a), w.low, w.high, w.units,
                        C.byref(handle), C.byref(chosen)))
                    selected = info.projections[stage]
                    assert (selected.route, selected.policy, selected.split, selected.compute_type, bytes(selected.config)) == (
                        chosen.route, chosen.policy, chosen.split, chosen.compute_type, bytes(chosen.config))
                    handles.append(handle); control_info.append(chosen)
                graph_safe = bool(kind) or all(i is None or i.route != 3 for i in control_info)
                # Retire each shape's activations/scratch before the next M.
                with Arena(sdk) as arena:
                    lanes = []
                    graphs = []
                    for lane in range(2):
                        lanes.append(dict(stream=arena.stream(), a=arena.buffer(m*channels*k*4),
                            ids=arena.buffer(m*topk*4), weights=arena.buffer(m*topk*4),
                            out=arena.buffer(m*n*4), scratch=arena.buffer(info.workspace_bytes)))
                        if aliases:
                            lanes[-1]['alias'] = arena.buffer(max(m*channels*k*4, m*n*4, m*topk*8) + 64)
                            lanes[-1]['alias_graphs'] = {}
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
                                if aliases:
                                    io['host'] = dict(a=a, ids=ids, weights=weights)
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
                                    x = source[:, np.arange(topk) % ch].reshape(-1, descriptors[stage].k)
                                    y = out.reshape(m*topk, -1)
                                    for expert in range(experts):
                                        mask = ids.reshape(-1) == expert
                                        if mask.any():
                                            chosen = control_info[stage]
                                            rounding = projection_output_compute(kind, m, chosen)
                                            try:
                                                check_output(y[mask], x[mask], gold[stage][expert], chosen.compute_type,
                                                             output_compute=rounding)
                                            except ValueError as exc:
                                                raise AssertionError(
                                                    f'FFN_PROJECTION_CHECK {context_label} stage={stage} '
                                                    f'expert={expert} repeat={repeat} lane={lane} '
                                                    f'output_rounding={rounding or "f32"} route={chosen.route} '
                                                    f'policy={chosen.policy} split={chosen.split} '
                                                    f'config={bytes(chosen.config).decode(errors="replace")}: {exc}') from exc
                                    return out.copy()

                                gate = project(0, io['a'].ptr, a)
                                if merged:
                                    gate, up = gate[..., :hidden], gate[..., hidden:]
                                else:
                                    up = project(1, io['a'].ptr, a)
                                # Independent token/slot-order transition, checked in F32
                                # before Down casts to its operand type. Do not inject the
                                # CPU exp/divide rounding into the separate GPU chain.
                                call = Transition(1, C.sizeof(Transition), m, topk, hidden, int(merged),
                                    Value(tmp[0].ptr, 0, 1, 0, 1, descriptors[0].n, 0, 0, 0),
                                    Value() if merged else Value(tmp[1].ptr, 0, 1, 0, 1, hidden, 0, 0, 0),
                                    0, 1, hidden, activation.ptr, None, None, io['stream'])
                                activation.poison(); arena.sync()
                                rc = api.transition(C.byref(call))
                                assert rc == 0, f'FFN_SWIGLU_RUN {context_label} rc={rc}'
                                arena.sync()
                                hidden_value = np.frombuffer(activation.bytes(), 'f4').reshape(m, topk, hidden).copy()
                                try:
                                    check_swiglu(hidden_value, gate, up)
                                except ValueError as exc:
                                    raise AssertionError(
                                        f'FFN_SWIGLU_CHECK {context_label} repeat={repeat} lane={lane}: {exc}') from exc
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
                                if repeat == 1 and graph_safe:
                                    graphs.append(Replay(sdk, io['stream'], run, 1))
                                api.check(graphs[lane]() if graphs else run())
                            arena.sync()
                            for lane, io in enumerate(lanes):
                                got = np.frombuffer(io['out'].bytes(), 'f4').reshape(m, n)
                                try:
                                    compare(got, expected[lane], denominators[lane])
                                except ValueError as exc:
                                    raise AssertionError(f'FFN_CHAIN_CHECK {context_label} repeat={repeat} lane={lane}: {exc}') from exc
                                for key in ('a', 'ids', 'weights', 'scratch'):
                                    io[key].guard()
                                if aliases:
                                    check_aliases(sdk, arena, api, plan, params, io, expected[lane],
                                                  denominators[lane], repeat, graph_safe, graphs, context_label)
                        print(f'FFN_ACCURACY kind={kind} q={q} down_q={down_q or q} m={m} '
                              f'n={hidden} k={k} swiglu=transition graph={int(graph_safe)} status=PASS', flush=True)
                    finally:
                        arena.sync()
                        for graph in graphs:
                            graph.close()
    finally:
        if control:
            projection.destroy(control)
        api.destroy(context)


def check_aliases(sdk, arena, api, plan, params, io, expected, denom, repeat, graph_safe, graphs, label):
    pool = io['alias']
    for case in (('a', 'ids', 'weights', 'routing') if params.kind else ('a',)):
        image = np.full(pool.size, 0xA5, dtype='u1')
        pointers = {key: io[key].ptr for key in ('a', 'ids', 'weights')}
        keys = ('ids', 'weights') if case == 'routing' else (case,)
        offset = 16  # Exercise partial, not only identical, input/output aliases.
        for key in keys:
            raw = io['host'][key].view('u1').reshape(-1)
            image[offset:offset + len(raw)] = raw
            pointers[key] = pool.ptr + offset
            offset += (len(raw) + 15) // 16 * 16
        pool.write(image)
        arena.sync()
        args = (plan, pointers['a'], pointers['ids'] if params.kind else None,
                pointers['weights'] if params.kind else None, pool.ptr,
                io['scratch'].ptr, io['scratch'].size, io['stream'])
        assert api.run(*args) != 0, 'v1 must retain its no-alias contract'
        if repeat == 1 and graph_safe:
            graph = Replay(sdk, io['stream'], lambda args=args: api.run_alias(*args), 1)
            io['alias_graphs'][case] = graph
            graphs.append(graph)
        graph = io['alias_graphs'].get(case)
        api.check(graph() if graph else api.run_alias(*args))
        arena.sync()
        got = np.frombuffer(pool.bytes(), 'f4', count=params.tokens*params.down.n).reshape(expected.shape)
        try:
            compare(got, expected, denom)
        except ValueError as exc:
            raise AssertionError(f'FFN_ALIAS_CHECK {label} alias={case} repeat={repeat}: {exc}') from exc
        io['scratch'].guard()
        print(f'FFN_ALIAS_CHECK {label} alias={case} repeat={repeat} graph={int(graph is not None)} status=PASS', flush=True)
