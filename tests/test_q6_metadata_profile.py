from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from quactlize.dispatch.table import Workload
from tools.profile_q6_metadata import measure, validate_prior


def test_one_copy_no_graphs_selected_arm_only():
    events = []

    class Fn:
        def __init__(self, name):
            self.name = name

        def __call__(self):
            events.append(self.name)
            return 0

    class Resources:
        def __init__(self, sdk):
            events.append('resources')

        def close(self):
            events.append('weights-closed')

    class Buffer:
        def __init__(self, owner, size):
            self.ptr = 128
            events.append(('buffer', size))

        def upload(self, value):
            events.append(('upload', value.nbytes))

        def guard(self):
            events.append('guard')

    class Context:
        def __init__(self, sdk, ring, point):
            assert len(ring.planes) == 1
            self.ring = ring
            self.r = NS(stream=123)
            self.out = NS(poison=lambda: events.append('poison'))

        def update(self, repeat):
            events.append(('input', repeat))

        def check(self):
            events.append('check')
            return {'error': 0}

        def close(self):
            events.append('context-closed')

    class Kernel:
        def __init__(self, b, library):
            assert len(b.ring.planes) == 1
            assert library == Path('old-build/bf16.so')

        def launch(self, arm):
            events.append(('launch', arm))
            return 0

    sdk = NS(lib=NS(hggcProfilerStart=Fn('start'), hggcProfilerStop=Fn('stop')),
             synchronize=lambda stream: events.append(('sync', stream)))
    api = NS(Weights=lambda *args: NS(planes={n: NS(nbytes=i) for n, i in
                                            (('low', 16), ('high', 8), ('units', 4))}),
             Resources=Resources, Buffer=Buffer, Context=Context, point=lambda w: {},
             checked=lambda rc, message: None)
    result = measure(NS(arm=1), NS(Kernel=Kernel), sdk,
                     Workload(14, 512, 2048, 8, compute='bf16', operation='indexed', experts=256, topk=8),
                     Path('old-build/bf16.so'), api)
    assert [e for e in events if isinstance(e, tuple) and e[0] == 'launch'] == [('launch', 1)] * 4
    assert events[events.index('start')+1:events.index('stop')] == [('launch', 1), ('sync', 123)]
    assert sum(e == 'check' for e in events) == 4
    assert events[-2:] == ['context-closed', 'weights-closed']
    assert result['weight_copies'] == 1 and result['weight_bytes'] == 28
    assert result['cold_ring'] is False and result['graph_api_calls'] == 0
    assert result['timings'] == [] and result['performance_admitted'] is False


def test_prior_gate():
    w = Workload(14, 512, 2048, 8)
    row = dict(status='PASS', profiled=False, raw_equal=True, timings=[{}, {}],
               authority_sha256='authority', library_sha256='image', workload=asdict(w))
    validate_prior(row, {}, {'sha256': 'image'}, w, lambda _: 'authority')
    for key, value in (('status', 'FAIL'), ('profiled', True), ('raw_equal', False),
                       ('timings', []), ('authority_sha256', 'other'),
                       ('library_sha256', 'other'), ('workload', {})):
        with pytest.raises(ValueError, match='completed numerical/timing point'):
            validate_prior(row | {key: value}, {}, {'sha256': 'image'}, w, lambda _: 'authority')


def test_acu_graph_opt_out_and_original_default():
    from dev.native_components.profile import acu_command
    direct = acu_command('acu', 'report', ['python', 'profile'], graph_profiling='none')
    original = acu_command('acu', 'report', ['python', 'profile'])
    assert direct[direct.index('--graph-profiling')+1] == 'none'
    assert original[original.index('--graph-profiling')+1] == 'node'
    assert direct[direct.index('--profile-from-start')+1] == 'no'
    assert direct[direct.index('--cache-control')+1] == 'all'
    assert direct[direct.index('--launch-count')+1] == '1'
    with pytest.raises(ValueError):
        acu_command('acu', 'report', [], graph_profiling='unsupported')


def test_parent_profiles_with_separate_driver():
    import inspect
    from tools.run_q6_metadata import collect
    source = inspect.getsource(collect)
    assert 'tools/profile_q6_metadata.py' in source
    assert "graph_profiling='none'" in source
    assert "'--profile-arm'" not in source
