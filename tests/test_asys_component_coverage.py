"""Missing activity remains a rejection; diagnostics never fabricate timing."""
from copy import deepcopy
import json

import pytest

from dev.strategy.asys_components import parse_component_rows
from tools import inspect_asys_component_coverage as inspect_tool


def fixture():
    node = dict(symbol='kernel', grid=[1,1,1], block=[32,1,1], shared_bytes=0)
    records = {'test-component': dict(identity=dict(node=node), calls_per_graph=16)}
    schedule = [dict(component='test-component', round=r, sample=s, kernels=16)
                for r in range(4) for s in range(3)]
    rows = [dict(node, name='kernel', kind='kernel', start=i*1100, end=i*1100+1000)
            for i in range(192)]
    return records, schedule, rows


@pytest.mark.parametrize('available', [0, 1, 16, 31, 191])
def test_missing_activity_reports_first_incomplete_sample(available):
    records, schedule, rows = fixture()
    before = deepcopy(records)
    with pytest.raises(ValueError) as caught:
        parse_component_rows(records, schedule, rows[:available], {}, 4, 3)
    text = str(caught.value)
    sample = available//16
    for marker in ('coverage incomplete', 'expected_total=192', f'actual_total={available}',
                   f'missing_total={192-available}', 'component=test-component',
                   f'round={sample//3}', f'sample={sample%3}',
                   f'first_activity={sample*16}', f'actual_sample={available%16}',
                   'no timing admitted'):
        assert marker in text
    assert records == before


def test_complete_coverage_retains_per_record_times_and_rejects_extra():
    records, schedule, rows = fixture()
    result = parse_component_rows(records, schedule, rows, {}, 4, 3)
    assert result['test-component']['mean_us'] == 1
    assert result['test-component']['samples_us'] == [[1]*3]*4
    with pytest.raises(ValueError, match='unaccounted'):
        parse_component_rows(records, schedule, rows+[rows[-1]], {}, 4, 3)


@pytest.mark.parametrize('available', [191, 192])
def test_inspector_is_read_only_and_not_an_admission(tmp_path, monkeypatch, available):
    records, schedule, rows = fixture()
    capture = dict(component_records=records, trace_schedule=schedule,
                   protocol=dict(rounds=4, samples=3))
    path = tmp_path/'capture.json'
    path.write_text(json.dumps(capture))
    before = path.read_bytes()
    monkeypatch.setattr(inspect_tool, 'activities', lambda *a, **kw: rows[:available])
    monkeypatch.setattr(inspect_tool, 'runtime_activities', lambda *a: [])
    result = inspect_tool.inspect(tmp_path)
    assert result['expected_kernels'] == 192 and result['actual_kernels'] == available
    assert not result['gpu_work'] and not result['performance_admitted']
    assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]
    assert (result['status'] == 'COMPONENT_COVERAGE_MATCHES') == (available == 192)
