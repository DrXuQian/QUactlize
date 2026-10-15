"""A bounded follow-up must not claim the full small-M denominator."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from dev.strategy.subset import point_filter, filtered_plan, completion_status
from quactlize.dispatch.table import Workload
from quactlize.runtime.tuning import digest


def test_filter_keeps_complete_candidate_pools_without_mutating_full_plan():
    points = [dict(id=w.key, workload=asdict(w), candidates=['rule'], champion_controls=['old'])
              for w in (Workload(8,1024,2048,1), Workload(8,1024,2048,8),
                        Workload(12,1024,2048,1), Workload(8,2048,512,1))]
    p = dict(points=points, work_counts={'points':4}, plan_sha256='old')
    before = deepcopy(p)
    selected = point_filter('{"qtype":8,"n":1024,"k":2048,"operation":"dense"}')
    got = filtered_plan(p, selected)
    assert got['points'] == points[:2] and got['registry_points'] == 4
    assert got['point_filter'] == selected and 'work_counts' not in got
    assert got['plan_sha256'] == digest({k:v for k,v in got.items() if k!='plan_sha256'})
    assert p == before and filtered_plan(p,{}) is p
    with pytest.raises(ValueError, match='no registered'):
        filtered_plan(p, {'qtype':14})


@pytest.mark.parametrize('text', ['[]', '{"qtype":9}', '{"m":1}', '{"n":true}',
                                  '{"k":0}', '{"k":2147483648}', '{"operation":"grouped"}'])
def test_invalid_or_unintended_filter_is_rejected(text):
    with pytest.raises(ValueError):
        point_filter(text)


def test_subset_is_never_called_complete_full_campaign():
    assert completion_status(12,12,0,{'qtype':8}) == 'COMPLETE_SMALLM_SUBSET'
    assert completion_status(12,12,1,{'qtype':8}) == 'INCOMPLETE'
    assert completion_status(11,12,0,{'qtype':8}) == 'INCOMPLETE'
    assert completion_status(1114,1114,0,{}) == 'COMPLETE_SMALLM'


def test_resume_refuses_changed_filter_before_rewriting_any_receipt(tmp_path):
    pytest.importorskip('torch')
    import tools.run_kpack_strategy_campaign as c
    path = tmp_path/'plan/plan.json'; path.parent.mkdir()
    old = '{"point_filter":{"qtype":8},"plan_sha256":"sealed"}'
    path.write_text(old)
    args = SimpleNamespace(output=tmp_path, point_filter={})
    with pytest.raises(ValueError, match='filter changed'):
        c.freeze_plan(args)
    assert path.read_text() == old


def test_filtered_decode_plan_preserves_partial_coverage_label(tmp_path):
    from dev.strategy.decode_plan import make_decode_plan
    from test_strategy_decode import original
    from test_strategy_resource_pilot import example
    probe = example(tmp_path)
    probe['resolved']['point_filter'] = {'qtype':12}
    result = make_decode_plan(probe, original(probe), pilot=False)
    assert result['point_filter'] == {'qtype':12}
    assert not result['full_smallm_cohort'] and not result['full_campaign']


def test_box_wrapper_forwards_filter_as_one_argument_and_keeps_failure_checks():
    from pathlib import Path
    text = (Path(__file__).parents[1]/'tools/run_kpack_strategy_campaign_box.sh').read_text()
    assert 'campaign_args+=(--point-filter "$POINT_FILTER")' in text
    assert 'campaign_args+=(--max-finalists "$MAX_FINALISTS")' in text
    assert 'set -euo pipefail' in text and '"${campaign_args[@]}"' in text
