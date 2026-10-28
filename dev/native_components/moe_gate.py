"""Actual-model indexed selected-only gate; no historical timing or new search."""
from dev.strategy import moe_model_plan as models
from quactlize.runtime.tuning import digest

COHORT = 'moe-native-models'


def plan(output, inventory):
    value = models.make_plan(output, inventory, selected_only=True)
    tc, vectors = {}, {}
    for point in value['points']:
        original, = [a for a in point['arms'] if a['id'] == 'original']
        point['arms'] = [original]
        # The selected helper is part of this exact recipe, not a fresh helper sweep.
        p = dict(original['recipe']['parameters'])
        point['sf_configs'] = [max(0,p.get('dequant_config',-1))] if (
            original['recipe']['route'] == 'sf' and point['workload']['qtype'] != 8) else []
        (tc if original['recipe']['kind'] == 'tc' else vectors)[original['module']] = original['spec']
    value.update(cohort=COHORT, tc=tc, vectors=vectors, native_comparison=True,
        remaining=['LARGE_M', 'BF16_STORAGE_ENDPOINTS', 'FUSED_GATE_UP_SWIGLU', 'MODEL_AB'])
    value['counts'].update(arms=len(value['points']), tc_modules=len(tc), vector_modules=len(vectors))
    value['sha256'] = digest({k:v for k,v in value.items() if k != 'sha256'})
    return value


def check_fixture(point, data, *, native):
    from dev.strategy.fixture_identity import expected
    from quactlize.dispatch.table import Workload
    if point['workload']['operation'] != 'indexed':
        return
    value = data if native else data['arms']['original']
    if (value.get('timing_input_repeat') != 1 or
            value.get('timed_fixture') != expected(Workload(**point['workload']),1)):
        raise ValueError('MoE native/candidate timed A or expert IDs differ')
