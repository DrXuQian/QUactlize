"""Product correctness-test support; performance runners are on experiments/ppu-tuning."""
from dataclasses import asdict
import json
from ci.support.full_case import full_case
from ci.support.eligibility import rejection
from quactlize.dispatch.dense_prefill import bucket_key
from quactlize.dispatch.table import Recipe
def execution_case(value):
    if value['route'] == 2:
        return full_case(value['dequant'], value['io'], dequant_first=value.get('dequant_first',False))
    recipe = Recipe.make('tc', 'sf' if value['route'] else 'fq',
                         **dict(value['tc'], dequant_config=value['dequant']))
    return dict(arm=recipe.route, helper_mode='cohort-current',
                helper_options=dict(vector_io=value['io']),
                recipe=json.loads(json.dumps(asdict(recipe))))


def legal(value, w):
    if value['route'] == 2:
        return True
    return rejection(Recipe.from_dict(execution_case(value)['recipe']), w) is None


def policy_lookup(policy, w):
    # Match the C++ selector, while still accepting a frozen older catalog in
    # historical experiment replays. A split M bucket is not the whole octave.
    intervals = policy.get('bucket_policy') == 'ACTIVE_MEANS_MNK_INTERVALS_V2'
    if (not (9 if intervals else 128)<=w.m<=8192 or w.operation!='dense' or
            w.compute!='f16' or w.storage!='f32' or w.input_stride!=w.k or w.output_stride!=w.n or
            w.m*w.n>2147483647):
        return 'OUTSIDE_NEW_TABLE', None
    shape = [w.qtype, w.m, w.n, w.k]
    for point in policy['points']:
        if point['shape'] == shape:
            return 'EXACT', point['winner']
    if not intervals and w.m>2048:
        return 'OUTSIDE_NEW_TABLE', None
    for bucket in policy['buckets']:
        if (list(bucket_key(shape)) == bucket['key'] and
                bucket.get('m_bounds',[0,8192])[0] <= w.m <= bucket.get('m_bounds',[0,8192])[1] and
                bucket['bounds'][0][0] <= w.n <= bucket['bounds'][0][1] and
                bucket['bounds'][1][0] <= w.k <= bucket['bounds'][1][1] and
                legal(policy['recipes'][bucket['recipe']], w)):
            return 'BUCKET_PREDICTION', bucket['recipe']
    return 'OUTSIDE_NEW_TABLE', None
