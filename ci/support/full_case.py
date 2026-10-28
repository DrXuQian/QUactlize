"""Product correctness-test support."""
from dataclasses import asdict
import json
from quactlize.dispatch.table import Recipe
def full_case(dequant, io, *, dequant_first=False):
    if type(dequant_first) is not bool:
        raise ValueError('invalid Full stage order')
    options = dict(dequant_variant=dequant, vector_io=io)
    recipe = Recipe.make('full', 'full', provider='typed-fp16', **options,
                         **(dict(dequant_first=True) if dequant_first else {}))
    return dict(arm='full-fp16', helper_mode='experimental', helper_options=options,
                recipe=json.loads(json.dumps(asdict(recipe))))
