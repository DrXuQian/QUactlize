"""Production ABI/catalog support."""
import json
from pathlib import Path
from quactlize.dispatch.catalog_components import require
from quactlize.dispatch.table import Recipe
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from quactlize.dispatch.catalog_recipe import recipe as normalize
ROOT = Path(__file__).resolve().parents[2]


SEED = ROOT/'policies/prefill_refresh_seed.json'


def load_seed():
    value = json.loads(SEED.read_text())
    require(value['sha256'] == digest({k: v for k, v in value.items() if k != 'sha256'}),
            'prefill shortlist seal differs')
    require(value['policy_sha256'] == sha(ROOT/'policies/kpack_prefill_confirmation_v1.json'),
            'production prefill policy changed; review the shortlist again')
    return value


def request(w):
    return (w.qtype, 0 if w.operation == 'dense' else 2, w.n, w.k, w.experts,
            w.topk, w.channels, w.m, int(w.compute == 'bf16'))


def recipe_of(row):
    require(row['status'] == 0 and row['policy'] == 19, 'current C++ prefill selector missed')
    if row['kind'] == 3 and row['route'] == 2:
        return Recipe.make('full', 'full', dequant_config=row['dequant_config'], provider='bf16-default')
    require(row['kind'] == 0 and row['route'] in (0, 1), 'unexpected current prefill family')
    return normalize(dict(row['parent'], kind='tc', split=row['split'],
        grid_mode=row['grid_mode'], grid_b=row['grid_b']), row['route'], row['dequant_config'])


def same_recipe(a, b):
    # Symbols are aliases; all arithmetic/geometry/helper/scheduler fields stay.
    return (a.kind == b.kind and a.route == b.route and
            {k: v for k, v in a.parameters if k != 'symbol'} ==
            {k: v for k, v in b.parameters if k != 'symbol'})
