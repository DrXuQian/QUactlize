"""Production ABI/catalog support."""
from quactlize.dispatch.table import Recipe
def recipe(config, route=None, dequant=-1):
    c = dict(config)
    kind = c.pop('kind', 'tc')
    if kind != 'tc':
        return Recipe.make(kind, 'fq', **c)
    r = c.pop('route', 0) if route is None else route
    if isinstance(r, str):
        r = int(r.startswith('sf'))
    c['persistent'] = c.pop('parent_persistent', c.get('persistent', -1))
    if isinstance(c.get('grid_mode'), str):
        c['grid_mode'] = {'implicit': 0, 'ordinary': 0, 'capacity': 2, 'balanced': 3}[c['grid_mode']]
    # Execution identity, not reported occupancy/shared-memory or source labels.
    fields = ('symbol', 'qtype', 'tm', 'tn', 'tk', 'wm', 'wn', 'stages', 'ap', 'dn',
              'persistent', 'split', 'grid_mode', 'grid_b')
    return Recipe.make('tc', ('fq', 'sf')[r % 2],
                       **{k: c[k] for k in fields if k in c}, dequant_config=dequant)
