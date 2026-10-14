"""One-axis template neighbors, independent of the held-out answer.

No N/K exception, latency lookup, Cartesian product or recursive expansion.
Seeds must already come from the blind prescreen. Compiler/shape/precision
legality is checked by the caller before a generated template is admitted.
"""
from functools import lru_cache

from quactlize.dispatch.table import Recipe


MODES = ('none', 'warps', 'simt', 'local')


def parent_name(p, route):
    return ('rule_' + route + '_' + '_'.join(k + str(p.get(k, 0)) for k in
            ('qtype','tm','tn','tk','wm','wn','stages','ap','dn','persistent'))).replace('-', 'm')


def adjacent(value, values):
    if value not in values:
        return ()
    i = values.index(value)
    return tuple(v for j, v in enumerate(values) if abs(j-i) == 1)


@lru_cache(maxsize=32768)
def one_axis(recipe, mode):
    if mode not in MODES:
        raise ValueError('unknown neighbor mode')
    p = dict(recipe.parameters)
    axes = {}
    if mode != 'none' and recipe.kind == 'simt':
        # Threads per CTA are a runtime-image axis, not a new TC compile.
        axes['warps'] = (2, 4, 8)
        if mode in ('simt', 'local'):
            axes.update(columns=(4, 8), values=(2, 4, 8),
                        variant=tuple(p['variant'] ^ bit for bit in (1, 2, 4)))
    elif mode == 'local' and recipe.kind == 'tc':
        # One small step, except for the declared operand delivery choices.
        for field, domain in (
                ('tm',(8,16,32,64,128,256)), ('tn',(16,32,64,128,256)),
                ('tk',(64,128,256)), ('wm',(8,16,32,64,128)),
                ('wn',(16,32,64,128)), ('stages',(2,3,4,6,8,12))):
            axes[field] = adjacent(p[field], domain)
        axes['dn'] = (16, 32, 64)
    # Q4's shape-specialized reader requires its own explicit carrier. Never
    # pretend arbitrary Q4 variants can run through the selected-only ABI.
    out = []
    for axis, values in axes.items():
        for value in values:
            if value == p[axis]:
                continue
            changed = p | {axis: value}
            if recipe.kind == 'tc':
                changed['symbol'] = parent_name(changed, recipe.route)
            out.append((Recipe.make(recipe.kind, recipe.route, **changed), axis))
    return tuple(out)
