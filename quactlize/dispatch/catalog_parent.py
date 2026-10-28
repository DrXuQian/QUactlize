"""Sealed production catalog validation/generation; no device experiments."""
def parent(recipe, workload):
    p = dict(recipe.parameters)
    result = {k: p[k] for k in ('symbol', 'tm', 'tn', 'tk', 'wm', 'wn', 'stages', 'ap', 'dn')} | dict(
        qtype=p.get('qtype', workload.qtype),
        route=recipe.route + ('-dense' if workload.operation == 'dense' else '-grouped'),
        persistent=p.get('persistent', 0 if workload.operation != 'dense' and recipe.route == 'fq' else -1))
    if 'dense_reducer' in p:result['dense_reducer']=p['dense_reducer']
    return result
