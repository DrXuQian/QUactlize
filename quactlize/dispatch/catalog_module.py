"""Sealed production catalog validation/generation; no device experiments."""
from quactlize.dispatch.catalog_parent import parent


def module_spec(w, r):
    if r.kind != 'tc':
        raise ValueError('resource probe requires a TC recipe')
    legacy = dict(r.parameters).get('legacy_adapters', False)
    spec = dict(parent=parent(r,w), compute=w.compute,
                endpoint='grouped' if w.operation!='dense' else
                         'native' if w.compute=='f16' and (w.m>8 or legacy) else 'typed')
    if legacy:
        if w.operation!='dense' or w.compute!='f16' or r.route!='fq':
            raise ValueError('legacy flat adapters require dense F16 FQ')
        spec['legacy_adapters'] = True
    return spec
