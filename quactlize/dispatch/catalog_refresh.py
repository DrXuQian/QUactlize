"""Sealed production catalog validation/generation; no device experiments."""
import math


from quactlize.dispatch.catalog_components import require, semantic


from quactlize.runtime.tuning import digest


SCHEMA = 'quactlize.decode-refresh-selection.v1'


ARCHIVES = {
    'parent': 'c7cf6c49661643ec90456964d06b7f48f2fdb8e63c4ccdd86a4d2368276139bb',
    'retry': '7a7e02783e670c4a7674c1ee5b43296e1f1b0114274370036cddb4b865710e3e',
    'prior_pass': '960595f5c597ce7ffb84c3f3d047e48beb3eef2fd7c381a04e5cb04c4dd5b49c',
}


SCOPE = 'ALTERNATING_ISOLATED_COMPONENT_SUM_NOT_MODEL_TPOT'


def identity(candidate):
    return digest(dict(config=semantic(candidate['config']),
                       dequant_config=candidate['dequant_config'],
                       implementation=candidate['implementation']))


def allowed(candidate, original):
    return (candidate['median_us'] <= original['median_us'] and
            candidate['spread_pct'] <= 5 and original['spread_pct'] <= 5 and
            sum(x <= y for x, y in zip(candidate['rounds'], original['rounds'])) >= 4)


def verify(table):
    require(table['schema'] == SCHEMA and table['archives'] == ARCHIVES and
            table['evidence_sha256'] == digest({k: v for k, v in table.items() if k != 'evidence_sha256'}),
            'refresh table identity differs')
    require(table['scope'] == SCOPE and not table['active'] and not table['production_changed'] and
            not table['runtime_router_readback'] and not table['frequency_normalization'] and
            not table['big_m_changed'], 'refresh scope differs')
    seen, points = set(), set()
    for row in table['rows']:
        k = tuple(row['key'])
        require(len(k) == 9 and k not in seen and 1 <= k[7] <= 8 and
                k[0] in (8, 10, 11, 12, 13, 14) and
                ((k[1] == 0 and k[4:7] == (1, 1, 1) and k[-1] == 0) or
                 (k[1] == 2 and k[4:6] == (256, 8) and k[6] in (1, 8) and k[-1] == 1)),
                'invalid/duplicate runtime key')
        seen.add(k)
        changed = (semantic(row['config']) != semantic(row['original_config']) or
                   row['dequant_config'] != row['original_dequant_config'] or
                   row['implementation'] != dict(kind='shipping'))
        require(row['changed'] == changed and row['status'] ==
                ('CONFIRMED_REPLACEMENT' if changed else 'RETAIN_INCUMBENT'),
                'retained/replacement identity differs')
        body = row['implementation']
        if body['kind'] != 'shipping':
            c = row['config']
            require(body['kind'] == 'q8-address' and type(body.get('body')) is int and
                    body['body'] in (0, 1, 3, 4, 5) and k[0:2] == (8, 0) and
                    c['kind'] == 'simt' and c['variant'] == 5 and c['values'] == 4 and
                    c['split'] == 1 and (c['columns'], c['warps']) in ((4, 8), (8, 4)),
                    'unmeasured Q8 implementation domain')
        for p in row['profiles']:
            require(p['point'] not in points, 'duplicate profile point')
            points.add(p['point'])
            require(math.isfinite(p['numerical_error']) and 0 <= p['numerical_error'] < .005 and
                    all(math.isfinite(p[f]) and p[f] > 0 for f in ('original_us', 'selected_us')) and
                    len(p['original_rounds_us']) == len(p['selected_rounds_us']) == 6,
                    'invalid profile proof')
            if row['changed']:
                require(p['raw_status'] == 'PASS' and p['selected_us'] <= p['original_us'] and
                        p['original_spread_pct'] <= 5 and p['selected_spread_pct'] <= 5 and
                        sum(x <= y for x, y in zip(p['selected_rounds_us'], p['original_rounds_us'])) >= 4,
                        'replacement lacks paired evidence')
    require(len(points) == table['summary']['points'] and len(seen) == table['summary']['host_keys'],
            'table coverage differs')
    require(len(points) == 1114 and len(seen) == 930 and
            table['summary']['changed'] == sum(r['changed'] for r in table['rows']) and
            table['summary']['retained'] == sum(not r['changed'] for r in table['rows']) and
            {p['point'] for p in table['pending']} == {p['point'] for r in table['rows']
                                                     for p in r['profiles'] if p['raw_status'] != 'PASS'},
            'summary or pending scope differs')
