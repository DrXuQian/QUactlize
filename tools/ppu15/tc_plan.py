"""Bounded TC correctness cohort, not a production policy or timing sweep."""
MS = (1, 8, 9, 16, 65, 128)
SPLITS = (1, 2)
NK = (256, 1024)
EXPERTS = 3


def row_counts(m, replay):
    """Fixed-capacity replay: skewed -> balanced -> different skewed expert."""
    if m < 1 or replay not in (0, 1, 2):
        raise ValueError('invalid grouped replay fixture')
    if replay == 1:
        return tuple(m // EXPERTS + int(e < m % EXPERTS) for e in range(EXPERTS))
    return (m, 0, 0) if replay == 0 else (0, 0, m)


def parents(cohort='dense'):
    if cohort == 'dense':
        pairs = ((12, 'fq-dense'), (12, 'sf-dense'), (14, 'fq-dense'),
                 (14, 'sf-dense'), (8, 'sf-dense'))
    elif cohort == 'supplement':
        pairs = [(q, route) for q in (10, 11, 13) for route in ('fq-dense', 'sf-dense')]
        pairs += [(q, route) for q in (10, 11, 12, 13, 14)
                  for route in ('fq-grouped', 'sf-grouped')]
        pairs += [(8, 'sf-grouped')]
    else:
        raise ValueError('unknown PPU15 TC cohort')
    return [dict(qtype=q, route=route, tm=64, tn=64, tk=256,
                 wm=16, wn=16, stages=2, ap=0, dn=64,
                 persistent=(1 if route == 'fq-grouped' else -1),
                 symbol=f'zw890_q{q}_{route.replace("-", "_")}'
                        f'{"_bf16" if route.endswith("grouped") else ""}_m64n64k256')
            for q, route in pairs]


def case_keys(parent):
    return {(parent['symbol'], m, split) for m in MS for split in SPLITS}


def verdict(parent, rows, rc):
    keys = [(r.get('parent'), r.get('m'), r.get('split')) for r in rows]
    expected = case_keys(parent)
    passed = sum(r.get('status') == 'PASS' and k in expected for r, k in zip(rows, keys))
    ok = rc == 0 and set(keys) == expected and len(keys) == passed == len(expected)
    return dict(status='PASS' if ok else 'FAIL', expected=len(expected), passed=passed, runner_rc=rc)


def plan(cohort='dense'):
    return dict(target='ppu_15', cohort=cohort, parents=parents(cohort), ms=MS, splits=SPLITS, n=NK[0], k=NK[1],
                expected=len(parents(cohort)) * len(MS) * len(SPLITS),
                scope='EXISTING_FQ_SF_COLLECTIVES_NATIVE_PPU15_CORRECTNESS_ONLY',
                precision='DENSE_F16_GROUPED_BF16_A_AND_C_FP32_ACCUMULATION',
                m_semantics='dense rows or concatenated grouped rows; NOT logical MoE tokens',
                production_enabled=False, performance_admitted=False,
                remaining=(['GROUPED_BF16', 'Q2_Q3_Q5_TC'] if cohort == 'dense' else []) +
                          ['M8_PACKED_A_RECIPE_TRANSLATION', 'PRODUCTION_SF_EXPANSION',
                           'INDEXED_ROUTER_FFN', 'FULL_LIBRARY_SELECTION', 'MODEL_INTEGRATION', 'PERFORMANCE'])
