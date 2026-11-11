"""Bounded TC correctness cohort, not a production policy or timing sweep."""
MS = (1, 8, 9, 16, 65, 128)
SPLITS = (1, 2)
NK = (256, 1024)


def parents():
    return [dict(qtype=q, route=route, tm=64, tn=64, tk=256,
                 wm=16, wn=16, stages=2, ap=0, dn=64, persistent=-1,
                 symbol=f'zw890_q{q}_{route.replace("-", "_")}_m64n64k256')
            for q, route in ((12, 'fq-dense'), (12, 'sf-dense'),
                             (14, 'fq-dense'), (14, 'sf-dense'), (8, 'sf-dense'))]


def case_keys(parent):
    return {(parent['symbol'], m, split) for m in MS for split in SPLITS}


def verdict(parent, rows, rc):
    keys = [(r.get('parent'), r.get('m'), r.get('split')) for r in rows]
    expected = case_keys(parent)
    passed = sum(r.get('status') == 'PASS' and k in expected for r, k in zip(rows, keys))
    ok = rc == 0 and set(keys) == expected and len(keys) == passed == len(expected)
    return dict(status='PASS' if ok else 'FAIL', expected=len(expected), passed=passed, runner_rc=rc)


def plan():
    return dict(target='ppu_15', parents=parents(), ms=MS, splits=SPLITS, n=NK[0], k=NK[1],
                expected=len(parents()) * len(MS) * len(SPLITS),
                scope='EXISTING_DENSE_FQ_SF_COLLECTIVES_NATIVE_PPU15_CORRECTNESS_ONLY',
                precision='F16_A_AND_C_FP32_ACCUMULATION',
                production_enabled=False, performance_admitted=False,
                remaining=['GROUPED_BF16', 'Q2_Q3_Q5_TC', 'M8_PACKED_A_RECIPE_TRANSLATION',
                           'FULL_LIBRARY_SELECTION', 'MODEL_INTEGRATION', 'PERFORMANCE'])
