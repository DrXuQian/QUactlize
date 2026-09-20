"""Matched Q4 gate/up+SwiGLU chains, never a production selector."""
from dataclasses import asdict

from dev.gemv_model.plan import Candidate, Point

SCHEMA = 'quactlize.tp2-paired-chain.v1'
BASE_SOURCE = '0aeffd7e977054bea05a369bc5c1a23e7380b588'
BASE_ARTIFACT = '46727e96de92c6a7da15aa12f066a31f1bed8673'
BASE_EXECUTION = 'f7a5a5c768b5a94a4c8c613f55e83bd592018b781a9df00a9134cb61698ba213'
BASE_FUSION = '56b7548a9a13f8066f36fdc272bd61271af4d27a5733c52d1f9e93feb93316d9'
POINTS = (Point('tp2-chain', 12, 512, 3072, 1, 2, 1, True),
          Point('model35-chain', 12, 512, 2048, 1, 2, 1, True))
ARMS = ('unfused', 'paired-shipping', 'paired-0', 'paired-1')
TOKEN_CONTROLS = tuple(range(1, 9))
ROUNDS, SAMPLES = 6, 15


def candidates(p):
    if p not in POINTS:
        raise ValueError('unknown paired-chain point')
    # Keep the previous paired winner. The second arm transfers the current
    # merged projection's reader, but retains paired sequential final sums.
    return [Candidate('historical-paired', changes=0 if p.k == 3072 else 1, fixed=True),
            Candidate('projection-reader-paired', warps=4, values=4,
                      changes=3 if p.k == 3072 else 1, fixed=True)]


def projection_config(q):
    if q == 12:
        return dict(variant=3, columns=4, warps=4, values=4, split=1)
    if q == 13:
        return dict(variant=3, columns=4, warps=2, values=8, split=1)
    raise ValueError('only the frozen Q4/Q5 chain is in scope')


def stages(arm):
    if arm not in ARMS:
        raise ValueError('unknown chain arm')
    middle = ('projection', 'activation') if arm == 'unfused' else ('paired-projection-activation',)
    return ('prepare-snapshot', *middle, 'down', 'weighted-finish')


def inventory():
    return [dict(point=asdict(p), candidates=[asdict(c) for c in candidates(p)]) for p in POINTS]
