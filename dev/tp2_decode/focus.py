"""Three low-MBU shards: retained winners plus bounded occupancy neighbors."""
from dataclasses import asdict, replace
import json
from pathlib import Path

from dev.gemv_model.plan import Candidate, Point

SCHEMA = 'quactlize.tp2-decode-focus.v1'
BASE_SOURCE = 'a94d15c45cdc75347690fc00ae1ed0d846c4c506'
BASE_ARTIFACT = 'd8852d419a550623c4fd96c723f749c128798774'
BASE_EXECUTION = '2cb65b500a1bc872347a9cbd7eb82ed849ff350616900acaaa6c8529fef3f0a0'
BASE_FUSION = 'c576a9cc496278f7a78277747d3e64cafed669ea6fd0cbc8f9ab637858af5e49'
TOKEN_CONTROLS = tuple(range(1, 9))
NAMES = ('tp2-q8-attn-kv', 'tp2-q8-shared-down', 'tp2-q5-down')
_rows = json.loads((Path(__file__).resolve().parents[2] /
                   'policies/kpack_decode_winners_v1.json').read_text())['rows']
_winners = {w['point']['name']: w for w in _rows if w['point']['name'] in NAMES}
POINTS = tuple(Point(**_winners[name]['point']) for name in NAMES)


def candidates(p):
    base = replace(Candidate(**_winners[p.name]['body']), name='clone')
    if p.q == 13:
        # Short K already has many CTAs. Test fewer output values/registers,
        # not a gratuitous Split-K producer plus another reduction launch.
        return [base, replace(base, name='fixed', fixed=True),
                replace(base, name='p4', values=4),
                replace(base, name='p4-fixed', values=4, fixed=True),
                replace(base, name='c8p4', columns=8, values=4),
                replace(base, name='c8p4-fixed', columns=8, values=4, fixed=True)]
    return [base, replace(base, name='fixed', fixed=True),
            replace(base, name='c4w2p4', columns=4, warps=2, values=4),
            replace(base, name='c4w2p4-fixed', columns=4, warps=2, values=4, fixed=True),
            replace(base, name='c4w2p2', columns=4, warps=2, values=2),
            replace(base, name='c4w4p2', columns=4, warps=4, values=2),
            replace(base, name='c4w2p2-no-hoist', columns=4, warps=2, values=2, hoist=False),
            replace(base, name='c4w2p4-s2', columns=4, warps=2, values=4,
                    split=2, vector_reduce=True)]


def inventory():
    return [dict(point=asdict(p), candidates=[asdict(c) for c in candidates(p)]) for p in POINTS]
