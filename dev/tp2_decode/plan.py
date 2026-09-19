"""Actual TP2 shards plus old-shape controls; no Cartesian-product search."""
from dataclasses import asdict, replace
from dev.gemv_model.plan import Candidate, Point, POINTS as OLD_POINTS

SCHEMA = 'quactlize.tp2-fastpaths.v1'
BASE_SOURCE = 'c05900e2d0fe691981ca6f768e72547536d4a588'
BASE_ARTIFACT = '006aa757f5b995ae4d8eac772cc2106782e14e90'
BASE_EXECUTION = 'b1816dec751969548326ace1793fe04fb7c15f0e581aecb0d59f952526fb2bca'
BASE_FUSION = '60235a0d49a57383a0b2c1ba3a5c31712fc2081c4763d789a13fd374e4643031'
TOKEN_CONTROLS = tuple(range(1, 9))

PRIORITY = (
    Point('tp2-q8-qkv', 8, 6144, 3072, tc=(16,64,16,64,8)),
    Point('tp2-q8-attn-gate', 8, 4096, 3072, tc=(16,64,16,64,8)),
    Point('tp2-q8-attn-q', 8, 8192, 3072, tc=(8,64,16,32,8)),
    Point('tp2-q8-out', 8, 3072, 4096),
    Point('tp2-q8-shared-projection', 8, 1024, 3072),
    Point('tp2-q8-shared-down', 8, 3072, 1024),
    Point('tp2-q8-attn-kv', 8, 256, 3072),
    Point('tp2-q6-head', 14, 124160, 3072, tc=(16,128,16,64,1)),
    Point('tp2-q4-gate-up', 12, 1024, 3072, 1, 2, 1),
    Point('tp2-q5-down', 13, 3072, 512, 1, 2, 8),
)
PAIRED = (Point('tp2-q8-paired-shared', 8, 1024, 3072, paired=True),
          Point('tp2-q4-paired-routed', 12, 512, 3072, 1, 2, 1, True))
CONTROLS = OLD_POINTS[:5]
POINTS = PRIORITY + PAIRED + CONTROLS


def candidates(p):
    if p.paired:
        old = p in CONTROLS
        base = Candidate('clone', variant=5 if p.q==8 else 3,
                         values=4 if old and p.q==8 else 8,
                         changes=1 if old and p.q==12 else 0,
                         fixed=old and p.q==12, hoist=old and p.q==8)
        vector = replace(base, values=4, fixed=False, hoist=p.q==8)
        return [base, replace(base, name='fixed', fixed=True),
                replace(vector, name='tile16'), replace(vector, name='tile16-fixed', fixed=True),
                replace(vector, name='tile16-w4', warps=4)]
    if p.q in (12, 13):
        base = Candidate('clone', values=4 if p.q==12 else 8, warps=4 if p.q==12 else 2,
                         changes=3 if p in CONTROLS else 0, fixed=p in CONTROLS)
        fold = replace(base, changes=3)
        return [base, replace(fold, name='unsigned-fold', fixed=False),
                replace(fold, name='unsigned-fold-fixed', fixed=True),
                replace(fold, name='c8p4', columns=8, values=4, fixed=False)]
    if p.q==14:
        base = Candidate('generic', columns=8, warps=4, values=4)
        return [base, replace(base, name='direct-meta', direct_meta=True),
                replace(base, name='direct-meta-fixed', direct_meta=True, fixed=True),
                replace(base, name='direct-meta-c4', columns=4, direct_meta=True)]
    if p.name=='q8-ssm-out':
        base = Candidate('clone', columns=8, warps=4, values=4, variant=5, split=8, fixed=True)
    elif p.name=='q8-shared-down':
        base = Candidate('clone', columns=8, warps=4, values=4, variant=5, hoist=True)
    else:
        base = Candidate('clone' if not p.tc else 'generic', variant=1,
                         warps=2 if p.name=='tp2-q8-shared-down' else 8,
                         values=4 if p.name=='tp2-q8-out' else 2)
    vector = Candidate('vector', columns=8, warps=4, values=4, variant=5)
    return [base, vector, replace(vector, name='hoist', hoist=True),
            replace(vector, name='hoist-fixed', hoist=True, fixed=True),
            replace(vector, name='c4-hoist', columns=4, warps=8, hoist=True),
            replace(vector, name='s2-vector', split=2, vector_reduce=True),
            replace(vector, name='s4-scalar', split=4),
            replace(vector, name='s4-vector', split=4, vector_reduce=True),
            replace(vector, name='s8-vector', split=8, vector_reduce=True)]


def inventory():
    return [dict(point=asdict(p), candidates=[asdict(c) for c in candidates(p)]) for p in POINTS]
