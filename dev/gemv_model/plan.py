"""Bounded M1 candidates for the measured paired-model decode bottlenecks."""

from dataclasses import asdict, dataclass, replace

SCHEMA = "quactlize.model-gemv-reader.v1"
BASE_SOURCE = "4f181a071ebf3715f90b2898033497342f9af4ca"
BASE_ARTIFACT = "6a9b89a322f1ccd5cf1b724325294f7d2ae129b1"
BASE_EXECUTION = "0a60a15a1acadea67b17f5c5506ce3d4bd2c9d4b6647c86d1f83fc6a6e119c89"
BASE_FUSION = "b73e9baac1bec19d4f9d1b689ce830afb86df6ea44434a40a9e8f0c2df70057b"


@dataclass(frozen=True)
class Candidate:
    name: str
    columns: int = 4
    warps: int = 8
    values: int = 8
    split: int = 1
    variant: int = 3
    changes: int = 0
    hoist: bool = False
    fixed: bool = False
    direct_meta: bool = False

    @property
    def tile_n(self):
        return self.columns * self.values


@dataclass(frozen=True)
class Point:
    name: str
    q: int
    n: int  # Logical output N; paired weights contain 2*N columns.
    k: int
    compute: int = 0
    mode: int = 0
    channels: int = 1
    paired: bool = False
    tc: tuple = ()  # TM, TK, WN, DN, split. TN64, WM=TM, stages2, AP0.

    @property
    def experts(self):
        return 256 if self.mode else 1

    @property
    def physical_n(self):
        return self.n * (2 if self.paired else 1)

    @property
    def parent(self):
        tm, tk, wn, dn, _ = self.tc
        return dict(symbol=f"closure_q{self.q}_0_tm{tm}_tn64_tk{tk}_wn{wn}_dn{dn}",
                    qtype=self.q, route="sf-dense" if self.q == 8 else "fq-dense",
                    tm=tm, tn=64, tk=tk, wm=tm, wn=wn, stages=2, ap=0, dn=dn, persistent=-1)


POINTS = (
    Point("q4-paired-routed", 12, 512, 2048, 1, 2, 1, True),
    Point("q5-routed-down", 13, 2048, 512, 1, 2, 8),
    Point("q8-paired-shared", 8, 512, 2048, paired=True),
    Point("q8-shared-down", 8, 2048, 512),
    Point("q8-ssm-out", 8, 2048, 4096),
    Point("q8-qkv", 8, 8192, 2048, tc=(8, 64, 16, 32, 8)),
    Point("q8-attn-gate", 8, 4096, 2048, tc=(16, 64, 16, 64, 8)),
    Point("q6-output", 14, 248320, 2048, tc=(16, 128, 16, 64, 1)),
)


def candidates(p):
    if p.name == "q4-paired-routed":
        base = Candidate("clone")
        h32 = replace(base, name="h32", changes=1)
        return [base, h32, replace(h32, name="h32-unsigned", changes=3),
                replace(h32, name="h32-fixed", fixed=True),
                replace(h32, name="h32-c8p4", columns=8, values=4),
                replace(h32, name="h32-c8p4w4", columns=8, values=4, warps=4),
                replace(h32, name="h32-c8p4w4-fixed", columns=8, values=4, warps=4, fixed=True),
                replace(h32, name="tile16-w8", values=4),
                replace(h32, name="tile16-w4", values=4, warps=4)]
    if p.name == "q5-routed-down":
        base = Candidate("clone", warps=2, changes=3)
        return [base, replace(base, name="fixed", fixed=True),
                replace(base, name="p4", values=4),
                replace(base, name="p4-fixed", values=4, fixed=True),
                replace(base, name="p4w4", values=4, warps=4),
                replace(base, name="c8p4", columns=8, values=4),
                replace(base, name="c8p4-fixed", columns=8, values=4, fixed=True)]
    if p.name == "q8-paired-shared":
        base = Candidate("clone", variant=5)
        return [base, replace(base, name="hoist", hoist=True),
                replace(base, name="fixed", fixed=True),
                replace(base, name="hoist-fixed", hoist=True, fixed=True),
                replace(base, name="c8p4", columns=8, values=4),
                replace(base, name="c8p4-hoist", columns=8, values=4, hoist=True),
                replace(base, name="c8p4w4-hoist", columns=8, values=4, warps=4, hoist=True),
                replace(base, name="tile16-w4", values=4, warps=4),
                replace(base, name="tile16-w8", values=4),
                replace(base, name="tile16-w4-hoist", values=4, warps=4, hoist=True),
                replace(base, name="tile16-w8-hoist", values=4, hoist=True)]
    if p.name == "q6-output":
        base = Candidate("generic", columns=8, warps=4, values=4)
        return [base, replace(base, name="direct", direct_meta=True),
                replace(base, name="direct-fixed", direct_meta=True, fixed=True),
                replace(base, name="generic-c4p4", columns=4),
                replace(base, name="direct-c4p4", columns=4, direct_meta=True),
                replace(base, name="direct-c4p8", columns=4, values=8, direct_meta=True),
                replace(base, name="direct-c8p4w2", warps=2, direct_meta=True)]
    if p.name == "q8-shared-down":
        base = Candidate("clone", columns=8, warps=4, values=4, variant=5, hoist=True)
        return [base, replace(base, name="fixed", fixed=True),
                replace(base, name="no-hoist", hoist=False),
                replace(base, name="w2", warps=2),
                replace(base, name="w2-fixed", warps=2, fixed=True),
                replace(base, name="c4w4", columns=4),
                replace(base, name="c4w2", columns=4, warps=2)]
    base = Candidate("clone" if not p.tc else "generic-s8", columns=8, warps=4,
                     values=4, variant=5, split=8)
    return [base, replace(base, name="s8-fixed", fixed=True),
            replace(base, name="s4", split=4),
            replace(base, name="s2", split=2), replace(base, name="s1", split=1),
            replace(base, name="s2-hoist", split=2, hoist=True),
            replace(base, name="s1-hoist-fixed", split=1, hoist=True, fixed=True),
            replace(base, name="c4w4-s2", columns=4, split=2),
            replace(base, name="c4w8-s1", columns=4, warps=8, split=1)]


def inventory():
    return [dict(point=asdict(p), candidates=[asdict(c) for c in candidates(p)]) for p in POINTS]
