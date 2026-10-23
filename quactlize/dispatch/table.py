"""Data-only selection records shared by import, offline audits and replay.

No GPU, compiler, torch or model-loader dependency. A bucket proposes recipes;
it never upgrades another shape/precision/router's measurement to exact.
"""
from dataclasses import asdict, dataclass, field
from functools import cached_property
import math

from quactlize.runtime.tuning import digest

QTYPES = (8, 10, 11, 12, 13, 14)
SCOPES = ('COMPLETE_CALL', 'RESIDENT_GEMM', 'COMPONENT_SUM', 'PROPOSAL')


@dataclass(frozen=True)
class Workload:
    qtype: int
    n: int
    k: int
    m: int
    operation: str = 'dense'
    compute: str = 'f16'
    storage: str = 'f32'
    experts: int = 1
    topk: int = 1
    channels: int = 1
    input_stride: int = 0
    output_stride: int = 0
    router: str = ''
    fusion: str = 'none'

    def __post_init__(self):
        values = (self.qtype, self.n, self.k, self.m, self.experts, self.topk, self.channels)
        if any(type(v) is not int or not 0 < v <= 2147483647 for v in values):
            raise ValueError('workload dimensions must be positive int32')
        if (self.qtype not in QTYPES or self.n % 256 or
                self.k % (512 if self.qtype in (11, 14) else 256) or
                self.compute not in ('f16', 'bf16') or self.storage not in ('f16', 'bf16', 'f32') or
                self.operation not in ('dense', 'indexed', 'grouped') or
                not isinstance(self.router, str) or not isinstance(self.fusion, str)):
            raise ValueError('unsupported workload contract')
        if self.operation == 'dense' and (self.experts, self.topk, self.channels, self.router) != (1, 1, 1, ''):
            raise ValueError('dense request has grouped fields')
        if self.topk > self.experts or self.channels not in (1, self.topk):
            raise ValueError('invalid indexed channels/top-k')
        for name, minimum in (('input_stride', self.k), ('output_stride', self.n)):
            value = getattr(self, name)
            if type(value) is not int or (value != 0 and value < minimum):
                raise ValueError('invalid storage stride')
            if value == 0:
                object.__setattr__(self, name, minimum)

    @property
    def mapping(self):
        return {8: 0x51384b5032540001, 12: 0x51344b5034540001}.get(self.qtype, 0x514b504b54000001)

    @cached_property
    def key(self):
        return digest(asdict(self) | {'mapping': self.mapping})

    @property
    def family(self):
        # Router changes can reuse proposals, NEVER an exact observation.
        return (self.qtype, self.operation, self.compute, self.storage,
                self.experts, self.topk, self.channels, self.fusion,
                self.input_stride == self.k, self.output_stride == self.n)

    @property
    def m_bucket(self):
        return ('decode', self.m) if self.m <= 8 else ('prefill', self.m.bit_length() - 1)


@dataclass(frozen=True)
class Recipe:
    kind: str
    route: str
    parameters: tuple

    def __post_init__(self):
        if self.kind not in ('simt', 'q4', 'tc', 'full') or self.route not in ('fq', 'sf', 'full'):
            raise ValueError('invalid recipe kind/route')
        if (self.kind == 'full') != (self.route == 'full'):
            raise ValueError('full route/kind mismatch')
        if not isinstance(self.parameters, tuple) or any(
                not isinstance(pair, tuple) or len(pair) != 2 or
                not isinstance(pair[0], str) or type(pair[1]) not in (int, str, bool)
                for pair in self.parameters):
            raise ValueError('recipe parameters must be scalar immutable pairs')
        if len(dict(self.parameters)) != len(self.parameters):
            raise ValueError('duplicate recipe parameter')
        object.__setattr__(self, 'parameters', tuple(sorted(self.parameters)))

    @classmethod
    def make(cls, kind, route, **parameters):
        return cls(kind, route, tuple(parameters.items()))

    @classmethod
    def from_dict(cls, value):
        return cls(value['kind'], value['route'], tuple(tuple(p) for p in value['parameters']))

    @cached_property
    def key(self):
        return digest(asdict(self))

    def eligible(self, w):
        """Cheap proposal pruning. The real backend MUST also validate resources."""
        p = dict(self.parameters)
        if self.kind == 'full':
            if p.get('provider') == 'typed-fp16':
                return w.m >= 128 and w.operation == 'dense' and w.compute == 'f16'
            return w.m >= 128 and w.qtype != 8
        if self.kind in ('simt', 'q4'):
            return w.m <= 8 and w.operation != 'grouped' and (self.kind != 'q4' or w.qtype == 12)
        if any(type(p.get(k)) is not int or p[k] <= 0 for k in ('tm', 'tn', 'tk', 'wm', 'wn', 'stages', 'split')):
            return False
        if p.get('qtype', w.qtype) != w.qtype or (w.qtype == 8 and self.route != 'sf'):
            return False
        tk, split = p['tk'], p['split']
        return (w.k % (tk * split) == 0 and w.k // (tk * split) >= p['stages'] - 1 and
                p['tm'] % p['wm'] == 0 and p['tn'] % p['wn'] == 0 and
                (not p.get('ap') or (w.operation == 'dense' and w.m == 1 and w.qtype in (10, 12))) and
                (w.operation != 'dense' or p['tm'] != 8 or w.m <= (64 if self.route == 'fq' else 8)))


@dataclass(frozen=True)
class Observation:
    workload: Workload
    recipe: Recipe
    source: str
    scope: str
    priority: int
    status: str
    evidence: str
    us: float | None = None
    details: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if self.scope not in SCOPES or not self.source or not self.evidence:
            raise ValueError('observation lacks provenance/scope')
        if self.us is not None and (not math.isfinite(self.us) or self.us <= 0):
            raise ValueError('invalid measured cost')


class StrategyTable:
    """One evidence catalog, with separate exact and family indices.

    Priority is an explicit import/review decision, not a comparison of raw
    microseconds across unrelated cohorts. All old records remain candidates.
    Default production selection is not modified by loading this catalog.
    """
    schema = 'quactlize.strategy-table.v1'

    def __init__(self, observations=()):
        self.observations = tuple(observations)
        self.exact, self.families = {}, {}
        for row in self.observations:
            self.exact.setdefault(row.workload.key, []).append(row)
            self.families.setdefault(row.workload.family, []).append(row)
        self.identity = digest([asdict(row) for row in self.observations])

    def lookup(self, request):
        rows = [r for r in self.exact.get(request.key, ()) if r.scope != 'PROPOSAL' and r.recipe.eligible(request)]
        return max(rows, key=lambda r: r.priority, default=None)

    def shortlist(self, request, incumbent, limit=8):
        """Explicit reference/control inventory, NOT blind fallback rules.

        Offline candidate generation uses dev.strategy.prescreen instead.
        This method never compiles, benchmarks or writes a selection cache.
        """
        if not 1 <= limit <= 32 or not incumbent.eligible(request):
            raise ValueError('bounded pool requires a legal incumbent')
        result, seen = [incumbent], {incumbent.key}
        for row in sorted(self.exact.get(request.key, ()), key=lambda r: -r.priority):
            if row.scope != 'PROPOSAL' and row.recipe.eligible(request) and row.recipe.key not in seen:
                result.append(row.recipe)
                seen.add(row.recipe.key)
        if len(result) > limit:
            raise ValueError('candidate budget would drop an existing exact winner')
        rows = []
        for row in self.families.get(request.family, ()):
            w = row.workload
            if (not row.recipe.eligible(request) or
                    max(w.n, request.n) > 2 * min(w.n, request.n) or
                    max(w.k, request.k) > 2 * min(w.k, request.k) or
                    (w.m <= 8) != (request.m <= 8)):
                continue
            distance = abs(math.log2(w.n / request.n)) + abs(math.log2(w.k / request.k))
            distance += 2 * abs(math.log2(w.m / request.m))
            rows.append(((w.key != request.key, distance, -row.priority, row.recipe.key), row))
        for _, row in sorted(rows, key=lambda item: item[0]):
            if len(result) == limit:
                break
            if row.recipe.key not in seen:
                result.append(row.recipe)
                seen.add(row.recipe.key)
        return tuple(result)

    def as_dict(self):
        return dict(schema=self.schema, identity=self.identity,
                    observations=[asdict(r) for r in self.observations])

    @classmethod
    def from_dict(cls, value):
        if value.get('schema') != cls.schema:
            raise ValueError('unknown strategy table')
        rows = []
        for raw in value['observations']:
            raw = dict(raw)
            raw['workload'] = Workload(**raw['workload'])
            raw['recipe'] = Recipe.from_dict(raw['recipe'])
            rows.append(Observation(**raw))
        table = cls(rows)
        if table.identity != value['identity']:
            raise ValueError('strategy table digest differs')
        return table
