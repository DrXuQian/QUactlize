"""Q4 decode-only incremental sweep: real families, SIMT and retained TC winners."""
from dataclasses import asdict, dataclass
from functools import lru_cache
import csv
import json
from pathlib import Path
import re

import numpy as np

from dev.gemv_ppu import moe_s1, smallm, moe_compare
from tools.plan_fq_kquant_kpack_perf import source_families
from tools.gguf_internal_shape_inventory import _routing_fixture, _splitmix64, ROUTING_SEED, _identity_sha256
from quactlize.runtime.compiler import sha, validate_parent
from quactlize.runtime.candidates import PARENT_FIELDS
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'quactlize.q4-decode-route-sweep.v1'
BUNDLE = ROOT/'prebuilt/ppu0010/q4-decode-sweep-v1'
DENSE = tuple(sorted(set(source_families()[0]) | set(smallm.SHAPES)))
GROUPED = moe_s1.SHAPES
MS = tuple(range(1, 9))
DENSE_REVIEW = ROOT/'docs/measurements/q4_smallm_20260913/summary.tsv'
MOE_REVIEW = ROOT/'docs/measurements/q4_moe_compare_20260913/summary.tsv'


@dataclass(frozen=True, order=True)
class Simt:
    reader: int
    variant: int
    warps: int
    values: int
    columns: int = 4

    @property
    def key(self):
        return f'r{self.reader}-v{self.variant}-w{self.warps}-p{self.values}-c{self.columns}'

    @property
    def tile_n(self):
        return 8 if self.reader == 0 else self.columns*self.values

    def geometry(self, n, rows):
        return dict(grid=rows*(n//self.tile_n), threads=self.warps*32,
                    shared_bytes=self.warps*self.tile_n*4, tile_n=self.tile_n,
                    rows_per_cta=1, split=1)


def old_dense_recipe(n, k, key):
    c = smallm.lookup(n, k, key)
    if c.family == 'meta':
        h, loading, amode, w = c.recipe
        if (loading, amode) != (1, 1):
            raise ValueError('unported historical META loading')
        return Simt(0, h, w, 1)
    if c.family == 'medium':
        p, w, reduction, unsigned = c.recipe
        return Simt(1, reduction+2*unsigned, w, p)
    v, cols, w, p = c.recipe
    return Simt(2, v, w, p, cols)


@lru_cache(None)
def simt_inventory(n, k):
    if (n, k) not in set(DENSE) | set(GROUPED):
        raise ValueError('unlisted weight family')
    pool = {Simt(**asdict(r)) for r in moe_s1.inventory()}
    # Transfer the admitted C8/P4 transaction-packing alternative to large N;
    # C8/P8 is NOT legal for the warp-cooperative metadata reader (64 lanes).
    if n >= 4096:
        pool |= {Simt(2, v, w, 4, 8) for v in (6, 7) for w in (4, 8)}
    if (n, k) in smallm.SHAPES:
        pool.add(old_dense_recipe(n, k, smallm.selected(n, k).key))
        for r in csv.DictReader(DENSE_REVIEW.open(), delimiter='\t'):
            if (int(r['N']), int(r['K'])) == (n, k):
                pool.add(old_dense_recipe(n, k, r['kpack_key']))
    for c in pool:
        if c.columns*c.values > 32 or c.columns not in (4, 8) or n % c.tile_n:
            raise ValueError('invalid cooperative reader width')
    return tuple(sorted(pool))


def recipe(n, k, key):
    candidates = [c for c in simt_inventory(n, k) if c.key == key]
    if len(candidates) != 1:
        raise ValueError('undeclared SIMT configuration')
    return candidates[0]


def real_ids(tokens):
    """Retain the old weighted-without-replacement router's exact ID bytes."""
    state = ROUTING_SEED
    ids = []
    for _ in range(tokens):
        selected = set(); remain = 16*4+240; picks = []
        for _ in range(8):
            state, value = _splitmix64(state); lottery = value % remain
            for e in range(256):
                if e in selected:
                    continue
                weight = 4 if e < 16 else 1
                if lottery < weight:
                    picks.append(e); selected.add(e); remain -= weight; break
                lottery -= weight
        ids.append(picks)
    old = _routing_fixture(256, 8, tokens)
    if _identity_sha256(ids) != old['token_routes_sha256']:
        raise ValueError('historical router bytes differ')
    return np.asarray(ids, dtype='<i4')


def routed_ids(tokens, router):
    if router == 'real':
        return real_ids(tokens)
    if router == 'spread':
        buckets = np.arange(tokens)
    elif router == 'cluster':
        buckets = np.zeros(tokens, dtype=int)
    elif re.fullmatch(r'repeat[2-7]', router) and tokens == 8:
        repeat = int(router[-1]); buckets = np.arange(tokens)//repeat
    else:
        raise ValueError('undeclared routed workload')
    return np.asarray([(np.arange(8)*17+3+int(b)*13) % 256 for b in buckets], dtype='<i4')


def workloads():
    result = []
    for n, k in DENSE:
        for m in MS:
            result.append(dict(id=f'dense-n{n}-k{k}-m{m}', operator='dense', n=n, k=k,
                               tokens=m, channels=1, router='dense', experts=1, rows=m))
    for n, k in GROUPED:
        axes = [(t, ch, r) for t in MS for ch in (1, 8) for r in ('spread', 'real')]
        axes += [(8, ch, r) for ch in (1, 8) for r in ['cluster']+[f'repeat{i}' for i in range(2, 8)]]
        for t, ch, router in axes:
            ids = routed_ids(t, router)
            result.append(dict(id=f'grouped-n{n}-k{k}-t{t}-ch{ch}-{router}', operator='grouped',
                               n=n, k=k, tokens=t, channels=ch, router=router, experts=256, rows=t*8,
                               ids_sha256=digest(ids.tolist())))
    if len(result) != 372 or len({r['id'] for r in result}) != 372:
        raise ValueError('decode workload denominator differs')
    return result


def family(w):
    return f'{w["operator"]}:{w["n"]}x{w["k"]}'


def request(w):
    return (12, 0 if w['operator'] == 'dense' else 2, w['rows'], w['n'], w['k'], w['experts'], w['tokens'])


def historical_tc():
    """All prior family anchors and published winners, not just one default."""
    data = json.loads(moe_compare.TACTICS.read_text())
    old_dense = json.loads((ROOT/'prebuilt/ppu0010/q4-smallm-v1/manifest.json').read_text())
    old_moe = json.loads((moe_compare.TC_BUNDLE/'manifest.json').read_text())
    registry = {n: {f:p[f] for f in PARENT_FIELDS} for n,p in data['parents'].items()}
    registry.update({r['parent']['symbol']:r['parent'] for m in (old_dense, old_moe) for r in m['modules']})
    families = {}; runtime = {}
    for op, shapes in (('dense', DENSE), ('grouped', GROUPED)):
        for n, k in shapes:
            f = f'{op}:{n}x{k}'; source_n = n//2 if op == 'grouped' and n == 1024 else n
            key = digest((12, 'fq-'+op, source_n, k, 32, 1 if op == 'dense' else 256))
            names = {r['parent'] for r in data['anchors'].get(key, []) if r['features'][0] <= (8 if op == 'dense' else 64)}
            families[f] = names; runtime[f] = set()
    for row in csv.DictReader(DENSE_REVIEW.open(), delimiter='\t'):
        f = f'dense:{row["N"]}x{row["K"]}'
        for field in ('tc_best_key', 'tc_current_key'):
            symbol, split = row[field].rsplit(':s', 1)
            families[f].add(symbol); runtime[f].add((symbol, int(split), 0, 0))
    for row in csv.DictReader(MOE_REVIEW.open(), delimiter='\t'):
        n, k = map(int, re.match(r'n(\d+)-k(\d+)', row['case']).groups())
        _, symbol, s, b, g = row['tc_key'].split(':')
        f = f'grouped:{n}x{k}'; families[f].add(symbol)
        runtime[f].add((symbol, int(s[1:]), int(b[1:]), int(g[1:])))
    parents = {n:registry[n] for names in families.values() for n in names}
    for p in parents.values():
        validate_parent(p)
    return parents, {f:sorted(v) for f,v in families.items()}, {f:sorted(v) for f,v in runtime.items()}


def tc_inventory(manifest, w):
    f = family(w); allowed = set(manifest['families'][f]); records = []
    exact = manifest['retained_runtime'][f]
    for record in manifest['modules']:
        p = record['parent']; name = p['symbol']
        if name not in allowed:
            continue
        for split in (1, 2, 4, 8):
            grids = [(b, g) for b in (1, 2, 4) for g in (2, 3)] if p['persistent'] == 1 else [(0, 0)]
            grids += [(b, g) for sym, s, b, g in exact if sym == name and s == split]
            for b, g in sorted(set(grids)):
                reason = None
                if p['ap'] and w['rows'] != 1:
                    reason = 'PACKED_ROW_A_REQUIRES_M1'
                elif w['k'] % (p['tk']*split) or w['k']//(p['tk']*split) < p['stages']-1:
                    reason = 'INSUFFICIENT_K_TILES_PER_PIPELINE_SLICE'
                records.append(dict(key=f'tc:{name}:s{split}:b{b}:g{g}', arm='tc', parent=name,
                                    split=split, grid_b=b, grid_mode=g, reason=reason))
    return records


def catalog(manifest, w):
    return [dict(key='simt:'+c.key, arm='simt', recipe=c.key, reason=None)
            for c in simt_inventory(w['n'], w['k'])] + tc_inventory(manifest, w)


def plan():
    return dict(schema=SCHEMA, qtype=12, dense_families=[list(x) for x in DENSE], grouped_families=[list(x) for x in GROUPED],
                dense_cases=96, grouped_cases=276, workloads=workloads(),
                scope='DECODE_ONLY_F32_ENDPOINTS_TC_CASTS_ADAPTERS_REAL_REDUCER_INCLUDED',
                production_changed=False, prefill_changed=False,
                pruning=['Q4 optimized readers only; other qtypes keep their current policy',
                         'M/tokens1..8; no prefill Cartesian product',
                         'C8/P8 exceeds warp-cooperative metadata width and is not instantiated',
                         'All historical parent and published SIMT/TC winners explicitly retained',
                         'AP1 is packed-row FP16 A delivery, not activation quantization; M1 only'])


def verify(bundle=BUNDLE):
    bundle = Path(bundle); m = json.loads((bundle/'manifest.json').read_text())
    if m.get('schema') != SCHEMA or m['plan'] != plan() or m['production_changed']:
        raise ValueError('decode sweep plan differs')
    for path, value in m['source_hashes'].items():
        if sha(ROOT/path) != value:
            raise ValueError('decode sweep source differs: '+path)
    for path, value in m['payloads'].items():
        p = (bundle/path).resolve(strict=True)
        if not p.is_relative_to(bundle.resolve()) or sha(p) != value:
            raise ValueError('decode sweep payload differs: '+path)
    parents, families, runtime = historical_tc()
    compiled = {r['parent']['symbol']:r['parent'] for r in m['modules']}
    if len(compiled) != len(m['modules']):
        raise ValueError('duplicate TC parent')
    for f, names in families.items():
        if not set(names) <= set(m['families'][f]):
            raise ValueError('historical TC parent omitted: '+f)
        if not set(map(tuple, runtime[f])) <= set(map(tuple, m['retained_runtime'][f])):
            raise ValueError('historical TC tactic omitted: '+f)
    for name, p in parents.items():
        if compiled.get(name) != p:
            raise ValueError('historical parent tuple differs')
    return m


def simt_source(n, k):
    prefix = '#include <hggc_runtime.h>\n#include "decode_kernel.cuh"\n'
    prefix += f'''extern "C" int q4_decode_sweep_run(qkg_call_v1 const* call,int id,
        quactlize_ppu_placed_arrangement_v2 const* arrangement) {{
    if(!call || call->n!={n} || call->k!={k} || call->rows>64) return QKG_SHAPE;
    if(call->mode==QKG_DENSE && call->rows>8) return QKG_SHAPE;
    qkg_sizes_v1 sizes{{}};
'''
    for i, r in enumerate(simt_inventory(n, k)):
        prefix += f'''    if(id=={i}) {{
        qkg_q4_s1_config_v1 config{{1,sizeof(config),{r.reader},{r.variant},{r.warps},{r.values}}};
        int rc=quactlize::execution::q4_s1::validate(*call,config,arrangement,sizes);
        if(rc) return rc;
        rc=quactlize::execution::q4_s1::buffers(*call,sizes);
        if(rc) return rc;
        return q4_decode_sweep::launch<{r.reader},{r.variant},{r.warps},{r.values},{r.columns},{n},{k}>(*call);
    }}
'''
    return prefix+'    return QKG_INVALID;\n}\n'
