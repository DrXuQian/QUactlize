"""Exact-only Q8 port: body identity, unchanged fallback and stale-DSO rejection."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

import pytest

from quactlize.dispatch.planning import plan_smallm, validate_inventory
from quactlize.execution import q8_refresh as binding
from quactlize.execution.simt_codegen import runtime_inventory
from quactlize.runtime.tuning import digest
from tools import generate_decode_selection as gen
from tools import generate_q8_refresh as port

ROOT = gen.ROOT


def test_generated_body_and_scope_match_measured_source():
    assert port.catalog() == (ROOT/'quactlize/execution/q8_refresh.inc').read_text()
    assert port.header() == (ROOT/'quactlize/execution/simt_q8_refresh.cuh').read_text()
    assert 'dev/' not in port.header()
    assert len(binding.rows(include_dense=False)) == 27
    assert len(binding.rows()) == 63
    assert {r['implementation']['body'] for r in binding.rows()} == {1, 3, 4, 5}


def test_regeneration_cannot_relabel_a_changed_kernel_with_old_timings(monkeypatch):
    original = port.body
    monkeypatch.setattr(port, 'body', lambda b: original(b)+'// edited\n')
    with pytest.raises(ValueError, match='new admission'):
        port.header()


def test_only_27_exact_keys_change_and_all_bucket_donors_are_retained():
    table = json.loads(binding.TABLE.read_text())
    old = gen.effective(json.loads(gen.EVIDENCE.read_text()), include_refresh=False)
    new = gen.effective(json.loads(gen.EVIDENCE.read_text()), include_dense=False)
    assert digest(old) == table['baseline_sha256']
    assert table['active'] is False  # The other160 replacements remain proposals.
    changed = {tuple(r['key']): r for r in binding.rows()}
    for a, b in zip(old['exact'], new['exact']):
        if tuple(a['key']) not in changed:
            assert a == b
        else:
            r = changed[tuple(a['key'])]
            assert b['config'] == r['config'] and b['implementation'] == r['implementation']
            assert b['guard'] == a['guard'] and b['bucket_config'] == a['config']
            assert b['bucket_dequant_config'] == a.get('dequant_config', -1)
    assert old['buckets'] == new['buckets'] and old['excluded'] == new['excluded']
    # Actual generated bucket choices, not only the intermediate Python rows.
    before, after = gen.header(old), gen.header(new)
    for array in ('kBucketChoices', 'kBucketIndex', 'kBucketDequant'):
        assert before.split(array+'[]={', 1)[1].split('};', 1)[0] == (
            after.split(array+'[]={', 1)[1].split('};', 1)[0])


@pytest.fixture(scope='module')
def plan(tmp_path_factory):
    return plan_smallm(tmp_path_factory.mktemp('q8-binding-query'))


def test_actual_selector_preserves_body_in_future_original_recipe(plan):
    from dev.strategy.decode_plan import original_recipe
    from dev.strategy.vector import spec
    from quactlize.dispatch.table import Workload
    rows = {tuple(r['key']): r for r in binding.rows()}
    from quactlize.execution.token_cta_policy import rows as token_rows
    superseded = {tuple(r['key']) for r in token_rows()}
    count = 0
    for r in plan['requests']:
        selected = rows.get(tuple(r['request']))
        if tuple(r['request']) in superseded:
            assert r['implementation']['producer']=='token-cta'
            continue
        if not selected:
            assert not r.get('implementation', {}).get('q8_body')
            continue
        count += 1
        assert r['status'] == 0 and r['kind'] == 1 and r['policy'] == 18
        assert r['config'] == {f: selected['config'][f] for f in binding.FIELDS}
        assert r['implementation']['q8_body'] == selected['implementation']['body']
        recipe = original_recipe(r)
        q, mode, n, k, e, top, ch, m, compute = r['request']
        actual = spec(Workload(q, n, k, m), recipe)
        assert actual['q8_body'] == selected['implementation']['body']
        assert asdict(recipe)['kind'] == 'simt'
    assert count == 62  # Q8 N1024/K5120/M2 now uses the later measured token body.


@pytest.mark.parametrize('fault', ['missing', 'body', 'catalog', 'key', 'geometry'])
def test_old_or_misbound_execution_cannot_pass_as_same_geometry(plan, fault):
    selected = [r for r in plan['requests'] if r.get('implementation', {}).get('q8_body')]
    p = dict(plan, requests=copy.deepcopy(selected))
    execution = dict(simt_configs={'8': [r.record() for r in runtime_inventory(8)]},
                     q8_refresh=binding.capability())
    validate_inventory(p, execution)
    if fault == 'missing':
        execution.pop('q8_refresh')
    elif fault == 'body':
        execution['q8_refresh']['body_sha256'] = '0'*64
    elif fault == 'catalog':
        execution['q8_refresh']['catalog_sha256'] = '0'*64
    elif fault == 'key':
        p['requests'][0]['request'][2] += 256
    else:
        p['requests'][0]['config']['warps'] = 2
    with pytest.raises(ValueError, match='Q8'):
        validate_inventory(p, execution)


def test_host_launch_scope_excludes_unmeasured_modes_storage_and_splits(tmp_path):
    code = r'''
#include "quactlize/execution/simt_strategy.hpp"
#include <cassert>
using namespace quactlize::execution::simt;
int main() {
int checked=0;
#define Q8R(N,K,M,C,W,Body) { \
qkg_simt_call_v2 d{};auto& c=d.call;d.compute_type=QKG_COMPUTE_F16; \
c.qtype=8;c.mode=QKG_DENSE;c.input_type=QKG_F32;c.experts=1;c.topk=1;c.channels=1; \
c.n=N;c.k=K;c.rows=M; \
qkg_simt_config_v1 f{1,sizeof(f),5,C,W,4,1}; \
assert(q8_refresh_body(d,f)==Body && implementation(d,f).q8_body==Body);++checked; \
for(int mode:{QKG_GROUPED,QKG_INDEXED}) {c.mode=mode;assert(!q8_refresh_body(d,f));}c.mode=QKG_DENSE; \
for(int s:{2,4,8}) {f.split=s;assert(!q8_refresh_body(d,f));}f.split=1; \
for(int input:{int(QKG_F16),int(QKG_SIMT_BF16)}) {c.input_type=input;assert(!q8_refresh_body(d,f));}c.input_type=QKG_F32; \
d.compute_type=QKG_COMPUTE_BF16;assert(!q8_refresh_body(d,f));d.compute_type=QKG_COMPUTE_F16; \
for(int m:{0,9,65536}) {c.rows=m;assert(!q8_refresh_body(d,f));}c.rows=M; \
c.n+=16;assert(!q8_refresh_body(d,f));c.n=N; \
c.k+=32;assert(!q8_refresh_body(d,f));c.k=K; \
c.topk=8;assert(!q8_refresh_body(d,f));c.topk=1; \
c.experts=256;assert(!q8_refresh_body(d,f));c.experts=1; \
c.channels=8;assert(!q8_refresh_body(d,f));c.channels=1; \
f.variant=4;assert(!q8_refresh_body(d,f));f.variant=5; \
f.values=2;assert(!q8_refresh_body(d,f));f.values=4; \
f.warps=2;assert(!q8_refresh_body(d,f)); \
}
#include "quactlize/execution/q8_refresh.inc"
#undef Q8R
assert(checked==63);
}
'''
    path = tmp_path/'scope.cpp'
    path.write_text('#include <initializer_list>\n'+code)
    subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(ROOT), path, '-o', tmp_path/'scope'], check=True)
    subprocess.run([tmp_path/'scope'], check=True)


def test_readiness_closes_source_binding_only():
    from dev.strategy.refresh_selection import readiness
    table = json.loads(binding.TABLE.read_text())
    old = gen.effective(json.loads(gen.EVIDENCE.read_text()), include_refresh=False)
    report = readiness(table, old, q8_bindings=binding.capability()['keys'])
    assert report['status'] == 'STRUCTURALLY_READY_NOT_DEVICE_ADMITTED'
    assert report['pending'] == [] and report['kernel_launches'] == 0
