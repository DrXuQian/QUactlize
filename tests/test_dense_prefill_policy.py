"""Real C++ selection replay against Box evidence. No simulated GPU timing."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from quactlize.dispatch.dense_prefill import EVIDENCE, HEADER, bucket_key, render, validate, audit_buckets
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[1]


def evidence():
    return json.loads(EVIDENCE.read_text())


def test_all_exact_rows_are_complete_call_minima():
    data = evidence()
    validate(data)
    assert render(data) == HEADER.read_text()
    assert len(data['points']) == 165
    assert len(data['reviews']) == 2
    assert sum(p['winner_evidence']['captures'] == 2 for p in data['points']) == 47
    assert sum(p['us'] > p['native_us'] for p in data['points']) == 12
    assert not data['model_performance_admitted']
    # Evidence keeps the native losses; export is not a zero-regression claim.
    assert max(100*(p['us']/p['native_us']-1) for p in data['points']) < 5


@pytest.mark.parametrize('fault', ['seal', 'winner', 'cost', 'bucket', 'recipe', 'precision'])
def test_mutated_catalog_fails(fault):
    data = copy.deepcopy(evidence())
    p = data['points'][0]
    if fault == 'seal':
        data['sha256'] = '0'*64
    else:
        if fault == 'winner':
            p['winner'] = next(r for r,_ in p['costs'] if r != p['winner'])
        elif fault == 'cost':
            next(c for c in p['costs'] if c[0] == p['winner'])[1] *= 2
        elif fault == 'bucket':
            data['buckets'][0]['bounds'][0][0] -= 256
        elif fault == 'recipe':
            data['recipes'][p['winner']]['io'] = 99
        elif fault == 'precision':
            data['arithmetic']['compute'] = 'BF16'
        data['sha256'] = digest({k:v for k,v in data.items() if k != 'sha256'})
    with pytest.raises(ValueError):
        validate(data)


def test_bucket_choice_was_measured_at_every_member():
    data = evidence()
    points = {p['key']:p for p in data['points']}
    for b in data['buckets']:
        assert b['predicted'] and not b['holdout_validated']
        for key in b['members']:
            p = points[key]
            assert list(bucket_key(p['shape'])) == b['key']
            assert dict(p['costs'])[b['recipe']] <= p['us']*1.05
    # Q4's conflicting coarse cells must be split, not forced to common FQ.
    assert len(data['buckets']) > len({bucket_key(p['shape']) for p in data['points']})


def test_bucket_audit_does_not_call_missing_measurements_a_win():
    audit = audit_buckets(evidence())
    assert sum(audit['counts'].values()) == 165
    assert audit['counts']['comparable'] == 33
    assert audit['counts']['candidate_not_measured_on_held_family'] == 17
    assert audit['worst_regret_pct'] < 6
    assert len(audit['over_5pct']) == 1 and not audit['new_model_validated']
    # N/K transposition used to produce 27% leave-family-out regret.
    assert bucket_key((12,2048,6144,5120)) != bucket_key((12,2048,5120,6144))


def test_cpp_exact_bucket_and_precision_domain(tmp_path):
    source = tmp_path/'query.cpp'
    source.write_text('''#include "quactlize/dispatch/dense_prefill.hpp"
#include <iostream>
int main() {
    using namespace quactlize::dispatch::dense_prefill;
    int q,m,n,k,compute,operation;
    while(std::cin>>q>>m>>n>>k>>compute>>operation) {
        qks_request_v1 r{1,sizeof(r),q,operation?2:(q==8?1:0),m,n,k,operation?256:1,m,
            q==8?UINT64_C(0x51384b5032540001):q==12?UINT64_C(0x51344b5034540001):UINT64_C(0x514b504b54000001)};
        auto d=select(r,compute);
        std::cout<<(d?int(d.choice-kChoices):-1)<<" "<<d.policy<<"\\n";
    }
}''')
    exe = tmp_path/'query'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT),source,'-o',exe],check=True)
    data = evidence()
    used = sorted({p['winner'] for p in data['points']} | {b['recipe'] for b in data['buckets']})
    exact = {tuple(p['shape']):p['winner'] for p in data['points']}

    def predicted(shape):
        q,m,n,k = shape
        if not (128 <= m <= 2048) or n%256 or k%(512 if q in (11,14) else 256):
            return -1, 0
        if shape in exact:
            return used.index(exact[shape]), 20
        matches = [b for b in data['buckets'] if list(bucket_key(shape)) == b['key'] and
                   b['bounds'][0][0] <= n <= b['bounds'][0][1] and
                   b['bounds'][1][0] <= k <= b['bounds'][1][1]]
        assert len(matches) <= 1
        if not matches:
            return -1, 0
        recipe = data['recipes'][matches[0]['recipe']]
        tc = recipe['tc']
        if tc and (k%(tc['tk']*tc['split']) or k//(tc['tk']*tc['split']) < tc['stages']-1 or n%tc['tn']):
            return -1, 0
        return used.index(matches[0]['recipe']), 21

    cases = []
    for shape in exact:
        q,m,n,k = shape
        cases.extend(((*shape,0,0),(*shape,1,0),(*shape,0,1)))
        cases.extend((q,mm,nn,kk,0,0) for mm in (m-1,m+1) for nn in (n-256,n,n+256)
                     for kk in (k-256,k,k+256) if nn>0 and kk>0)
    cases += [(12,m,1024,5120,0,0) for m in (0,1,8,9,64,127,4096)]
    cases += [(10,2048,5120,5120,0,0)] # Q2 has no measurement in this cohort
    text = ''.join(' '.join(map(str,c))+'\n' for c in cases)
    rows = [tuple(map(int,line.split())) for line in subprocess.check_output([exe],input=text,text=True).splitlines()]
    assert len(rows) == len(cases)
    for c,actual in zip(cases,rows):
        assert actual == ((-1,0) if c[4] or c[5] else predicted(c[:4])), c
