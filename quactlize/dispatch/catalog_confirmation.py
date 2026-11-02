"""Build-time validation of compact model-tuning confirmations, not a tuner."""
import math
import statistics

from quactlize.runtime.tuning import digest

SOURCE = 'e76f9dea18bcddcf7c2ae66d15052b07f285bf45'
POLICY = '55d25805cc7cd35009d9cb9057acaa12e33dcfed192968e4fcccf811bdba674c'
ARCHIVE = 'b9cf9a9b04315d6ba175a0de3c53c9f75aa8073db67691ee1f7cdb6440315890'
COHORTS = {
    SOURCE: (POLICY, ARCHIVE, {(8,1024,2048), (8,2048,3072)}),
    'bb72f9676ec9255c2daec9b0984f42edb321f5c6': (
        'a4deee3523f47d420221e82bd194b02ba2e53875566de9ac0a05146db58137e0',
        'dace9a58a89fb104bfaf35675589d5d60812e5d7c907eab847960cc8ada9a9ca',
        {(12,51200,5120), *((q,34816,5120) for q in (11,12,13,14))}),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def mean(values):
    flat = [v for row in values for v in row] if values and isinstance(values[0], list) else values
    require(len(flat) >= 3 and all(math.isfinite(v) and v > 0 for v in flat), 'invalid confirmed samples')
    return statistics.fmean(flat)


def validate(value):
    require(value['schema'] == 'quactlize.model-tuning-confirmation.v1' and
            value['sha256'] == digest({k:v for k,v in value.items() if k != 'sha256'}),
            'model confirmation seal differs')
    w, c, p = value['workload'], value['contract'], value['provenance']
    cohort = COHORTS.get(p['quactlize_commit'])
    require(cohort is not None, 'unreviewed model confirmation source')
    policy, archive, families = cohort
    require((w['operation'], w['compute'], w['storage'], w['experts'], w['topk'], w['channels']) ==
            ('dense', 'f16', 'f32', 1, 1, 1) and (w['qtype'], w['n'], w['k']) in families and
            w['m'] in (*range(1,9),16,32,64,128,256,512,1024,2048,4096,5120) and
            w['fusion'] == 'none' and not w['router'] and
            (w['input_stride'],w['output_stride']) == (w['k'],w['n']), 'model confirmation workload differs')
    require(c == dict(accumulator='f32', activation_included=False, epilogue='IDENTITY', form='concat',
            input_type='f32', layout='CANONICAL_KPACK', logical_n=w['n']//2, output_n=w['n'],
            output_type='f32', round_projection=0, row_order='original', scope='PROJECTION_ONLY', weight_n=w['n']),
            'model confirmation arithmetic differs')
    require(p['timing_method'] == 'ASYS_GPU_ACTIVITY' and
            value['policy_sha256'] == policy and value['archive_sha256'] == archive and
            value['correctness'] == 'PASS', 'model confirmation provenance/correctness differs')
    # The compact export carries the passed numerical receipt's digest, not
    # its numeric error. Do not invent an error value or reuse another case's.
    proof = value['proof']
    hashes = [value['observation_sha256'], p['source_identity'], p['binary_manifest_sha256'],
              proof.get('sha256', proof.get('measurement_sha256', ''))]
    require(all(len(h) == 64 and all(c in '0123456789abcdef' for c in h) for h in hashes),
            'model confirmation proof binding missing')
    close = lambda a,b:math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-8)
    require(close(mean(value['samples_us']), value['mean_us']) and
            math.isfinite(value['baseline_us']) and 0 < value['mean_us'] <= value['baseline_us'],
            'model confirmation mean/regression differs')
    parts = value['components']
    require(parts and all(v['uses'] == 1 and close(mean(v['samples_us']),v['mean_us']) for v in parts) and
            close(math.fsum(v['mean_us'] for v in parts),value['mean_us']), 'model confirmation helper sum differs')


def smallm_configuration(value):
    w = value['workload']; r = value['binding']['recipe']; p = dict(r['parameters'])
    require(w['m'] <= 8, 'not a small-M confirmation')
    if r['kind'] == 'tc':
        require(r['route'] in ('fq','sf') and p.pop('dequant_config') == -1, 'unsupported dense TC contract')
        p['parent_persistent'] = p.pop('persistent')
        return dict(kind='tc', route=int(r['route']=='sf'), **p), dict(kind='shipping')
    require(r['kind'] == 'simt' and r['route'] == 'fq', 'unsupported dense reader')
    if 'cta_tokens' in p:
        require(p.pop('cta_tokens') == w['m'], 'token ownership differs')
        impl = dict(kind='token-cta', tokens=w['m'], window=p.pop('cta_window',0))
    elif 'q8_body' in p:
        impl = dict(kind='q8-address', body=p.pop('q8_body'))
    else:
        impl = dict(kind='shipping')
    require(set(p) == {'variant','columns','warps','values','split'}, 'unbound dense reader option')
    return dict(kind='simt', **p), impl


def prefill_configuration(value):
    from quactlize.dispatch.dense_prefill import execution
    parts = {str(i):v for i,v in enumerate(value['components'])}
    return execution(value['binding'], dict(components=list(parts)), parts)
