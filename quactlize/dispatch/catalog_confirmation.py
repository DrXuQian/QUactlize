"""Build-time validation of compact model-tuning confirmations, not a tuner."""
import math
import statistics

from quactlize.runtime.tuning import digest

SOURCE = 'e76f9dea18bcddcf7c2ae66d15052b07f285bf45'
POLICY = '55d25805cc7cd35009d9cb9057acaa12e33dcfed192968e4fcccf811bdba674c'
ARCHIVE = 'b9cf9a9b04315d6ba175a0de3c53c9f75aa8073db67691ee1f7cdb6440315890'


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
    require((w['qtype'], w['operation'], w['compute'], w['storage'], w['experts'], w['topk'], w['channels']) ==
            (8, 'dense', 'f16', 'f32', 1, 1, 1) and (w['n'], w['k']) in ((1024,2048),(2048,3072)) and
            w['m'] in (*range(1,9),16,32,64,128,256,512,1024,2048,4096,5120) and
            w['fusion'] == 'none' and not w['router'] and
            (w['input_stride'],w['output_stride']) == (w['k'],w['n']), 'model confirmation workload differs')
    require(c == dict(accumulator='f32', activation_included=False, epilogue='IDENTITY', form='concat',
            input_type='f32', layout='CANONICAL_KPACK', logical_n=w['n']//2, output_n=w['n'],
            output_type='f32', round_projection=0, row_order='original', scope='PROJECTION_ONLY', weight_n=w['n']),
            'model confirmation arithmetic differs')
    require(p['quactlize_commit'] == SOURCE and p['timing_method'] == 'ASYS_GPU_ACTIVITY' and
            value['policy_sha256'] == POLICY and value['archive_sha256'] == ARCHIVE and
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
        require(r['route'] == 'sf' and p.pop('dequant_config') == -1, 'unsupported dense TC contract')
        p['parent_persistent'] = p.pop('persistent')
        return dict(kind='tc', route=1, **p), dict(kind='shipping')
    require(r['kind'] == 'simt' and r['route'] == 'fq', 'unsupported dense reader')
    if 'cta_tokens' in p:
        require(p.pop('cta_tokens') == w['m'], 'token ownership differs')
        impl = dict(kind='token-cta', tokens=w['m'], window=p.pop('cta_window'))
    else:
        impl = dict(kind='q8-address', body=p.pop('q8_body'))
    require(set(p) == {'variant','columns','warps','values','split'}, 'unbound dense reader option')
    return dict(kind='simt', **p), impl


def prefill_configuration(value):
    from quactlize.dispatch.dense_prefill import execution
    parts = {str(i):v for i,v in enumerate(value['components'])}
    return execution(value['binding'], dict(components=list(parts)), parts)
