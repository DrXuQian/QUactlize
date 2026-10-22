"""Physical high-word ownership only; matched N64 K128/K256 controls.

No metadata, consumer, output, arithmetic or production-selection change.
Four Q6/Q5 shape families, two precisions, four arms per precision.
"""
import re
from . import round2
from quactlize.dispatch.table import Workload

ROUND = 5
FORMATS = (13, 14)
DEQUANT_VARIANTS = (20, 21, 22, 23)
DYNAMIC_VARIANTS = (22, 23)
IO_VARIANTS = (1, 2)
DYNAMIC_IO = 2


def workloads():
    return [Workload(q, n, k, 2048, storage='f32', compute='f16')
            for q in (14, 13) for n, k in ((5120, 25600), (1024, 5120))]


def pool(point):
    w = workloads()[point]
    return [dict(name=f'r5-{dtype}-dq{variant}', storage=storage,
                 vector_io=2 if w.n == 1024 or (w.qtype == 14 and storage) else 1,
                 dequant_variant=variant, math_mode=16, algorithm=-1)
            for dtype, storage in (('bf16', 1), ('fp16', 0)) for variant in (20, 22, 21, 23)]


def stages(point, arm):
    return ('dequant',)


def composed(point, rows):
    result = []
    incumbent = 21 if workloads()[point].n > 1024 else 20
    for storage in (1, 0):
        subset = [r for r in rows if r['arm']['storage'] == storage]
        control = next(r['arm'] for r in subset if r['arm']['dequant_variant'] == incumbent)
        candidate = min((r for r in subset if r['arm']['dequant_variant'] >= 22),
                        key=lambda r: r['median_us']['dequant'])['arm']
        result += [control | dict(name=f'composed-{storage}-incumbent'),
                   candidate | dict(name=f'composed-{storage}-screen-choice')]
    return result


def profile_jobs(results):
    jobs = []
    for point, w in enumerate(workloads()):
        row = next(r for r in results if r['point'] == point and r['status'] == 'PASS')
        storage = int(w.qtype == 14)
        candidates = [r for r in row['arms'] if r['arm']['storage'] == storage and r['arm']['dequant_variant'] >= 22]
        candidate = min(candidates, key=lambda r: r['median_us']['dequant'])['arm']
        control = next(r['arm'] for r in row['arms'] if r['arm']['storage'] == storage and
                       r['arm']['dequant_variant'] == candidate['dequant_variant'] - 2)
        # Matched tile / dtype / IO, ABBA, no whole-call capture or cold ring.
        for repeat, order in enumerate(((control, candidate), (candidate, control))):
            for arm in order:
                jobs.append(dict(point=point, stage=1, arm=arm['name'], pair=point, repeat=repeat,
                                 role='control' if arm == control else 'candidate'))
    return jobs


def check_profile(path, job):
    if job['stage'] != 1:
        raise ValueError('R5 profiles isolated dequant only')
    arm = next(a for a in pool(job['point']) if a['name'] == job['arm'])
    w = workloads()[job['point']]
    variant = arm['dequant_variant']
    tk = 128 if variant in (20, 22) else 256
    expected = 'dense_dequant_unique_high' if variant >= 22 else 'dense_dequant_cooperative'
    result = round2.check_profile(path, job, expected_kernel=expected,
                                  geometry=([w.n // 64, w.k // tk, 1], 128))
    suffix = r', (\d+)u?>' if variant >= 22 else r', 64u?, (\d+)u?, true>'
    match = re.search(expected + r'<\(gguf_scale::KType\)([34]), (true|false)' + suffix, result['launches'][0][1])
    args = (str(w.qtype - 10), 'true' if arm['storage'] else 'false', str(tk))
    if not match or match.groups() != args:
        raise ValueError('R5 profile precision/format/tile differs')
    return result | dict(qtype=w.qtype, storage=arm['storage'], variant=variant,
                         tile=[64, tk], high_ownership='UNIQUE_PHYSICAL' if variant >= 22 else 'LOGICAL_K32',
                         candidate_binding='NUMERIC_RECEIPT_EXACT_SYMBOL_GEOMETRY')


cycle_pairs = round2.cycle_pairs
