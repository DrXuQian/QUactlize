"""Qwen TP-local helper geometries; synthetic projections, not a GEMM runner."""
from collections import Counter
from dataclasses import asdict

from dev.moe_scaffold.protocol import Case

MS = (*range(1, 9), 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 5120)
ROUTERS = ('spread', 'cluster', 'repeat2', 'repeat4', 'random')
# Audited GGUF headers and llama CPU TP split, actual-next-plans-r2:
# Gate/Up Q4_K [E,512,K], Down Q5_K [E,K,512], E256/top8.
# This is a frozen geometry registry, not fresh model-file verification.
MODELS = (('qwen35-35b-a3b', 1, 2048), ('qwen35-122b-a10b', 2, 3072))


def model_plan(smoke=False, phase='all'):
    if phase not in ('all', 'prepare', 'transition', 'finish'):
        raise ValueError('unknown helper phase')
    cases, consumers = {}, {}

    # Unused dimensions stay one: each entry profiles one helper phase, not a
    # fake full network. Do not allocate model-wide tensors for unrelated phases.
    for name, _, k in MODELS:
        for m in MS:
            for router in (('spread',) if m == 1 else ROUTERS):
                c = Case(m, columns=k, hidden=1, output=1, padding=0,
                         profile=router, phase='prepare')
                cases[c.key] = c; consumers[c.key] = {name}
                for splits, storage in ((1, -1), (1, 0), (2, -1), (4, -1), (8, -1)):
                    c = Case(m, columns=1, hidden=1, output=k, padding=0, phase='finish',
                             profile=router, down_splits=splits, projection_dtype=storage)
                    cases[c.key] = c; consumers[c.key] = {name}
            # Direct SIMT final F32 output is in token/slot order: no map reads,
            # so do not multiply its measurement by five irrelevant ID patterns.
            c = Case(m, columns=1, hidden=1, output=k, padding=0, phase='finish',
                     profile='spread', down_order=1, projection_dtype=0)
            cases[c.key] = c; consumers[c.key] = {name}
            for merged in (0, 1):
                for state, splits, storage in ((0, 1, -1), (0, 1, 0),
                                               (1, 2, -1), (1, 4, -1), (1, 8, -1)):
                    c = Case(m, columns=1, hidden=512, output=1, padding=0, phase='transition',
                             profile='spread', merged=merged, state=state, splits=splits,
                             projection_dtype=storage)
                    cases[c.key] = c; consumers.setdefault(c.key, set()).add(name)
            for storage in (-1, 0):
                c = Case(m, columns=1, hidden=512, output=1, padding=0, phase='transition',
                         profile='spread', state=2, projection_dtype=storage)
                cases[c.key] = c; consumers.setdefault(c.key, set()).add(name)

    def smoke_case(c):
        if c.phase == 'prepare':
            return (c.tokens, c.profile) in ((1, 'spread'), (5120, 'cluster'))
        if c.phase == 'finish':
            return (c.tokens == 5120 and c.profile == 'cluster' and c.down_splits == 8 or
                    c.tokens == 8 and c.down_order == 1)
        return ((c.tokens, c.state, c.splits, c.merged, c.projection_dtype) in
                ((8, 0, 1, 0, 0), (1, 0, 1, 1, -1), (5120, 1, 8, 1, -1), (128, 2, 1, 1, -1)))

    all_count = len(cases)
    selected = sorted((c for c in cases.values() if (not smoke or smoke_case(c)) and
                       (phase == 'all' or c.phase == phase)), key=lambda c: (c.phase, c.tokens, c.key))
    coverage = dict(cohort='qwen-models', cases=len(selected), full_cases=all_count,
        phases=dict(Counter(c.phase for c in selected)),
        models=[dict(model=n, tp=t, input=k, hidden=512, output=k, experts=256, topk=8,
                     gate_up_qtype=12, down_qtype=13) for n, t, k in MODELS],
        tokens=sorted({c.tokens for c in selected}), routers=list(ROUTERS),
        consumers={c.key: sorted(consumers[c.key]) for c in selected},
        layout='CONTIGUOUS_F32_INPUT_BF16_INTERMEDIATE_F32_WEIGHTED_OUTPUT',
        projection_states='FINAL_BF16_OR_F32_PARTIALS_2_4_8_ACTIVATED_BF16_OR_F32',
        historical_times_reused=False, actual_model_weights_used=False,
        model_ids='SYNTHETIC_NOT_CAPTURED_MODEL_ROUTES',
        scope='HELPER_GEOMETRIES_NOT_GEMM_OR_WHOLE_CHAIN_OR_MODEL',
        deferred=['DEEPSEEK', 'MODEL_SHAPED_ORDER_CONVERSION', 'ACTUAL_GEMM_CHAIN', 'NATIVE_MODEL_AB'])
    return selected, coverage


def estimated_work(c):
    rows = c.tokens*c.topk
    if c.phase == 'prepare': return 1_000_000 + rows*c.columns
    if c.phase == 'transition': return 1_000_000 + rows*c.hidden*max(1, c.splits)*2
    if c.phase == 'finish': return 1_000_000 + rows*c.output*max(1, c.down_splits)
    return 1_000_000 + rows*(c.columns+c.hidden*max(1,c.splits)*2+c.output*max(1,c.down_splits))


def assignments(cases, workers):
    """Keep all candidates for a point on one device; spread expensive points."""
    if workers < 1: raise ValueError('need at least one worker')
    loads, assigned = [0]*workers, [[] for _ in range(workers)]
    for c in sorted(cases, key=lambda c: (-estimated_work(c), c.key)):
        i = min(range(workers), key=lambda i: (loads[i], i))
        assigned[i].append(asdict(c)); loads[i] += estimated_work(c)
    return assigned
