"""Product correctness-test support."""
from functools import lru_cache
from quactlize.runtime.compiler import validate_parent
def parent(recipe, workload):
    p = dict(recipe.parameters)
    result = {k: p[k] for k in ('symbol', 'tm', 'tn', 'tk', 'wm', 'wn', 'stages', 'ap', 'dn')} | dict(
        qtype=p.get('qtype', workload.qtype),
        route=recipe.route + ('-dense' if workload.operation == 'dense' else '-grouped'),
        persistent=p.get('persistent', 0 if workload.operation != 'dense' and recipe.route == 'fq' else -1))
    if 'dense_reducer' in p:result['dense_reducer']=p['dense_reducer']
    return result


@lru_cache(maxsize=65536)
def rejection(recipe, workload, max_threads=1024):
    """Hard semantic/launch filters only; no invented occupancy or SMEM cost.

The 1024-thread ceiling is the current builder's 32-warps constraint, not a
guessed device attribute. A probed device may impose a tighter ceiling. Exact
shared storage/register occupancy remain the compiled resource query's job.
"""
    if not recipe.eligible(workload):
        return 'SHAPE_PRECISION_OR_PIPELINE_CONTRACT'
    p = dict(recipe.parameters)
    if 'dense_reducer' in p and (type(p['dense_reducer']) is not int or p['dense_reducer'] not in (2,4,8) or recipe.kind!='tc' or
            workload.operation!='dense' or workload.compute!='f16' or workload.m<=8 or
            p.get('split') not in (2,4,8) or p.get('grid_mode',0)!=0):
        return 'COMPACT_REDUCER_REQUIRES_NATIVE_DENSE_PREFILL_SPLITK'
    if recipe.kind == 'tc':
        try:
            c = parent(recipe, workload)
            validate_parent(c)
        except (ValueError, KeyError) as exc:
            return 'PARENT_CONTRACT: ' + str(exc)
        # ppu_tactic_space.hpp::cta_warps, asserted against TiledMma there.
        threads = 32 * (p['tm'] // p['wm']) * (p['tn'] // p['wn'])
        if threads > min(1024, max_threads):
            return 'CTA_THREAD_LIMIT'
        persistent = p.get('grid_mode', 0) >= 2
        if workload.operation == 'dense' and persistent and (recipe.route == 'fq' or p['split'] != 1):
            return 'DENSE_PERSISTENT_REQUIRES_SF_S1'
        if workload.operation != 'dense' and recipe.route == 'fq' and p.get('persistent', 0) != int(persistent):
            return 'GROUPED_SCHEDULE_DIFFERS_FROM_COMPILED_PARENT'
    if recipe.kind == 'simt':
        if (p.get('variant') not in range(6) or p.get('columns') not in (4, 8) or
                p.get('warps') not in (2, 4, 8) or p.get('values') not in (2, 4, 8) or
                p['columns'] * p['values'] > 32 or p.get('split') not in (1, 2, 4, 8)):
            return 'SIMT_CONFIG_CONTRACT'
        if workload.qtype == 8 and p['variant'] & 2:
            return 'Q8_SIMT_VARIANT'
        if p['variant'] >= 4 and (workload.qtype != 8 or workload.storage != 'f32'):
            return 'Q8_F32_VECTOR_ONLY'
        if p['warps'] * 32 > max_threads:
            return 'CTA_THREAD_LIMIT'
    if recipe.kind == 'q4':
        reader, variant, warps, values, columns = (p.get(k) for k in ('reader','variant','warps','values','columns'))
        if (any(type(x) is not int for x in (reader,variant,warps,values,columns)) or
                columns not in (4,8) or not 1<=warps<=32 or columns*values>32 or
                not (reader==0 and variant in (0,1) and values==1 or
                     reader==1 and variant in range(4) and values in (2,4) or
                     reader==2 and variant in range(8) and values in (4,8))):
            return 'Q4_READER_CONTRACT'
        if reader and (warps*32//columns)%8:
            return 'Q4_WORKER_GROUP_ALIGNMENT'
        if warps*32>max_threads:
            return 'CTA_THREAD_LIMIT'
        if workload.storage!='f32' and not (workload.storage=='bf16' and workload.compute=='bf16'):
            return 'Q4_DECODE_STORAGE_CONTRACT'
    return None
